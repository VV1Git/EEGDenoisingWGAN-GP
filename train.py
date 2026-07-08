import torch
import torch.nn as nn
import torch.optim as optim
import os
from contextlib import nullcontext
from tqdm import tqdm

# Import data preparation utilities from your custom file
from eeg_data_generator import prepare_eeg_data, EEGNoiseDataset, DataLoader, split_train_test

# Import shared variables
from variables import (
    EEG_FILE, EOG_FILE, EMG_FILE, SNR_RANGE_DB, NUM_NOISE_VARIANTS,
    LEARNING_RATE, BATCH_SIZE, CHANNELS_EEG, NUM_EPOCHS, FEATURES_CRITIC, FEATURES_GEN,
    CRITIC_ITERATIONS, LAMBDA_GP, LOGS_DIR, LAMBDA_L1, GEN_NUM_LAYERS, DISC_NUM_LAYERS
)

from utils import gradient_penalty, save_checkpoint
from model import Discriminator, Generator, initialize_weights

# --- Hyperparameters etc. ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Input shape is fixed at (B, 1, 512) every step, so the cuDNN autotuner picks
# an optimal conv algorithm once and never thrashes. GPU-only; a no-op on CPU.
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Seed for reproducible weight initialization and data shuffling
torch.manual_seed(42)

SAMPLES_PER_EPOCH = None # This will be set after loading data

# --- Directory Setup ---
os.makedirs(LOGS_DIR, exist_ok=True)
print(f"Created/Ensured '{LOGS_DIR}' directory exists.")

# --- Data Loading and Preparation ---
try:
    clean_eeg_np, eog_noise_np, emg_noise_np = prepare_eeg_data(
        EEG_FILE, EOG_FILE, EMG_FILE, SNR_RANGE_DB
    )
    SAMPLES_PER_EPOCH = clean_eeg_np.shape[1]
    print(f"Detected samples per epoch: {SAMPLES_PER_EPOCH}")
except (FileNotFoundError, ValueError) as e:
    print(f"Error preparing data: {e}")
    print("Please ensure your dataset files are correctly placed and named.")
    exit()

# --- Data Splitting ---
# Training uses only the training clean epochs AND the training noise pools, so no
# test-set clean epoch or noise realization is ever seen during training.
train_clean_eeg_np, _, train_eog_np, _, train_emg_np, _ = split_train_test(
    clean_eeg_np, eog_noise_np, emg_noise_np
)

print(f"\nTraining on {train_clean_eeg_np.shape[0]} clean EEG epochs "
      f"(EOG pool {0 if train_eog_np is None else len(train_eog_np)}, "
      f"EMG pool {0 if train_emg_np is None else len(train_emg_np)}).")

# Create EEGNoiseDataset for training data
eeg_denoising_train_dataset = EEGNoiseDataset(
    train_clean_eeg_np, train_eog_np, train_emg_np, SNR_RANGE_DB,
    num_noise_variants_per_clean_epoch=NUM_NOISE_VARIANTS
)

# --- Optimized DataLoader ---
# Cap workers: each sample is only microseconds of numpy work, so ~8 workers
# saturate this trivial pipeline. More over-subscribe (extra IPC + per-epoch
# re-spawn cost, which is large on Windows spawn) and contend with the
# 5-critic-iteration compute. persistent_workers avoids re-spawning every epoch;
# prefetch_factor deepens the queue; pin_memory only helps for CUDA H2D copies.
num_cpu_cores = os.cpu_count() or 0
num_workers_to_use = min(8, num_cpu_cores)

loader = DataLoader(
    eeg_denoising_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers_to_use,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers_to_use > 0),
    prefetch_factor=(4 if num_workers_to_use > 0 else None),
)
print(f"DataLoader configured with num_workers={num_workers_to_use} and pin_memory={torch.cuda.is_available()} (for CUDA).")
print(f"Training will use {len(eeg_denoising_train_dataset)} augmented samples per epoch.")

# --- Initialize Generator and Critic Models ---
gen = Generator(CHANNELS_EEG, SAMPLES_PER_EPOCH, FEATURES_GEN, dropout_p=0.2, num_layers=GEN_NUM_LAYERS).to(device)
critic = Discriminator(CHANNELS_EEG, SAMPLES_PER_EPOCH, FEATURES_CRITIC, num_layers=DISC_NUM_LAYERS).to(device)

initialize_weights(gen)
initialize_weights(critic)

# --- Optional torch.compile (opt-in, GPU only) ---
# torch.compile can fuse the many tiny 1D-conv kernels this net is bottlenecked
# on. It is opt-in (EEG_COMPILE=1) and GPU-only: on CPU it mostly adds startup
# cost, and the GP create_graph=True double-backward can trip older Inductor
# versions (caught below). We keep handles to the *uncompiled* modules so the
# saved state_dict keys match evaluate.py / comparisons (compiled modules
# prefix keys with "_orig_mod.").
gen_raw, critic_raw = gen, critic
USE_COMPILE = torch.cuda.is_available() and os.environ.get("EEG_COMPILE", "0") == "1"
if USE_COMPILE:
    try:
        gen = torch.compile(gen)
        critic = torch.compile(critic)
        print("torch.compile enabled for gen/critic.")
    except Exception as e:
        gen, critic = gen_raw, critic_raw
        print(f"torch.compile unavailable, continuing eager: {e}")


def _unwrap(module):
    """Return the original (uncompiled) module so checkpoint keys stay compatible."""
    return getattr(module, "_orig_mod", module)

# --- Optimizers ---
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

gen.train()
critic.train()

l1_loss_fn = nn.L1Loss()

# --- Automatic Mixed Precision (GPU only) ---
# AMP (fp16 autocast + GradScaler) is the biggest GPU throughput lever but
# changes GPU numerics, so it is entirely gated on CUDA. On CPU use_amp is
# False, amp_ctx() is nullcontext(), and the scaler is disabled -> the CPU path
# runs pure fp32 with no autocast/scaler and is byte-identical to before.
# CRITICAL: the WGAN-GP gradient penalty (create_graph=True double-backward) is
# computed in fp32 OUTSIDE autocast (see utils.gradient_penalty), and the critic
# loss that contains the GP term is NOT scaled (scaler.scale would corrupt the
# second-order graph). Only the generator's first-order loss is scaled.
use_amp = torch.cuda.is_available()
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


def amp_ctx():
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

# --- Training Loop ---
print("\nStarting WGAN-GP training for EEG Denoising...")

for epoch in range(NUM_EPOCHS):
    for batch_idx, (noisy_signals, clean_signals) in enumerate(tqdm(loader)):
        # non_blocking overlaps the H2D DMA with compute when the source is
        # pinned (GPU path). On CPU it is a no-op and numerically identical.
        noisy_signals = noisy_signals.to(device, non_blocking=True)
        clean_signals = clean_signals.to(device, non_blocking=True)

        for _ in range(CRITIC_ITERATIONS):
            # The generator is not updated during critic iterations, so generate
            # the fake signals without building a generator graph (saves memory
            # and compute). The gradient penalty re-enables grad on the
            # interpolated signals internally.
            with torch.no_grad():
                with amp_ctx():
                    fake_denoised_signals = gen(noisy_signals)
            # The GP path must be fp32; drop any fp16 from the autocast forward.
            fake_denoised_signals = fake_denoised_signals.float()

            # Critic forward runs under autocast (fp16 matmuls/convs = the speed
            # win on Tensor cores); the Wasserstein term is cast back to fp32
            # before it is combined with the fp32 gradient penalty.
            with amp_ctx():
                critic_real = critic(clean_signals).reshape(-1)
                critic_fake = critic(fake_denoised_signals).reshape(-1)
            w_term = -(critic_real.float().mean() - critic_fake.float().mean())

            # Gradient penalty is computed in fp32 outside autocast (see
            # utils.gradient_penalty). Pass clean_signals as fp32 too.
            gp = gradient_penalty(critic, clean_signals.float(), fake_denoised_signals, device=device)

            loss_critic = w_term + LAMBDA_GP * gp

            # The critic loss contains the create_graph=True GP term, so it must
            # NOT be scaled by GradScaler (scaling would corrupt the second-order
            # graph). Plain backward/step in both CPU and GPU paths.
            critic.zero_grad(set_to_none=True)
            loss_critic.backward()
            opt_critic.step()

        # Generator update: a plain first-order loss -> fully GradScaler-safe and
        # where fp16 pays off most. On CPU the scaler is disabled (pass-through).
        with amp_ctx():
            fake_denoised_signals = gen(noisy_signals)
            gen_fake_score = critic(fake_denoised_signals).reshape(-1)
            adv_loss = -torch.mean(gen_fake_score)
            l1_loss = l1_loss_fn(fake_denoised_signals, clean_signals)
            loss_gen = adv_loss + LAMBDA_L1 * l1_loss

        gen.zero_grad(set_to_none=True)
        scaler.scale(loss_gen).backward()
        scaler.step(opt_gen)
        scaler.update()

    if (epoch + 1) % 50 == 0:
        checkpoint_filename = os.path.join(LOGS_DIR, f"gen_epoch_{epoch+1}.pth.tar")
        save_checkpoint({'gen': _unwrap(gen).state_dict(), 'opt_gen': opt_gen.state_dict()}, checkpoint_filename)
        print(f"Saved generator checkpoint for Epoch {epoch+1} to '{checkpoint_filename}'")

    print(f"Epoch {epoch+1} finished. Loss Critic: {loss_critic.item():.4f}, Loss Generator: {loss_gen.item():.4f}, L1 Loss: {l1_loss.item():.4f}")

print("\nTraining complete!")

# --- Save final generator model to 'model' directory ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)
final_gen_filename = os.path.join(MODEL_DIR, "final_generator_model.pth.tar")
# _unwrap keeps state_dict keys free of the "_orig_mod." prefix that
# torch.compile adds, so evaluate.py / comparisons load the checkpoint cleanly.
save_checkpoint({'gen': _unwrap(gen).state_dict(), 'opt_gen': opt_gen.state_dict()}, final_gen_filename)
print(f"Saved final generator model to '{final_gen_filename}'")
