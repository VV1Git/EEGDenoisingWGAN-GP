import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split # type:ignore

# Import data preparation utilities from your custom file
from eeg_data_generator import prepare_eeg_data, EEGNoiseDataset, DataLoader

# Import shared variables
from variables import (
    EEG_FILE, EOG_FILE, EMG_FILE, SNR_RANGE_DB, TRAIN_SPLIT_RATIO, NUM_NOISE_VARIANTS,
    LEARNING_RATE, BATCH_SIZE, CHANNELS_EEG, NUM_EPOCHS, FEATURES_CRITIC, FEATURES_GEN,
    CRITIC_ITERATIONS, LAMBDA_GP, LOGS_DIR
)

from utils import gradient_penalty, save_checkpoint, load_checkpoint
from model import Discriminator, Generator, initialize_weights

# --- Hyperparameters etc. ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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

# --- Data Splitting (Training and Testing) ---
train_clean_eeg_np, test_clean_eeg_np = train_test_split(
    clean_eeg_np, test_size=(1 - TRAIN_SPLIT_RATIO), random_state=42
)

print(f"\nDataset split into:")
print(f"  Training clean EEG epochs: {train_clean_eeg_np.shape[0]}")
print(f"  Testing clean EEG epochs: {test_clean_eeg_np.shape[0]}")

# Create EEGNoiseDataset for training data
eeg_denoising_train_dataset = EEGNoiseDataset(
    train_clean_eeg_np, eog_noise_np, emg_noise_np, SNR_RANGE_DB,
    num_noise_variants_per_clean_epoch=NUM_NOISE_VARIANTS
)

# --- Optimized DataLoader ---
num_cpu_cores = os.cpu_count()
num_workers_to_use = num_cpu_cores if num_cpu_cores else 0

loader = DataLoader(
    eeg_denoising_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers_to_use,
    pin_memory=True,
)
print(f"DataLoader configured with num_workers={num_workers_to_use} and pin_memory=True (for CUDA).")
print(f"Training will use {len(eeg_denoising_train_dataset)} augmented samples per epoch.")

# --- Initialize Generator and Critic Models ---
gen = Generator(CHANNELS_EEG, SAMPLES_PER_EPOCH, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_EEG, SAMPLES_PER_EPOCH, FEATURES_CRITIC).to(device)

initialize_weights(gen)
initialize_weights(critic)

# --- Optimizers ---
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

gen.train()
critic.train()

# --- Training Loop ---
print("\nStarting WGAN-GP training for EEG Denoising...")

for epoch in range(NUM_EPOCHS):
    for batch_idx, (noisy_signals, clean_signals) in enumerate(tqdm(loader)):
        noisy_signals = noisy_signals.to(device)
        clean_signals = clean_signals.to(device)

        for _ in range(CRITIC_ITERATIONS):
            fake_denoised_signals = gen(noisy_signals)
            critic_real = critic(clean_signals).reshape(-1)
            critic_fake = critic(fake_denoised_signals.detach()).reshape(-1)

            gp = gradient_penalty(critic, clean_signals, fake_denoised_signals, device=device)

            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            )

            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        gen_fake_score = critic(fake_denoised_signals).reshape(-1)
        loss_gen = -torch.mean(gen_fake_score)

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    if (epoch + 1) % 50 == 0:
        checkpoint_filename = os.path.join(LOGS_DIR, f"gen_epoch_{epoch+1}.pth.tar")
        save_checkpoint({'gen': gen.state_dict(), 'opt_gen': opt_gen.state_dict()}, checkpoint_filename)
        print(f"Saved generator checkpoint for Epoch {epoch+1} to '{checkpoint_filename}'")

    print(f"Epoch {epoch+1} finished. Loss Critic: {loss_critic.item():.4f}, Loss Generator: {loss_gen.item():.4f}")

print("\nTraining complete!")

# --- Save final generator model to 'model' directory ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)
final_gen_filename = os.path.join(MODEL_DIR, "final_generator_model.pth.tar")
save_checkpoint({'gen': gen.state_dict(), 'opt_gen': opt_gen.state_dict()}, final_gen_filename)
print(f"Saved final generator model to '{final_gen_filename}'")
