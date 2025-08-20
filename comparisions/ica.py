import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from scipy.signal import welch
from scipy.stats import pearsonr
from tqdm import tqdm
from sklearn.decomposition import FastICA
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress all ConvergenceWarning messages
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Import shared variables and data utilities
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from variables import (
    EEG_FILE, EOG_FILE, EMG_FILE, SNR_RANGE_DB_EVAL, SNR_RANGE_DB,
    BATCH_SIZE, EEG_BANDS, SAMPLING_RATE, PSD_SAMPLE_INDEX_FOR_VIZ, TRAIN_SPLIT_RATIO, NUM_NOISE_VARIANTS
)
from eeg_data_generator import prepare_eeg_data, EEGNoiseDataset, DataLoader

# --- Metric Calculation Functions (copied from evaluate.py) ---
def calculate_rrmse(clean_signal, denoised_signal):
    clean_signal = clean_signal.flatten()
    denoised_signal = denoised_signal.flatten()
    rmse = np.sqrt(np.mean((clean_signal - denoised_signal)**2))
    rms_clean = np.sqrt(np.mean(clean_signal**2))
    if rms_clean == 0:
        return np.inf if rmse > 0 else 0.0
    return rmse / rms_clean

def calculate_cc(clean_signal, denoised_signal):
    clean_signal = clean_signal.flatten()
    denoised_signal = denoised_signal.flatten()
    if np.std(clean_signal) == 0 or np.std(denoised_signal) == 0:
        return 1.0 if np.allclose(clean_signal, denoised_signal) else 0.0
    return pearsonr(clean_signal, denoised_signal)[0]

def plot_multi_snr_samples(snrs, noisy_samples, clean_samples, denoised_samples, save_path):
    num = len(snrs)
    fig, axes = plt.subplots(num, 1, figsize=(15, 3 * num), sharex=True)
    if num == 1:
        axes = [axes]
    fig.suptitle("ICA Sample Denoising Comparison for SNRs: " + ", ".join([str(s) for s in snrs]), fontsize=16)
    for idx, (snr, noisy, clean, denoised) in enumerate(zip(snrs, noisy_samples, clean_samples, denoised_samples)):
        ax = axes[idx]
        ax.plot(clean, label='Clean EEG', color='blue', alpha=0.7)
        ax.plot(noisy, label='Noisy EEG', color='red', alpha=0.7)
        ax.plot(denoised, label='ICA Denoised EEG', color='green', linestyle='--', alpha=0.8)
        ax.set_title(f"SNR {snr} dB")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close(fig)

# --- ICA Denoising Function ---
def ica_denoise(noisy_signal, n_components=3):
    """
    Apply ICA to denoise a single-channel EEG signal by separating sources and reconstructing
    the signal after removing the component with the highest kurtosis (assumed artifact).
    Args:
        noisy_signal: shape (samples,) or (1, samples)
        n_components: number of ICA components
    Returns:
        denoised_signal: shape (samples,)
    """
    # ICA expects shape (samples, channels), so stack the signal with delayed versions
    # to create a pseudo-multichannel input if needed
    x = noisy_signal.flatten()
    X = np.stack([np.roll(x, shift) for shift in range(n_components)], axis=1)
    ica = FastICA(n_components=n_components, random_state=0, max_iter=10000, tol=0.1)
    S_ = ica.fit_transform(X)
    # Remove the component with the highest absolute kurtosis (likely artifact)
    kurt = np.abs(np.apply_along_axis(lambda s: np.mean((s - np.mean(s))**4) / (np.var(s)**2), 0, S_))
    artifact_idx = np.argmax(kurt)
    S_[:, artifact_idx] = 0
    X_denoised = ica.inverse_transform(S_)
    # Return the first channel (original signal)
    return X_denoised[:, 0]

# --- Main ICA Evaluation Logic ---
def main():
    ICA_EVAL_PLOTS_DIR = os.path.join("comparisions", "ica_evaluation_plots")
    os.makedirs(ICA_EVAL_PLOTS_DIR, exist_ok=True)
    print(f"Created/Ensured '{ICA_EVAL_PLOTS_DIR}' directory exists for ICA evaluation plots.")

    # Load data
    clean_eeg_all, eog_noise, emg_noise = prepare_eeg_data(
        EEG_FILE, EOG_FILE, EMG_FILE, [-100, -100]
    )
    SAMPLES_PER_EPOCH = clean_eeg_all.shape[1]
    _, test_clean_eeg_np = train_test_split(
        clean_eeg_all, test_size=(1 - TRAIN_SPLIT_RATIO), random_state=42
    )

    snr_values_db = SNR_RANGE_DB_EVAL
    rrmse_temporal_per_snr = []
    cc_per_snr = []
    snr_samples = []

    print("\n--- ICA evaluation across different SNRs ---")
    for current_snr_db in snr_values_db:
        print(f"\nEvaluating at SNR: {current_snr_db} dB")
        test_dataset_current_snr = EEGNoiseDataset(
            test_clean_eeg_np, eog_noise, emg_noise, [current_snr_db, current_snr_db],
            num_noise_variants_per_clean_epoch=NUM_NOISE_VARIANTS
        )
        test_loader_current_snr = DataLoader(
            test_dataset_current_snr,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        batch_rrmse_temporal = []
        batch_cc = []
        example_saved = False

        for batch_idx, (noisy_signals, clean_signals) in enumerate(tqdm(test_loader_current_snr, desc=f"SNR {current_snr_db}dB")):
            noisy_signals_np = noisy_signals.numpy()
            clean_signals_np = clean_signals.numpy()
            for i in range(noisy_signals_np.shape[0]):
                noisy = noisy_signals_np[i, 0, :]
                clean = clean_signals_np[i, 0, :]
                denoised = ica_denoise(noisy)
                batch_rrmse_temporal.append(calculate_rrmse(clean, denoised))
                batch_cc.append(calculate_cc(clean, denoised))
                if not example_saved:
                    snr_samples.append((current_snr_db, noisy, clean, denoised))
                    example_saved = True

        rrmse_temporal_per_snr.append(np.mean(batch_rrmse_temporal))
        cc_per_snr.append(np.mean(batch_cc))

    # Save grouped plots for SNRs in pairs
    group_size = 2
    for idx in range(0, len(snr_samples), group_size):
        group = snr_samples[idx:idx+group_size]
        snrs = [item[0] for item in group]
        noisy_samples = [item[1] for item in group]
        clean_samples = [item[2] for item in group]
        denoised_samples = [item[3] for item in group]
        save_path = os.path.join(ICA_EVAL_PLOTS_DIR, f"ica_multi_snr_sample_denoising_{'_'.join(str(s) for s in snrs)}.png")
        plot_multi_snr_samples(snrs, noisy_samples, clean_samples, denoised_samples, save_path)
        print(f"Saved ICA grouped sample denoising plot for SNRs {snrs} to '{save_path}'")

    # Plot RRMSE Temporal vs SNR
    plt.figure(figsize=(6, 5))
    plt.plot(snr_values_db, rrmse_temporal_per_snr, marker='o', linestyle='-', color='blue')
    plt.title('ICA RRMSE Temporal vs. Input SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('RRMSE Temporal')
    plt.grid(True)
    plt.savefig(os.path.join(ICA_EVAL_PLOTS_DIR, 'ica_RRMSE_Temporal_vs_SNR.png'))
    plt.close()
    print(f"Saved ICA RRMSE Temporal plot to '{os.path.join(ICA_EVAL_PLOTS_DIR, 'ica_RRMSE_Temporal_vs_SNR.png')}'")

    # Plot CC vs SNR
    plt.figure(figsize=(6, 5))
    plt.plot(snr_values_db, cc_per_snr, marker='o', linestyle='-', color='blue')
    plt.title('ICA Pearson\'s CC vs. Input SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Pearson\'s CC')
    plt.grid(True)
    plt.savefig(os.path.join(ICA_EVAL_PLOTS_DIR, 'ica_CC_vs_SNR.png'))
    plt.close()
    print(f"Saved ICA Pearson's CC plot to '{os.path.join(ICA_EVAL_PLOTS_DIR, 'ica_CC_vs_SNR.png')}'")

    print("\nICA evaluation complete.")

if __name__ == "__main__":
    main()
