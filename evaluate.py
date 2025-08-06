import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt # type:ignore
import os
from sklearn.model_selection import train_test_split # type:ignore
from scipy.signal import welch # For power spectral density calculation # type:ignore
from scipy.integrate import simpson # For integrating PSD over frequency bands # type:ignore
from scipy.stats import pearsonr # For Pearson's correlation coefficient # type:ignore
from tqdm import tqdm # Added tqdm import
from sklearn.metrics.pairwise import cosine_similarity # For cosine similarity # type:ignore

# Import shared variables
from variables import (
    EEG_FILE, EOG_FILE, EMG_FILE, SNR_RANGE_DB_EVAL, SNR_RANGE_DB, SAVED_MODEL_PATH, EVAL_PLOTS_DIR,
    CHANNELS_EEG, FEATURES_GEN, BATCH_SIZE, EEG_BANDS, SAMPLING_RATE, PSD_SAMPLE_INDEX_FOR_VIZ, TRAIN_SPLIT_RATIO, NUM_NOISE_VARIANTS
)

# Import necessary components from your project files
from model import Generator # Only need the Generator for evaluation
from utils import load_checkpoint # To load the saved model
from eeg_data_generator import prepare_eeg_data, EEGNoiseDataset, DataLoader

# --- Configuration ---
# Set device for evaluation
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Paths to your data files (should be consistent with training setup)
# SNR_RANGE_DB is now used for the range of SNRs to *test*

# Output directory for evaluation plots
os.makedirs(EVAL_PLOTS_DIR, exist_ok=True)
print(f"Created/Ensured '{EVAL_PLOTS_DIR}' directory exists for evaluation plots.")

# Model architecture parameters (MUST match the trained generator's parameters)
CHANNELS_EEG = 1
SAMPLES_PER_EPOCH = None # Will be set after loading data

# EEG frequency bands and sampling rate for power analysis

# Data split ratio (MUST match training.py)
# Number of noise variants per clean epoch for augmentation (MUST match training.py)


# --- Metric Calculation Functions ---

def calculate_rrmse(clean_signal, denoised_signal):
    """Calculates Relative Root Mean Squared Error (RRMSE) in the temporal domain.
    Lower RRMSE indicates better denoising.
    """
    # Ensure inputs are NumPy arrays
    clean_signal = clean_signal.flatten() # Flatten to 1D
    denoised_signal = denoised_signal.flatten()

    # Calculate RMSE
    rmse = np.sqrt(np.mean((clean_signal - denoised_signal)**2))
    
    # Calculate RMS of the clean signal
    rms_clean = np.sqrt(np.mean(clean_signal**2))

    if rms_clean == 0: # Avoid division by zero for flat clean signals
        return np.inf if rmse > 0 else 0.0
    return rmse / rms_clean

def calculate_rrmse_spectral(clean_signal, denoised_signal, sampling_rate):
    """Calculates Relative Root Mean Squared Error (RRMSE) in the spectral domain.
    Compares the PSDs of the clean and denoised signals.
    """
    # Ensure inputs are 1D arrays for PSD calculation
    if clean_signal.ndim > 1: clean_signal = clean_signal.flatten()
    if denoised_signal.ndim > 1: denoised_signal = denoised_signal.flatten()

    # Compute PSDs
    f_clean, Pxx_clean = welch(clean_signal, fs=sampling_rate, nperseg=sampling_rate, return_onesided=True)
    f_denoised, Pxx_denoised = welch(denoised_signal, fs=sampling_rate, nperseg=sampling_rate, return_onesided=True)

    # Ensure frequency bins match (they should if sampling_rate and nperseg are same)
    if not np.array_equal(f_clean, f_denoised):
        raise ValueError("Frequency bins for clean and denoised PSDs do not match.")

    # Calculate RMSE between PSDs
    rmse_psd = np.sqrt(np.mean((Pxx_clean - Pxx_denoised)**2))
    
    # Calculate RMS of clean PSD
    rms_clean_psd = np.sqrt(np.mean(Pxx_clean**2))

    if rms_clean_psd == 0:
        return np.inf if rmse_psd > 0 else 0.0
    return rmse_psd / rms_clean_psd


def calculate_cc(clean_signal, denoised_signal):
    """Calculates Pearson's Correlation Coefficient (CC).
    Higher CC (closer to 1) indicates better preservation of signal shape.
    """
    # Ensure inputs are NumPy arrays
    clean_signal = clean_signal.flatten() # Flatten to 1D
    denoised_signal = denoised_signal.flatten()

    # Handle cases where std dev might be zero (flat signals)
    if np.std(clean_signal) == 0 or np.std(denoised_signal) == 0:
        return 1.0 if np.allclose(clean_signal, denoised_signal) else 0.0 # Perfect correlation if identical and flat, else 0
    
    # Pearsonr returns (correlation_coefficient, p_value)
    return pearsonr(clean_signal, denoised_signal)[0]

def calculate_band_power_ratios(signal, sampling_rate, bands):
    """
    Calculates the power in specified EEG frequency bands and their ratios to total power.

    Args:
        signal (np.ndarray): The EEG signal. Can be 1D (samples,) or 2D (batch_size, samples).
        sampling_rate (int): The sampling rate of the EEG signal (Hz).
        bands (dict): Dictionary defining frequency bands, e.g.,
                      {'delta': [0.5, 4], 'theta': [4, 8], ...}

    Returns:
        dict: A dictionary with band power ratios (e.g., {'delta_ratio': 0.2, ...}).
    """
    # If signal is (batch_size, 1, samples), it's already squeezed to (batch_size, samples)
    # If signal is (samples,), it remains 1D.
    
    # Compute Power Spectral Density (PSD) using Welch's method
    # f: array of sample frequencies (1D)
    # Pxx: Power spectral density. If signal is 1D, Pxx is 1D. If signal is 2D (batch_size, samples), Pxx is (batch_size, num_freq_bins)
    f, Pxx = welch(signal, fs=sampling_rate, nperseg=sampling_rate, return_onesided=True, axis=-1) # nperseg = 2 seconds window

    # Integrate PSD to get total power.
    # If Pxx is (batch_size, num_freq_bins), total_power will be (batch_size,)
    total_power = simpson(Pxx, f, axis=-1) 

    band_ratios = {}
    for band_name, (low_freq, high_freq) in bands.items():
        # Find frequencies within the band using a boolean mask
        freq_mask = (f >= low_freq) & (f <= high_freq)
        
        # Integrate PSD over the band frequencies using the mask
        # band_power will be (batch_size,) if Pxx was (batch_size, num_freq_bins), or scalar if Pxx was 1D
        band_power = simpson(Pxx[..., freq_mask], f[freq_mask], axis=-1)
        
        # Calculate ratio, handle division by zero
        # Ensure total_power is not zero before division
        ratio = np.where(total_power == 0, 0, band_power / total_power)
        
        # Average ratio across the batch (if batch exists), otherwise just the scalar ratio
        band_ratios[f'{band_name}_ratio'] = np.mean(ratio) if ratio.ndim > 0 else ratio

    return band_ratios

def calculate_cosine_similarity_power_ratios(clean_signal_np, denoised_signal_np, sampling_rate, bands):
    """
    Calculates the cosine similarity between the vector of band power ratios
    of the clean signal and the denoised signal.

    Args:
        clean_signal_np (np.ndarray): A single clean EEG signal (1D array).
        denoised_signal_np (np.ndarray): A single denoised EEG signal (1D array).
        sampling_rate (int): The sampling rate of the EEG signal (Hz).
        bands (dict): Dictionary defining frequency bands.

    Returns:
        float: Cosine similarity value.
    """
    # Calculate band power ratios for clean and denoised signals
    clean_ratios = calculate_band_power_ratios(clean_signal_np, sampling_rate, bands)
    denoised_ratios = calculate_band_power_ratios(denoised_signal_np, sampling_rate, bands)

    # Create vectors of ratios in a consistent order
    clean_ratio_vector = np.array([clean_ratios[f'{band}_ratio'] for band in bands.keys()])
    denoised_ratio_vector = np.array([denoised_ratios[f'{band}_ratio'] for band in bands.keys()])

    # Reshape for sklearn's cosine_similarity (expects 2D arrays: (n_samples, n_features))
    clean_ratio_vector = clean_ratio_vector.reshape(1, -1)
    denoised_ratio_vector = denoised_ratio_vector.reshape(1, -1)

    # Calculate cosine similarity
    # cosine_similarity returns a 2D array, so take the [0, 0] element
    return cosine_similarity(clean_ratio_vector, denoised_ratio_vector)[0, 0]


# --- Visualization Function (reused from training.py, adapted for evaluation) ---
def plot_evaluation_samples(noisy_sample, clean_sample, denoised_sample, run_label, num_plots=4, save_path=None):
    """
    Plots a few noisy, clean, and denoised EEG signals for evaluation visualization.

    Args:
        noisy_sample (torch.Tensor): A batch of noisy EEG signals.
        clean_sample (torch.Tensor): A batch of corresponding clean EEG signals.
        denoised_sample (torch.Tensor): A batch of denoised EEG signals from the generator.
        run_label (str): A label for the evaluation run (e.g., "Test Set Evaluation").
        num_plots (int): Number of samples to plot from the batch.
        save_path (str, optional): If provided, the plot will be saved to this path.
    """
    noisy_sample_np = noisy_sample.cpu().detach().numpy()
    clean_sample_np = clean_sample.cpu().detach().numpy()
    denoised_sample_np = denoised_sample.cpu().detach().numpy()

    fig = plt.figure(figsize=(15, 3 * num_plots))
    plt.suptitle(f"{run_label} - Sample Denoising Comparison", fontsize=16)

    for i in range(min(num_plots, noisy_sample_np.shape[0])):
        ax = plt.subplot(num_plots, 1, i + 1)
        ax.plot(clean_sample_np[i, 0, :], label='Clean EEG', color='blue', alpha=0.7)
        ax.plot(noisy_sample_np[i, 0, :], label='Noisy EEG', color='red', alpha=0.7)
        ax.plot(denoised_sample_np[i, 0, :], label='Denoised EEG', color='green', linestyle='--', alpha=0.8)
        ax.set_title(f"Sample {i+1}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

def plot_psd_comparison(clean_signal_np, noisy_signal_np, denoised_signal_np, sampling_rate, bands, save_path=None):
    """
    Plots and compares the Power Spectral Density (PSD) of clean, noisy, and denoised signals,
    highlighting EEG frequency bands.

    Args:
        clean_signal_np (np.ndarray): A single clean EEG signal (1D array).
        noisy_signal_np (np.ndarray): A single noisy EEG signal (1D array).
        denoised_signal_np (np.ndarray): A single denoised EEG signal (1D array).
        sampling_rate (int): The sampling rate of the EEG signal (Hz).
        bands (dict): Dictionary defining frequency bands.
        save_path (str, optional): If provided, the plot will be saved to this path.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle('Power Spectral Density Comparison', fontsize=16)

    signal_types = {
        'Clean Signal': clean_signal_np.flatten(),
        'Contaminated Signal': noisy_signal_np.flatten(),
        'Denoised Signal': denoised_signal_np.flatten()
    }
    
    band_colors = {
        'delta': 'yellow',
        'theta': 'orange',
        'alpha': 'lightgreen',
        'beta': 'skyblue',
        'gamma': 'plum'
    }

    for i, (title, signal) in enumerate(signal_types.items()):
        ax = axes[i]
        f, Pxx = welch(signal, fs=sampling_rate, nperseg=sampling_rate, return_onesided=True)
        
        ax.plot(f, Pxx, color='blue')
        ax.set_title(title)
        ax.set_xlabel('Frequency (Hz)')
        if i == 0: # Only set ylabel for the first subplot
            ax.set_ylabel('Power (V**2/Hz)')
        
        # Highlight frequency bands
        for band_name, (low_freq, high_freq) in bands.items():
            ax.axvspan(low_freq, high_freq, color=band_colors[band_name], alpha=0.3, label=band_name.capitalize())
        
        ax.set_xlim(0, 80) # Limit frequency range for better visualization
        ax.grid(True, linestyle=':', alpha=0.6)
        # Add a legend only to the first plot to avoid redundancy
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            # Sort legend labels by frequency band order
            sorted_labels = [b.capitalize() for b in bands.keys()]
            order = [labels.index(l) for l in sorted_labels if l in labels]
            ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper right', fontsize='small')


    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap

    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


# --- Main Evaluation Logic ---
def main():
    # 1. Load and prepare data (ensure consistent splitting with training)
    try:
        clean_eeg_all, eog_noise, emg_noise = prepare_eeg_data(
            EEG_FILE, EOG_FILE, EMG_FILE, [-100, -100] # Use a dummy SNR range for loading, actual SNR will be set per test
        )
        SAMPLES_PER_EPOCH = clean_eeg_all.shape[1]
    except (FileNotFoundError, ValueError) as e:
        print(f"Error preparing data: {e}")
        print("Please ensure your dataset files are correctly placed and named.")
        return

    # Re-split data to ensure the test set is exactly what was unseen during training
    # Use the same random_state as in training.py
    TRAIN_SPLIT_RATIO = 0.9 # Changed to 90% training, 10% testing
    _, test_clean_eeg_np = train_test_split(
        clean_eeg_all, test_size=(1 - TRAIN_SPLIT_RATIO), random_state=42
    )

    test_dataset = EEGNoiseDataset(
        test_clean_eeg_np, eog_noise, emg_noise, SNR_RANGE_DB,
        num_noise_variants_per_clean_epoch=NUM_NOISE_VARIANTS # Pass the new parameter
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # Important: Do NOT shuffle test data for consistent evaluation
        # num_workers=4, # You can add num_workers here for faster evaluation if needed
    )
    print(f"Loaded test dataset with {len(test_dataset)} samples.")

    # 2. Load the trained Generator model
    generator = Generator(CHANNELS_EEG, SAMPLES_PER_EPOCH, FEATURES_GEN).to(device)
    try:
        checkpoint = torch.load(SAVED_MODEL_PATH, map_location=device)
        generator.load_state_dict(checkpoint['gen'])
        print(f"Successfully loaded generator model from '{SAVED_MODEL_PATH}'")
    except Exception as e:
        print(f"Error loading generator model from '{SAVED_MODEL_PATH}': {e}")
        print("Please ensure the model path is correct and the file exists.")
        return

    generator.eval() # Set generator to evaluation mode (disables dropout, batch norm updates)

    # --- Data collection for SNR vs. Metrics plots ---
    snr_values_db = SNR_RANGE_DB_EVAL # Renamed from SNR_RANGE_DB_FOR_TESTING
    rrmse_temporal_per_snr = []
    rrmse_spectral_per_snr = []
    cc_per_snr = []
    cosine_sim_power_ratios_at_neg14db = [] # New list for cosine similarity at -14dB

    print("\n--- Starting evaluation across different SNRs ---")
    for current_snr_db in snr_values_db:
        print(f"\nEvaluating at SNR: {current_snr_db} dB")
        # Create a test dataset specifically for this SNR
        # The EEGNoiseDataset's __getitem__ will now use this specific SNR
        test_dataset_current_snr = EEGNoiseDataset(
            test_clean_eeg_np, eog_noise, emg_noise, [current_snr_db, current_snr_db], # Fixed SNR for testing
            num_noise_variants_per_clean_epoch=NUM_NOISE_VARIANTS # Pass the new parameter
        )
        test_loader_current_snr = DataLoader(
            test_dataset_current_snr,
            batch_size=BATCH_SIZE,
            shuffle=False, # Do NOT shuffle for consistent evaluation
        )

        batch_rrmse_temporal = []
        batch_rrmse_spectral = []
        batch_cc = []
        batch_cosine_sim_power_ratios = [] # For current SNR if needed, or specifically for -14dB

        with torch.no_grad():
            for batch_idx, (noisy_signals, clean_signals) in enumerate(tqdm(test_loader_current_snr, desc=f"SNR {current_snr_db}dB")):
                noisy_signals = noisy_signals.to(device)
                clean_signals = clean_signals.to(device)
                denoised_signals = generator(noisy_signals)

                noisy_signals_np = noisy_signals.cpu().numpy()
                clean_signals_np = clean_signals.cpu().numpy()
                denoised_signals_np = denoised_signals.cpu().numpy()

                for i in range(noisy_signals.shape[0]):
                    # Temporal RRMSE
                    batch_rrmse_temporal.append(calculate_rrmse(clean_signals_np[i], denoised_signals_np[i]))
                    # Spectral RRMSE
                    batch_rrmse_spectral.append(calculate_rrmse_spectral(clean_signals_np[i], denoised_signals_np[i], SAMPLING_RATE))
                    # Pearson's CC
                    batch_cc.append(calculate_cc(clean_signals_np[i], denoised_signals_np[i]))
                    
                    # Calculate Cosine Similarity of Power Ratios specifically at -14dB
                    if current_snr_db == -14:
                        batch_cosine_sim_power_ratios.append(
                            calculate_cosine_similarity_power_ratios(
                                clean_signals_np[i, 0, :], # Pass 1D signal
                                denoised_signals_np[i, 0, :], # Pass 1D signal
                                SAMPLING_RATE,
                                EEG_BANDS
                            )
                        )
            
            # Aggregate metrics for the current SNR
            rrmse_temporal_per_snr.append(np.mean(batch_rrmse_temporal))
            rrmse_spectral_per_snr.append(np.mean(batch_rrmse_spectral))
            cc_per_snr.append(np.mean(batch_cc))

            # If we are at -14dB, aggregate the cosine similarity for this SNR
            if current_snr_db == -14:
                cosine_sim_power_ratios_at_neg14db.append(np.mean(batch_cosine_sim_power_ratios))


    print("\n--- Plotting SNR vs. Metrics ---")
    # Plot RRMSE Temporal vs SNR
    plt.figure(figsize=(6, 5))
    plt.plot(snr_values_db, rrmse_temporal_per_snr, marker='o', linestyle='-', color='blue')
    plt.title('RRMSE Temporal vs. Input SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('RRMSE Temporal')
    plt.grid(True)
    plt.savefig(os.path.join(EVAL_PLOTS_DIR, 'RRMSE_Temporal_vs_SNR.png'))
    plt.close()
    print(f"Saved RRMSE Temporal plot to '{os.path.join(EVAL_PLOTS_DIR, 'RRMSE_Temporal_vs_SNR.png')}'")

    # Plot RRMSE Spectral vs SNR
    plt.figure(figsize=(6, 5))
    plt.plot(snr_values_db, rrmse_spectral_per_snr, marker='o', linestyle='-', color='blue')
    plt.title('RRMSE Spectral vs. Input SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('RRMSE Spectral')
    plt.grid(True)
    plt.savefig(os.path.join(EVAL_PLOTS_DIR, 'RRMSE_Spectral_vs_SNR.png'))
    plt.close()
    print(f"Saved RRMSE Spectral plot to '{os.path.join(EVAL_PLOTS_DIR, 'RRMSE_Spectral_vs_SNR.png')}'")

    # Plot CC vs SNR
    plt.figure(figsize=(6, 5))
    plt.plot(snr_values_db, cc_per_snr, marker='o', linestyle='-', color='blue')
    plt.title('Pearson\'s CC vs. Input SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Pearson\'s CC')
    plt.grid(True)
    plt.savefig(os.path.join(EVAL_PLOTS_DIR, 'CC_vs_SNR.png'))
    plt.close()
    print(f"Saved Pearson's CC plot to '{os.path.join(EVAL_PLOTS_DIR, 'CC_vs_SNR.png')}'")


    # --- Original Aggregated Metrics (kept for overall performance at random SNRs) ---
    print("\n--- Aggregated Evaluation Metrics (Overall Test Set) ---")
    # Re-create a test loader with the original random SNR range for overall metrics
    # Note: SNR_RANGE_DB here is the range used during training (e.g., [-5, 5])
    test_dataset_overall = EEGNoiseDataset(
        test_clean_eeg_np, eog_noise, emg_noise, SNR_RANGE_DB, # Use the original range for overall metrics
        num_noise_variants_per_clean_epoch=NUM_NOISE_VARIANTS # Pass the new parameter
    )
    test_loader_overall = DataLoader(
        test_dataset_overall,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    all_rrmse_overall = []
    all_cc_overall = []
    clean_band_ratios_agg_overall = {band: [] for band in EEG_BANDS.keys()}
    noisy_band_ratios_agg_overall = {band: [] for band in EEG_BANDS.keys()}
    denoised_band_ratios_agg_overall = {band: [] for band in EEG_BANDS.keys()}

    with torch.no_grad():
        for batch_idx, (noisy_signals, clean_signals) in enumerate(tqdm(test_loader_overall, desc="Overall Metrics")):
            noisy_signals = noisy_signals.to(device)
            clean_signals = clean_signals.to(device)
            denoised_signals = generator(noisy_signals)

            noisy_signals_np = noisy_signals.cpu().numpy()
            clean_signals_np = clean_signals.cpu().numpy()
            denoised_signals_np = denoised_signals.cpu().numpy()

            for i in range(noisy_signals.shape[0]):
                all_rrmse_overall.append(calculate_rrmse(clean_signals_np[i], denoised_signals_np[i]))
                all_cc_overall.append(calculate_cc(clean_signals_np[i], denoised_signals_np[i]))
            
            clean_ratios_batch = calculate_band_power_ratios(clean_signals_np.squeeze(1), SAMPLING_RATE, EEG_BANDS)
            noisy_ratios_batch = calculate_band_power_ratios(noisy_signals_np.squeeze(1), SAMPLING_RATE, EEG_BANDS)
            denoised_ratios_batch = calculate_band_power_ratios(denoised_signals_np.squeeze(1), SAMPLING_RATE, EEG_BANDS)

            for band in EEG_BANDS.keys():
                clean_band_ratios_agg_overall[band].append(clean_ratios_batch[f'{band}_ratio'])
                noisy_band_ratios_agg_overall[band].append(noisy_ratios_batch[f'{band}_ratio'])
                denoised_band_ratios_agg_overall[band].append(denoised_ratios_batch[f'{band}_ratio'])
            
            if batch_idx == 0: # Only plot one example from the overall test set
                plot_evaluation_samples(
                    noisy_signals[:4],
                    clean_signals[:4],
                    denoised_signals[:4],
                    run_label="Overall Test Set Example",
                    num_plots=4,
                    save_path=os.path.join(EVAL_PLOTS_DIR, "overall_test_denoising_example.png")
                )
                print(f"Saved overall example denoising plot to '{os.path.join(EVAL_PLOTS_DIR, 'overall_test_denoising_example.png')}'")
                
                # Plot PSD comparison for one sample from this batch
                # Use the new PSD_SAMPLE_INDEX_FOR_VIZ variable
                plot_psd_comparison(
                    clean_signals_np[PSD_SAMPLE_INDEX_FOR_VIZ, 0, :], # Access specific sample and channel
                    noisy_signals_np[PSD_SAMPLE_INDEX_FOR_VIZ, 0, :],
                    denoised_signals_np[PSD_SAMPLE_INDEX_FOR_VIZ, 0, :],
                    SAMPLING_RATE,
                    EEG_BANDS,
                    save_path=os.path.join(EVAL_PLOTS_DIR, "psd_comparison_example.png")
                )
                print(f"Saved PSD comparison plot to '{os.path.join(EVAL_PLOTS_DIR, 'psd_comparison_example.png')}'")


    avg_rrmse_overall = np.mean(all_rrmse_overall)
    avg_cc_overall = np.mean(all_cc_overall)

    print(f"Overall Average RRMSE on Test Set: {avg_rrmse_overall:.4f}")
    print(f"Overall Average Pearson's CC on Test Set: {avg_cc_overall:.4f}")

    print("\nOverall Average Band Power Ratios (Clean vs. Noisy vs. Denoised):")
    for band in EEG_BANDS.keys():
        avg_clean_ratio = np.mean(clean_band_ratios_agg_overall[band])
        avg_noisy_ratio = np.mean(noisy_band_ratios_agg_overall[band])
        avg_denoised_ratio = np.mean(denoised_band_ratios_agg_overall[band])
        print(f"  {band.capitalize()} Band:")
        print(f"    Clean: {avg_clean_ratio:.4f}")
        print(f"    Noisy: {avg_noisy_ratio:.4f}")
        print(f"    Denoised: {avg_denoised_ratio:.4f}")
        
        labels = ['Clean', 'Noisy', 'Denoised']
        values = [avg_clean_ratio, avg_noisy_ratio, avg_denoised_ratio]
        
        plt.figure(figsize=(6, 4))
        plt.bar(labels, values, color=['blue', 'red', 'green'])
        plt.title(f'Overall {band.capitalize()} Band Power Ratio')
        plt.ylabel('Power Ratio')
        plt.ylim(0, 1)
        plt.savefig(os.path.join(EVAL_PLOTS_DIR, f'overall_{band}_power_ratio_bar.png'))
        plt.close()
        print(f"    Saved overall {band.capitalize()} band power ratio plot to '{os.path.join(EVAL_PLOTS_DIR, f'overall_{band}_power_ratio_bar.png')}'")

    # Report Cosine Similarity of Power Ratios at -14dB
    if cosine_sim_power_ratios_at_neg14db:
        avg_cosine_sim_at_neg14db = np.mean(cosine_sim_power_ratios_at_neg14db)
        print(f"\nAverage Cosine Similarity of Power Ratios (Clean vs. Denoised) at -14dB: {avg_cosine_sim_at_neg14db:.4f}")


if __name__ == "__main__":
    main()
