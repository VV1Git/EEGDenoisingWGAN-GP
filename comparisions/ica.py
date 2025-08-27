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

import torch  # Add torch for GPU acceleration

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
    fig, axes = plt.subplots(num, 1, figsize=(10, 3 * num), sharex=True)
    # No suptitle
    if num == 1:
        axes = [axes]
    for idx, (snr, noisy, clean, denoised) in enumerate(zip(snrs, noisy_samples, clean_samples, denoised_samples)):
        ax = axes[idx]
        ax.plot(clean, label='Clean EEG', color='blue', alpha=0.7)
        ax.plot(noisy, label='Noisy EEG', color='red', linestyle='--', alpha=0.7)
        ax.plot(denoised, label='ICA Denoised EEG', color='green', linestyle='-', alpha=0.8)
        ax.set_title("ICA", fontsize=24)  # Method name as title
        ax.set_xlabel("Sample Index", fontsize=18)
        ax.set_ylabel("Amplitude", fontsize=18)
        ax.legend()
        ax.grid(True)
        # ax.set_yscale('log')  # Remove log scale
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(save_path)
    plt.close(fig)

def calculate_rrmse_spectral(denoised_signal, clean_signal, sampling_rate):
    """
    Calculates the Relative Root Mean Squared Error (RRMSE) in the spectral domain.
    Compares the PSDs of the clean and denoised signals.
    """
    if clean_signal.ndim > 1: clean_signal = clean_signal.flatten()
    if denoised_signal.ndim > 1: denoised_signal = denoised_signal.flatten()
    f_clean, Pxx_clean = welch(clean_signal, fs=sampling_rate, nperseg=sampling_rate, return_onesided=True)
    f_denoised, Pxx_denoised = welch(denoised_signal, fs=sampling_rate, nperseg=sampling_rate, return_onesided=True)
    if not np.array_equal(f_clean, f_denoised):
        raise ValueError("Frequency bins for clean and denoised PSDs do not match.")
    rmse_psd = np.sqrt(np.mean((Pxx_clean - Pxx_denoised)**2))
    rms_clean_psd = np.sqrt(np.mean(Pxx_clean**2))
    if rms_clean_psd == 0:
        return float('inf') if rmse_psd > 0 else 0.0
    return rmse_psd / rms_clean_psd

def calculate_band_power_ratios(signal, sampling_rate, bands):
    f, Pxx = welch(signal, fs=sampling_rate, nperseg=sampling_rate, return_onesided=True, axis=-1)
    total_power = np.trapz(Pxx, f, axis=-1)
    band_ratios = {}
    for band_name, (low_freq, high_freq) in bands.items():
        freq_mask = (f >= low_freq) & (f <= high_freq)
        band_power = np.trapz(Pxx[..., freq_mask], f[freq_mask], axis=-1)
        ratio = np.where(total_power == 0, 0, band_power / total_power)
        band_ratios[f'{band_name}_ratio'] = np.mean(ratio) if np.ndim(ratio) > 0 else ratio
    return band_ratios

def plot_psd_comparison(clean_signal_np, noisy_signal_np, denoised_signal_np, sampling_rate, bands, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle("ICA", fontsize=24)  # Add overall title
    titles = ["Clean", "Noisy", "Denoised"]
    signals = [clean_signal_np.flatten(), noisy_signal_np.flatten(), denoised_signal_np.flatten()]
    band_colors = {
        'delta': 'yellow',
        'theta': 'orange',
        'alpha': 'lightgreen',
        'beta': 'skyblue',
        'gamma': 'plum'
    }
    for i, (ax, signal, subtitle) in enumerate(zip(axes, signals, titles)):
        f, Pxx = welch(signal, fs=sampling_rate, nperseg=sampling_rate, return_onesided=True)
        ax.plot(f, Pxx, color='blue')
        ax.set_title(subtitle)
        ax.set_xlabel('Frequency (Hz)', fontsize=18)
        if i == 0:
            ax.set_ylabel('Power (V**2/Hz)', fontsize=18)
        for band_name, (low_freq, high_freq) in bands.items():
            ax.axvspan(low_freq, high_freq, color=band_colors[band_name], alpha=0.3, label=band_name.capitalize())
        ax.set_xlim(0, 80)
        ax.grid(True, linestyle=':', alpha=0.6)
        # ax.set_yscale('log')  # Remove log scale
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            sorted_labels = [b.capitalize() for b in bands.keys()]
            order = [labels.index(l) for l in sorted_labels if l in labels]
            ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper right', fontsize=15)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()

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
    # Move data to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_torch = torch.from_numpy(x).float().to(device)
    X = torch.stack([torch.roll(x_torch, shifts=shift, dims=0) for shift in range(n_components)], dim=1)
    X_cpu = X.cpu().numpy()  # FastICA only supports CPU, so move back to CPU

    ica = FastICA(n_components=n_components, random_state=0, max_iter=10000, tol=0.1)
    S_ = ica.fit_transform(X_cpu)
    # Remove the component with the highest absolute kurtosis (likely artifact)
    kurt = np.abs(np.apply_along_axis(lambda s: np.mean((s - np.mean(s))**4) / (np.var(s)**2), 0, S_))
    artifact_idx = np.argmax(kurt)
    S_[:, artifact_idx] = 0
    X_denoised = ica.inverse_transform(S_)
    # Return the first channel (original signal)
    denoised = X_denoised[:, 0]
    # Move result to GPU if available, then back to CPU numpy for further processing
    denoised_torch = torch.from_numpy(denoised).float().to(device)
    return denoised_torch.cpu().numpy()

# --- Main ICA Evaluation Logic ---
def main():
    ICA_EVAL_PLOTS_DIR = os.path.join("comparisions", "ica_evaluation_plots")
    os.makedirs(ICA_EVAL_PLOTS_DIR, exist_ok=True)
    print(f"Created/Ensured '{ICA_EVAL_PLOTS_DIR}' directory exists for ICA evaluation plots.")

    # --- Shared sample for all methods ---
    SHARED_SAMPLE_PATH = os.path.join(ICA_EVAL_PLOTS_DIR, "..", "shared_sample_denoising_-6.npz")

    # Load data
    clean_eeg_all, eog_noise, emg_noise = prepare_eeg_data(
        EEG_FILE, EOG_FILE, EMG_FILE, [-100, -100]
    )
    SAMPLES_PER_EPOCH = clean_eeg_all.shape[1]
    _, test_clean_eeg_np = train_test_split(
        clean_eeg_all, test_size=(1 - TRAIN_SPLIT_RATIO), random_state=42
    )

    # --- Generate or load shared sample for -6dB ---
    if not os.path.exists(SHARED_SAMPLE_PATH):
        # Pick a fixed clean and noise epoch for all methods
        np.random.seed(42)
        clean_idx = 0
        clean_epoch = test_clean_eeg_np[clean_idx].astype(np.float64).flatten()
        noise_type = 'both'
        eog = eog_noise[0] if eog_noise is not None else np.zeros_like(clean_epoch)
        emg = emg_noise[0] if emg_noise is not None else np.zeros_like(clean_epoch)
        noise_epoch = eog + emg
        # SNR -6 dB
        snr_db = -6
        clean_power = np.mean(clean_epoch**2)
        noise_power = np.mean(noise_epoch**2)
        snr_linear = 10**(snr_db / 10)
        alpha = np.sqrt(clean_power / (snr_linear * noise_power)) if noise_power > 0 else 0
        noisy_signal = clean_epoch + alpha * noise_epoch
        # Remove mean
        noisy_signal = noisy_signal - np.mean(noisy_signal)
        clean_epoch = clean_epoch - np.mean(clean_epoch)
        np.savez(SHARED_SAMPLE_PATH, clean=clean_epoch, noisy=noisy_signal)
        print(f"Saved shared sample for -6dB to {SHARED_SAMPLE_PATH}")
    else:
        arr = np.load(SHARED_SAMPLE_PATH)
        clean_epoch = arr["clean"]
        noisy_signal = arr["noisy"]

    snr_values_db = SNR_RANGE_DB_EVAL
    rrmse_temporal_per_snr = []
    rrmse_spectral_per_snr = []
    cc_per_snr = []
    band_power_ratios_per_snr = {band: {'clean': [], 'noisy': [], 'denoised': []} for band in EEG_BANDS.keys()}
    snr_samples = []
    example_psd_saved = False
    sample_saved_for_minus6db = False

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
        batch_rrmse_spectral = []
        batch_cc = []
        clean_band_ratios_agg = {band: [] for band in EEG_BANDS.keys()}
        noisy_band_ratios_agg = {band: [] for band in EEG_BANDS.keys()}
        denoised_band_ratios_agg = {band: [] for band in EEG_BANDS.keys()}
        example_saved = False

        for batch_idx, (noisy_signals, clean_signals) in enumerate(tqdm(test_loader_current_snr, desc=f"SNR {current_snr_db}dB")):
            noisy_signals_np = noisy_signals.numpy()
            clean_signals_np = clean_signals.numpy()
            for i in range(noisy_signals_np.shape[0]):
                noisy = noisy_signals_np[i, 0, :]
                clean = clean_signals_np[i, 0, :]
                denoised = ica_denoise(noisy)
                batch_rrmse_temporal.append(calculate_rrmse(clean, denoised))
                batch_rrmse_spectral.append(calculate_rrmse_spectral(denoised, clean, SAMPLING_RATE))
                batch_cc.append(calculate_cc(clean, denoised))
                # Band power ratios
                clean_ratios = calculate_band_power_ratios(clean, SAMPLING_RATE, EEG_BANDS)
                noisy_ratios = calculate_band_power_ratios(noisy, SAMPLING_RATE, EEG_BANDS)
                denoised_ratios = calculate_band_power_ratios(denoised, SAMPLING_RATE, EEG_BANDS)
                for band in EEG_BANDS.keys():
                    clean_band_ratios_agg[band].append(clean_ratios[f'{band}_ratio'])
                    noisy_band_ratios_agg[band].append(noisy_ratios[f'{band}_ratio'])
                    denoised_band_ratios_agg[band].append(denoised_ratios[f'{band}_ratio'])
                if not example_saved:
                    # Only save -6dB sample for multi_snr_sample_denoising
                    if current_snr_db == -6 and not sample_saved_for_minus6db:
                        snr_samples = [(current_snr_db, noisy, clean, denoised)]
                        sample_saved_for_minus6db = True
                    # Save PSD plot for the first SNR only
                    if not example_psd_saved:
                        plot_psd_comparison(
                            clean, noisy, denoised, SAMPLING_RATE, EEG_BANDS,
                            save_path=os.path.join(ICA_EVAL_PLOTS_DIR, "psd_comparison_example.png")
                        )
                        example_psd_saved = True
                    example_saved = True

        rrmse_temporal_per_snr.append(np.mean(batch_rrmse_temporal))
        rrmse_spectral_per_snr.append(np.mean(batch_rrmse_spectral))
        cc_per_snr.append(np.mean(batch_cc))
        for band in EEG_BANDS.keys():
            band_power_ratios_per_snr[band]['clean'].append(np.mean(clean_band_ratios_agg[band]))
            band_power_ratios_per_snr[band]['noisy'].append(np.mean(noisy_band_ratios_agg[band]))
            band_power_ratios_per_snr[band]['denoised'].append(np.mean(denoised_band_ratios_agg[band]))

    # Band power ratio bar plots for each band
    x = np.arange(len(snr_values_db))
    width = 0.25
    for band in EEG_BANDS.keys():
        plt.figure(figsize=(10, 6))
        clean_vals = band_power_ratios_per_snr[band]['clean']
        noisy_vals = band_power_ratios_per_snr[band]['noisy']
        denoised_vals = band_power_ratios_per_snr[band]['denoised']
        max_val = max(
            max(clean_vals) if clean_vals else 0,
            max(noisy_vals) if noisy_vals else 0,
            max(denoised_vals) if denoised_vals else 0,
        )
        plt.bar(x - width, clean_vals, width, label='Clean', color='blue')
        plt.bar(x, noisy_vals, width, label='Noisy', color='red')
        plt.bar(x + width, denoised_vals, width, label='Denoised', color='green')
        plt.title("ICA", fontsize=24)
        plt.xlabel('SNR (dB)', fontsize=18)
        plt.ylabel('Power Ratio', fontsize=18)
        plt.ylim(0, max_val * 1.05 if max_val > 0 else 1)
        plt.xticks(x, [str(snr) for snr in snr_values_db])
        plt.grid(axis='y')
        plt.legend()
        fname = f'overall_{band}_power_ratio_vs_snr.png'
        plt.savefig(os.path.join(ICA_EVAL_PLOTS_DIR, fname))
        plt.close()
        print(f"Saved overall {band.capitalize()} band power ratio vs SNR bar chart to '{os.path.join(ICA_EVAL_PLOTS_DIR, fname)}'")

    # Only save the -6dB grouped sample plot
    if True:  # Always save the shared sample
        denoised = ica_denoise(noisy_signal)
        sample_txt_path = os.path.join(ICA_EVAL_PLOTS_DIR, "sample_denoising_-6.txt")
        with open(sample_txt_path, "w") as f:
            f.write("Index\tClean\tNoisy\tDenoised\n")
            for i in range(len(clean_epoch)):
                f.write(f"{i}\t{clean_epoch[i]}\t{noisy_signal[i]}\t{denoised[i]}\n")
        print(f"Saved sample denoising signals to '{sample_txt_path}'")

    # RRMSE Temporal vs SNR
    plt.figure(figsize=(6, 5))
    plt.plot(snr_values_db, rrmse_temporal_per_snr, marker='o', linestyle='-', color='blue')
    plt.title('ICA', fontsize=24)
    plt.xlabel('SNR (dB)', fontsize=18)
    plt.ylabel('RRMSE Temporal', fontsize=18)
    plt.grid(True)
    plt.savefig(os.path.join(ICA_EVAL_PLOTS_DIR, 'RRMSE_Temporal_vs_SNR.png'))
    plt.close()

    # RRMSE Spectral vs SNR
    plt.figure(figsize=(6, 5))
    plt.plot(snr_values_db, rrmse_spectral_per_snr, marker='o', linestyle='-', color='blue')
    plt.title('ICA', fontsize=24)
    plt.xlabel('SNR (dB)', fontsize=18)
    plt.ylabel('RRMSE Spectral', fontsize=18)
    plt.grid(True)
    plt.savefig(os.path.join(ICA_EVAL_PLOTS_DIR, 'RRMSE_Spectral_vs_SNR.png'))
    plt.close()

    # CC vs SNR
    plt.figure(figsize=(6, 5))
    plt.plot(snr_values_db, cc_per_snr, marker='o', linestyle='-', color='blue')
    plt.title('ICA', fontsize=24)
    plt.xlabel('SNR (dB)', fontsize=18)
    plt.ylabel('Pearson\'s CC', fontsize=18)
    plt.grid(True)
    plt.savefig(os.path.join(ICA_EVAL_PLOTS_DIR, 'CC_vs_SNR.png'))
    plt.close()

    # --- Print summary statistics at the end ---
    print("\n--- Summary Statistics Across SNRs ---")
    print(f"Average CC across SNRs: {np.mean(cc_per_snr):.4f} ± {np.std(cc_per_snr):.4f}")
    print(f"Average RRMSE (Temporal) across SNRs: {np.mean(rrmse_temporal_per_snr):.4f} ± {np.std(rrmse_temporal_per_snr):.4f}")
    print(f"Average RRMSE (Spectral) across SNRs: {np.mean(rrmse_spectral_per_snr):.4f} ± {np.std(rrmse_spectral_per_snr):.4f}")

    print("\nPSD Ratio (Denoised/Clean) across SNRs for each frequency band:")
    for band in EEG_BANDS.keys():
        denoised = np.array(band_power_ratios_per_snr[band]['denoised'])
        clean = np.array(band_power_ratios_per_snr[band]['clean'])
        ratio = denoised / (clean + 1e-12)  # avoid division by zero
        print(f"  {band.capitalize()}: Mean={np.mean(ratio):.4f}, Std={np.std(ratio):.4f}")

    # --- Save CC and RRMSE vs SNR data to text files for overlay plotting ---
    cc_txt_path = os.path.join(ICA_EVAL_PLOTS_DIR, "cc_vs_snr.txt")
    rrmse_txt_path = os.path.join(ICA_EVAL_PLOTS_DIR, "rrmse_vs_snr.txt")
    rrmse_spectral_txt_path = os.path.join(ICA_EVAL_PLOTS_DIR, "rrmse_spectral_vs_snr.txt")
    with open(cc_txt_path, "w") as f:
        f.write("SNR_dB\tCC\n")
        for snr, cc in zip(snr_values_db, cc_per_snr):
            f.write(f"{snr}\t{cc}\n")
    with open(rrmse_txt_path, "w") as f:
        f.write("SNR_dB\tRRMSE\n")
        for snr, rrmse in zip(snr_values_db, rrmse_temporal_per_snr):
            f.write(f"{snr}\t{rrmse}\n")
    with open(rrmse_spectral_txt_path, "w") as f:
        f.write("SNR_dB\tRRMSE_Spectral\n")
        for snr, rrmse_spec in zip(snr_values_db, rrmse_spectral_per_snr):
            f.write(f"{snr}\t{rrmse_spec}\n")
    print(f"Saved CC vs SNR data to '{cc_txt_path}'")
    print(f"Saved RRMSE vs SNR data to '{rrmse_txt_path}'")
    print(f"Saved RRMSE Spectral vs SNR data to '{rrmse_spectral_txt_path}'")

    # --- Grouped bar chart: average power ratios for each band (ICA, half AR-WGAN width) ---
    band_names = list(EEG_BANDS.keys())
    avg_clean = [np.mean(band_power_ratios_per_snr[band]['clean']) for band in band_names]
    avg_noisy = [np.mean(band_power_ratios_per_snr[band]['noisy']) for band in band_names]
    avg_denoised = [np.mean(band_power_ratios_per_snr[band]['denoised']) for band in band_names]

    x = np.arange(len(band_names))
    width = 0.25
    plt.figure(figsize=(7, 8))  # ICA: half as wide as AR-WGAN
    plt.bar(x - width, avg_clean, width, label='Clean', color='blue')
    plt.bar(x, avg_noisy, width, label='Noisy', color='red')
    plt.bar(x + width, avg_denoised, width, label='Denoised', color='green')
    plt.title("ICA", fontsize=24)
    plt.xlabel('EEG Band', fontsize=18)
    plt.ylabel('Average Power Ratio', fontsize=18)
    plt.xticks(x, [b.capitalize() for b in band_names])
    plt.ylim(0, max(avg_clean + avg_noisy + avg_denoised) * 1.05)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(ICA_EVAL_PLOTS_DIR, "overall_band_power_ratios_grouped.png"))
    plt.close()
    print(f"Saved grouped band power ratio bar chart to '{os.path.join(ICA_EVAL_PLOTS_DIR, 'overall_band_power_ratios_grouped.png')}'")

    print("\nICA evaluation complete.")

if __name__ == "__main__":
    main()
