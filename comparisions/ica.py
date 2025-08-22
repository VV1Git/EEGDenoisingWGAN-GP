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
    fig, axes = plt.subplots(num, 1, figsize=(10, 3 * num), sharex=True)
    if num == 1:
        axes = [axes]
    for idx, (snr, noisy, clean, denoised) in enumerate(zip(snrs, noisy_samples, clean_samples, denoised_samples)):
        ax = axes[idx]
        ax.plot(clean, label='Clean EEG', color='blue', alpha=0.7)
        ax.plot(noisy, label='Noisy EEG', color='red', linestyle='--', alpha=0.7)
        ax.plot(denoised, label='ICA Denoised EEG', color='green', linestyle='-', alpha=0.8)
        ax.set_title("ICA")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
        ax.set_title("ICA")
        ax.set_xlabel('Frequency (Hz)')
        if i == 0:
            ax.set_ylabel('Power (V**2/Hz)')
        for band_name, (low_freq, high_freq) in bands.items():
            ax.axvspan(low_freq, high_freq, color=band_colors[band_name], alpha=0.3, label=band_name.capitalize())
        ax.set_xlim(0, 80)
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            sorted_labels = [b.capitalize() for b in bands.keys()]
            order = [labels.index(l) for l in sorted_labels if l in labels]
            ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper right', fontsize='small')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
    rrmse_spectral_per_snr = []
    cc_per_snr = []
    band_power_ratios_per_snr = {band: {'clean': [], 'noisy': [], 'denoised': []} for band in EEG_BANDS.keys()}
    snr_samples = []
    example_psd_saved = False

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
                    snr_samples.append((current_snr_db, noisy, clean, denoised))
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
        plt.title("ICA")
        plt.xlabel('SNR (dB)')
        plt.ylabel('Power Ratio')
        plt.ylim(0, max_val * 1.05 if max_val > 0 else 1)
        plt.xticks(x, [str(snr) for snr in snr_values_db])
        plt.grid(axis='y')
        plt.legend()
        fname = f'overall_{band}_power_ratio_vs_snr.png'
        plt.savefig(os.path.join(ICA_EVAL_PLOTS_DIR, fname))
        plt.close()
        print(f"Saved overall {band.capitalize()} band power ratio vs SNR bar chart to '{os.path.join(ICA_EVAL_PLOTS_DIR, fname)}'")

    # Grouped sample plots for SNRs in pairs (14 images for 14 SNRs, 2 per plot)
    group_size = 2
    for idx in range(0, len(snr_samples), group_size):
        group = snr_samples[idx:idx+group_size]
        snrs = [item[0] for item in group]
        noisy_samples = [item[1] for item in group]
        clean_samples = [item[2] for item in group]
        denoised_samples = [item[3] for item in group]
        save_path = os.path.join(ICA_EVAL_PLOTS_DIR, f"multi_snr_sample_denoising_{'_'.join(str(s) for s in snrs)}.png")
        plot_multi_snr_samples(snrs, noisy_samples, clean_samples, denoised_samples, save_path)
        print(f"Saved grouped sample denoising plot for SNRs {snrs} to '{save_path}'")

    # RRMSE Temporal vs SNR
    plt.figure(figsize=(6, 5))
    plt.plot(snr_values_db, rrmse_temporal_per_snr, marker='o', linestyle='-', color='blue')
    plt.title('RRMSE Temporal vs. Input SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('RRMSE Temporal')
    plt.grid(True)
    plt.savefig(os.path.join(ICA_EVAL_PLOTS_DIR, 'RRMSE_Temporal_vs_SNR.png'))
    plt.close()

    # RRMSE Spectral vs SNR
    plt.figure(figsize=(6, 5))
    plt.plot(snr_values_db, rrmse_spectral_per_snr, marker='o', linestyle='-', color='blue')
    plt.title('RRMSE Spectral vs. Input SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('RRMSE Spectral')
    plt.grid(True)
    plt.savefig(os.path.join(ICA_EVAL_PLOTS_DIR, 'RRMSE_Spectral_vs_SNR.png'))
    plt.close()

    # CC vs SNR
    plt.figure(figsize=(6, 5))
    plt.plot(snr_values_db, cc_per_snr, marker='o', linestyle='-', color='blue')
    plt.title('Pearson\'s CC vs. Input SNR')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Pearson\'s CC')
    plt.grid(True)
    plt.savefig(os.path.join(ICA_EVAL_PLOTS_DIR, 'CC_vs_SNR.png'))
    plt.close()

    print("\nICA evaluation complete.")

if __name__ == "__main__":
    main()
