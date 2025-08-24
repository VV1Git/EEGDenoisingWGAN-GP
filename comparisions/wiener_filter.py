import numpy as np # type:ignore
import scipy.signal as signal # type:ignore
import matplotlib.pyplot as plt # type:ignore
from scipy.signal import welch # type:ignore
from os.path import join 
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eeg_data_generator import prepare_eeg_data, EEGNoiseDataset
from variables import SNR_RANGE_DB_EVAL, EVAL_PLOTS_DIR, EEG_BANDS, SAMPLING_RATE

# --- Variables from variables.py ---
# This section contains the relevant variables copied from your project's variables.py.
# This makes the script self-contained and runnable on its own.



# --- Denoising Function ---
def wiener_filter_scipy(noisy_signal, mysize=None, noise=None):
    """
    Applies a Wiener filter to a 1D signal using scipy's built-in function.

    The Wiener filter minimizes the mean-squared error between the original
    signal and the filtered signal. It works by estimating a clean signal from
    a noisy one based on the statistical properties of the signal and noise.

    Args:
        noisy_signal (np.ndarray): The 1D signal to be denoised.
        mysize (int or tuple, optional): A scalar or tuple of integers that
                                         specifies the size of the neighborhood
                                         over which local noise estimation is
                                         performed.
        noise (float, optional): The noise power to use. If None, it is estimated
                                 from the local variance.

    Returns:
        np.ndarray: The denoised signal.
    """
    # Ensure input is 1D float64
    noisy_signal = np.asarray(noisy_signal, dtype=np.float64).flatten()
    # Use a reasonable window size for EEG (e.g., 31 samples ~0.12s at 256Hz)
    if mysize is None:
        mysize = 31
    denoised_signal = signal.wiener(noisy_signal, mysize=mysize, noise=noise)
    return denoised_signal


# --- Utility Functions for Evaluation Metrics ---

def calculate_cc(denoised_signal, clean_signal):
    """
    Calculates the Correlation Coefficient (CC) between two signals.
    A higher CC indicates better denoising performance.

    Args:
        denoised_signal (np.ndarray): The denoised signal.
        clean_signal (np.ndarray): The ground-truth clean signal.

    Returns:
        float: The correlation coefficient.
    """
    if denoised_signal.shape != clean_signal.shape:
        raise ValueError("Signals must have the same shape.")
    
    # Calculate the Pearson correlation coefficient
    corr_matrix = np.corrcoef(denoised_signal, clean_signal)
    return corr_matrix[0, 1]

def calculate_rrmse(denoised_signal, clean_signal):
    """
    Calculates the Relative Root Mean Squared Error (RRMSE).
    A lower RRMSE indicates better denoising performance.

    Args:
        denoised_signal (np.ndarray): The denoised signal.
        clean_signal (np.ndarray): The ground-truth clean signal.

    Returns:h
        float: The RRMSE value.
    """
    mse = np.mean((denoised_signal - clean_signal)**2)
    rmse = np.sqrt(mse)
    clean_signal_rms = np.sqrt(np.mean(clean_signal**2))
    
    # Avoid division by zero
    if clean_signal_rms == 0:
        return float('inf')
        
    return rmse / clean_signal_rms

def calculate_psd(signal, fs):
    """
    Calculates the Power Spectral Density (PSD) of a signal using Welch's method.

    Args:
        signal (np.ndarray): The input signal.
        fs (float): The sampling frequency.

    Returns:
        tuple: A tuple containing the frequency array and the PSD array.
    """
    # nperseg is the number of data points per segment
    f, Pxx = welch(signal, fs=fs, nperseg=256, noverlap=128)
    return f, Pxx

def calculate_band_power(psd_freqs, psd_data, band):
    """
    Calculates the power within a specific EEG frequency band.
    
    Args:
        psd_freqs (np.ndarray): The frequency array from PSD.
        psd_data (np.ndarray): The PSD array.
        band (str): The name of the band (e.g., 'alpha').
        
    Returns:
        float: The average power in the specified band.
    """
    if band not in EEG_BANDS:
        raise ValueError(f"Unknown band: {band}. Must be one of {list(EEG_BANDS.keys())}")
        
    low_freq, high_freq = EEG_BANDS[band]
    band_indices = np.where((psd_freqs >= low_freq) & (psd_freqs <= high_freq))
    band_power = np.trapz(psd_data[band_indices], psd_freqs[band_indices])
    return band_power

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

# --- Plotting Utility Functions ---

def create_and_save_plot(x, y, xlabel, ylabel, title, filename):
    """
    Creates a line plot and saves it to the specified directory.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title("Wiener Filter", fontsize=24, y=1.01)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.xticks()
    plt.yticks()
    
    # Ensure the directory exists
    os.makedirs(EVAL_PLOTS_DIR, exist_ok=True)
    save_path = join(EVAL_PLOTS_DIR, filename)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close() # Close the plot figure to free up memory

def create_and_save_psd_plot(signals_dict, fs, title, filename):
    """
    Creates a PSD plot comparing multiple signals and saves it.
    """
    plt.figure(figsize=(10, 6))
    for name, signal_data in signals_dict.items():
        f, Pxx = calculate_psd(signal_data, fs)
        plt.plot(f, Pxx, label=name)  # No log scale
    plt.title("Wiener Filter", fontsize=24, y=1.01)
    plt.xlabel('Frequency (Hz)', fontsize=18)
    plt.ylabel('Power/Frequency (dB/Hz)', fontsize=18)
    plt.legend()
    plt.grid(True)
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    # plt.yscale('log')  # Removed log scale
    plt.xticks()
    plt.yticks()
    save_path = join(EVAL_PLOTS_DIR, filename)
    plt.savefig(save_path)
    print(f"PSD plot saved to {save_path}")
    plt.close()

def create_and_save_band_power_plots(band_power_data, bands, title, filename_prefix):
    """
    Creates bar plots for band power ratios and saves them.
    """
    os.makedirs(EVAL_PLOTS_DIR, exist_ok=True)
    
    for band in bands:
        noisy_ratios = [data[0] for data in band_power_data[band]]
        denoised_ratios = [data[1] for data in band_power_data[band]]
        snrs = SNR_RANGE_DB_EVAL
        
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.35
        x = np.arange(len(snrs))
        
        ax.bar(x - width/2, noisy_ratios, width, label='Noisy')
        ax.bar(x + width/2, denoised_ratios, width, label='Denoised')
        
        ax.set_title("Wiener Filter", fontsize=24, y=1.01)
        ax.set_xlabel('SNR (dB)', fontsize=18)
        ax.set_ylabel(f'{band.capitalize()} Power Ratio', fontsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(snrs)
        ax.legend()
        plt.grid(True, axis='y')
        plt.tight_layout(rect=[0, 0.04, 1, 0.97])
        plt.xticks()
        plt.yticks()
        
        save_path = join(EVAL_PLOTS_DIR, f'overall_{band}_power_ratio.png')
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.close(fig)

def plot_multi_snr_samples(snrs, noisy_samples, clean_samples, denoised_samples, save_path):
    """
    Plots sample denoising results for multiple SNRs in a single figure.
    """
    num = len(snrs)
    fig, axes = plt.subplots(num, 1, figsize=(10, 3 * num), sharex=True)
    # No suptitle
    if num == 1:
        axes = [axes]
    for idx, (snr, noisy, clean, denoised) in enumerate(zip(snrs, noisy_samples, clean_samples, denoised_samples)):
        ax = axes[idx]
        ax.plot(clean, label='Clean EEG', color='blue', alpha=0.7)
        ax.plot(noisy, label='Noisy EEG', color='red', linestyle='--', alpha=0.7)
        ax.plot(denoised, label='Denoised EEG', color='green', linestyle='-', alpha=0.8)
        ax.set_title("Wiener Filter", fontsize=24)  # Method name as title
        ax.set_xlabel("Sample Index", fontsize=18)
        ax.set_ylabel("Amplitude", fontsize=18)
        ax.legend()
        ax.grid(True)
        # ax.set_yscale('log')  # Remove log scale
        plt.xticks()
        plt.yticks()
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.savefig(save_path)
    plt.close(fig)

# --- Main Logic for Generating Plots ---
if __name__ == "__main__":
    print(f"Generating evaluation plots and saving to '{EVAL_PLOTS_DIR}'...")

    # --- Load data using eeg_data_generator.py ---
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    EEG_FILE = os.path.join(DATA_DIR, "EEG_all_epochs.npy")
    EOG_FILE = os.path.join(DATA_DIR, "EOG_all_epochs.npy")
    EMG_FILE = os.path.join(DATA_DIR, "EMG_all_epochs.npy")
    SNR_RANGE_DB = [-14, 6]  # Min/max SNR for dataset, not used directly here

    # Set EVAL_PLOTS_DIR to be inside the comparisions folder
    EVAL_PLOTS_DIR = os.path.join(os.path.dirname(__file__), "wiener_evaluation_plots")
    os.makedirs(EVAL_PLOTS_DIR, exist_ok=True)

    clean_eeg_np, eog_noise_np, emg_noise_np = prepare_eeg_data(EEG_FILE, EOG_FILE, EMG_FILE, SNR_RANGE_DB)
    time_len_samples = clean_eeg_np.shape[1]

    # --- Metric storage for plotting ---
    cc_scores = []
    rrmse_temporal_scores = []
    rrmse_spectral_scores = []
    band_power_ratios_per_snr = {band: {'clean': [], 'noisy': [], 'denoised': []} for band in EEG_BANDS.keys()}
    snr_samples = []  # List of (snr, noisy, clean, denoised)
    example_psd_saved = False
    sample_saved_for_minus6db = False

    for snr_db in SNR_RANGE_DB_EVAL:
        cc_list = []
        rrmse_list = []
        rrmse_spectral_list = []
        clean_band_ratios_agg = {band: [] for band in EEG_BANDS.keys()}
        noisy_band_ratios_agg = {band: [] for band in EEG_BANDS.keys()}
        denoised_band_ratios_agg = {band: [] for band in EEG_BANDS.keys()}
        example_saved = False

        for i in range(clean_eeg_np.shape[0]):
            clean_epoch = clean_eeg_np[i].astype(np.float64).flatten()
            noise_type = np.random.choice(['eog', 'emg', 'both'])
            if noise_type == 'eog' and eog_noise_np is not None:
                noise_epoch = eog_noise_np[np.random.randint(len(eog_noise_np))]
            elif noise_type == 'emg' and emg_noise_np is not None:
                noise_epoch = emg_noise_np[np.random.randint(len(emg_noise_np))]
            else:
                eog = eog_noise_np[np.random.randint(len(eog_noise_np))] if eog_noise_np is not None else np.zeros_like(clean_epoch)
                emg = emg_noise_np[np.random.randint(len(emg_noise_np))] if emg_noise_np is not None else np.zeros_like(clean_epoch)
                noise_epoch = eog + emg

            noise_epoch = np.asarray(noise_epoch, dtype=np.float64).flatten()
            if len(noise_epoch) != time_len_samples:
                if len(noise_epoch) > time_len_samples:
                    noise_epoch = noise_epoch[:time_len_samples]
                else:
                    noise_epoch = np.concatenate((noise_epoch, np.zeros(time_len_samples - len(noise_epoch))))

            clean_power = np.mean(clean_epoch**2)
            noise_power = np.mean(noise_epoch**2)
            if clean_power == 0 or noise_power == 0:
                continue
            snr_linear = 10**(snr_db / 10)
            alpha = np.sqrt(clean_power / (snr_linear * noise_power))
            noisy_signal = clean_epoch + alpha * noise_epoch

            noisy_signal = noisy_signal - np.mean(noisy_signal)
            clean_epoch = clean_epoch - np.mean(clean_epoch)
            denoised_signal = wiener_filter_scipy(noisy_signal, mysize=31)

            cc = calculate_cc(denoised_signal, clean_epoch)
            rrmse_temporal = calculate_rrmse(denoised_signal, clean_epoch)
            rrmse_spectral = calculate_rrmse_spectral(denoised_signal, clean_epoch, SAMPLING_RATE)
            cc_list.append(cc)
            rrmse_list.append(rrmse_temporal)
            rrmse_spectral_list.append(rrmse_spectral)

            # Band power ratios for clean, noisy, denoised
            psd_freqs, psd_noisy = calculate_psd(noisy_signal, SAMPLING_RATE)
            _, psd_denoised = calculate_psd(denoised_signal, SAMPLING_RATE)
            _, psd_clean = calculate_psd(clean_epoch, SAMPLING_RATE)
            total_power_clean = np.trapz(psd_clean, psd_freqs)
            total_power_noisy = np.trapz(psd_noisy, psd_freqs)
            total_power_denoised = np.trapz(psd_denoised, psd_freqs)
            for band in EEG_BANDS.keys():
                low, high = EEG_BANDS[band]
                band_mask = (psd_freqs >= low) & (psd_freqs <= high)
                clean_band = np.trapz(psd_clean[band_mask], psd_freqs[band_mask])
                noisy_band = np.trapz(psd_noisy[band_mask], psd_freqs[band_mask])
                denoised_band = np.trapz(psd_denoised[band_mask], psd_freqs[band_mask])
                clean_ratio = clean_band / total_power_clean if total_power_clean > 0 else 0
                noisy_ratio = noisy_band / total_power_noisy if total_power_noisy > 0 else 0
                denoised_ratio = denoised_band / total_power_denoised if total_power_denoised > 0 else 0
                clean_band_ratios_agg[band].append(clean_ratio)
                noisy_band_ratios_agg[band].append(noisy_ratio)
                denoised_band_ratios_agg[band].append(denoised_ratio)

            # Save one example per SNR for grouped plots and PSD
            if not example_saved:
                # Only save -6dB sample for multi_snr_sample_denoising
                if snr_db == -6 and not sample_saved_for_minus6db:
                    snr_samples = [(snr_db, noisy_signal, clean_epoch, denoised_signal)]
                    sample_saved_for_minus6db = True
                # Save PSD and grouped plot for the first SNR only (like evaluate.py)
                if not example_psd_saved:
                    # PSD comparison plot
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
                    fig.suptitle('Wiener Filter', fontsize=24)  # Add overall title
                    signal_types = {
                        'Clean Signal': clean_epoch,
                        'Contaminated Signal': noisy_signal,
                        'Denoised Signal': denoised_signal
                    }
                    band_colors = {
                        'delta': 'yellow',
                        'theta': 'orange',
                        'alpha': 'lightgreen',
                        'beta': 'skyblue',
                        'gamma': 'plum'
                    }
                    for j, (title, sig) in enumerate(signal_types.items()):
                        ax = axes[j]
                        f, Pxx = calculate_psd(sig, SAMPLING_RATE)
                        ax.plot(f, Pxx, color='blue')
                        ax.set_title(title)
                        ax.set_xlabel('Frequency (Hz)', fontsize=18)
                        if j == 0:
                            ax.set_ylabel('Power (V**2/Hz)', fontsize=18)
                        for band_name, (low, high) in EEG_BANDS.items():
                            ax.axvspan(low, high, color=band_colors[band_name], alpha=0.3, label=band_name.capitalize())
                        ax.set_xlim(0, 80)
                        ax.grid(True, linestyle=':', alpha=0.6)
                        # ax.set_yscale('log')  # Remove log scale
                        if j == 0:
                            handles, labels = ax.get_legend_handles_labels()
                            sorted_labels = [b.capitalize() for b in EEG_BANDS.keys()]
                            order = [labels.index(l) for l in sorted_labels if l in labels]
                            ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper right', fontsize='small')
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plt.savefig(join(EVAL_PLOTS_DIR, "psd_comparison_example.png"))
                    plt.close(fig)
                    example_psd_saved = True
                example_saved = True

        cc_scores.append(np.mean(cc_list))
        rrmse_temporal_scores.append(np.mean(rrmse_list))
        rrmse_spectral_scores.append(np.mean(rrmse_spectral_list))
        for band in EEG_BANDS.keys():
            band_power_ratios_per_snr[band]['clean'].append(np.mean(clean_band_ratios_agg[band]))
            band_power_ratios_per_snr[band]['noisy'].append(np.mean(noisy_band_ratios_agg[band]))
            band_power_ratios_per_snr[band]['denoised'].append(np.mean(denoised_band_ratios_agg[band]))
        print(f"SNR: {snr_db} dB | CC: {cc_scores[-1]:.4f} | RRMSE: {rrmse_temporal_scores[-1]:.4f} | RRMSE Spectral: {rrmse_spectral_scores[-1]:.4f}")

    # --- Plotting: match evaluate.py exactly ---
    import numpy as np
    # RRMSE Temporal vs SNR
    plt.figure(figsize=(6, 5))
    plt.plot(SNR_RANGE_DB_EVAL, rrmse_temporal_scores, marker='o', linestyle='-', color='blue')
    plt.title("Wiener Filter", fontsize=24)
    plt.xlabel('SNR (dB)', fontsize=18)
    plt.ylabel('RRMSE Temporal', fontsize=18)
    plt.grid(True)
    plt.savefig(join(EVAL_PLOTS_DIR, 'RRMSE_Temporal_vs_SNR.png'))
    plt.close()

    # RRMSE Spectral vs SNR
    plt.figure(figsize=(6, 5))
    plt.plot(SNR_RANGE_DB_EVAL, rrmse_spectral_scores, marker='o', linestyle='-', color='blue')
    plt.title("Wiener Filter", fontsize=24)
    plt.xlabel('SNR (dB)', fontsize=18)
    plt.ylabel('RRMSE Spectral', fontsize=18)
    plt.grid(True)
    plt.savefig(join(EVAL_PLOTS_DIR, 'RRMSE_Spectral_vs_SNR.png'))
    plt.close()

    # CC vs SNR
    plt.figure(figsize=(6, 5))
    plt.plot(SNR_RANGE_DB_EVAL, cc_scores, marker='o', linestyle='-', color='blue')
    plt.title("Wiener Filter", fontsize=24)
    plt.xlabel('SNR (dB)', fontsize=18)
    plt.ylabel('Pearson\'s CC', fontsize=18)
    plt.grid(True)
    plt.savefig(join(EVAL_PLOTS_DIR, 'CC_vs_SNR.png'))
    plt.close()

    # Band power ratio bar plots for each band
    x = np.arange(len(SNR_RANGE_DB_EVAL))
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
        plt.title("Wiener Filter", fontsize=24)
        plt.xlabel('SNR (dB)', fontsize=18)
        plt.ylabel('Power Ratio', fontsize=18)
        plt.ylim(0, max_val * 1.05 if max_val > 0 else 1)
        plt.xticks(x, [str(snr) for snr in SNR_RANGE_DB_EVAL])
        plt.grid(axis='y')
        plt.legend()
        fname = f'overall_{band}_power_ratio_vs_snr.png'
        plt.savefig(join(EVAL_PLOTS_DIR, fname))
        plt.close()
        print(f"Saved overall {band.capitalize()} band power ratio vs SNR bar chart to '{join(EVAL_PLOTS_DIR, fname)}'")

    # Only save the -6dB grouped sample plot
    if snr_samples:
        snrs = [item[0] for item in snr_samples]
        noisy_samples = [item[1] for item in snr_samples]
        clean_samples = [item[2] for item in snr_samples]
        denoised_samples = [item[3] for item in snr_samples]
        save_path = join(EVAL_PLOTS_DIR, f"multi_snr_sample_denoising_-6.png")
        plot_multi_snr_samples(snrs, noisy_samples, clean_samples, denoised_samples, save_path)
        print(f"Saved grouped sample denoising plot for SNR -6 dB to '{save_path}'")

    # --- Print summary statistics at the end ---
    print("\n--- Summary Statistics Across SNRs ---")
    print(f"Average CC across SNRs: {np.mean(cc_scores):.4f} ± {np.std(cc_scores):.4f}")
    print(f"Average RRMSE (Temporal) across SNRs: {np.mean(rrmse_temporal_scores):.4f} ± {np.std(rrmse_temporal_scores):.4f}")
    print(f"Average RRMSE (Spectral) across SNRs: {np.mean(rrmse_spectral_scores):.4f} ± {np.std(rrmse_spectral_scores):.4f}")

    print("\nPSD Ratio (Denoised/Clean) across SNRs for each frequency band:")
    for band in EEG_BANDS.keys():
        denoised = np.array(band_power_ratios_per_snr[band]['denoised'])
        clean = np.array(band_power_ratios_per_snr[band]['clean'])
        ratio = denoised / (clean + 1e-12)  # avoid division by zero
        print(f"  {band.capitalize()}: Mean={np.mean(ratio):.4f}, Std={np.std(ratio):.4f}")

    # --- Grouped bar chart: average power ratios for each band (Wiener, half AR-WGAN width) ---
    band_names = list(EEG_BANDS.keys())
    avg_clean = [np.mean(band_power_ratios_per_snr[band]['clean']) for band in band_names]
    avg_noisy = [np.mean(band_power_ratios_per_snr[band]['noisy']) for band in band_names]
    avg_denoised = [np.mean(band_power_ratios_per_snr[band]['denoised']) for band in band_names]

    x = np.arange(len(band_names))
    width = 0.25
    plt.figure(figsize=(7, 8))  # Wiener: half as wide as AR-WGAN
    plt.bar(x - width, avg_clean, width, label='Clean', color='blue')
    plt.bar(x, avg_noisy, width, label='Noisy', color='red')
    plt.bar(x + width, avg_denoised, width, label='Denoised', color='green')
    plt.title("Wiener Filter", fontsize=24)
    plt.xlabel('EEG Band', fontsize=18)
    plt.ylabel('Average Power Ratio', fontsize=18)
    plt.xticks(x, [b.capitalize() for b in band_names])
    plt.ylim(0, max(avg_clean + avg_noisy + avg_denoised) * 1.05)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_PLOTS_DIR, "overall_band_power_ratios_grouped.png"))
    plt.close()
    print(f"Saved grouped band power ratio bar chart to '{os.path.join(EVAL_PLOTS_DIR, 'overall_band_power_ratios_grouped.png')}'")

    print("\nAll evaluation plots have been generated and saved to the 'wiener_evaluation_plots' folder.")

