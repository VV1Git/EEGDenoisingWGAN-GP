import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import welch
from os.path import join
import os

# --- Variables from variables.py ---
# This section contains the relevant variables copied from your project's variables.py.
# This makes the script self-contained and runnable on its own.
SNR_RANGE_DB_EVAL = np.arange(-14, 6, 2)
EVAL_PLOTS_DIR = 'evaluation_plots'
EEG_BANDS = {
    'delta': [0.5, 4],
    'theta': [4, 8],
    'alpha': [8, 13],
    'beta': [13, 30],
    'gamma': [30, 100]
}
SAMPLING_RATE = 512


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
    # We use the 'scipy.signal.wiener' function, which provides a robust implementation.
    # It can work adaptively without an explicit noise power spectrum.
    # The 'mysize' parameter is used to specify the neighborhood for noise estimation.
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

    Returns:
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

# --- Plotting Utility Functions ---

def create_and_save_plot(x, y, xlabel, ylabel, title, filename):
    """
    Creates a line plot and saves it to the specified directory.
    
    Args:
        x (list or np.ndarray): The x-axis data.
        y (list or np.ndarray): The y-axis data.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        title (str): The title of the plot.
        filename (str): The name of the file to save (e.g., 'CC_VS_SNR.png').
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    
    # Ensure the directory exists
    os.makedirs(EVAL_PLOTS_DIR, exist_ok=True)
    save_path = join(EVAL_PLOTS_DIR, filename)
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.close() # Close the plot figure to free up memory

def create_and_save_psd_plot(signals_dict, fs, title, filename):
    """
    Creates a PSD plot comparing multiple signals and saves it.
    
    Args:
        signals_dict (dict): A dictionary mapping signal names to their data arrays.
        fs (float): The sampling frequency.
        title (str): The title of the plot.
        filename (str): The name of the file to save (e.g., 'psd_comparison.png').
    """
    plt.figure(figsize=(10, 6))
    for name, signal_data in signals_dict.items():
        f, Pxx = calculate_psd(signal_data, fs)
        plt.semilogy(f, Pxx, label=name)
    
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Ensure the directory exists
    os.makedirs(EVAL_PLOTS_DIR, exist_ok=True)
    save_path = join(EVAL_PLOTS_DIR, filename)
    plt.savefig(save_path)
    print(f"PSD plot saved to {save_path}")
    plt.close()

def create_and_save_band_power_plots(band_power_data, bands, title, filename_prefix):
    """
    Creates bar plots for band power ratios and saves them.
    
    Args:
        band_power_data (dict): A dictionary where keys are band names and
                                values are a list of tuples (noisy_ratio, denoised_ratio).
        bands (list): A list of band names to plot.
        title (str): The overall title for the plot.
        filename_prefix (str): The prefix for the saved files.
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
        
        ax.set_title(f'Overall {band.capitalize()} Power Ratio vs. SNR')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(f'{band.capitalize()} Power Ratio')
        ax.set_xticks(x)
        ax.set_xticklabels(snrs)
        ax.legend()
        plt.grid(True, axis='y')
        plt.tight_layout()
        
        save_path = join(EVAL_PLOTS_DIR, f'overall_{band}_power_ratio.png')
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.close(fig)

# --- Main Logic for Generating Plots ---
if __name__ == "__main__":
    print(f"Generating evaluation plots and saving to '{EVAL_PLOTS_DIR}'...")
    
    # A single, representative clean signal for the simulation
    # In a real scenario, you'd load a dataset of clean epochs.
    # Here, we generate a synthetic EEG-like signal for demonstration.
    time_len_samples = SAMPLING_RATE * 5 # 5 seconds
    t = np.linspace(0, 5, time_len_samples, endpoint=False)
    clean_signal = np.sum([np.sin(2 * np.pi * f * t) for f in [4, 8, 12, 20, 40]], axis=0)
    
    # Lists to store metrics for plotting
    cc_scores = []
    rrmse_temporal_scores = []
    band_power_data = {band: [] for band in EEG_BANDS.keys()}
    
    # --- The Main Evaluation Loop ---
    for snr_db in SNR_RANGE_DB_EVAL:
        # Convert SNR from dB to linear scale
        snr_linear = 10**(snr_db / 10)
        
        # Calculate required noise power
        clean_power = np.mean(clean_signal**2)
        noise_power = clean_power / snr_linear
        
        # Generate noise with the calculated power
        noise = np.random.randn(len(clean_signal)) * np.sqrt(noise_power)
        noisy_signal = clean_signal + noise

        # Apply the Wiener filter
        denoised_signal = wiener_filter_scipy(noisy_signal, mysize=None)
        
        # Calculate metrics for the current SNR level
        cc = calculate_cc(denoised_signal, clean_signal)
        rrmse_temporal = calculate_rrmse(denoised_signal, clean_signal)
        
        cc_scores.append(cc)
        rrmse_temporal_scores.append(rrmse_temporal)

        # Calculate and store band power ratios
        psd_freqs, psd_noisy = calculate_psd(noisy_signal, SAMPLING_RATE)
        _, psd_denoised = calculate_psd(denoised_signal, SAMPLING_RATE)
        _, psd_clean = calculate_psd(clean_signal, SAMPLING_RATE)

        for band in EEG_BANDS.keys():
            noisy_power = calculate_band_power(psd_freqs, psd_noisy, band)
            denoised_power = calculate_band_power(psd_freqs, psd_denoised, band)
            clean_power_band = calculate_band_power(psd_freqs, psd_clean, band)
            
            # Avoid division by zero
            noisy_ratio = noisy_power / clean_power_band if clean_power_band > 0 else 0
            denoised_ratio = denoised_power / clean_power_band if clean_power_band > 0 else 0
            
            band_power_data[band].append((noisy_ratio, denoised_ratio))
        
        print(f"SNR: {snr_db} dB | CC: {cc:.4f} | RRMSE: {rrmse_temporal:.4f}")

    # --- Create and Save All Plots ---
    
    # CC vs. SNR plot
    create_and_save_plot(
        SNR_RANGE_DB_EVAL, 
        cc_scores, 
        'SNR (dB)', 
        'Correlation Coefficient', 
        'Correlation Coefficient vs. SNR',
        'CC_VS_SNR.png'
    )
    
    # RRMSE vs. SNR plot
    create_and_save_plot(
        SNR_RANGE_DB_EVAL, 
        rrmse_temporal_scores, 
        'SNR (dB)', 
        'Relative Root Mean Squared Error (RRMSE)', 
        'RRMSE Temporal vs. SNR',
        'RRMSE_Temporal_vs_SNR.png'
    )
    
    # Create individual band power ratio plots
    create_and_save_band_power_plots(
        band_power_data, 
        list(EEG_BANDS.keys()), 
        'Overall Band Power Ratio vs. SNR',
        'overall'
    )
    
    # Create a PSD comparison plot for a specific SNR level (e.g., 0 dB)
    psd_signals = {
        'Original Clean Signal': clean_signal,
        'Noisy Signal': noisy_signal,
        'Denoised Signal (Wiener)': denoised_signal,
    }
    create_and_save_psd_plot(
        psd_signals, 
        SAMPLING_RATE, 
        f'Power Spectral Density Comparison (SNR = {SNR_RANGE_DB_EVAL[-1]} dB)', # Use the highest SNR for a clear plot
        'PSD_comparison_example.png'
    )

    print("\nAll evaluation plots have been generated and saved to the 'evaluation_plots' folder.")
