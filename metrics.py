"""Shared evaluation metrics for EEG denoising.

These functions are the single source of truth for all quantitative metrics so
that AR-WGAN (evaluate.py), ICA (comparisons/ica.py) and the Wiener filter
(comparisons/wiener_filter.py) are all scored with *identical* code. Any
difference in metric definitions between methods would make the head-to-head
comparison (finalplots overlays) unfair, so every method imports from here.

Convention: the clean (ground-truth) signal is always the FIRST argument.
"""
import numpy as np
from scipy.signal import welch
from scipy.integrate import simpson
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


def calculate_rrmse(clean_signal, denoised_signal):
    """Relative Root Mean Squared Error in the temporal domain (lower is better)."""
    clean_signal = clean_signal.flatten()
    denoised_signal = denoised_signal.flatten()
    rmse = np.sqrt(np.mean((clean_signal - denoised_signal) ** 2))
    rms_clean = np.sqrt(np.mean(clean_signal ** 2))
    if rms_clean == 0:  # Avoid division by zero for flat clean signals
        return np.inf if rmse > 0 else 0.0
    return rmse / rms_clean


def calculate_rrmse_spectral(clean_signal, denoised_signal, sampling_rate):
    """Relative RMSE between the Welch PSDs of the clean and denoised signals."""
    if clean_signal.ndim > 1:
        clean_signal = clean_signal.flatten()
    if denoised_signal.ndim > 1:
        denoised_signal = denoised_signal.flatten()

    f_clean, Pxx_clean = welch(clean_signal, fs=sampling_rate, nperseg=sampling_rate, return_onesided=True)
    f_denoised, Pxx_denoised = welch(denoised_signal, fs=sampling_rate, nperseg=sampling_rate, return_onesided=True)

    if not np.array_equal(f_clean, f_denoised):
        raise ValueError("Frequency bins for clean and denoised PSDs do not match.")

    rmse_psd = np.sqrt(np.mean((Pxx_clean - Pxx_denoised) ** 2))
    rms_clean_psd = np.sqrt(np.mean(Pxx_clean ** 2))
    if rms_clean_psd == 0:
        return np.inf if rmse_psd > 0 else 0.0
    return rmse_psd / rms_clean_psd


def calculate_cc(clean_signal, denoised_signal):
    """Pearson's correlation coefficient between clean and denoised signals."""
    clean_signal = clean_signal.flatten()
    denoised_signal = denoised_signal.flatten()
    # Handle flat signals (zero std) to avoid NaN from pearsonr
    if np.std(clean_signal) == 0 or np.std(denoised_signal) == 0:
        return 1.0 if np.allclose(clean_signal, denoised_signal) else 0.0
    return pearsonr(clean_signal, denoised_signal)[0]


def calculate_band_power_ratios(signal, sampling_rate, bands):
    """Ratio of power in each EEG band to total power.

    Accepts a 1D signal (samples,) or a 2D batch (batch, samples); for a batch
    the per-band ratio is averaged across the batch.
    """
    f, Pxx = welch(signal, fs=sampling_rate, nperseg=sampling_rate, return_onesided=True, axis=-1)
    total_power = simpson(Pxx, x=f, axis=-1)

    band_ratios = {}
    for band_name, (low_freq, high_freq) in bands.items():
        freq_mask = (f >= low_freq) & (f <= high_freq)
        band_power = simpson(Pxx[..., freq_mask], x=f[freq_mask], axis=-1)
        ratio = np.where(total_power == 0, 0, band_power / total_power)
        band_ratios[f'{band_name}_ratio'] = np.mean(ratio) if ratio.ndim > 0 else ratio
    return band_ratios


def calculate_cosine_similarity_power_ratios(clean_signal_np, denoised_signal_np, sampling_rate, bands):
    """Cosine similarity between the clean and denoised band-power-ratio vectors."""
    clean_ratios = calculate_band_power_ratios(clean_signal_np, sampling_rate, bands)
    denoised_ratios = calculate_band_power_ratios(denoised_signal_np, sampling_rate, bands)

    clean_ratio_vector = np.array([clean_ratios[f'{band}_ratio'] for band in bands.keys()]).reshape(1, -1)
    denoised_ratio_vector = np.array([denoised_ratios[f'{band}_ratio'] for band in bands.keys()]).reshape(1, -1)
    return cosine_similarity(clean_ratio_vector, denoised_ratio_vector)[0, 0]
