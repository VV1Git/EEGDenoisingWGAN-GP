"""Classical baseline denoisers shared by the comparison scripts and the poster
generator, so there is one implementation of each (no per-file copies)."""
import numpy as np
import warnings
from scipy.signal import wiener as scipy_wiener
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def ica_denoise(noisy_signal, n_components=3):
    """Denoise a single-channel EEG signal with ICA.

    A pseudo-multichannel input is built by stacking time-delayed copies of the
    signal. ICA separates the sources, the component with the highest absolute
    kurtosis (the most artifact-like one) is removed, and the signal is
    reconstructed.

    Args:
        noisy_signal: array of shape (samples,) or (1, samples).
        n_components: number of ICA components / delay taps.

    Returns:
        The denoised signal, shape (samples,).
    """
    x = noisy_signal.flatten()
    X = np.stack([np.roll(x, shift) for shift in range(n_components)], axis=1)

    ica = FastICA(n_components=n_components, random_state=0, max_iter=1000, tol=1e-4)
    sources = ica.fit_transform(X)
    kurtosis = np.abs(np.apply_along_axis(
        lambda s: np.mean((s - np.mean(s)) ** 4) / (np.var(s) ** 2), 0, sources))
    sources[:, np.argmax(kurtosis)] = 0
    denoised = ica.inverse_transform(sources)[:, 0]

    # The delay-embedded single-channel unmixing is occasionally ill-conditioned,
    # and on a small fraction of epochs the reconstruction blows up (non-finite or
    # enormous amplitude). Treat that as a failed separation and return the input
    # unchanged, so one bad epoch does not dominate the averaged error metrics.
    # This is a no-op on well-behaved epochs (a valid denoising never exceeds the
    # input amplitude by 10x).
    if (not np.all(np.isfinite(denoised))) or np.max(np.abs(denoised)) > 10 * (np.max(np.abs(x)) + 1e-12):
        return x
    return denoised


def wiener_denoise(noisy_signal, mysize=31, noise=None):
    """Denoise a 1D signal with scipy's Wiener filter.

    Args:
        noisy_signal: the 1D signal to filter.
        mysize: neighborhood size for local noise estimation (31 samples is about
            0.06 s at the 512 Hz sampling rate).
        noise: noise power to use; estimated from local variance when None.

    Returns:
        The denoised signal.
    """
    noisy_signal = np.asarray(noisy_signal, dtype=np.float64).flatten()
    return scipy_wiener(noisy_signal, mysize=mysize, noise=noise)
