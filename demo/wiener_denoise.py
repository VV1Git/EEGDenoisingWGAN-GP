import numpy as np
import scipy.signal as signal

def wiener_filter(noisy_signal, mysize=31, noise=None):
    """
    Applies a Wiener filter to a 1D signal using scipy's built-in function.
    
    Args:
        noisy_signal (np.ndarray): The 1D signal to be denoised.
        mysize (int): The size of the neighborhood for local noise estimation.
        noise (float, optional): The noise power to use. If None, it is estimated.
    
    Returns:
        np.ndarray: The denoised signal.
    """
    # Ensure input is 1D float64
    noisy_signal = np.asarray(noisy_signal, dtype=np.float64).flatten()
    
    # Apply Wiener filter
    denoised_signal = signal.wiener(noisy_signal, mysize=mysize, noise=noise)
    
    return denoised_signal
