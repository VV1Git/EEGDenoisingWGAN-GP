import numpy as np
from sklearn.decomposition import FastICA
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

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
    x = noisy_signal.flatten()
    
    # Create pseudo-multichannel input using time-delayed versions
    X = np.stack([np.roll(x, shift=shift) for shift in range(n_components)], axis=1)
    
    # Apply ICA
    ica = FastICA(n_components=n_components, random_state=0, max_iter=10000, tol=0.1)
    S_ = ica.fit_transform(X)
    
    # Remove the component with the highest absolute kurtosis (likely artifact)
    kurt = np.abs(np.apply_along_axis(lambda s: np.mean((s - np.mean(s))**4) / (np.var(s)**2), 0, S_))
    artifact_idx = np.argmax(kurt)
    S_[:, artifact_idx] = 0
    
    # Reconstruct signal
    X_denoised = ica.inverse_transform(S_)
    
    # Return the first channel (original signal)
    return X_denoised[:, 0]
