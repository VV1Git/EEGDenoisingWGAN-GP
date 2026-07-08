"""Shared plotting helpers used by every evaluator, so the figure code is not
copied per method."""
import matplotlib.pyplot as plt  # type: ignore
from scipy.signal import welch  # type: ignore

BAND_COLORS = {
    'delta': 'yellow',
    'theta': 'orange',
    'alpha': 'lightgreen',
    'beta': 'skyblue',
    'gamma': 'plum',
}


def plot_psd_comparison(clean_signal, noisy_signal, denoised_signal, sampling_rate,
                        bands, method_name, save_path=None):
    """Plot the power spectral density of the clean, noisy, and denoised signals
    side by side, with the EEG frequency bands shaded. ``method_name`` is used as
    the figure title (for example "AR-WGAN", "ICA", or "Wiener Filter")."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(method_name, fontsize=24)
    titles = ["Clean", "Noisy", "Denoised"]
    signals = [clean_signal.flatten(), noisy_signal.flatten(), denoised_signal.flatten()]

    for i, (ax, signal, subtitle) in enumerate(zip(axes, signals, titles)):
        f, Pxx = welch(signal, fs=sampling_rate, nperseg=sampling_rate, return_onesided=True)
        ax.plot(f, Pxx, color='blue')
        ax.set_title(subtitle)
        ax.set_xlabel('Frequency (Hz)', fontsize=18)
        if i == 0:
            ax.set_ylabel('Power (V**2/Hz)', fontsize=18)
        for band_name, (low_freq, high_freq) in bands.items():
            ax.axvspan(low_freq, high_freq, color=BAND_COLORS[band_name], alpha=0.3,
                       label=band_name.capitalize())
        ax.set_xlim(0, 80)
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            sorted_labels = [b.capitalize() for b in bands.keys()]
            order = [labels.index(l) for l in sorted_labels if l in labels]
            ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
                      loc='upper right', fontsize=15)

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    if save_path:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
