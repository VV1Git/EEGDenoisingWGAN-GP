"""
EEG Denoising Demo - Interactive Slideshow
===========================================
This demo shows the performance of three different denoising methods on noisy EEG data:
1. AR-WGAN (Deep Learning Model)
2. ICA (Independent Component Analysis)
3. Wiener Filter

Auto-advances every 15 seconds
Press Q or ESC to quit
"""

import numpy as np
import torch
import matplotlib
# Try to use a backend that doesn't require tkinter
import os
import sys
try:
    matplotlib.use('TkAgg')  # Try TkAgg backend first (requires python3-tk)
except:
    try:
        matplotlib.use('Qt5Agg')  # Try Qt5 backend
    except:
        try:
            matplotlib.use('GTK3Agg')  # Try GTK3 backend
        except:
            matplotlib.use('Agg')  # Fallback to non-interactive backend
            print("Warning: Using non-interactive backend. Keyboard controls may not work.")
            print("To enable interactive features, install tkinter: sudo apt-get install python3-tk")
            print("Or install PyQt5: pip install PyQt5")
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent
from scipy.signal import welch
from scipy.stats import pearsonr

# Import the denoising methods
from model import Generator
from ica_denoise import ica_denoise
from wiener_denoise import wiener_filter

# Configuration
SAMPLING_RATE = 512
EEG_BANDS = {
    'delta': [0.5, 4],
    'theta': [4, 8],
    'alpha': [8, 13],
    'beta': [13, 30],
    'gamma': [30, 100]
}
SNR_DB = -6  # Signal-to-noise ratio in dB for synthetic noise
AUTO_ADVANCE_INTERVAL = 5  # seconds


class EEGDemoSlideshow:
    """Interactive slideshow for EEG denoising demonstration."""
    
    def __init__(self, script_dir):
        self.script_dir = script_dir
        self.current_slide = 0
        self.slides_cache = {}  # Cache for generated slides
        self.fig = None
        self.timer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("=" * 70)
        print("EEG DENOISING INTERACTIVE DEMO")
        print("=" * 70)
        print(f"Using device: {self.device}")
        
        # Load model
        self.gen_model = self.load_model()
        
        # Load dataset
        self.load_dataset()
        
        print("\n" + "=" * 70)
        print("CONTROLS:")
        print("  SPACE / RIGHT ARROW : Next slide")
        print("  LEFT ARROW          : Previous slide")
        print("  Q / ESC             : Quit")
        print(f"  Auto-advance        : Every {AUTO_ADVANCE_INTERVAL} seconds")
        print("=" * 70)
        print(f"\nDataset size: {len(self.clean_eeg)} epochs")
        print("Slides generated on-demand...\n")
    
    def load_model(self):
        """Load the trained generator model."""
        model_path = os.path.join(self.script_dir, 'final_generator_model.pth.tar')
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            sys.exit(1)
        
        print("Loading AR-WGAN model...")
        gen = Generator(channels_eeg=1, seq_len=512, features_g=32)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'gen' in checkpoint:
            gen.load_state_dict(checkpoint['gen'])
        else:
            gen.load_state_dict(checkpoint)
        
        gen.to(self.device)
        gen.eval()
        print("✓ Model loaded successfully!")
        return gen
    
    def load_dataset(self):
        """Load the dataset."""
        eeg_path = os.path.join(self.script_dir, 'EEG_all_epochs.npy')
        eog_path = os.path.join(self.script_dir, 'EOG_all_epochs.npy')
        emg_path = os.path.join(self.script_dir, 'EMG_all_epochs.npy')
        
        if not all(os.path.exists(p) for p in [eeg_path, eog_path, emg_path]):
            print("Error: Dataset files not found!")
            print(f"Expected files in {self.script_dir}:")
            print("  - EEG_all_epochs.npy")
            print("  - EOG_all_epochs.npy")
            print("  - EMG_all_epochs.npy")
            sys.exit(1)
        
        print("Loading dataset...")
        self.clean_eeg = np.load(eeg_path, allow_pickle=True)
        self.eog_noise = np.load(eog_path, allow_pickle=True)
        self.emg_noise = np.load(emg_path, allow_pickle=True)
        print(f"✓ Loaded {len(self.clean_eeg)} EEG epochs")
    
    def generate_slide(self, slide_idx):
        """Generate a slide on-demand."""
        if slide_idx in self.slides_cache:
            return self.slides_cache[slide_idx]
        
        # Wrap around if we exceed dataset size
        i = slide_idx % len(self.clean_eeg)
        
        # Get clean signal
        clean_signal = self.clean_eeg[i].flatten()
        clean_signal = self.normalize_signal(clean_signal)
        
        # Create noise (combination of EOG and EMG)
        noise_signal = (self.eog_noise[i % len(self.eog_noise)] + self.emg_noise[i % len(self.emg_noise)]).flatten()
        noise_signal = self.normalize_signal(noise_signal)
        
        # Ensure same length
        min_len = min(len(clean_signal), len(noise_signal))
        clean_signal = clean_signal[:min_len]
        noise_signal = noise_signal[:min_len]
        
        # Create noisy signal
        noisy_signal = self.create_noisy_signal(clean_signal, noise_signal, SNR_DB)
        
        # Remove DC offset
        clean_signal = clean_signal - np.mean(clean_signal)
        noisy_signal = noisy_signal - np.mean(noisy_signal)
        
        # Denoise with all methods
        denoised_gan = self.denoise_with_gan(noisy_signal)
        denoised_ica = ica_denoise(noisy_signal, n_components=3)
        denoised_wiener = wiener_filter(noisy_signal, mysize=31)
        
        # Store slide data
        slide_data = {
            'clean': clean_signal,
            'noisy': noisy_signal,
            'denoised_gan': denoised_gan,
            'denoised_ica': denoised_ica,
            'denoised_wiener': denoised_wiener
        }
        
        # Cache the slide
        self.slides_cache[slide_idx] = slide_data
        
        return slide_data
    
    def normalize_signal(self, signal):
        """Normalize signal to [-1, 1] range."""
        min_val = signal.min()
        max_val = signal.max()
        if max_val == min_val:
            return np.zeros_like(signal)
        return 2 * (signal - min_val) / (max_val - min_val) - 1
    
    def create_noisy_signal(self, clean_signal, noise_signal, snr_db):
        """Add noise to clean signal at specified SNR."""
        clean_power = np.mean(clean_signal**2)
        noise_power = np.mean(noise_signal**2)
        
        if clean_power == 0 or noise_power == 0:
            return clean_signal
        
        snr_linear = 10**(snr_db / 10)
        alpha = np.sqrt(clean_power / (snr_linear * noise_power))
        
        noisy_signal = clean_signal + alpha * noise_signal
        return noisy_signal
    
    def denoise_with_gan(self, noisy_signal):
        """Denoise signal using the AR-WGAN model."""
        noisy_tensor = torch.from_numpy(noisy_signal).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            denoised_tensor = self.gen_model(noisy_tensor)
        
        denoised = denoised_tensor.cpu().numpy().squeeze()
        return denoised
    
    def calculate_psd(self, signal):
        """Calculate Power Spectral Density."""
        f, Pxx = welch(signal, fs=SAMPLING_RATE, nperseg=SAMPLING_RATE, return_onesided=True)
        return f, Pxx
    
    def calculate_metrics(self, clean, denoised):
        """Calculate performance metrics."""
        # Correlation coefficient
        corr, _ = pearsonr(clean, denoised)
        
        # Temporal RRMSE (Relative Root Mean Square Error)
        mse = np.mean((clean - denoised) ** 2)
        rmse = np.sqrt(mse)
        rrmse_temporal = rmse / (np.std(clean) + 1e-10)  # as decimal
        
        # Spectral RRMSE
        f_clean, psd_clean = self.calculate_psd(clean)
        f_denoised, psd_denoised = self.calculate_psd(denoised)
        
        mse_spectral = np.mean((psd_clean - psd_denoised) ** 2)
        rmse_spectral = np.sqrt(mse_spectral)
        rrmse_spectral = rmse_spectral / (np.std(psd_clean) + 1e-10)  # as decimal
        
        return {
            'correlation': corr,
            'rrmse_temporal': rrmse_temporal,
            'rrmse_spectral': rrmse_spectral
        }
    
    def plot_slide(self, slide_idx):
        """Plot a single slide with all comparisons."""
        # Generate slide data on-demand
        data = self.generate_slide(slide_idx)
        
        # Create figure only once
        if self.fig is None:
            self.fig = plt.figure(figsize=(18, 10))  # Aspect ratio ~3:2, fits better on screen
            self.fig.canvas.manager.set_window_title('EEG Denoising Demo')
        else:
            self.fig.clear()
        
        # Adjust grid to make time series twice as wide as PSD (2:1 ratio)
        # Increase top margin and hspace to prevent overlap
        gs = self.fig.add_gridspec(3, 3, hspace=0.5, wspace=0.3, top=0.90, bottom=0.06, left=0.05, right=0.98)
        
        methods = [
            ('AR-WGAN (Deep Learning)', data['denoised_gan'], 'green'),
            ('ICA (Independent Component Analysis)', data['denoised_ica'], 'orange'),
            ('Wiener Filter (Classical)', data['denoised_wiener'], 'purple')
        ]
        
        band_colors = {
            'delta': 'yellow',
            'theta': 'orange',
            'alpha': 'lightgreen',
            'beta': 'skyblue',
            'gamma': 'plum'
        }
        
        # Main title - no sample number
        self.fig.suptitle(
            'EEG Denoising Comparison',
            fontsize=15, fontweight='bold', y=0.96
        )
        
        for idx, (method_name, denoised, color) in enumerate(methods):
            # Calculate metrics
            metrics = self.calculate_metrics(data['clean'], denoised)
            
            # Time domain plot (spans 2 columns)
            ax_time = self.fig.add_subplot(gs[idx, 0:2])
            time_axis = np.arange(len(data['clean'])) / SAMPLING_RATE
            
            ax_time.plot(time_axis, data['clean'], label='Clean EEG', color='blue', alpha=0.7, linewidth=1.2)
            ax_time.plot(time_axis, data['noisy'], label='Noisy (SNR=-6dB)', color='red', linestyle='--', alpha=0.6, linewidth=0.9)
            ax_time.plot(time_axis, denoised, label=f'Denoised', color=color, linewidth=1.5, alpha=0.9)
            
            ax_time.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
            ax_time.set_ylabel('Amplitude', fontsize=10, fontweight='bold')
            
            # Add metrics to title - changed to decimal format
            title_text = (f'{method_name} - Time Domain\n'
                         f'Corr: {metrics["correlation"]:.3f} | '
                         f'RRMSE(t): {metrics["rrmse_temporal"]:.4f} | '
                         f'RRMSE(f): {metrics["rrmse_spectral"]:.4f}')
            ax_time.set_title(title_text, fontsize=9, fontweight='bold', pad=8)
            ax_time.legend(loc='upper right', fontsize=8)
            ax_time.grid(True, alpha=0.3, linestyle=':')
            ax_time.tick_params(labelsize=8)
            
            # Frequency domain plot (PSD) - spans 1 column
            ax_freq = self.fig.add_subplot(gs[idx, 2])
            
            f_clean, Pxx_clean = self.calculate_psd(data['clean'])
            f_noisy, Pxx_noisy = self.calculate_psd(data['noisy'])
            f_denoised, Pxx_denoised = self.calculate_psd(denoised)
            
            ax_freq.plot(f_clean, Pxx_clean, label='Clean', color='blue', alpha=0.7, linewidth=1.2)
            ax_freq.plot(f_noisy, Pxx_noisy, label='Noisy', color='red', linestyle='--', alpha=0.6, linewidth=0.9)
            ax_freq.plot(f_denoised, Pxx_denoised, label=f'Denoised', color=color, linewidth=1.5, alpha=0.9)
            
            # Shade EEG bands
            for band_name, (low_freq, high_freq) in EEG_BANDS.items():
                ax_freq.axvspan(low_freq, high_freq, color=band_colors[band_name], alpha=0.15)
            
            # Add band labels only on top row to avoid overlap
            if idx == 0:
                y_pos = ax_freq.get_ylim()[1] * 0.90
                for band_name, (low_freq, high_freq) in EEG_BANDS.items():
                    mid_freq = (low_freq + high_freq) / 2
                    ax_freq.text(mid_freq, y_pos, band_name[:1].upper(),
                               ha='center', va='top', fontsize=6, fontweight='bold')
            
            ax_freq.set_xlabel('Freq (Hz)', fontsize=9, fontweight='bold')
            ax_freq.set_ylabel('PSD', fontsize=9, fontweight='bold')
            ax_freq.set_title('PSD', fontsize=9, fontweight='bold', pad=8)
            ax_freq.legend(loc='upper right', fontsize=7)
            ax_freq.set_xlim(0, 80)
            ax_freq.grid(True, alpha=0.3, linestyle=':')
            ax_freq.tick_params(labelsize=7)
        
        # Connect keyboard event handler (only once)
        if not hasattr(self, '_key_connected'):
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            self._key_connected = True
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def auto_advance(self):
        """Auto-advance to next slide."""
        self.current_slide += 1
        print(f"[Auto] → Slide {self.current_slide + 1}")
        self.plot_slide(self.current_slide)
        self.reset_timer()
    
    def reset_timer(self):
        """Reset the auto-advance timer."""
        if self.timer is not None:
            self.timer.stop()
        self.timer = self.fig.canvas.new_timer(interval=AUTO_ADVANCE_INTERVAL * 1000)
        self.timer.add_callback(self.auto_advance)
        self.timer.start()
    
    def on_key_press(self, event):
        """Handle keyboard events."""
        if event.key in ['right', ' ']:  # Next slide
            self.current_slide += 1
            print(f"→ Slide {self.current_slide + 1}")
            self.plot_slide(self.current_slide)
            self.reset_timer()
        
        elif event.key == 'left':  # Previous slide
            if self.current_slide > 0:
                self.current_slide -= 1
                print(f"← Slide {self.current_slide + 1}")
                self.plot_slide(self.current_slide)
                self.reset_timer()
            else:
                print("(Already at first slide)")
        
        elif event.key in ['q', 'escape']:  # Quit
            print("\nExiting demo...")
            if self.timer is not None:
                self.timer.stop()
            plt.close('all')
            sys.exit(0)
    
    def run(self):
        """Start the slideshow."""
        self.plot_slide(self.current_slide)
        print(f"\nShowing slide {self.current_slide + 1}")
        print(f"Auto-advancing every {AUTO_ADVANCE_INTERVAL} seconds...")
        print("Use arrow keys or spacebar to navigate, Q or ESC to quit.")
        
        # Start auto-advance timer
        self.reset_timer()
        
        plt.show()


def main():
    """Main entry point."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Create and run slideshow
        slideshow = EEGDemoSlideshow(script_dir)
        slideshow.run()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        plt.close('all')
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
