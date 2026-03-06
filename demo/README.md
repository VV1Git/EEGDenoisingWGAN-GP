# EEG Denoising Demo - Interactive Slideshow

This is a **fully self-contained** interactive demo showcasing three different EEG denoising methods:
1. **AR-WGAN** - Deep Learning based denoising using Wasserstein GAN
2. **ICA** - Independent Component Analysis
3. **Wiener Filter** - Classical signal processing method

## Requirements

This demo requires Python 3.7+ and the following packages:
- numpy>=1.21.0
- torch>=1.10.0
- scipy>=1.7.0
- matplotlib>=3.4.0
- scikit-learn>=1.0.0

## Installation

Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Demo

Simply run:
```bash
python main.py
```

The demo will:
1. Load the pre-trained AR-WGAN model
2. Load the EEG dataset
3. Prepare 10 demonstration slides
4. Launch an interactive slideshow

## Controls

- **SPACE** or **RIGHT ARROW** - Next slide
- **LEFT ARROW** - Previous slide
- **Q** or **ESC** - Quit the demo
- Close the window to exit

## What You'll See

Each slide displays a comparison of all three denoising methods with:

### Left Column - Time Domain Plots
Shows the EEG signal over time (in seconds) for each method:
- **Blue line**: Clean EEG (ground truth)
- **Red dashed line**: Noisy EEG (with SNR = -6dB)
- **Colored line**: Denoised result (Green=AR-WGAN, Orange=ICA, Purple=Wiener)

### Right Column - Frequency Domain (PSD) Plots
Shows the Power Spectral Density analysis for each method:
- Same color scheme as time domain
- Highlighted EEG frequency bands:
  - **Delta** (0.5-4 Hz) - Deep sleep
  - **Theta** (4-8 Hz) - Drowsiness, meditation
  - **Alpha** (8-13 Hz) - Relaxed, eyes closed
  - **Beta** (13-30 Hz) - Active thinking, focus
  - **Gamma** (30-100 Hz) - High-level processing

## Files Included

- `main.py` - Interactive slideshow application
- `model.py` - AR-WGAN Generator architecture
- `ica_denoise.py` - ICA denoising implementation
- `wiener_denoise.py` - Wiener filter implementation
- `final_generator_model.pth.tar` - Pre-trained AR-WGAN model (~260 MB)
- `EEG_all_epochs.npy` - Clean EEG dataset
- `EOG_all_epochs.npy` - EOG (eye movement) artifact data
- `EMG_all_epochs.npy` - EMG (muscle) artifact data
- `requirements.txt` - Python dependencies
- `README.md` - This file

## Portability

✅ **This demo folder is completely self-contained!**

You can:
1. Copy the entire `demo` folder to any computer
2. Install Python 3.7+ and the requirements
3. Run `python main.py`
4. No additional setup or configuration needed!

All necessary code, model weights, and data are included in this folder.

## Technical Details

- **Sampling Rate**: 512 Hz
- **Noise Type**: Combination of EOG and EMG artifacts
- **Signal-to-Noise Ratio**: -6 dB (challenging scenario)
- **Number of Slides**: 10 different EEG samples
- **Model**: U-Net style Generator with Wasserstein GAN training

## Troubleshooting

**Issue**: "No module named 'tkinter'"
- **Solution**: Install tkinter for your system:
  - Ubuntu/Debian: `sudo apt-get install python3-tk`
  - Fedora: `sudo dnf install python3-tkinter`
  - macOS: Included with Python
  - Windows: Included with Python

**Issue**: Demo runs but no window appears
- **Solution**: Make sure you have a graphical display. The demo requires an interactive display to show plots.

**Issue**: CUDA out of memory
- **Solution**: The demo will automatically use CPU if CUDA is not available. Performance may be slightly slower but will work fine.

## License

This demo is part of the EEG Denoising research project.
