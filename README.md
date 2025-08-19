# EEGDenoisingWGAN-GP

This repository provides a deep learning framework for denoising EEG (Electroencephalography) signals using a Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP). The approach is specifically designed to remove physiological artifacts such as EOG (Electrooculogram) and EMG (Electromyogram) noise from EEG recordings, leveraging state-of-the-art generative modeling techniques for effective denoising.

## Features

- **WGAN-GP Architecture**: Utilizes a Wasserstein GAN with gradient penalty for stable and robust adversarial training.
- **Custom Data Generation**: Dynamically creates noisy-clean EEG pairs by adding scaled EOG/EMG noise to clean EEG signals at random Signal-to-Noise Ratios (SNR).
- **Configurable Data Augmentation**: Supports multiple noise variants for each clean epoch to increase dataset diversity and improve generalization.
- **End-to-End PyTorch Pipeline**: Seamless integration from data loading and augmentation to training and evaluation.
- **U-Net Inspired Generator**: The generator model follows an encoder-decoder (U-Net-like) architecture, which is well-suited for 1D time-series denoising.

## How It Works

1. **Data Preparation**: 
    - Loads clean EEG, EOG, and EMG signals from `.npy` files.
    - Normalizes each epoch to the range [-1, 1] for stable neural network training.
    - Dynamically generates noisy EEG signals by mixing clean EEG with EOG/EMG noise at random SNRs.

2. **Dataset & Dataloader**:
    - The custom `EEGNoiseDataset` class produces paired noisy and clean EEG epochs for training.
    - DataLoader shuffles and batches data efficiently for GPU/CPU processing.

3. **Model Architecture**:
    - **Generator**: Encoder-decoder network that attempts to recover clean EEG from noisy input.
    - **Critic (Discriminator)**: Evaluates the realism of the denoised output versus ground truth clean EEG.

4. **Training Loop**:
    - Adversarial training where the generator learns to produce clean signals that "fool" the critic.
    - Uses gradient penalty to stabilize WGAN training for time-series data.

5. **Evaluation**:
    - Provides evaluation scripts to test denoising performance at various SNR levels.
    - Calculates metrics such as Relative Root Mean Squared Error (RRMSE) for quantitative assessment.

## Requirements

- Python 3.x
- PyTorch
- NumPy

(See `requirements.txt` if available for full dependency list.)

## Getting Started

1. **Prepare Dataset**: Place your preprocessed EEG, EOG, and EMG `.npy` files in the `dataset/` directory.
2. **Configure Parameters**: Adjust settings in `variables.py` for your dataset and training preferences.
3. **Train the Model**: Run `train.py` to begin WGAN-GP training.
4. **Evaluate**: Use `evaluate.py` to assess model performance and generate evaluation plots.

## Repository Structure

- `train.py` – Training loop and pipeline
- `model.py` – Generator and Critic architectures
- `eeg_data_generator.py` – Data preparation and augmentation
- `variables.py` – Central configuration
- `utils.py` – Utility functions (e.g., gradient penalty)
- `evaluate.py` – Evaluation scripts

