# Centralized configuration and shared variables for EEG Denoising GAN project

import numpy as np

# --- File Paths ---
EEG_FILE = 'dataset/EEG_all_epochs.npy'
EOG_FILE = 'dataset/EOG_all_epochs.npy'
EMG_FILE = 'dataset/EMG_all_epochs.npy'

# --- SNR Range ---
SNR_RANGE_DB = [-5, 5]  # For training
SNR_RANGE_DB_EVAL = np.arange(-14, 6, 2)  # For evaluation (e.g., -14, -12, ..., 4 dB)

# --- Data Split Ratio ---
TRAIN_SPLIT_RATIO = 0.9  # 90% for training, 10% for testing

# --- Data Augmentation ---
NUM_NOISE_VARIANTS = 8  # Number of noise variants per clean epoch

# --- Model Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
CHANNELS_EEG = 1
NUM_EPOCHS = 150
FEATURES_CRITIC = 16
FEATURES_GEN = 16
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

# --- Logging/Output Directories ---
LOGS_DIR = 'logs'
EVAL_PLOTS_DIR = 'evaluation_plots'
SAVED_MODEL_PATH = 'model/final_generator_model.pth.tar'

# --- EEG Frequency Bands and Sampling Rate ---
EEG_BANDS = {
    'delta': [0.5, 4],
    'theta': [4, 8],
    'alpha': [8, 13],
    'beta': [13, 30],
    'gamma': [30, 100]
}
SAMPLING_RATE = 512

# --- Visualization ---
PSD_SAMPLE_INDEX_FOR_VIZ = 0
