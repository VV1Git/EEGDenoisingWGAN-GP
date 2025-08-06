import numpy as np
import os
from sklearn.model_selection import train_test_split # type:ignore

# Import data preparation utilities from your custom file
# Ensure 'eeg_data_generator.py' is in the same directory or accessible in PYTHONPATH
from eeg_data_generator import prepare_eeg_data, EEGNoiseDataset

def main():
    """
    This script demonstrates the data splitting and calculates the number of
    augmented data fragment pairs for training and testing datasets.
    """
    # --- Configuration (MUST match training.py and evaluate_model.py) ---
    EEG_FILE = 'dataset/EEG_all_epochs.npy'
    EOG_FILE = 'dataset/EOG_all_epochs.npy'
    EMG_FILE = 'dataset/EMG_all_epochs.npy'
    SNR_RANGE_DB = [-5, 5] # This range is used by EEGNoiseDataset, but not for counting here.

    TRAIN_SPLIT_RATIO = 0.9 # 90% for training, 10% for testing
    NUM_NOISE_VARIANTS = 4 # Number of noise variants per clean epoch for augmentation

    print("--- Data Split and Augmentation Calculation ---")

    # 1. Load raw data
    try:
        clean_eeg_all, eog_noise_np, emg_noise_np = prepare_eeg_data(
            EEG_FILE, EOG_FILE, EMG_FILE, SNR_RANGE_DB # SNR_RANGE_DB is a placeholder for prepare_eeg_data
        )
        print(f"\nTotal raw clean EEG epochs loaded: {clean_eeg_all.shape[0]}")
        print(f"Total raw EOG noise epochs loaded: {eog_noise_np.shape[0]}")
        print(f"Total raw EMG noise epochs loaded: {emg_noise_np.shape[0]}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error preparing data: {e}")
        print("Please ensure your dataset files are correctly placed and named.")
        return

    # 2. Perform the 90/10 split on the *clean EEG epochs*
    # This ensures that the training and testing sets have completely unseen clean data.
    train_clean_eeg_np, test_clean_eeg_np = train_test_split(
        clean_eeg_all, test_size=(1 - TRAIN_SPLIT_RATIO), random_state=42
    )

    print(f"\nAfter 90/10 split on clean EEG epochs:")
    print(f"  Training clean EEG epochs: {train_clean_eeg_np.shape[0]}")
    print(f"  Testing clean EEG epochs: {test_clean_eeg_np.shape[0]}")

    # 3. Instantiate EEGNoiseDataset for training and testing
    # The 'num_noise_variants_per_clean_epoch' parameter determines the effective size.
    train_dataset = EEGNoiseDataset(
        train_clean_eeg_np, eog_noise_np, emg_noise_np, SNR_RANGE_DB,
        num_noise_variants_per_clean_epoch=NUM_NOISE_VARIANTS
    )

    test_dataset = EEGNoiseDataset(
        test_clean_eeg_np, eog_noise_np, emg_noise_np, SNR_RANGE_DB,
        num_noise_variants_per_clean_epoch=NUM_NOISE_VARIANTS
    )

    # 4. Report the total number of augmented data fragment pairs
    print(f"\nAugmented data fragment pairs per epoch (with {NUM_NOISE_VARIANTS} variants per clean epoch):")
    print(f"  Training augmented samples: {len(train_dataset)}")
    print(f"  Testing augmented samples: {len(test_dataset)}")

    # Verify the calculation
    expected_train_samples = train_clean_eeg_np.shape[0] * NUM_NOISE_VARIANTS
    expected_test_samples = test_clean_eeg_np.shape[0] * NUM_NOISE_VARIANTS
    print(f"\nVerification:")
    print(f"  Expected training samples: {expected_train_samples}")
    print(f"  Expected testing samples: {expected_test_samples}")
    print(f"  Total expected samples: {expected_train_samples + expected_test_samples}")


if __name__ == "__main__":
    main()
