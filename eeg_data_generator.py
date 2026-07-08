import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os

# Import shared variables
from variables import EEG_FILE, EOG_FILE, EMG_FILE, SNR_RANGE_DB, TRAIN_SPLIT_RATIO

# --- 1. Custom Dataset Class ---
class EEGNoiseDataset(Dataset):
    """
    A PyTorch Dataset for generating noisy-clean EEG pairs.

    This dataset takes clean EEG epochs and separate EOG/EMG noise epochs.
    For each request, it randomly selects a clean EEG epoch, a type of noise
    (EOG or EMG), and a random SNR within the specified range. It then
    synthesizes a noisy EEG signal by adding the scaled noise to the clean EEG.

    The __getitem__ method returns a tuple: (noisy_eeg_tensor, clean_eeg_tensor),
    each with a shape of (1, num_samples) to represent a single channel.
    """
    def __init__(self, clean_eeg, eog_noise, emg_noise, snr_range_db, num_noise_variants_per_clean_epoch=1):
        """
        Initializes the dataset with clean EEG and noise components.

        Args:
            clean_eeg (np.ndarray): NumPy array of clean EEG epochs.
                                    Expected shape: (num_epochs, num_samples)
            eog_noise (np.ndarray): NumPy array of EOG noise epochs.
                                    Expected shape: (num_epochs, num_samples)
            emg_noise (np.ndarray): NumPy array of EMG noise epochs.
                                    Expected shape: (num_epochs, num_samples)
            snr_range_db (list): A list [min_snr_db, max_snr_db] for random SNR generation.
            num_noise_variants_per_clean_epoch (int): Number of noise variants per clean epoch (for augmentation).
        """
        # Pre-convert to contiguous float32 once at init. The consumers all
        # ultimately build float32 tensors, so caching a contiguous float32 copy
        # here (a) halves host memory bandwidth on the ~millions of __getitem__
        # calls, (b) avoids a per-sample dtype-cast copy, and (c) produces
        # smaller/contiguous buffers for faster pinned H2D transfers to the GPU.
        # np.asarray(..., dtype=float32) is a no-op if already float32/contiguous.
        self.clean_eeg = np.ascontiguousarray(clean_eeg, dtype=np.float32)
        self.eog_noise = np.ascontiguousarray(eog_noise, dtype=np.float32) if eog_noise is not None else eog_noise
        self.emg_noise = np.ascontiguousarray(emg_noise, dtype=np.float32) if emg_noise is not None else emg_noise
        self.snr_range_db = snr_range_db
        self.num_noise_variants_per_clean_epoch = num_noise_variants_per_clean_epoch

        # Determine the number of samples in each epoch (assuming consistent length for all epochs)
        if self.clean_eeg.shape[0] > 0:
            self.num_samples_per_epoch = self.clean_eeg.shape[1]
        else:
            raise ValueError("Clean EEG data is empty. Cannot determine epoch length.")

        # The number of pairs to generate is based on the number of available clean EEG epochs.
        # You could also choose to generate a fixed larger number by sampling with replacement.
        self.num_pairs = len(clean_eeg) * self.num_noise_variants_per_clean_epoch

        print(f"Dataset initialized with {len(clean_eeg)} clean EEG epochs.")
        print(f"EOG noise epochs available: {len(eog_noise)}")
        print(f"EMG noise epochs available: {len(emg_noise)}")

    def __len__(self):
        """
        Returns the total number of (noisy, clean) pairs that can be generated.
        """
        return self.num_pairs

    def __getitem__(self, idx):
        """
        Generates a (noisy_eeg, clean_eeg) pair for a given index.

        Args:
            idx (int): The index of the clean EEG epoch to use.

        Returns:
            tuple: A tuple containing:
                - noisy_epoch_tensor (torch.Tensor): The synthesized noisy EEG epoch.
                                                     Shape: (1, num_samples)
                - clean_epoch_tensor (torch.Tensor): The corresponding clean EEG epoch.
                                                     Shape: (1, num_samples)
        """
        # Adjust index to support multiple variants per clean epoch
        clean_idx = idx // self.num_noise_variants_per_clean_epoch

        # Select the clean EEG epoch for this index
        clean_epoch = self.clean_eeg[clean_idx]

        # Randomly choose noise type: 0='eog', 1='emg', 2='both'.
        # An integer draw is far cheaper than np.random.choice on a list of
        # strings (no index-array build / object-dtype handling / allocation)
        # while giving the identical uniform-1/3 distribution over the three
        # categories. All epochs are length num_samples_per_epoch by
        # construction, so the former per-branch length-fixups were dead code.
        noise_type = np.random.randint(3)

        # Select noise(s) accordingly
        if noise_type == 0:  # 'eog'
            noise_pool = self.eog_noise
            if len(noise_pool) == 0:
                noise_epoch = np.zeros_like(clean_epoch)
            else:
                noise_epoch = noise_pool[np.random.randint(len(noise_pool))]
        elif noise_type == 1:  # 'emg'
            noise_pool = self.emg_noise
            if len(noise_pool) == 0:
                noise_epoch = np.zeros_like(clean_epoch)
            else:
                noise_epoch = noise_pool[np.random.randint(len(noise_pool))]
        else:  # 'both'
            # Combine EOG and EMG artifacts
            if len(self.eog_noise) == 0 and len(self.emg_noise) == 0:
                noise_epoch = np.zeros_like(clean_epoch)
            else:
                # If one pool is empty, just use the other
                if len(self.eog_noise) == 0:
                    noise_epoch = self.emg_noise[np.random.randint(len(self.emg_noise))]
                elif len(self.emg_noise) == 0:
                    noise_epoch = self.eog_noise[np.random.randint(len(self.eog_noise))]
                else:
                    eog_epoch = self.eog_noise[np.random.randint(len(self.eog_noise))]
                    emg_epoch = self.emg_noise[np.random.randint(len(self.emg_noise))]
                    noise_epoch = eog_epoch + emg_epoch

        # Synthesize the noisy signal by adding scaled noise
        noisy_epoch = self._add_noise_with_snr(clean_epoch, noise_epoch, self.snr_range_db)

        # Convert NumPy arrays to PyTorch tensors.
        # Arrays are already contiguous float32 (cast once at __init__ and the
        # SNR-mix math preserves float32), so no per-sample .float() copy is
        # needed. torch.from_numpy(...).float() would only re-copy identical data.
        # np.ascontiguousarray guards against a rare non-contiguous view (e.g.
        # the length-fixup path) so from_numpy never sees a non-contiguous array.
        noisy_epoch = np.ascontiguousarray(noisy_epoch, dtype=np.float32)
        clean_epoch = np.ascontiguousarray(clean_epoch, dtype=np.float32)
        # .unsqueeze(0) adds a channel dimension (e.g., (512,) -> (1, 512) for single-channel EEG)
        noisy_epoch_tensor = torch.from_numpy(noisy_epoch).unsqueeze(0)
        clean_epoch_tensor = torch.from_numpy(clean_epoch).unsqueeze(0)

        return noisy_epoch_tensor, clean_epoch_tensor

    def _add_noise_with_snr(self, clean_signal, noise_signal, snr_range_db):
        """
        Adds noise to a clean signal to achieve a randomly selected target SNR within a range.

        Args:
            clean_signal (np.ndarray): The clean signal (1D array).
            noise_signal (np.ndarray): The noise signal (1D array).
            snr_range_db (list): [min_snr_db, max_snr_db] for random SNR generation.

        Returns:
            np.ndarray: The noisy signal (clean_signal + scaled_noise_signal).
        """
        # Calculate power (mean squared value) of the clean signal
        clean_power = np.mean(clean_signal**2)
        # Calculate power (mean squared value) of the noise signal
        noise_power = np.mean(noise_signal**2)

        # Handle edge cases where signal or noise power might be zero to prevent division by zero
        if clean_power == 0:
            # If clean signal is flat, just return the noise signal (or zeros if noise is also flat)
            return noise_signal
        if noise_power == 0:
            # If noise signal is flat, return the clean signal as no noise can be added
            return clean_signal

        # Randomly select a target SNR in dB within the specified range
        target_snr_db = np.random.uniform(snr_range_db[0], snr_range_db[1])
        # Convert SNR from dB to linear scale: SNR_linear = 10^(SNR_dB / 10)
        target_snr_linear = 10**(target_snr_db / 10)

        # Calculate the scaling factor (alpha) for the noise signal.
        # We want: clean_power / (alpha^2 * noise_power) = target_snr_linear
        # So: alpha^2 = clean_power / (target_snr_linear * noise_power)
        # alpha = sqrt(clean_power / (target_snr_linear * noise_power))
        alpha = np.sqrt(clean_power / (target_snr_linear * noise_power))

        # Scale the noise signal
        scaled_noise_signal = alpha * noise_signal

        # Add the scaled noise to the clean signal to create the noisy signal
        return clean_signal + scaled_noise_signal

# --- 2. Main Data Loading and Preparation Function ---
def prepare_eeg_data(eeg_file, eog_file, emg_file, snr_range_db):
    """
    Loads raw EEG and artifact data from .npy files and prepares them for the dataset.
    Includes validation and optional normalization.

    Args:
        eeg_file (str): Path to the clean EEG .npy file.
        eog_file (str): Path to the EOG noise .npy file.
        emg_file (str): Path to the EMG noise .npy file.
        snr_range_db (list): SNR range for noise addition.

    Returns:
        tuple: (clean_eeg_data, eog_noise_data, emg_noise_data) as normalized NumPy arrays.
               Returns None for noise data if the file is not found.

    Raises:
        FileNotFoundError: If the clean EEG data file is not found.
        ValueError: If neither EOG nor EMG noise data can be loaded.
    """
    clean_eeg_data = None
    eog_noise_data = None
    emg_noise_data = None

    # Helper function to load .npy files and handle basic shape/error checking
    def load_npy_file_safe(path, name):
        if not os.path.exists(path):
            print(f"Warning: {name} file not found at '{path}'. Skipping loading.")
            return None
        try:
            data = np.load(path, allow_pickle=True)
            # Ensure data is 2D (epochs, samples)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            elif data.ndim > 2:
                print(f"Warning: {name} data has more than 2 dimensions ({data.shape}). "
                      f"Assuming first dimension is epochs and attempting to flatten or select first channel.")
                # If it's (epochs, channels, samples), take the first channel
                if data.shape[1] > 1:
                    data = data[:, 0, :]
                else: # Otherwise, flatten the remaining dimensions
                    data = data.reshape(data.shape[0], -1)
            print(f"Loaded {name} data with shape: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading {name} from '{path}': {e}")
            return None

    # Load all necessary data files
    clean_eeg_data = load_npy_file_safe(eeg_file, "Clean EEG")
    eog_noise_data = load_npy_file_safe(eog_file, "EOG Noise")
    emg_noise_data = load_npy_file_safe(emg_file, "EMG Noise")

    # Essential validation: Clean EEG data must be present
    if clean_eeg_data is None:
        raise FileNotFoundError(f"Clean EEG data is essential and could not be loaded from '{eeg_file}'.")

    # Essential validation: At least one type of noise data must be present
    if eog_noise_data is None and emg_noise_data is None:
        raise ValueError("Neither EOG nor EMG noise data could be loaded. At least one is required for synthetic noise generation.")

    # --- Optional: Normalize all loaded data ---
    # Normalizing data to a consistent range (e.g., [-1, 1]) is crucial for
    # stable neural network training, especially for GANs.
    def normalize_data_to_range(data):
        if data is None:
            return None
        # Normalize each epoch independently to [-1, 1]
        # This handles potential variations in amplitude across different epochs
        min_vals = data.min(axis=1, keepdims=True)
        max_vals = data.max(axis=1, keepdims=True)
        # Avoid division by zero for flat signals (where min_val == max_val)
        range_vals = max_vals - min_vals
        normalized_data = np.where(range_vals == 0, 0, 2 * (data - min_vals) / range_vals - 1)
        return normalized_data

    clean_eeg_data = normalize_data_to_range(clean_eeg_data)
    eog_noise_data = normalize_data_to_range(eog_noise_data)
    emg_noise_data = normalize_data_to_range(emg_noise_data)

    return clean_eeg_data, eog_noise_data, emg_noise_data


# --- 3. Train/Test Split (single source of truth) ---
def split_train_test(clean_eeg, eog_noise, emg_noise, train_ratio=TRAIN_SPLIT_RATIO, seed=42):
    """Split the clean EEG and both noise pools into train and test sets.

    Every script (training and all evaluators) calls this with the same seed, so
    the held-out sets are identical everywhere. The noise pools are split too, so
    no test-time EOG/EMG epoch is ever seen during training.

    Returns:
        (train_clean, test_clean, train_eog, test_eog, train_emg, test_emg).
        A noise pool that is None or has a single epoch is returned unchanged for
        both train and test.
    """
    test_size = 1 - train_ratio

    def _split(pool):
        if pool is None or len(pool) < 2:
            return pool, pool
        return train_test_split(pool, test_size=test_size, random_state=seed)

    train_clean, test_clean = train_test_split(clean_eeg, test_size=test_size, random_state=seed)
    train_eog, test_eog = _split(eog_noise)
    train_emg, test_emg = _split(emg_noise)
    return train_clean, test_clean, train_eog, test_eog, train_emg, test_emg


def make_or_load_shared_sample(path, clean_epoch, eog, emg, snr_db=-6, seed=42):
    """Return a fixed (clean, noisy) pair at the given SNR, shared by all methods.

    The pair is generated once from the given clean epoch and EOG+EMG artifacts,
    mean-subtracted, and cached to ``path`` so every evaluator plots the exact
    same sample. If the cache exists it is loaded and reused.
    """
    if os.path.exists(path):
        arr = np.load(path)
        return arr["clean"], arr["noisy"]

    np.random.seed(seed)
    clean_epoch = np.asarray(clean_epoch, dtype=np.float64).flatten()
    eog = eog if eog is not None else np.zeros_like(clean_epoch)
    emg = emg if emg is not None else np.zeros_like(clean_epoch)
    noise_epoch = np.asarray(eog, dtype=np.float64).flatten() + np.asarray(emg, dtype=np.float64).flatten()

    clean_power = np.mean(clean_epoch ** 2)
    noise_power = np.mean(noise_epoch ** 2)
    snr_linear = 10 ** (snr_db / 10)
    alpha = np.sqrt(clean_power / (snr_linear * noise_power)) if noise_power > 0 else 0
    noisy = clean_epoch + alpha * noise_epoch

    noisy = noisy - np.mean(noisy)
    clean_epoch = clean_epoch - np.mean(clean_epoch)
    np.savez(path, clean=clean_epoch, noisy=noisy)
    return clean_epoch, noisy


# --- Example Usage (This block will run when the script is executed directly) ---
if __name__ == "__main__":
    print("--- Starting EEG Data Preparation ---")

    # 1. Prepare raw data (load and normalize)
    # This step handles file loading, basic validation, and normalization.
    try:
        clean_eeg_np, eog_noise_np, emg_noise_np = prepare_eeg_data(EEG_FILE, EOG_FILE, EMG_FILE, SNR_RANGE_DB)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nData preparation failed: {e}")
        print("Please ensure your .npy files exist at the specified paths and are correctly structured.")
        exit() # Exit if essential data cannot be loaded

    # 2. Create the custom PyTorch Dataset
    # This dataset will dynamically generate noisy-clean pairs during training.
    eeg_denoising_dataset = EEGNoiseDataset(clean_eeg_np, eog_noise_np, emg_noise_np, SNR_RANGE_DB)

    # 3. Create a PyTorch DataLoader for efficient batching during training
    batch_size = 32 # Define your desired batch size
    # The DataLoader shuffles data for better training and handles batching.
    eeg_dataloader = DataLoader(eeg_denoising_dataset, batch_size=batch_size, shuffle=True)

    print(f"\nSuccessfully created PyTorch DataLoader with batch size {batch_size}.")
    print(f"Total batches available per epoch: {len(eeg_dataloader)}")

    # 4. Example of iterating through the DataLoader (simulating a training loop)
    print("\n--- Demonstrating one batch from the DataLoader ---")
    for i, (noisy_batch, clean_batch) in enumerate(eeg_dataloader):
        print(f"Batch {i+1}:")
        # The shape will be (batch_size, channels, num_samples)
        print(f"  Noisy EEG batch shape: {noisy_batch.shape}")
        print(f"  Clean EEG batch shape: {clean_batch.shape}")

        # You would typically move these batches to your training device (CPU/GPU)
        # e.g., noisy_batch = noisy_batch.to(device)
        # e.g., clean_batch = clean_batch.to(device)

        # Then, you would feed 'noisy_batch' to your Generator model
        # and use both 'noisy_batch' and 'clean_batch' with your Critic model
        # to calculate losses and perform backpropagation.

        # For demonstration, we'll just show the first batch and break.
        break

    print("\nData generation setup complete. The 'eeg_dataloader' is ready to be used in your PyTorch GAN training loop.")