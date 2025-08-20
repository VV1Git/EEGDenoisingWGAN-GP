import numpy as np
import matplotlib.pyplot as plt
from eeg_data_generator import prepare_eeg_data, EEGNoiseDataset

def plot_eeg_overlay(clean_eeg, noisy_eeg, db_list, num_examples=4):
    plt.figure(figsize=(12, 2.5 * num_examples))
    for i in range(num_examples):
        artefact = noisy_eeg[i] - clean_eeg[i]  # Changed order here
        plt.subplot(num_examples, 1, i + 1)
        plt.plot(clean_eeg[i].squeeze(), label="Clean EEG", color='blue')
        plt.plot(noisy_eeg[i].squeeze(), label="Noisy EEG", color='red', alpha=0.6)
        plt.plot(artefact.squeeze(), label="Artefact (Noisy - Clean)", color='purple', alpha=0.7)
        plt.title(f"EEG Epoch {i+1}" + f" (SNR: {db_list[i]:.2f} dB)")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def main():
    EEG_FILE = 'dataset/EEG_all_epochs.npy'
    EOG_FILE = 'dataset/EOG_all_epochs.npy'
    EMG_FILE = 'dataset/EMG_all_epochs.npy'
    SNR_RANGE_DB = [-35, -20]
    BATCH_SIZE = 4

    try:
        clean_eeg_all, eog_noise_np, emg_noise_np = prepare_eeg_data(
            EEG_FILE, EOG_FILE, EMG_FILE, SNR_RANGE_DB
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error preparing data: {e}")
        return

    # Use only a subset for visualization
    clean_eeg_subset = clean_eeg_all[:BATCH_SIZE]
    # Generate a random SNR for each sample
    db_list = np.random.uniform(SNR_RANGE_DB[0], SNR_RANGE_DB[1], size=BATCH_SIZE)


    noisy_eeg_list = []
    clean_eeg_list = []
    for i in range(BATCH_SIZE):
        # Re-generate noisy signal for the specific SNR
        sample_dataset = EEGNoiseDataset(
            clean_eeg_subset[i:i+1], eog_noise_np, emg_noise_np, [db_list[i], db_list[i]],
            num_noise_variants_per_clean_epoch=1
        )
        noisy, clean = sample_dataset[0]
        noisy_eeg_list.append(noisy.numpy().squeeze())
        clean_eeg_list.append(clean.numpy().squeeze())

    plot_eeg_overlay(clean_eeg_list, noisy_eeg_list, db_list, num_examples=BATCH_SIZE)

if __name__ == "__main__":
    main()
