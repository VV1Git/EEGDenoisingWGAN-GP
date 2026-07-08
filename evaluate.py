import torch
import numpy as np
import matplotlib.pyplot as plt # type:ignore
import os
from tqdm import tqdm

# Import shared variables
from variables import (
    EEG_FILE, EOG_FILE, EMG_FILE, SNR_RANGE_DB_EVAL, SNR_RANGE_DB, SAVED_MODEL_PATH, EVAL_PLOTS_DIR,
    CHANNELS_EEG, FEATURES_GEN, BATCH_SIZE, EEG_BANDS, SAMPLING_RATE, PSD_SAMPLE_INDEX_FOR_VIZ,
    NUM_NOISE_VARIANTS, SHARED_SAMPLE_PATH
)

# Import the Generator, the shared evaluation metrics (single source of truth,
# identical across AR-WGAN / ICA / Wiener), the shared data helpers, and the
# shared PSD plot.
from model import Generator
from metrics import (
    calculate_rrmse, calculate_rrmse_spectral, calculate_cc,
    calculate_band_power_ratios, calculate_cosine_similarity_power_ratios,
)
from eeg_data_generator import (
    prepare_eeg_data, EEGNoiseDataset, DataLoader, split_train_test, make_or_load_shared_sample,
)
from plots import plot_psd_comparison

# --- Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Output directory for evaluation plots
os.makedirs(EVAL_PLOTS_DIR, exist_ok=True)
print(f"Created/Ensured '{EVAL_PLOTS_DIR}' directory exists for evaluation plots.")

SAMPLES_PER_EPOCH = None  # Will be set after loading data


# --- Main Evaluation Logic ---
def main():
    # 1. Load and prepare data (ensure consistent splitting with training)
    try:
        clean_eeg_all, eog_noise, emg_noise = prepare_eeg_data(
            EEG_FILE, EOG_FILE, EMG_FILE, [-100, -100] # Use a dummy SNR range for loading, actual SNR will be set per test
        )
        SAMPLES_PER_EPOCH = clean_eeg_all.shape[1]
    except (FileNotFoundError, ValueError) as e:
        print(f"Error preparing data: {e}")
        print("Please ensure your dataset files are correctly placed and named.")
        return

    # Re-derive the held-out test set with the shared split so the clean epochs
    # AND the noise pools are identical to (and disjoint from) what train.py used.
    _, test_clean_eeg_np, _, test_eog, _, test_emg = split_train_test(
        clean_eeg_all, eog_noise, emg_noise
    )
    print(f"Held-out test set: {test_clean_eeg_np.shape[0]} clean EEG epochs.")

    # 2. Load the trained Generator model
    generator = Generator(CHANNELS_EEG, SAMPLES_PER_EPOCH, FEATURES_GEN).to(device)
    try:
        checkpoint = torch.load(SAVED_MODEL_PATH, map_location=device)
        generator.load_state_dict(checkpoint['gen'])
        print(f"Successfully loaded generator model from '{SAVED_MODEL_PATH}'")
    except Exception as e:
        print(f"Error loading generator model from '{SAVED_MODEL_PATH}': {e}")
        print("Please ensure the model path is correct and the file exists.")
        return

    generator.eval() # Set generator to evaluation mode (disables dropout, batch norm updates)

    # --- Data collection for SNR vs. Metrics plots ---
    snr_values_db = SNR_RANGE_DB_EVAL
    rrmse_temporal_per_snr = []
    rrmse_spectral_per_snr = []
    cc_per_snr = []
    cosine_sim_power_ratios_at_neg14db = [] # cosine similarity of power ratios at -14 dB

    # --- Store band power ratios for all SNRs for plotting later ---
    band_power_ratios_per_snr = {band: {'clean': [], 'noisy': [], 'denoised': []} for band in EEG_BANDS.keys()}

    # --- Shared sample for all methods (fixed -6 dB clean/noisy pair) ---
    clean_epoch, noisy_signal = make_or_load_shared_sample(
        SHARED_SAMPLE_PATH,
        test_clean_eeg_np[0],
        test_eog[0] if test_eog is not None else None,
        test_emg[0] if test_emg is not None else None,
    )

    print("\n--- Starting evaluation across different SNRs ---")
    for current_snr_db in snr_values_db:
        print(f"\nEvaluating at SNR: {current_snr_db} dB")
        # Create a test dataset specifically for this SNR
        # The EEGNoiseDataset's __getitem__ will now use this specific SNR
        test_dataset_current_snr = EEGNoiseDataset(
            test_clean_eeg_np, test_eog, test_emg, [current_snr_db, current_snr_db], # Fixed SNR for testing
            num_noise_variants_per_clean_epoch=NUM_NOISE_VARIANTS # Pass the new parameter
        )
        # num_workers/pin_memory are gated on CUDA so the metric-producing CPU
        # path stays serial (num_workers=0) and RNG-order-identical to before;
        # the __getitem__ noise synthesis uses the un-seeded global numpy RNG,
        # so adding workers there would change the drawn noisy signals.
        test_loader_current_snr = DataLoader(
            test_dataset_current_snr,
            batch_size=BATCH_SIZE,
            shuffle=False, # Do NOT shuffle for consistent evaluation
            num_workers=(4 if device == "cuda" else 0),
            pin_memory=(device == "cuda"),
            persistent_workers=(device == "cuda"),
        )

        batch_rrmse_temporal = []
        batch_rrmse_spectral = []
        batch_cc = []
        batch_cosine_sim_power_ratios = []

        # --- Band power ratio aggregation for each SNR ---
        clean_band_ratios_agg = {band: [] for band in EEG_BANDS.keys()}
        noisy_band_ratios_agg = {band: [] for band in EEG_BANDS.keys()}
        denoised_band_ratios_agg = {band: [] for band in EEG_BANDS.keys()}

        with torch.no_grad():
            for batch_idx, (noisy_signals, clean_signals) in enumerate(tqdm(test_loader_current_snr, desc=f"SNR {current_snr_db}dB")):
                # Only noisy_signals is fed to the model, so only it needs the
                # device. clean_signals is used solely for numpy metrics, so we
                # keep it on the CPU (avoids a wasted H2D + D2H round-trip).
                noisy_signals = noisy_signals.to(device, non_blocking=True)
                denoised_signals = generator(noisy_signals)

                noisy_signals_np = noisy_signals.cpu().numpy()
                clean_signals_np = clean_signals.numpy()
                denoised_signals_np = denoised_signals.cpu().numpy()

                for i in range(noisy_signals.shape[0]):
                    # Temporal RRMSE
                    batch_rrmse_temporal.append(calculate_rrmse(clean_signals_np[i], denoised_signals_np[i]))
                    # Spectral RRMSE
                    batch_rrmse_spectral.append(calculate_rrmse_spectral(clean_signals_np[i], denoised_signals_np[i], SAMPLING_RATE))
                    # Pearson's CC
                    batch_cc.append(calculate_cc(clean_signals_np[i], denoised_signals_np[i]))
                    
                    # Calculate Cosine Similarity of Power Ratios specifically at -14dB
                    if current_snr_db == -14:
                        batch_cosine_sim_power_ratios.append(
                            calculate_cosine_similarity_power_ratios(
                                clean_signals_np[i, 0, :], # Pass 1D signal
                                denoised_signals_np[i, 0, :], # Pass 1D signal
                                SAMPLING_RATE,
                                EEG_BANDS
                            )
                        )

                    # --- Band power ratios for each SNR ---
                    clean_ratios = calculate_band_power_ratios(clean_signals_np[i, 0, :], SAMPLING_RATE, EEG_BANDS)
                    noisy_ratios = calculate_band_power_ratios(noisy_signals_np[i, 0, :], SAMPLING_RATE, EEG_BANDS)
                    denoised_ratios = calculate_band_power_ratios(denoised_signals_np[i, 0, :], SAMPLING_RATE, EEG_BANDS)
                    for band in EEG_BANDS.keys():
                        clean_band_ratios_agg[band].append(clean_ratios[f'{band}_ratio'])
                        noisy_band_ratios_agg[band].append(noisy_ratios[f'{band}_ratio'])
                        denoised_band_ratios_agg[band].append(denoised_ratios[f'{band}_ratio'])

            # Aggregate metrics for the current SNR
            rrmse_temporal_per_snr.append(np.mean(batch_rrmse_temporal))
            rrmse_spectral_per_snr.append(np.mean(batch_rrmse_spectral))
            cc_per_snr.append(np.mean(batch_cc))

            # If we are at -14dB, aggregate the cosine similarity for this SNR
            if current_snr_db == -14:
                cosine_sim_power_ratios_at_neg14db.append(np.mean(batch_cosine_sim_power_ratios))

        # --- Save band power ratio bar plots for each SNR ---
        for band in EEG_BANDS.keys():
            avg_clean_ratio = np.mean(clean_band_ratios_agg[band])
            avg_noisy_ratio = np.mean(noisy_band_ratios_agg[band])
            avg_denoised_ratio = np.mean(denoised_band_ratios_agg[band])
            # --- Instead of plotting here, store for later ---
            band_power_ratios_per_snr[band]['clean'].append(avg_clean_ratio)
            band_power_ratios_per_snr[band]['noisy'].append(avg_noisy_ratio)
            band_power_ratios_per_snr[band]['denoised'].append(avg_denoised_ratio)

    # --- After SNR loop: Plot band power ratios vs SNR for each band as bar chart ---
    for band in EEG_BANDS.keys():
        x = np.arange(len(snr_values_db))
        width = 0.25
        clean_vals = band_power_ratios_per_snr[band]['clean']
        noisy_vals = band_power_ratios_per_snr[band]['noisy']
        denoised_vals = band_power_ratios_per_snr[band]['denoised']
        max_val = max(
            max(clean_vals) if clean_vals else 0,
            max(noisy_vals) if noisy_vals else 0,
            max(denoised_vals) if denoised_vals else 0,
        )
        plt.figure(figsize=(20, 6))  # width doubled from 10 to 20
        plt.bar(x - width, clean_vals, width, label='Clean', color='blue')
        plt.bar(x, noisy_vals, width, label='Noisy', color='red')
        plt.bar(x + width, denoised_vals, width, label='Denoised', color='green')
        plt.title("AR-WGAN", fontsize=24)
        plt.xlabel('SNR (dB)', fontsize=18)
        plt.ylabel('Power Ratio', fontsize=18)
        plt.ylim(0, max_val * 1.05 if max_val > 0 else 1)
        plt.xticks(x, [str(snr) for snr in snr_values_db])
        plt.grid(axis='y')
        plt.legend()
        fname = f'overall_{band}_power_ratio_vs_snr.png'
        plt.savefig(os.path.join(EVAL_PLOTS_DIR, fname))
        plt.close()
        print(f"Saved overall {band.capitalize()} band power ratio vs SNR bar chart to '{os.path.join(EVAL_PLOTS_DIR, fname)}'")

    # Save the -6 dB shared sample: clean, noisy, and AR-WGAN denoised
    noisy = noisy_signal
    clean = clean_epoch
    with torch.no_grad():
        denoised = generator(torch.from_numpy(noisy).float().unsqueeze(0).unsqueeze(0).to(device)).cpu().detach().numpy().flatten()
    sample_txt_path = os.path.join(EVAL_PLOTS_DIR, "sample_denoising_-6.txt")
    with open(sample_txt_path, "w") as f:
        f.write("Index\tClean\tNoisy\tDenoised\n")
        for i in range(len(clean)):
            f.write(f"{i}\t{clean[i]}\t{noisy[i]}\t{denoised[i]}\n")
    print(f"Saved sample denoising signals to '{sample_txt_path}'")

    print("\n--- Plotting SNR vs. Metrics ---")
    # Plot RRMSE Temporal vs SNR
    plt.figure(figsize=(12, 5))
    plt.plot(snr_values_db, rrmse_temporal_per_snr, marker='o', linestyle='-', color='blue')
    plt.title("AR-WGAN", fontsize=24)
    plt.xlabel('SNR (dB)', fontsize=18)
    plt.ylabel('RRMSE Temporal', fontsize=18)
    plt.grid(True)
    plt.savefig(os.path.join(EVAL_PLOTS_DIR, 'RRMSE_Temporal_vs_SNR.png'))
    plt.close()
    print(f"Saved RRMSE Temporal plot to '{os.path.join(EVAL_PLOTS_DIR, 'RRMSE_Temporal_vs_SNR.png')}'")

    # Plot RRMSE Spectral vs SNR
    plt.figure(figsize=(12, 5))
    plt.plot(snr_values_db, rrmse_spectral_per_snr, marker='o', linestyle='-', color='blue')
    plt.title("AR-WGAN", fontsize=24)
    plt.xlabel('SNR (dB)', fontsize=18)
    plt.ylabel('RRMSE Spectral', fontsize=18)
    plt.grid(True)
    plt.savefig(os.path.join(EVAL_PLOTS_DIR, 'RRMSE_Spectral_vs_SNR.png'))
    plt.close()
    print(f"Saved RRMSE Spectral plot to '{os.path.join(EVAL_PLOTS_DIR, 'RRMSE_Spectral_vs_SNR.png')}'")

    # Plot CC vs SNR
    plt.figure(figsize=(12, 5))
    plt.plot(snr_values_db, cc_per_snr, marker='o', linestyle='-', color='blue')
    plt.title("AR-WGAN", fontsize=24)
    plt.xlabel('SNR (dB)', fontsize=18)
    plt.ylabel('Pearson\'s CC', fontsize=18)
    plt.grid(True)
    plt.savefig(os.path.join(EVAL_PLOTS_DIR, 'CC_vs_SNR.png'))
    plt.close()
    print(f"Saved Pearson's CC plot to '{os.path.join(EVAL_PLOTS_DIR, 'CC_vs_SNR.png')}'")

    # --- Save CC and RRMSE vs SNR data to text files for overlay plotting ---
    cc_txt_path = os.path.join(EVAL_PLOTS_DIR, "cc_vs_snr.txt")
    rrmse_txt_path = os.path.join(EVAL_PLOTS_DIR, "rrmse_vs_snr.txt")
    rrmse_spectral_txt_path = os.path.join(EVAL_PLOTS_DIR, "rrmse_spectral_vs_snr.txt")
    with open(cc_txt_path, "w") as f:
        f.write("SNR_dB\tCC\n")
        for snr, cc in zip(snr_values_db, cc_per_snr):
            f.write(f"{snr}\t{cc}\n")
    with open(rrmse_txt_path, "w") as f:
        f.write("SNR_dB\tRRMSE\n")
        for snr, rrmse in zip(snr_values_db, rrmse_temporal_per_snr):
            f.write(f"{snr}\t{rrmse}\n")
    with open(rrmse_spectral_txt_path, "w") as f:
        f.write("SNR_dB\tRRMSE_Spectral\n")
        for snr, rrmse_spec in zip(snr_values_db, rrmse_spectral_per_snr):
            f.write(f"{snr}\t{rrmse_spec}\n")
    print(f"Saved CC vs SNR data to '{cc_txt_path}'")
    print(f"Saved RRMSE vs SNR data to '{rrmse_txt_path}'")
    print(f"Saved RRMSE Spectral vs SNR data to '{rrmse_spectral_txt_path}'")

    # --- Original Aggregated Metrics (kept for overall performance at random SNRs) ---
    print("\n--- Aggregated Evaluation Metrics (Overall Test Set) ---")
    # Re-create a test loader with the original random SNR range for overall metrics
    # Note: SNR_RANGE_DB here is the range used during training (e.g., [-5, 5])
    test_dataset_overall = EEGNoiseDataset(
        test_clean_eeg_np, test_eog, test_emg, SNR_RANGE_DB, # Use the original range for overall metrics
        num_noise_variants_per_clean_epoch=NUM_NOISE_VARIANTS # Pass the new parameter
    )
    test_loader_overall = DataLoader(
        test_dataset_overall,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=(4 if device == "cuda" else 0),
        pin_memory=(device == "cuda"),
        persistent_workers=(device == "cuda"),
    )

    all_rrmse_overall = []
    all_cc_overall = []
    clean_band_ratios_agg_overall = {band: [] for band in EEG_BANDS.keys()}
    noisy_band_ratios_agg_overall = {band: [] for band in EEG_BANDS.keys()}
    denoised_band_ratios_agg_overall = {band: [] for band in EEG_BANDS.keys()}

    with torch.no_grad():
        for batch_idx, (noisy_signals, clean_signals) in enumerate(tqdm(test_loader_overall, desc="Overall Metrics")):
            # clean_signals is only used for numpy metrics; keep it on CPU.
            noisy_signals = noisy_signals.to(device, non_blocking=True)
            denoised_signals = generator(noisy_signals)

            noisy_signals_np = noisy_signals.cpu().numpy()
            clean_signals_np = clean_signals.numpy()
            denoised_signals_np = denoised_signals.cpu().numpy()

            for i in range(noisy_signals.shape[0]):
                all_rrmse_overall.append(calculate_rrmse(clean_signals_np[i], denoised_signals_np[i]))
                all_cc_overall.append(calculate_cc(clean_signals_np[i], denoised_signals_np[i]))
            
            clean_ratios_batch = calculate_band_power_ratios(clean_signals_np.squeeze(1), SAMPLING_RATE, EEG_BANDS)
            noisy_ratios_batch = calculate_band_power_ratios(noisy_signals_np.squeeze(1), SAMPLING_RATE, EEG_BANDS)
            denoised_ratios_batch = calculate_band_power_ratios(denoised_signals_np.squeeze(1), SAMPLING_RATE, EEG_BANDS)

            for band in EEG_BANDS.keys():
                clean_band_ratios_agg_overall[band].append(clean_ratios_batch[f'{band}_ratio'])
                noisy_band_ratios_agg_overall[band].append(noisy_ratios_batch[f'{band}_ratio'])
                denoised_band_ratios_agg_overall[band].append(denoised_ratios_batch[f'{band}_ratio'])
            
            if batch_idx == 0: # Only plot one example from the overall test set
                # Plot PSD comparison for one sample from this batch
                plot_psd_comparison(
                    clean_signals_np[PSD_SAMPLE_INDEX_FOR_VIZ, 0, :], # Access specific sample and channel
                    noisy_signals_np[PSD_SAMPLE_INDEX_FOR_VIZ, 0, :],
                    denoised_signals_np[PSD_SAMPLE_INDEX_FOR_VIZ, 0, :],
                    SAMPLING_RATE,
                    EEG_BANDS,
                    method_name="AR-WGAN",
                    save_path=os.path.join(EVAL_PLOTS_DIR, "psd_comparison_example.png")
                )
                print(f"Saved PSD comparison plot to '{os.path.join(EVAL_PLOTS_DIR, 'psd_comparison_example.png')}'")


    avg_rrmse_overall = np.mean(all_rrmse_overall)
    avg_cc_overall = np.mean(all_cc_overall)

    print(f"Overall Average RRMSE on Test Set: {avg_rrmse_overall:.4f}")
    print(f"Overall Average Pearson's CC on Test Set: {avg_cc_overall:.4f}")

    print("\nOverall Average Band Power Ratios (Clean vs. Noisy vs. Denoised):")
    for band in EEG_BANDS.keys():
        avg_clean_ratio = np.mean(clean_band_ratios_agg_overall[band])
        avg_noisy_ratio = np.mean(noisy_band_ratios_agg_overall[band])
        avg_denoised_ratio = np.mean(denoised_band_ratios_agg_overall[band])
        print(f"  {band.capitalize()} Band:")
        print(f"    Clean: {avg_clean_ratio:.4f}")
        print(f"    Noisy: {avg_noisy_ratio:.4f}")
        print(f"    Denoised: {avg_denoised_ratio:.4f}")

    # Report Cosine Similarity of Power Ratios at -14dB
    if cosine_sim_power_ratios_at_neg14db:
        avg_cosine_sim_at_neg14db = np.mean(cosine_sim_power_ratios_at_neg14db)
        print(f"\nAverage Cosine Similarity of Power Ratios (Clean vs. Denoised) at -14dB: {avg_cosine_sim_at_neg14db:.4f}")

    # --- Save Band Power Ratios to text file for graphs.py ---
    band_powers_txt_path = os.path.join(EVAL_PLOTS_DIR, "band_power_ratios.txt")
    with open(band_powers_txt_path, "w") as f:
        f.write("Band\tClean\tNoisy\tDenoised\n")
        for band in EEG_BANDS.keys():
            avg_clean = np.mean(clean_band_ratios_agg_overall[band])
            avg_noisy = np.mean(noisy_band_ratios_agg_overall[band])
            avg_denoised = np.mean(denoised_band_ratios_agg_overall[band])
            f.write(f"{band}\t{avg_clean}\t{avg_noisy}\t{avg_denoised}\n")
    print(f"Saved aggregated band power ratios to '{band_powers_txt_path}'")

    # --- Print summary statistics at the end ---
    print("\n--- Summary Statistics Across SNRs ---")
    print(f"Average CC across SNRs: {np.mean(cc_per_snr):.4f} ± {np.std(cc_per_snr):.4f}")
    print(f"Average RRMSE (Temporal) across SNRs: {np.mean(rrmse_temporal_per_snr):.4f} ± {np.std(rrmse_temporal_per_snr):.4f}")
    print(f"Average RRMSE (Spectral) across SNRs: {np.mean(rrmse_spectral_per_snr):.4f} ± {np.std(rrmse_spectral_per_snr):.4f}")

    print("\nPSD Ratio (Denoised/Clean) across SNRs for each frequency band:")
    for band in EEG_BANDS.keys():
        denoised = np.array(band_power_ratios_per_snr[band]['denoised'])
        clean = np.array(band_power_ratios_per_snr[band]['clean'])
        ratio = denoised / (clean + 1e-12)  # avoid division by zero
        print(f"  {band.capitalize()}: Mean={np.mean(ratio):.4f}, Std={np.std(ratio):.4f}")

    # --- Grouped bar chart: average power ratios for each band (AR-WGAN, twice as wide) ---
    band_names = list(EEG_BANDS.keys())
    avg_clean = [np.mean(band_power_ratios_per_snr[band]['clean']) for band in band_names]
    avg_noisy = [np.mean(band_power_ratios_per_snr[band]['noisy']) for band in band_names]
    avg_denoised = [np.mean(band_power_ratios_per_snr[band]['denoised']) for band in band_names]

    x = np.arange(len(band_names))
    width = 0.25
    plt.figure(figsize=(14, 8))  # AR-WGAN: twice as wide
    plt.bar(x - width, avg_clean, width, label='Clean', color='blue')
    plt.bar(x, avg_noisy, width, label='Noisy', color='red')
    plt.bar(x + width, avg_denoised, width, label='Denoised', color='green')
    plt.title("AR-WGAN", fontsize=24)
    plt.xlabel('EEG Band', fontsize=18)
    plt.ylabel('Average Power Ratio', fontsize=18)
    plt.xticks(x, [b.capitalize() for b in band_names])
    plt.ylim(0, max(avg_clean + avg_noisy + avg_denoised) * 1.05)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_PLOTS_DIR, "overall_band_power_ratios_grouped.png"))
    plt.close()
    print(f"Saved grouped band power ratio bar chart to '{os.path.join(EVAL_PLOTS_DIR, 'overall_band_power_ratios_grouped.png')}'")


if __name__ == "__main__":
    main()
