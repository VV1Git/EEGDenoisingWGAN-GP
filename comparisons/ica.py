import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from joblib import Parallel, delayed  # ships as a scikit-learn dependency

# Import shared variables, data utilities, the shared metrics, the shared ICA
# denoiser (the same one postergen.py uses), and the shared PSD plot. Using the
# same metrics as evaluate.py, on the same held-out test split, guarantees ICA is
# scored fairly against AR-WGAN and the Wiener filter.
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from variables import (
    EEG_FILE, EOG_FILE, EMG_FILE, SNR_RANGE_DB_EVAL,
    BATCH_SIZE, EEG_BANDS, SAMPLING_RATE, NUM_NOISE_VARIANTS, SHARED_SAMPLE_PATH
)
from eeg_data_generator import (
    prepare_eeg_data, EEGNoiseDataset, DataLoader, split_train_test, make_or_load_shared_sample,
)
from metrics import calculate_rrmse, calculate_rrmse_spectral, calculate_cc, calculate_band_power_ratios
from baselines import ica_denoise
from plots import plot_psd_comparison


# --- Main ICA Evaluation Logic ---
def main():
    ICA_EVAL_PLOTS_DIR = os.path.join(os.path.dirname(__file__), "ica_evaluation_plots")
    os.makedirs(ICA_EVAL_PLOTS_DIR, exist_ok=True)
    print(f"Created/Ensured '{ICA_EVAL_PLOTS_DIR}' directory exists for ICA evaluation plots.")

    # Load data and take the held-out test split (clean + noise pools) so ICA is
    # evaluated on exactly the epochs and noise the model never trained on.
    clean_eeg_all, eog_noise, emg_noise = prepare_eeg_data(
        EEG_FILE, EOG_FILE, EMG_FILE, [-100, -100]
    )
    _, test_clean_eeg_np, _, test_eog, _, test_emg = split_train_test(
        clean_eeg_all, eog_noise, emg_noise
    )

    # --- Shared -6 dB sample for all methods ---
    clean_epoch, noisy_signal = make_or_load_shared_sample(
        SHARED_SAMPLE_PATH,
        test_clean_eeg_np[0],
        test_eog[0] if test_eog is not None else None,
        test_emg[0] if test_emg is not None else None,
    )

    snr_values_db = SNR_RANGE_DB_EVAL
    rrmse_temporal_per_snr = []
    rrmse_spectral_per_snr = []
    cc_per_snr = []
    band_power_ratios_per_snr = {band: {'clean': [], 'noisy': [], 'denoised': []} for band in EEG_BANDS.keys()}
    example_psd_saved = False

    print("\n--- ICA evaluation across different SNRs ---")
    for current_snr_db in snr_values_db:
        print(f"\nEvaluating at SNR: {current_snr_db} dB")
        test_dataset_current_snr = EEGNoiseDataset(
            test_clean_eeg_np, test_eog, test_emg, [current_snr_db, current_snr_db],
            num_noise_variants_per_clean_epoch=NUM_NOISE_VARIANTS
        )
        test_loader_current_snr = DataLoader(
            test_dataset_current_snr,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        batch_rrmse_temporal = []
        batch_rrmse_spectral = []
        batch_cc = []
        clean_band_ratios_agg = {band: [] for band in EEG_BANDS.keys()}
        noisy_band_ratios_agg = {band: [] for band in EEG_BANDS.keys()}
        denoised_band_ratios_agg = {band: [] for band in EEG_BANDS.keys()}

        for batch_idx, (noisy_signals, clean_signals) in enumerate(tqdm(test_loader_current_snr, desc=f"SNR {current_snr_db}dB")):
            noisy_signals_np = noisy_signals.numpy()
            clean_signals_np = clean_signals.numpy()
            # Denoise the whole batch in parallel. ica_denoise is deterministic
            # (FastICA random_state=0) and pure, so parallel execution returns
            # bit-identical results regardless of worker count/order; collecting
            # them in sample order below preserves the exact metric aggregation.
            denoised_batch = Parallel(n_jobs=-1, backend="loky")(
                delayed(ica_denoise)(noisy_signals_np[i, 0, :])
                for i in range(noisy_signals_np.shape[0])
            )
            for i in range(noisy_signals_np.shape[0]):
                noisy = noisy_signals_np[i, 0, :]
                clean = clean_signals_np[i, 0, :]
                denoised = denoised_batch[i]
                batch_rrmse_temporal.append(calculate_rrmse(clean, denoised))
                batch_rrmse_spectral.append(calculate_rrmse_spectral(clean, denoised, SAMPLING_RATE))
                batch_cc.append(calculate_cc(clean, denoised))
                # Band power ratios
                clean_ratios = calculate_band_power_ratios(clean, SAMPLING_RATE, EEG_BANDS)
                noisy_ratios = calculate_band_power_ratios(noisy, SAMPLING_RATE, EEG_BANDS)
                denoised_ratios = calculate_band_power_ratios(denoised, SAMPLING_RATE, EEG_BANDS)
                for band in EEG_BANDS.keys():
                    clean_band_ratios_agg[band].append(clean_ratios[f'{band}_ratio'])
                    noisy_band_ratios_agg[band].append(noisy_ratios[f'{band}_ratio'])
                    denoised_band_ratios_agg[band].append(denoised_ratios[f'{band}_ratio'])
                # Save one PSD comparison plot (first sample of the first SNR)
                if not example_psd_saved:
                    plot_psd_comparison(
                        clean, noisy, denoised, SAMPLING_RATE, EEG_BANDS,
                        method_name="ICA",
                        save_path=os.path.join(ICA_EVAL_PLOTS_DIR, "psd_comparison_example.png")
                    )
                    example_psd_saved = True

        rrmse_temporal_per_snr.append(np.mean(batch_rrmse_temporal))
        rrmse_spectral_per_snr.append(np.mean(batch_rrmse_spectral))
        cc_per_snr.append(np.mean(batch_cc))
        for band in EEG_BANDS.keys():
            band_power_ratios_per_snr[band]['clean'].append(np.mean(clean_band_ratios_agg[band]))
            band_power_ratios_per_snr[band]['noisy'].append(np.mean(noisy_band_ratios_agg[band]))
            band_power_ratios_per_snr[band]['denoised'].append(np.mean(denoised_band_ratios_agg[band]))

    # Band power ratio bar plots for each band
    x = np.arange(len(snr_values_db))
    width = 0.25
    for band in EEG_BANDS.keys():
        plt.figure(figsize=(10, 6))
        clean_vals = band_power_ratios_per_snr[band]['clean']
        noisy_vals = band_power_ratios_per_snr[band]['noisy']
        denoised_vals = band_power_ratios_per_snr[band]['denoised']
        max_val = max(
            max(clean_vals) if clean_vals else 0,
            max(noisy_vals) if noisy_vals else 0,
            max(denoised_vals) if denoised_vals else 0,
        )
        plt.bar(x - width, clean_vals, width, label='Clean', color='blue')
        plt.bar(x, noisy_vals, width, label='Noisy', color='red')
        plt.bar(x + width, denoised_vals, width, label='Denoised', color='green')
        plt.title("ICA", fontsize=24)
        plt.xlabel('SNR (dB)', fontsize=18)
        plt.ylabel('Power Ratio', fontsize=18)
        plt.ylim(0, max_val * 1.05 if max_val > 0 else 1)
        plt.xticks(x, [str(snr) for snr in snr_values_db])
        plt.grid(axis='y')
        plt.legend()
        fname = f'overall_{band}_power_ratio_vs_snr.png'
        plt.savefig(os.path.join(ICA_EVAL_PLOTS_DIR, fname))
        plt.close()
        print(f"Saved overall {band.capitalize()} band power ratio vs SNR bar chart to '{os.path.join(ICA_EVAL_PLOTS_DIR, fname)}'")

    # Save the -6 dB shared sample: clean, noisy, and ICA denoised
    denoised = ica_denoise(noisy_signal)
    sample_txt_path = os.path.join(ICA_EVAL_PLOTS_DIR, "sample_denoising_-6.txt")
    with open(sample_txt_path, "w") as f:
        f.write("Index\tClean\tNoisy\tDenoised\n")
        for i in range(len(clean_epoch)):
            f.write(f"{i}\t{clean_epoch[i]}\t{noisy_signal[i]}\t{denoised[i]}\n")
    print(f"Saved sample denoising signals to '{sample_txt_path}'")

    # RRMSE Temporal vs SNR
    plt.figure(figsize=(6, 5))
    plt.plot(snr_values_db, rrmse_temporal_per_snr, marker='o', linestyle='-', color='blue')
    plt.title('ICA', fontsize=24)
    plt.xlabel('SNR (dB)', fontsize=18)
    plt.ylabel('RRMSE Temporal', fontsize=18)
    plt.grid(True)
    plt.savefig(os.path.join(ICA_EVAL_PLOTS_DIR, 'RRMSE_Temporal_vs_SNR.png'))
    plt.close()

    # RRMSE Spectral vs SNR
    plt.figure(figsize=(6, 5))
    plt.plot(snr_values_db, rrmse_spectral_per_snr, marker='o', linestyle='-', color='blue')
    plt.title('ICA', fontsize=24)
    plt.xlabel('SNR (dB)', fontsize=18)
    plt.ylabel('RRMSE Spectral', fontsize=18)
    plt.grid(True)
    plt.savefig(os.path.join(ICA_EVAL_PLOTS_DIR, 'RRMSE_Spectral_vs_SNR.png'))
    plt.close()

    # CC vs SNR
    plt.figure(figsize=(6, 5))
    plt.plot(snr_values_db, cc_per_snr, marker='o', linestyle='-', color='blue')
    plt.title('ICA', fontsize=24)
    plt.xlabel('SNR (dB)', fontsize=18)
    plt.ylabel('Pearson\'s CC', fontsize=18)
    plt.grid(True)
    plt.savefig(os.path.join(ICA_EVAL_PLOTS_DIR, 'CC_vs_SNR.png'))
    plt.close()

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

    # --- Save CC and RRMSE vs SNR data to text files for overlay plotting ---
    cc_txt_path = os.path.join(ICA_EVAL_PLOTS_DIR, "cc_vs_snr.txt")
    rrmse_txt_path = os.path.join(ICA_EVAL_PLOTS_DIR, "rrmse_vs_snr.txt")
    rrmse_spectral_txt_path = os.path.join(ICA_EVAL_PLOTS_DIR, "rrmse_spectral_vs_snr.txt")
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

    # --- Save Band Power Ratios to text file for graphs.py ---
    # Calculate overall averages first (similar to how they are calculated for the plot below)
    band_names = list(EEG_BANDS.keys())
    avg_clean_all = [np.mean(band_power_ratios_per_snr[band]['clean']) for band in band_names]
    avg_noisy_all = [np.mean(band_power_ratios_per_snr[band]['noisy']) for band in band_names]
    avg_denoised_all = [np.mean(band_power_ratios_per_snr[band]['denoised']) for band in band_names]

    band_powers_txt_path = os.path.join(ICA_EVAL_PLOTS_DIR, "band_power_ratios.txt")
    with open(band_powers_txt_path, "w") as f:
        f.write("Band\tClean\tNoisy\tDenoised\n")
        for i, band in enumerate(band_names):
            f.write(f"{band}\t{avg_clean_all[i]}\t{avg_noisy_all[i]}\t{avg_denoised_all[i]}\n")
    print(f"Saved aggregated band power ratios to '{band_powers_txt_path}'")

    # --- Grouped bar chart: average power ratios for each band (ICA, half AR-WGAN width) ---
    band_names = list(EEG_BANDS.keys())
    avg_clean = [np.mean(band_power_ratios_per_snr[band]['clean']) for band in band_names]
    avg_noisy = [np.mean(band_power_ratios_per_snr[band]['noisy']) for band in band_names]
    avg_denoised = [np.mean(band_power_ratios_per_snr[band]['denoised']) for band in band_names]

    x = np.arange(len(band_names))
    width = 0.25
    plt.figure(figsize=(7, 8))  # ICA: half as wide as AR-WGAN
    plt.bar(x - width, avg_clean, width, label='Clean', color='blue')
    plt.bar(x, avg_noisy, width, label='Noisy', color='red')
    plt.bar(x + width, avg_denoised, width, label='Denoised', color='green')
    plt.title("ICA", fontsize=24)
    plt.xlabel('EEG Band', fontsize=18)
    plt.ylabel('Average Power Ratio', fontsize=18)
    plt.xticks(x, [b.capitalize() for b in band_names])
    plt.ylim(0, max(avg_clean + avg_noisy + avg_denoised) * 1.05)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(ICA_EVAL_PLOTS_DIR, "overall_band_power_ratios_grouped.png"))
    plt.close()
    print(f"Saved grouped band power ratio bar chart to '{os.path.join(ICA_EVAL_PLOTS_DIR, 'overall_band_power_ratios_grouped.png')}'")

    print("\nICA evaluation complete.")

if __name__ == "__main__":
    main()
