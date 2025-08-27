import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def combine_method_plots(ica_path, wiener_path, arwgan_path, output_path):
    """
    Combines ICA and Wiener plots on the top row, AR-WGAN on the bottom row, and saves the result.
    All images are resized to the same width as the smallest among them, and stacked accordingly.
    """
    ica_img = Image.open(ica_path)
    wiener_img = Image.open(wiener_path)
    arwgan_img = Image.open(arwgan_path)

    # Resize all images to the same width (minimum width among all)
    min_width = min(ica_img.width, wiener_img.width, arwgan_img.width)
    ica_img = ica_img.resize((min_width, int(ica_img.height * min_width / ica_img.width)))
    wiener_img = wiener_img.resize((min_width, int(wiener_img.height * min_width / wiener_img.width)))
    arwgan_img = arwgan_img.resize((min_width * 2, int(arwgan_img.height * (min_width * 2) / arwgan_img.width)))

    # Add a small border (reduce further to 2px)
    border = -4

    # Create a new blank image to combine with reduced border
    top_height = max(ica_img.height, wiener_img.height)
    bottom_height = arwgan_img.height
    total_width = min_width * 2 + border * 3
    total_height = top_height + bottom_height + border * 3

    combined = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    # Paste ICA (top-left)
    combined.paste(ica_img, (border, border))
    # Paste Wiener (top-right)
    combined.paste(wiener_img, (min_width + border * 2, border))
    # Paste AR-WGAN (bottom, centered horizontally)
    combined.paste(arwgan_img, (border, top_height + border * 2))

    combined.save(output_path)

def plot_overlay_metric(metric_name, ylabel, output_path, ica_dir, wiener_dir, arwgan_dir):
    """
    Overlay the metric (CC or RRMSE) vs SNR for all three methods.
    """
    method_dirs = {
        "ICA": ica_dir,
        "Wiener": wiener_dir,
        "AR-WGAN": arwgan_dir,
    }
    colors = {
        "ICA": "tab:orange",
        "Wiener": "tab:green",
        "AR-WGAN": "tab:blue",
    }
    metric_file = {
        "CC": "cc_vs_snr.txt",
        "RRMSE": "rrmse_vs_snr.txt",
        "RRMSE_Spectral": "rrmse_spectral_vs_snr.txt",
    }[metric_name]
    plt.figure(figsize=(8, 6))
    for method, dir_path in method_dirs.items():
        txt_path = os.path.join(dir_path, metric_file)
        if not os.path.exists(txt_path):
            print(f"Missing {metric_name} data for {method} at {txt_path}")
            continue
        data = np.loadtxt(txt_path, skiprows=1)
        snr = data[:, 0]
        value = data[:, 1]
        plt.plot(snr, value, marker='o', linestyle='-', color=colors[method], label=method)
    plt.xlabel("SNR (dB)", fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.title(f"{metric_name.replace('_', ' ')} vs SNR (All Methods)", fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Overlay plot saved to {output_path}")

def plot_overlay_sample_denoising(ica_dir, wiener_dir, arwgan_dir, output_path):
    """
    Overlay the denoised signals from all methods on the same clean signal.
    """
    method_dirs = {
        "ICA": ica_dir,
        "Wiener": wiener_dir,
        "AR-WGAN": arwgan_dir,
    }
    colors = {
        "ICA": "tab:orange",
        "Wiener": "red",
        "AR-WGAN": "tab:blue",
    }
    sample_files = {
        method: os.path.join(dir_path, "sample_denoising_-6.txt")
        for method, dir_path in method_dirs.items()
    }
    # Load the clean signal from any method (they should be identical)
    for method in method_dirs:
        if os.path.exists(sample_files[method]):
            data = np.loadtxt(sample_files[method], skiprows=1)
            idx = data[:, 0]
            clean = data[:, 1]
            break
    else:
        print("No sample signal files found for overlay.")
        return

    plt.figure(figsize=(14, 6))
    plt.plot(idx, clean, color='green', label='Clean', linewidth=2, alpha=1.0)
    for method in method_dirs:
        if os.path.exists(sample_files[method]):
            data = np.loadtxt(sample_files[method], skiprows=1)
            denoised = data[:, 3]
            plt.plot(
                idx, denoised,
                color=colors[method],
                label=f"{method} Denoised",
                linewidth=1.5,
                alpha=1.0
            )
        else:
            print(f"Missing sample signal for {method} at {sample_files[method]}")
    plt.xlabel("Sample Index", fontsize=16)
    plt.ylabel("Amplitude", fontsize=16)
    plt.title("Sample Denoising at SNR -6 dB (All Methods)", fontsize=20)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Overlay sample denoising plot saved to {output_path}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ica_dir = os.path.join(base_dir, "comparisions", "ica_evaluation_plots")
    wiener_dir = os.path.join(base_dir, "comparisions", "wiener_evaluation_plots")
    arwgan_dir = os.path.join(base_dir, "evaluation_plots")
    final_dir = os.path.join(base_dir, "finalplots")
    os.makedirs(final_dir, exist_ok=True)

    # List of plot types to combine (filenames must match across methods)
    plot_types = [
        "RRMSE_Temporal_vs_SNR.png",
        "RRMSE_Spectral_vs_SNR.png",
        "CC_vs_SNR.png",
    ]
    # Add band power ratio plots
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    for band in bands:
        plot_types.append(f"overall_{band}_power_ratio_vs_snr.png")

    # Add grouped band power ratio plot
    plot_types.append("overall_band_power_ratios_grouped.png")

    # Add multi_snr_sample_denoising plots (find all matching files)
    ica_multi = [f for f in os.listdir(ica_dir) if f.startswith("multi_snr_sample_denoising_")]
    for fname in ica_multi:
        plot_types.append(fname)

    for plot_name in plot_types:
        ica_path = os.path.join(ica_dir, plot_name)
        wiener_path = os.path.join(wiener_dir, plot_name)
        arwgan_path = os.path.join(arwgan_dir, plot_name)
        output_path = os.path.join(final_dir, plot_name)
        if os.path.exists(ica_path) and os.path.exists(wiener_path) and os.path.exists(arwgan_path):
            combine_method_plots(ica_path, wiener_path, arwgan_path, output_path)
            print(f"Combined plot saved to {output_path}")
        else:
            print(f"Skipping {plot_name}: missing one or more method plots.")

    # Overlay CC vs SNR and RRMSE vs SNR for all methods
    plot_overlay_metric(
        metric_name="CC",
        ylabel="Pearson's CC",
        output_path=os.path.join(final_dir, "CC_vs_SNR_overlay.png"),
        ica_dir=ica_dir,
        wiener_dir=wiener_dir,
        arwgan_dir=arwgan_dir,
    )
    plot_overlay_metric(
        metric_name="RRMSE",
        ylabel="RRMSE Temporal",
        output_path=os.path.join(final_dir, "RRMSE_Temporal_vs_SNR_overlay.png"),
        ica_dir=ica_dir,
        wiener_dir=wiener_dir,
        arwgan_dir=arwgan_dir,
    )
    plot_overlay_metric(
        metric_name="RRMSE_Spectral",
        ylabel="RRMSE Spectral",
        output_path=os.path.join(final_dir, "RRMSE_Spectral_vs_SNR_overlay.png"),
        ica_dir=ica_dir,
        wiener_dir=wiener_dir,
        arwgan_dir=arwgan_dir,
    )

    # Overlay sample denoising for all methods
    plot_overlay_sample_denoising(
        ica_dir=ica_dir,
        wiener_dir=wiener_dir,
        arwgan_dir=arwgan_dir,
        output_path=os.path.join(final_dir, "sample_denoising_-6_overlay.png"),
    )

if __name__ == "__main__":
    main()
