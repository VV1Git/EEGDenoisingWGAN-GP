import os
from PIL import Image

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

if __name__ == "__main__":
    main()
