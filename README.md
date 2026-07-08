# EEGDenoisingWGAN-GP

Code for denoising EEG signals with a Wasserstein GAN with gradient penalty
(WGAN-GP). The model removes eye (EOG) and muscle (EMG) artifacts from
single-channel EEG. It is trained and tested on the EEGdenoiseNet benchmark and
compared against ICA and a Wiener filter.

## How it works

The generator is a 1D U-Net that maps a noisy EEG epoch to a clean one. The
critic scores clean epochs against the generator's output, and the gradient
penalty keeps training stable. The generator is trained with an adversarial term
plus an L1 reconstruction term against the clean signal.

Training data is built as it is needed. `prepare_eeg_data` loads the clean EEG,
EOG, and EMG arrays and normalizes each epoch to the range [-1, 1].
`EEGNoiseDataset` then adds a randomly chosen artifact (EOG, EMG, or both) at a
random SNR to make each noisy/clean pair.

The clean EEG and both noise pools are split into train and test with
`split_train_test`, using the same seed everywhere. Training uses only the
training epochs and the training noise. Every evaluator uses only the test epochs
and the test noise, so nothing the model saw during training appears at test
time.

## Requirements

Python 3, plus numpy, scipy, matplotlib, scikit-learn, tqdm, torch, joblib, and
Pillow. See `requirements.txt`.

## Getting started

Set up a virtual environment from the repo root:

```bash
py -3.14 -m venv .venv          # or: python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Put the `.npy` files for clean EEG, EOG, and EMG in `dataset/`. Settings such as
the learning rate, batch size, and SNR range are in `variables.py`.

Then run, from the repo root, in this order:

1. `train.py` trains the model and saves the generator to `model/`.
2. `evaluate.py` scores the trained generator across SNR levels and writes plots
   and metric text files to `evaluation_plots/`.
3. `comparisons/ica.py` and `comparisons/wiener_filter.py` do the same for the
   two baselines.
4. `graphs.py` reads the per-method outputs and builds the combined figures in
   `finalplots/`.
5. `postergen.py` builds the poster figures in `posterplots/`.

AR-WGAN, ICA, and the Wiener filter all use the same held-out test split, the
same noise mixing, and the same metric and denoiser code, so the comparison is
fair.

## Layout

- `train.py`: the training loop.
- `model.py`: the generator (U-Net) and the critic.
- `eeg_data_generator.py`: data loading, the noisy/clean dataset, the train/test
  split, and the shared -6 dB sample.
- `variables.py`: configuration.
- `utils.py`: the gradient penalty and checkpoint helpers.
- `metrics.py`: the metric functions (CC, RRMSE, band-power ratios), used by
  every evaluator.
- `baselines.py`: the ICA and Wiener denoisers, used by the comparison scripts
  and the poster generator.
- `plots.py`: the shared PSD comparison plot.
- `evaluate.py`: evaluation of the trained model.
- `comparisons/`: the ICA and Wiener-filter evaluations.
- `graphs.py`: combines the per-method results into `finalplots/`.
- `postergen.py`: the poster figures.
