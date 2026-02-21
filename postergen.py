import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.signal import welch, wiener
from scipy.interpolate import griddata
from matplotlib.patches import Patch

def _load_first_epoch_npy(path):
	if not os.path.exists(path):
		raise FileNotFoundError(path)
	arr = np.load(path, allow_pickle=False)
	arr = np.asarray(arr, dtype=float)
	if arr.ndim == 0:
		raise ValueError(f"Unexpected scalar in {path}")
	if arr.ndim == 1:
		return arr.ravel()
	# prefer first epoch/row
	return arr.reshape(arr.shape[0], -1)[0, :].ravel()

def _add_eeg_band_shading_and_legend(ax, xlim=(0, 40)):
	"""Shade canonical EEG bands and add legend key (style-only)."""
	bands = [
		("Delta", (0.5, 4.0), "#f1c40f"),
		("Theta", (4.0, 8.0), "#f39c12"),
		("Alpha", (8.0, 13.0), "#2ecc71"),
		("Beta",  (13.0, 30.0), "#5dade2"),
		("Gamma", (30.0, 80.0), "#d7bde2"),
	]
	xmin, xmax = xlim
	handles = []
	for name, (lo, hi), color in bands:
		lo_c = max(lo, xmin)
		hi_c = min(hi, xmax)
		if hi_c <= lo_c:
			continue
		ax.axvspan(lo_c, hi_c, color=color, alpha=0.16, zorder=0)
		handles.append(Patch(facecolor=color, edgecolor="none", alpha=0.35, label=name))
	leg = ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95)
	leg.get_frame().set_edgecolor("#dddddd")

def plot_cleaned_eeg_with_psd_from_dataset():
	"""
	Save posterplots/cleaned_eeg_psd.png : cleaned EEG (top) + PSD (bottom).
	"""
	base_dir = os.path.dirname(os.path.abspath(__file__))
	dataset_dir = os.path.join(base_dir, "dataset")
	poster_dir = os.path.join(base_dir, "posterplots")
	os.makedirs(poster_dir, exist_ok=True)

	eeg_path = os.path.join(dataset_dir, "EEG_all_epochs.npy")
	eeg = _load_first_epoch_npy(eeg_path)

	fs = 250  # default sampling rate
	n = min(len(eeg), fs * 4)
	eeg = eeg[:n] - np.mean(eeg[:n])
	t = np.arange(n) / fs

	nperseg = min(len(eeg), fs * 2)
	freqs, psd = welch(eeg, fs=fs, nperseg=nperseg)

	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [1, 1]})
	fig.patch.set_facecolor("white")

	eeg_color = "#2c3e50"
	psd_line = "#2980b9"
	psd_fill = "#3498db"

	ax1.plot(t, eeg, color=eeg_color, linewidth=1.4)
	ax1.set_title("Cleaned EEG Signal (Pre-processed)", fontsize=12, fontweight="bold", loc="left")
	ax1.set_ylabel("Amplitude (µV)")
	ax1.set_xlim(t[0], t[-1] if t.size else 0)
	ax1.grid(True, linestyle="--", alpha=0.4)
	ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

	ax2.fill_between(freqs, psd, color=psd_fill, alpha=0.32)
	ax2.plot(freqs, psd, color=psd_line, linewidth=2)
	ax2.set_title("Power Spectral Density", fontsize=12, fontweight="bold", loc="left")
	ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel(r"Power ($V^2/Hz$)")
	ax2.set_xlim(0, 40)
	_add_eeg_band_shading_and_legend(ax2, xlim=(0, 40))
	ax2.grid(True, linestyle="--", alpha=0.4)
	ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

	plt.tight_layout()
	out_path = os.path.join(poster_dir, "cleaned_eeg_psd.png")
	plt.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved {out_path}")

def plot_eog_emg_pair_from_dataset():
	"""
	Save posterplots/eog_emg_pair.png : EOG (top) and EMG (bottom) with height ratio 10:3.
	"""
	base_dir = os.path.dirname(os.path.abspath(__file__))
	dataset_dir = os.path.join(base_dir, "dataset")
	poster_dir = os.path.join(base_dir, "posterplots")
	os.makedirs(poster_dir, exist_ok=True)

	eog_path = os.path.join(dataset_dir, "EOG_all_epochs.npy")
	emg_path = os.path.join(dataset_dir, "EMG_all_epochs.npy")

	eog = _load_first_epoch_npy(eog_path)
	emg = _load_first_epoch_npy(emg_path)

	fs = 250
	n_eog = min(len(eog), fs * 4)
	n_emg = min(len(emg), fs * 4)
	t_eog = np.arange(n_eog) / fs
	t_emg = np.arange(n_emg) / fs
	eog = eog[:n_eog] - np.mean(eog[:n_eog])
	emg = emg[:n_emg] - np.mean(emg[:n_emg])

	# EOG much larger than EMG: 10:3 ratio
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 6), gridspec_kw={"height_ratios": [10, 3]})
	fig.patch.set_facecolor("white")

	ax1.plot(t_eog, eog, color="#c0392b", linewidth=1.8)
	ax1.set_title("EOG Artifact (Representative)", fontsize=12, fontweight="bold", loc="left")
	ax1.set_ylabel("Amplitude (µV)")
	ax1.set_xlim(t_eog[0], t_eog[-1] if t_eog.size else 0)
	ax1.grid(True, linestyle="--", alpha=0.4)
	ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

	ax2.plot(t_emg, emg, color="#8e44ad", linewidth=1.8)
	ax2.set_title("EMG Artifact (Representative)", fontsize=12, fontweight="bold", loc="left")
	ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Amplitude (µV)")
	ax2.set_xlim(t_emg[0], t_emg[-1] if t_emg.size else 0)
	ax2.grid(True, linestyle="--", alpha=0.4)
	ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

	plt.tight_layout()
	out_path = os.path.join(poster_dir, "eog_emg_pair.png")
	plt.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved {out_path}")

def plot_wiener_example_from_dataset():
	"""
	Save posterplots/wiener_example.png : noisy (red dashed), Wiener denoised (green front), clean (blue).
	"""
	base_dir = os.path.dirname(os.path.abspath(__file__))
	dataset_dir = os.path.join(base_dir, "dataset")
	poster_dir = os.path.join(base_dir, "posterplots")
	os.makedirs(poster_dir, exist_ok=True)

	eeg_path = os.path.join(dataset_dir, "EEG_all_epochs.npy")
	artifact_path = os.path.join(dataset_dir, "EOG_all_epochs.npy")
	if not os.path.exists(artifact_path):
		artifact_path = os.path.join(dataset_dir, "EMG_all_epochs.npy")

	clean = _load_first_epoch_npy(eeg_path)
	art = _load_first_epoch_npy(artifact_path) if os.path.exists(artifact_path) else None

	# trim/pad artifact to EEG length
	L = len(clean)
	if art is not None:
		if len(art) >= L:
			art = art[:L]
		else:
			art = np.pad(art, (0, L - len(art)), mode="wrap")
		art_scale = 0.5 * (np.ptp(clean) / (np.ptp(art) + 1e-12))
		artifact = art * art_scale
	else:
		rng = np.random.default_rng(0)
		artifact = 0.2 * (rng.normal(size=clean.shape) * np.std(clean))

	noisy = clean + artifact

	# stronger smoothing for Wiener: use larger window (in samples)
	fs = 250
	win_samples = max(3, int(0.35 * fs))
	denoised = wiener(noisy, mysize=win_samples)

	# Single-panel figure sized 5x4 (width x height)
	fig, ax = plt.subplots(1, 1, figsize=(5, 4))
	t = np.arange(L) / fs
	# thicker lines; put denoised (green) in front with higher zorder
	ax.plot(t, noisy, linestyle="--", color="red", linewidth=1.8, label="Noisy", zorder=2)
	ax.plot(t, denoised, color="green", linewidth=2.6, label="Denoised", zorder=4)
	ax.plot(t, clean, color="blue", linewidth=1.8, label="Clean", zorder=1)
	ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude (µV)")
	ax.set_title("Wiener Filter")
	ax.legend(loc="upper right", frameon=True)
	ax.grid(True, linestyle="--", alpha=0.4)
	ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
	plt.tight_layout()
	out_path = os.path.join(poster_dir, "wiener_example.png")
	plt.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved {out_path}")

def plot_wiener_residual_from_dataset():
	"""
	Save posterplots/wiener_residual.png : residual (noisy - denoised) shown in a compact 3:1 figure.
	"""
	base_dir = os.path.dirname(os.path.abspath(__file__))
	dataset_dir = os.path.join(base_dir, "dataset")
	poster_dir = os.path.join(base_dir, "posterplots")
	os.makedirs(poster_dir, exist_ok=True)

	# reuse same sources & logic as the main wiener example
	eeg_path = os.path.join(dataset_dir, "EEG_all_epochs.npy")
	artifact_path = os.path.join(dataset_dir, "EOG_all_epochs.npy")
	if not os.path.exists(artifact_path):
		artifact_path = os.path.join(dataset_dir, "EMG_all_epochs.npy")

	clean = _load_first_epoch_npy(eeg_path)
	art = _load_first_epoch_npy(artifact_path) if os.path.exists(artifact_path) else None

	L = len(clean)
	if art is not None:
		if len(art) >= L:
			art = art[:L]
		else:
			art = np.pad(art, (0, L - len(art)), mode="wrap")
		art_scale = 0.5 * (np.ptp(clean) / (np.ptp(art) + 1e-12))
		artifact = art * art_scale
	else:
		rng = np.random.default_rng(0)
		artifact = 0.2 * (rng.normal(size=clean.shape) * np.std(clean))

	noisy = clean + artifact

	# same Wiener smoothing as example
	fs = 250
	win_samples = max(3, int(0.35 * fs))
	denoised = wiener(noisy, mysize=win_samples)

	residual = noisy - denoised

	# compact 3:1 figure (width x height)
	fig, ax = plt.subplots(1, 1, figsize=(6, 2))
	t = np.arange(L) / fs
	ax.plot(t, residual, color="#8e44ad", linewidth=1.6)
	ax.set_xlabel("Time (s)")
	ax.set_ylabel("Residual")
	ax.set_title("Wiener Residual")
	ax.grid(True, linestyle="--", alpha=0.4)
	ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
	plt.tight_layout()
	out_path = os.path.join(poster_dir, "wiener_residual.png")
	plt.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved {out_path}")

def plot_stochastic_manifold_from_dataset():
	"""
	Generate a '3D stochastic manifold map' by plotting signal trajectories in phase space.
	Uses EEG vs EOG vs EMG if available, otherwise uses Time-Delay Embedding of EEG.
	Saves to posterplots/stochastic_manifold.png with 1:1 aspect ratio.
	"""
	base_dir = os.path.dirname(os.path.abspath(__file__))
	dataset_dir = os.path.join(base_dir, "dataset")
	poster_dir = os.path.join(base_dir, "posterplots")
	os.makedirs(poster_dir, exist_ok=True)

	eeg_path = os.path.join(dataset_dir, "EEG_all_epochs.npy")
	eog_path = os.path.join(dataset_dir, "EOG_all_epochs.npy")
	emg_path = os.path.join(dataset_dir, "EMG_all_epochs.npy")

	try:
		x = _load_first_epoch_npy(eeg_path)
	except FileNotFoundError:
		print("No EEG data found for manifold map.")
		return

	# Try to load others to form 3 axes
	y = None
	z = None
	if os.path.exists(eog_path):
		y = _load_first_epoch_npy(eog_path)
	if os.path.exists(emg_path):
		z = _load_first_epoch_npy(emg_path)

	# If missing dimensions, use Time Delay Embedding on EEG
	tau = 4
	if y is None or z is None:
		if len(x) < 3 * tau:
			return
		if y is None:
			y = np.roll(x, -tau)
		if z is None:
			z = np.roll(x, -2 * tau)

	# Trim to common length
	n = min(len(x), len(y), len(z), 250 * 20)
	x = x[:n]; y = y[:n]; z = z[:n]

	# Normalize
	def normalize(arr):
		return (arr - np.mean(arr)) / (np.std(arr) + 1e-9)
	x = normalize(x); y = normalize(y); z = normalize(z)

	# Subsample for triangulation/detail control (keep shape but limit points)
	max_pts = 3000
	step = max(1, n // max_pts)
	idx = np.arange(0, n, step)
	xs = x[idx]; ys = y[idx]; zs = z[idx]
	t_indices = np.linspace(0, 1, xs.size)

	# Create 3D figure (1:1)
	fig = plt.figure(figsize=(6, 6))
	ax = fig.add_subplot(111, projection='3d')
	fig.patch.set_facecolor("white")

	# Build a triangulation in the projected (x,y) plane and plot a trisurf as a manifold sheet.
	tri = mtri.Triangulation(xs, ys)
	ax.plot_trisurf(xs, ys, zs, triangles=tri.triangles,
	                cmap='plasma', linewidth=0.2, antialiased=True, alpha=0.75, shade=True)

	# Overlay larger points for visibility, colored by time
	ax.scatter(xs, ys, zs, c=t_indices, cmap='plasma', s=30, alpha=0.95, edgecolors='k', linewidths=0.15, depthshade=True)

	# Zoom: use percentile window to crop to main cloud
	pad = 0.06
	xlo, xhi = np.percentile(xs, [2 + pad, 98 - pad])
	ylo, yhi = np.percentile(ys, [2 + pad, 98 - pad])
	zlo, zhi = np.percentile(zs, [2 + pad, 98 - pad])
	ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_zlim(zlo, zhi)

	# Style similar to other figures
	ax.set_axis_off()
	ax.set_title("Stochastic Manifold Map", fontsize=14, fontweight='bold', color="#333333")
	ax.view_init(elev=30, azim=45)

	plt.tight_layout()
	out_path = os.path.join(poster_dir, "stochastic_manifold.png")
	plt.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved {out_path}")

if __name__ == "__main__":
	plot_cleaned_eeg_with_psd_from_dataset()
	plot_eog_emg_pair_from_dataset()
	plot_wiener_example_from_dataset()
	plot_wiener_residual_from_dataset()
	plot_stochastic_manifold_from_dataset()
