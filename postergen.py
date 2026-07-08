import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.interpolate import griddata
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import torch

# Add project root to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
	from variables import (
		EEG_FILE, EOG_FILE, EMG_FILE, CHANNELS_EEG, FEATURES_GEN, SAVED_MODEL_PATH,
		SAMPLING_RATE
	)
	from model import Generator
	from eeg_data_generator import prepare_eeg_data
	from baselines import ica_denoise, wiener_denoise
except ImportError:
	print("Warning: Project modules not found. Comparison functions may fail.")

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
	denoised = wiener_denoise(noisy, mysize=win_samples)

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
	denoised = wiener_denoise(noisy, mysize=win_samples)

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
	Plot a large 3D point cloud (no surface) using EEG/EOG/EMG or time-delay embedding.
	Add extra jittered points for density and render a colored grid floor where each
	block color encodes local point density. Saves to posterplots/stochastic_manifold.png.
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

	# Normalize for nicer plotting
	def normalize(arr):
		return (arr - np.mean(arr)) / (np.std(arr) + 1e-9)
	x = normalize(x); y = normalize(y); z = normalize(z)

	# Subsample for plotting density control
	max_pts = 5000
	step = max(1, n // max_pts)
	idx = np.arange(0, n, step)
	X = x[idx]; Y = y[idx]; Z = z[idx]
	t_indices = np.linspace(0, 1, X.size)

	# Add more points by jittering a random subset to increase visual density
	extra_frac = 0.6  # fraction of points to generate as jittered copies
	n_extra = int(len(X) * extra_frac)
	if n_extra > 0:
		rng = np.random.default_rng(1)
		sel = rng.integers(0, len(X), size=n_extra)
		jitter_scale = 0.02 * np.ptp(np.vstack([X, Y, Z]), axis=1).max()
		X_extra = X[sel] + rng.normal(scale=jitter_scale, size=n_extra)
		Y_extra = Y[sel] + rng.normal(scale=jitter_scale, size=n_extra)
		Z_extra = Z[sel] + rng.normal(scale=jitter_scale, size=n_extra)
		# concatenate
		X = np.concatenate([X, X_extra])
		Y = np.concatenate([Y, Y_extra])
		Z = np.concatenate([Z, Z_extra])
		t_extra = rng.random(n_extra)
		t_indices = np.concatenate([t_indices, t_extra])

	# Create 3D scatter (1:1)
	fig = plt.figure(figsize=(6, 6))
	ax = fig.add_subplot(111, projection='3d')
	fig.patch.set_facecolor("white")

	# Build a 2D grid on the XY projection and compute counts per cell
	grid_res = 48
	xmin, xmax = np.percentile(X, [1, 99])
	ymin, ymax = np.percentile(Y, [1, 99])
	x_edges = np.linspace(xmin, xmax, grid_res + 1)
	y_edges = np.linspace(ymin, ymax, grid_res + 1)
	H, xe, ye = np.histogram2d(X, Y, bins=[x_edges, y_edges])
	# normalize counts for color mapping
	norm = Normalize(vmin=0, vmax=H.max() if H.max() > 0 else 1)
	cmap = cm.get_cmap('Blues')

	# Create mesh centers and Z plane slightly below the cloud
	x_centers = 0.5 * (xe[:-1] + xe[1:])
	y_centers = 0.5 * (ye[:-1] + ye[1:])
	Xg, Yg = np.meshgrid(x_centers, y_centers)
	# place grid at a z slightly below the min Z to avoid overlap
	zmin = np.min(Z)
	zrange = np.ptp(Z) if np.ptp(Z) > 0 else 1.0
	Zplane = np.full_like(Xg, zmin - 0.04 * zrange)

	# Map counts -> facecolors
	facecolors = cmap(norm(H.T))  # transpose so orientation matches mesh grid

	# Plot colored grid blocks as a flat surface with facecolors
	ax.plot_surface(Xg, Yg, Zplane, rstride=1, cstride=1, facecolors=facecolors,
	                shade=False, linewidth=0, antialiased=True, alpha=0.9)

	# Large blue-ish points colored by time
	sc = ax.scatter(X, Y, Z, c=t_indices, cmap='Blues', s=28, alpha=0.95, edgecolors='k', linewidths=0.08)

	# Axes, labels and grid for meaning
	ax.set_xlabel("Dim 1")
	ax.set_ylabel("Dim 2")
	ax.set_zlabel("Dim 3")
	ax.grid(True, linestyle="--", alpha=0.25)
	ax.set_title("Stochastic Manifold Point Cloud", fontsize=14, fontweight='bold', color="#222f4a")

	# Try to set equal aspect for 3D if available
	try:
		ax.set_box_aspect((1, 1, 1))
	except Exception:
		pass

	# Adjust view
	ax.view_init(elev=30, azim=45)

	plt.tight_layout()
	out_path = os.path.join(poster_dir, "stochastic_manifold.png")
	plt.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved {out_path}")

def plot_js_wasserstein_surfaces():
	"""
	Generate side-by-side 3D topographical loss surfaces:
	- Left: Jensen-Shannon-like (bumpy, hole hard to find)
	- Right: Wasserstein-like (smoother funnel with clearer hole)
	This version increases jaggedness by combining low-frequency and higher-resolution
	random fields and applying a non-linear accentuation so peaks/valleys are less rounded.
	Save to posterplots/js_wasserstein_compare.png.
	"""
	base_dir = os.path.dirname(os.path.abspath(__file__))
	poster_dir = os.path.join(base_dir, "posterplots")
	os.makedirs(poster_dir, exist_ok=True)

	# create grid
	n = 200
	lim = 1.6
	x = np.linspace(-lim, lim, n)
	y = np.linspace(-lim, lim, n)
	X, Y = np.meshgrid(x, y)
	R = np.sqrt(X**2 + Y**2)

	# base radial shape made less funnel-like (lower exponent, small anisotropy)
	base = 0.9 * (R**1.05) + 0.15 * (X * 0.5)  # slight tilt

	# central holes (depths)
	hole_small = -1.8 * np.exp(-(R / 0.14)**2)
	hole_large = -2.6 * np.exp(-(R / 0.2)**2)

	# generate two low-frequency random fields via coarse grids + interpolation
	rng = np.random.default_rng(42)
	coarse = 16  # coarse grid resolution controls low freq
	xc = np.linspace(-lim, lim, coarse)
	yc = np.linspace(-lim, lim, coarse)
	(Xc, Yc) = np.meshgrid(xc, yc)
	random_coarse_js = rng.normal(scale=1.0, size=Xc.shape)
	random_coarse_w = rng.normal(scale=1.0, size=Xc.shape)
	pts_coarse = np.column_stack((Xc.ravel(), Yc.ravel()))
	js_field = griddata(pts_coarse, random_coarse_js.ravel(), (X, Y), method="cubic")
	w_field = griddata(pts_coarse, random_coarse_w.ravel(), (X, Y), method="cubic")
	# fallback linear where cubic produced NaNs
	if np.isnan(js_field).any():
		js_field_lin = griddata(pts_coarse, random_coarse_js.ravel(), (X, Y), method="linear")
		js_field[np.isnan(js_field)] = js_field_lin[np.isnan(js_field)]
	if np.isnan(w_field).any():
		w_field_lin = griddata(pts_coarse, random_coarse_w.ravel(), (X, Y), method="linear")
		w_field[np.isnan(w_field)] = w_field_lin[np.isnan(w_field)]

	# add a higher-resolution random field to create sharper jaggedness
	coarse_high = 40
	xch = np.linspace(-lim, lim, coarse_high)
	ych = np.linspace(-lim, lim, coarse_high)
	(Xch, Ych) = np.meshgrid(xch, ych)
	random_coarse_js_high = rng.normal(scale=0.8, size=Xch.shape)
	random_coarse_w_high = rng.normal(scale=0.6, size=Xch.shape)
	pts_high = np.column_stack((Xch.ravel(), Ych.ravel()))
	js_field_high = griddata(pts_high, random_coarse_js_high.ravel(), (X, Y), method="cubic")
	w_field_high = griddata(pts_high, random_coarse_w_high.ravel(), (X, Y), method="cubic")
	# fallback linear
	if np.isnan(js_field_high).any():
		js_field_high_lin = griddata(pts_high, random_coarse_js_high.ravel(), (X, Y), method="linear")
		js_field_high[np.isnan(js_field_high)] = js_field_high_lin[np.isnan(js_field_high)]
	if np.isnan(w_field_high).any():
		w_field_high_lin = griddata(pts_high, random_coarse_w_high.ravel(), (X, Y), method="linear")
		w_field_high[np.isnan(w_field_high)] = w_field_high_lin[np.isnan(w_field_high)]

	# combine fields and apply non-linear accentuation to reduce roundness and boost jaggedness
	js_combined = 1.2 * js_field + 0.95 * js_field_high
	w_combined = 0.7 * w_field + 0.5 * w_field_high

	# non-linear sharpening: raise absolute values to a power >1, keep sign
	def sharpen(arr, power=1.25, scale=1.0):
		return scale * np.sign(arr) * (np.abs(arr) ** power)

	js_noise = sharpen(js_combined, power=1.22, scale=1.0) + 0.08 * rng.normal(size=X.shape)
	w_noise = sharpen(w_combined, power=1.10, scale=0.8) + 0.06 * rng.normal(size=X.shape)

	# reduce tapering so jaggedness persists toward center (less rounded)
	taper = np.exp(- (R / (lim * 0.9))**2)
	js_noise *= (0.45 + 0.55 * (1 - taper))
	w_noise *= (0.4 + 0.6 * (1 - taper))

	# Build surfaces: less funnel-like base, larger jagged contribution for JS
	JS = base + 1.4 * js_noise + 0.55 * hole_small
	W = base + 1.0 * w_noise + 1.0 * hole_large

	# Clip for display
	vmin, vmax = -6.0, 8.0
	JS = np.clip(JS, vmin, vmax)
	W = np.clip(W, vmin, vmax)

	# Plotting with colormap that shows detail
	fig = plt.figure(figsize=(14, 5.5), constrained_layout=True)
	cmap = plt.get_cmap("viridis")

	# Left: Jensen-Shannon (more jagged)
	ax1 = fig.add_subplot(1, 2, 1, projection="3d")
	surf1 = ax1.plot_surface(X, Y, JS, cmap=cmap, linewidth=0, antialiased=True, rcount=160, ccount=160, alpha=0.95)
	ax1.set_title("Jensen–Shannon Loss Surface", fontsize=12, fontweight="bold")
	ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("Loss")
	ax1.view_init(elev=35, azim=-55)
	ax1.dist = 10
	cb1 = fig.colorbar(surf1, ax=ax1, shrink=0.6, pad=0.02)
	cb1.ax.set_title("Loss", fontsize=9)

	# subtle marker at center (hole location)
	ax1.scatter([0], [0], [JS.min()], color="navy", s=12, alpha=0.8)

	# Right: Wasserstein (still smoother but more jagged than before)
	ax2 = fig.add_subplot(1, 2, 2, projection="3d")
	surf2 = ax2.plot_surface(X, Y, W, cmap=cmap, linewidth=0, antialiased=True, rcount=160, ccount=160, alpha=0.95)
	ax2.set_title("Wasserstein-like Surface", fontsize=12, fontweight="bold")
	ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("Loss")
	ax2.view_init(elev=35, azim=-55)
	ax2.dist = 10
	cb2 = fig.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.02)
	cb2.ax.set_title("Loss", fontsize=9)

	# add gentle contours on Wasserstein to show structure
	ax2.contour(X, Y, W, zdir='z', offset=W.min() - 0.6, cmap=cmap, linewidths=0.6)

	# visual tweaks
	for ax in (ax1, ax2):
		ax.set_box_aspect((1, 1, 0.45))
		ax.xaxis.pane.fill = False
		ax.yaxis.pane.fill = False
		ax.zaxis.pane.fill = False
		ax.grid(False)

	out_path = os.path.join(poster_dir, "js_wasserstein_compare.png")
	plt.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close(fig)
	print(f"Saved {out_path}")

def plot_weight_clipping_vs_gp():
	"""
	Illustrate weight clipping vs gradient-penalty (Lipschitz) discriminator responses.
	X axis: fake data position 0..10. Y axis: discriminator score 0..1.
	Weight clipping: diagonal up to x=6, then saturates at 1.
	GP+Lipschitz: smooth monotonic transition from 0->1 across 0..10.
	"""
	base_dir = os.path.dirname(os.path.abspath(__file__))
	poster_dir = os.path.join(base_dir, "posterplots")
	os.makedirs(poster_dir, exist_ok=True)

	# x domain
	x = np.linspace(0, 10, 400)

	# Weight clipping: diagonal up to x=6, then saturate at 1
	y_wc = np.minimum(x / 6.0, 1.0)

	# Gradient penalty + Lipschitz: smooth sigmoidal/tanh mapping between 0 and 1
	# Use a tanh centered at 5 with width chosen to span 0..10 smoothly
	y_gp = 0.5 * (1.0 + np.tanh((x - 5.0) / 2.2))

	# Plot
	fig, ax = plt.subplots(1, 1, figsize=(9, 3))
	fig.patch.set_facecolor("white")

	ax.plot(x, y_wc, linestyle="--", color="#c0392b", linewidth=2.2, label="Weight clipping")
	ax.plot(x, y_gp, linestyle="-", color="#2980b9", linewidth=2.4, label="Gradient penalty + Lipschitz")

	# Visual cues: mark the saturation point for weight clipping
	ax.axvline(6.0, color="#7f8c8d", linewidth=0.9, linestyle=":")
	ax.text(6.05, 0.03, "clipping threshold", color="#7f8c8d", fontsize=8, va="bottom")

	ax.set_xlim(0, 10)
	ax.set_ylim(-0.02, 1.02)
	ax.set_xlabel("Data position")
	ax.set_ylabel("Discriminator score")
	ax.set_title("Weight Clipping vs Gradient Penalty + Lipschitz")
	ax.grid(True, linestyle="--", alpha=0.28)
	ax.legend(loc="upper left", frameon=True, fontsize="small")

	plt.tight_layout()
	out_path = os.path.join(poster_dir, "wc_vs_gp.png")
	plt.savefig(out_path, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved {out_path}")

# --- Denoising Helpers ---

def get_denoised_ica(noisy_signal, n_components=3):
	"""ICA denoising via the shared baseline (identical to comparisons/ica.py)."""
	try:
		return ica_denoise(noisy_signal, n_components=n_components)
	except Exception:
		return noisy_signal.flatten()

def get_denoised_wiener(noisy_signal, mysize=31):
	"""Wiener denoising via the shared baseline."""
	return wiener_denoise(noisy_signal, mysize=mysize)

def load_arwgan_generator():
	"""Load the trained generator model."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	# Initialize model structure (assuming 512 samples or dynamic)
	# Passing dummy samples size, model usually adapts or we resize input
	gen = Generator(CHANNELS_EEG, 512, FEATURES_GEN).to(device)
	if os.path.exists(SAVED_MODEL_PATH):
		try:
			checkpoint = torch.load(SAVED_MODEL_PATH, map_location=device)
			gen.load_state_dict(checkpoint['gen'])
			gen.eval()
			return gen, device
		except Exception as e:
			print(f"Error loading model: {e}")
			return None, device
	print(f"Warning: Model not found at {SAVED_MODEL_PATH}")
	return None, device

def get_denoised_arwgan(noisy_signal, generator_model, device):
	"""Apply AR-WGAN generator."""
	# Reshape for model: (batch, channels, time) -> (1, 1, 512)
	# Ensure input is 512 length
	target_len = 512
	if len(noisy_signal) != target_len:
		# Simple resize or crop for inference demo
		sig = np.resize(noisy_signal, target_len)
	else:
		sig = noisy_signal
		
	inp = torch.from_numpy(sig).float().unsqueeze(0).unsqueeze(0).to(device)
	with torch.no_grad():
		out = generator_model(inp)
	return out.cpu().numpy().flatten()

def prepare_comparison_data(target_snr_db=0):
	"""Load a clean epoch and create a noisy version at target SNR."""
	# Load or use default paths from imports
	clean_all, eog_noise, emg_noise = prepare_eeg_data(EEG_FILE, EOG_FILE, EMG_FILE, [-100, -100])
	
	# Pick a specific sample index that looks "interesting"
	idx = 25  
	if idx >= len(clean_all): idx = 0
	clean = clean_all[idx].flatten()
	
	# Create mixed noise
	eog = eog_noise[idx % len(eog_noise)] if eog_noise is not None else np.zeros_like(clean)
	emg = emg_noise[idx % len(emg_noise)] if emg_noise is not None else np.zeros_like(clean)
	noise = eog + emg
	
	# Adjust SNR
	clean_p = np.mean(clean**2)
	noise_p = np.mean(noise**2)
	if noise_p == 0: return clean[:512], clean[:512]
	
	snr_linear = 10**(target_snr_db / 10)
	alpha = np.sqrt(clean_p / (snr_linear * noise_p))
	noisy = clean + alpha * noise
	
	# Crop to 512 samples for consistency
	return clean[:512], noisy[:512]

# --- New Comparison Plotting Functions ---

def _method_psd(sig):
	"""Welch PSD (single-sided) used by the poster comparison figures."""
	return welch(sig, fs=SAMPLING_RATE, nperseg=min(len(sig), 256))

def _denoise_all_methods(noisy, clean, gen, device):
	"""Denoise one noisy epoch with ICA, Wiener, and AR-WGAN and return the
	three denoised signals. Shared by the poster comparison figures."""
	den_ica = get_denoised_ica(noisy)
	den_wiener = get_denoised_wiener(noisy)
	if gen is not None:
		den_arwgan = get_denoised_arwgan(noisy, gen, device)
		if len(den_arwgan) != len(clean):
			den_arwgan = np.resize(den_arwgan, len(clean))
	else:
		den_arwgan = np.zeros_like(clean)
	return den_ica, den_wiener, den_arwgan

def plot_all_methods_comparison_0db():
	"""
	Generate time-series comparison of ICA, Wiener, and AR-WGAN at 0 dB SNR.
	Saves to posterplots/comparison_0db_time.png.
	"""
	base_dir = os.path.dirname(os.path.abspath(__file__))
	poster_dir = os.path.join(base_dir, "posterplots")
	os.makedirs(poster_dir, exist_ok=True)
	
	clean, noisy = prepare_comparison_data(target_snr_db=0)
	
	# Denoise with all methods (shared helper)
	gen, device = load_arwgan_generator()
	den_ica, den_wiener, den_arwgan = _denoise_all_methods(noisy, clean, gen, device)

	# 2. Plot stacked
	fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
	
	methods = [
		("ICA", den_ica, "#2ecc71"), 
		("Wiener Filter", den_wiener, "#e67e22"), 
		("AR-WGAN", den_arwgan, "#9b59b6")
	]
	
	t = np.arange(len(clean))
	
	# Time-series fonts: 1.5x baseline
	TITLE_FS = 21  # 14 * 1.5
	LABEL_FS = 18  # 12 * 1.5
	for ax, (name, sig, color) in zip(axes, methods):
		ax.plot(t, clean, color="#34495e", linewidth=1.5, label="Clean", alpha=0.9, zorder=1)
		ax.plot(t, noisy, color="#e74c3c", linewidth=1.0, linestyle="--", label="Noisy (0 dB)", alpha=0.6, zorder=0)
		ax.plot(t, sig, color=color, linewidth=2.0, label=f"{name} Denoised", zorder=2)
		ax.set_title(f"{name}", fontsize=TITLE_FS, fontweight='bold', loc='left')
		ax.set_ylabel("Amplitude", fontsize=LABEL_FS)
		ax.legend(loc="upper right", framealpha=0.95, fontsize=12)
		ax.grid(True, linestyle=":", alpha=0.5)

	axes[-1].set_xlabel("Sample Index", fontsize=LABEL_FS)
	for ax in axes:
		ax.tick_params(axis='both', which='major', labelsize=int(LABEL_FS * 0.6))
	
	out_path = os.path.join(poster_dir, "comparison_0db_time.png")
	plt.savefig(out_path, dpi=300)
	plt.close()
	print(f"Saved {out_path}")

def plot_psd_comparison_all_methods():
	"""
	Generate PSD comparison of ICA, Wiener, AR-WGAN vs Clean/Noisy.
	Includes shaded frequency bands.
	Saves to posterplots/comparison_psd_all.png.
	"""
	base_dir = os.path.dirname(os.path.abspath(__file__))
	poster_dir = os.path.join(base_dir, "posterplots")
	os.makedirs(poster_dir, exist_ok=True)
	
	clean, noisy = prepare_comparison_data(target_snr_db=0)
	
	# Denoise with all methods (shared helper)
	gen, device = load_arwgan_generator()
	den_ica, den_wiener, den_arwgan = _denoise_all_methods(noisy, clean, gen, device)

	# Calculate PSDs (shared helper)
	f, psd_clean = _method_psd(clean)
	_, psd_noisy = _method_psd(noisy)
	_, psd_ica = _method_psd(den_ica)
	_, psd_wiener = _method_psd(den_wiener)
	_, psd_arwgan = _method_psd(den_arwgan)

	# Plot
	fig, ax = plt.subplots(figsize=(10, 6))
	fig.patch.set_facecolor("white")
	
	ax.plot(f, psd_clean, color="black", linewidth=2.0, label="Clean", alpha=0.8)
	ax.plot(f, psd_noisy, color="red", linewidth=1.5, linestyle="--", label="Noisy (0 dB)", alpha=0.6)
	ax.plot(f, psd_ica, color="#2ecc71", linewidth=2.0, label="ICA")
	ax.plot(f, psd_wiener, color="#e67e22", linewidth=2.0, label="Wiener")
	ax.plot(f, psd_arwgan, color="#9b59b6", linewidth=2.0, label="AR-WGAN")
	
	ax.set_title("PSD Comparison (0 dB Input SNR)", fontsize=14, fontweight="bold")
	ax.set_xlabel("Frequency (Hz)")
	ax.set_ylabel(r"Power Spectral Density ($V^2/Hz$)")
	ax.set_xlim(0, 50)  # Focus on EEG range
	
	# Add bands background
	_add_eeg_band_shading_and_legend(ax, xlim=(0, 50))
	
	# Custom legend for lines (separate from bands)
	lines_legend = ax.legend(loc="upper right", frameon=True)
	ax.add_artist(lines_legend)
	
	ax.grid(True, linestyle="--", alpha=0.4)
	
	plt.tight_layout()
	out_path = os.path.join(poster_dir, "comparison_psd_all.png")
	plt.savefig(out_path, dpi=300)
	plt.close()
	print(f"Saved {out_path}")

def plot_methods_for_snrs(snrs):
	"""
	For each SNR in snrs, create:
	  - posterplots/comparison_{snr}db_time.png  (stacked time-series: ICA / Wiener / AR-WGAN)
	  - posterplots/comparison_{snr}db_psd.png   (PSD comparison with EEG-band shading)
	"""
	base_dir = os.path.dirname(os.path.abspath(__file__))
	poster_dir = os.path.join(base_dir, "posterplots")
	os.makedirs(poster_dir, exist_ok=True)

	# attempt to load generator once
	gen, device = load_arwgan_generator()

	# Font scaling: keep PSD font sizing as before, but use explicit time-series sizes = 1.5x baseline
	FONT_SCALE = 2.25
	base_title = 12
	base_label = 10
	base_tick = 9
	base_legend = 10
	# PSD font sizes (leave these for PSD plotting)
	title_fs_psd = int(round(base_title * FONT_SCALE))
	label_fs_psd = int(round(base_label * FONT_SCALE))
	tick_fs = int(round(base_tick * FONT_SCALE))
	legend_fs = int(round(base_legend * FONT_SCALE))

	# Time-series explicit sizes (1.5x baseline)
	title_fs_ts = int(round(14 * 2))   # baseline title was ~14 -> 21
	label_fs_ts = int(round(12 * 2))   # baseline label ~12 -> 18

	for snr in snrs:
		clean, noisy = prepare_comparison_data(target_snr_db=snr)

		# Denoise with all methods (shared helper)
		den_ica, den_wiener, den_arwgan = _denoise_all_methods(noisy, clean, gen, device)

		# --- Time series figure (3 stacked rows: ICA / Wiener / AR-WGAN) ---
		fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
		methods = [
			("ICA", den_ica, "#2ecc71"),
			("Wiener Filter", den_wiener, "#e67e22"),
			("AR-WGAN", den_arwgan, "#9b59b6"),
		]
		t = np.arange(len(clean))
		for ax, (name, sig, color) in zip(axes, methods):
			ax.plot(t, clean, color="#34495e", linewidth=1.4, label="Clean", zorder=2)
			ax.plot(t, noisy, color="#e74c3c", linewidth=1.0, linestyle="--", label=f"Noisy ({snr} dB)", alpha=0.6, zorder=0)
			ax.plot(t, sig, color=color, linewidth=2.0, label=f"{name} Denoised", zorder=3)
			# use time-series sizes
			ax.set_title(f"{name} — SNR {snr} dB", fontsize=title_fs_ts, loc="left")
			ax.set_ylabel("Amplitude", fontsize=label_fs_ts)
			ax.tick_params(axis='both', which='major', labelsize=tick_fs)
			ax.grid(True, linestyle=":", alpha=0.5)
			ax.legend(loc="upper right", framealpha=0.95, fontsize=legend_fs)
		axes[-1].set_xlabel("Sample Index", fontsize=label_fs_ts)
		axes[-1].tick_params(axis='x', labelsize=tick_fs)
		out_time = os.path.join(poster_dir, f"comparison_{snr}db_time.png")
		plt.savefig(out_time, dpi=300, bbox_inches="tight")
		plt.close(fig)
		print(f"Saved {out_time}")

		# --- PSD figure (shared PSD helper) ---
		fc, p_clean = _method_psd(clean)
		_, p_noisy = _method_psd(noisy)
		_, p_ica = _method_psd(den_ica)
		_, p_wiener = _method_psd(den_wiener)
		_, p_arwgan = _method_psd(den_arwgan)

		fig, ax = plt.subplots(figsize=(10, 6))
		ax.plot(fc, p_clean, color="black", linewidth=2.0, label="Clean")
		ax.plot(fc, p_noisy, color="red", linewidth=1.5, linestyle="--", label=f"Noisy ({snr} dB)")
		ax.plot(fc, p_ica, color="#2ecc71", linewidth=1.8, label="ICA")
		ax.plot(fc, p_wiener, color="#e67e22", linewidth=1.8, label="Wiener")
		ax.plot(fc, p_arwgan, color="#9b59b6", linewidth=1.8, label="AR-WGAN")

		# PSD fonts unchanged (use PSD font sizes)
		ax.set_title(f"PSD Comparison — SNR {snr} dB", fontsize=title_fs_psd, fontweight="bold")
		ax.set_xlabel("Frequency (Hz)", fontsize=label_fs_psd)
		ax.set_ylabel(r"Power Spectral Density ($V^2/Hz$)", fontsize=label_fs_psd)
		ax.set_xlim(0, 50)
		ax.tick_params(axis='both', which='major', labelsize=tick_fs)
		_add_eeg_band_shading_and_legend(ax, xlim=(0, 50))

		legend = ax.legend(loc="upper right", frameon=True, fontsize=legend_fs)
		ax.add_artist(legend)
		ax.grid(True, linestyle="--", alpha=0.4)

		out_psd = os.path.join(poster_dir, f"comparison_{snr}db_psd.png")
		plt.tight_layout()
		plt.savefig(out_psd, dpi=300, bbox_inches="tight")
		plt.close(fig)
		print(f"Saved {out_psd}")

if __name__ == "__main__":
	#plot_cleaned_eeg_with_psd_from_dataset()
	#plot_eog_emg_pair_from_dataset()
	#plot_wiener_example_from_dataset()
	#plot_wiener_residual_from_dataset()
	#plot_stochastic_manifold_from_dataset()
	#plot_js_wasserstein_surfaces()
	#plot_weight_clipping_vs_gp()
	plot_all_methods_comparison_0db()
	plot_psd_comparison_all_methods()
	# Generate per-SNR comparisons (time series + PSD) for requested SNRs (only -8 and -14 dB)
	plot_methods_for_snrs([-8, -14])
