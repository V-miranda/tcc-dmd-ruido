#!/usr/bin/env python3
"""
plot_turbo_modes.py
===================
Runs ONLY the custom BOPDMD (bopdmdTurbo) on one snapshot field of the
pitching-airfoil DNS dataset and plots

    1. the DMD modes (spatial structures) on the physical grid
    2. the DMD spectrum (three views, see plot_spectrum)

so they can be compared with the figures of the paper.

Steps:
    load field  ->  (optional noise)  ->  remove temporal mean of the training
    window  ->  (optional delay embedding)  ->  fit bopdmdTurbo  ->  plots

Outputs (folder OUTDIR):
    spectrum.png                  spectrum figure
    modes_real.png / modes_abs.png   mode maps (depending on PLOT_PARTS)
    modes_table.csv               frequency, growth rate, amplitude, importance
    turbo_decomposition.npz       omega, amplitudes, modes (complex64), grid, settings

How the mode maps are drawn
    * modes are complex and defined up to a complex constant, so for the "real"
      map each mode is rotated so that its point of largest magnitude is real and
      positive, then divided by its maximum -> colour scale is [-1, 1].
    * "abs" shows |phi| / max|phi| (scale [0, 1]).
    * only one mode of each complex-conjugate pair is shown (frequency >= 0);
      a mode with f ~ 0 is the (near-)stationary mode.

Usage:  edit the SETTINGS block, then   python plot_turbo_modes.py
Requirements: numpy, h5py, pandas, matplotlib, scipy/jax (for bopdmdTurbo), pydmd
"""

import importlib
import math
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# =============================================================================
# SETTINGS
# =============================================================================
FILE = "airfoilDNS/airfoilDNS_a30f0p05.h5"
GRID_FILE = "airfoilDNS/airfoilDNS_grid.h5"   # datasets "x" and "y"; None = use array indices
FIELD = "uy"                # "uy" (vertical velocity), "ux" or "vort"
TIME_NAME = "t_field"
F_PITCH = None              # pitching frequency; None = read it from the file name (f0p1 -> 0.1)

# Fit settings (defaults = reference case of the benchmark)
RANK = 20
N_SNAP = 400                # first N_SNAP snapshots are used (file has 401)
TEST_FRAC = 0.2             # last fraction held out for the forecast error; 0 = fit everything
NOISE = 0.0                 # fraction of the fluctuation std added as Gaussian noise
DELAY = 1                   # number of stacked time-shifted copies (1 = none)
SEED = 0

TURBO_MODULE = "bopdmdTurboV3"   # file bopdmdTurboV3.py next to this script
TURBO_EXTRA = {}                 # extra keyword arguments for the turbo BOPDMD
NUM_TRIALS = 0                   # 0 = no bagging
TOL = 1e-6
MAXITER = 200
VERBOSE = False                  # True prints the optimiser's progress

# Plot settings
MODE_SELECTION = "importance"    # "importance": N_MODES most important modes
                                 # "harmonics" : modes closest to f_P, 2 f_P, ..., N_MODES f_P
N_MODES = 6
PLOT_PARTS = ["real", "abs"]     # any of "real", "abs"
XLIM = None                      # e.g. (-1.0, 6.0) to zoom on the airfoil and near wake
YLIM = None                      # e.g. (-1.5, 1.5)
FREQ_MAX = 1.0                   # frequency axis limit of the spectrum plots
AIRFOIL = True                   # overlay the airfoil (xa, ya) on the mode maps, if available
DPI = 200
OUTDIR = Path("turbo_modes_out_a30f0p05")


# =============================================================================
# Data
# =============================================================================
def load_field():
    """Return X (n_space x n_time), dt, spatial shape and the airfoil outline."""
    with h5py.File(FILE, "r") as f:
        t_all = f[TIME_NAME][...].ravel()
        ds = f[FIELD]
        time_axis = max(i for i, s in enumerate(ds.shape) if s == len(t_all))
        data = ds[...]

        airfoil = None
        if AIRFOIL and "xa" in f and "ya" in f:
            try:
                airfoil = []
                for name in ("xa", "ya"):
                    a = np.asarray(f[name])
                    if a.ndim > 1 and "t_force" in f:
                        nt_f = len(np.ravel(f["t_force"]))
                        ax = [i for i, s in enumerate(a.shape) if s == nt_f]
                        if ax:
                            a = np.take(a, 0, axis=ax[0])      # outline at the first instant
                    airfoil.append(np.ravel(a))
            except Exception:                                  # outline is optional
                airfoil = None

    spatial_shape = tuple(s for i, s in enumerate(data.shape) if i != time_axis)
    X = np.moveaxis(data, time_axis, -1).reshape(-1, data.shape[time_axis])
    dt = float(np.median(np.diff(t_all)))
    print(f"[data] field {FIELD}: X = {X.shape}, spatial shape = {spatial_shape}, dt = {dt:g}")
    return X, dt, spatial_shape, airfoil


def load_grid(spatial_shape):
    """Return (x, y, transpose) or None. transpose=True if the field is stored as [x, y]."""
    if GRID_FILE is None or not Path(GRID_FILE).exists():
        print("[grid] grid file not found -> plotting against array indices")
        return None
    try:
        with h5py.File(GRID_FILE, "r") as g:
            x = np.asarray(g["x"]).squeeze()
            y = np.asarray(g["y"]).squeeze()

        def to_1d(a):
            if a.ndim == 2:                                   # meshgrid -> 1-D axis
                return a[:, 0] if np.allclose(a, a[:, :1]) else a[0, :]
            return a

        x, y = to_1d(x), to_1d(y)
        if spatial_shape == (len(x), len(y)):
            return x, y, True
        if spatial_shape == (len(y), len(x)):
            return x, y, False
        print(f"[grid] grid {len(x)}x{len(y)} does not match field {spatial_shape} -> using indices")
    except Exception as exc:
        print(f"[grid] could not read grid ({exc}) -> using indices")
    return None


def delay_embed(F, d):
    """Stack d time-shifted copies of F: (n, m) -> (n*d, m-d+1)."""
    if d <= 1:
        return F
    k = F.shape[1] - d + 1
    return np.vstack([F[:, i:i + k] for i in range(d)])


# =============================================================================
# Fit
# =============================================================================
def fit_turbo(F_train, t_fit):
    sys.path.insert(0, str(Path.cwd()))
    BOPDMD = importlib.import_module(TURBO_MODULE).BOPDMD
    model = BOPDMD(svd_rank=RANK, num_trials=NUM_TRIALS,
                   varpro_opts_dict={"tol": TOL, "maxiter": MAXITER, "verbose": VERBOSE},
                   **TURBO_EXTRA)
    t0 = time.perf_counter()
    model.fit(F_train, t=t_fit)
    print(f"[fit] bopdmdTurbo, rank {RANK}: {time.perf_counter() - t0:.1f} s")
    return model


def reconstruction_errors(modes, amps, omega, X, mean, t, m_train):
    """Relative errors of the reconstruction (train) and forecast (test) against X."""
    m, n = X.shape[1], X.shape[0]
    err2, ref2 = np.empty(m), np.empty(m)
    chunk = max(1, int(1e7 // n))
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        X_hat = (modes @ (amps[:, None] * np.exp(np.outer(omega, t[s:e])))).real
        X_true = X[:, s:e] - mean
        err2[s:e] = np.sum((X_hat - X_true) ** 2, axis=0)
        ref2[s:e] = np.sum(X_true ** 2, axis=0)
    train = math.sqrt(err2[:m_train].sum() / ref2[:m_train].sum())
    test = math.sqrt(err2[m_train:].sum() / ref2[m_train:].sum()) if m_train < m else float("nan")
    return train, test


# =============================================================================
# Mode selection and table
# =============================================================================
def select_modes(freq, score, f_pitch):
    """Indices of the modes to draw (one per conjugate pair, f >= 0)."""
    pos = np.where(freq > 1e-6)[0]
    zero = np.where(np.abs(freq) <= 1e-6)[0]            # stationary modes: keep only the strongest
    pool = np.concatenate([pos, zero[np.argsort(-score[zero])][:1]]).astype(int)
    if MODE_SELECTION == "harmonics" and f_pitch:
        sel = []
        for h in range(1, N_MODES + 1):
            k = pool[np.argmin(np.abs(freq[pool] - h * f_pitch))]
            if k not in sel:
                sel.append(int(k))
        return sel
    return [int(k) for k in pool[np.argsort(-score[pool])][:N_MODES]]


# =============================================================================
# Plots
# =============================================================================
def plot_spectrum(omega, score, freq, dt, f_pitch, sel, fname):
    import matplotlib.pyplot as plt

    growth = omega.real
    s = score / score.max()
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)

    # (a) frequency vs growth rate
    ax = axs[0]
    sc = ax.scatter(freq, growth, s=25 + 300 * s, c=np.log10(s), cmap="viridis",
                    edgecolor="k", linewidth=0.4, zorder=3)
    for k in sel:
        ax.annotate(str(k), (freq[k], growth[k]), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
    if f_pitch:
        for h in range(1, int(FREQ_MAX / f_pitch) + 1):
            ax.axvline(h * f_pitch, color="gray", ls=":", lw=0.8, zorder=1)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(-FREQ_MAX, FREQ_MAX)
    ax.set_xlabel(r"frequency  $f = \mathrm{Im}(\omega)/2\pi$")
    ax.set_ylabel(r"growth rate  $\mathrm{Re}(\omega)$")
    ax.set_title("Continuous-time eigenvalues (dotted: harmonics of $f_P$)")
    fig.colorbar(sc, ax=ax, label=r"$\log_{10}$ relative importance")

    # (b) discrete-time eigenvalues on the unit circle
    ax = axs[1]
    lam = np.exp(omega * dt)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), "k-", lw=0.8)
    ax.scatter(lam.real, lam.imag, s=25 + 300 * s, c=np.log10(s), cmap="viridis",
               edgecolor="k", linewidth=0.4, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\mathrm{Re}(\lambda)$")
    ax.set_ylabel(r"$\mathrm{Im}(\lambda)$")
    ax.set_title(r"Discrete-time eigenvalues  $\lambda = e^{\omega \Delta t}$")

    # (c) amplitude spectrum
    ax = axs[2]
    pos = freq >= -1e-6
    floor = s[pos].min() / 3
    ax.vlines(freq[pos], floor, s[pos], color="C0", lw=1.2)
    ax.plot(freq[pos], s[pos], "o", color="C0", ms=5)
    if f_pitch:
        for h in range(1, int(FREQ_MAX / f_pitch) + 1):
            ax.axvline(h * f_pitch, color="gray", ls=":", lw=0.8, zorder=0)
    ax.set_yscale("log")
    ax.set_xlim(0, FREQ_MAX)
    ax.set_xlabel(r"frequency  $f$")
    ax.set_ylabel(r"$|b_k|\,\|\phi_k\|$  (normalised)")
    ax.set_title("Amplitude spectrum")

    fig.savefig(fname, dpi=DPI)
    plt.close(fig)


def plot_modes(modes, sel, freq, growth, amps, spatial_shape, grid, airfoil, part, fname):
    import matplotlib.pyplot as plt

    N = len(sel)
    ncols = 2 if N > 1 else 1
    nrows = math.ceil(N / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.8 * ncols, 3.3 * nrows),
                             squeeze=False, constrained_layout=True)
    im = None
    for ax in axes.ravel()[N:]:
        ax.axis("off")

    for ax, k in zip(axes.ravel(), sel):
        phi = modes[:, k]
        if part == "real":
            phi = phi * np.exp(-1j * np.angle(phi[np.argmax(np.abs(phi))]))   # fix the phase
            Z = phi.real
            cmap, vmin = "RdBu_r", -1.0
        else:
            Z = np.abs(phi)
            cmap, vmin = "magma", 0.0
        Z = (Z / np.max(np.abs(Z))).reshape(spatial_shape)

        if grid is not None:
            x, y, transpose = grid
            im = ax.pcolormesh(x, y, Z.T if transpose else Z, cmap=cmap, vmin=vmin, vmax=1.0,
                               shading="auto", rasterized=True)
            ax.set_aspect("equal")
            ax.set_xlabel(r"$x/c$")
            ax.set_ylabel(r"$y/c$")
            if airfoil is not None:
                ax.plot(airfoil[0], airfoil[1], "k-", lw=1.0)
            if XLIM:
                ax.set_xlim(XLIM)
            if YLIM:
                ax.set_ylim(YLIM)
        else:
            Zp = Z.T if Z.shape[0] > Z.shape[1] else Z        # assume the longer axis is x
            im = ax.imshow(Zp, origin="lower", cmap=cmap, vmin=vmin, vmax=1.0, aspect="equal")
            ax.set_xlabel("x index")
            ax.set_ylabel("y index")
        ax.set_title(f"mode {k}:  f = {freq[k]:.4f},  Re($\\omega$) = {growth[k]:+.2e},  "
                     f"|b| = {abs(amps[k]):.2e}", fontsize=10)

    label = (r"$\mathrm{Re}(\phi)/\max$" if part == "real" else r"$|\phi|/\max$")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label=f"{FIELD}: {label}")
    fig.savefig(fname, dpi=DPI)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    import matplotlib
    matplotlib.use("Agg")

    OUTDIR.mkdir(exist_ok=True)
    X, dt, spatial_shape, airfoil = load_field()
    grid = load_grid(spatial_shape)

    f_pitch = F_PITCH
    if f_pitch is None:
        match = re.search(r"f(\d+)p(\d+)", Path(FILE).name)
        f_pitch = float(f"{match.group(1)}.{match.group(2)}") if match else None

    m = min(N_SNAP, X.shape[1])
    X = X[:, :m]
    n = X.shape[0]
    t = np.arange(m) * dt
    m_train = int(round(m * (1 - TEST_FRAC)))
    k = m_train - DELAY + 1
    if RANK >= k:
        raise SystemExit("rank >= number of (delayed) training snapshots")

    Xn = X
    if NOISE > 0:
        rng = np.random.default_rng(SEED)
        sigma = NOISE * np.std(X - X.mean(axis=1, keepdims=True))
        Xn = X + sigma * rng.standard_normal(X.shape)
    mean = Xn[:, :m_train].mean(axis=1, keepdims=True)
    F_train = delay_embed(Xn[:, :m_train] - mean, DELAY)

    model = fit_turbo(F_train, t[:k])
    omega = np.asarray(model.eigs)
    amps = np.asarray(model.amplitudes).ravel()
    modes = np.asarray(model.modes)[:n]              # physical block of the (delayed) modes

    train_err, test_err = reconstruction_errors(modes, amps, omega, X, mean, t, m_train)
    print(f"[error] train = {train_err:.3e}, test = {test_err:.3e}")

    freq = omega.imag / (2 * np.pi)
    score = np.abs(amps) * np.linalg.norm(modes, axis=0)
    sel = select_modes(freq, score, f_pitch)

    # table of all modes
    table = pd.DataFrame({
        "mode": np.arange(len(omega)), "freq": freq, "growth_rate": omega.real,
        "abs_amplitude": np.abs(amps), "mode_norm": np.linalg.norm(modes, axis=0),
        "importance": score / score.max(), "plotted": [i in sel for i in range(len(omega))],
    }).sort_values("importance", ascending=False)
    table.to_csv(OUTDIR / "modes_table.csv", index=False)
    print(table.head(12).to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # plots
    plot_spectrum(omega, score, freq, dt, f_pitch, sel, OUTDIR / "spectrum.png")
    for part in PLOT_PARTS:
        plot_modes(modes, sel, freq, omega.real, amps, spatial_shape, grid, airfoil, part,
                   OUTDIR / f"modes_{part}.png")

    # raw decomposition, so the figures can be redrawn without refitting
    np.savez_compressed(
        OUTDIR / "turbo_decomposition.npz",
        omega=omega, amplitudes=amps, modes=modes.astype(np.complex64),
        spatial_shape=np.array(spatial_shape), dt=dt, f_pitch=f_pitch or np.nan,
        x=grid[0] if grid else np.nan, y=grid[1] if grid else np.nan,
        settings=str(dict(FIELD=FIELD, RANK=RANK, N_SNAP=N_SNAP, TEST_FRAC=TEST_FRAC,
                          NOISE=NOISE, DELAY=DELAY, TOL=TOL, MAXITER=MAXITER)))
    print(f"\nResults written to {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
