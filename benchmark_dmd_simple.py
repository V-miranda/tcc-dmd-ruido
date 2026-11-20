#!/usr/bin/env python3
"""
benchmark_dmd_simple.py
=======================
Compares three DMD implementations on one HDF5 snapshot file
(e.g. the pitching-airfoil DNS dataset):

    dmd     -> pydmd.DMD
    bopdmd  -> pydmd.BOPDMD
    turbo   -> bopdmdTurbo.BOPDMD    (own version; module name set in TURBO_MODULE)

What it does, step by step:
    1. Load the snapshots into a matrix X (space x time).
    2. Optionally add Gaussian noise, remove the temporal mean and
       (optionally) stack time-delayed copies of the snapshots.
    3. Fit each method on the first (1 - TEST_FRAC) of the snapshots.
       A warm-up fit (same shapes, untimed) runs first, so that one-off costs
       (JAX jit compilation, BLAS/thread start-up, cache warming) are not
       counted in the fit time.
    4. Reconstruct ALL snapshots from the DMD modes and measure
         - train error    (reconstruction of the data the model saw)
         - test error     (forecast of the held-out final snapshots)
    5. Record fit time, peak memory and the dominant frequencies.
    6. Repeat while sweeping ONE parameter at a time
       (rank, number of snapshots, noise, delay), keeping the others at BASE.

Every run happens in a fresh Python process, so the peak-memory numbers
of one run are not polluted by the previous one.

Usage:  edit the SETTINGS block below, then run
        python benchmark_dmd_simple.py

Requirements: numpy, h5py, pandas, matplotlib, psutil (Windows only), pydmd
"""

import re
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# =============================================================================
# SETTINGS  (everything you normally want to change is here)
# =============================================================================
FILE = "airfoilDNS/airfoilDNS_a25f0p1.h5"
FIELD = "uy"                # vertical (wall-normal) velocity; other options: "ux" (streamwise velocity), "vort"
TIME_NAME = "t_field"       # dataset holding the snapshot times
T_STRIDE = 1                # keep every n-th snapshot
SPACE_STRIDE = 1            # keep every n-th grid point (in each space direction)
F_PITCH = None              # pitching frequency; None = read it from the file name (f0p1 -> 0.1)

METHODS = ["dmd", "bopdmd", "turbo"]

# Reference case; each sweep changes one of these and keeps the other two.
# delay = number of stacked time-shifted copies (1 = no delay embedding)
BASE = {"rank": 20, "n_snap": 300, "noise": 0.0, "delay": 1}
SWEEPS = {
    "rank":   [10, 20, 40],
    "n_snap": [200, 300, 400],
    "noise":  [0.0, 0.05, 0.2],     # fraction of the fluctuation standard deviation
    "delay":  [1, 2, 4],            # delay sweep
}

REPS = 3                    # repetitions per case (median is reported)
TEST_FRAC = 0.2             # last 20 % of the snapshots are held out for the forecast test
OUTDIR = Path("bench_out")

# BOPDMD options (identical for pydmd's BOPDMD and the turbo version, so the
# stopping condition is the same for both).
#   tol     = 1e-6 : PyDMD default; a tighter value is pointless on noisy data and
#                    makes L-BFGS-B (ftol = tol**2) run until maxiter.
#   maxiter = 200  : above PyDMD's default (30) and turbo's default (100), so
#                    neither method is cut off at rank 40 / delay 4.
BOP_KWARGS = {"num_trials": 0,          # 0 = no bagging (single opt-DMD fit)
              "varpro_opts_dict": {"tol": 1e-6, "maxiter": 200, "verbose": False}}
TURBO_MODULE = "bopdmdTurboV3"          # file bopdmdTurboV3.py next to this script
TURBO_EXTRA = {}            # extra keyword arguments only for the turbo BOPDMD

# Warm-up: before each timed fit, the same fit (same data shapes) is run once,
# untimed, on a throw-away model with only WARMUP_MAXITER iterations.
WARMUP = True
WARMUP_MAXITER = 2

LABELS = {"dmd": "PyDMD DMD", "bopdmd": "PyDMD BOPDMD", "turbo": "bopdmdTurbo"}


# =============================================================================
# 1. Data loading
# =============================================================================
def load_data():
    """Return X (n_space x n_time), dt and the pitching frequency."""
    with h5py.File(FILE, "r") as f:
        t_all = f[TIME_NAME][...].ravel()
        ds = f[FIELD]

        # the time axis is the one whose length equals len(t_all)
        time_axis = max(i for i, s in enumerate(ds.shape) if s == len(t_all))

        # stride in space and in time, in a single read
        index = [slice(None, None, SPACE_STRIDE)] * ds.ndim
        index[time_axis] = slice(None, None, T_STRIDE)
        data = ds[tuple(index)]

    # (..., time) -> (n_space, n_time)
    X = np.moveaxis(data, time_axis, -1).reshape(-1, data.shape[time_axis])
    dt = float(np.median(np.diff(t_all[::T_STRIDE])))

    f_pitch = F_PITCH
    if f_pitch is None:
        match = re.search(r"f(\d+)p(\d+)", Path(FILE).name)
        f_pitch = float(f"{match.group(1)}.{match.group(2)}") if match else None

    print(f"[data] X = {X.shape}, dt = {dt:g}, f_pitch = {f_pitch}")
    return X, dt, f_pitch


# =============================================================================
# 2. One run (executed in its own process)
# =============================================================================
def peak_memory_gb():
    """Highest memory the current process has used so far, in GB."""
    try:
        import resource                                   # Linux / macOS
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / (1024**3 if sys.platform == "darwin" else 1024**2)
    except ImportError:                                   # Windows
        import psutil
        return psutil.Process().memory_info().peak_wset / 1024**3


def delay_embed(F, d):
    """Stack d time-shifted copies of F: (n, m) -> (n*d, m-d+1)."""
    if d <= 1:
        return F
    k = F.shape[1] - d + 1
    return np.vstack([F[:, i:i + k] for i in range(d)])


def build_model(method, rank, maxiter=None):
    """Import and create the model (outside the timed region).
    maxiter overrides the iteration limit (used by the warm-up fit)."""
    kwargs = {**BOP_KWARGS, "varpro_opts_dict": dict(BOP_KWARGS["varpro_opts_dict"])}
    if maxiter is not None:
        kwargs["varpro_opts_dict"]["maxiter"] = maxiter
    if method == "dmd":
        from pydmd import DMD
        return DMD(svd_rank=rank)
    if method == "bopdmd":
        from pydmd import BOPDMD
        return BOPDMD(svd_rank=rank, **kwargs)
    if method == "turbo":
        import importlib
        sys.path.insert(0, str(Path.cwd()))               # find the module next to the script
        BOPDMD = importlib.import_module(TURBO_MODULE).BOPDMD
        return BOPDMD(svd_rank=rank, **kwargs, **TURBO_EXTRA)
    raise ValueError(f"unknown method {method}")


def fit_model(model, method, F, t, dt):
    """Fit the model and return the continuous-time eigenvalues omega."""
    if method == "dmd":
        model.fit(F)
        return np.log(np.asarray(model.eigs, dtype=complex)) / dt   # discrete -> continuous
    model.fit(F, t=t)
    return np.asarray(model.eigs)


def run_one(job):
    """Fit one method on one case and return a dict of metrics."""
    method, rank, m, noise = job["method"], job["rank"], job["n_snap"], job["noise"]
    d, dt = job["delay"], job["dt"]

    # ---- prepare data (not timed) -------------------------------------------
    X = np.array(np.load(job["data_path"], mmap_mode="r")[:, :m])   # clean data
    n = X.shape[0]
    m_train = int(round(m * (1 - TEST_FRAC)))
    k = m_train - d + 1                                    # columns left after delay embedding
    if rank >= k:
        return {"status": "skipped: rank >= number of (delayed) training snapshots"}

    Xn = X
    if noise > 0:
        rng = np.random.default_rng(job["seed"])
        sigma = noise * np.std(X - X.mean(axis=1, keepdims=True))
        Xn = X + sigma * rng.standard_normal(X.shape)

    mean = Xn[:, :m_train].mean(axis=1, keepdims=True)
    F_train = delay_embed(Xn[:, :m_train] - mean, d)       # what the model is fitted on
    t = np.arange(m) * dt

    # ---- warm-up (untimed) --------------------------------------------------
    # Same shapes as the real fit, so jit-compiled functions and library
    # start-up costs are paid here and not inside the timed region.
    warmup_time = 0.0
    if WARMUP:
        import gc
        import warnings
        warm = build_model(method, rank, maxiter=WARMUP_MAXITER)
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_model(warm, method, F_train, t[:k], dt)
        warmup_time = time.perf_counter() - t0
        del warm
        gc.collect()

    model = build_model(method, rank)

    # ---- fit (timed) --------------------------------------------------------
    t0 = time.perf_counter()
    omega = fit_model(model, method, F_train, t[:k], dt)
    fit_time = time.perf_counter() - t0
    peak_mem = peak_memory_gb()      # process peak so far (includes data and warm-up)

    modes = np.asarray(model.modes)[:n]                    # keep the physical block of delayed modes
    amps = np.asarray(model.amplitudes).ravel()

    # ---- reconstruct all snapshots, compare with the CLEAN data --------------
    # x(t) = Phi @ (b * exp(omega * t)); done in chunks of columns to save memory
    err2 = np.empty(m)       # squared error per snapshot
    ref2 = np.empty(m)       # squared norm of the true snapshot
    chunk = max(1, int(1e7 // n))
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        X_hat = (modes @ (amps[:, None] * np.exp(np.outer(omega, t[s:e])))).real
        X_true = X[:, s:e] - mean
        err2[s:e] = np.sum((X_hat - X_true) ** 2, axis=0)
        ref2[s:e] = np.sum(X_true ** 2, axis=0)
    train_err = np.sqrt(err2[:m_train].sum() / ref2[:m_train].sum())
    test_err = np.sqrt(err2[m_train:].sum() / ref2[m_train:].sum())

    # ---- spectrum -----------------------------------------------------------
    T = m_train * dt                                       # length of the training window
    score = np.abs(amps) * np.linalg.norm(modes, axis=0)   # importance of each mode
    dominant = []                                          # 5 most important distinct frequencies
    for i in np.argsort(-score):
        freq = abs(omega[i].imag) / (2 * np.pi)
        if all(abs(freq - x) > 0.5 / T for x in dominant):
            dominant.append(freq)
        if len(dominant) == 5:
            break

    out = {
        "status": "ok",
        "fit_time_s": fit_time,
        "warmup_time_s": warmup_time,
        "peak_mem_gb": peak_mem,
        "train_err": train_err,
        "test_err": test_err,
        "n_unstable": int(np.sum(omega.real * T > 0.01)),   # modes that grow by > 0.01 e-folds
        "dominant_freqs": ";".join(f"{x:.4g}" for x in dominant),
    }
    if job["f_pitch"]:
        nearest = min(dominant, key=lambda x: abs(x - job["f_pitch"]))
        out["pitch_rel_dev"] = abs(nearest - job["f_pitch"]) / job["f_pitch"]
    return out


def run_isolated(job):
    """Run one job in a brand-new process so memory is measured cleanly."""
    ctx = mp.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
            return pool.submit(run_one, job).result()
    except Exception as exc:                               # import error, out of memory, ...
        return {"status": f"failed: {type(exc).__name__}: {exc}"}


# =============================================================================
# 3. Plots
# =============================================================================
def plot_sweeps(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [("fit_time_s", "fit time [s]"), ("peak_mem_gb", "peak memory [GB]"),
               ("train_err", "train error"), ("test_err", "forecast (test) error")]

    for param in SWEEPS:
        # keep only the rows where the OTHER parameters are at their base value
        rows = summary
        for other, value in BASE.items():
            if other != param:
                rows = rows[rows[other] == value]

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, (col, title) in zip(axes.ravel(), metrics):
            for method in METHODS:
                r = rows[rows["method"] == method].sort_values(param)
                if len(r):
                    ax.plot(r[param], r[col], "o-", label=LABELS[method])
            ax.set_xlabel(param)
            ax.set_title(title)
            if col != "peak_mem_gb":
                ax.set_yscale("log")
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUTDIR / f"sweep_{param}.png", dpi=150)
        plt.close(fig)


# =============================================================================
# 4. Main
# =============================================================================
def main():
    OUTDIR.mkdir(exist_ok=True)

    # load once, save to disk so every worker process can read it
    X, dt, f_pitch = load_data()
    data_path = str(OUTDIR / "X_tmp.npy")
    np.save(data_path, X)
    n_available = X.shape[1]
    del X

    # list of cases: one parameter changes at a time, the rest stay at BASE
    cases = []
    for name, values in SWEEPS.items():
        for v in values:
            case = {**BASE, name: v}
            case["n_snap"] = min(case["n_snap"], n_available)
            if case not in cases:
                cases.append(case)

    rows = []
    total = len(cases) * len(METHODS) * REPS
    count = 0
    for case in cases:
        for method in METHODS:
            for rep in range(REPS):
                count += 1
                print(f"[{count}/{total}] {method:7s} {case} rep={rep} ... ", end="", flush=True)
                job = {**case, "method": method, "seed": rep, "dt": dt,
                       "f_pitch": f_pitch, "data_path": data_path}
                result = run_isolated(job)
                if result["status"] == "ok":
                    print(f"{result['fit_time_s']:.2f} s, test err {result['test_err']:.3g}")
                else:
                    print(result["status"])
                rows.append({**case, "method": method, "rep": rep, **result})

    Path(data_path).unlink(missing_ok=True)

    # ---- save and summarise ---------------------------------------------------
    results = pd.DataFrame(rows)
    results.to_csv(OUTDIR / "results_all_runs.csv", index=False)

    ok = results[results["status"] == "ok"]
    if ok.empty:
        print("No successful runs.")
        return
    keys = ["method", "rank", "n_snap", "noise", "delay"]
    numeric = ["fit_time_s", "warmup_time_s", "peak_mem_gb", "train_err", "test_err", "n_unstable", "pitch_rel_dev"]
    numeric = [c for c in numeric if c in ok.columns]
    summary = ok.groupby(keys)[numeric].median().reset_index()   # median over repetitions
    summary.to_csv(OUTDIR / "results_summary.csv", index=False)

    print("\n" + summary.sort_values(["rank", "n_snap", "noise", "delay", "method"])
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    plot_sweeps(summary)
    print(f"\nResults written to {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
