#!/usr/bin/env python3
"""Espectro DMD do caso B a partir do modes_table.csv do plot_turbo_modes.py.

Uso:
    python plot_modal_spectrum.py modes_table.csv saida/a30f0p05_spectrum \
        [--dt 0.1] [--fp 0.05] [--fn 0.2418]

Grava saida/<nome>.png (300 dpi), .pdf e .svg. Painéis:
  (a) taxa de crescimento Re(omega) em função da frequência (tamanho = importância);
  (b) autovalores de tempo discreto lambda = exp(omega*dt), ampliados perto de lambda = 1;
  (c) espectro de amplitude (importância relativa |b|·||phi||, normalizada pelo máximo).
Só os modos de frequência positiva são mostrados; os negativos são os conjugados.
As linhas pontilhadas marcam os múltiplos de f_P. Os quatro modos de maior importância
são identificados pelo combinação de f_n e f_P mais próxima (f_n, f_n -+ f_P, f_P).
"""
import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK2, GRID, AXIS, SURFACE = "#1d1d1b", "#52514e", "#e4e3df", "#9a9992", "#ffffff"
BLUE = "#2a78d6"      # série principal
MUTED = "#8fb5e8"     # modos de menor importância (mesma cor, mais clara)

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42, "svg.fonttype": "none",
})


def comma(ax, axis):
    f = mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out", help="caminho de saída sem extensão")
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--fp", type=float, default=0.05)
    ap.add_argument("--fn", type=float, default=0.2418)
    ap.add_argument("--top", type=int, default=4, help="número de modos rotulados")
    a = ap.parse_args()

    d = pd.read_csv(a.csv)
    d = d[d["freq"] > 0].sort_values("importance", ascending=False).reset_index(drop=True)
    f, g, imp = d["freq"].to_numpy(), d["growth_rate"].to_numpy(), d["importance"].to_numpy()
    lam = np.exp((g + 2j * np.pi * f) * a.dt)
    size = 25 + 230 * imp                      # área do marcador proporcional à importância
    top = np.arange(len(d)) < a.top

    # rótulo: combinação nominal mais próxima
    cands = {r"$f_n$": a.fn, r"$f_P$": a.fp, r"$f_n + f_P$": a.fn + a.fp, r"$f_n - f_P$": a.fn - a.fp}
    labels = {}
    for i in np.where(top)[0]:
        labels[i] = min(cands, key=lambda k: abs(cands[k] - f[i]))

    fig = plt.figure(figsize=(6.4, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], width_ratios=[1.1, 1.0])
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])

    harm = np.arange(1, int(0.8 / a.fp) + 1) * a.fp

    # (a) taxa de crescimento
    for h in harm:
        axa.axvline(h, color=GRID, lw=0.8, ls=":", zorder=1)
    axa.axhline(0, color=AXIS, lw=0.9, zorder=2)
    axa.scatter(f[~top], g[~top] * 1e3, s=size[~top], c=MUTED, ec=SURFACE, lw=1.2, zorder=3)
    axa.scatter(f[top], g[top] * 1e3, s=size[top], c=BLUE, ec=SURFACE, lw=1.2, zorder=4)
    offs = {r"$f_n$": (0, 14), r"$f_n - f_P$": (-22, 12), r"$f_n + f_P$": (18, -17), r"$f_P$": (0, 12)}
    for i, lab in labels.items():
        axa.annotate(lab, (f[i], g[i] * 1e3), xytext=offs[lab], textcoords="offset points",
                     ha="center", va="center", fontsize=8.5, color=INK)
    axa.set_xlim(0, 0.8)
    axa.set_xlabel("Frequência $f = \\mathrm{Im}(\\omega)/2\\pi$")
    axa.set_ylabel("$\\mathrm{Re}(\\omega)\\ (\\times 10^{-3})$")
    axa.set_title("(a) Taxa de crescimento", loc="left", fontweight="normal")
    axa.grid(False)
    comma(axa, "x")
    axa.set_ylim(min(g.min() * 1e3 - 2, -4), max(g.max() * 1e3 + 4, 9))

    # (b) autovalores de tempo discreto, ampliados
    th = np.linspace(-0.1, 0.9, 300)
    axb.plot(np.cos(th), np.sin(th), color=AXIS, lw=1.0, zorder=2)
    axb.scatter(lam.real[~top], lam.imag[~top], s=size[~top], c=MUTED, ec=SURFACE, lw=1.2, zorder=3)
    axb.scatter(lam.real[top], lam.imag[top], s=size[top], c=BLUE, ec=SURFACE, lw=1.2, zorder=4)
    axb.set_aspect("equal", adjustable="box")
    axb.set_xlim(0.58, 1.02)
    axb.set_ylim(-0.02, 0.50)
    axb.set_xlabel("$\\mathrm{Re}(\\lambda)$")
    axb.set_ylabel("$\\mathrm{Im}(\\lambda)$")
    axb.set_title("(b) Autovalores discretos", loc="left", fontweight="normal")
    axb.annotate("círculo unitário", (np.cos(0.5), np.sin(0.5)), xytext=(-8, -2), textcoords="offset points",
                 ha="right", va="top", fontsize=8, color=INK2)
    comma(axb, "x"); comma(axb, "y")

    # (c) espectro de amplitude
    for h in harm:
        axc.axvline(h, color=GRID, lw=0.8, ls=":", zorder=1)
    axc.vlines(f, 1e-3, imp, color=np.where(top, BLUE, MUTED), lw=1.6, zorder=2)
    axc.scatter(f[~top], imp[~top], s=34, c=MUTED, ec=SURFACE, lw=1.2, zorder=3)
    axc.scatter(f[top], imp[top], s=40, c=BLUE, ec=SURFACE, lw=1.2, zorder=4)
    for i, lab in labels.items():
        dx = {r"$f_n - f_P$": -0.014, r"$f_n + f_P$": 0.014}.get(lab, 0)
        ha = {r"$f_n - f_P$": "right", r"$f_n + f_P$": "left"}.get(lab, "center")
        dy = 1.0 if dx == 0 else 0.0
        axc.annotate(lab, (f[i] + dx, imp[i] * (1.18 if dy else 1.0)), ha=ha,
                     va="bottom" if dy else "center", fontsize=8.5, color=INK)
    axc.set_yscale("log")
    axc.set_ylim(1.5e-2, 2.2)
    axc.set_xlim(0, 0.8)
    axc.set_xlabel("Frequência $f$")
    axc.set_ylabel("Importância relativa\n$|b_k|\\,\\|\\phi_k\\|$ (normalizada)")
    axc.set_title("(c) Espectro de amplitude", loc="left", fontweight="normal")
    axc.grid(False)
    axc.yaxis.set_major_formatter(mpl.ticker.LogFormatterSciNotation())
    comma(axc, "x")
    axc.text(0.995, 0.95, "linhas pontilhadas: múltiplos de $f_P$", transform=axc.transAxes,
             ha="right", va="top", fontsize=8, color=INK2)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    for ext, kw in (("png", {"dpi": 300}), ("pdf", {}), ("svg", {})):
        fig.savefig(f"{a.out}.{ext}", **kw)
    print("gravado:", a.out + ".{png,pdf,svg}")


if __name__ == "__main__":
    main()
