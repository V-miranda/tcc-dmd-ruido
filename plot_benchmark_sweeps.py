#!/usr/bin/env python3
"""Figuras das varreduras do benchmark (tempo, memória, erro de treino e de previsão).

Uso:
    python plot_benchmark_sweeps.py results_summary.csv saida/ [--sufixo ""]

Lê o CSV de resumo do benchmark (uma linha por método e configuração) e grava,
para cada varredura (posto, número de snapshots, ruído, atraso), um PNG (300 dpi)
e um PDF vetorial: sweep_rank, sweep_n_snap, sweep_noise, sweep_delay.

A configuração de referência é rank=20, n_snap=300, noise=0, delay=1; cada varredura
varia um único parâmetro a partir dela. Métodos sem resultado numa configuração
(por exemplo, falha do BOP-DMD com atraso) aparecem como lacuna; descreva a falha na legenda.
"""
import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------
# Estilo
# --------------------------------------------------------------------------
INK = "#1d1d1b"          # texto principal
INK2 = "#52514e"         # texto secundário
GRID = "#e4e3df"         # grade, recessiva
AXIS = "#9a9992"
SURFACE = "#ffffff"

# Três séries: azul, laranja, verde-água (slots 1-3 da paleta, validados contra
# daltonismo). Marcadores e traços distintos garantem leitura em preto e branco.
METHODS = [
    # chave no CSV, rótulo, cor, marcador, estilo de linha
    ("dmd",    "DMD",                   "#2a78d6", "o", "-"),
    ("bopdmd", "BOP-DMD (PyDMD)",       "#eb6834", "s", "--"),
    ("turbo",  "BOP-DMD-Turbo", "#1baf7a", "^", "-"),
]

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",   # texto editável no SVG (Inkscape/Illustrator)
})

BASE = dict(rank=20, n_snap=300, noise=0.0, delay=1)

PANELS = [
    # coluna, título, escala y, rótulo y
    ("fit_time_s",  "Tempo de ajuste",           "log",    "Tempo (s)"),
    ("peak_mem_gb", "Pico de memória",           "linear", "Memória (GB)"),
    ("train_err",   "Erro de treino",            "log",    r"$\varepsilon_{tr}$"),
    ("test_err",    "Erro de previsão (teste)",  "log",    r"$\varepsilon_{te}$"),
]

SWEEPS = {
    "rank":   dict(col="rank",   xlabel="Posto de truncamento $r$",  ticks=[10, 20, 40],
                   fmt=lambda v: f"{v:g}", file="sweep_rank"),
    "n_snap": dict(col="n_snap", xlabel="Número de snapshots $m$",    ticks=[200, 300, 400],
                   fmt=lambda v: f"{v:g}", file="sweep_n_snap"),
    "noise":  dict(col="noise",  xlabel="Nível de ruído $\\eta$",     ticks=[0, 0.05, 0.2],
                   fmt=lambda v: f"{v:g}".replace(".", ","), file="sweep_noise"),
    "delay":  dict(col="delay",  xlabel="Atraso temporal $d$",        ticks=[1, 2, 4],
                   fmt=lambda v: f"{v:g}", file="sweep_delay"),
}


def select(df, sweep_col):
    """Linhas da varredura: o parâmetro varia e os demais ficam na referência."""
    mask = np.ones(len(df), dtype=bool)
    for k, v in BASE.items():
        if k != sweep_col:
            mask &= np.isclose(df[k].to_numpy(dtype=float), v)
    return df[mask]


def decimal_comma(ax, axis="y"):
    """Vírgula decimal nos rótulos numéricos (padrão ABNT)."""
    fmt = mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def make_figure(df, key, outdir, suffix=""):
    sw = SWEEPS[key]
    sub = select(df, sw["col"])
    xs = np.array(sw["ticks"], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.3), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.06, h_pad=0.08, wspace=0.10, hspace=0.12)

    failed_methods = []
    for ax, (col, title, yscale, ylabel), letter in zip(axes.ravel(), PANELS, "abcd"):
        for mkey, label, color, marker, ls in METHODS:
            d = sub[sub["method"] == mkey].set_index(sw["col"])[col]
            d = d.reindex(xs)
            if d.isna().any() and mkey not in failed_methods and d.notna().any():
                failed_methods.append(mkey)
            ax.plot(xs, d.to_numpy(), ls, color=color, lw=1.8, marker=marker, ms=6.5,
                    mfc=color, mec=SURFACE, mew=1.4, solid_capstyle="round",
                    solid_joinstyle="round", zorder=3, label=label, clip_on=False)
        ax.set_yscale(yscale)
        ax.set_xticks(xs)
        ax.set_xticklabels([sw["fmt"](v) for v in xs])
        pad = 0.08 * (xs.max() - xs.min())
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        ax.set_xlabel(sw["xlabel"])
        ax.set_ylabel(ylabel)
        ax.set_title(f"({letter}) {title}", loc="left", fontweight="normal")
        ax.grid(axis="x", visible=False)
        if yscale == "linear":
            ax.set_ylim(bottom=0)
            ax.margins(y=0.08)
            decimal_comma(ax)
        else:
            ax.margins(y=0.15)
            lo, hi = ax.get_ylim()
            if hi / lo < 40:   # faixa estreita: rótulos em números simples
                ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10, subs=(1, 2, 3, 5), numticks=12))
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(
                    lambda v, _: f"{v:g}".replace(".", ",")))
                ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
            else:
                ax.yaxis.set_major_formatter(mpl.ticker.LogFormatterSciNotation())
        ax.tick_params(which="minor", length=2, width=0.6, color=AXIS)

    handles = [Line2D([0], [0], color=c, lw=1.8, ls=ls, marker=m, ms=6.5, mfc=c, mec=SURFACE, mew=1.4)
               for _, _, c, m, ls in METHODS]
    fig.legend(handles, [m[1] for m in METHODS], loc="outside upper center", ncol=3,
               fontsize=9, handlelength=2.6, columnspacing=2.2, labelcolor=INK)

    base = os.path.join(outdir, sw["file"] + suffix)
    fig.savefig(base + ".png", dpi=300)
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".svg")
    plt.close(fig)
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="results_summary.csv")
    ap.add_argument("outdir", help="pasta de saída")
    ap.add_argument("--sufixo", default="", help="sufixo dos nomes de arquivo")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)
    for key in SWEEPS:
        print("gravado:", make_figure(df, key, args.outdir, args.sufixo) + ".{png,pdf,svg}")


if __name__ == "__main__":
    main()
