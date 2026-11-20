"""
Figura de comparação entre os modos do DMD de referência (rotina calc_dmd.m
do banco de dados de Towne et al., 2023, reproduzida em Python) e os do
bopdmdTurbo no caso B400, para as quatro frequências de maior importância.
Mostra a parte real de u_y, com a fase de cada modo do bopdmdTurbo alinhada
à do modo de referência e cada campo normalizado pelo percentil 99,5.

Usa a mesma estrutura de pastas do compare_modes_towne.py:
    dados/airfoilDNS_a30f0p05.h5 (ou a30f0p05_uy_parte*.h5)
    dados/airfoilDNS_grid.h5
    bopdmdTurboV3.py (na pasta deste script)
A figura é gravada em resultados/modos_B400_ref.pdf e .png.

Uso:
    python figura_modos_ref.py
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import compare_modes_towne as C  # noqa: E402

GRADE = C.DADOS / "airfoilDNS_grid.h5"
SAIDA = BASE / "resultados"
M, POSTO, POSTO_REF = 400, 20, 100
F_P, F_N = 0.05, 0.2381  # f_n obtida pelo DMD de referência


def br(v, nd=4):
    """Número com vírgula decimal."""
    return f"{v:.{nd}f}".replace(".", ",")


def main():
    if not GRADE.exists():
        raise FileNotFoundError(f"Arquivo de grade não encontrado: {GRADE}")
    with h5py.File(GRADE, "r") as f:
        x = np.asarray(f["x"]).squeeze()
        y = np.asarray(f["y"]).squeeze()

    X, t, grade = C.load_field(C.arquivos_caso_b())
    ny, nx = grade
    if (ny, nx) != (y.size, x.size):
        raise ValueError(f"Malha dos dados {grade} diferente da grade "
                         f"({y.size}, {x.size}).")

    # bopdmdTurbo, caso B400 (mesmo pré-processamento do TCC)
    Xc, tc = C.train_fluctuations(X, t, M)
    om_t, P_t, _ = C.run_turbo(Xc, tc, POSTO)
    del Xc
    # Referência: DMD exato sobre todos os snapshots, média removida
    X -= X.mean(axis=1, keepdims=True)
    om_r, P_r, _ = C.exact_dmd(X, t, POSTO_REF)
    del X

    alvos = [F_N, F_N - F_P, F_N + F_P, F_P]
    f_t = om_t.imag / (2 * np.pi)
    f_r = om_r.imag / (2 * np.pi)

    fig, axs = plt.subplots(4, 2, figsize=(6.3, 6.6), sharex=True,
                            sharey=True, constrained_layout=True)
    niveis = np.linspace(-1, 1, 21)
    for i, alvo in enumerate(alvos):
        kr = int(np.argmin(np.abs(f_r - alvo)))
        kt = int(np.argmin(np.abs(f_t - alvo)))
        pr, pt = P_r[:, kr], P_t[:, kt]
        c = np.vdot(pr, pt)
        s = abs(c) / (np.linalg.norm(pr) * np.linalg.norm(pt))
        pt = pt * np.conj(c) / abs(c)  # alinha a fase à da referência
        for j, (modo, f) in enumerate(((pr, f_r[kr]), (pt, f_t[kt]))):
            campo = modo.real.reshape(ny, nx)
            # percentil evita que picos junto ao aerofólio apaguem a esteira
            campo = campo / np.percentile(np.abs(campo), 99.5)
            ax = axs[i, j]
            cs = ax.contourf(x, y, campo, levels=niveis, cmap="RdBu_r",
                             extend="both")
            ax.set_aspect("equal")
            rot = "Referência" if j == 0 else r"$\mathtt{bopdmdTurbo}$"
            extra = "" if j == 0 else f", $s$ = {br(s, 3)}"
            ax.set_title(f"{rot}: $f$ = {br(f)}{extra}", fontsize=8.5)
            ax.tick_params(labelsize=7.5)
            if j == 0:
                ax.set_ylabel("$y/c$", fontsize=8.5)
            if i == 3:
                ax.set_xlabel("$x/c$", fontsize=8.5)
    cb = fig.colorbar(cs, ax=axs, shrink=0.6, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.set_yticklabels(["−1", "−0,5", "0", "0,5", "1"])
    cb.ax.tick_params(labelsize=7.5)
    cb.set_label(r"Re$(\phi_{u_y})$ normalizada (percentil 99,5)", fontsize=8.5)

    SAIDA.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        caminho = SAIDA / f"modos_B400_ref.{ext}"
        fig.savefig(caminho, dpi=200)
        print(f"Gravado: {caminho}")


if __name__ == "__main__":
    main()
