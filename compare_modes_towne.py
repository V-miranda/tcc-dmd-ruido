"""
Comparação quantitativa entre os modos do bopdmdTurbo (caso B do TCC) e os
modos de um DMD de referência, nos moldes da Fig. 21 de Towne et al. (2023).

Caso B: airfoilDNS_a30f0p05.h5, campo uy, alpha0 = 30 graus, f_P = 0,05.

O artigo não informa número de snapshots, posto nem pré-processamento do DMD
da figura. Por isso, a referência é calculada com DMD exato (Tu et al., 2014)
sobre um trecho configurável dos dados, e o posto da referência é varrido.
Opcionalmente, a referência pode ser lida de um .mat gerado com o calc_dmd.m
do próprio banco de dados (opção --ref-mat).

Para cada modo do bopdmdTurbo de frequência positiva, entre os mais
importantes, são reportados:
  - o modo de referência de frequência mais próxima, o desvio de frequência
    e a similaridade de forma entre os dois;
  - o modo de referência de maior similaridade de forma, e a frequência dele.

Similaridade de forma (insensível à fase e à escala arbitrárias do modo):
    s = |phi_ref^H phi| / (||phi_ref||_2 ||phi||_2),  0 <= s <= 1.

Estrutura de pastas esperada (tudo relativo a este script):
    compare_modes_towne.py
    figura_modos_ref.py
    bopdmdTurboV3.py
    dados/
        airfoilDNS_a30f0p05.h5   (ou a30f0p05_uy_parte1.h5, parte2.h5, ...)
        airfoilDNS_grid.h5       (usado pela figura)
    resultados/                  (criada automaticamente)

Uso típico:
    python compare_modes_towne.py --m 400 --ref-ranks 20 100
    python compare_modes_towne.py --m 300 --ref-ranks 20 100
"""

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np


# --------------------------------------------------------------------------
# Caminhos: tudo relativo à pasta deste script
# --------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DADOS = BASE / "dados"


def arquivos_caso_b():
    """Arquivo completo do caso B, se existir em dados/, ou as partes
    exportadas (a30f0p05_uy_parte1.h5, parte2.h5, ...)."""
    completo = DADOS / "airfoilDNS_a30f0p05.h5"
    if completo.exists():
        return [completo]
    partes = sorted(DADOS.glob("a30f0p05_uy_parte*.h5"))
    if not partes:
        raise FileNotFoundError(
            f"Nenhum arquivo do caso B encontrado em {DADOS}. Esperado "
            "airfoilDNS_a30f0p05.h5 ou a30f0p05_uy_parte*.h5."
        )
    return partes


# --------------------------------------------------------------------------
# Leitura dos dados
# --------------------------------------------------------------------------
def load_field(h5_paths, field="uy", time_key="t_field"):
    """Retorna X (n x nt, pontos nas linhas, tempo nas colunas) e t (nt,).
    Aceita um arquivo ou várias partes consecutivas no tempo."""
    if isinstance(h5_paths, (str, Path)):
        h5_paths = [h5_paths]
    blocos, tempos, grid_shape = [], [], None
    for path in h5_paths:
        with h5py.File(path, "r") as f:
            data = np.asarray(f[field])
            t = np.asarray(f[time_key], dtype=np.float64).squeeze()
        nt = t.size
        axes = [ax for ax, size in enumerate(data.shape) if size == nt]
        if len(axes) != 1:
            raise ValueError(
                f"Não foi possível identificar o eixo do tempo em {path}: "
                f"shape={data.shape}, len(t)={nt}."
            )
        data = np.moveaxis(data, axes[0], -1)
        if grid_shape is None:
            grid_shape = data.shape[:-1]
        elif data.shape[:-1] != grid_shape:
            raise ValueError("As partes têm malhas diferentes.")
        blocos.append(data.reshape(-1, nt))  # mantém o tipo do arquivo
        tempos.append(t)
    # Ordena as partes pelo primeiro instante e preenche X sem cópias extras.
    ordem = np.argsort([tt[0] for tt in tempos])
    t = np.concatenate([tempos[i] for i in ordem])
    if np.any(np.diff(t) <= 0):
        raise ValueError("Instantes repetidos ou fora de ordem entre as partes.")
    X = np.empty((blocos[0].shape[0], t.size), dtype=np.float64)
    col = 0
    for i in ordem:
        nb = blocos[i].shape[1]
        X[:, col:col + nb] = blocos[i]
        col += nb
        blocos[i] = None
    return X, t, grid_shape


def train_fluctuations(X, t, m):
    """Mesmo pré-processamento do TCC (Seção de pré-processamento), com d = 1
    e sem ruído: primeiros m snapshots, 80% para treino, média do treino
    removida."""
    X = X[:, :m]
    t = t[:m]
    m_tr = int(round(0.8 * m))
    Xtr = X[:, :m_tr]
    Xc = Xtr - Xtr.mean(axis=1, keepdims=True)
    return Xc, t[:m_tr]


# --------------------------------------------------------------------------
# DMD de referência (DMD exato)
# --------------------------------------------------------------------------
def exact_dmd(Xc, t, rank):
    """DMD exato (Tu et al., 2014), igual ao calc_dmd.m do banco de dados.
    A SVD de X1 é obtida pelo método dos snapshots (autodecomposição de
    X1^T X1), que evita a matriz U completa e reduz o uso de memória quando
    n >> m. Retorna omega, modos e amplitudes."""
    dt = np.mean(np.diff(t))
    X1, X2 = Xc[:, :-1], Xc[:, 1:]
    G = X1.T @ X1
    s2, V = np.linalg.eigh(G)
    idx = np.argsort(s2)[::-1][:rank]
    s = np.sqrt(np.clip(s2[idx], 0.0, None))
    V = V[:, idx]
    U = X1 @ (V / s)
    Atilde = U.T @ X2 @ (V / s)
    lam, W = np.linalg.eig(Atilde)
    Phi = (X2 @ (V / s)) @ W
    omega = np.log(lam.astype(complex)) / dt
    b = np.linalg.lstsq(Phi, X1[:, 0].astype(complex), rcond=None)[0]
    return omega, Phi, b


def load_ref_mat(path, modes_key, eigs_key, amps_key, eigs_kind, dt):
    """Lê a referência de um .mat (por exemplo, gerado com o calc_dmd.m)."""
    from scipy.io import loadmat

    d = loadmat(path)
    Phi = np.asarray(d[modes_key])
    eigs = np.asarray(d[eigs_key]).squeeze()
    if eigs.ndim == 2:  # matriz diagonal
        eigs = np.diag(eigs)
    if eigs_kind == "discrete":
        omega = np.log(eigs.astype(complex)) / dt
    else:
        omega = eigs.astype(complex)
    if amps_key and amps_key in d:
        b = np.asarray(d[amps_key]).squeeze()
    else:
        b = np.ones(Phi.shape[1])
    return omega, Phi, b


# --------------------------------------------------------------------------
# bopdmdTurbo
# --------------------------------------------------------------------------
def run_turbo(Xc, t, rank, turbo_dir=BASE):
    path = Path(turbo_dir) / "bopdmdTurboV3.py"
    if not path.exists():
        raise FileNotFoundError(f"bopdmdTurboV3.py não encontrado em {turbo_dir}.")
    spec = importlib.util.spec_from_file_location("bopdmdTurboV3", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bopdmdTurboV3"] = mod
    spec.loader.exec_module(mod)
    model = mod.BOPDMD(
        svd_rank=rank,
        num_trials=0,
        varpro_opts_dict={"tol": 1e-6, "maxiter": 200},
    )
    model.fit(Xc, t)
    return np.asarray(model.eigs), np.asarray(model.modes), np.asarray(
        model.amplitudes
    )


# --------------------------------------------------------------------------
# Comparação
# --------------------------------------------------------------------------
def positive(omega, Phi, b):
    """Mantém um modo de cada par: frequência positiva, ordenado por
    importância |b| ||phi||."""
    f = omega.imag / (2 * np.pi)
    keep = f > 0
    imp = np.abs(b) * np.linalg.norm(Phi, axis=0)
    idx = np.where(keep)[0]
    idx = idx[np.argsort(-imp[idx])]
    return f[idx], omega.real[idx], Phi[:, idx], imp[idx]


def similarity(A, B):
    """Matriz |a_i^H b_j| / (||a_i|| ||b_j||)."""
    An = A / np.linalg.norm(A, axis=0)
    Bn = B / np.linalg.norm(B, axis=0)
    return np.abs(An.conj().T @ Bn)


def compare(turbo, ref, n_modes):
    f_t, g_t, P_t, i_t = turbo
    f_r, g_r, P_r, i_r = ref
    n = min(n_modes, f_t.size)
    S = similarity(P_t[:, :n], P_r)
    rows = []
    for k in range(n):
        j_f = int(np.argmin(np.abs(f_r - f_t[k])))
        j_s = int(np.argmax(S[k]))
        rows.append(
            {
                "ordem_turbo": k + 1,
                "f_turbo": f_t[k],
                "re_turbo": g_t[k],
                "imp_rel_turbo": i_t[k] / i_t[0],
                "f_ref_proxima": f_r[j_f],
                "desvio_f_rel": abs(f_r[j_f] - f_t[k]) / f_t[k],
                "sim_freq_proxima": S[k, j_f],
                "ordem_ref_proxima": j_f + 1,
                "f_ref_mais_similar": f_r[j_s],
                "sim_max": S[k, j_s],
                "ordem_ref_mais_similar": j_s + 1,
            }
        )
    return rows


def write_outputs(rows, out_prefix, label):
    path_csv = f"{out_prefix}_{label}.csv"
    with open(path_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    path_tex = f"{out_prefix}_{label}.tex"
    with open(path_tex, "w", encoding="utf-8") as fh:
        fh.write("% Gerado por compare_modes_towne.py\n")
        fh.write("\\begin{tabular}{cccccc}\n\\toprule\n")
        fh.write(
            "Modo & $f$ (\\texttt{bopdmdTurbo}) & $f$ (ref.) & "
            "Desvio de $f$ & $s$ (freq. próxima) & $s_{\\max}$ \\\\\n"
        )
        fh.write("\\midrule\n")
        def br(x, nd):  # separador decimal em português
            return f"{x:.{nd}f}".replace(".", "{,}")

        for r in rows:
            fh.write(
                f"{r['ordem_turbo']} & ${br(r['f_turbo'], 4)}$ & "
                f"${br(r['f_ref_proxima'], 4)}$ & "
                f"${br(100 * r['desvio_f_rel'], 2)}$\\% & "
                f"${br(r['sim_freq_proxima'], 3)}$ & "
                f"${br(r['sim_max'], 3)}$ \\\\\n"
            )
        fh.write("\\bottomrule\n\\end{tabular}\n")
    return path_csv, path_tex


def print_rows(rows, label):
    print(f"\n=== Referência: {label} ===")
    print(
        f"{'k':>2} {'f_turbo':>8} {'f_ref':>8} {'desvio':>8} "
        f"{'s_freq':>7} {'s_max':>6} {'f(s_max)':>9}"
    )
    for r in rows:
        print(
            f"{r['ordem_turbo']:>2} {r['f_turbo']:8.4f} "
            f"{r['f_ref_proxima']:8.4f} {100 * r['desvio_f_rel']:7.2f}% "
            f"{r['sim_freq_proxima']:7.3f} {r['sim_max']:6.3f} "
            f"{r['f_ref_mais_similar']:9.4f}"
        )


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", nargs="+", default=None,
                   help="arquivos de dados (padrão: os do caso B na pasta dados/)")
    p.add_argument("--field", default="uy")
    p.add_argument("--turbo-dir", default=str(BASE),
                   help="pasta que contém bopdmdTurboV3.py (padrão: a deste script)")
    p.add_argument("--m", type=int, default=400,
                   help="snapshots do caso do bopdmdTurbo (B300 ou B400)")
    p.add_argument("--rank", type=int, default=20,
                   help="posto do bopdmdTurbo")
    p.add_argument("--ref-m", type=int, default=None,
                   help="snapshots da referência (padrão: todos os disponíveis; "
                        "a referência usa 100%% deles, sem divisão treino/teste)")
    p.add_argument("--ref-ranks", type=int, nargs="+", default=[10, 20, 40],
                   help="postos da referência DMD exato")
    p.add_argument("--ref-mat", default=None,
                   help=".mat com a referência (ex.: saída do calc_dmd.m)")
    p.add_argument("--mat-modes", default="Phi")
    p.add_argument("--mat-eigs", default="lambda")
    p.add_argument("--mat-amps", default="b")
    p.add_argument("--mat-eigs-kind", choices=["discrete", "continuous"],
                   default="discrete")
    p.add_argument("--n-modes", type=int, default=6,
                   help="número de modos do bopdmdTurbo comparados")
    p.add_argument("--out", default=None,
                   help="prefixo dos arquivos de saída (padrão: resultados/comparacao_B<m>)")
    a = p.parse_args()

    h5 = a.h5 or arquivos_caso_b()
    if a.out is None:
        (BASE / "resultados").mkdir(exist_ok=True)
        a.out = str(BASE / "resultados" / f"comparacao_B{a.m}")
    X, t, grid = load_field(h5, a.field)
    dt = float(np.mean(np.diff(t)))
    print(f"Dados: {X.shape[0]} pontos (malha {grid}), {X.shape[1]} snapshots, "
          f"dt = {dt:.4g}")

    # bopdmdTurbo, mesmo pré-processamento do caso B do TCC.
    Xc, tc = train_fluctuations(X, t, a.m)
    om_t, P_t, b_t = run_turbo(Xc, tc, a.rank, a.turbo_dir)
    turbo = positive(om_t, P_t, b_t)
    T = tc[-1] - tc[0] + dt
    print(f"bopdmdTurbo: m = {a.m}, m_tr = {Xc.shape[1]}, r = {a.rank}, "
          f"T = {T:.1f}, tolerância de frequência 1/(2T) = {1 / (2 * T):.4f}")

    # Referência(s).
    refs = []
    if a.ref_mat:
        om_r, P_r, b_r = load_ref_mat(a.ref_mat, a.mat_modes, a.mat_eigs,
                                      a.mat_amps, a.mat_eigs_kind, dt)
        if P_r.shape[0] != X.shape[0]:
            raise ValueError("Os modos do .mat não têm o mesmo número de "
                             "pontos que os dados.")
        refs.append(("calc_dmd", (om_r, P_r, b_r)))
    else:
        ref_m = a.ref_m or X.shape[1]
        del Xc
        Xr = X[:, :ref_m]
        Xr -= Xr.mean(axis=1, keepdims=True)  # in place, para poupar memória
        for r in a.ref_ranks:
            refs.append((f"dmd_exato_m{ref_m}_r{r}",
                         exact_dmd(Xr, t[:ref_m], r)))

    for label, ref in refs:
        rows = compare(turbo, positive(*ref), a.n_modes)
        print_rows(rows, label)
        c, x = write_outputs(rows, a.out, label)
        print(f"Gravados: {c}, {x}")


if __name__ == "__main__":
    main()
