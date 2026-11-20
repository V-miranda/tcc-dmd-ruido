# Análise de diferentes métodos DMD com dados ruidosos

Códigos do Trabalho de Conclusão de Curso de **Vinicius Miranda Rodrigues** (Engenharia Aeroespacial, UFSC – Centro Tecnológico de Joinville), com orientação do Prof. Dr. Juan Pablo de Lima Costa Salazar e coorientação do Prof. Dr. Filipe Dutra da Silva. A defesa está marcada para 27/11/2026.

O trabalho compara o DMD original (Schmid, 2010) e o BOP-DMD do [PyDMD](https://github.com/PyDMD/PyDMD) com dados ruidosos. Ele também propõe o **bopdmdTurbo**, uma modificação do BOP-DMD que reduz o uso de memória RAM no ajuste de dados de grande dimensão.

## O bopdmdTurbo

O BOP-DMD do PyDMD resolve o problema do opt-DMD por projeção de variáveis com o método de Levenberg-Marquardt. A cada iteração, ele forma o Jacobiano do resíduo, de $m_s n_s \times r$ elementos complexos, e faz sua fatoração QR. Sem projeção nos modos POD (`use_proj=False`), essas matrizes chegam a dezenas de GB. O `bopdmdTurboV3.py` mantém a formulação do VarPro, a inicialização, o *bagging* e a interface da classe `BOPDMD`, mas:

- troca o Levenberg-Marquardt pelo **L-BFGS-B** (SciPy);
- calcula o gradiente por **diferenciação automática** no modo reverso (JAX, `float64`/`complex128`, compilado com `jit`), sem formar o Jacobiano;
- aplica **regularização de Tikhonov adaptativa** ao subproblema linear, com $\lambda_{ef} = \lambda \|\Phi\|_F^2$;
- resolve o subproblema linear por **fatoração de Cholesky**;
- impõe as restrições `stable`, `imag` e `limited` como **limites** do L-BFGS-B. `conjugate_pairs` e funções do usuário são aplicadas só à estimativa inicial e ao resultado final;
- acrescenta os parâmetros `m_history` (memória do L-BFGS-B, padrão 20) e `compression` (seleção de *snapshots* por QR com pivoteamento, como no VarProDMD).

Os parâmetros do Levenberg-Marquardt (`init_lambda`, `maxlam`, `lamup`, `use_levmarq`, `eps_stall`, `use_fulljac`) continuam aceitos por compatibilidade, mas são ignorados.

### Uso

O módulo substitui diretamente a classe do PyDMD:

```python
# from pydmd import BOPDMD
from bopdmdTurboV3 import BOPDMD

dmd = BOPDMD(
    svd_rank=20,
    num_trials=0,            # 0 = sem bagging (um único ajuste opt-DMD)
    use_proj=False,          # ajuste sobre os dados completos
    varpro_opts_dict={"tol": 1e-6, "maxiter": 200, "m_history": 20},
)
dmd.fit(X, t)                # X: (n_espaço, n_tempo), t: instantes dos snapshots
```

Mantenha `jax_debug_nans` desativado (o módulo já faz isso). Com ele ativo, o primeiro NaN levanta uma exceção antes de a regularização ser ajustada.

## Conteúdo do repositório

| Arquivo | Função |
|---|---|
| `bopdmdTurboV3.py` | Implementação do bopdmdTurbo (derivada de `pydmd/bopdmd.py`) |
| `benchmark_dmd_simple.py` | *Benchmark* DMD × BOP-DMD × bopdmdTurbo com projeção POD (`use_proj=True`); saída em `bench_out/` |
| `benchmark_dmd_simple_noproj.py` | O mesmo *benchmark* sem projeção (`use_proj=False`); saída em `bench_out_noproj/` |
| `plot_benchmark_sweeps.py` | Figuras das varreduras (tempo, memória, erro de treino e de previsão) a partir do `results_summary.csv` |
| `compare_modes_towne.py` | Compara os modos do bopdmdTurbo (caso B) com um DMD exato de referência, por frequência e similaridade de forma |
| `figura_modos_ref.py` | Figura dos quatro modos principais do caso B400: referência × bopdmdTurbo |
| `plot_modal_spectrum.py` | Espectro modal (taxa de crescimento, autovalores discretos e amplitudes) a partir de um `modes_table.csv` |
| `Gui_FWH_SU2.py` | Analogia de Ffowcs Williams e Hawkings (formulação sólida), acelerada com Numba |
| `main_multiple.py` | Propaga a pressão de parede reconstruída do jato até o observador com o `Gui_FWH_SU2.py` |
| `fft_plotter.py` | Densidade espectral pelo método de Welch e SPL em função do número de Strouhal |

## Dados

**Aerofólio em arfagem.** Conjunto *Low-Reynolds-number pitching airfoil direct numerical simulations* (Towne et al., 2023), disponível no [Deep Blue Data](https://deepblue.lib.umich.edu/data) da Universidade de Michigan. Os dados não estão no repositório. Os casos usados são:

- `airfoilDNS_a25f0p1.h5` ($\alpha_0 = 25^\circ$, $f_P = 0{,}1$): *benchmark* e caso A da análise modal;
- `airfoilDNS_a30f0p05.h5` ($\alpha_0 = 30^\circ$, $f_P = 0{,}05$): caso B da análise modal;
- `airfoilDNS_grid.h5`: malha, usada nas figuras dos modos.

Os scripts de *benchmark* leem o arquivo indicado em `FILE` (padrão `airfoilDNS/airfoilDNS_a25f0p1.h5`). Os scripts da análise modal esperam os arquivos em `dados/` e gravam em `resultados/`.

**Jato instalado.** A pressão de parede da placa plana próxima ao jato vem de uma simulação LES-LBM do grupo (Da Silva et al., 2025). Esses dados não são públicos. Os caminhos em `main_multiple.py` e `fft_plotter.py` apontam para o cluster onde as simulações foram feitas e precisam ser ajustados para outro ambiente.

## Requisitos

Python 3.10 ou superior (o `main_multiple.py` usa f-strings no formato do 3.12). Pacotes:

```bash
pip install numpy scipy pydmd jax dask h5py pandas matplotlib numba
pip install psutil   # apenas no Windows, para medir o pico de memória
```

Os resultados do TCC foram obtidos em CPU, em um nó com dois Intel Xeon Gold 5318Y (48 núcleos) e 251 GB de RAM, sob Linux. O BOP-DMD do PyDMD sem projeção chega a cerca de 107 GB de pico nos casos do *benchmark*.

## Como reproduzir

### *Benchmark*

Ajuste o bloco `SETTINGS` no início do script (arquivo, campo, caso de referência, varreduras, repetições) e execute:

```bash
python benchmark_dmd_simple.py          # com projeção POD
python benchmark_dmd_simple_noproj.py   # sem projeção
python plot_benchmark_sweeps.py bench_out/results_summary.csv bench_out/
```

Configuração padrão:

- caso de referência $r = 20$, $m = 300$, $\eta = 0$, $d = 1$;
- uma varredura por parâmetro: $r \in \{10, 20, 40\}$, $m \in \{200, 300, 400\}$, $\eta \in \{0;\ 0{,}05;\ 0{,}2\}$ e $d \in \{1, 2, 4\}$;
- três repetições por caso, com a mediana reportada;
- 80% dos *snapshots* para treino e 20% para a previsão;
- `tol = 1e-6`, `maxiter = 200` e `num_trials = 0` nos dois métodos de otimização.

Cada ajuste roda em um processo novo, depois de um ajuste de aquecimento não cronometrado com 2 iterações, para que a compilação do JAX e a inicialização das bibliotecas não entrem no tempo medido.

Saídas: `results_all_runs.csv` (todas as repetições), `results_summary.csv` (medianas) e figuras `sweep_*.png`/`.pdf`.

### Análise modal (caso B)

```bash
python compare_modes_towne.py --m 400 --ref-ranks 20 100
python compare_modes_towne.py --m 300 --ref-ranks 20 100
python figura_modos_ref.py
```

`compare_modes_towne.py --help` lista as demais opções. Entre elas estão a leitura da referência a partir de um `.mat` gerado pelo `calc_dmd.m` do banco de dados (`--ref-mat`) e o número de modos comparados (`--n-modes`).

### Ruído do jato

1. Ajuste o DMD, o BOP-DMD ou o bopdmdTurbo à pressão de parede e grave a reconstrução em HDF5 (conjunto `Pressure` ou `reconstructed_data`).
2. Edite os caminhos, o `casename` e o `dt` em `main_multiple.py` e execute o script para obter o sinal acústico no observador.
3. Use `fft_plotter.py` para calcular e comparar os espectros.

## Principais resultados

- Com projeção POD, os dois métodos de otimização têm erros de reconstrução e de previsão inferiores aos do DMD original em todos os níveis de ruído, e o pico de memória é dominado pelos dados.
- Sem projeção, o pico de memória do bopdmdTurbo é de 8 a 19 vezes menor que o do BOP-DMD (4,2–6,6 GB contra 42–107 GB). O tempo de ajuste é de 16 a 160 vezes menor.
- Com atraso ($d = 2$ e $d = 4$), o BOP-DMD sem projeção falha com `OverflowError` por causa do tamanho do Jacobiano, e o bopdmdTurbo conclui o ajuste.
- O preço é um erro de previsão maior nos casos de janela curta, posto alto e ruído elevado, com taxas de crescimento menos precisas.
- No jato, o bopdmdTurbo chega a 1500 modos, contra 100 do BOP-DMD. É o método cujo espectro de campo distante mais se aproxima do obtido com a pressão original.

Os detalhes estão no texto do TCC.

## Referências principais

- ASKHAM, T.; KUTZ, J. N. Variable projection methods for an optimized dynamic mode decomposition. *SIAM Journal on Applied Dynamical Systems*, v. 17, 2018.
- SASHIDHAR, D.; KUTZ, J. N. Bagging, optimized dynamic mode decomposition for robust, stable forecasting with spatial and temporal uncertainty quantification. *Philosophical Transactions of the Royal Society A*, 2022.
- SCHMID, P. J. Dynamic mode decomposition of numerical and experimental data. *Journal of Fluid Mechanics*, v. 656, 2010.
- TOWNE, A. et al. A database for reduced-complexity modeling of fluid flows. *AIAA Journal*, 2023.
- DEMO, N.; TEZZELE, M.; ROZZA, G. PyDMD: Python Dynamic Mode Decomposition. *Journal of Open Source Software*, 2018.
- ICHINAGA, S. M. et al. PyDMD: a Python package for robust dynamic mode decomposition. *Journal of Machine Learning Research*, 2024.

## Uso de inteligência artificial

O Claude Opus 5.5 (Anthropic) ajudou a gerar e revisar parte do código. Todo o código foi revisado pelo autor e verificado com a suíte de testes `tests/test_bopdmd.py` do PyDMD e pela comparação com o PyDMD original e com a literatura.

## Licença

O `bopdmdTurboV3.py` é derivado do PyDMD, distribuído sob a licença MIT, e mantém essa origem. Os demais arquivos ainda não têm licença definida.
