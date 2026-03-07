# Static Density-Density Response of the 3D Homogeneous Electron Gas via DMC

This repository contains analysis code for computing the **static density-density response function** $\chi(q)$ of the three-dimensional **homogeneous electron gas (3D HEG)** using **Diffusion Monte Carlo (DMC)** as implemented in [QMCPACK](https://qmcpack.org). Results are compared against the Moroni *et al.* (1995) parametrization and other reference models.

---

## Table of Contents

1. [Physics Background](#physics-background)
2. [Method](#method)
3. [Finite-Size Correction](#finite-size-correction)
4. [Repository Structure](#repository-structure)
5. [Installation](#installation)
6. [Data Requirements](#data-requirements)
7. [Usage](#usage)
8. [Module Reference](#module-reference)
9. [References](#references)

---

## Physics Background

### The Homogeneous Electron Gas

The **homogeneous electron gas** (HEG, also called *uniform electron gas* or *jellium*) is a model system of $N_e$ interacting electrons in a uniform positive background (neutralizing "jellium"). It is the foundational model of density functional theory (DFT) and a key benchmark for quantum many-body methods.

The single parameter controlling the physics is the **Wigner-Seitz radius**:

$$r_s = \left(\frac{3}{4\pi n_0}\right)^{1/3}$$

where $n_0 = N_e / L^3$ is the average electron density. Small $r_s$ (high density) means weakly correlated electrons; large $r_s$ (low density) means strongly correlated. All calculations use **atomic (Hartree) units**: $\hbar = m_e = e = 1$.

**Fermi wavevector**: $k_F = (9\pi/4)^{1/3} / r_s$.

### Static Density-Density Response Function

The **static density-density response function** $\chi(q)$ describes how the electron density responds to a static external perturbation of wavevector $\mathbf{q}$:

$$\delta n(\mathbf{q}) = \chi(q) \, v_{\text{ext}}(\mathbf{q})$$

It encodes the full exchange-correlation physics of the interacting system. Key properties:

- $\chi(q) < 0$ for all $q$ (density increases where potential is attractive)
- $\chi(q) \to -n_0 / (4\pi/q^2)$ as $q \to 0$ (compressibility sum rule)
- $\chi(q) \to \chi_0(q)$ as $q \to \infty$ (high-$q$ limit, response approaches non-interacting)
- The ratio $S(q) = -\chi(q) / (\pi n_0)$ is related to the static structure factor

### Non-Interacting Lindhard Function

The non-interacting response is given analytically by the **Lindhard function**:

$$\chi_0(q) = -\frac{k_F}{2\pi^2} \left[ \frac{1}{2} + \frac{1 - x^2}{4x} \ln\left|\frac{1+x}{1-x}\right| \right], \quad x = \frac{q}{2k_F}$$

### Local Field Factor and Reference Models

The full interacting $\chi(q)$ is related to $\chi_0(q)$ through the **local field factor** $G(q)$:

$$\chi(q)^{-1} = \chi_0(q)^{-1} - v_c(q) [1 - G(q)]$$

where $v_c(q) = 4\pi/q^2$ is the Coulomb interaction. Setting $G = 0$ gives the **Random Phase Approximation (RPA)**. The key reference models used here are:

- **RPA**: $G(q) = 0$
- **Moroni *et al.*** (1995): parametrized $G(q; r_s)$ fitted to QMC data at $r_s = 1, 2, 5, 10$
- **Corradini *et al.*** (1998): alternative parametrization of the exchange-correlation kernel $f_{xc}(q; r_s)$

---

## Method

### Extracting $\chi(q)$ from Total Energies

The response function is extracted by applying a static **cosine perturbation** to the full Hamiltonian:

$$H(v_\mathbf{q}) = H_0 + 2 v_\mathbf{q} \sum_i \cos(\mathbf{q} \cdot \mathbf{r}_i)$$

For small perturbation amplitude $v_\mathbf{q}$, the total energy per electron expands as:

$$\frac{E(v_\mathbf{q})}{N_e} = \frac{E_0}{N_e} + \frac{\chi(q)}{n_0} v_\mathbf{q}^2 + O(v_\mathbf{q}^4)$$

Therefore, **$\chi(q)/n_0$ is the quadratic coefficient** of $E(v_\mathbf{q})$. In practice:

1. Run QMCPACK DMC for several $v_\mathbf{q}$ values (including $v_\mathbf{q} = 0$)
2. Fit $E(v_\mathbf{q})$ to a **quadratic** or **quartic** polynomial: $E_0 + A v_\mathbf{q}^2 + B v_\mathbf{q}^4$
3. Extract $\chi(q) = A \cdot n_0$

### Trial Wavefunction

The DMC calculation uses a Slater-Jastrow trial wavefunction perturbed at wavevector $\mathbf{q}$:

$$\Psi_T^v(\{\mathbf{r}_i\}) = D_\uparrow^v D_\downarrow^v \cdot \exp\!\left(-\sum_{i<j} u(r_{ij})\right)$$

where:
- $D_s^v$: Slater determinant from the single-particle Hamiltonian $H_{\text{sp}} + 2\alpha \cos(\mathbf{q} \cdot \mathbf{r})$ with **$\alpha \neq v_\mathbf{q}$** in general
- $u(r)$: Slater-Jastrow correlation factor optimized with VMC
- $\alpha$: optimal perturbation for the trial state, found by minimizing the VMC energy; computed as $\alpha^*(v_\mathbf{q}) = v_\mathbf{q} \cdot f(q/k_F)$ where $f$ interpolates between the low-$q$ (RPA) and high-$q$ (non-interacting) limits via a Fermi-function form

A **long-range Jastrow** correction has been implemented to reduce variance, especially at high $r_s$.

### $q$-point indexing

Wavevectors are indexed as integer triples $(n_x, n_y, n_z)$ with $\mathbf{q} = (2\pi/L)(n_x, n_y, n_z)$. Only the unique shells (distinct $|\mathbf{q}|$) are computed. The function `gen_qidx(n_shell)` generates all unique shells up to shell index `n_shell`.

---

## Finite-Size Correction

DMC energies are computed in a simulation cell of finite size $N_e$. The finite-size correction to $\chi$ uses the known analytic form of the non-interacting finite-size error:

$$\chi^{-1}(q; \infty) = \chi^{-1}(q; N_e) + \left[\chi_0^{-1}(q; \infty) - \chi_0^{-1}(q; N_e)\right]$$

where $\chi_0(q; N_e)$ is the finite-system Lindhard function (sum over discrete $k$-grid), computed using `anal_chi02`. The DFT single-particle energies at finite $v_\mathbf{q}$ provide $\chi_0(q; N_e)$ via the same quadratic-fit approach.

---

## Repository Structure

```
pp/
├── README.md              # This file
├── low-q.ipynb            # Main analysis notebook (chi(q) for all rs, figures)
│
├── io_utils.py            # File I/O, directory parsing, E(vq) energy caching
├── physics.py             # Analytical formulas: gas params, Lindhard, LFF models, FS correction
├── fitting.py             # Fitting pipeline: E(vq) → chi(q), bootstrap errors, get_chi
├── plotting.py            # Visualization: plot_chi, plot_E_of_vq, plot_variance
│
├── output/                # Cached E_all matrices (auto-generated)
│   ├── E_all_rs2.0-n162.npz
│   ├── E_all_rs5.0-n54.npz
│   ├── E_all_rs10.0-n162.npz
│   ├── E_all_rs15.0-n162.npz
│   ├── E_all_rs20.0-n162.npz
│   └── E_all_rs25.0-n162.npz
│
├── Figures/               # Output PDF/PNG figures (git-ignored)
├── context.md             # Extended physics notes and code documentation
├── requirements.txt       # Python package requirements (pip)
└── environment.yml        # Conda environment spec
```

### Module overview

| File | Contents |
|------|----------|
| `io_utils.py` | `get_energy`, `get_energy_pwscf`, `collect_q_and_vq`, `get_E_all`, `load_or_compute_E`, `load_raw_blocks`, `get_variance_for_run` |
| `physics.py` | `get_gas_params`, `get_qs`, `gen_qidx`, `chi0q`, `anal_chi02`, `guess_alpha2`, `FS_correct`, `get_chi_Moroni`, `get_chi_RPA`, `get_chi_corradini`, `G_Moroni`, `corradini_pz` |
| `fitting.py` | `get_chi_q`, `get_chi0_q`, `get_correction`, `get_chi`, `bootstrap_chi_error`, `analyze_vq_range`, `fit_E_of_vq`, `fit_quality_report` |
| `plotting.py` | `plot_chi`, `plot_E_of_vq`, `plot_variance` |

---

## Installation

### Prerequisites

- Python ≥ 3.10
- A QMCPACK run tree (see [Data Requirements](#data-requirements))
- [`qharv`](https://github.com/Paul-St-Young/harvest_qmcpack) — QMCPACK output parser

### Option A: pip (venv)

```bash
git clone git@github.com:moegunes/static_response-3DHEG.git
cd static_response-3DHEG

python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

pip install -r requirements.txt

# qharv is not on PyPI — install from source:
pip install git+https://github.com/Paul-St-Young/harvest_qmcpack.git
```

### Option B: conda

```bash
git clone git@github.com:moegunes/static_response-3DHEG.git
cd static_response-3DHEG

conda env create -f environment.yml
conda activate heg-response
```

### Launch the notebook

```bash
source .venv/bin/activate   # or: conda activate heg-response
jupyter lab low-q.ipynb
```

---

## Data Requirements

The analysis reads QMCPACK output files from a specific directory tree. The expected layout is:

```
{main_dir}/rs{rs:.1f}-n{Ne}/qv{qx}_{qy}_{qz}-vq{vq:.4f}/{dft_func}-e{ecut}-qa{alpha:.3f}-thr1.0d-{thr}/{wf}-t{tau*Ne*rs}-ts{ts:.4f}-nw{nw}/qmc.s002.stat.h5
```

**Example:**
```
runs/rs5.0-n54/qv1_0_0-vq0.0020/ni-e125-qa0.172-thr1.0d-10/sj-t4218.75-ts0.2500-nw1024/qmc.s002.stat.h5
```

The run directories used in the paper are on the Flatiron Institute cluster (`/mnt/ceph/users/mgunes/HEG_dmc/static_response/`). Raw QMC data is not included in this repository due to size, but the **cached energy matrices** in `output/` allow reproducing all figures without re-running QMCPACK.

### Datasets in `output/`

| File | $r_s$ | $N_e$ | Source directory |
|------|-------|-------|-----------------|
| `E_all_rs2.0-n162.npz` | 2 | 162 | `v4.0/runs` |
| `E_all_rs5.0-n54.npz` | 5 | 54 | `lr_jastrow_full/runs` |
| `E_all_rs10.0-n162.npz` | 10 | 162 | `v4.0/runs` |
| `E_all_rs15.0-n162.npz` | 15 | 162 | `v4.00/runs` |
| `E_all_rs20.0-n162.npz` | 20 | 162 | `v4.00/runs` |
| `E_all_rs25.0-n162.npz` | 25 | 162 | `v4.00/runs` |

Each `.npz` file contains:
- `E_all`: 2D array `(n_q, n_vq)` — DMC total energies (Ha/electron)
- `dE_all`: 2D array `(n_q, n_vq)` — statistical errors
- `E_dft`: 2D array `(n_q, n_vq)` — DFT (Quantum Espresso) energies for FS correction
- `dE_dft`: corresponding DFT errors
- `qidx_list`: list of q-index triples `[nx, ny, nz]`
- `vq_list`: array of perturbation amplitudes
- `main_dir`: path to the source run directory (for cache staleness check)

---

## Usage

### Quick start — reproduce figures

Open `low-q.ipynb` in Jupyter and run cells corresponding to the desired $r_s$. Each cell loads the cached energy matrix from `output/` and produces a $\chi(q)$ plot.

### Computing $\chi(q)$ for a given $(r_s, N_e)$

```python
from io_utils import collect_q_and_vq
from physics import gen_qidx
from fitting import get_chi
from plotting import plot_chi
import matplotlib.pyplot as plt

main_dir = "/path/to/qmcpack/runs"
rs, Ne = 5, 54

# Discover available q-points and vq values from the run directory
qidx_list, vq_list = collect_q_and_vq(main_dir, rs, Ne)

# (Optional) restrict to specific shells / vq range
# qidx_list = gen_qidx(4)
# vq_list = [0.0, 0.004, 0.008, 0.012]

# Run full pipeline: E(vq) fit → chi → FS correction → bootstrap errors
chi, dchi, fit_quality = get_chi(
    main_dir, vq_list, qidx_list, rs, Ne,
    dft_func='ni',        # 'ni' = non-interacting DFT for FS correction
    vq_fit='quadratic',   # 'quadratic' or 'quartic'
    verbose=True,
    n_boot=800,
)

# Plot
fig, ax = plt.subplots(dpi=200)
plot_chi(qidx_list, chi, dchi, rs, Ne, chi_ref='both', ax=ax)
fig.savefig('chi_rs5.pdf', format='pdf', bbox_inches='tight')
```

### Using the energy cache

The first call to `get_chi` (or `load_or_compute_E`) for a given $(r_s, N_e)$ will read all available HDF5 files and write a cache to `./output/E_all_rs{rs:.1f}-n{Ne:d}.npz`. Subsequent calls load from cache (~milliseconds vs. minutes for full I/O).

```python
from io_utils import load_or_compute_E

E_all, dE_all, E_dft, dE_dft, qidx_list, vq_list = load_or_compute_E(
    main_dir, rs=10, Ne=162,
    vq_subset=[0, 0.001, 0.002, 0.003],   # optional subset
    qidx_subset=gen_qidx(5),              # optional subset
)
```

---

## Module Reference

### `io_utils.py`

```
get_energy(h5_file)                      → (E, dE)
get_energy_pwscf(xml_path)               → (E, dE)
collect_q_and_vq(main_dir, rs, Ne)       → (qidx_list, vq_list)
get_E_all(main_dir, rs, Ne)              → (E_all, dE_all, E_dft, dE_dft, qidx_list, vq_list)
load_or_compute_E(main_dir, rs, Ne, ...) → (E_all, dE_all, E_dft, dE_dft, qidx_list, vq_list)
get_variance_for_run(...)                → (var, dvar)
load_raw_blocks(h5_file, ...)            → blocks array
```

### `physics.py`

```
get_gas_params(rs, Ne)             → (kF, n0, NF, L)
get_qs(qidxl, Ne, rs)              → array of |q|
gen_qidx(n_shell)                  → list of [nx,ny,nz] shells
chi0q(q, Ne, rs)                   → Lindhard χ₀(q) [thermodynamic limit]
anal_chi02(rs, Ne, qidx_list)      → χ₀(q, N) on finite k-grid
guess_alpha2(rs, Ne, qidx)         → optimal α for trial wavefunction
FS_correct(chi, correction, ...)   → finite-size corrected χ
get_chi_Moroni(rs, Ne, q)          → Moroni reference χ(q)
get_chi_RPA(rs, Ne, q)             → RPA χ(q)
get_chi_corradini(rs, Ne, q)       → Corradini reference χ(q)
G_Moroni(rs, q)                    → local field factor G(q)
corradini_pz(rs, q)                → fxc parametrization
```

### `fitting.py`

```
fit_E_of_vq(E, dE, vq, func)                     → (popt, pcov)
fit_quality_report(popt, pcov, E, dE, vq, func)   → dict (SNR, chi2r, ...)
get_chi_q(main_dir, Ne, rs, vq_list, qidxl, ...)  → (chi, dchi, fit_quality)
get_chi0_q(main_dir, Ne, rs, vq_list, qidxl, ...) → (chi0, dchi0, fit_quality)
get_correction(main_dir, qidxl, rs, Ne, ...)      → FS correction array
get_chi(main_dir, vql, qidxl, rs, Ne, ...)        → (chi, dchi, fit_quality)
bootstrap_chi_error(fit_quality, vq, n0, fs_fn)   → (boot_err, boot_samples)
analyze_vq_range(rs, Ne, qidx_list, vq_max, ...)  → list of per-q analysis dicts
```

### `plotting.py`

```
plot_chi(qidxl, chi, dchi, rs, Ne, ...)   → ax
plot_E_of_vq(q, vq_list, rs, Ne, ...)    → ax
plot_variance(main_dir, rs, Ne, ...)      → (ax, x, var, dvar)
```

---

## References

1. **Moroni, S., Ceperley, D. M., & Senatore, G.** (1995).
   *Static response from quantum Monte Carlo calculations.*
   **Phys. Rev. Lett. 75**, 689.
   https://doi.org/10.1103/PhysRevLett.75.689

2. **Corradini, M., Del Sole, R., Onida, G., & Palummo, M.** (1998).
   *Analytical expressions for the local-field factor G(q) and the exchange-correlation kernel Kxc(r) of the homogeneous electron gas.*
   **Phys. Rev. B 57**, 14569.
   https://doi.org/10.1103/PhysRevB.57.14569

3. **Kim, J. et al.** (2018).
   *QMCPACK: an open source ab initio quantum Monte Carlo package for the electronic structure of atoms, molecules and solids.*
   **J. Phys.: Condens. Matter 30**, 195901.
   https://doi.org/10.1088/1361-648X/aab9c3

4. **Giannozzi, P. et al.** (2009).
   *QUANTUM ESPRESSO: a modular and open-source software project for quantum simulations of materials.*
   **J. Phys.: Condens. Matter 21**, 395502.
   https://doi.org/10.1088/0953-8984/21/39/395502

5. **Ceperley, D. M., & Alder, B. J.** (1980).
   *Ground State of the Electron Gas by a Stochastic Method.*
   **Phys. Rev. Lett. 45**, 566.
   https://doi.org/10.1103/PhysRevLett.45.566

6. **Young, P.** `harvest_qmcpack` (qharv): Python tools for harvesting QMCPACK output.
   https://github.com/Paul-St-Young/harvest_qmcpack

---

## Citation

If you use this code or data in your work, please cite this repository and the relevant physics references above.

```bibtex
@misc{gunes2026heg_response,
  author       = {Güneş, M.},
  title        = {Static density-density response of the 3D HEG via DMC},
  year         = {2026},
  url          = {https://github.com/moegunes/static_response-3DHEG},
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
