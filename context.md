# Context: Static Response of the 3D Homogeneous Electron Gas via DMC

## Physics Overview

This project computes the **static density-density response function** $\chi(q)$ of the three-dimensional homogeneous electron gas (3D HEG) using **Diffusion Monte Carlo (DMC)** as implemented in QMCPACK. The goal is to benchmark and extend the classic results of Moroni *et al.* (1995) to higher densities ($r_s > 10$) using modern QMC techniques (backflow, larger systems, twist averaging).

### Key Physical Quantities

- **$r_s$** (Wigner-Seitz radius): inverse density parameter. $r_s = (3/(4\pi n_0))^{1/3}$. Low $r_s$ = high density (weakly correlated), high $r_s$ = low density (strongly correlated).
- **$N_e$**: number of electrons in the simulation cell (typically 54, 162, 294, 406).
- **$\chi(q)$**: static density response function — tells how the electron density responds to an external perturbation of wavevector $q$.
- **$\chi_0(q)$**: Lindhard function — the *non-interacting* response (analytically known).
- **$k_F$**: Fermi wavevector. $k_F = (9\pi/4)^{1/3}/r_s$.
- **$n_0 = N_e / L^3$**: average electron density.
- **$G(q)$**: local field factor — encodes exchange-correlation effects beyond RPA.

### Method: Extracting $\chi(q)$ from Total Energies

The response function is obtained by applying a cosine perturbation to the Hamiltonian:

$$H' = H + v_{\text{ext}}(\mathbf{r}), \quad v_{\text{ext}}(\mathbf{r}) = 2v_\mathbf{q} \cos(\mathbf{q} \cdot \mathbf{r})$$

The total energy of the perturbed system expands as:

$$E(v_\mathbf{q}) = E_0 + \frac{\chi(q)}{n_0} v_\mathbf{q}^2 + O(v_\mathbf{q}^4)$$

So $\chi(q)/n_0$ is the **curvature** of $E(v_q)$ at $v_q = 0$. In practice:

1. Run QMCPACK for several values of $v_q$ (including $v_q = 0$).
2. Fit $E(v_q)$ to a quartic: $E_0 + A v_q^2 + B v_q^4$.
3. Extract $\chi(q) = A \cdot n_0$.

### Trial Wavefunction

$$\Psi_T^v = D_\uparrow^v D_\downarrow^v \prod_{i<j} e^{-u(r_{ij})}$$

- **$D_s^v$**: Slater determinant from a non-interacting system perturbed by $v_{\text{ext}}(\mathbf{r}) = 2\alpha \cos(\mathbf{q} \cdot \mathbf{r})$.
- **$u(r)$**: Jastrow factor optimized via VMC.
- **$\alpha$**: perturbation amplitude for the trial wavefunction. Not necessarily equal to $v_q$; an optimal $\alpha^*(v_q)$ is found by minimizing VMC/DMC energy. Its ratio $\alpha^*/v_q$ follows a $\tanh(q/k_F)$-like interpolation.
- A **long-range Jastrow** factor has been implemented to reduce variance and smooth $\chi(q)$.

### Finite-Size Corrections

$$\chi^{-1}(q, \infty) = \chi^{-1}(q, N) + [\chi_0^{-1}(q, \infty) - \chi_0^{-1}(q, N)]$$

This corrects the finite-size interacting $\chi$ using the known finite-size error of the non-interacting $\chi_0$.

### Reference Models

- **RPA**: $\chi_{\text{RPA}} = \chi_0 / (1 - \chi_0 V_c)$ where $V_c = 4\pi/q^2$.
- **Moroni *et al.***: Uses a fitted local field factor $G(q)$ to obtain $\chi(q)$ via $f_{xc} = -V_c G(q)$.
- **Corradini *et al.***: Alternative parametrization of $G(q)$.

---

## Code Structure

### Workspace Layout

```
pp/
├── low-q.ipynb          # Main analysis notebook
├── utils.py             # All utility functions
├── context.md           # This file
├── Figures/             # Output plots
└── vscode/              # Virtual environment (sympy, jupyter_core)
```

### Key Functions in `utils.py`

| Function | Purpose |
|----------|---------|
| `get_energy(h5_file)` | Extract DMC energy ± error from QMCPACK `.stat.h5` file |
| `get_energy_pwscf(path)` | Extract DFT band energy from Quantum Espresso XML |
| `get_chi_q(main_dir, Ne, rs, vq_list, qidx_list, ...)` | **Core**: fits $E(v_q)$ to quartic, returns $\chi(q)$ and $d\chi(q)$ |
| `get_chi0_q(...)` | Same but for non-interacting (DFT) energies |
| `chi0q(q, Ne, rs)` | Analytical Lindhard function (thermodynamic limit) |
| `anal_chi02(rs, Ne, qidx_list, n_shell)` | Analytical $\chi_0(q, N)$ on a finite k-grid |
| `get_gas_params(rs, Ne)` | Returns $k_F, n_0, N_F, L$ |
| `get_qs(qidx, Ne, rs)` | Converts q-indices to physical $|q|$ values |
| `gen_qidx(mq)` | Generate unique q-index shells up to max index `mq` |
| `guess_alpha2(rs, Ne, qidx)` | Compute optimal $\alpha$ via Fermi-function interpolation |
| `collect_q_and_vq(runs_path, rs, n)` | Parse directory tree for available q-points and $v_q$ values |
| `G_Moroni(rs, q)` | Moroni local field factor parametrization |
| `corradini_pz(rs, q)` | Corradini $f_{xc}$ parametrization |

### Key Functions in `low-q.ipynb`

| Function | Purpose |
|----------|---------|
| `FS_correct(chiq, qidxl, rs, Ne, dft_func)` | Apply finite-size correction to raw DMC $\chi$ |
| `get_chi(vql, qidxl, rs, Ne, dft_func)` | Full pipeline: fit → correct → propagate errors |
| `get_chi_Moroni(rs, Ne, qlist)` | Compute Moroni reference $\chi(q)$ |
| `get_chi_RPA(rs, Ne, qlist)` | Compute RPA $\chi(q)$ |
| `get_chi_corradini(rs, Ne, qlist)` | Compute Corradini reference $\chi(q)$ |
| `plot_chi(qidxl, chi, dchi, rs, Ne, ...)` | Plot $-\chi(q)/n_0$ vs $q/k_F$ with reference curves |
| `plot_E_of_vq(q, vq_list, rs, Ne, ...)` | Plot $E(v_q)$ parabola + fit for a single q-point |
| `E_of_vq(q, vql, rs, Ne, ...)` | Compute $E(v_q)$ array + quartic fit coefficients |

### Directory Layout for QMC Runs

```
{main_dir}/rs{rs:.1f}-n{Ne}/qv{qx}_{qy}_{qz}-vq{vq:.4f}/{dft_func}-e{ecut}-qa{alpha:.3f}-thr1.0d-{thr}/{wf}-t{tpmult*Ne*rs}-ts{ts:.4f}-nw{nw}/qmc.s002.stat.h5
```

Example:
```
runs/rs5.0-n54/qv1_0_0-vq0.0020/ni-e125-qa0.172-thr1.0d-10/sj-t4218.75-ts0.2500-nw1024/qmc.s002.stat.h5
```

---

## Current Challenges

1. **$v_q$ range sensitivity**: $\chi(q)$ (the curvature) is sensitive to which $v_q$ values are used in the quartic fit. For large $r_s$, the signal-to-noise ratio degrades because the energy scale shrinks as $\max\{v_q\} \propto 1/r_s$.

2. **Fitting robustness**: The quartic fit $E_0 + A v_q^2 + B v_q^4$ via `curve_fit` can give unstable $A$ when:
   - The $v_q$ range is too narrow (signal within error bars)
   - The $v_q$ range is too wide (higher-order terms matter)
   - Statistical noise in DMC energies is comparable to the curvature signal

3. **Low-q regime**: For small $q$, $|\chi(q)|$ is small (proportional to $q^2$), making the energy variation tiny and hard to resolve above statistical noise.

4. **High $r_s$**: Everything gets harder — smaller energy scales, larger fluctuations, breakdown of RPA-like assumptions.

---

## Units and Conventions

- **Atomic (Hartree) units** throughout: $\hbar = m_e = e = 1$.
- Energies in Hartree, lengths in Bohr radii.
- $\chi(q)$ is negative (for repulsive interactions), so $-\chi(q)/n_0$ is plotted as a positive quantity.
- $q$ is plotted in units of $k_F$ (i.e., $q/k_F$).
- The factor of 2 accounts for spin: two spin channels contribute independently.
