# Static response of the 3D HEG using diffusion Monte Carlo

Analysis code for computing the **static density-density response function** $\chi(q)$ of the three-dimensional **homogeneous electron gas (3D HEG)** via **Diffusion Monte Carlo (DMC)** ([QMCPACK](https://qmcpack.org)). Trial wavefunctions produced by DFT from [Quantum ESPRESSO](https://www.quantum-espresso.org).

---

## Method

### Extracting $\chi(q)$ from total energies

A static cosine perturbation is added to the Hamiltonian:

$$
H(v_q) = H_0 + 2 v_q \sum_i \cos(\mathbf{q} \cdot \mathbf{r}_i)
$$

For small amplitude $v_q$, the DMC ground-state energy per electron expands as:

$$
\frac{E(v_q)}{N_e} = \frac{E_0}{N_e} + \frac{\chi(q)}{n_0} v_q^2 + O(v_q^4)
$$

so $\chi(q)/n_0$ is the quadratic coefficient of a polynomial fit to $E(v_q)$. DMC energies are computed by QMCPACK using a Slater-Jastrow trial wavefunction perturbed at each wavevector $\mathbf{q}$.

### Finite-size correction

DMC energies are computed in a finite simulation cell of $N_e$ electrons. The finite-size error in $\chi$ is corrected using the known analytic form of the non-interacting finite-size bias:

$$
\chi^{-1}(q; \infty) = \chi^{-1}(q; N_e) + \left[\chi_0^{-1}(q; \infty) - \chi_0^{-1}(q; N_e)\right]
$$

where $\chi_0$ is the non-interacting Lindhard function.

---

## Installation

### Prerequisites

- Python ≥ 3.10
- [`qharv`](https://github.com/Paul-St-Young/harvest_qmcpack) — QMCPACK output parser (not on PyPI)

### Option A: pip (venv)

```bash
git clone git@github.com:moegunes/static_response-3DHEG.git
cd static_response-3DHEG

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
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

Pre-computed energy matrices for all (r_s, N_e) datasets are provided in `output/` as `.npz` files. **No raw QMC data is needed to reproduce the figures** — simply clone the repository and run the notebook cells. Running the analysis for cases where datasets are not available, the code will try to build `E_all.npz` from scratch. Note that unless you have access to the workflow, this is impossible to do.

---

## Usage

### Reproduce figures

Open `low-q.ipynb` in Jupyter and run cell by cell. Each section loads the cached `.npz` for a given r_s and produces the corresponding χ(q) figure.

### Compute χ(q) programmatically

```python
from utils.io_utils import collect_q_and_vq
from utils.fitting import get_chi
from utils.plotting import plot_chi
import matplotlib.pyplot as plt

main_dir = "/path/to/qmcpack/runs"
rs, Ne = 5, 54

qidx_list, vq_list = collect_q_and_vq(main_dir, rs, Ne)

chi, dchi, fit_quality = get_chi(
    main_dir, vq_list, qidx_list, rs, Ne,
    dft_func='ni',       # DFT functional for FS correction
    vq_fit='quadratic',  # 'quadratic' or 'quartic'
    n_boot=800,
)

fig, ax = plt.subplots(dpi=200)
plot_chi(qidx_list, chi, dchi, rs, Ne, chi_ref='both', ax=ax)
fig.savefig('chi_rs5.pdf', bbox_inches='tight')
```

---

## References

1. Moroni, S., Ceperley, D. M., & Senatore, G. (1995). *Static response from quantum Monte Carlo calculations.* **Phys. Rev. Lett. 75**, 689. https://doi.org/10.1103/PhysRevLett.75.689

2. Corradini, M., Del Sole, R., Onida, G., & Palummo, M. (1998). *Analytical expressions for the local-field factor G(q) and the exchange-correlation kernel.* **Phys. Rev. B 57**, 14569. https://doi.org/10.1103/PhysRevB.57.14569

3. Kim, J. et al. (2018). *QMCPACK: an open source ab initio quantum Monte Carlo package.* **J. Phys.: Condens. Matter 30**, 195901. https://doi.org/10.1088/1361-648X/aab9c3

4. Giannozzi, P. et al. (2009). *QUANTUM ESPRESSO: a modular and open-source software project.* **J. Phys.: Condens. Matter 21**, 395502. https://doi.org/10.1088/0953-8984/21/39/395502
