# static_response-3DHEG

Analysis code for the **static density-density response function** χ(q) of the
three-dimensional homogeneous electron gas (3D HEG), computed with Diffusion
Monte Carlo (DMC) as implemented in [QMCPACK](https://qmcpack.org/).

The goal is to benchmark and extend the classic results of Moroni *et al.*
(1995) to a wider range of electron densities — including the strongly
correlated regime r_s ≥ 10 — using modern QMC techniques (backflow
wavefunctions, larger simulation cells, twist averaging).

---

## Physics background

The static density-density response function χ(q) is extracted from DMC
energies via the *perturbation method*: an external cosine potential

    H' = λ Σᵢ cos(q · rᵢ)

is added to the Hamiltonian.  In linear response, the DMC energy per electron
shifts as

    E(λ)/N ≈ E₀/N − [χ(q) / (4n)] λ²

so that a quadratic fit in λ gives

    χ(q) = −4n a

where a is the λ² coefficient and n = 3/(4π r_s³) is the electron density.

The local field factor G(q) — the key exchange-correlation quantity — is then
obtained from the Dyson equation

    G(q) = 1 + (1/v(q)) [1/χ(q) − 1/χ₀(q)]

where χ₀(q) is the non-interacting (Lindhard) response and v(q) = 4π/q² is
the Coulomb potential.

---

## Repository layout

```
output/                     # QMCPACK energy matrices (one .npz per run)
│  generate_sample_data.py  # regenerate the bundled synthetic data
│  rs1.0/N14/energies.npz
│  rs2.0/N14/energies.npz
│  rs2.0/N38/energies.npz
│  rs5.0/N14/energies.npz
│  rs10.0/N14/energies.npz
│  rs20.0/N14/energies.npz
src/
│  utils.py      physical constants, density/kF/L helpers, q-vector generation
│  lindhard.py   Lindhard function χ₀(q), RPA response χ_RPA(q), G(q)
│  response.py   fit E(λ), compute χ(q) from an energy matrix
│  data_io.py    save/load .npz energy matrices
scripts/
│  compute_chi.py  command-line analysis script
tests/
│  test_utils.py, test_lindhard.py, test_response.py, test_data_io.py
requirements.txt
```

### Energy matrix format

Each `.npz` file produced by QMCPACK (or the sample-data script) contains:

| Array           | Shape      | Description                        |
|-----------------|------------|------------------------------------|
| `energies`      | (n_q, n_λ) | DMC energy per electron (Hartree)  |
| `energy_errors` | (n_q, n_λ) | Statistical uncertainties          |
| `q_vectors`     | (n_q, 3)   | Full q-vectors (Bohr⁻¹)            |
| `q_magnitudes`  | (n_q,)     | |q| values (Bohr⁻¹)               |
| `lambdas`       | (n_λ,)     | Perturbation strengths (Hartree)   |
| `rs`            | scalar     | Wigner-Seitz radius (Bohr)         |
| `N`             | scalar     | Number of electrons                |

Additional `meta_*` keys store free-form metadata (method, backflow flag, …).

---

## Quick start

```bash
pip install -r requirements.txt

# Compute χ(q) for rs=2, N=14 (uses bundled sample data)
python scripts/compute_chi.py --rs 2.0 --N 14

# Save numerical results and generate a plot
python scripts/compute_chi.py --rs 2.0 --N 14 \
    --save-results results_rs2_N14.npz --plot

# Run the test suite
python -m pytest tests/ -v
```

To use your own QMCPACK output, save the energy matrices to `output/` with
`src.data_io.save_energy_matrix` (see `output/generate_sample_data.py` for a
complete example), then run `compute_chi.py` as above.

---

## References

* S. Moroni, D. M. Ceperley, G. Senatore, *Phys. Rev. Lett.* **75**, 689 (1995)
  — original DMC calculation of χ(q) for r_s = 2, 5, 10.
* D. M. Ceperley and B. J. Alder, *Phys. Rev. Lett.* **45**, 566 (1980)
  — DMC ground-state energies of the homogeneous electron gas.
* J. Kim *et al.*, *J. Phys.: Condens. Matter* **30**, 195901 (2018)
  — QMCPACK code description.
