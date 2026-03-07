#!/usr/bin/env python3
"""Generate synthetic QMCPACK-like energy matrices for sample data.

This script creates physically realistic (but synthetic) DMC energy matrices
using the RPA response function as a model χ(q).  The resulting .npz files
in output/ stand in for real QMCPACK output during development and testing.

Usage
-----
    python output/generate_sample_data.py
"""

import os
import sys

import numpy as np

# Allow running from the repo root or from the output/ directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from src.data_io import get_output_path, save_energy_matrix
from src.lindhard import chi_rpa
from src.utils import density_from_rs, q_vectors_along_axis

# ---------------------------------------------------------------------------
# Configurations to generate
# ---------------------------------------------------------------------------
CONFIGS = [
    {"rs": 1.0, "N": 14},
    {"rs": 2.0, "N": 14},
    {"rs": 2.0, "N": 38},
    {"rs": 5.0, "N": 14},
    {"rs": 10.0, "N": 14},
    {"rs": 20.0, "N": 14},
]

# Perturbation strengths (Hartree)
LAMBDAS = np.array([0.0, 0.01, 0.05, 0.1, 0.2])

# DMC-like statistical error per electron (Hartree)
STAT_ERROR = 5e-4

# Number of q-multiples along (1,0,0)
N_Q = 4

# Random seed for reproducibility
RNG = np.random.default_rng(42)


def _heg_energy_per_electron(rs):
    """Approximate HEG ground-state energy per electron (Hartree).

    Uses the Perdew-Zunger parametrisation of the Ceperley-Alder DMC data.
    """
    E_kin = 2.21 / rs**2
    E_x = -0.9163 / rs

    if rs >= 1.0:
        gamma, beta1, beta2 = -0.1423, 1.0529, 0.3334
        E_c = gamma / (1.0 + beta1 * np.sqrt(rs) + beta2 * rs)
    else:
        A, B, C, D = 0.0311, -0.048, 0.002, -0.0116
        ln_rs = np.log(rs)
        E_c = A * ln_rs + B + C * rs * ln_rs + D * rs

    return E_kin + E_x + E_c


def generate_energy_matrix(rs, N, lambdas, n_q=N_Q):
    """Generate a synthetic energy matrix for given (rs, N).

    The signal uses the RPA response function; Gaussian noise mimics DMC
    statistical errors.

    Returns
    -------
    q_vecs : ndarray, shape (n_q, 3)
    q_mags : ndarray, shape (n_q,)
    energies : ndarray, shape (n_q, n_lambda)
    errors : ndarray, shape (n_q, n_lambda)
    """
    n = density_from_rs(rs)
    E0 = _heg_energy_per_electron(rs)

    q_vecs, q_mags = q_vectors_along_axis(rs, N, n_multiples=n_q)
    n_lambda = len(lambdas)

    energies = np.zeros((n_q, n_lambda))
    errors = np.full((n_q, n_lambda), STAT_ERROR)

    for i, q in enumerate(q_mags):
        chi = chi_rpa(q, rs)

        for j, lam in enumerate(lambdas):
            # Linear-response energy shift: ΔE/N = −χ(q) λ² / (4n)
            dE = -chi * lam**2 / (4.0 * n)
            # Small quartic correction (higher-order perturbation theory)
            b_quartic = 0.1 * chi**2 / n
            dE += b_quartic * lam**4

            noise = RNG.normal(0.0, STAT_ERROR)
            energies[i, j] = E0 + dE + noise

    return q_vecs, q_mags, energies, errors


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    for cfg in CONFIGS:
        rs, N = cfg["rs"], cfg["N"]
        print(f"Generating  rs={rs:.1f},  N={N} …")

        q_vecs, q_mags, energies, errors = generate_energy_matrix(rs, N, LAMBDAS)

        path = get_output_path(base_dir, rs, N)
        save_energy_matrix(
            path=path,
            energies=energies,
            q_vectors=q_vecs,
            q_magnitudes=q_mags,
            lambdas=LAMBDAS,
            rs=rs,
            N=N,
            energy_errors=errors,
            metadata={
                "method": "DMC-RPA-synthetic",
                "description": (
                    "Synthetic data generated from the RPA response function. "
                    "Replace with real QMCPACK output."
                ),
                "stat_error": STAT_ERROR,
            },
        )
        print(f"  → {path}.npz  ({len(q_mags)} q-points, {len(LAMBDAS)} λ values)")

    print("Done.")


if __name__ == "__main__":
    main()
