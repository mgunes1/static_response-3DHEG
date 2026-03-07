"""Tests for src/response.py."""
import numpy as np
import pytest

from src.lindhard import chi_rpa
from src.response import (
    chi_from_energy_perturbation,
    compute_chi_q,
    fit_energy_perturbation,
)
from src.utils import density_from_rs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_energies(lambdas, chi_true, rs, noise=0.0, rng=None):
    """Create E/N vs λ data consistent with a given χ."""
    n = density_from_rs(rs)
    E0 = -0.5  # arbitrary offset
    energies = E0 - chi_true * lambdas**2 / (4.0 * n)
    if noise:
        rng = rng or np.random.default_rng(0)
        energies += rng.normal(0.0, noise, size=lambdas.shape)
    return energies


# ---------------------------------------------------------------------------
# fit_energy_perturbation
# ---------------------------------------------------------------------------

class TestFitEnergyPerturbation:
    def test_exact_quadratic(self):
        lambdas = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
        E0_true, a_true = -0.5, -0.03
        energies = E0_true + a_true * lambdas**2
        fit = fit_energy_perturbation(lambdas, energies)
        assert np.isclose(fit["E0"], E0_true, atol=1e-10)
        assert np.isclose(fit["a"], a_true, atol=1e-10)

    def test_recovers_a_with_noise(self):
        lambdas = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
        E0_true, a_true = -0.5, -0.02
        rng = np.random.default_rng(123)
        energies = E0_true + a_true * lambdas**2 + rng.normal(0, 1e-6, lambdas.shape)
        fit = fit_energy_perturbation(lambdas, energies)
        assert np.isclose(fit["a"], a_true, atol=2e-4)

    def test_quartic_term_absorbed(self):
        lambdas = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
        E0_true, a_true, b_true = -0.5, -0.02, 0.005
        energies = E0_true + a_true * lambdas**2 + b_true * lambdas**4
        fit = fit_energy_perturbation(lambdas, energies)
        assert np.isclose(fit["a"], a_true, atol=1e-8)
        assert np.isclose(fit["b"], b_true, atol=1e-8)

    def test_dict_keys_present(self):
        lambdas = np.linspace(0, 0.2, 5)
        energies = -0.5 + (-0.01) * lambdas**2
        fit = fit_energy_perturbation(lambdas, energies)
        for key in ("E0", "a", "b", "cov", "E0_err", "a_err", "b_err"):
            assert key in fit


# ---------------------------------------------------------------------------
# chi_from_energy_perturbation
# ---------------------------------------------------------------------------

class TestChiFromEnergyPerturbation:
    def test_recovers_chi(self):
        rs, N = 2.0, 14
        q_mag = 0.8
        chi_true = chi_rpa(q_mag, rs)

        lambdas = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
        energies = _make_synthetic_energies(lambdas, chi_true, rs)

        result = chi_from_energy_perturbation(lambdas, energies, q_mag, rs, N)
        assert np.isclose(result["chi"], chi_true, rtol=1e-6)

    def test_result_keys(self):
        lambdas = np.array([0.0, 0.1, 0.2])
        energies = -0.5 + (-0.01) * lambdas**2
        result = chi_from_energy_perturbation(lambdas, energies, 1.0, 2.0, 14)
        for key in ("chi", "chi_err", "fit", "q", "rs", "N"):
            assert key in result

    def test_chi_negative(self):
        # χ(q) < 0 is physical for the HEG
        rs, N = 2.0, 14
        q_mag = 0.8
        chi_true = chi_rpa(q_mag, rs)
        lambdas = np.array([0.0, 0.05, 0.1, 0.2])
        energies = _make_synthetic_energies(lambdas, chi_true, rs)
        result = chi_from_energy_perturbation(lambdas, energies, q_mag, rs, N)
        assert result["chi"] < 0.0


# ---------------------------------------------------------------------------
# compute_chi_q
# ---------------------------------------------------------------------------

class TestComputeChiQ:
    def test_recovers_chi_matrix(self):
        rs, N = 2.0, 14
        lambdas = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
        q_mags = np.array([0.8, 1.0, 1.5])
        chi_true = chi_rpa(q_mags, rs)

        # Build energy matrix with exact signal (no noise)
        energy_matrix = np.zeros((len(q_mags), len(lambdas)))
        for i, q in enumerate(q_mags):
            energy_matrix[i] = _make_synthetic_energies(lambdas, chi_true[i], rs)

        result = compute_chi_q(energy_matrix, lambdas, q_mags, rs, N)
        assert np.allclose(result["chi"], chi_true, rtol=1e-5)

    def test_output_shapes(self):
        rs, N = 2.0, 14
        lambdas = np.array([0.0, 0.1, 0.2])
        q_mags = np.array([0.8, 1.2, 1.8])
        energy_matrix = np.zeros((3, 3)) - 0.5

        result = compute_chi_q(energy_matrix, lambdas, q_mags, rs, N)
        assert result["q"].shape == (3,)
        assert result["chi"].shape == (3,)
        assert result["chi_err"].shape == (3,)

    def test_with_error_matrix(self):
        rs, N = 2.0, 14
        lambdas = np.array([0.0, 0.1, 0.2])
        q_mags = np.array([0.8, 1.2])
        energy_matrix = np.zeros((2, 3)) - 0.5
        error_matrix = np.full((2, 3), 1e-3)

        result = compute_chi_q(
            energy_matrix, lambdas, q_mags, rs, N, error_matrix=error_matrix
        )
        assert result["chi"].shape == (2,)
