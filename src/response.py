"""Compute the static density-density response function χ(q) from DMC energies.

The standard perturbation method (Moroni et al. 1995) adds a cosine external
potential to the DMC Hamiltonian:

    H' = λ Σᵢ cos(q · rᵢ)

In linear response, the DMC energy per electron shifts as:

    E(λ)/N ≈ E₀/N − [χ(q) / (4n)] λ²

where n is the electron density.  A quadratic fit in λ therefore gives:

    χ(q) = −4n a

where a is the λ² coefficient of the fit.
"""

import numpy as np
from scipy.optimize import curve_fit

from .utils import density_from_rs


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def _poly2_model(lam, E0, a, b):
    """E(λ) = E0 + a λ² + b λ⁴"""
    return E0 + a * lam**2 + b * lam**4


def fit_energy_perturbation(lambdas, energies, errors=None):
    """Fit DMC energy vs perturbation strength.

    Fits the model  E(λ) = E₀ + a λ² + b λ⁴  using least-squares.

    Parameters
    ----------
    lambdas : array_like, shape (n_lambda,)
        Perturbation strengths λ in Hartree.
    energies : array_like, shape (n_lambda,)
        DMC energies per electron in Hartree.
    errors : array_like, shape (n_lambda,), optional
        Statistical uncertainties on energies (used as weights).

    Returns
    -------
    dict with keys
        E0, a, b          – fit parameters
        E0_err, a_err, b_err  – 1-σ uncertainties from covariance matrix
        cov               – full 3×3 covariance matrix
    """
    lambdas = np.asarray(lambdas, dtype=float)
    energies = np.asarray(energies, dtype=float)

    zero_mask = lambdas == 0.0
    E0_guess = float(energies[zero_mask].mean()) if zero_mask.any() else float(energies[0])

    sigma = errors if errors is not None else None

    try:
        popt, pcov = curve_fit(
            _poly2_model,
            lambdas,
            energies,
            p0=[E0_guess, -1e-3, 0.0],
            sigma=sigma,
            absolute_sigma=(sigma is not None),
            maxfev=10_000,
        )
    except RuntimeError:
        # Fallback: simple polynomial fit in λ²
        lam2 = lambdas**2
        coeffs = np.polyfit(lam2, energies - E0_guess, 2)
        popt = [E0_guess, float(coeffs[1]), float(coeffs[0])]
        pcov = np.zeros((3, 3))

    return {
        "E0": popt[0],
        "a": popt[1],
        "b": popt[2],
        "cov": pcov,
        "E0_err": float(np.sqrt(max(pcov[0, 0], 0.0))),
        "a_err": float(np.sqrt(max(pcov[1, 1], 0.0))),
        "b_err": float(np.sqrt(max(pcov[2, 2], 0.0))),
    }


# ---------------------------------------------------------------------------
# χ(q) from a single q-vector
# ---------------------------------------------------------------------------

def chi_from_energy_perturbation(lambdas, energies, q_mag, rs, N, errors=None):
    """Compute χ(q) from DMC energies at one q-vector.

    Parameters
    ----------
    lambdas : array_like, shape (n_lambda,)
        Perturbation strengths in Hartree.
    energies : array_like, shape (n_lambda,)
        DMC energies *per electron* in Hartree.
    q_mag : float
        |q| in 1/Bohr (informational, stored in output).
    rs : float
        Wigner-Seitz radius in Bohr.
    N : int
        Number of electrons.
    errors : array_like, shape (n_lambda,), optional
        Statistical uncertainties on energies.

    Returns
    -------
    dict with keys
        chi      – response function in 1 / (Hartree · Bohr³)
        chi_err  – 1-σ statistical uncertainty
        fit      – full fit result dict
        q, rs, N
    """
    n = density_from_rs(rs)
    fit = fit_energy_perturbation(lambdas, energies, errors)

    chi = -4.0 * n * fit["a"]
    chi_err = 4.0 * n * fit["a_err"]

    return {
        "chi": chi,
        "chi_err": chi_err,
        "fit": fit,
        "q": float(q_mag),
        "rs": float(rs),
        "N": int(N),
    }


# ---------------------------------------------------------------------------
# χ(q) for a full energy matrix
# ---------------------------------------------------------------------------

def compute_chi_q(energy_matrix, lambdas, q_magnitudes, rs, N, error_matrix=None):
    """Compute χ(q) for all q-vectors from an energy matrix.

    Parameters
    ----------
    energy_matrix : ndarray, shape (n_q, n_lambda)
        DMC energies per electron in Hartree for each (q, λ) pair.
    lambdas : array_like, shape (n_lambda,)
        Perturbation strengths in Hartree.
    q_magnitudes : array_like, shape (n_q,)
        |q| values in 1/Bohr.
    rs : float
        Wigner-Seitz radius in Bohr.
    N : int
        Number of electrons.
    error_matrix : ndarray, shape (n_q, n_lambda), optional
        Statistical uncertainties on DMC energies.

    Returns
    -------
    dict with keys
        q        – ndarray of |q| values (1/Bohr)
        chi      – ndarray of χ(q) values (1/(Hartree·Bohr³))
        chi_err  – ndarray of 1-σ statistical uncertainties
        rs, N
    """
    lambdas = np.asarray(lambdas, dtype=float)
    q_magnitudes = np.asarray(q_magnitudes, dtype=float)
    energy_matrix = np.asarray(energy_matrix, dtype=float)

    n_q = len(q_magnitudes)
    chi_values = np.zeros(n_q)
    chi_errors = np.zeros(n_q)

    for i, q_mag in enumerate(q_magnitudes):
        errors_i = error_matrix[i] if error_matrix is not None else None
        result = chi_from_energy_perturbation(
            lambdas, energy_matrix[i], q_mag, rs, N, errors_i
        )
        chi_values[i] = result["chi"]
        chi_errors[i] = result["chi_err"]

    return {
        "q": q_magnitudes,
        "chi": chi_values,
        "chi_err": chi_errors,
        "rs": float(rs),
        "N": int(N),
    }
