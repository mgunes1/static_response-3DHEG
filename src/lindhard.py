"""Lindhard function and free-electron response for the 3D HEG.

All quantities are in Hartree atomic units.
"""

import numpy as np
from .utils import fermi_wavevector, coulomb_potential


def lindhard_function(x):
    """Dimensionless static Lindhard function L(x).

    L(x) = 1/2 + (1 - x²) / (4x) * ln|(1 + x) / (1 - x)|

    Parameters
    ----------
    x : float or array_like
        Reduced wavevector x = q / (2 kF).

    Returns
    -------
    float or ndarray
        Lindhard function values.
    """
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x.copy())

    result = np.empty_like(x)
    eps = 1e-8

    mask_zero = x < eps
    mask_one = np.abs(x - 1.0) < eps
    mask_normal = ~mask_zero & ~mask_one

    # General case
    xn = x[mask_normal]
    log_arg = np.abs((1.0 + xn) / (1.0 - xn))
    result[mask_normal] = 0.5 + (1.0 - xn**2) / (4.0 * xn) * np.log(log_arg)

    # x → 0: L(x) → 1 − x²/3 (uniform limit)
    xz = x[mask_zero]
    result[mask_zero] = 1.0 - xz**2 / 3.0

    # x = 1: L(1) = 1/2 exactly
    result[mask_one] = 0.5

    return result[0] if scalar else result


def chi0(q, rs):
    """Non-interacting (Lindhard) static density-density response function.

    χ₀(q) = −(kF / π²) L(q / 2kF)

    Parameters
    ----------
    q : float or array_like
        Wavevector magnitude in 1/Bohr.
    rs : float
        Wigner-Seitz radius in Bohr.

    Returns
    -------
    float or ndarray
        χ₀(q) in units of 1 / (Hartree · Bohr³).
    """
    kf = fermi_wavevector(rs)
    x = np.asarray(q) / (2.0 * kf)
    return -(kf / np.pi**2) * lindhard_function(x)


def chi_rpa(q, rs):
    """Random-Phase-Approximation static response function.

    χ_RPA(q) = χ₀(q) / (1 − v(q) χ₀(q))

    Parameters
    ----------
    q : float or array_like
        Wavevector magnitude in 1/Bohr.
    rs : float
        Wigner-Seitz radius in Bohr.

    Returns
    -------
    float or ndarray
        χ_RPA(q) in units of 1 / (Hartree · Bohr³).
    """
    c0 = chi0(q, rs)
    v = coulomb_potential(q)
    return c0 / (1.0 - v * c0)


def local_field_factor(q, chi_dmc, rs):
    """Extract the local field factor G(q) from a DMC response function.

    Derived from the Dyson equation

        χ(q) = χ₀(q) / (1 − v(q) [1 − G(q)] χ₀(q))

    which gives

        G(q) = 1 + (1 / v(q)) * (1/χ_DMC(q) − 1/χ₀(q))

    When χ_DMC = χ_RPA (G=0 by construction) this identity is satisfied
    exactly.

    Parameters
    ----------
    q : float or array_like
        Wavevector magnitude in 1/Bohr.
    chi_dmc : float or array_like
        DMC response function in 1 / (Hartree · Bohr³).
    rs : float
        Wigner-Seitz radius in Bohr.

    Returns
    -------
    float or ndarray
        Dimensionless local field factor G(q).
    """
    c0 = chi0(q, rs)
    v = coulomb_potential(q)
    return 1.0 + (1.0 / v) * (1.0 / np.asarray(chi_dmc) - 1.0 / c0)
