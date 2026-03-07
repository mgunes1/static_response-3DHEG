"""Physical constants and utility functions for 3D HEG calculations.

All quantities are in Hartree atomic units (energy: Hartree, length: Bohr).
"""

import numpy as np

_FOUR_PI = 4.0 * np.pi


def density_from_rs(rs):
    """Electron density from Wigner-Seitz radius.

    Parameters
    ----------
    rs : float
        Wigner-Seitz radius in Bohr.

    Returns
    -------
    float
        Electron density n in electrons/Bohr^3.
    """
    return 3.0 / (_FOUR_PI * rs**3)


def fermi_wavevector(rs):
    """Fermi wavevector of the 3D HEG.

    Parameters
    ----------
    rs : float
        Wigner-Seitz radius in Bohr.

    Returns
    -------
    float
        Fermi wavevector kF in 1/Bohr.
    """
    return (9.0 * np.pi / 4.0) ** (1.0 / 3.0) / rs


def fermi_energy(rs):
    """Fermi energy of the 3D HEG.

    Parameters
    ----------
    rs : float
        Wigner-Seitz radius in Bohr.

    Returns
    -------
    float
        Fermi energy EF in Hartree.
    """
    kf = fermi_wavevector(rs)
    return 0.5 * kf**2


def box_length(rs, N):
    """Side length of the cubic simulation box.

    Parameters
    ----------
    rs : float
        Wigner-Seitz radius in Bohr.
    N : int
        Number of electrons.

    Returns
    -------
    float
        Box side length L in Bohr.
    """
    n = density_from_rs(rs)
    return (N / n) ** (1.0 / 3.0)


def coulomb_potential(q):
    """Fourier transform of the Coulomb potential in 3D.

    v(q) = 4π / q²

    Parameters
    ----------
    q : float or ndarray
        Wavevector magnitude in 1/Bohr.

    Returns
    -------
    float or ndarray
        Coulomb potential in Hartree·Bohr³.
    """
    return _FOUR_PI / np.asarray(q) ** 2


def q_vectors_along_axis(rs, N, n_multiples=4, axis=0):
    """Accessible q-vectors along one lattice axis for a cubic cell.

    Returns q = m * (2π/L) * e_axis for m = 1, …, n_multiples.

    Parameters
    ----------
    rs : float
        Wigner-Seitz radius in Bohr.
    N : int
        Number of electrons.
    n_multiples : int
        Number of multiples of the minimum q to include.
    axis : int
        Cartesian axis (0=x, 1=y, 2=z).

    Returns
    -------
    q_vecs : ndarray, shape (n_multiples, 3)
        q-vectors in 1/Bohr.
    q_mags : ndarray, shape (n_multiples,)
        Magnitudes |q| in 1/Bohr.
    """
    L = box_length(rs, N)
    q_unit = 2.0 * np.pi / L
    q_vecs = np.zeros((n_multiples, 3))
    q_mags = np.zeros(n_multiples)
    for m in range(1, n_multiples + 1):
        q_vecs[m - 1, axis] = m * q_unit
        q_mags[m - 1] = m * q_unit
    return q_vecs, q_mags


def unique_q_magnitudes(rs, N, n_shells=3):
    """All distinct |q| values accessible in a cubic simulation cell.

    Enumerates integer triples (nx, ny, nz) with nx²+ny²+nz² ≤ n_shells²
    and returns the unique magnitudes |q| = (2π/L)|n|.

    Parameters
    ----------
    rs : float
        Wigner-Seitz radius in Bohr.
    N : int
        Number of electrons.
    n_shells : int
        Maximum |n| to include.

    Returns
    -------
    q_mags : ndarray
        Unique |q| values sorted in ascending order (1/Bohr).
    n_sq_vals : ndarray
        Corresponding integer |n|² values.
    """
    L = box_length(rs, N)
    q_unit = 2.0 * np.pi / L

    seen = set()
    n_sq_list = []
    for nx in range(-n_shells, n_shells + 1):
        for ny in range(-n_shells, n_shells + 1):
            for nz in range(-n_shells, n_shells + 1):
                n_sq = nx * nx + ny * ny + nz * nz
                if n_sq == 0 or n_sq > n_shells**2:
                    continue
                if n_sq not in seen:
                    seen.add(n_sq)
                    n_sq_list.append(n_sq)

    n_sq_arr = np.array(sorted(n_sq_list))
    q_mags = q_unit * np.sqrt(n_sq_arr.astype(float))
    return q_mags, n_sq_arr
