"""I/O utilities for QMCPACK energy matrices.

Energy matrices are stored as compressed NumPy archives (.npz) with a
consistent layout that includes the energy values, q-vectors, λ values,
and metadata describing the calculation.

Directory layout
----------------
output/
    rs{X.X}/
        N{Y}/
            energies.npz          # standard DMC run
            energies_backflow.npz # optional backflow run
"""

import os

import numpy as np


def save_energy_matrix(
    path,
    energies,
    q_vectors,
    q_magnitudes,
    lambdas,
    rs,
    N,
    energy_errors=None,
    metadata=None,
):
    """Save an energy matrix to a compressed .npz file.

    Parameters
    ----------
    path : str
        Destination path (the .npz extension is appended if absent).
    energies : ndarray, shape (n_q, n_lambda)
        DMC energies *per electron* in Hartree.
    q_vectors : ndarray, shape (n_q, 3)
        Full q-vectors in 1/Bohr.
    q_magnitudes : ndarray, shape (n_q,)
        |q| values in 1/Bohr.
    lambdas : array_like, shape (n_lambda,)
        Perturbation strengths in Hartree.
    rs : float
        Wigner-Seitz radius in Bohr.
    N : int
        Number of electrons.
    energy_errors : ndarray, shape (n_q, n_lambda), optional
        Statistical uncertainties on DMC energies.
    metadata : dict, optional
        Arbitrary string/scalar metadata (e.g. method, backflow flag).
    """
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    arrays = {
        "energies": np.asarray(energies, dtype=float),
        "q_vectors": np.asarray(q_vectors, dtype=float),
        "q_magnitudes": np.asarray(q_magnitudes, dtype=float),
        "lambdas": np.asarray(lambdas, dtype=float),
        "rs": np.float64(rs),
        "N": np.int64(N),
    }

    if energy_errors is not None:
        arrays["energy_errors"] = np.asarray(energy_errors, dtype=float)

    if metadata is not None:
        for key, val in metadata.items():
            if isinstance(val, str):
                arrays[f"meta_{key}"] = np.array(val, dtype=object)
            else:
                arrays[f"meta_{key}"] = np.asarray(val)

    np.savez_compressed(path, **arrays)


def load_energy_matrix(path):
    """Load an energy matrix from a .npz file.

    Parameters
    ----------
    path : str
        Path to the file (the .npz extension is appended if absent).

    Returns
    -------
    dict with keys
        energies, q_vectors, q_magnitudes, lambdas, rs, N,
        energy_errors (if present), metadata (dict, if present).
    """
    if not path.endswith(".npz"):
        path = path + ".npz"

    data = np.load(path, allow_pickle=True)

    result = {
        "energies": data["energies"],
        "q_vectors": data["q_vectors"],
        "q_magnitudes": data["q_magnitudes"],
        "lambdas": data["lambdas"],
        "rs": float(data["rs"]),
        "N": int(data["N"]),
    }

    if "energy_errors" in data:
        result["energy_errors"] = data["energy_errors"]

    metadata = {}
    for key in data.files:
        if key.startswith("meta_"):
            val = data[key]
            metadata[key[5:]] = val.item() if val.ndim == 0 else val
    if metadata:
        result["metadata"] = metadata

    return result


def get_output_path(base_dir, rs, N, suffix=""):
    """Return the canonical path for an energy matrix file.

    Parameters
    ----------
    base_dir : str
        Root output directory (e.g. ``"output"``).
    rs : float
        Wigner-Seitz radius.
    N : int
        Number of electrons.
    suffix : str, optional
        Extra suffix appended to ``"energies"`` (e.g. ``"_backflow"``).

    Returns
    -------
    str
        Path without the ``.npz`` extension.
    """
    rs_str = f"rs{rs:.1f}"
    N_str = f"N{N}"
    return os.path.join(base_dir, rs_str, N_str, f"energies{suffix}")
