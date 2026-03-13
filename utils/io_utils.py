"""
I/O utilities for the static response project.

Functions for reading QMC / DFT energies, parsing directory trees,
and caching the full energy matrix E(q, vq).
"""

import os
import re
import warnings
import xml.etree.ElementTree as ET

import numpy as np

# ---------------------------------------------------------------------------
# Energy extraction
# ---------------------------------------------------------------------------


def get_energy(h5_file, nequil=50):
    """
    Extract energy and error from QMCPACK stat.h5 file.
    Uses proper autocorrelation correction via harvest_qmcpack.

    Parameters
    ----------
    h5_file : str
        Path to stat.h5 file.
    nequil : int
        Number of equilibration blocks to discard (default: 50).

    Returns
    -------
    E : float
        Mean energy.
    dE : float
        Error accounting for autocorrelation.
    """
    import h5py

    try:
        from qharv.reel.stat_h5 import mean_and_err

        with h5py.File(h5_file, "r") as f:
            E, dE = mean_and_err(f, "LocalEnergy", nequil)
        return E[0], dE[0]
    except ImportError:
        raise ImportError(
            "qharv.reel.stat_h5 not found. "
            "Install qharv or check that h5_file is valid."
        )


def get_energy_pwscf(path):
    """
    Extract DFT band energy (<eband>) from Quantum Espresso XML output.

    Parameters
    ----------
    path : str
        Path to pwscf.xml or a directory containing it.

    Returns
    -------
    float
        The band energy in Hartree.
    """
    xml_path = path
    if os.path.isdir(xml_path):
        xml_path = os.path.join(xml_path, "pwscf.xml")

    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"File not found: {xml_path}")

    try:
        for event, elem in ET.iterparse(xml_path, events=("end",)):
            if elem.tag == "eband":
                text = elem.text.strip() if elem.text else ""
                try:
                    return float(text)
                except ValueError:
                    raise ValueError(f"Cannot parse etot value: '{text}'")
    except ET.ParseError as e:
        raise ValueError(f"XML parse error: {e}")

    raise ValueError("<etot> tag not found in XML file")


def get_variance(scalar_file, nequil=50):
    """
    Extract variance and error from QMCPACK scalar.dat file.

    Parameters
    ----------
    scalar_file : str
        Path to scalar.dat file.
    nequil : int
        Equilibration blocks to discard.

    Returns
    -------
    var : float
        Mean variance.
    dvar : float
        Error on the variance.
    """
    try:
        from qharv.reel.scalar_dat import read, single_column

        df = read(scalar_file)
        var, dvar, _ = single_column(df, "Variance", nequil)
        return var, dvar
    except ImportError:
        import pandas as pd

        df = pd.read_csv(scalar_file, comment="#", sep=r"\s+", header=None)
        with open(scalar_file, "r") as f:
            header = f.readline().strip()
        if header.startswith("#"):
            cols = header.replace("#", "").split()
            df.columns = cols[: len(df.columns)]

        if (
            "Variance" not in df.columns
            and "LocalEnergy" in df.columns
            and "LocalEnergy_sq" in df.columns
        ):
            df["Variance"] = df["LocalEnergy_sq"] - df["LocalEnergy"] ** 2

        var_vals = df["Variance"].values[nequil:]
        var = np.mean(var_vals)
        dvar = np.std(var_vals, ddof=1) / np.sqrt(len(var_vals))
        return var, dvar


# ---------------------------------------------------------------------------
# Directory parsing
# ---------------------------------------------------------------------------


def collect_q_and_vq(runs_path, rs, n):
    """
    Parse q-indices and vq values from a QMC run directory tree.

    Parameters
    ----------
    runs_path : str
        Path to the ``runs`` directory.
    rs : float
        Wigner-Seitz radius.
    n : int
        Particle number.

    Returns
    -------
    qidx_list : list of [int, int, int]
        Unique q-indices, sorted by magnitude.
    vq_list : list of float
        Sorted unique vq values.
    """
    try:
        data = np.load(
            runs_path + "/output/E_all_rs{:.1f}-n{:d}.npz".format(rs, n),
            allow_pickle=True,
        )
        qidx_list = [list(q) for q in data["qlist"]]
        vq_list = list(data["vqlist"])
        print(f"  [cache hit] loaded q/vq from {runs_path}")
        return qidx_list, vq_list

    except (FileNotFoundError, KeyError):
        print(f"  [cache miss] no q/vq cache at {runs_path} — parsing directory tree")
        rsn_dir = os.path.join(runs_path, f"rs{rs:.1f}-n{n}")
        if not os.path.isdir(rsn_dir):
            raise FileNotFoundError(f"Directory not found: {rsn_dir}")

        q_pattern = re.compile(r"qv(-?\d+)_(-?\d+)_(-?\d+)-vq([\d.]+)$")

        qidx_set = set()
        vq_set = set()

        with os.scandir(rsn_dir) as entries:
            for entry in entries:
                if not entry.is_dir():
                    continue
                match = q_pattern.match(entry.name)
                if match:
                    qidx_set.add(
                        (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                    )
                    vq_set.add(float(match.group(4)))

        qidx_list = [list(q) for q in sorted(qidx_set)]
        vq_list = sorted(vq_set)
        return qidx_list, vq_list


# ---------------------------------------------------------------------------
# QMC run parameters
# ---------------------------------------------------------------------------


def qmc_params_default(rs, Ne):
    """Return default QMC simulation parameters."""
    ecut_pre = 125
    wf = "sj"
    dft_func = "ni"
    ts = rs / (12 * Ne**0.5) if wf == "sjb" else rs / 20
    ss = 3 if wf == "sjb" else 2
    nw = 1024
    tpmult = 15.625
    return ecut_pre, wf, dft_func, ts, ss, nw, tpmult


def _build_h5_path(main_dir, rs, Ne, q, vq, pwscf=False):
    """Construct the path to a stat.h5 file for given run parameters."""
    from .physics import guess_alpha2

    alpha = guess_alpha2(rs, Ne, q)
    ecut_pre, wf, dft_func, ts, ss, nw, tpmult = qmc_params_default(rs, Ne)
    if rs > 29:
        if pwscf:
            thr = 10
            return (
                f"{main_dir}/rs{rs:.1f}-n{Ne:d}/"
                f"qv{q[0]:d}_{q[1]:d}_{q[2]:d}-vq{vq:.5f}/"
                f"{dft_func}-e{ecut_pre}-qa{alpha:.3f}-thr1.0d-{thr}/"
                f"scf/qeout"
            )
        else:
            return (
                f"{main_dir}/rs{rs:.1f}-n{Ne:d}/"
                f"qv{q[0]:d}_{q[1]:d}_{q[2]:d}-vq{vq:.5f}/"
                f"{dft_func}-e{ecut_pre}-qa{alpha:.3f}-thr1.0d-10/"
                f"{wf}-t{tpmult * Ne * rs}-ts{ts:.4f}-nw{nw}/"
                f"qmc.s00{ss}.stat.h5"
            )
    else:
        if pwscf:
            thr = 10
            return (
                f"{main_dir}/rs{rs:.1f}-n{Ne:d}/"
                f"qv{q[0]:d}_{q[1]:d}_{q[2]:d}-vq{vq:.4f}/"
                f"{dft_func}-e{ecut_pre}-qa{alpha:.3f}-thr1.0d-{thr}/"
                f"scf/qeout"
            )
        else:
            return (
                f"{main_dir}/rs{rs:.1f}-n{Ne:d}/"
                f"qv{q[0]:d}_{q[1]:d}_{q[2]:d}-vq{vq:.4f}/"
                f"{dft_func}-e{ecut_pre}-qa{alpha:.3f}-thr1.0d-10/"
                f"{wf}-t{tpmult * Ne * rs}-ts{ts:.4f}-nw{nw}/"
                f"qmc.s00{ss}.stat.h5"
            )


def get_variance_for_run(main_dir, rs, Ne, q, vq, nequil=50):
    """
    Get variance for a specific QMC run.

    Returns
    -------
    (var, dvar) or (None, None) if the file is missing.
    """
    scalar_path = _build_h5_path(main_dir, rs, Ne, q, vq)
    try:
        return get_variance(scalar_path, nequil)
    except (FileNotFoundError, OSError):
        return None, None


# ---------------------------------------------------------------------------
# E_all caching
# ---------------------------------------------------------------------------


def _cache_path(rs, Ne):
    """Return the file path for the cached E_all."""
    return f"./output/E_all_rs{rs:.1f}-n{Ne:d}.npz"


def _subset_E(E_full, dE_full, full_qlist, full_vqlist, req_qlist, req_vqlist):
    """
    Extract a (q, vq) subset from the full cached energy arrays.

    Returns
    -------
    (E_sub, dE_sub) : ndarrays (n_q_req, n_v_req), or None if any
                      requested point is missing from the cache.
    """
    full_vqlist = np.asarray(full_vqlist, dtype=float)
    req_vqlist = np.asarray(req_vqlist, dtype=float)

    q_lookup = {tuple(q): i for i, q in enumerate(full_qlist)}
    q_idx = []
    for q in req_qlist:
        key = tuple(q)
        if key not in q_lookup:
            return None
        q_idx.append(q_lookup[key])

    vq_idx = []
    for vq in req_vqlist:
        diffs = np.abs(full_vqlist - vq)
        best = np.argmin(diffs)
        if diffs[best] > 1e-8:
            return None
        vq_idx.append(best)

    return E_full[np.ix_(q_idx, vq_idx)], dE_full[np.ix_(q_idx, vq_idx)]


def get_E_all(main_dir, rs, Ne):
    """
    Discover ALL available (q, vq) combinations from the directory tree,
    extract E(vq) for every combination, and cache the full matrix to disk.

    Returns
    -------
    E_all : ndarray (n_q, n_v)
    dE_all : ndarray (n_q, n_v)
    qidx_list : list of [qx, qy, qz]
    vq_list : sorted list of float
    """

    qidx_list, vq_list = collect_q_and_vq(main_dir, rs, Ne)
    n_q, n_v = len(qidx_list), len(vq_list)
    E_all = np.zeros((n_q, n_v))  # np.full((n_q, n_v), np.nan)
    dE_all = np.zeros((n_q, n_v))  # np.full((n_q, n_v), np.nan)

    for iq, q in enumerate(qidx_list):
        for iv, vq in enumerate(vq_list):
            h5path = _build_h5_path(main_dir, rs, Ne, q, vq)
            try:
                E, dE = get_energy(h5path)
                E_all[iq, iv] = E / Ne
                dE_all[iq, iv] = dE / Ne
            except (FileNotFoundError, OSError) as e:
                raise FileNotFoundError(
                    f"Required file not found (check for still running DMC simulations or errors): {h5path}"
                ) from e

    os.makedirs("./output", exist_ok=True)
    np.savez(
        _cache_path(rs, Ne),
        E_all=E_all,
        dE_all=dE_all,
        qlist=qidx_list,
        vqlist=vq_list,
        main_dir=main_dir,
    )
    print(f"  [get_E_all] cached {n_q} q × {n_v} vq → {_cache_path(rs, Ne)}")
    return E_all, dE_all, qidx_list, vq_list


def load_or_compute_E(main_dir, rs, Ne, qidx_list, vq_list):
    """
    Return E_all and dE_all for the requested (qidx_list, vq_list) subset.

    Loads from cache if available and matching; otherwise rebuilds.

    Returns
    -------
    E_sub : ndarray (n_q_req, n_v_req)
    dE_sub : ndarray (n_q_req, n_v_req)
    """
    cache = _cache_path(rs, Ne)

    if os.path.exists(cache):
        data = np.load(cache, allow_pickle=True)
        stored_dir = str(data["main_dir"])

        if stored_dir == main_dir:
            full_qlist = [list(q) for q in data["qlist"]]
            result = _subset_E(
                data["E_all"],
                data["dE_all"],
                full_qlist,
                data["vqlist"],
                qidx_list,
                vq_list,
            )
            if result is not None:
                print(
                    f"  [cache hit] loaded {len(qidx_list)} q × "
                    f"{len(vq_list)} vq from {cache}"
                )
                return result
            else:
                print("  [cache miss] requested q/vq not fully covered — rebuilding")
        else:
            print(
                f"  [cache stale] main_dir changed "
                f"({stored_dir} → {main_dir}) — rebuilding"
            )
            full_qlist = [list(q) for q in data["qlist"]]
            result = _subset_E(
                data["E_all"],
                data["dE_all"],
                full_qlist,
                data["vqlist"],
                qidx_list,
                vq_list,
            )
            return result

    # Rebuild cache with ALL available data
    warnings.warn(
        f"Pre-computed cache not found at '{cache}'. "
        "Attempting to rebuild from raw QMC data in main_dir. "
        "Note: This will not work unless you have access to the raw data. ",
        UserWarning,
        stacklevel=2,
    )
    get_E_all(main_dir, rs, Ne)

    data = np.load(cache, allow_pickle=True)
    full_qlist = [list(q) for q in data["qlist"]]
    result = _subset_E(
        data["E_all"],
        data["dE_all"],
        full_qlist,
        data["vqlist"],
        qidx_list,
        vq_list,
    )
    if result is None:
        raise ValueError(
            f"Requested q/vq not found in directory tree at {main_dir}. "
            f"Available q: {full_qlist}, available vq: {list(data['vqlist'])}"
        )
    return result
