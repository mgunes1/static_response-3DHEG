"""
I/O utilities for the static response project.

Functions for reading QMC / DFT energies, parsing directory trees,
and caching the full energy matrix E(q, vq).
"""

import glob
import json
import os
import re
import warnings
import xml.etree.ElementTree as ET

import numpy as np

# ---------------------------------------------------------------------------
# Alpha manifest (per-(rs, Ne, q, vq) optimized trial-WF prefactor)
# ---------------------------------------------------------------------------


# Key format used by v5.0/workflow/snakefile aggregate_alpha_manifest.
def _manifest_key(q, vq):
    return "%d_%d_%d/%s" % (int(q[0]), int(q[1]), int(q[2]), ("%.5f" % float(vq)))


def load_alpha_manifest(main_dir, rs, Ne):
    """Load the alpha manifest written by the v5.0 workflow.

    Returns a dict {key: entry} where key is "qx_qy_qz/vq" and entry contains
    {alpha_opt, status, reason, ...}. Returns None if no manifest exists.
    """
    fpath = os.path.join(main_dir, f"alpha_manifest_rs{rs:.1f}-n{Ne:d}.json")
    if not os.path.isfile(fpath):
        return None
    with open(fpath) as f:
        data = json.load(f)
    return data.get("entries", {})


def _alpha_for(manifest, rs, Ne, q, vq):
    """Resolve alpha via manifest with guess_alpha2 fallback (warns)."""
    from .physics import guess_alpha2

    if manifest is not None:
        entry = manifest.get(_manifest_key(q, vq))
        if (
            entry is not None
            and entry.get("alpha_opt") is not None
            and entry.get("status") == "ok"
        ):
            return float(entry["alpha_opt"])
        if entry is not None and entry.get("status") in (
            "failed",
            "rescan_widen",
            "rescan_shift",
        ):
            warnings.warn(
                f"alpha manifest entry for q={q}, vq={vq} has status="
                f"{entry.get('status')}; falling back to guess_alpha2",
                UserWarning,
                stacklevel=3,
            )
    return float(guess_alpha2(rs, Ne, q))


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
        print(f"[cache miss] no q/vq cache at {runs_path} — parsing directory tree")
        rsn_dir = runs_path  # os.path.join(runs_path, f"rs{rs:.1f}-n{n}")
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


def _build_h5_path(
    main_dir, rs, Ne, q, vq, pwscf=False, variance=False, vmc=False, alpha_manifest=None
):
    """Construct the path to a stat.h5 file for given run parameters.

    alpha_manifest : dict or None
        If provided (from load_alpha_manifest), the per-(q,vq) optimized alpha
        is used; otherwise falls back to guess_alpha2 with a warning.
    """
    alpha = _alpha_for(alpha_manifest, rs, Ne, q, vq)
    ecut_pre, wf, dft_func, ts, ss, nw, tpmult = qmc_params_default(rs, Ne)
    if rs > 29:
        if pwscf:
            thr = 10
            pattern = (
                f"{main_dir}"
                f"qv{q[0]:d}_{q[1]:d}_{q[2]:d}-vq{vq:.5f}/"
                f"*-qa*-thr1.0d-10/"
                f"scf/qeout"
            )
            matches = glob.glob(pattern)

            if len(matches) == 0:
                raise FileNotFoundError(f"No match for pattern:\n{pattern}")
            elif len(matches) > 1:
                raise RuntimeError(f"Multiple matches found:\n{matches}")

            path = matches[0]
            return path
        else:
            if variance:
                return (
                    # f"{main_dir}/rs{rs:.1f}-n{Ne:d}/"
                    f"{main_dir}/"
                    f"qv{q[0]:d}_{q[1]:d}_{q[2]:d}-vq{vq:.5f}/"
                    f"{dft_func}-e{ecut_pre}-qa{alpha:.3f}-thr1.0d-10/"
                    f"{wf}-t{tpmult * Ne * rs}-ts{ts:.4f}-nw{nw}/"
                    f"qmc.s00{ss}.scalar.dat"
                )
            else:
                if vmc:
                    pattern = (
                        f"{main_dir}"
                        f"qv{q[0]:d}_{q[1]:d}_{q[2]:d}-vq{vq:.5f}/"
                        f"*-qa*-thr1.0d-10/"
                        f"opt-sj/"
                        f"qmc.s011.stat.h5"
                    )
                else:
                    pattern = (
                        f"{main_dir}"
                        f"qv{q[0]:d}_{q[1]:d}_{q[2]:d}-vq{vq:.5f}/"
                        f"*-qa*-thr1.0d-10/"
                        f"sj*/"
                        f"qmc.s002.stat.h5"
                    )

                matches = glob.glob(pattern)

                if len(matches) == 0:
                    raise FileNotFoundError(f"No match for pattern:\n{pattern}")
                elif len(matches) > 1:
                    raise RuntimeError(f"Multiple matches found:\n{matches}")

                path = matches[0]
                return path
    else:
        if pwscf:
            thr = 10
            pattern = (
                f"{main_dir}"
                f"qv{q[0]:d}_{q[1]:d}_{q[2]:d}-vq{vq:.5f}/"
                f"*-qa*-thr1.0d-10/"
                f"scf/qeout"
            )
            matches = glob.glob(pattern)

            if len(matches) == 0:
                raise FileNotFoundError(f"No match for pattern:\n{pattern}")
            elif len(matches) > 1:
                raise RuntimeError(f"Multiple matches found:\n{matches}")

            path = matches[0]
            return path
        else:
            if variance:
                return (
                    # f"{main_dir}/rs{rs:.1f}-n{Ne:d}/"
                    f"{main_dir}/"
                    f"qv{q[0]:d}_{q[1]:d}_{q[2]:d}-vq{vq:.4f}/"
                    f"{dft_func}-e{ecut_pre}-qa{alpha:.3f}-thr1.0d-10/"
                    f"{wf}-t{tpmult * Ne * rs}-ts{ts:.4f}-nw{nw}/"
                    f"qmc.s00{ss}.scalar.dat"
                )
            else:
                if vmc:
                    pattern = (
                        f"{main_dir}"
                        f"qv{q[0]:d}_{q[1]:d}_{q[2]:d}-vq{vq:.5f}/"
                        f"*-qa*-thr1.0d-10/"
                        f"opt-sj/"
                        f"qmc.s011.stat.h5"
                    )
                else:
                    pattern = (
                        f"{main_dir}"
                        f"qv{q[0]:d}_{q[1]:d}_{q[2]:d}-vq{vq:.5f}/"
                        f"*-qa*-thr1.0d-10/"
                        f"sj*/"
                        f"qmc.s002.stat.h5"
                    )

                matches = glob.glob(pattern)

                if len(matches) == 0:
                    raise FileNotFoundError(f"No match for pattern:\n{pattern}")
                elif len(matches) > 1:
                    raise RuntimeError(f"Multiple matches found:\n{matches}")

                path = matches[0]
                return path


def get_variance_for_run(main_dir, rs, Ne, q, vq, nequil=50, alpha_manifest=None):
    """
    Get variance for a specific QMC run.

    Returns
    -------
    (var, dvar) or (None, None) if the file is missing.
    """
    scalar_path = _build_h5_path(
        main_dir, rs, Ne, q, vq, variance=True, alpha_manifest=alpha_manifest
    )
    try:
        return get_variance(scalar_path, nequil)
    except (FileNotFoundError, OSError):
        return None, None


# ---------------------------------------------------------------------------
# E_all caching
# ---------------------------------------------------------------------------


def _cache_path(rs, Ne, pwscf=False, vmc=False, prefix=""):
    """Return the file path for the cached E_all."""
    if pwscf:
        return f"./output/{prefix}xE_DFT_rs{rs:.1f}-n{Ne:d}.npz"
    elif vmc:
        return f"./output/{prefix}xE_VMC_rs{rs:.1f}-n{Ne:d}.npz"
    else:
        return f"./output/{prefix}xE_QMC_rs{rs:.1f}-n{Ne:d}.npz"


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


def get_E_all(main_dir, rs, Ne, prefix="", alpha_manifest=None):
    """
    Discover ALL available (q, vq) combinations from the directory tree,
    extract E(vq) for every combination, and cache the full matrix to disk.

    Parameters
    ----------
    alpha_manifest : dict or None
        If provided (from load_alpha_manifest), per-point optimized alphas are
        used for path resolution; otherwise falls back to guess_alpha2.

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
    E_dft_all = np.zeros((n_q, n_v))  # np.full((n_q, n_v), np.nan)
    dE_dft_all = np.zeros((n_q, n_v))  # np.full((n_q, n_v), np.nan)
    E_vmc_all = np.zeros((n_q, n_v))  # np.full((n_q, n_v), np.nan)
    dE_vmc_all = np.zeros((n_q, n_v))  # np.full((n_q, n_v), np.nan)

    for iq, q in enumerate(qidx_list):
        for iv, vq in enumerate(vq_list):
            h5path = _build_h5_path(
                main_dir, rs, Ne, q, vq, alpha_manifest=alpha_manifest
            )
            h5path_dft = _build_h5_path(
                main_dir, rs, Ne, q, vq, pwscf=True, alpha_manifest=alpha_manifest
            )
            h5path_vmc = _build_h5_path(
                main_dir, rs, Ne, q, vq, vmc=True, alpha_manifest=alpha_manifest
            )
            try:
                E, dE = get_energy(h5path)
                E_all[iq, iv] = E / Ne
                dE_all[iq, iv] = dE / Ne

                E_vmc, dE_vmc = get_energy(h5path_vmc)
                E_vmc_all[iq, iv] = E_vmc / Ne
                dE_vmc_all[iq, iv] = dE_vmc / Ne

                E_dft, dE_dft = get_energy_pwscf(h5path_dft), 0.0
                E_dft_all[iq, iv] = E_dft / Ne
                dE_dft_all[iq, iv] = dE_dft / Ne
            except (FileNotFoundError, OSError) as e:
                raise FileNotFoundError(
                    f"Required file not found (check for still running DMC simulations or errors): {h5path}"
                ) from e

    os.makedirs("./output", exist_ok=True)
    np.savez(
        _cache_path(rs, Ne, prefix=prefix),
        E_all=E_all,
        dE_all=dE_all,
        qlist=qidx_list,
        vqlist=vq_list,
        main_dir=main_dir,
    )
    np.savez(
        _cache_path(rs, Ne, vmc=True, prefix=prefix),
        E_all=E_vmc_all,
        dE_all=dE_vmc_all,
        qlist=qidx_list,
        vqlist=vq_list,
        main_dir=main_dir,
    )
    np.savez(
        _cache_path(rs, Ne, pwscf=True, prefix=prefix),
        E_all=E_dft_all,
        dE_all=dE_dft_all,
        qlist=qidx_list,
        vqlist=vq_list,
        main_dir=main_dir,
    )
    print(
        f"  [get_E_all] cached {n_q} q × {n_v} vq → {_cache_path(rs, Ne, prefix=prefix)}"
    )
    return E_all, dE_all, qidx_list, vq_list


# ---------------------------------------------------------------------------
# v5.1 workflow: analytical alpha, no -thr1.0d-10 suffix
# ---------------------------------------------------------------------------


def _build_path(
    rs_n_dir,
    rs,
    Ne,
    q,
    vq,
    func="ni",
    ecut_pre=125,
    wf="sj",
    tpmult=15.625,
    nwalker=1024,
    variance=False,
    pwscf=False,
):
    """Construct output file path for the v5.1 (analytical-alpha) workflow.

    rs_n_dir : path to runs/rs{rs:.1f}-n{Ne:d}/
    """
    from .physics import get_alpha, q_over_kf

    ecut_pre, wf, func, ts, ss, nwalker, tpmult = qmc_params_default(rs, Ne)
    alpha = get_alpha(q_over_kf(rs, Ne, q), rs)
    alpha_str = "%.3f" % alpha
    ts = rs / (12 * Ne**0.5) if wf == "sjb" else rs / 20
    tproj = tpmult * Ne * rs
    ss = 3 if wf == "sjb" else 2

    base = (
        f"{rs_n_dir}/"
        f"qv{q[0]:d}_{q[1]:d}_{q[2]:d}-vq{vq:.5f}/"
        f"{func}-e{ecut_pre}-qa{alpha_str}"
    )

    if pwscf:
        # QE save directory XML (data-file-schema.xml for QE >= 6.4)
        pattern = f"{base}/scf/qeout/*.save/data-file-schema.xml"
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No DFT XML found: {pattern}")
        return matches[0]

    run_dir = f"{base}/{wf}-t{tproj}-ts{ts:.4f}-nw{nwalker}"
    if variance:
        return f"{run_dir}/qmc.s00{ss}.scalar.dat"
    return f"{run_dir}/qmc.s00{ss}.stat.h5"


def _parse_dmc_xml(dmc_xml_path):
    """Extract last QMC block params and first Jastrow block from dmc.xml.

    Returns (qmc_dict, jastrow_dict) where each dict has named params
    plus '_xml' key with the raw XML string of the whole block.
    """
    tree = ET.parse(dmc_xml_path)
    root = tree.getroot()

    qmc_els = root.findall("qmc")
    last_qmc = qmc_els[-1] if qmc_els else None
    qmc_info = {}
    if last_qmc is not None:
        for p in last_qmc.findall("parameter"):
            name = p.get("name", "")
            qmc_info[name] = p.text.strip() if p.text else ""
        qmc_info["_xml"] = ET.tostring(last_qmc, encoding="unicode")

    jastrow = root.find(".//jastrow")
    jastrow_info = {}
    if jastrow is not None:
        jastrow_info["type"] = jastrow.get("type", "")
        jastrow_info["name"] = jastrow.get("name", "")
        jastrow_info["function"] = jastrow.get("function", "")
        jastrow_info["_xml"] = ET.tostring(jastrow, encoding="unicode")

    return qmc_info, jastrow_info


def collect_and_save(
    rs_n_dir,
    rs,
    Ne,
    qidxl,
    vql,
    outpath=None,
    nequil=50,
):
    """Collect DMC energies, DFT energies, and variance for all (q, vq).

    Saves an npz to outpath (default: {rs_n_dir}/E_all.npz) with keys:
      rs, Ne, E_dmc_all, dE_dmc_all, var_all, dvar_all,
      qlist, vqlist, E_dft_all
    All energies and variances are per electron (divided by Ne).
    """
    ecut_pre, wf, func, ts, ss, nwalker, tpmult = qmc_params_default(rs, Ne)

    vql_f = [float(v) for v in vql]
    n_q, n_v = len(qidxl), len(vql_f)
    kw = dict(func=func, ecut_pre=ecut_pre, wf=wf, tpmult=tpmult, nwalker=nwalker)

    E_all = np.zeros((n_q, n_v))
    dE_all = np.zeros((n_q, n_v))
    E_dft_all = np.zeros((n_q, n_v))
    var_all = np.zeros((n_q, n_v))
    dvar_all = np.zeros((n_q, n_v))

    for iq, q in enumerate(qidxl):
        for iv, vq in enumerate(vql_f):
            h5 = _build_path(rs_n_dir, rs, Ne, q, vq, **kw)
            dft = _build_path(rs_n_dir, rs, Ne, q, vq, pwscf=True, **kw)
            sdat = _build_path(rs_n_dir, rs, Ne, q, vq, variance=True, **kw)

            E, dE = get_energy(h5, nequil)
            E_all[iq, iv] = E / Ne
            dE_all[iq, iv] = dE / Ne

            E_dft_all[iq, iv] = get_energy_pwscf(dft) / Ne

            var, dvar = get_variance(sdat, nequil)
            var_all[iq, iv] = var / Ne
            dvar_all[iq, iv] = dvar / Ne

    if outpath is None:
        outpath = os.path.join(rs_n_dir, "E_all.npz")

    os.makedirs(os.path.dirname(os.path.abspath(outpath)), exist_ok=True)
    np.savez(
        outpath,
        rs=rs,
        Ne=Ne,
        E_dmc_all=E_all,
        dE_dmc_all=dE_all,
        var_all=var_all,
        dvar_all=dvar_all,
        qlist=qidxl,
        vqlist=vql_f,
        E_dft_all=E_dft_all,
    )
    return outpath


def load_or_compute_E(main_dir, rs, Ne, qidx_list, vq_list, alpha_manifest=None):
    """
    Return E_all and dE_all for the requested (qidx_list, vq_list) subset.

    Loads from cache if available and matching; otherwise rebuilds.

    Parameters
    ----------
    alpha_manifest : dict or None
        Forwarded to get_E_all when a cache rebuild is needed.

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
    get_E_all(main_dir, rs, Ne, alpha_manifest=alpha_manifest)

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
