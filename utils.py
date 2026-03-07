# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes ,inset_axes, mark_inset
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np


def get_energy_pwscf(path):
    # Resolve directory to file
    xml_path = path
    if os.path.isdir(xml_path):
        xml_path = os.path.join(xml_path, "pwscf.xml")

    # Check existence
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"File not found: {xml_path}")

    # Stream-parse XML to find <etot>
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


def get_energy(h5_file, nequil=50):
    """
    Extract energy and error from QMCPACK stat.h5 file.
    Uses proper autocorrelation correction via harvest_qmcpack.

    Args:
        h5_file (str): path to stat.h5 file
        nequil (int): number of equilibration blocks to discard (default: 50)

    Returns:
        tuple: (E, dE) energy mean and error accounting for autocorrelation
    """
    import h5py

    try:
        from qharv.reel.stat_h5 import mean_and_err

        with h5py.File(h5_file, "r") as f:
            E, dE = mean_and_err(f, "LocalEnergy", nequil)
        return E[0], dE[0]
    except ImportError:
        raise ImportError(
            "Cannot call mean_and_err. Please ensure qharv is installed and the h5 file is valid. Either qharv.reel.stat_h5 module not found, or there is an error in h5_file."
        )


def get_chi0_q(
    main_dir,
    Ne,
    rs,
    vq_list,
    qidx_list,
    ecut_pre=125,
    wf="sj",
    from_pwscf=False,
    prefix="",
    dft_func="ni",
    vq_fit="quadratic",
):
    from scipy.optimize import curve_fit

    chi_q = np.zeros(len(qidx_list))
    dchi_q = np.zeros(len(qidx_list))
    for iq in range(len(qidx_list)):
        E_list = []
        q = qidx_list[iq]
        alpha = guess_alpha2(rs, Ne, qidx_list[iq])
        for vq in vq_list:
            dir_h5 = f"{main_dir}/{prefix}rs{rs:.1f}-n{Ne:d}/qv{q[0]:d}_{q[1]:d}_{q[2]:d}-vq{vq:.4f}/{dft_func}-e{ecut_pre}-qa{alpha:.3f}-thr1.0d-10/scf/qeout"
            E = get_energy_pwscf(dir_h5)
            E_list.append(E / Ne)

        def quartic(x, A, B, e0):
            return e0 + A * x**2 + B * x**4

        def quadratic(x, A, e0):
            return e0 + A * x**2

        if vq_fit == "quadratic":
            popt, pcov = curve_fit(quadratic, np.array(vq_list) * alpha, E_list)
            A = popt[0]

        elif vq_fit == "quartic":
            popt, pcov = curve_fit(quartic, np.array(vq_list) * alpha, E_list)
            A = popt[0]

        n0 = get_gas_params(rs, Ne)[1]
        chi_q[iq] = A * n0
        dchi_q[iq] = 0
    return chi_q, dchi_q


def fit_funcs(vq_fit):
    def quartic(x, A, B, e0):
        return e0 + A * x**2 + B * x**4

    def quadratic(x, A, e0):
        return e0 + A * x**2

    fit_funcs = {"quartic": quartic, "quadratic": quadratic}
    try:
        return fit_funcs[vq_fit]
    except KeyError:
        raise ValueError(
            f"Invalid fit type '{vq_fit}'. Choose 'quartic' or 'quadratic'. If new form, add it to fit_funcs."
        )


def fit_quality_report(
    poptl, pcovl, E_all, dE_all, vq_arr, qidx_list, vq_fit, verbose=False
):
    """
    Generate fit quality diagnostic from a produced fit of E(vq).
    """
    fit_quality = []
    for iq in range(len(qidx_list)):
        q = qidx_list[iq]
        E_arr = E_all[iq, :]
        dE_arr = dE_all[iq, :]
        popt = poptl[iq]
        pcov = pcovl[iq]
        E_pred = fit_funcs(vq_fit)(vq_arr, *popt)
        residuals = E_arr - E_pred
        n_params = 3 if vq_fit == "quartic" else 2  # (e0, A, B) or (e0, A)
        ndof = max(len(vq_arr) - n_params, 1)
        chi2_red = np.sum((residuals / dE_arr) ** 2) / ndof

        # Signal-to-noise: how far does the parabola rise above noise?
        mean_dE = np.mean(dE_arr[1:]) if len(dE_arr) > 1 else dE_arr[0]
        delta_E = np.max(np.abs(E_arr[1:] - E_arr[0])) if len(E_arr) > 1 else 0
        snr = delta_E / mean_dE if mean_dE > 0 else 0.0

        # Significance of curvature
        A = popt[0]
        try:
            dA = np.sqrt(pcov[0, 0])
        except ImportError:
            dA = 1e8
        A_sig = abs(A) / dA if dA > 0 else np.inf

        # Reliability flag
        reliable = (snr > 5.0) and (A_sig > 2.0) and (chi2_red < 10.0)

        info = {
            "reduced_chi2": chi2_red,
            "signal_to_noise": snr,
            "A_significance": A_sig,
            "reliable": reliable,
            "E_list": E_arr,
            "dE_list": dE_arr,
            "popt": popt,
            "pcov": pcov,
            "q": q,
        }

        fit_quality.append(info)

        if verbose:
            tag = "OK" if info["reliable"] else "UNRELIABLE"
            print(
                f"  q={q}  chi/n0={A:.4e}  SNR={info['signal_to_noise']:.1f}  "
                f"|A|/dA={info['A_significance']:.1f}  "
                f"chi2r={info['reduced_chi2']:.2f}  [{tag}]"
            )

    return fit_quality


def qmc_params_default(rs, Ne):
    # Load simulation parameters
    ecut_pre = 125
    wf = "sj"
    dft_func = "ni"
    ts = rs / (12 * Ne**0.5) if wf == "sjb" else rs / 20
    ss = 3 if wf == "sjb" else 2
    nw = 1024
    tpmult = 15.625

    return ecut_pre, wf, dft_func, ts, ss, nw, tpmult


def _cache_path(rs, Ne):
    """Return the path for the cached E_all file."""
    return f"./output/E_all_rs{rs:.1f}-n{Ne:d}.npz"


def _subset_E(E_full, dE_full, full_qlist, full_vqlist, req_qlist, req_vqlist):
    """
    Extract the subset of (q, vq) from the full cached arrays.

    Parameters
    ----------
    E_full, dE_full : ndarray (n_q_full, n_v_full)
    full_qlist : list of [qx, qy, qz]
    full_vqlist : array of floats
    req_qlist : list of [qx, qy, qz]  — requested q-points
    req_vqlist : array-like of floats  — requested vq values

    Returns
    -------
    (E_sub, dE_sub) : ndarrays (n_q_req, n_v_req), or None if any
                      requested point is missing.
    """
    full_vqlist = np.asarray(full_vqlist, dtype=float)
    req_vqlist = np.asarray(req_vqlist, dtype=float)

    # --- row indices (q-points) ---
    # Build lookup: tuple(q) -> index
    q_lookup = {tuple(q): i for i, q in enumerate(full_qlist)}
    q_idx = []
    for q in req_qlist:
        key = tuple(q)
        if key not in q_lookup:
            return None
        q_idx.append(q_lookup[key])

    # --- column indices (vq values, with float tolerance) ---
    vq_idx = []
    for vq in req_vqlist:
        diffs = np.abs(full_vqlist - vq)
        best = np.argmin(diffs)
        if diffs[best] > 1e-8:
            return None
        vq_idx.append(best)

    E_sub = E_full[np.ix_(q_idx, vq_idx)]
    dE_sub = dE_full[np.ix_(q_idx, vq_idx)]
    return E_sub, dE_sub


def get_E_all(main_dir, rs, Ne):
    """
    Discover ALL available (q, vq) combinations from the directory tree,
    extract E(vq) for every combination, and cache the full matrix to disk.

    The cache file is ``./output/E_all_rs{rs}-n{Ne}.npz`` and stores
    ``E_all``, ``dE_all``, ``qlist``, ``vqlist``, and ``main_dir``.

    Returns
    -------
    E_all : ndarray, shape (n_q, n_v)
    dE_all : ndarray, shape (n_q, n_v)
    qidx_list : list of [qx, qy, qz]
    vq_list : sorted list of floats
    """
    qidx_list, vq_list = collect_q_and_vq(main_dir, rs, Ne)
    n_q = len(qidx_list)
    n_v = len(vq_list)
    E_all = np.full((n_q, n_v), np.nan)
    dE_all = np.full((n_q, n_v), np.nan)
    ecut_pre, wf, dft_func, ts, ss, nw, tpmult = qmc_params_default(rs, Ne)

    for iq in range(n_q):
        q = qidx_list[iq]
        alpha = guess_alpha2(rs, Ne, q)
        for iv, vq in enumerate(vq_list):
            dir_h5 = (
                f"{main_dir}/rs{rs:.1f}-n{Ne:d}/"
                f"qv{q[0]:d}_{q[1]:d}_{q[2]:d}-vq{vq:.4f}/"
                f"{dft_func}-e{ecut_pre}-qa{alpha:.3f}-thr1.0d-10/"
                f"{wf}-t{tpmult * Ne * rs}-ts{ts:.4f}-nw{nw}/"
                f"qmc.s00{ss}.stat.h5"
            )
            try:
                E, dE = get_energy(dir_h5)
                E_all[iq, iv] = E / Ne
                dE_all[iq, iv] = dE / Ne
            except (FileNotFoundError, OSError):
                print(f"  [get_E_all] skipping missing file: {dir_h5}")

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

    1. If a cache file exists and ``main_dir`` matches, try to subset.
    2. If the cache is missing, stale, or doesn't cover the request,
       call ``get_E_all`` to rebuild it from scratch, then subset.

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
                    f"  [cache hit] loaded {len(qidx_list)} q × {len(vq_list)} vq "
                    f"from {cache}"
                )
                return result
            else:
                print("  [cache miss] requested q/vq not fully covered — rebuilding")
        else:
            print(
                f"  [cache stale] main_dir changed "
                f"({stored_dir} → {main_dir}) — rebuilding"
            )

    # Rebuild cache with ALL available data
    _, _, full_qlist, full_vqlist = get_E_all(main_dir, rs, Ne)

    # Now reload and subset
    data = np.load(cache, allow_pickle=True)
    full_qlist_loaded = [list(q) for q in data["qlist"]]
    result = _subset_E(
        data["E_all"],
        data["dE_all"],
        full_qlist_loaded,
        data["vqlist"],
        qidx_list,
        vq_list,
    )
    if result is None:
        raise ValueError(
            f"Requested q/vq not found in the directory tree at {main_dir}. "
            f"Available q: {full_qlist_loaded}, available vq: {list(data['vqlist'])}"
        )
    return result


def fit_E_of_vq(E_arr, dE_arr, vq_arr, fit_func):
    """
    Fit E_arr as a function of vq for a chosen form: quadratic or quartic, i.e. E0 + Ax^2 [+ Bx^4]
    """
    from scipy.optimize import curve_fit

    popt, pcov = curve_fit(fit_func, vq_arr, E_arr, sigma=dE_arr, absolute_sigma=True)
    return popt, pcov


"""
 main_dir,
    Ne,
    rs,
    vq_list,
    qidx_list,
    ecut_pre=125,
    wf="sj",
    dft_func="ni",
    verbose=False,
    vq_fit="quadratic",
"""


def get_chi_q(
    main_dir,
    Ne,
    rs,
    vq_list,
    qidx_list,
    verbose=False,
    vq_fit="quadratic",
):
    """
    Extract chi(q) from quartic fit of E(vq) for each q-point.

    Returns
    -------
    chi_q : array
        Response function for each q-point: chi = A * n0
    dchi_q : array
        Statistical error on chi (from covariance of fit)
    fit_quality : list of dict
        Per-q-point diagnostics:
          - 'reduced_chi2': reduced chi-squared of the quartic fit
          - 'signal_to_noise': (max|E(vq)-E(0)|) / mean(dE)
          - 'A_significance': |A| / dA
          - 'reliable': bool — True if the fit is considered trustworthy
          - 'E_list', 'dE_list': raw data arrays
          - 'popt': fit parameters [A, B]
          - 'q': q-index
    """

    chi_q = np.zeros(len(qidx_list))
    dchi_q = np.zeros(len(qidx_list))

    n0 = get_gas_params(rs, Ne)[1]

    E_all, dE_all = load_or_compute_E(main_dir, rs, Ne, qidx_list, vq_list)

    poptl = []  # np.zeros(len(qidx_list))
    pcovl = []  # np.zeros(len(qidx_list))
    for iq in range(len(qidx_list)):
        E_arr = E_all[iq, :]
        dE_arr = dE_all[iq, :]
        vq_arr = np.array(vq_list)

        # --- fitting ---
        popt, pcov = fit_E_of_vq(E_arr, dE_arr, vq_arr, fit_funcs(vq_fit))
        A = popt[0]
        try:
            dA = np.sqrt(pcov[0, 0])
        except ImportError:
            dA = 1e8
        poptl.append(popt)
        pcovl.append(pcov)

        # --- storing ---
        chi_q[iq] = A * n0
        dchi_q[iq] = dA * n0

    # --- fit quality diagnostics ---
    print("FIT REPORT GENERATING")
    fit_quality = fit_quality_report(
        poptl, pcovl, E_all, dE_all, vq_arr, qidx_list, vq_fit, verbose
    )

    return chi_q, dchi_q, fit_quality


def bootstrap_chi_error(
    fit_quality_list,
    vq_arr,
    n0,
    fs_correct_fn,
    fit_type="quadratic",
    n_boot=500,
    seed=None,
):
    """
    Parametric bootstrap for chi(q) error bars.

    For each bootstrap iteration, perturbs E_i -> E_i + dE_i * N(0,1)
    and fits E(vq) = E0 + A*vq^2 [+ B*vq^4] with E0 as a free parameter
    via curve_fit. This correctly propagates E0 uncertainty into dA.

    Parameters
    ----------
    fit_quality_list : list of dict
        From get_chi_q; each entry must contain 'E_list' and 'dE_list'.
    vq_arr : array
        The vq values used in the fit.
    n0 : float
        Number density.
    fs_correct_fn : callable
        chi_arr -> chi_corr_arr  (finite-size correction).
    fit_type : str
        'quadratic' or 'quartic'.
    n_boot : int
        Number of bootstrap iterations.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    boot_err : array (n_q,)
    boot_samples : array (n_boot, n_q)
    """
    rng = np.random.default_rng(seed)
    vq_arr = np.asarray(vq_arr, dtype=float)
    n_v = len(vq_arr)
    n_q = len(fit_quality_list)

    # Stack energies and errors: shape (n_q, n_v)
    E_all = np.array([fq["E_list"] for fq in fit_quality_list])  # (n_q, n_v)
    dE_all = np.array([fq["dE_list"] for fq in fit_quality_list])  # (n_q, n_v)

    # Generate ALL perturbations at once: shape (n_boot, n_q, n_v)
    noise = rng.standard_normal((n_boot, n_q, n_v))
    # Perturbed energies: broadcast (n_boot, n_q, n_v)
    E_pert = E_all[np.newaxis, :, :] + dE_all[np.newaxis, :, :] * noise

    chi_boot = np.empty((n_boot, n_q))

    for b in range(n_boot):
        for iq in range(n_q):
            E_b = E_pert[b, iq, :]  # (n_v,) perturbed energies for this q
            sigma_b = dE_all[iq, :]  # (n_v,) errors (fixed across boots)
            fit_func = fit_funcs(fit_type)
            popt, _ = fit_E_of_vq(E_b, sigma_b, vq_arr, fit_func)
            try:
                chi_boot[b, iq] = popt[0] * n0
            except RuntimeError:
                chi_boot[b, iq] = np.nan

    # Apply FS correction to each row
    boot_samples = np.empty_like(chi_boot)
    for b in range(n_boot):
        try:
            boot_samples[b] = fs_correct_fn(chi_boot[b])
        except Exception:
            boot_samples[b] = np.nan

    boot_err = np.nanstd(boot_samples, axis=0)
    return boot_err, boot_samples


def load_raw_blocks(
    main_dir,
    Ne,
    rs,
    vq_list,
    qidx_list,
    ecut_pre=125,
    dft_func="ni",
    wf="sj",
    nequil=50,
):
    """
    Load raw QMC block energies for all (q, vq) combinations.

    Reads each stat.h5 file once and returns the per-block energies
    (after equilibration), divided by Ne.

    Parameters
    ----------
    main_dir, Ne, rs, vq_list, qidx_list, ecut_pre, dft_func, wf :
        Same as get_chi_q.
    nequil : int
        Number of equilibration blocks to discard (default: 50).

    Returns
    -------
    raw_blocks : ndarray, shape (n_q, n_v, n_blocks)
        Per-block energies E/Ne after equilibration.
        n_blocks = total_blocks - nequil (typically 350).
    """
    import h5py

    ts = rs / (12 * Ne**0.5) if wf == "sjb" else rs / 20
    ss = 3 if wf == "sjb" else 2
    nw = 1024
    tpmult = 15.625

    n_q = len(qidx_list)
    n_v = len(vq_list)

    # Read first file to determine n_blocks
    q0 = qidx_list[0]
    alpha0 = guess_alpha2(rs, Ne, q0)
    path0 = (
        f"{main_dir}/rs{rs:.1f}-n{Ne:d}/"
        f"qv{q0[0]:d}_{q0[1]:d}_{q0[2]:d}-vq{vq_list[0]:.4f}/"
        f"{dft_func}-e{ecut_pre}-qa{alpha0:.3f}-thr1.0d-10/"
        f"{wf}-t{tpmult * Ne * rs}-ts{ts:.4f}-nw{nw}/"
        f"qmc.s00{ss}.stat.h5"
    )
    with h5py.File(path0, "r") as f:
        total_blocks = f["LocalEnergy/value"].shape[0]
    n_blocks = total_blocks - nequil

    raw_blocks = np.empty((n_q, n_v, n_blocks), dtype=np.float64)

    for iq, q in enumerate(qidx_list):
        alpha = guess_alpha2(rs, Ne, q)
        for iv, vq in enumerate(vq_list):
            h5path = (
                f"{main_dir}/rs{rs:.1f}-n{Ne:d}/"
                f"qv{q[0]:d}_{q[1]:d}_{q[2]:d}-vq{vq:.4f}/"
                f"{dft_func}-e{ecut_pre}-qa{alpha:.3f}-thr1.0d-10/"
                f"{wf}-t{tpmult * Ne * rs}-ts{ts:.4f}-nw{nw}/"
                f"qmc.s00{ss}.stat.h5"
            )
            with h5py.File(h5path, "r") as f:
                vals = f["LocalEnergy/value"][nequil:, 0]
            raw_blocks[iq, iv, : len(vals)] = vals / Ne

    return raw_blocks


def get_qs(qidx, Ne, rs):
    L = (4 * np.pi / 3 * rs**3 * Ne) ** (1 / 3)
    qs = 2 * np.pi / L * np.array(qidx)

    return np.linalg.norm(qs, axis=1)


def gen_qidx(mq):
    """
    Generate unique q-indices, returning only one representative per shell (unique |q|²).
    Sorted by increasing magnitude.
    """
    ql = []
    for qx in range(mq):
        for qy in range(qx + 1):
            for qz in range(qy + 1):
                q_sq = qx**2 + qy**2 + qz**2
                ql.append((q_sq, [qx, qy, qz]))

    # Sort by q_sq magnitude
    ql.sort(key=lambda x: x[0])

    # Remove duplicates: keep only first occurrence of each |q|²
    seen_q_sq = set()
    unique_ql = []
    for q_sq, qidx in ql:
        if q_sq not in seen_q_sq and q_sq > 0:  # exclude [0,0,0]
            seen_q_sq.add(q_sq)
            unique_ql.append(qidx)

    return unique_ql


def guess_alpha(rs, nelec, qidx):
    kf = (9 * np.pi / 4) ** (1 / 3) / rs  # 3D gas
    alat = (nelec * 4 * np.pi / 3) ** (1.0 / 3) * rs
    blat = 2 * np.pi / alat
    qvec = blat * np.array(qidx)
    qmag = np.linalg.norm(qvec)
    alpha = np.tanh(qmag / kf)
    return alpha


def guess_alpha2(rs, nelec, qidx):
    a = 1.2
    kf = (9 * np.pi / 4) ** (1 / 3) / rs  # 3D gas
    alat = (nelec * 4 * np.pi / 3) ** (1.0 / 3) * rs
    blat = 2 * np.pi / alat
    qvec = blat * np.array(qidx)
    qmag = np.linalg.norm(qvec)
    alpha = 2 / (1 + np.exp(-a * (qmag / kf) ** 2)) - 1
    return alpha


def get_gas_params(rs, Ne):
    L = (4 * np.pi / 3 * rs**3 * Ne) ** (1 / 3)
    n0 = 1.0 / (rs**3.0 * 4.0 * np.pi / 3.0)
    kF = (3 * np.pi**2 * n0) ** (1 / 3)
    NF = kF / (1 * np.pi**2)
    return kF, n0, NF, L


def collect_q_and_vq(
    runs_path,
    rs,
    n,
):
    """
    Parse q indices and vq values from a QMC run directory tree.

    Parameters
    ----------
    runs_path : str or Path
        Path to the `runs` directory
    rs : float or str
        rs value (e.g. 5.0)
    n : int
        particle number (e.g. 294)

    Returns
    -------
    qidx_list : list of list[int, int, int]
        Unique q indices [qx, qy, qz]
    vq_list : list of float
        Sorted unique vq values
    """
    import os
    import re

    rsn_dir = os.path.join(runs_path, f"rs{rs:.1f}-n{n}")

    if not os.path.isdir(rsn_dir):
        raise FileNotFoundError(f"Directory not found: {rsn_dir}")

    # Pre-compiled pattern for speed
    q_pattern = re.compile(r"qv(-?\d+)_(-?\d+)_(-?\d+)-vq([\d.]+)$")

    qidx_set = set()
    vq_set = set()

    # os.scandir is faster than pathlib.iterdir
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


def get_variance(scalar_file, nequil=50):
    """
    Extract variance and error from QMCPACK scalar.dat file.
    Uses proper autocorrelation correction via qharv.reel.scalar_dat.

    Args:
        scalar_file (str): path to scalar.dat file
        nequil (int): number of equilibration blocks to discard (default: 50)

    Returns:
        tuple: (var, dvar) variance mean and error accounting for autocorrelation
    """
    try:
        from qharv.reel.scalar_dat import read, single_column

        df = read(scalar_file)
        var, dvar, _ = single_column(df, "Variance", nequil)
        return var, dvar
    except ImportError:
        print(
            "COULD NOT FIND THE qharv.reel.scalar_dat MODULE. USING FALLBACK METHOD FOR VARIANCE CALCULATION."
        )
        # Fallback: manual calculation without autocorrelation
        import pandas as pd

        df = pd.read_csv(scalar_file, comment="#", sep=r"\s+", header=None)
        # Try to get column names from file
        with open(scalar_file, "r") as f:
            header = f.readline().strip()
        if header.startswith("#"):
            cols = header.replace("#", "").split()
            df.columns = cols[: len(df.columns)]
        # Calculate variance from LocalEnergy_sq - LocalEnergy^2 if needed
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


def get_variance_for_run(
    main_dir, rs, Ne, q, vq, ecut_pre=125, dft_func="ni", wf="sj", nequil=50
):
    """
    Get variance for a specific QMC run.

    Args:
        main_dir: base runs directory
        rs, Ne: system parameters
        q: q-index [qx, qy, qz]
        vq: perturbation strength
        ecut_pre, dft_func, wf: run parameters
        nequil: equilibration blocks

    Returns:
        tuple: (var, dvar) or (None, None) if file not found
    """
    alpha = guess_alpha2(rs, Ne, q)
    ts = rs / (12 * Ne**0.5) if wf == "sjb" else rs / 20
    ss = 3 if wf == "sjb" else 2
    nw = 1024
    tpmult = 15.625

    scalar_path = (
        f"{main_dir}/rs{rs:.1f}-n{Ne:d}/"
        f"qv{q[0]:d}_{q[1]:d}_{q[2]:d}-vq{vq:.4f}/"
        f"{dft_func}-e{ecut_pre}-qa{alpha:.3f}-thr1.0d-10/"
        f"{wf}-t{tpmult * Ne * rs}-ts{ts:.4f}-nw{nw}/qmc.s00{ss}.scalar.dat"
    )

    try:
        return get_variance(
            scalar_path, nequil
        )  # [0]/Ne, get_variance(scalar_path, nequil)[1]/Ne
    except (FileNotFoundError, OSError):
        return None, None


def plot_variance(
    main_dir,
    rs,
    Ne,
    qidx_list=None,
    vq_list=None,
    ecut_pre=125,
    dft_func="ni",
    wf="sj",
    nequil=50,
    vs="vq",
    fixed_val=None,
    ax=None,
    **plot_kwargs,
):
    """
    Plot variance vs either q (for fixed vq) or vs vq (for fixed q).

    Args:
        main_dir: base runs directory
        rs, Ne: system parameters
        qidx_list: list of q-indices [[qx,qy,qz], ...]
        vq_list: list of vq values
        ecut_pre, dft_func, wf, nequil: run parameters
        vs: 'vq' to plot vs vq (need fixed q), 'q' to plot vs q (need fixed vq)
        fixed_val: the fixed q-index (if vs='vq') or fixed vq (if vs='q')
        ax: matplotlib axes (created if None)
        **plot_kwargs: passed to errorbar

    Returns:
        ax, x_vals, var_vals, dvar_vals
    """

    if ax is None:
        fig, ax = plt.subplots(dpi=150)

    var_vals = []
    dvar_vals = []
    x_vals = []

    if vs == "vq":
        # Plot variance vs vq for a fixed q
        if fixed_val is None:
            raise ValueError("Need fixed q-index for vs='vq'")
        q = fixed_val
        for vq in vq_list:
            var, dvar = get_variance_for_run(
                main_dir, rs, Ne, q, vq, ecut_pre, dft_func, wf, nequil
            )
            if var is not None:
                x_vals.append(vq)
                var_vals.append(var)
                dvar_vals.append(dvar)
        xlabel = r"$v_q$"
        title = rf"$N_e={Ne}$,$r_s={rs}$, $q=({q[0]},{q[1]},{q[2]})$, $\alpha={guess_alpha2(rs, Ne, q):.3f}$"

    elif vs == "q":
        # Plot variance vs |q| for a fixed vq
        kF, n0, NF, L = get_gas_params(rs, Ne)
        if fixed_val is None:
            raise ValueError("Need fixed vq for vs='q'")
        vq = fixed_val
        q_mags = get_qs(qidx_list, Ne, rs)
        for i, q in enumerate(qidx_list):
            var, dvar = get_variance_for_run(
                main_dir, rs, Ne, q, vq, ecut_pre, dft_func, wf, nequil
            )
            if var is not None:
                x_vals.append(q_mags[i] / kF)  # plot vs |q|/kF
                var_vals.append(var)
                dvar_vals.append(dvar)
        xlabel = r"$|q|/k_F$"
        title = rf"$N_e={Ne}$,$r_s={rs}$, $v_q={vq:.4f}$"
    else:
        raise ValueError("vs must be 'vq' or 'q'")

    x_vals = np.array(x_vals)
    var_vals = np.array(var_vals)
    dvar_vals = np.array(dvar_vals)

    # Sort by x
    order = np.argsort(x_vals)
    x_vals = x_vals[order]
    var_vals = var_vals[order]
    dvar_vals = dvar_vals[order]

    ax.errorbar(x_vals, var_vals, yerr=dvar_vals, fmt="o-", capsize=3, **plot_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Variance")
    ax.set_title(title)

    return ax, x_vals, var_vals, dvar_vals


def analyze_vq_range(
    rs,
    Ne,
    qidx_list,
    vq_max,
    n_vq=7,
    dE_typical=None,
    chi_ref="Moroni",
    snr_threshold=3.0,
    verbose=True,
):
    """
    Analyze whether a given vq range provides sufficient signal-to-noise
    for extracting chi(q) at each q-point, using the known reference chi
    (Moroni or Corradini) to estimate the expected energy variation.

    The idea: for a quartic fit E(vq) = E0 + (chi/n0)*vq^2 + B*vq^4,
    the energy variation at vq_max is approximately:
        delta_E ≈ |chi/n0| * vq_max^2

    If delta_E / dE < snr_threshold, the signal is buried in noise
    and the fit will be unreliable.

    Parameters
    ----------
    rs : float
        Wigner-Seitz radius
    Ne : int
        Number of electrons
    qidx_list : list of [qx, qy, qz]
        q-point indices to analyze
    vq_max : float
        Maximum vq in the proposed range
    n_vq : int
        Number of vq points (including 0) in the range
    dE_typical : float or None
        Typical DMC error bar on E/Ne (Hartree per electron).
        If None, uses a rough estimate: 5e-5 * rs (scales with rs).
    chi_ref : str
        Reference model: 'Moroni' or 'Corradini'
    snr_threshold : float
        Minimum acceptable signal-to-noise ratio (default: 3)
    verbose : bool
        Print per-q analysis

    Returns
    -------
    analysis : list of dict
        Per-q-point analysis with keys:
          - 'q': q-index
          - 'q_over_kF': |q|/kF
          - 'chi_ref_over_n0': expected |chi/n0| from reference
          - 'delta_E': expected energy variation at vq_max
          - 'dE': typical error bar
          - 'snr': delta_E / dE
          - 'acceptable': bool (snr > snr_threshold)
          - 'vq_max_suggested': suggested vq_max to achieve snr_threshold
            (only if current range is unacceptable)
    """
    from chi_utils import G_Moroni, chi0q, corradini_pz

    kF, n0, NF, L = get_gas_params(rs, Ne)
    ql = get_qs(qidx_list, Ne, rs)

    # Reference chi
    if chi_ref == "Moroni":
        Vc = 4 * np.pi / ql**2
        chi0 = chi0q(ql, Ne, rs)
        G = G_Moroni(rs, ql)
        fxc = -Vc * G
        chi_ref_vals = chi0 / (1 - chi0 * (Vc + fxc))
    elif chi_ref == "Corradini":
        Vc = 4 * np.pi / ql**2
        chi0 = chi0q(ql, Ne, rs)
        fxc = corradini_pz(rs, ql)
        chi_ref_vals = chi0 / (1 - chi0 * (Vc + fxc))
    else:
        raise ValueError(f"Unknown chi_ref: {chi_ref}")

    if dE_typical is None:
        dE_typical = 5e-5 * rs  # rough empirical scaling

    analysis = []
    if verbose:
        print(
            f"{'q':>12s} {'q/kF':>6s} {'|chi/n0|':>10s} {'delta_E':>10s} "
            f"{'dE':>10s} {'SNR':>6s} {'status':>12s} {'vq_max_sug':>12s}"
        )
        print("-" * 82)

    for iq, q in enumerate(qidx_list):
        chi_over_n0 = abs(chi_ref_vals[iq] / n0)
        delta_E = chi_over_n0 * vq_max**2
        snr = delta_E / dE_typical
        acceptable = snr > snr_threshold

        # Suggest a vq_max that would achieve the threshold
        if not acceptable and chi_over_n0 > 0:
            vq_max_sug = np.sqrt(snr_threshold * dE_typical / chi_over_n0)
        else:
            vq_max_sug = vq_max

        info = {
            "q": q,
            "q_over_kF": ql[iq] / kF,
            "chi_ref_over_n0": chi_over_n0,
            "delta_E": delta_E,
            "dE": dE_typical,
            "snr": snr,
            "acceptable": acceptable,
            "vq_max_suggested": vq_max_sug,
        }
        analysis.append(info)

        if verbose:
            status = "OK" if acceptable else "TOO NOISY"
            sug_str = f"{vq_max_sug:.6f}" if not acceptable else "---"
            print(
                f"  {str(q):>10s} {ql[iq] / kF:6.3f} {chi_over_n0:10.4e} "
                f"{delta_E:10.4e} {dE_typical:10.4e} {snr:6.1f} "
                f"{status:>12s} {sug_str:>12s}"
            )

    if verbose:
        n_bad = sum(1 for a in analysis if not a["acceptable"])
        print(
            f"\n  Summary: {len(analysis) - n_bad}/{len(analysis)} q-points acceptable "
            f"with vq_max={vq_max:.6f}, SNR threshold={snr_threshold}"
        )
        if n_bad > 0:
            max_sug = max(
                a["vq_max_suggested"] for a in analysis if not a["acceptable"]
            )
            print(f"  Suggested vq_max to make all points acceptable: {max_sug:.6f}")
            print(
                "  WARNING: large vq_max may enter non-quadratic regime. "
                "Use quartic fit and verify with chi0 from DFT."
            )

    return analysis


def get_chi_q_old(
    main_dir,
    Ne,
    rs,
    vq_list,
    qidx_list,
    ecut_pre=125,
    wf="sj",
    dft_func="ni",
    verbose=False,
    vq_fit="quadratic",
):
    """
    !!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!ABANDONED!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!
    Extract chi(q) from quartic fit of E(vq) for each q-point.

    Returns
    -------
    chi_q : array
        Response function for each q-point: chi = A * n0
    dchi_q : array
        Statistical error on chi (from covariance of fit)
    fit_quality : list of dict
        Per-q-point diagnostics:
          - 'reduced_chi2': reduced chi-squared of the quartic fit
          - 'signal_to_noise': (max|E(vq)-E(0)|) / mean(dE)
          - 'A_significance': |A| / dA
          - 'reliable': bool — True if the fit is considered trustworthy
          - 'E_list', 'dE_list': raw data arrays
          - 'popt': fit parameters [A, B]
          - 'q': q-index
    """
    from scipy.optimize import curve_fit

    chi_q = np.zeros(len(qidx_list))
    dchi_q = np.zeros(len(qidx_list))
    fit_quality = []

    ts = rs / (12 * Ne**0.5) if wf == "sjb" else rs / 20
    ss = 3 if wf == "sjb" else 2
    nw = 1024
    tpmult = 15.625
    L = (4 * np.pi / 3 * rs**3 * Ne) ** (1 / 3)
    n0 = Ne / L**3

    for iq in range(len(qidx_list)):
        E_list = []
        dE_list = []
        q = qidx_list[iq]
        alpha = guess_alpha2(rs, Ne, q)

        for vq in vq_list:
            dir_h5 = (
                f"{main_dir}/rs{rs:.1f}-n{Ne:d}/"
                f"qv{q[0]:d}_{q[1]:d}_{q[2]:d}-vq{vq:.4f}/"
                f"{dft_func}-e{ecut_pre}-qa{alpha:.3f}-thr1.0d-10/"
                f"{wf}-t{tpmult * Ne * rs}-ts{ts:.4f}-nw{nw}/"
                f"qmc.s00{ss}.stat.h5"
            )
            E, dE = get_energy(dir_h5)
            E_list.append(E / Ne)
            dE_list.append(dE / Ne)

        E_arr = np.array(E_list)
        dE_arr = np.array(dE_list)
        vq_arr = np.array(vq_list)

        # --- quartic fit: E0 + A*x^2 + B*x^4 ---
        # E0_fixed = E_arr[0]

        def quartic(x, A, B, e0):
            return e0 + A * x**2 + B * x**4

        def quadratic(x, A, e0):
            return e0 + A * x**2

        if vq_fit == "quadratic":
            # popt, pcov = np.polyfit(vq_arr**2, E_arr-E0_fixed, 1, w=1/dE_arr, cov=True)
            popt, pcov = curve_fit(
                quadratic, vq_arr, E_arr, sigma=dE_arr, absolute_sigma=True
            )
            A = popt[0]
            dA = np.sqrt(pcov[0, 0])
            E_pred = quadratic(vq_arr, *popt)

        elif vq_fit == "quartic":
            # popt, pcov = np.polyfit(vq_arr**2, E_arr-E0_fixed, 2, w=1/dE_arr, cov=True)
            popt, pcov = curve_fit(
                quartic, vq_arr, E_arr, sigma=dE_arr, absolute_sigma=True
            )
            A = popt[0]
            try:
                dA = np.sqrt(pcov[0, 0])
            except ImportError:
                dA = 1e8
            E_pred = quartic(vq_arr, *popt)
        # --- fit quality diagnostics ---
        print("Reporting")
        residuals = E_arr - E_pred
        n_params = 3 if vq_fit == "quartic" else 2  # (e0, A, B) or (e0, A)
        ndof = max(len(vq_arr) - n_params, 1)
        chi2_red = np.sum((residuals / dE_arr) ** 2) / ndof

        # Signal-to-noise: how far does the parabola rise above noise?
        mean_dE = np.mean(dE_arr[1:]) if len(dE_arr) > 1 else dE_arr[0]
        delta_E = np.max(np.abs(E_arr[1:] - E_arr[0])) if len(E_arr) > 1 else 0
        snr = delta_E / mean_dE if mean_dE > 0 else 0.0

        # Significance of curvature
        A_sig = abs(A) / dA if dA > 0 else np.inf

        # Reliability flag
        reliable = (snr > 5.0) and (A_sig > 2.0) and (chi2_red < 10.0)

        info = {
            "reduced_chi2": chi2_red,
            "signal_to_noise": snr,
            "A_significance": A_sig,
            "reliable": reliable,
            "E_list": E_arr,
            "dE_list": dE_arr,
            "popt": popt,
            "q": q,
        }

        chi_q[iq] = A * n0
        dchi_q[iq] = dA * n0
        fit_quality.append(info)

        if verbose:
            tag = "OK" if info["reliable"] else "UNRELIABLE"
            print(
                f"  q={q}  chi/n0={A:.4e}  SNR={info['signal_to_noise']:.1f}  "
                f"|A|/dA={info['A_significance']:.1f}  "
                f"chi2r={info['reduced_chi2']:.2f}  [{tag}]"
            )

    return chi_q, dchi_q, fit_quality


def bootstrap_chi_error_blocks(
    raw_blocks, vq_arr, n0, fs_correct_fn, fit_type="quadratic", n_boot=500, seed=None
):
    """
    !!!!!!!!!!!!!!!!!!!!!!!!!
    !!!!!!!!ABANDONED!!!!!!!!
    !!!!!!!!!!!!!!!!!!!!!!!!!

    Non-parametric block bootstrap for chi(q) error bars.

    Instead of assuming E ~ N(mean, sigma), resamples the actual QMC block
    energies with replacement, computes block means, fits, and applies
    FS correction. This captures the true error distribution including
    any non-Gaussianity and autocorrelation structure.

    Parameters
    ----------
    raw_blocks : ndarray, shape (n_q, n_v, n_blocks)
        Per-block energies from load_raw_blocks.
    vq_arr : array (n_v,)
        The vq values used in the fit.
    n0 : float
        Number density.
    fs_correct_fn : callable
        chi_arr -> chi_corr_arr  (finite-size correction).
    fit_type : str
        'quadratic' or 'quartic'.
    n_boot : int
        Number of bootstrap iterations.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    boot_err : array (n_q,)
    boot_samples : array (n_boot, n_q)
    """
    rng = np.random.default_rng(seed)
    vq_arr = np.asarray(vq_arr, dtype=float)
    n_q, n_v, n_blocks = raw_blocks.shape

    def quartic(x, A, B, e0):
        return e0 + A * x**2 + B * x**4

    def quadratic(x, A, e0):
        return e0 + A * x**2

    chi_boot = np.empty((n_boot, n_q))

    for b in range(n_boot):
        # Resample block indices independently for each (q, vq)
        idx = rng.integers(0, n_blocks, size=(n_q, n_v, n_blocks))

        # Compute resampled means: for each (q, v), mean of selected blocks
        q_idx = np.arange(n_q)[:, None, None]
        v_idx = np.arange(n_v)[None, :, None]
        resampled = raw_blocks[q_idx, v_idx, idx]  # (n_q, n_v, n_blocks)
        E_boot = np.mean(resampled, axis=2)  # (n_q, n_v)

        for iq in range(n_q):
            E_b = E_boot[iq, :]  # (n_v,) perturbed energies for this q
            fit_func = fit_funcs(fit_type)
            sigma_b = np.std(resampled[iq, :, :], axis=1) / np.sqrt(
                n_blocks
            )  # bootstrap std error
            popt, _ = fit_E_of_vq(E_b, sigma_b, vq_arr, fit_func)
            try:
                chi_boot[b, iq] = popt[0] * n0
            except RuntimeError:
                chi_boot[b, iq] = np.nan

    # Apply FS correction
    boot_samples = np.empty_like(chi_boot)
    for b in range(n_boot):
        try:
            boot_samples[b] = fs_correct_fn(chi_boot[b])
        except Exception:
            boot_samples[b] = np.nan

    boot_err = np.nanstd(boot_samples, axis=0)
    return boot_err, boot_samples
