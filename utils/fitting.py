"""
Fitting and chi(q) extraction pipeline.

E(vq) fitting, chi(q) extraction, bootstrap error propagation,
finite-size corrected pipeline (get_chi), and vq-range analysis.
"""

import numpy as np
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Fit functions
# ---------------------------------------------------------------------------


def _fit_func(vq_fit):
    """
    Return the named fit function for E(vq).

    Options: 'quadratic' -> E0 + A*x^2
             'quartic'   -> E0 + A*x^2 + B*x^4
    """

    def quartic(x, A, B, e0):
        return e0 + A * x**2 + B * x**4

    def quadratic(x, A, e0):
        return e0 + A * x**2

    funcs = {"quartic": quartic, "quadratic": quadratic}
    try:
        return funcs[vq_fit]
    except KeyError:
        raise ValueError(
            f"Invalid fit type '{vq_fit}'. Choose 'quartic' or 'quadratic'."
        )


def fit_E_of_vq(E_arr, dE_arr, vq_arr, func):
    """
    Fit E(vq) with the given function using curve_fit.

    Parameters
    ----------
    E_arr : array  — energies
    dE_arr : array — errors (absolute sigma)
    vq_arr : array — vq values
    func : callable — fit function (from _fit_func)

    Returns
    -------
    popt, pcov : as from scipy.optimize.curve_fit
    """
    return curve_fit(func, vq_arr, E_arr, sigma=dE_arr, absolute_sigma=True)


# ---------------------------------------------------------------------------
# Fit quality diagnostics
# ---------------------------------------------------------------------------


def fit_quality_report(
    poptl, pcovl, E_all, dE_all, vq_arr, qidx_list, vq_fit, verbose=False
):
    """
    Generate per-q fit quality diagnostics.

    Returns a list of dicts with keys:
      reduced_chi2, signal_to_noise, A_significance, reliable,
      E_list, dE_list, popt, pcov, q
    """
    fit_quality = []
    func = _fit_func(vq_fit)
    n_params = 3 if vq_fit == "quartic" else 2

    for iq, q in enumerate(qidx_list):
        E_arr = E_all[iq, :]
        dE_arr = dE_all[iq, :]
        popt = poptl[iq]
        pcov = pcovl[iq]

        E_pred = func(vq_arr, *popt)
        residuals = E_arr - E_pred
        ndof = max(len(vq_arr) - n_params, 1)
        chi2_red = np.sum((residuals / dE_arr) ** 2) / ndof

        mean_dE = np.mean(dE_arr[1:]) if len(dE_arr) > 1 else dE_arr[0]
        delta_E = np.max(np.abs(E_arr[1:] - E_arr[0])) if len(E_arr) > 1 else 0
        snr = delta_E / mean_dE if mean_dE > 0 else 0.0

        A = popt[0]
        dA = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else 1e8
        A_sig = abs(A) / dA if dA > 0 else np.inf

        reliable = (snr > 5.0) and (A_sig > 2.0) and (chi2_red < 10.0)

        if not reliable:
            if snr <= 5.0:
                reason = f"Low signal-to-noise ratio. SNR = {snr:.2f}"
            elif A_sig <= 2.0:
                reason = f"Low significance. |A|/dA = {A_sig:.2f}"
            elif chi2_red >= 10.0:
                reason = f"High reduced chi^2. chi^2 = {chi2_red:.2f}"
        else:
            reason = "None"

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
            "vqlist": vq_arr,
            "vq_fit": vq_fit,
            "reason": reason,
        }
        fit_quality.append(info)

        if verbose:
            tag = "OK" if reliable else "UNRELIABLE"
            if not reliable:
                print(
                    f"  q={q}  chi/n0={A:.4e}  SNR={snr:.1f}  "
                    f"|A|/dA={A_sig:.1f}  chi2r={chi2_red:.2f}  [{tag}]"
                )

    return fit_quality


# ---------------------------------------------------------------------------
# get_chi0_q  (DFT-level chi0 from pwscf energies)
# ---------------------------------------------------------------------------


def get_chi0_q(
    main_dir,
    Ne,
    rs,
    vq_list,
    qidx_list,
    ecut_pre=125,
    dft_func="ni",
    vq_fit="quadratic",
):
    """
    Extract non-interacting chi_0(q) by fitting DFT (pwscf) energies
    vs vq for each q-point.

    Returns
    -------
    chi_q : array   — DFT chi_0 for each q
    dchi_q : array  — (zeros — DFT has no stochastic error)
    """
    from .io_utils import _build_h5_path, get_energy_pwscf
    from .physics import get_gas_params, guess_alpha2

    chi_q = np.zeros(len(qidx_list))
    dchi_q = np.zeros(len(qidx_list))

    for iq, q in enumerate(qidx_list):
        E_list = []
        alpha = guess_alpha2(rs, Ne, q)
        for vq in vq_list:
            path = _build_h5_path(main_dir, rs, Ne, q, vq, pwscf=True)
            E = get_energy_pwscf(path)
            E_list.append(E / Ne)

        func = _fit_func(vq_fit)
        popt, _ = curve_fit(func, np.array(vq_list) * alpha, E_list)
        n0 = get_gas_params(rs, Ne)[1]
        chi_q[iq] = popt[0] * n0

    return chi_q, dchi_q


# ---------------------------------------------------------------------------
# Core: get_chi_q
# ---------------------------------------------------------------------------


def get_chi_q(main_dir, Ne, rs, vq_list, qidx_list, verbose=False, vq_fit="quadratic"):
    """
    Extract chi(q) from E(vq) fits using cached energies.

    Returns
    -------
    chi_q : array
    dchi_q : array
    fit_quality : list of dict
    """
    from .io_utils import load_or_compute_E
    from .physics import get_gas_params

    n0 = get_gas_params(rs, Ne)[1]
    E_all, dE_all = load_or_compute_E(main_dir, rs, Ne, qidx_list, vq_list)

    chi_q = np.zeros(len(qidx_list))
    dchi_q = np.zeros(len(qidx_list))
    vq_arr = np.asarray(vq_list, dtype=float)

    poptl, pcovl = [], []
    func = _fit_func(vq_fit)
    if verbose:
        print(
            f"Fitting E(vq) for {len(qidx_list)} q-points with '{vq_fit}' function..."
        )
    for iq in range(len(qidx_list)):
        popt, pcov = fit_E_of_vq(E_all[iq], dE_all[iq], vq_arr, func)
        A = popt[0]
        dA = np.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else 1e8
        poptl.append(popt)
        pcovl.append(pcov)
        chi_q[iq] = A * n0
        dchi_q[iq] = dA * n0
    if verbose:
        print("Fit complete. Generating fit quality report...")
    fit_quality = fit_quality_report(
        poptl, pcovl, E_all, dE_all, vq_arr, qidx_list, vq_fit, verbose
    )
    if verbose:
        print("-" * 82)
        print("Successfully completed")
        print("-" * 82)
    return chi_q, dchi_q, fit_quality


# ---------------------------------------------------------------------------
# Full pipeline: get_chi  (fit → FS correct → bootstrap)
# ---------------------------------------------------------------------------


def get_correction(main_dir, qidxl, rs, Ne, vq_list, qidx_list):
    """
    Compute the finite-size correction:  chi0_inf^{-1} - chi0_grid^{-1}.

    Parameters
    ----------
    main_dir : str
        Runs directory (for DFT chi0 extraction).
    qidxl : list
        q-indices to correct.
    rs, Ne : float, int
        System parameters.
    vq_list, qidx_list : list
        Full vq / q lists for DFT chi0 fit.
    """
    from .physics import anal_chi02, chi0q, get_qs

    ql = get_qs(qidxl, Ne, rs)
    chi0_infty = chi0q(ql, Ne, rs)
    chi0_a = anal_chi02(rs, Ne, qidxl)
    #chi00_q = get_chi0_q(main_dir, Ne, rs, vq_list, qidxl, dft_func="ni", ecut_pre=125)[0]
    return chi0_infty ** (-1) - chi0_a ** (-1)


def get_chi(
    main_dir,
    vql,
    qidxl,
    rs,
    Ne,
    dft_func="ni",
    vq_fit="quadratic",
    verbose=False,
    n_boot=800,
    seed=42,
    bootstrap_method="parametric",
    vq_list_dft=None,
    qidx_list_dft=None,
):
    """
    Full pipeline: fit E(vq) → extract chi → FS-correct → bootstrap errors.

    Parameters
    ----------
    main_dir : str
        Runs directory.
    vql : list
        vq values for the chi fit.
    qidxl : list
        q-indices for the chi fit.
    rs, Ne : system parameters.
    dft_func : str
        DFT functional label for FS correction.
    vq_fit : str
        'quadratic' or 'quartic'.
    n_boot : int
        Number of bootstrap iterations.
    seed : int
        RNG seed.
    bootstrap_method : str
        'parametric' or 'block'.
    vq_list_dft, qidx_list_dft : list, optional
        vq / q lists for DFT chi0 (defaults to vql / qidxl).

    Returns
    -------
    chi_q_corr : array   — FS-corrected chi(q)
    boot_err : array     — bootstrap standard deviation
    fit_quality : list of dict
    """
    from .physics import FS_correct, get_gas_params

    if vq_list_dft is None:
        vq_list_dft = vql
    if qidx_list_dft is None:
        qidx_list_dft = qidxl

    chi_q, dchi_q, fit_quality = get_chi_q(
        main_dir,
        Ne,
        rs,
        vql,
        qidxl,
        verbose=verbose,
        vq_fit=vq_fit,
    )

    correction = get_correction(main_dir, qidxl, rs, Ne, vq_list_dft, qidx_list_dft)
    chi_q_corr = FS_correct(chi_q, correction, rs, Ne, dft_func=dft_func)

    # Bootstrap error
    kF, n0, NF, L = get_gas_params(rs, Ne)

    def fs_fn(chi_arr):
        return FS_correct(chi_arr, correction, rs, Ne, dft_func=dft_func)

    boot_err, _ = bootstrap_chi_error(
        fit_quality,
        np.array(vql),
        n0,
        fs_fn,
        fit_type=vq_fit,
        n_boot=n_boot,
        seed=seed,
    )
    return chi_q_corr, boot_err, fit_quality


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


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

    Perturbs E_i → E_i + dE_i * N(0,1), refits, applies FS correction.

    Returns
    -------
    boot_err : array (n_q,)
    boot_samples : array (n_boot, n_q)
    """
    rng = np.random.default_rng(seed)
    vq_arr = np.asarray(vq_arr, dtype=float)
    n_v = len(vq_arr)
    n_q = len(fit_quality_list)

    E_all = np.array([fq["E_list"] for fq in fit_quality_list])
    dE_all = np.array([fq["dE_list"] for fq in fit_quality_list])

    noise = rng.standard_normal((n_boot, n_q, n_v))
    E_pert = E_all[np.newaxis, :, :] + dE_all[np.newaxis, :, :] * noise

    func = _fit_func(fit_type)
    chi_boot = np.empty((n_boot, n_q))

    for b in range(n_boot):
        for iq in range(n_q):
            popt, _ = fit_E_of_vq(E_pert[b, iq], dE_all[iq], vq_arr, func)
            chi_boot[b, iq] = popt[0] * n0
            """
            try:
                chi_boot[b, iq] = popt[0] * n0
            except (RuntimeError, IndexError):
                chi_boot[b, iq] = np.nan
            """

    boot_samples = np.empty_like(chi_boot)
    for b in range(n_boot):
        boot_samples[b] = fs_correct_fn(chi_boot[b])
        """
        try:
            boot_samples[b] = fs_correct_fn(chi_boot[b])
        except Exception:
            boot_samples[b] = np.nan
        """

    boot_err = np.nanstd(boot_samples, axis=0)
    return boot_err, boot_samples


# ---------------------------------------------------------------------------
# vq-range analysis
# ---------------------------------------------------------------------------


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
    Pre-flight check: predict signal-to-noise for a proposed vq range
    using a reference chi (Moroni or Corradini).

    Returns a list of per-q dicts with keys: q, q_over_kF, chi_ref_over_n0,
    delta_E, dE, snr, acceptable, vq_max_suggested.
    """
    from .physics import (
        G_Moroni,
        chi0q,
        corradini_pz,
        get_gas_params,
        get_qs,
    )

    kF, n0, NF, L = get_gas_params(rs, Ne)
    ql = get_qs(qidx_list, Ne, rs)

    if chi_ref == "Moroni":
        Vc = 4 * np.pi / ql**2
        chi0 = chi0q(ql, Ne, rs)
        G = G_Moroni(rs, ql)
        chi_ref_vals = chi0 / (1 - chi0 * (Vc - Vc * G))
    elif chi_ref == "Corradini":
        Vc = 4 * np.pi / ql**2
        chi0 = chi0q(ql, Ne, rs)
        fxc = corradini_pz(rs, ql)
        chi_ref_vals = chi0 / (1 - chi0 * (Vc + fxc))
    else:
        raise ValueError(f"Unknown chi_ref: {chi_ref}")

    if dE_typical is None:
        dE_typical = 5e-5 * rs

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
            f"\n  Summary: {len(analysis) - n_bad}/{len(analysis)} q-points "
            f"acceptable with vq_max={vq_max:.6f}, SNR threshold={snr_threshold}"
        )
        if n_bad > 0:
            max_sug = max(
                a["vq_max_suggested"] for a in analysis if not a["acceptable"]
            )
            print(f"  Suggested vq_max: {max_sug:.6f}")

    return analysis
