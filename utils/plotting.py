"""
Visualization utilities for the static response project.

plot_chi, plot_E_of_vq, plot_variance.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_chi(
    qidxl,
    chi,
    dchi,
    rs,
    Ne,
    chi_ref="Moroni",
    ax=None,
    c="k",
    fit_quality=None,
    unreliable_color="red",
    unreliable_marker="o",
    **kwargs,
):
    """
    Plot -chi(q)/n0 vs q/kF with error bars and reference curves.

    Parameters
    ----------
    qidxl : list of [qx, qy, qz]
    chi, dchi : arrays of chi values and errors
    rs, Ne : system parameters
    chi_ref : 'Moroni', 'Corradini', 'both', or None
    fit_quality : list of dicts (from get_chi_q) — flags unreliable points
    **kwargs : forwarded to the data marker plot
    """
    from .physics import (
        get_chi_corradini,
        get_chi_Moroni,
        get_chi_RPA,
        get_gas_params,
        get_qs,
    )

    if ax is None:
        fig, ax = plt.subplots()

    kF, n0, NF, L = get_gas_params(rs, Ne)
    ql = get_qs(qidxl, Ne, rs)

    # Reference curves
    q_fine = np.linspace(0.01, 7.5, 5000)

    if chi_ref in ("Moroni", "both"):
        chi_M = get_chi_Moroni(rs, Ne, q_fine)
        ax.plot(q_fine / kF, -chi_M / n0, "k-", label=r"Moroni $\text{et. al}$")
    if chi_ref in ("Corradini", "both"):
        chi_C = get_chi_corradini(rs, Ne, q_fine)
        ax.plot(q_fine / kF, -chi_C / n0, "k-.", label=r"Corradini $\text{et. al}$")
    if chi_ref in ("Corradini", "both"):
        chi_R = get_chi_RPA(rs, Ne, q_fine)
        ax.plot(q_fine / kF, -chi_R / n0, "k--", label=r"RPA")

    # Reliable vs unreliable
    if fit_quality is not None:
        reliable = np.array([fq["reliable"] for fq in fit_quality])
    else:
        reliable = np.ones(len(chi), dtype=bool)

    marker = "o"
    defaults = {
        "marker": marker,
        "markeredgecolor": c,
        "markerfacecolor": "white",
        "markersize": 5,
        "linestyle": "",
        "label": rf"$N_e = {Ne}$",
        "fillstyle": "none",
    }
    defaults.update(kwargs)

    ax.errorbar(
        ql / kF, -chi / n0, yerr=dchi / n0, linestyle="", alpha=1, color="black"
    )

    if np.any(reliable):
        ax.plot(ql[reliable] / kF, -chi[reliable] / n0, **defaults)

    if np.any(~reliable):
        bad_defaults = dict(defaults)
        bad_defaults.update(
            {
                "marker": unreliable_marker,
                "markeredgecolor": unreliable_color,
                "markerfacecolor": unreliable_color,
                "markersize": 7,
                "label": None,
            }
        )
        ax.plot(ql[~reliable] / kF, -chi[~reliable] / n0, **bad_defaults)
        # printing the unreliable q points and their indices
        print("\nUnreliable q points:")
        print("-" * 41)
        for i, q in enumerate(np.array(qidxl)[~reliable]):
            iq_unreliable = np.array(range(len(qidxl)))[~reliable][i]
            print(
                f"q = {q}.\nFitted with {fit_quality[iq_unreliable]['vq_fit']} function by following vq points: {fit_quality[iq_unreliable]['vqlist']}.\nReason: {fit_quality[iq_unreliable]['reason']}"
            )
            print("-" * 41)

    ax.set_ylabel(r"$-\chi(q)/n_0$")
    ax.set_xlabel(r"$q/q_F$")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, NF / n0)


def plot_E_of_vq(
    q,
    vq_list,
    rs,
    Ne,
    main_dir,
    alpha=None,
    ax=None,
    fit_line="r",
    pwscf=False,
    ecut_pre=125,
    **kwargs,
):
    """
    Plot E(vq) for a single q-point with quartic fit.

    Parameters
    ----------
    q : [qx, qy, qz]
    vq_list : array of vq values
    rs, Ne, main_dir, dft_func : run identifiers
    pwscf : bool — if True, plot DFT energies instead of DMC
    ecut_pre : int — plane-wave cutoff (for DFT path)
    **kwargs : forwarded to data markers
    """
    from scipy.optimize import curve_fit

    from .io_utils import _build_h5_path, get_energy, get_energy_pwscf
    from .physics import (
        anal_chi02,
        get_chi_Moroni,
        get_gas_params,
        get_qs,
        guess_alpha2,
    )

    if ax is None:
        fig, ax = plt.subplots()

    if alpha is None:
        alpha = guess_alpha2(rs, Ne, q)

    n0 = get_gas_params(rs, Ne)[1]

    # Collect energies
    E_list, dE_list = [], []
    for vq in vq_list:
        if pwscf:
            path = _build_h5_path(main_dir, rs, Ne, q, vq, pwscf=True)
            E = get_energy_pwscf(path)
            dE = 1e-17
        else:
            path = _build_h5_path(main_dir, rs, Ne, q, vq, pwscf=False)

            E, dE = get_energy(path)
        E_list.append(E / Ne)
        dE_list.append(dE / Ne)

    E_arr = np.array(E_list)
    dE_arr = np.array(dE_list)
    vq_arr = np.array(vq_list)

    def quartic(x, A, B):
        return E_arr[0] + A * x**2 + B * x**4

    if pwscf:
        popt, _ = curve_fit(
            quartic, vq_arr * alpha, E_arr, sigma=dE_arr, absolute_sigma=True
        )
    else:
        popt, _ = curve_fit(quartic, vq_arr, E_arr, sigma=dE_arr, absolute_sigma=True)

    # Plot
    if pwscf:
        finex = np.linspace(vq_arr[0] * alpha, vq_arr[-1] * alpha)
        ax.plot(finex, E_arr[0] + anal_chi02(rs, Ne, [q]) / n0 * finex**2, "k")
        ax.plot(finex, E_arr[0] + popt[0] * finex**2 + popt[1] * finex**4, fit_line)
        ax.errorbar(
            vq_arr * alpha, E_arr, yerr=dE_arr, linestyle="", alpha=0.7, color="black"
        )
        marker_defaults = {
            "markerfacecolor": "white",
            "markeredgecolor": "k",
            "marker": "o",
            "fillstyle": "none",
            "linestyle": "",
        }
        marker_defaults.update(kwargs)
        ax.plot(vq_arr * alpha, E_arr, **marker_defaults)
        ax.set_xlim(0, finex[-1] * 1.2)
        print(f"E_GS = {E_arr[0]:.3f}")
        print(f"anal chi = {-anal_chi02(rs, Ne, [q]) / n0}")
        print(f"dft chi = {-popt[0]}")
        ax.set_ylabel(r"$E_\mathrm{DFT}$")
        ax.set_xlabel(r"$\alpha$")
    else:
        finex = np.linspace(vq_arr[0], vq_arr[-1])
        ax.plot(finex, E_arr[0] + popt[0] * finex**2 + popt[1] * finex**4, fit_line)
        ax.plot(
            finex,
            E_arr[0] + get_chi_Moroni(rs, Ne, get_qs([q], Ne, rs)) / n0 * finex**2,
            "k",
        )
        print(f"-chi/n0 = {-popt[0]:.3f}")
        ax.errorbar(vq_arr, E_arr, yerr=dE_arr, linestyle="", alpha=0.7, color="black")
        marker_defaults = {
            "markerfacecolor": "white",
            "markeredgecolor": "k",
            "marker": "o",
            "fillstyle": "none",
            "linestyle": "",
        }
        marker_defaults.update(kwargs)
        ax.plot(vq_arr, E_arr, **marker_defaults)
        ax.set_ylabel(r"$E_\mathrm{DMC}$")
        ax.set_xlabel(r"$v_q$")

    kF = get_gas_params(rs, Ne)[0]
    ax.set_title(
        rf"$N_e = {Ne}, r_s = {rs}, q/kF = {get_qs([q], Ne, rs) / kF}, "
        rf"\alpha= {alpha:.3f}$"
    )


def plot_variance(
    main_dir,
    rs,
    Ne,
    qidx_list=None,
    vq_list=None,
    nequil=50,
    vs="vq",
    fixed_val=None,
    ax=None,
    **plot_kwargs,
):
    """
    Plot QMC variance vs either q (for fixed vq) or vs vq (for fixed q).

    Parameters
    ----------
    vs : 'vq' or 'q'
    fixed_val : the fixed q-index (if vs='vq') or fixed vq (if vs='q')
    **plot_kwargs : forwarded to errorbar

    Returns
    -------
    ax, x_vals, var_vals, dvar_vals
    """
    from .io_utils import get_variance_for_run
    from .physics import get_gas_params, get_qs, guess_alpha2

    if ax is None:
        fig, ax = plt.subplots(dpi=150)

    var_vals, dvar_vals, x_vals = [], [], []

    if vs == "vq":
        if fixed_val is None:
            raise ValueError("Need fixed q-index for vs='vq'")
        q = fixed_val
        for vq in vq_list:
            var, dvar = get_variance_for_run(main_dir, rs, Ne, q, vq, nequil=nequil)
            if var is not None:
                x_vals.append(vq)
                var_vals.append(var)
                dvar_vals.append(dvar)
        xlabel = r"$v_q$"
        title = (
            rf"$N_e={Ne}$,$r_s={rs}$, "
            rf"$q=({q[0]},{q[1]},{q[2]})$, "
            rf"$\alpha={guess_alpha2(rs, Ne, q):.3f}$"
        )

    elif vs == "q":
        kF, n0, NF, L = get_gas_params(rs, Ne)
        if fixed_val is None:
            raise ValueError("Need fixed vq for vs='q'")
        vq = fixed_val
        q_mags = get_qs(qidx_list, Ne, rs)
        for i, q in enumerate(qidx_list):
            var, dvar = get_variance_for_run(main_dir, rs, Ne, q, vq, nequil=nequil)
            if var is not None:
                x_vals.append(q_mags[i] / kF)
                var_vals.append(var)
                dvar_vals.append(dvar)
        xlabel = r"$|q|/k_F$"
        title = rf"$N_e={Ne}$,$r_s={rs}$, $v_q={vq:.4f}$"
    else:
        raise ValueError("vs must be 'vq' or 'q'")

    x_vals = np.array(x_vals)
    var_vals = np.array(var_vals)
    dvar_vals = np.array(dvar_vals)

    order = np.argsort(x_vals)
    x_vals, var_vals, dvar_vals = x_vals[order], var_vals[order], dvar_vals[order]

    ax.errorbar(x_vals, var_vals, yerr=dvar_vals, fmt="o-", capsize=3, **plot_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Variance")
    ax.set_title(title)

    return ax, x_vals, var_vals, dvar_vals
