import numpy as np

PARAMS_MORONI = {"a1": 2.15, "a2": 0.435, "b1": 1.57, "b2": 0.409}

PARAMS_CORRADINI = PARAMS_MORONI

PARAMS_KAPLAN = {
    "a0j": -0.00451760,
    "a1j": 0.0155766,
    "a2j": 0.422624,
    "a3j": 3.516054,
    "a4j": 1.015830,
}

PARAMS_N = (
    2.5881728,
    3.32121643,
    0.09195465,
)  # fitted by fixing B(rs) to Moroni's form (which is calculated up to rs=10)
PARAMS_N2 = (
    1.53561128,
    2.8426441,
    0.01958829,
)  # fitted after optimizing B(rs) to our form (which is calculated up to rs=80).
# Optimization of B was done by setting n,B free together. After obtaining the new B(rs), it was fixed and n(rs) was optimized and fitted to the form given in get_n().

PARAMS_N3 = (
    44.30346968,
    1.44807839,
    0.14616063,
    0.71333594,
)  # fitted both n And B free together.

PARAMS_B_NEW = {
    "a1": 2.421104280498847,
    "a2": 0,
    "b1": 1.5169121108381027,
    "b2": 0.009417565545155299,
}

PARAMS_KAPLAN_NEW = {
    "a0j": -0.0148281396829882,
    "a1j": 0.01801880545317576,
    "a2j": 0.16798453028516483,
    "a3j": 3.516054,
    "a4j": 1.015830,
}


def get_n(rs):
    """n(rs) = n_inf + dn * exp(-k * rs)  (exponential approach to large-rs limit)."""
    n_inf, dn, k = PARAMS_N2
    a, b, c, d = PARAMS_N3
    return n_inf + dn * np.exp(-k * np.asarray(rs, dtype=float))
    # return a * np.power(rs, -b) + c * rs ** (d)


# ---------------------------------------------------------------------------
# Fit forms from literature
# ---------------------------------------------------------------------------
def form_Moroni(q, rs, a1, a2, b1, b2, n=None):
    q = q + 1e-18
    rho_avg = 1.0 / (rs**3 * 4 * np.pi / 3)
    k_F = (3 * np.pi**2 * rho_avg) ** (1 / 3)
    Q = q / k_F

    diff_mu = diffvc(rho_avg)
    A = 1 / 4 - (k_F**2) / (4 * np.pi) * diff_mu

    diff_rse = diffv_cep(rs)
    C = np.pi / (2 * k_F) * (-diff_rse)

    if n is None:
        n = 4 if rs >= 10 else 8

    x = rs**0.5
    B = (1 + a1 * x + a2 * x**3) / (3 + b1 * x + b2 * x**3)

    G = (((A - C) ** (-n) + (Q**2 / B) ** n) ** (-1 / n) + C) * Q**2
    return G


def form_Corradini(q, rs, a1, a2, b1, b2):
    q = q + 1e-18
    rho_avg = 1.0 / (rs**3 * 4 * np.pi / 3)
    k_F = (3 * np.pi**2 * rho_avg) ** (1 / 3)
    Q = q / k_F

    diff_mu = diffvc(rho_avg)
    A = 1 / 4 - (k_F**2) / (4 * np.pi) * diff_mu

    diff_rse = diffv_cep(rs)
    C = np.pi / (2 * k_F) * (-diff_rse)
    x = rs**0.5
    B = (1 + a1 * x + a2 * x**3) / (3 + b1 * x + b2 * x**3)

    g = B / (A - C)
    alpha = 1.5 / (rs**0.25) * A / (B * g)
    beta = 1.2 / (B * g)

    G = C * Q**2 + (B * Q**2) / (g + Q**2) + alpha * Q**4 * np.exp(-beta * Q**2)
    return G




def form_Kaplan(rs, q, a0j, a1j, a2j, a3j, a4j):
    """
    Kaplan & Kukkonen (2022) static density local field factor G+(rs, q).
    Implements Eqs. (2)-(3) of Kaplan & Kukkonen, PRB 105, 035123 (2022):

    Parameters
    ----------
    rs : float
        Wigner-Seitz radius (atomic units). Valid range: ~0.1 to 100.
    q : float or array-like
        Wavevector (atomic units).
    a0j, a1j, a2j, a3j, a4j : float
        Fit parameters for G+.

    Returns
    -------
    G : float or ndarray
        Static density local field factor G+(rs, q).

    Asymptotic limits satisfied
    ---------------------------
    Small q:  G+ → A+(rs)·x²          (compressibility sum rule)
    Large q:  G+ → C(rs)·x² + B+(rs)  (Moroni et al.)
    """

    kF = _kF_from_rs(rs)
    x = np.asarray(q, dtype=float) / kF

    A_j = _A_plus(rs)
    B_j = _B_plus(rs)
    C_rs = _C(rs, kF)
    alpha_j = a0j + a1j * np.exp(-a2j * rs)

    y = x**4 / 16.0
    H = H_smooth_step(y, a3j, a4j)

    small_q = x**2 * (A_j + alpha_j * x**4) * H
    large_q = (C_rs * x**2 + B_j) * (1.0 - H)

    return small_q + large_q


# ---------------------------------------------------------------------------
# Reference local field factors
# ---------------------------------------------------------------------------


def G_Moroni(rs, q, n=None):
    if n is None:
        n = 4 if rs >= 10 else 8
    if n == "opt":
        n = get_n(rs)
    n = get_n(rs)
    return form_Moroni(q, rs, *PARAMS_MORONI.values(), n=n)


def G_mhg(rs, q):
    n = get_n(rs)
    return form_Moroni(q, rs, *PARAMS_B_NEW.values(), n=n)


def G_Corradini(rs, q):
    return form_Corradini(q, rs, *PARAMS_CORRADINI.values())


def G_Kaplan(rs, q):
    return form_Kaplan(rs, q, *PARAMS_KAPLAN.values())


# ---------------------------------------------------------------------------
# Ceperley-Alder correlation derivatives (used by G_Moroni & corradini)
# ---------------------------------------------------------------------------


def diffv_cep(r_s):
    """d(r_s * epsilon_c) / d(r_s)  for the Ceperley-Alder parametrization."""
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334
    denom = (beta1 * np.sqrt(r_s) + beta2 * r_s + 1) ** 2
    return beta1 * gamma * np.sqrt(r_s) / (2 * denom) + gamma / denom


def diffvc(rho):
    """d(mu_c) / d(n_0)  (correlation chemical-potential derivative)."""
    third = 1.0 / 3.0
    a = 0.0311
    c = 0.0020
    d = -0.0116
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334

    r_s = (3.0 / (4.0 * np.pi * rho)) ** third

    stor1 = (1.0 + beta1 * np.sqrt(r_s) + beta2 * r_s) ** (-3.0)
    stor2 = (
        -0.41666667 * beta1 * (r_s ** (-0.5))
        - 0.5833333 * (beta1**2)
        - 0.66666667 * beta2
    )
    stor3 = -1.75 * beta1 * beta2 * np.sqrt(r_s) - 1.3333333 * r_s * (beta2**2)
    reshigh = gamma * stor1 * (stor2 + stor3)
    reslow = a / r_s + 0.66666667 * (c * np.log(r_s) + d) + 0.33333333 * c

    reshigh = reshigh * (-4.0 * np.pi / 9.0) * (r_s**4)
    reslow = reslow * (-4.0 * np.pi / 9.0) * (r_s**4)

    filterlow = r_s < 1
    filterhigh = r_s >= 1
    return reslow * filterlow + reshigh * filterhigh


def _kF_from_rs(rs):
    """Fermi wavevector from Wigner-Seitz radius: kF = (9π/4)^(1/3) / rs."""
    return (9.0 * np.pi / 4.0) ** (1.0 / 3.0) / rs


# ─────────────────────────────────────────────────────────────────────────────
# Exchange energy per electron and its rs-derivatives (analytic)
# ─────────────────────────────────────────────────────────────────────────────
# e_x(rs) = -Cx / rs,  Cx = (3/4π)(9π/4)^(1/3)

_CX = (3.0 / (4.0 * np.pi)) * (9.0 * np.pi / 4.0) ** (1.0 / 3.0)


def _ex(rs):
    """LDA exchange energy per electron."""
    return -_CX / rs


def _dex_drs(rs):
    """de_x/drs."""
    return _CX / rs**2


def _d2ex_drs2(rs):
    """d²e_x/drs²."""
    return -2.0 * _CX / rs**3


# ─────────────────────────────────────────────────────────────────────────────
# PW92 correlation energy per electron and its rs-derivatives
# ─────────────────────────────────────────────────────────────────────────────


def _ec_pw92(rs):
    """
    Perdew-Wang 92 correlation energy per electron (Hartree).

    J. P. Perdew & Y. Wang, Phys. Rev. B 45, 13244 (1992).
    """
    A = 0.031091
    a1 = 0.21370
    b1, b2, b3, b4 = 7.5957, 3.5876, 1.6382, 0.49294

    Q0 = -2.0 * A * (1.0 + a1 * rs)
    Q1 = 2.0 * A * (b1 * rs**0.5 + b2 * rs + b3 * rs**1.5 + b4 * rs**2)
    return Q0 * np.log(1.0 + 1.0 / Q1)


def _dec_drs(rs):
    """de_c/drs via central differences."""
    h = max(1e-4 * rs, 1e-6)
    return (_ec_pw92(rs + h) - _ec_pw92(rs - h)) / (2.0 * h)


def _d2ec_drs2(rs):
    """d²e_c/drs² via central differences."""
    h = max(1e-4 * rs, 1e-6)
    return (_ec_pw92(rs + h) - 2.0 * _ec_pw92(rs) + _ec_pw92(rs - h)) / h**2


# ─────────────────────────────────────────────────────────────────────────────
# A+(rs) — compressibility sum rule
# ─────────────────────────────────────────────────────────────────────────────


def _A_plus(rs):
    """
    A+(rs) from the compressibility sum rule.

    From the paper:
        A+(rs) = -(kF²/4π) · ∂²ε_xc^LDA/∂n²
    """
    kF = _kF_from_rs(rs)

    # Combined xc derivatives wrt rs (exchange analytic + correlation numerical)
    de_xc = _dex_drs(rs) + _dec_drs(rs)
    d2e_xc = _d2ex_drs2(rs) + _d2ec_drs2(rs)

    # d²(n·e_xc)/dn² expressed through rs-derivatives
    d2_vol_dn2 = (4.0 * np.pi * rs**4 / 27.0) * (rs * d2e_xc - 2.0 * de_xc)

    return -(kF**2 / (4.0 * np.pi)) * d2_vol_dn2


# ─────────────────────────────────────────────────────────────────────────────
# B+(rs) — large-q constant, Moroni et al. (1995)
# ─────────────────────────────────────────────────────────────────────────────


def _B_plus(rs):
    """
    Large-q constant B+(rs).

    lim_{q→∞} G+(rs,q) = C(rs)·x² + B+(rs)

    Moroni et al. parameterization as given in the paper:
        B+(rs) = [1 + 2.15·rs^(1/2) + 0.435·rs^(3/2)]
               / [3 + 1.57·rs^(1/2) + 0.409·rs^(3/2)]
    """
    s = rs**0.5
    return (1.0 + 2.15 * s + 0.435 * s**3) / (3.0 + 1.57 * s + 0.409 * s**3)


# ─────────────────────────────────────────────────────────────────────────────
# C(rs) — large-q x² coefficient
# ─────────────────────────────────────────────────────────────────────────────


def _C(rs, kF=None):
    """
    Large-q x² coefficient C(rs).

    Uses PW92 correlation only (no exchange), per Corradini et al. 1998
    and Kaplan & Kukkonen 2022.
    """
    if kF is None:
        kF = _kF_from_rs(rs)

    h = max(1e-4 * rs, 1e-6)
    d_rse_c = ((rs + h) * _ec_pw92(rs + h) - (rs - h) * _ec_pw92(rs - h)) / (2.0 * h)

    return -(np.pi / (2.0 * kF)) * d_rse_c


# ─────────────────────────────────────────────────────────────────────────────
# Smoothed step function
# ─────────────────────────────────────────────────────────────────────────────


def H_smooth_step(y, beta, gamma):
    ebg = np.exp(beta * gamma)
    eby = np.exp(-beta * y)
    return (ebg - 1.0) * eby / (1.0 + (ebg - 2.0) * eby)
