"""
Physics formulas for the 3D homogeneous electron gas.

Lindhard function, local field factors (Moroni, Corradini), LDA f_xc,
finite-size chi0 on a discrete k-grid, gas parameters, alpha heuristics,
finite-size correction, and reference chi models (RPA, Moroni, Corradini).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Gas parameters
# ---------------------------------------------------------------------------


def get_gas_params(rs, Ne):
    """
    Return fundamental parameters of the homogeneous electron gas.

    Returns
    -------
    kF : float  — Fermi wavevector
    n0 : float  — electron density
    NF : float  — density of states at the Fermi level
    L  : float  — simulation cell side length
    """
    L = (4 * np.pi / 3 * rs**3 * Ne) ** (1 / 3)
    n0 = 1.0 / (rs**3 * 4 * np.pi / 3)
    kF = (3 * np.pi**2 * n0) ** (1 / 3)
    NF = kF / (np.pi**2)
    return kF, n0, NF, L


# ---------------------------------------------------------------------------
# q-grid helpers
# ---------------------------------------------------------------------------


def get_qs(qidx, Ne, rs):
    """Convert integer q-indices to physical |q| magnitudes (a.u.)."""
    L = (4 * np.pi / 3 * rs**3 * Ne) ** (1 / 3)
    qs = 2 * np.pi / L * np.array(qidx)
    return np.linalg.norm(qs, axis=1)


def gen_qidx(mq):
    """
    Generate unique q-index representatives, one per shell (unique |q|²),
    sorted by increasing magnitude.  Excludes [0,0,0].
    """
    ql = []
    for qx in range(mq):
        for qy in range(qx + 1):
            for qz in range(qy + 1):
                q_sq = qx**2 + qy**2 + qz**2
                ql.append((q_sq, [qx, qy, qz]))

    ql.sort(key=lambda x: x[0])

    seen = set()
    unique = []
    for q_sq, qidx in ql:
        if q_sq not in seen and q_sq > 0:
            seen.add(q_sq)
            unique.append(qidx)
    return unique


def get_shell_points(shell_number):
    """
    Return all integer k-grid points (i, j, k) inside the cumulative
    shell ``shell_number``.  Shell 1 = origin only; shell 2 = origin + 6
    nearest neighbours, etc.
    """
    G = shell_number * 2

    pts = []
    radii = set()
    for i in range(-G, G + 1):
        for j in range(-G, G + 1):
            for k in range(-G, G + 1):
                m = i * i + j * j + k * k
                pts.append((m, [i, j, k]))
                radii.add(m)

    radii = sorted(radii)
    if shell_number < 1 or shell_number > len(radii):
        raise ValueError("shell_number out of range for this G.")
    m_cut = radii[shell_number - 1]

    return np.array(sorted([p for m, p in pts if m <= m_cut]))


# ---------------------------------------------------------------------------
# Perturbation-amplitude heuristics
# ---------------------------------------------------------------------------


def guess_alpha2(rs, Ne, qidx):
    """
    Estimate optimal perturbation scaling alpha via a logistic-like function
    of (q / kF)².
    """
    a = 1.2
    kf = (9 * np.pi / 4) ** (1 / 3) / rs
    alat = (Ne * 4 * np.pi / 3) ** (1 / 3) * rs
    blat = 2 * np.pi / alat
    qvec = blat * np.array(qidx)
    qmag = np.linalg.norm(qvec)
    return 2 / (1 + np.exp(-a * (qmag / kf) ** 2)) - 1


# ---------------------------------------------------------------------------
# Lindhard function
# ---------------------------------------------------------------------------


def chi0q(q, Ne, rs):
    """
    Lindhard susceptibility chi_0(q) for one spin species
    (thermodynamic limit).
    """
    n0 = (4 * np.pi / 3 * rs**3) ** (-1) / 2
    kF = (6 * np.pi**2 * n0) ** (1 / 3)

    q = np.where(q < 1e-10, 1e-10, q)
    Q = q / kF
    return (
        -kF / (2 * np.pi**2) * (1 - (Q / 4 - 1 / Q) * np.log(np.abs((Q + 2) / (Q - 2))))
    )


def anal_chi02(rs, Ne, qidx_list):
    """
    Finite-size Lindhard chi_0(q) via exact summation over the discrete
    k-grid (for a simulation cell with Ne electrons).
    """
    L = (4 * np.pi / 3 * rs**3 * Ne) ** (1 / 3)
    n0 = Ne / L**3
    kF = (3 * np.pi**2 * n0) ** (1 / 3)

    def epsilon(k):
        return np.linalg.norm(k) ** 2 / 2

    def n_k(k):
        return 1 if epsilon(k) <= epsilon(kF) else 0

    _n_shell_map = {54: 4, 162: 7, 294: 10, 406: 13}
    if Ne not in _n_shell_map:
        raise ValueError(f"Ne={Ne} not supported. Add the corresponding n_shell value.")
    k_grid = get_shell_points(_n_shell_map[Ne]) * 2 * np.pi / L
    q_grid = 2 * np.pi / L * np.array(qidx_list)

    chi0 = np.zeros(len(q_grid))
    for i in range(len(q_grid)):
        chi0_q = 0.0
        for k in k_grid:
            k_plus_q = k + q_grid[i]
            eps_k = epsilon(k)
            eps_kq = epsilon(k_plus_q)
            nk = n_k(k)
            nkq = n_k(k_plus_q)
            chi0_q += (nk - nkq + 1e-9) / (eps_k - eps_kq + 1e-9)
        chi0[i] = chi0_q * 4

    chi0 /= L**3
    return chi0


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


def fxc_lda_scalar(rs):
    """
    Full LDA f_xc (exchange + correlation, scalar-relativistic)
    for a single rs value.
    """
    a = 0.0311
    c = 0.0020
    d = -0.0116
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334
    rho = 4 * np.pi / 3 * rs**3

    if rs >= 1:
        stor1 = (1.0 + beta1 * np.sqrt(rs) + beta2 * rs) ** (-3.0)
        stor2 = (
            -0.41666667 * beta1 * (rs ** (-0.5))
            - 0.5833333 * (beta1**2)
            - 0.66666667 * beta2
        )
        stor3 = -1.75 * beta1 * beta2 * np.sqrt(rs) - 1.3333333 * rs * (beta2**2)
        dvc = gamma * stor1 * (stor2 + stor3)
    else:
        dvc = a / rs + 0.66666667 * (c * np.log(rs) + d) + 0.33333333 * c

    dvc = dvc * (-4.0 * np.pi / 9.0) * (rs**4)

    vxnr = -((3 * rho / np.pi) ** (1 / 3))
    b = 0.0140 / rs
    rel = -0.5 + 1.5 * np.log(b + np.sqrt(1.0 + b * b)) / (b * np.sqrt(1.0 + b * b))
    bb = b * b
    bb1 = 1.0 + bb
    difrel = (1.5 / (b * bb1)) - 1.5 * np.log(b + np.sqrt(bb1)) * (1.0 + 2.0 * bb) * (
        bb1 ** (-1.5)
    ) / bb
    difrel = difrel * (-0.0140) / (rs * rs)
    dvx = (0.610887057 / (rs * rs)) * rel + vxnr * difrel
    dvx = dvx * (-4.0 * np.pi / 9.0) * (rs**4)
    return dvc + dvx


# ---------------------------------------------------------------------------
# Local field factors
# ---------------------------------------------------------------------------


def get_fxc_from_chi(rs, Ne, qidx_list, chi):
    """Extract f_xc(q) from chi(q) via the relation chi = chi0 / (1 - chi0 * (Vc + fxc))."""
    ql = get_qs(qidx_list, Ne, rs)
    Vc = 4 * np.pi / ql**2
    chi0 = anal_chi02(rs, Ne, qidx_list)
    fxc = 1 / chi0 - 1 / chi - Vc
    return fxc


# chi_RPA = chi0 / (1 - chi0 *Vc)
# 1/chi_RPA = 1/chi0 - Vc
def get_G_from_chi(rs, Ne, qidx_list, chi):
    """Extract G(q) from chi(q) via the relation chi = chi0 / (1 - chi0 * (Vc + fxc))."""
    ql = get_qs(qidx_list, Ne, rs)
    Vc = 4 * np.pi / ql**2
    fxc = get_fxc_from_chi(rs, Ne, qidx_list, chi)
    G = -fxc / Vc  # 1 + (1 / chi - 1 / chi0)/Vc
    return G


def G_Moroni(rs, q):
    """Moroni et al. local field factor G(q)."""
    q = q + 1e-18
    rho_avg = 1.0 / (rs**3 * 4 * np.pi / 3)
    k_F = (3 * np.pi**2 * rho_avg) ** (1 / 3)
    Q = q / k_F

    diff_mu = diffvc(rho_avg)
    A = 1 / 4 - (k_F**2) / (4 * np.pi) * diff_mu

    diff_rse = diffv_cep(rs)
    C = np.pi / (2 * k_F) * (-diff_rse)

    a1, a2, b1, b2 = 2.15, 0.435, 1.57, 0.409
    n = 4 if rs == 10 else 8
    x = rs**0.5
    B = (1 + a1 * x + a2 * x**3) / (3 + b1 * x + b2 * x**3)

    G = (((A - C) ** (-n) + (Q**2 / B) ** n) ** (-1 / n) + C) * Q**2
    return G


def corradini_pz(rs, q):
    """Corradini–Perdew-Zunger f_xc(q) in atomic units."""
    q = q + 1e-18
    rho_avg = 1.0 / (rs**3 * 4 * np.pi / 3)
    k_F = (3 * np.pi**2 * rho_avg) ** (1 / 3)
    Q = q / k_F

    diff_mu = diffvc(rho_avg)
    A = 1 / 4 - (k_F**2) / (4 * np.pi) * diff_mu

    diff_rse = diffv_cep(rs)
    C = np.pi / (2 * k_F) * (-diff_rse)

    a1, a2, b1, b2 = 2.15, 0.435, 1.57, 0.409
    x = rs**0.5
    B = (1 + a1 * x + a2 * x**3) / (3 + b1 * x + b2 * x**3)

    g = B / (A - C)
    alpha = 1.5 / (rs**0.25) * A / (B * g)
    beta = 1.2 / (B * g)
    Gcor = C * Q**2 + (B * Q**2) / (g + Q**2) + alpha * Q**4 * np.exp(-beta * Q**2)
    return -4 * np.pi / q**2 * Gcor


# ---------------------------------------------------------------------------
# Reference chi models
# ---------------------------------------------------------------------------


def get_chi_Moroni(rs, Ne, qlist):
    """Compute chi(q) using Moroni G(q) local field factor."""
    Vc = 4 * np.pi / qlist**2
    chi0 = chi0q(qlist, Ne, rs)
    G = G_Moroni(rs, qlist)
    fxc = -Vc * G
    return chi0 / (1 - chi0 * (Vc + fxc))


def get_chi_RPA(rs, Ne, qlist):
    """Compute RPA chi(q) = chi_0 / (1 - chi_0 * V_c)."""
    Vc = 4 * np.pi / qlist**2
    chi0 = chi0q(qlist, Ne, rs)
    return chi0 / (1 - chi0 * Vc)


def get_chi_corradini(rs, Ne, qlist):
    """Compute chi(q) using Corradini f_xc(q)."""
    Vc = 4 * np.pi / qlist**2
    chi0 = chi0q(qlist, Ne, rs)
    fxc = corradini_pz(rs, qlist)
    return chi0 / (1 - chi0 * (Vc + fxc))


# ---------------------------------------------------------------------------
# Finite-size correction
# ---------------------------------------------------------------------------


def FS_correct(chiq, correction, rs, Ne, dft_func=None):
    """
    Apply additive finite-size correction to chi^{-1}(q).

    chi_corrected = (chi_raw^{-1} + correction)^{-1}
    """
    chi_inv = chiq ** (-1) + correction
    return chi_inv ** (-1)
