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


def q_over_kf(rs, Ne, qidx):
    """q/kF for cubic box; rs-independent."""
    rs = float(rs)
    Ne = int(Ne)
    q_mag = np.linalg.norm(qidx)
    L = (4 * np.pi / 3 * rs**3 * Ne) ** (1 / 3)
    kF = (9 * np.pi / 4) ** (1 / 3) / rs
    q_norm = 2 * np.pi / L * q_mag

    return q_norm / kF


def get_alpha(qidx, rs, Ne):
    """Analytical fit to optimal alpha/vq from v5.0 scan (Moroni 1995 range)."""
    A0, a = 1.92757669, -0.60504965
    B0, b = 1.16030916, 0.30130412
    A = A0 * float(rs) ** a
    B = B0 * float(rs) ** b

    q_over_kF = q_over_kf(rs, Ne, qidx)
    return 1 - np.exp(-A * float(q_over_kF) ** B)


def guess_alpha2(rs, nelec, qidx):
    """
    Guess alpha based on a simple hyperbolic tangent function.
    Usage range: rs = 0-5
    """
    kf = (9 * np.pi / 4) ** (1 / 3) / rs  # 3D gas
    alat = (nelec * 4 * np.pi / 3) ** (1.0 / 3) * rs
    blat = 2 * np.pi / alat
    qvec = blat * np.array(qidx)
    qmag = np.linalg.norm(qvec)
    alpha = np.tanh(qmag / kf)
    return alpha


def guess_alpha1(rs, nelec, qidx):
    """
    Guess alpha based on a sigmoid function.
    Usage range: rs = 5-50 (needs to be tested more thoroughly)
    """
    a = 1.2
    kf = (9 * np.pi / 4) ** (1 / 3) / rs  # 3D gas
    alat = (nelec * 4 * np.pi / 3) ** (1.0 / 3) * rs
    blat = 2 * np.pi / alat
    qvec = blat * np.array(qidx)
    qmag = np.linalg.norm(qvec)
    alpha = 2 / (1 + np.exp(-a * (qmag / kf) ** 2)) - 1
    return alpha


def guess_alpha2x(rs, nelec, qidx):
    """
    Guess alpha based on a sigmoid function.
    Usage range: rs > 50 (needs to be tested more thoroughly)
    """
    A, B, C = 1.69387154, 0.15297875, 0.94657354
    kf = (9 * np.pi / 4) ** (1 / 3) / rs  # 3D gas
    alat = (nelec * 4 * np.pi / 3) ** (1.0 / 3) * rs
    blat = 2 * np.pi / alat
    qvec = blat * np.array(qidx)
    qmag = np.linalg.norm(qvec)
    alpha = (
        (B * A ** (-1) + 1) / (B + A * np.exp(-C * (qmag / kf) ** 2)) - A ** (-1)
    ) * B
    return alpha


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
# Local field factors
# ---------------------------------------------------------------------------


def get_G_from_chi(rs, Ne, qidx_list, chi):
    """Extract G(q) from chi(q) via the relation chi = chi0 / (1 - chi0 * (Vc + fxc))."""
    ql = get_qs(qidx_list, Ne, rs)
    Vc = 4 * np.pi / ql**2
    chi0 = chi0q(ql, Ne, rs)
    G = 1 + (1 / chi - 1 / chi0) / Vc
    return G


# ---------------------------------------------------------------------------
# Reference chi models
# ---------------------------------------------------------------------------


def get_chi_Moroni(rs, Ne, qlist, n=None):
    """Compute chi(q) using Moroni G(q) local field factor."""
    from .fitting_G import G_Moroni

    Vc = 4 * np.pi / qlist**2
    chi0 = chi0q(qlist, Ne, rs)
    G = G_Moroni(rs, qlist, n=n)
    fxc = -Vc * G
    return chi0 / (1 - chi0 * (Vc + fxc))


def get_chi_RPA(rs, Ne, qlist):
    """Compute RPA chi(q) = chi_0 / (1 - chi_0 * V_c)."""
    Vc = 4 * np.pi / qlist**2
    chi0 = chi0q(qlist, Ne, rs)
    return chi0 / (1 - chi0 * Vc)


def get_chi_corradini(rs, Ne, qlist):
    """Compute chi(q) using Corradini f_xc(q)."""
    from .fitting_G import G_Corradini

    Vc = 4 * np.pi / qlist**2
    chi0 = chi0q(qlist, Ne, rs)
    fxc = -Vc * G_Corradini(rs, qlist)
    return chi0 / (1 - chi0 * (Vc + fxc))


def get_chi_gunes(rs, Ne, qlist):
    """Compute chi(q) using gunes f_xc(q)."""
    from .fitting_G import G_mhg

    Vc = 4 * np.pi / qlist**2
    chi0 = chi0q(qlist, Ne, rs)
    fxc = -Vc * G_mhg(rs, qlist)
    return chi0 / (1 - chi0 * (Vc + fxc))


def get_chi_kaplan(rs, Ne, qlist):
    """Compute chi(q) using Kaplan f_xc(q)."""
    from .fitting_G import G_Kaplan

    Vc = 4 * np.pi / qlist**2
    chi0 = chi0q(qlist, Ne, rs)
    fxc = -Vc * G_Kaplan(rs, qlist)
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
