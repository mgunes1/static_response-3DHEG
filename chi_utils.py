import numpy as np


def chi0q(q, Ne, rs):  # one spin
    "Lindhard function (reciprocal space) from Vignale, eqn. 18 in the report"
    # L = (4*np.pi/3*rs**3*Ne)**(1/3)
    n0 = (4 * np.pi / 3 * rs**3) ** (-1) / 2
    kF = (6 * np.pi**2 * n0) ** (1 / 3)

    filter_low = q < 1e-10
    filter_high = q >= 1e-10
    q = q * filter_high + 1e-10 * filter_low
    Q = q / (kF)
    res = (
        -kF / (2 * np.pi**2) * (1 - (Q / 4 - 1 / Q) * np.log(np.abs((Q + 2) / (Q - 2))))
    )
    return res


def corradini_pz(r_s, q):
    q = q + 1e-18
    # Q is given in multiples of k_F (Q = q / k_F)
    rho_avg = 1.0 / (r_s**3.0 * 4.0 * np.pi / 3.0)
    k_F = (3.0 * np.pi**2.0 * rho_avg) ** (1.0 / 3.0)
    Q = q / k_F
    # Q = q
    e = 1  # atomic units
    #   diff_mu = 1                 # How to model this for HEG?
    #                                 This should be d \mu_c / d n_0
    diff_mu = diffvc(rho_avg)
    A = 1.0 / 4.0 - (k_F**2.0) / (4.0 * np.pi * e**2.0) * diff_mu
    #   diff_rse = 1                # How to model this for HEG? e_c(r_s) !!
    #                                 This should be d(r_s * e_c) / d r_s
    diff_rse = diffv_cep(r_s)
    C = np.pi / (2.0 * e**2.0 * k_F) * (-diff_rse)
    a1 = 2.15
    a2 = 0.435
    b1 = 1.57
    b2 = 0.409
    x = r_s ** (1.0 / 2.0)
    B = (1.0 + a1 * x + a2 * x**3.0) / (3.0 + b1 * x + b2 * x**3.0)
    g = B / (A - C)
    alpha = 1.5 / (r_s ** (1.0 / 4.0)) * A / (B * g)
    beta = 1.2 / (B * g)
    Gcor = (
        C * Q**2.0
        + (B * Q**2.0) / (g + Q**2.0)
        + alpha * Q**4.0 * np.exp(-beta * Q**2.0)
    )
    return -4.0 * np.pi * e**2.0 / (q**2.0) * Gcor


def G_Moroni(rs, q):
    q = q + 1e-18
    # Q is given in multiples of k_F (Q = q / k_F)
    rho_avg = 1.0 / (rs**3.0 * 4.0 * np.pi / 3.0)
    k_F = (3.0 * np.pi**2.0 * rho_avg) ** (1.0 / 3.0)
    Q = q / k_F
    # Q = q
    e = 1  # atomic units
    #   diff_mu = 1                 # How to model this for HEG?
    #                                 This should be d \mu_c / d n_0
    diff_mu = diffvc(rho_avg)
    A = 1.0 / 4.0 - (k_F**2.0) / (4.0 * np.pi * e**2.0) * diff_mu
    #   diff_rse = 1                # How to model this for HEG? e_c(r_s) !!
    #                                 This should be d(r_s * e_c) / d r_s
    diff_rse = diffv_cep(rs)
    C = np.pi / (2.0 * e**2.0 * k_F) * (-diff_rse)
    a1 = 2.15
    a2 = 0.435
    b1 = 1.57
    b2 = 0.409
    n = 8
    if rs == 10:
        n = 4
    x = rs ** (1.0 / 2.0)
    B = (1.0 + a1 * x + a2 * x**3.0) / (3.0 + b1 * x + b2 * x**3.0)
    G = (((A - C) ** (-n) + (Q**2 / B) ** n) ** (-1 / n) + C) * Q**2
    return G


def diffv_cep(r_s):
    # Multiplied v_cep from above with r_s and differentiated wrt r_s
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334
    # res = gamma * (beta1 * np.sqrt(r_s) + 2) / \
    #    (2 * (beta1 * np.sqrt(r_s) + beta2 * r_s + 1) ** 2)
    res = (beta1 * gamma * np.sqrt(r_s)) / (
        2 * (beta1 * np.sqrt(r_s) + beta2 * r_s + 1) ** 2
    ) + gamma / (beta1 * np.sqrt(r_s) + beta2 * r_s + 1) ** 2
    return res


def diffvc(rho):
    # from dp-code
    third = 1.0 / 3.0
    a = 0.0311
    # b = -0.0480  #  never used?
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
    from numpy import pi, sqrt

    a = 0.0311
    b = -0.0480
    c = 0.0020
    d = -0.0116
    gamma = -0.1423
    beta1 = 1.0529
    beta2 = 0.3334
    rho = 4 * pi / 3 * rs**3
    # rho=0.1
    # rs=(3.0/(4.0*pi*rho))**athird
    # rs=8
    # two different calculations depending if Rs <> 1
    if rs >= 1:
        stor1 = (1.0 + beta1 * sqrt(rs) + beta2 * rs) ** (-3.0)
        stor2 = (
            -0.41666667 * beta1 * (rs ** (-0.5))
            - 0.5833333 * (beta1**2)
            - 0.66666667 * beta2
        )
        stor3 = -1.75 * beta1 * beta2 * sqrt(rs) - 1.3333333 * rs * (beta2**2)
        diffvc = gamma * stor1 * (stor2 + stor3)
    else:
        diffvc = a / rs + 0.66666667 * (c * np.log(rs) + d) + 0.33333333 * c

    diffvc = diffvc * (-4.0 * pi / 9.0) * (rs**4)

    ##### diffVx
    # real rho,rs,b,bb,bb1
    # real rel, vxnr, difrel
    vxnr = -((3 * rho / pi) ** 0.3333333333)
    b = 0.0140 / rs
    rel = -0.5 + 1.5 * np.log(b + sqrt(1.0 + b * b)) / (b * sqrt(1.0 + b * b))
    bb = b * b
    bb1 = 1.0 + bb
    difrel = (1.5 / (b * bb1)) - 1.5 * np.log(b + sqrt(bb1)) * (1.0 + 2.0 * bb) * (
        bb1 ** (-1.5)
    ) / bb
    difrel = difrel * (-0.0140) / (rs * rs)
    diffvx = (0.610887057 / (rs * rs)) * rel + vxnr * difrel
    diffvx = diffvx * (-4.0 * pi / 9.0) * (rs**4)
    return diffvc + diffvx


def anal_chi02(rs, Ne, qidx_list):
    L = (4 * np.pi / 3 * rs**3 * Ne) ** (1 / 3)
    n0 = Ne / L**3

    kF = (3 * np.pi**2 * n0) ** (1 / 3)  # Fermi energy

    # Define the dispersion relation
    def epsilon(k):
        return np.linalg.norm(k) ** 2 / 2

    # Define the step function for n_k at T = 0
    def n_k(k):
        return 1 if epsilon(k) <= epsilon(kF) else 0

    ########
    # ic = 9
    ########
    # cl=2*np.pi/L #spacing of the k grid.
    # k_vals = np.arange(-ic, ic+1)*cl

    # Generate k-point grid
    # kx, ky, kz = np.meshgrid(k_vals, k_vals, k_vals, indexing='ij')
    # k_grid = np.stack([kx.ravel( ), ky.ravel(), kz.ravel()], axis=-1)
    if Ne == 54:
        n_shell = 4
    elif Ne == 162:
        n_shell = 7
    elif Ne == 294:
        n_shell = 10
    elif Ne == 406:
        n_shell = 13
    else:
        # no other choice, raise error.
        raise ValueError(
            f"Ne={Ne} is not a supported value. If this is a new calculation please add the corresponding n_shell value in this function."
        )

    k_grid = get_shell_points(n_shell) * 2 * np.pi / L

    q_grid = 2 * np.pi / L * np.array(qidx_list)
    # q_vals=np.linalg.norm(qs,axis=1)
    vq = 0
    # Compute the Lindhard function in the static limit
    chi0 = np.zeros(len(q_grid))
    Etot = np.zeros(len(q_grid))
    for i in range(len(q_grid)):
        chi0_q = 0
        count = 0
        Etot_q = 0
        for k in k_grid:
            count += 2
            k_plus_q = k + q_grid[i]
            epsilon_k = epsilon(k)
            epsilon_k_plus_q = epsilon(k_plus_q)
            n_k_val = n_k(k)
            n_k_plus_q_val = n_k(k_plus_q)

            # print(n_k_val,n_k_plus_q_val)
            chi0_q += (n_k_val - n_k_plus_q_val + 0.000000001) / (
                epsilon_k - epsilon_k_plus_q + 0.000000001
            )
            Etot_q += (epsilon_k + epsilon_k_plus_q) / 2 - np.sqrt(
                ((epsilon_k - epsilon_k_plus_q) / 2) ** 2 + vq**2
            )

        Etot[i] = Etot_q
        chi0[i] = chi0_q * 4

    # Normalize by the volume of the system (L^3)
    chi0 /= L**3

    return chi0


def get_shell_points(shell_number):
    """
    Return all integer k-grid points (i,j,k) inside the cumulative shell 'shell_number'.
    G sets the cube range: -G ≤ i,j,k ≤ G. Typically G is the maximum |k| index included.

    Example:
        shell_number=1 → [(0,0,0)]
        shell_number=2 → (0,0,0) + the 6 axis points → 7 total
    """
    G = shell_number * 2

    # Collect all points and their squared radii
    pts = []
    radii = set()

    for i in range(-G, G + 1):
        for j in range(-G, G + 1):
            for k in range(-G, G + 1):
                m = i * i + j * j + k * k
                pts.append((m, [i, j, k]))
                radii.add(m)

    # Sort distinct radii -> these define the shells
    radii = sorted(radii)

    # Select the m corresponding to this shell
    if shell_number < 1 or shell_number > len(radii):
        raise ValueError("shell_number is out of range for this G.")
    m_cut = radii[shell_number - 1]

    # Return all points with m <= m_cut (cumulative)
    return np.array(sorted([p for m, p in pts if m <= m_cut]))
