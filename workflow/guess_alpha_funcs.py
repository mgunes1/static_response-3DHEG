import numpy as np


def guess_alpha0(rs, nelec, qidx):
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


def guess_alpha2(rs, nelec, qidx):
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
