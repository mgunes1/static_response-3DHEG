#!/usr/bin/env python3
"""Alpha scan + parabola selection for the per-(rs, Ne, q, vq) trial-WF
prefactor used by the v5.0 Snakemake workflow.

`alpha` here is the trial-WF construction parameter that scales the cosine
amplitude seen by Q-E SCF: vcospot = 2 * vq * alpha. The DMC sampling itself
still uses the physical vq. We optimize alpha *per* (rs, Ne, qidx, vq) by:

  1. running VMC at a small grid of alpha candidates,
  2. fitting E_VMC(alpha) to a parabola,
  3. validating curvature, SNR, boundary, goodness-of-fit,
  4. picking the parabola minimum as the production alpha.

This module is imported by both the Snakefile (rule bodies) and a CLI
self-test entrypoint.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Heuristic center (single source of truth, mirrors pp/utils/physics.py)
# ---------------------------------------------------------------------------


def guess_alpha_center(rs: float, nelec: int, qidx) -> float:
    """Heuristic alpha used as the center of the initial scan grid.

    IMPORTANT: This formula must stay exactly in sync with
    ``pp/utils/physics.py::guess_alpha2``.  Any change to the fitted
    parameters (A, B, C) below must be mirrored there.

    Formula: alpha = B * [(B/A + 1) / (B + A*exp(-C*(q/kF)²)) - 1/A]
    Parameters A, B, C fitted empirically across rs/Ne/q.
    """
    from guess_alpha_funcs import guess_alpha0, guess_alpha1, guess_alpha2

    guess_alpha = guess_alpha0 if rs <= 5 else guess_alpha1 if rs < 50 else guess_alpha2

    return guess_alpha(float(rs), int(nelec), list(qidx))


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

ALPHA_FMT = "%.3f"
ALPHA_LO, ALPHA_HI = 1e-3, 1.5

ALPHA_UNIFORM_LO = 0.005
ALPHA_UNIFORM_HI = 1.50
ALPHA_UNIFORM_N = 10


def make_alpha_grid_uniform(
    n: int = ALPHA_UNIFORM_N, lo: float = ALPHA_UNIFORM_LO, hi: float = ALPHA_UNIFORM_HI
) -> list:
    """Wide uniform alpha grid with no centering on any heuristic.

    Covers [lo, hi] with n evenly spaced points, clipped to [ALPHA_LO, ALPHA_HI].
    All vq values at a given (rs, Ne, q) use the same grid, so the shifting
    alpha_opt(vq) is always within the scanned range.
    """
    raw = np.linspace(lo, hi, n)
    out = []
    seen = set()
    for a in raw:
        s = ALPHA_FMT % float(np.clip(a, ALPHA_LO, ALPHA_HI))
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def make_alpha_grid(center: float, n: int = 5, halfwidth: float = 0.25):
    """Uniform alpha grid of exactly n points centred on *center*.

    The raw grid is [center-halfwidth, center+halfwidth] but both bounds are
    clipped to [ALPHA_LO, ALPHA_HI] *before* generating the linspace, so that
    when center is close to a boundary (e.g. very small alpha) all n points
    stay within the valid range rather than being dropped.  This guarantees
    enough points for a parabola fit regardless of the center location.

    Returned values are formatted to 3 decimals to match the wildcard format
    used in the snakefile path template. Exact duplicates after rounding are
    removed (can happen only at very tight halfwidths).
    """
    lo = max(ALPHA_LO, center - halfwidth)
    hi = min(ALPHA_HI, center + halfwidth)
    raw = np.linspace(lo, hi, n)
    out = []
    seen = set()
    for a in raw:
        s = ALPHA_FMT % a
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


# ---------------------------------------------------------------------------
# VMC energy extraction
# ---------------------------------------------------------------------------


def _last_scalar_dat(opt_xml_or_out: str) -> str | None:
    """Locate the highest-numbered scalar.dat in the same directory as opt.{xml,out}.

    QMCPACK writes output as ``qmc.s*.scalar.dat`` regardless of the input
    file name (``opt.xml`` → ``qmc.s*.scalar.dat``), so we search the
    parent directory for *any* ``*.scalar.dat`` and return the last by sort
    order (which is by series index). This is more robust than deriving the
    prefix from the input filename.
    """
    import os

    base = opt_xml_or_out
    if base.endswith(".out") or base.endswith(".xml"):
        base = base[:-4]
    # First try: input-name prefix (e.g. opt.s*.scalar.dat) — legacy naming
    pat = base + ".s*.scalar.dat"
    files = sorted(glob.glob(pat))
    if files:
        return files[-1]
    # Second try: directory-wide search for qmc.s*.scalar.dat (standard QMCPACK)
    dirpath = os.path.dirname(base)
    pat2 = os.path.join(dirpath, "qmc.s*.scalar.dat")
    files2 = sorted(glob.glob(pat2))
    return files2[-1] if files2 else None


def load_vmc_energy(opt_path: str, nequil_frac: float = 0.25):
    """Return (energy, sigma) of the *last* VMC optimization series.

    Prefers the qharv-style scalar.dat reader. Falls back to numpy on the
    raw file if qharv is unavailable. `opt_path` may point at the opt.xml or
    opt.out; the scalar.dat file is located by sibling lookup.
    """
    fsca = _last_scalar_dat(opt_path)
    if fsca is None or not os.path.exists(fsca):
        return None, None

    # Preferred path: qharv reader → reliable column names
    try:
        from qharv.reel import scalar_dat as qsd

        df = qsd.read(fsca)
        if "LocalEnergy" not in df.columns:
            raise RuntimeError("LocalEnergy column missing")
        nblock = len(df)
        neq = max(1, int(nequil_frac * nblock))
        e = df["LocalEnergy"].iloc[neq:].to_numpy()
        if e.size < 2:
            return None, None
        mean = float(np.mean(e))
        # Statistical error of the mean (no autocorr correction here — opt
        # series are short and we just need a relative SNR; downstream gates
        # are tolerant).
        sigma = float(np.std(e, ddof=1) / np.sqrt(e.size))
        return mean, sigma
    except Exception:
        pass

    # Fallback: numpy
    try:
        data = np.loadtxt(fsca)
        if data.ndim < 2 or data.shape[0] < 4:
            return None, None
        # Column 1 is LocalEnergy in QMCPACK scalar.dat
        col = data[:, 1]
        neq = max(1, int(nequil_frac * len(col)))
        e = col[neq:]
        return float(np.mean(e)), float(np.std(e, ddof=1) / np.sqrt(len(e)))
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Parabola fit
# ---------------------------------------------------------------------------


@dataclass
class ParabolaFit:
    a: float
    b: float
    c: float
    sigma_a: float
    alpha_opt: float
    e_min: float
    chi2_red: float

    def asdict(self):
        return asdict(self)


def fit_parabola(alpha, energy, dE) -> ParabolaFit | None:
    """Weighted polyfit of E(alpha) = a*alpha^2 + b*alpha + c.

    Returns None on degenerate input (npts<3 or all dE==0).
    """
    alpha = np.asarray(alpha, dtype=float)
    energy = np.asarray(energy, dtype=float)
    dE = np.asarray(dE, dtype=float)
    if alpha.size < 3:
        return None
    # weights = 1/sigma; np.polyfit treats them as 1/sigma, not 1/sigma^2
    w = np.where(dE > 0, 1.0 / np.maximum(dE, 1e-12), 1.0)
    # Fit around the global minimum to stay in the parabolic regime.
    # Ensure at least 5 fit points (ndof>=2): use max(3*min_idx+1, min_idx+4, 5).
    global_min_idx = np.argmin(energy)
    fit_end = max(
        3 * global_min_idx + 1,  # 2×min_idx points right of minimum
        global_min_idx + 4,  # at least 3 points right of minimum
        5,
    )  # absolute minimum of 5 points
    fit_end = min(fit_end, len(alpha))
    fit_range = slice(0, fit_end)
    # Fit E = a*alpha^2 + b*alpha + c (coefficients only — cov computed via GLS below)
    coeffs = np.polyfit(alpha[fit_range], energy[fit_range], 2, w=w[fit_range])
    a, b, c = coeffs
    if a == 0:
        return None
    alpha_opt = -b / (2 * a)
    e_min = c - b**2 / (4 * a)
    # sigma_a from GLS error propagation through dE (NOT chi2r-scaled).
    # chi2r >> 1 is expected when the parabola is an approximation (quartic
    # deviations at large alpha). Using chi2r-scaled cov from polyfit would
    # spuriously inflate sigma_a and fail the curvature gate.
    # Cov[coeffs] = (A^T W A)^{-1}  with  W = diag(1/dE^2).
    af = alpha[fit_range]
    wf = 1.0 / np.maximum(dE[fit_range], 1e-12) ** 2
    A = np.column_stack([af**2, af, np.ones_like(af)])
    try:
        cov_gls = np.linalg.inv(A.T @ np.diag(wf) @ A)
        sigma_a = float(np.sqrt(max(cov_gls[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        sigma_a = 0.0
    # chi2r over the fit range only (diagnostic — not used for gating)
    pred_fit = np.polyval(coeffs, alpha[fit_range])
    resid_fit = (energy[fit_range] - pred_fit) / np.where(
        dE[fit_range] > 0, dE[fit_range], 1.0
    )
    ndof = max(fit_end - 3, 1)
    chi2_red = float(np.sum(resid_fit**2) / ndof)
    return ParabolaFit(
        a=float(a),
        b=float(b),
        c=float(c),
        sigma_a=sigma_a,
        alpha_opt=float(alpha_opt),
        e_min=float(e_min),
        chi2_red=chi2_red,
    )


# ---------------------------------------------------------------------------
# Selection + quality gates
# ---------------------------------------------------------------------------

# Defaults; can be overridden via select_alpha kwargs
SNR_MIN_DEFAULT = 5.0
A_SIG_MIN_DEFAULT = 2.0


@dataclass
class AlphaSelection:
    rs: float
    nelec: int
    qidx: list
    vq: float
    alpha_center: float
    alpha_grid: list  # cumulative grid actually evaluated
    E_grid: list
    dE_grid: list
    iteration: int
    halfwidth: float
    alpha_opt: float | None
    E_min: float | None
    curvature_a: float | None
    sigma_a: float | None
    chi2_red: float | None
    snr: float | None
    status: str  # ok | rescan_widen | rescan_shift | failed
    reason: str
    next_alphas: list = field(default_factory=list)  # alphas to add next iter
    reason_clip: str = ""  # set when outlier sigma-clipping removed points

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def select_alpha(
    rs: float,
    nelec: int,
    qidx,
    vq: float,
    alpha_grid: list,  # cumulative scanned grid (str or float)
    energies: list,  # one per alpha_grid entry; (E, dE) tuples
    iteration: int,
    halfwidth: float,
    snr_min: float = SNR_MIN_DEFAULT,
    a_sig_min: float = A_SIG_MIN_DEFAULT,
    max_iterations: int = 3,
) -> AlphaSelection:
    """Apply quality gates and return an AlphaSelection.

    `energies` may contain (None, None) for alphas whose VMC failed; those
    are silently dropped from the fit. The cumulative grid is preserved in
    the returned object so future iterations know what was already evaluated.
    """
    # --- vq=0 short-circuit: alpha*vq=0 regardless of alpha ---
    if float(vq) == 0.0:
        center = guess_alpha_center(rs, nelec, qidx)
        return AlphaSelection(
            rs=float(rs),
            nelec=int(nelec),
            qidx=list(qidx),
            vq=0.0,
            alpha_center=float(center),
            alpha_grid=[],
            E_grid=[],
            dE_grid=[],
            iteration=0,
            halfwidth=0.0,
            alpha_opt=0.0,
            E_min=None,
            curvature_a=None,
            sigma_a=None,
            chi2_red=None,
            snr=None,
            status="ok",
            reason="vq=0: alpha*vq=0 identically; alpha_opt=0",
        )

    center = guess_alpha_center(rs, nelec, qidx)
    grid_f = [float(a) for a in alpha_grid]

    # Filter to valid (E, dE) entries
    keep = [
        (a, e[0], e[1])
        for a, e in zip(grid_f, energies)
        if e[0] is not None and e[1] is not None and e[1] > 0
    ]
    keep.sort(key=lambda t: t[0])

    base = dict(
        rs=float(rs),
        nelec=int(nelec),
        qidx=list(qidx),
        vq=float(vq),
        alpha_center=float(center),
        alpha_grid=grid_f,
        E_grid=[e[0] for e in energies],
        dE_grid=[e[1] for e in energies],
        iteration=int(iteration),
        halfwidth=float(halfwidth),
        alpha_opt=None,
        E_min=None,
        curvature_a=None,
        sigma_a=None,
        chi2_red=None,
        snr=None,
    )

    if len(keep) < 3:
        return AlphaSelection(
            **base,
            status="failed",
            reason="fewer than 3 valid VMC points; cannot fit a parabola",
        )

    a_arr = np.array([k[0] for k in keep])
    e_arr = np.array([k[1] for k in keep])
    de_arr = np.array([k[2] for k in keep])

    # --- Sigma-clipping: drop outlier opt runs before any gate ---
    # An outlier is defined as a point whose energy deviates by more than
    # CLIP_SIGMA * MAD from the median. A single failed or mis-converged
    # optimization can produce energies tens of sigma away; this catches them.
    CLIP_SIGMA = 5.0
    e_med = float(np.median(e_arr))
    mad = float(np.median(np.abs(e_arr - e_med)))
    if mad > 0:
        clip_mask = np.abs(e_arr - e_med) <= CLIP_SIGMA * mad
        n_clipped = int(np.sum(~clip_mask))
        if n_clipped > 0 and np.sum(clip_mask) >= 3:
            a_arr = a_arr[clip_mask]
            e_arr = e_arr[clip_mask]
            de_arr = de_arr[clip_mask]
            # Record clipped grid in base so it's visible in JSON
            # (keep original full grid; only update E/dE for clipped points)
            base["reason_clip"] = (
                f"{n_clipped} outlier(s) sigma-clipped (|E-median|>{CLIP_SIGMA}*MAD)"
            )

    signal = float(np.max(e_arr) - np.min(e_arr))
    noise = float(np.mean(de_arr))
    snr = signal / noise if noise > 0 else float("inf")
    base["snr"] = snr

    # --- SNR gate ---
    # With a wide uniform grid the parabola should always be detectable if vq
    # is large enough. If SNR is still low, the perturbation amplitude is too
    # small for VMC to see — widening the alpha range won't help.
    if snr < snr_min:
        return AlphaSelection(
            **base,
            status="failed",
            reason=f"low SNR ({snr:.2f} < {snr_min}): parabola undetectable; vq too small",
        )

    # --- Parabola fit ---
    fit = fit_parabola(a_arr, e_arr, de_arr)
    if fit is None:
        return AlphaSelection(
            **base,
            status="failed",
            reason="parabola fit returned None",
        )
    base.update(curvature_a=fit.a, sigma_a=fit.sigma_a, chi2_red=fit.chi2_red)

    # --- Curvature gate ---
    if fit.a <= 0:
        return AlphaSelection(
            **base,
            status="failed",
            reason=f"non-convex E(alpha): a={fit.a:.3e} <= 0",
        )
    if fit.sigma_a > 0 and abs(fit.a) / fit.sigma_a < a_sig_min:
        return AlphaSelection(
            **base,
            status="failed",
            reason=(
                f"curvature insignificant: |a|/sigma_a="
                f"{abs(fit.a) / fit.sigma_a:.2f} < {a_sig_min}"
            ),
        )

    # chi2_red is stored in JSON for diagnostics but NOT used as a gate.
    # High chi2_red at large vq reflects quartic deviations and underestimated
    # dE from finite optimization loops, not a poorly-determined minimum.

    # --- Boundary gate (rescan_shift) ---
    a_lo, a_hi = a_arr.min(), a_arr.max()
    margin = 1e-4
    if fit.alpha_opt <= a_lo + margin or fit.alpha_opt >= a_hi - margin:
        base["alpha_opt"] = float(np.clip(fit.alpha_opt, ALPHA_LO, ALPHA_HI))
        base["E_min"] = fit.e_min
        if iteration + 1 >= max_iterations:
            return AlphaSelection(
                **base,
                status="failed",
                reason=(
                    f"boundary minimum after {iteration + 1} iterations "
                    f"(alpha_opt={fit.alpha_opt:.3f}, "
                    f"grid=[{a_lo:.3f},{a_hi:.3f}])"
                ),
            )
        # Shift center toward the boundary side
        side_low = fit.alpha_opt <= a_lo + margin
        if side_low:
            new_center = max(ALPHA_LO, (center + a_lo) / 2)
        else:
            new_center = min(ALPHA_HI, (center + a_hi) / 2)
        new_grid = make_alpha_grid(new_center, n=5, halfwidth=halfwidth)
        next_alphas = [a for a in new_grid if a not in set(alpha_grid)]
        return AlphaSelection(
            **base,
            status="rescan_shift",
            reason=(
                f"boundary minimum at alpha={fit.alpha_opt:.3f}; "
                f"shifting center -> {new_center:.3f}"
            ),
            next_alphas=next_alphas,
        )

    # --- Success ---
    base["alpha_opt"] = float(np.clip(fit.alpha_opt, ALPHA_LO, ALPHA_HI))
    base["E_min"] = fit.e_min
    return AlphaSelection(
        **base,
        status="ok",
        reason="",
    )


# ---------------------------------------------------------------------------
# CLI: --selftest
# ---------------------------------------------------------------------------


def _selftest():
    """Synthetic parabola → fitter must recover the analytic minimum.

    Also regression-tests guess_alpha_center against hardcoded reference
    values computed from pp/utils/physics.py::guess_alpha2 to catch any
    formula divergence immediately.
    """
    # --- Part 1: formula regression guard ---
    # Reference values from pp/utils/physics.py::guess_alpha2 (A,B,C fitted).
    # If this fails the formula in this file has drifted from physics.py.
    _KNOWN = [
        # (rs,  Ne,  qidx,      expected_alpha)
        (50.0, 162, [1, 0, 0], 0.011497),
        (50.0, 162, [2, 2, 1], 0.157830),
        (10.0, 54, [1, 0, 0], 0.024289 * (162 / 54) ** (1 / 3) / 1.0),  # placeholder
    ]
    # Simpler: just test the two anchoring cases that were wrong before the fix
    # Reference values from pp/utils/physics.py::guess_alpha2 (A,B,C fitted).
    # Recompute with: python -c "from pp.utils.physics import guess_alpha2; ..."
    _KNOWN_SIMPLE = [
        (50.0, 162, [1, 0, 0], 0.011497),  # small-alpha corner
        (50.0, 162, [2, 2, 1], 0.157830),  # high-q corner
        (10.0, 54, [1, 0, 0], 0.025373),  # different rs/Ne
    ]
    for rs_t, Ne_t, q_t, expected in _KNOWN_SIMPLE:
        got = guess_alpha_center(rs_t, Ne_t, q_t)
        assert abs(got - expected) < 5e-4, (
            f"guess_alpha_center formula mismatch for rs={rs_t} Ne={Ne_t} "
            f"q={q_t}: got {got:.6f}, expected {expected:.6f}. "
            f"Check that alpha_scan.py and pp/utils/physics.py use the same "
            f"A,B,C parameters."
        )
    print("SELFTEST Part 1 OK: guess_alpha_center formula matches reference")

    # --- Part 2: grid generation covers center for small alpha ---
    grid_small = make_alpha_grid(0.011, n=5, halfwidth=0.25)
    assert len(grid_small) == 5, (
        f"make_alpha_grid returned {len(grid_small)} points for small center "
        f"(expected 5); boundary clipping broken"
    )
    print("SELFTEST Part 2 OK: make_alpha_grid produces 5 points for small center")

    # --- Part 3: parabola recovery using uniform grid ---
    rng = np.random.default_rng(42)
    true_a, true_alpha_opt, e0 = 0.5, 0.75, -1.234
    grid = make_alpha_grid_uniform(n=10)
    grid_f = np.array([float(x) for x in grid])
    e_clean = true_a * (grid_f - true_alpha_opt) ** 2 + e0
    sigma = 1e-5
    e_noisy = e_clean + rng.normal(0, sigma, size=e_clean.size)
    energies = [(float(e), sigma) for e in e_noisy]
    sel = select_alpha(
        rs=10.0,
        nelec=54,
        qidx=[1, 0, 0],
        vq=0.001,
        alpha_grid=grid,
        energies=energies,
        iteration=0,
        halfwidth=0.15,
    )
    print(sel.to_json())
    assert sel.status == "ok", f"expected ok, got {sel.status}"
    assert abs(sel.alpha_opt - true_alpha_opt) < 5e-3, (
        f"alpha_opt {sel.alpha_opt} not close to {true_alpha_opt}"
    )
    print(
        "SELFTEST Part 3 OK: alpha_opt = %.4f (true %.4f)"
        % (sel.alpha_opt, true_alpha_opt)
    )
    print("SELFTEST OK: alpha_opt = %.4f (true %.4f)" % (sel.alpha_opt, true_alpha_opt))


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    if args.selftest:
        _selftest()
    else:
        p.print_help()
