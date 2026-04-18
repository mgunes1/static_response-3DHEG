#!/usr/bin/env python3
"""Post-run validation for alpha-scan results.

Usage
-----
    python validate_alpha_scan.py --rs 50.0 --nelec 162 --runs_dir runs/

Reads all alpha_selected.json files for the given (rs, Ne) and reports:
  - per-point status (ok / rescan_* / failed)
  - parabola curvature, SNR, chi2_red
  - alpha_opt vs guess_alpha2 heuristic
  - alpha_manifest JSON summary

Optionally (--plot) generates E(alpha) parabola plots for spot-checks.

After running get_chi with / without the manifest, also compares
fit-quality metrics for DMC E(vq) fits (requires v4.00-style reference run
in --ref_dir).
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np


def load_selected_jsons(runs_dir: str, rs: float, ne: int):
    """Glob all alpha_selected.json files for this (rs, Ne)."""
    pattern = os.path.join(
        runs_dir,
        f"rs{rs:.1f}-n{ne:d}",
        "qv*",
        "ni-e*-alpha_selected.json",
    )
    paths = sorted(glob.glob(pattern))
    results = []
    for p in paths:
        with open(p) as f:
            results.append((p, json.load(f)))
    return results


def summarize(records):
    """Print a table of per-point alpha scan results."""
    print(f"\n{'q':>12}  {'vq':>9}  {'status':>14}  "
          f"{'alpha_opt':>10}  {'alpha_heur':>10}  "
          f"{'SNR':>8}  {'chi2_red':>9}  reason")
    print("-" * 100)
    n_ok = n_rescan = n_failed = 0
    for path, r in records:
        qidx = r.get("qidx", ["?", "?", "?"])
        vq = r.get("vq", float("nan"))
        status = r.get("status", "?")
        alpha_opt = r.get("alpha_opt")
        alpha_center = r.get("alpha_center")
        snr = r.get("snr")
        chi2 = r.get("chi2_red")
        reason = r.get("reason", "")[:40]

        q_str = f"[{qidx[0]},{qidx[1]},{qidx[2]}]"
        alpha_opt_s = f"{alpha_opt:.4f}" if alpha_opt is not None else "   None"
        alpha_h_s = f"{alpha_center:.4f}" if alpha_center is not None else "   None"
        snr_s = f"{snr:.1f}" if snr is not None else "    —"
        chi2_s = f"{chi2:.2f}" if chi2 is not None else "   —"

        print(f"{q_str:>12}  {vq:>9.5f}  {status:>14}  "
              f"{alpha_opt_s:>10}  {alpha_h_s:>10}  "
              f"{snr_s:>8}  {chi2_s:>9}  {reason}")

        if status == "ok":
            n_ok += 1
        elif status.startswith("rescan"):
            n_rescan += 1
        else:
            n_failed += 1

    print("-" * 100)
    total = len(records)
    print(f"  ok={n_ok}/{total}  rescan={n_rescan}/{total}  failed={n_failed}/{total}\n")
    return n_ok, n_rescan, n_failed


def alpha_delta_stats(records):
    """Compute statistics of alpha_opt - alpha_heuristic."""
    deltas = []
    for _, r in records:
        if r.get("status") == "ok" and r.get("alpha_opt") is not None:
            delta = r["alpha_opt"] - r["alpha_center"]
            deltas.append(delta)
    if not deltas:
        print("No ok points to compute alpha delta stats.")
        return
    deltas = np.array(deltas)
    print("alpha_opt - alpha_heuristic statistics (ok points only):")
    print(f"  mean  = {deltas.mean():+.4f}")
    print(f"  std   = {deltas.std():.4f}")
    print(f"  min   = {deltas.min():+.4f}")
    print(f"  max   = {deltas.max():+.4f}")
    print(f"  |delta| > 0.05: {np.sum(np.abs(deltas) > 0.05)} / {len(deltas)}")
    print()


def plot_parabolas(records, output_dir="."):
    """Plot E(alpha) with fitted parabola for each point. Requires matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)
    for path, r in records:
        if r.get("status") != "ok":
            continue
        alpha_grid = np.array(r.get("alpha_grid", []))
        E_grid = np.array(r.get("E_grid", []))
        dE_grid = np.array(r.get("dE_grid", []))
        if alpha_grid.size < 3:
            continue
        qidx = r["qidx"]
        vq = r["vq"]
        alpha_opt = r["alpha_opt"]
        curvature_a = r.get("curvature_a")
        E_min = r.get("E_min")

        fig, ax = plt.subplots(figsize=(5, 3.5))
        mask = [e is not None and de is not None for e, de in zip(E_grid, dE_grid)]
        ag = alpha_grid[mask]
        eg = E_grid[mask]
        deg = dE_grid[mask]

        ax.errorbar(ag, eg, yerr=deg, fmt="o", ms=4, label="VMC")

        # Draw fitted parabola
        if curvature_a is not None and E_min is not None:
            a_fit = curvature_a
            al_fine = np.linspace(ag.min() - 0.05, ag.max() + 0.05, 200)
            # Reconstruct parabola from curvature, opt, and min
            e_fine = a_fit * (al_fine - alpha_opt) ** 2 + E_min
            ax.plot(al_fine, e_fine, "-", lw=1.2, label="fit")
            ax.axvline(alpha_opt, color="r", ls="--", lw=0.8,
                       label=f"α_opt={alpha_opt:.3f}")

        ax.set_xlabel("α")
        ax.set_ylabel("E_VMC / N")
        ax.set_title(f"q={qidx}  vq={vq:.5f}  SNR={r.get('snr', 0):.1f}")
        ax.legend(fontsize=8)
        fig.tight_layout()

        fname = os.path.join(
            output_dir,
            f"parabola_q{qidx[0]}{qidx[1]}{qidx[2]}_vq{vq:.5f}.png",
        )
        fig.savefig(fname, dpi=120)
        plt.close(fig)
        print(f"  saved: {fname}")


def compare_fit_quality(ref_dir, new_dir, rs, Ne, qidxl, vql,
                        runs_dir, alpha_manifest):
    """
    Compare DMC E(vq) fit-quality metrics for a given (rs, Ne):
      - without manifest (ref_dir, guess_alpha2)
      - with manifest (new_dir, alpha_manifest)

    Requires the pp package to be importable.
    """
    pp_root = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "pp")
    )
    sys.path.insert(0, pp_root)
    try:
        from utils.fitting import get_chi_q
        from utils.io_utils import load_alpha_manifest
    except ImportError as e:
        print(f"Cannot import pp/utils: {e}. Skipping fit-quality comparison.")
        return

    print("=== DMC E(vq) fit-quality comparison ===")
    print(f"  ref_dir (guess_alpha2): {ref_dir}")
    print(f"  new_dir (manifest):     {new_dir}")

    for label, directory, manifest in [
        ("guess_alpha2", ref_dir, None),
        ("manifest", new_dir, alpha_manifest),
    ]:
        try:
            chi_q, dchi_q, fq = get_chi_q(
                directory, Ne, rs, vql, qidxl,
                alpha_manifest=manifest, verbose=False,
            )
            print(f"\n  [{label}]")
            for iq, (q, chi, dchi, fqi) in enumerate(
                zip(qidxl, chi_q, dchi_q, fq)
            ):
                snr = fqi.get("snr", float("nan"))
                chi2 = fqi.get("chi2_red", float("nan"))
                sig = fqi.get("A_significance", float("nan"))
                print(f"    q={q}  chi={chi:.4f}±{dchi:.4f}"
                      f"  SNR={snr:.1f}  chi2_red={chi2:.2f}  A/dA={sig:.1f}")
        except Exception as e:
            print(f"  [{label}] ERROR: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rs", type=float, required=True)
    p.add_argument("--nelec", type=int, required=True)
    p.add_argument("--runs_dir", default="runs",
                   help="Path to the v5.0 runs directory (default: runs/)")
    p.add_argument("--plot", action="store_true",
                   help="Generate E(alpha) parabola PNG plots")
    p.add_argument("--plot_dir", default="output/alpha_scan_plots",
                   help="Directory for plot output (default: output/alpha_scan_plots)")
    p.add_argument("--compare", action="store_true",
                   help="Compare fit quality vs ref (ref_dir must also be set)")
    p.add_argument("--ref_dir", default=None,
                   help="Reference runs dir for fit-quality comparison (v4.00-style)")
    args = p.parse_args()

    records = load_selected_jsons(args.runs_dir, args.rs, args.nelec)
    if not records:
        print(f"No alpha_selected.json files found in "
              f"{args.runs_dir}/rs{args.rs:.1f}-n{args.nelec}/")
        sys.exit(1)

    print(f"\nLoaded {len(records)} alpha_selected.json files "
          f"for rs={args.rs:.1f}, Ne={args.nelec}")
    summarize(records)
    alpha_delta_stats(records)

    if args.plot:
        print("Generating parabola plots ...")
        plot_parabolas(records, output_dir=args.plot_dir)

    if args.compare and args.ref_dir:
        # Build manifest dict from ok records
        manifest = {}
        for _, r in records:
            if r.get("status") == "ok" and r.get("alpha_opt") is not None:
                q = r["qidx"]
                vq = r["vq"]
                key = "%d_%d_%d/%.5f" % (q[0], q[1], q[2], float(vq))
                manifest[key] = r

        # Collect unique q and vq from records
        qidxl_seen = []
        vql_seen = set()
        for _, r in records:
            q = r["qidx"]
            if q not in qidxl_seen:
                qidxl_seen.append(q)
            vql_seen.add(float(r["vq"]))
        vql_sorted = sorted(vql_seen)

        compare_fit_quality(
            ref_dir=args.ref_dir,
            new_dir=args.runs_dir,
            rs=args.rs, Ne=args.nelec,
            qidxl=qidxl_seen,
            vql=vql_sorted,
            runs_dir=args.runs_dir,
            alpha_manifest=manifest,
        )


if __name__ == "__main__":
    main()
