#!/usr/bin/env python3
"""Compute χ(q) for the 3D HEG from QMCPACK energy matrices.

Example
-------
    python scripts/compute_chi.py --rs 2.0 --N 14
    python scripts/compute_chi.py --rs 2.0 --N 14 --plot
    python scripts/compute_chi.py --rs 2.0 --N 14 --save-results results_rs2_N14.npz
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_io import get_output_path, load_energy_matrix
from src.lindhard import chi0, chi_rpa, local_field_factor
from src.response import compute_chi_q
from src.utils import fermi_wavevector


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute static χ(q) for the 3D HEG from QMCPACK energies."
    )
    p.add_argument("--rs", type=float, required=True, help="Wigner-Seitz radius (Bohr)")
    p.add_argument("--N", type=int, required=True, help="Number of electrons")
    p.add_argument(
        "--output-dir",
        default="output",
        help="Directory containing energy matrices (default: output/)",
    )
    p.add_argument("--plot", action="store_true", help="Save a χ(q) plot")
    p.add_argument(
        "--save-results", default=None, metavar="FILE", help="Save results to .npz"
    )
    return p.parse_args()


def main():
    args = parse_args()
    rs, N = args.rs, args.N

    data_path = get_output_path(args.output_dir, rs, N)
    print(f"Loading  {data_path}.npz")

    try:
        data = load_energy_matrix(data_path)
    except FileNotFoundError:
        print(
            f"Error: no energy matrix at {data_path}.npz\n"
            "Run the QMCPACK workflow and store results in output/, "
            "or generate sample data with  python output/generate_sample_data.py"
        )
        sys.exit(1)

    energies = data["energies"]
    lambdas = data["lambdas"]
    q_mags = data["q_magnitudes"]
    errors = data.get("energy_errors")

    print(f"rs = {rs},  N = {N}")
    print(f"q-points : {len(q_mags)}")
    print(f"λ values : {lambdas}")

    # Compute χ(q) from the energy matrix
    result = compute_chi_q(energies, lambdas, q_mags, rs, N, error_matrix=errors)

    # Analytical references
    kf = fermi_wavevector(rs)
    q_norm = result["q"] / (2.0 * kf)
    chi_free = chi0(result["q"], rs)
    chi_rpa_vals = chi_rpa(result["q"], rs)
    G_q = local_field_factor(result["q"], result["chi"], rs)

    # Print table
    header = f"{'q/(2kF)':>10} {'χ_DMC':>14} {'χ_err':>12} {'χ₀':>14} {'χ_RPA':>14} {'G(q)':>10}"
    print(f"\n{header}")
    print("-" * len(header))
    for i in range(len(result["q"])):
        print(
            f"{q_norm[i]:10.4f} {result['chi'][i]:14.6f} {result['chi_err'][i]:12.6f}"
            f" {chi_free[i]:14.6f} {chi_rpa_vals[i]:14.6f} {G_q[i]:10.4f}"
        )

    # Optionally save results
    if args.save_results:
        np.savez(
            args.save_results,
            q=result["q"],
            q_norm=q_norm,
            chi_dmc=result["chi"],
            chi_err=result["chi_err"],
            chi0=chi_free,
            chi_rpa=chi_rpa_vals,
            G_q=G_q,
            rs=rs,
            N=N,
        )
        print(f"\nResults saved to {args.save_results}")

    # Optionally plot
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plot.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        q_fine = np.linspace(0.01 * 2.0 * kf, 3.0 * 2.0 * kf, 300)
        q_fine_norm = q_fine / (2.0 * kf)

        # --- left panel: χ(q) / χ₀(q) ---
        ax1.axhline(1.0, color="k", ls="--", alpha=0.4, label="Free electron")
        ax1.plot(
            q_fine_norm,
            chi_rpa(q_fine, rs) / chi0(q_fine, rs),
            "b--",
            label="RPA",
        )
        ax1.errorbar(
            q_norm,
            result["chi"] / chi_free,
            yerr=result["chi_err"] / np.abs(chi_free),
            fmt="ro",
            capsize=4,
            label=f"DMC (N={N})",
        )
        ax1.set_xlabel(r"$q\,/\,(2k_F)$", fontsize=12)
        ax1.set_ylabel(r"$\chi(q)\,/\,\chi_0(q)$", fontsize=12)
        ax1.set_title(rf"Static response, $r_s={rs}$", fontsize=12)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # --- right panel: G(q) ---
        ax2.axhline(0.0, color="k", ls="--", alpha=0.4)
        ax2.errorbar(
            q_norm,
            G_q,
            fmt="ro",
            capsize=4,
            label=f"DMC (N={N})",
        )
        ax2.set_xlabel(r"$q\,/\,(2k_F)$", fontsize=12)
        ax2.set_ylabel(r"$G(q)$", fontsize=12)
        ax2.set_title(rf"Local field factor, $r_s={rs}$", fontsize=12)
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plot_file = f"chi_q_rs{rs:.1f}_N{N}.png"
        plt.savefig(plot_file, dpi=150)
        print(f"Plot saved to {plot_file}")


if __name__ == "__main__":
    main()
