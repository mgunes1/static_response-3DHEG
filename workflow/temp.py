import matplotlib

matplotlib.use("Agg")
import json
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/mnt/ceph/users/mgunes/HEG_dmc/static_response/v5.0/workflow")
from alpha_scan import select_alpha

run_dir = "/mnt/ceph/users/mgunes/HEG_dmc/static_response/v5.0/runs/rs50.0-n162"
vq_list = ["0.00000", "0.00001", "0.00002", "0.00003", "0.00004"]
vq_float = [0.0, 1e-5, 2e-5, 3e-5, 4e-5]

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle(
    "E(α) scan — q=[2,2,1], rs=50.0, Ne=162\nTop: raw data  |  Bottom: with σ-clipping applied",
    fontsize=12,
    fontweight="bold",
    y=1.01,
)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

for col_idx, (vq_str, vq_f, col) in enumerate(zip(vq_list, vq_float, colors)):
    d = json.load(open(f"{run_dir}/qv2_2_1-vq{vq_str}/ni-e125-alpha_iter0.json"))
    g = np.array(d["alpha_grid"], dtype=float)
    E = np.array(d["E_grid"], dtype=float)
    dE = np.array(d["dE_grid"], dtype=float)

    # Outlier mask via MAD
    e_med = np.median(E)
    mad = np.median(np.abs(E - e_med))
    is_out = (np.abs(E - e_med) > 5 * mad) if mad > 0 else np.zeros(len(E), bool)

    # ---------- ROW 0: raw ----------
    ax0 = axes[0, col_idx]
    ax0.errorbar(
        g[~is_out], E[~is_out], yerr=dE[~is_out], fmt="o", color=col, capsize=3, ms=5
    )
    if any(is_out):
        ax0.errorbar(
            g[is_out],
            E[is_out],
            yerr=dE[is_out],
            fmt="x",
            color="red",
            capsize=3,
            ms=9,
            mew=2.5,
            label="outlier",
        )
        ax0.legend(fontsize=7, loc="best")
    status_raw = d["status"]
    snr_raw = d.get("snr", 0)
    ax0.set_title(f"vq={vq_str}\n{status_raw} | snr={snr_raw:.1f}", fontsize=9)
    if col_idx == 0:
        ax0.set_ylabel("E/N (Ha)", fontsize=9)

    # ---------- ROW 1: clipped ----------
    ax1 = axes[1, col_idx]
    energies = list(zip(d["E_grid"], d["dE_grid"]))
    result = select_alpha(
        50.0,
        162,
        [2, 2, 1],
        vq_f,
        d["alpha_grid"],
        energies,
        iteration=0,
        halfwidth=0.25,
    )

    gc = g[~is_out]
    Ec = E[~is_out]
    dEc = dE[~is_out]
    ax1.errorbar(gc, Ec, yerr=dEc, fmt="o", color=col, capsize=3, ms=5, label="clean")
    if any(is_out):
        ax1.errorbar(
            g[is_out],
            E[is_out],
            yerr=dE[is_out],
            fmt="x",
            color="lightgray",
            capsize=3,
            ms=7,
            mew=1.5,
            alpha=0.5,
            label="clipped",
        )
        ax1.legend(fontsize=7)

    # Draw fit if ok
    if result.status == "ok" and result.alpha_opt is not None:
        a_, b_, c_ = (
            result.curvature_a,
            -2 * result.curvature_a * result.alpha_opt,
            result.E_min + result.curvature_a * result.alpha_opt**2,
        )
        gf = np.linspace(gc.min() - 0.05, gc.max() + 0.05, 200)
        ax1.plot(gf, result.curvature_a * gf**2 + b_ * gf + c_, "-k", lw=1.2)
        ax1.axvline(
            result.alpha_opt,
            color="red",
            ls="--",
            lw=0.9,
            label=f"α*={result.alpha_opt:.3f}",
        )
        ax1.legend(fontsize=7)

    clip_note = f"\n({result.reason_clip[:25]})" if result.reason_clip else ""
    ax1.set_title(f"{result.status} | snr={result.snr:.1f}{clip_note}", fontsize=8)
    ax1.set_xlabel("α", fontsize=9)
    if col_idx == 0:
        ax1.set_ylabel("E/N (Ha) [clipped]", fontsize=9)

    # Shared y-axis limits: zoom in on clean range
    spread = max(np.ptp(Ec) * 2.5, 4e-4)
    for ax in [ax0, ax1]:
        ax.set_ylim(np.median(Ec) - spread, np.median(Ec) + spread)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("α", fontsize=9)

plt.tight_layout()
plt.savefig("/tmp/E_alpha_q221_before_after.png", dpi=150, bbox_inches="tight")
print("Saved /tmp/E_alpha_q221_before_after.png")
