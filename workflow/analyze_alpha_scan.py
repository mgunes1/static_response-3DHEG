#!/usr/bin/env python3
"""Alpha scan analysis: plot E(alpha) for all (q, vq), show fit quality.

Usage:
    python analyze_alpha_scan.py --runs_dir runs --rs 50.0 --nelec 162 \
        [--outdir runs/plots] [--alpha_max_fit 0.50]

alpha_max_fit: upper alpha cutoff used for the restricted parabola fit
(default 0.50).  The full-grid fit is also shown for comparison.
"""
import argparse, glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── helpers ─────────────────────────────────────────────────────────────────

def _fit_parabola(a_arr, e_arr, de_arr):
    """Weighted polyfit; return (coeffs, cov, alpha_opt) or None on failure."""
    if len(a_arr) < 3:
        return None
    w = 1.0 / np.maximum(de_arr, 1e-12)
    try:
        coeffs, cov = np.polyfit(a_arr, e_arr, 2, w=w, cov=True)
    except Exception:
        return None
    a, b, c = coeffs
    if a == 0:
        return None
    alpha_opt = -b / (2 * a)
    sigma_a   = float(np.sqrt(max(cov[0, 0], 0)))
    pred      = np.polyval(coeffs, a_arr)
    resid     = (e_arr - pred) / np.where(de_arr > 0, de_arr, 1.0)
    ndof      = max(len(a_arr) - 3, 1)
    chi2r     = float(np.sum(resid**2) / ndof)
    return dict(coeffs=coeffs, alpha_opt=float(alpha_opt),
                a=float(a), sigma_a=sigma_a, chi2r=chi2r)


def load_scan_data(runs_dir, rs, nelec, qidx, vq_str):
    """Collect cumulative (alpha, E, dE) from all iter JSONs for one (q, vq)."""
    qx, qy, qz = qidx
    base = os.path.join(runs_dir, f'rs{rs}-n{nelec}',
                        f'qv{qx}_{qy}_{qz}-vq{vq_str}')
    # pick up all iter JSONs sorted by iteration
    pattern = os.path.join(base, 'ni-e125-alpha_iter*.json')
    files   = sorted(glob.glob(pattern),
                     key=lambda p: int(p.split('alpha_iter')[1].split('.json')[0]))
    if not files:
        return None

    # use the LAST iter json which has the cumulative grid
    with open(files[-1]) as f:
        d = json.load(f)

    # also load alpha_selected.json for the final status / alpha_opt
    sel_path = os.path.join(base, 'ni-e125-alpha_selected.json')
    sel = None
    if os.path.exists(sel_path):
        with open(sel_path) as f:
            sel = json.load(f)

    alphas_raw = d['alpha_grid']
    E_raw      = d['E_grid']
    dE_raw     = d['dE_grid']
    # mask out None values (VMC that failed to produce output)
    valid = [e is not None and de is not None for e, de in zip(E_raw, dE_raw)]
    alphas = np.array([a for a, v in zip(alphas_raw, valid) if v], dtype=float)
    E      = np.array([e for e, v in zip(E_raw,      valid) if v], dtype=float)
    dE     = np.array([d_ for d_, v in zip(dE_raw,   valid) if v], dtype=float)
    return dict(
        alphas=alphas[valid], E=E[valid], dE=dE[valid],
        vq=float(d['vq']),
        status=sel['status']  if sel else 'unknown',
        reason=sel.get('reason', '') if sel else '',
        alpha_opt_sel=sel.get('alpha_opt') if sel else None,
        snr=d.get('snr'),
        chi2r=d.get('chi2_red'),
        n_iter=len(files),
    )


# ── per-vq panel ─────────────────────────────────────────────────────────────

def plot_panel(ax, data, alpha_max_fit=0.50, rs=None, nelec=None, qidx=None):
    alphas, E, dE = data['alphas'], data['E'], data['dE']
    vq      = data['vq']
    status  = data['status']
    ok      = status == 'ok'
    color   = '#2ca02c' if ok else '#d62728'

    # ── data points ──
    ax.errorbar(alphas, E, yerr=dE, fmt='o', color=color,
                ms=4, lw=0.8, capsize=2, label='VMC data', zorder=3)

    a_lo, a_hi = alphas.min(), alphas.max()
    a_plot = np.linspace(max(0, a_lo - 0.05), min(1.6, a_hi + 0.05), 300)

    # ── full-range parabola fit ──
    fit_full = _fit_parabola(alphas, E, dE)
    if fit_full is not None:
        e_fit = np.polyval(fit_full['coeffs'], a_plot)
        ax.plot(a_plot, e_fit, '--', color='grey', lw=1.0,
                label=f"fit full (a/sigma={fit_full['a']/max(fit_full['sigma_a'],1e-30):.1f})")

    # ── restricted parabola fit (alpha <= alpha_max_fit) ──
    mask_r = alphas <= alpha_max_fit
    if mask_r.sum() >= 3:
        fit_r = _fit_parabola(alphas[mask_r], E[mask_r], dE[mask_r])
        if fit_r is not None:
            a_plot_r = np.linspace(max(0, alphas[mask_r].min()-0.05),
                                   min(alpha_max_fit+0.1, a_hi+0.05), 300)
            e_fit_r  = np.polyval(fit_r['coeffs'], a_plot_r)
            ax.plot(a_plot_r, e_fit_r, '-', color='steelblue', lw=1.5,
                    label=f"fit alpha≤{alpha_max_fit:.2f} "
                          f"(alpha*={fit_r['alpha_opt']:.3f}, "
                          f"a/sigma={fit_r['a']/max(fit_r['sigma_a'],1e-30):.1f})")
            # mark restricted-fit alpha_opt
            if 0 < fit_r['alpha_opt'] < 1.6:
                ax.axvline(fit_r['alpha_opt'], color='steelblue', lw=1.0,
                           ls=':', alpha=0.7)

    # ── mark raw argmin ──
    idx_min = np.argmin(E)
    ax.axvline(alphas[idx_min], color='orange', lw=1.0, ls='-.',
               label=f"argmin data = {alphas[idx_min]:.3f}")

    # ── mark selected alpha_opt (if ok) ──
    if data['alpha_opt_sel'] is not None:
        ax.axvline(data['alpha_opt_sel'], color=color, lw=1.5, ls='--',
                   label=f"alpha_opt(selected)={data['alpha_opt_sel']:.3f}")

    # ── shade restricted region ──
    ax.axvspan(a_lo, min(alpha_max_fit, a_hi), alpha=0.06, color='steelblue',
               label=f'parabolic regime (alpha≤{alpha_max_fit:.2f})')

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$E_\mathrm{VMC} / N_e$ (Ha)')
    snr_str  = f"{data['snr']:.0f}" if data['snr'] else 'N/A'
    chi2_str = f"{data['chi2r']:.0f}" if data['chi2r'] else 'N/A'
    title = (f"vq={vq:.5f}  status={status}  SNR={snr_str}  chi2={chi2_str}"
             f"  iter={data['n_iter']}")
    ax.set_title(title, fontsize=8, color=color)
    if data['reason']:
        ax.text(0.02, 0.04, f"[!] {data['reason']}", transform=ax.transAxes,
                fontsize=7, color='#d62728', va='bottom')
    ax.legend(fontsize=6, loc='upper right')


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs_dir',     default='runs')
    p.add_argument('--rs',           type=float, required=True)
    p.add_argument('--nelec',        type=int,   required=True)
    p.add_argument('--qidx',         nargs=3, type=int, default=[2, 2, 1],
                   metavar=('QX','QY','QZ'))
    p.add_argument('--alpha_max_fit',type=float, default=0.50,
                   help='Upper alpha cutoff for restricted parabola fit')
    p.add_argument('--outdir',       default=None,
                   help='Directory to save plots (default: runs_dir/plots)')
    args = p.parse_args()

    outdir = args.outdir or os.path.join(args.runs_dir, 'plots')
    os.makedirs(outdir, exist_ok=True)

    qx, qy, qz = args.qidx
    rs_str = '%.1f' % args.rs

    # ── discover all vq values for this (rs, nelec, q) ──
    base_glob = os.path.join(args.runs_dir,
                             f'rs{rs_str}-n{args.nelec}',
                             f'qv{qx}_{qy}_{qz}-vq*',
                             'ni-e125-alpha_iter0.json')
    iter0_files = sorted(glob.glob(base_glob))
    if not iter0_files:
        print(f'No alpha_iter0.json found under {base_glob}')
        sys.exit(1)

    vq_strs = []
    for fp in iter0_files:
        # extract vq string from path segment qv...-vq<VQ>
        seg = [s for s in fp.split(os.sep) if s.startswith('qv')]
        if seg:
            vq_strs.append(seg[0].split('-vq')[1])

    # ── load scan data ──
    all_data = {}
    for vq_str in vq_strs:
        d = load_scan_data(args.runs_dir, rs_str, args.nelec,
                           args.qidx, vq_str)
        if d is not None and d['vq'] > 0:   # skip vq=0
            all_data[float(vq_str)] = d

    if not all_data:
        print('No non-zero vq scan data found.')
        sys.exit(1)

    vqs = sorted(all_data.keys())
    n   = len(vqs)

    # ── Figure 1: E(alpha) panels ──────────────────────────────────────────
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5*ncols, 4.2*nrows),
                             squeeze=False)
    for i, vq in enumerate(vqs):
        ax = axes[i // ncols][i % ncols]
        plot_panel(ax, all_data[vq], alpha_max_fit=args.alpha_max_fit,
                   rs=args.rs, nelec=args.nelec, qidx=args.qidx)

    # hide unused axes
    for j in range(n, nrows*ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(f'E(alpha) scan — rs={args.rs}, Ne={args.nelec}, '
                 f'q=[{qx},{qy},{qz}]', fontsize=11)
    fig.tight_layout()
    fname1 = os.path.join(outdir,
        f'alpha_scan_E_alpha_rs{args.rs}_n{args.nelec}_q{qx}{qy}{qz}.pdf')
    fig.savefig(fname1, format='pdf', bbox_inches='tight')
    print(f'Saved: {fname1}')
    plt.close(fig)

    # ── Figure 2: alpha_opt vs vq ───────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))

    ax_opt, ax_snr = axes2

    for vq in vqs:
        d = all_data[vq]
        ok = d['status'] == 'ok'
        c  = '#2ca02c' if ok else '#d62728'
        mk = 'o' if ok else 'X'

        # restricted-range alpha_opt (always computable)
        alphas, E, dE = d['alphas'], d['E'], d['dE']
        mask_r = alphas <= args.alpha_max_fit
        aopt_r = None
        if mask_r.sum() >= 3:
            fr = _fit_parabola(alphas[mask_r], E[mask_r], dE[mask_r])
            if fr and 0 < fr['alpha_opt'] < 1.6:
                aopt_r = fr['alpha_opt']

        # selected alpha_opt
        aopt_sel = d['alpha_opt_sel']
        # raw argmin
        aopt_raw = alphas[np.argmin(E)]

        ax_opt.plot(vq, aopt_raw, marker='v', color=c, ms=7, alpha=0.6,
                    label='argmin data' if vq == vqs[0] else '')
        if aopt_r is not None:
            ax_opt.plot(vq, aopt_r, marker='s', color='steelblue', ms=7,
                        alpha=0.8,
                        label=f'fit alpha≤{args.alpha_max_fit:.2f}' if vq == vqs[0] else '')
        if aopt_sel is not None:
            ax_opt.plot(vq, aopt_sel, marker=mk, color=c, ms=9, lw=1.5,
                        markeredgecolor='k', markeredgewidth=0.5,
                        label='selected (ok)' if ok and vq==vqs[0] else
                              'failed (no DMC)' if not ok and vq==vqs[0] else '')

        ax_snr.semilogy(vq, d['snr'] if d['snr'] else np.nan,
                        marker=mk, color=c, ms=8)

    ax_opt.set_xlabel('vq (Ha)')
    ax_opt.set_ylabel('alpha_opt')
    ax_opt.set_title(f'alpha_opt vs vq  (q=[{qx},{qy},{qz}])')
    ax_opt.legend(fontsize=7)
    ax_opt.axhline(0, color='grey', lw=0.5, ls='--')

    ax_snr.set_xlabel('vq (Ha)')
    ax_snr.set_ylabel('SNR  [E(alpha) signal/noise]')
    ax_snr.set_title('SNR vs vq')
    ax_snr.axhline(5, color='grey', ls='--', lw=0.8, label='SNR=5 gate')
    ax_snr.legend(fontsize=7)

    fig2.suptitle(f'Alpha scan summary — rs={args.rs}, Ne={args.nelec}, '
                  f'q=[{qx},{qy},{qz}]', fontsize=11)
    fig2.tight_layout()
    fname2 = os.path.join(outdir,
        f'alpha_scan_summary_rs{args.rs}_n{args.nelec}_q{qx}{qy}{qz}.png')
    fig2.savefig(fname2, dpi=150, bbox_inches='tight')
    print(f'Saved: {fname2}')
    plt.close(fig2)

    # ── Text summary ─────────────────────────────────────────────────────────
    print()
    print(f"{'vq':>10}  {'status':>8}  {'SNR':>8}  {'chi2r':>10}  "
          f"{'alpha_opt(sel)':>12}  {'alpha_argmin':>10}  reason")
    print('-' * 90)
    for vq in vqs:
        d = all_data[vq]
        aopt = f"{d['alpha_opt_sel']:.4f}" if d['alpha_opt_sel'] is not None else '  N/A  '
        argm = f"{d['alphas'][np.argmin(d['E'])]:.4f}"
        snr  = f"{d['snr']:.1f}"  if d['snr']  else '  N/A'
        chi2 = f"{d['chi2r']:.1f}" if d['chi2r'] else '  N/A'
        print(f"{vq:>10.5f}  {d['status']:>8}  {snr:>8}  {chi2:>10}  "
              f"{aopt:>12}  {argm:>10}  {d['reason'][:50]}")


if __name__ == '__main__':
    main()
