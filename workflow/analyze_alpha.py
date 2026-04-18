#!/usr/bin/env python3
"""
Analyze VMC optimization results for different alpha values.
Fit E(alpha) to find optimal alpha that minimizes energy.

Usage:
  python analyze_alpha.py --rs 20.0 --nelec 162 --qidx 1 0 0 --vq 0.001
  python analyze_alpha.py --all  # analyze all available data
"""

import numpy as np
import argparse
import os
import glob
from scipy.optimize import minimize_scalar
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

def parse_opt_out(fopt_out):
    """
    Parse VMC optimization output to get final energy.
    Returns (energy, error) or (None, None) if parsing fails.
    """
    try:
        with open(fopt_out, 'r') as f:
            lines = f.readlines()
        
        # Look for final optimized energy
        # This parsing may need adjustment based on actual QMCPACK output format
        for line in reversed(lines):
            if 'LocalEnergy' in line and '=' in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if 'LocalEnergy' in p and i+2 < len(parts):
                        energy = float(parts[i+2])
                        error = float(parts[i+4]) if i+4 < len(parts) else 0.0
                        return energy, error
        
        # Alternative parsing - look for energy in scalar.dat files
        scalar_file = fopt_out.replace('.out', '.s001.scalar.dat')
        if os.path.exists(scalar_file):
            data = np.loadtxt(scalar_file)
            if len(data) > 0:
                # Average over last portion of data
                n_skip = len(data) // 2
                energy = np.mean(data[n_skip:, 1])  # Assuming column 1 is LocalEnergy
                error = np.std(data[n_skip:, 1]) / np.sqrt(len(data) - n_skip)
                return energy, error
                
    except Exception as e:
        print(f"  Warning: Could not parse {fopt_out}: {e}")
    
    return None, None


def collect_alpha_data(rs, nelec, qidx, vq, base_path='runs'):
    """
    Collect E(alpha) data for a given (rs, nelec, qidx, vq) combination.
    """
    qx, qy, qz = qidx
    pattern = f'{base_path}/rs{rs:.1f}-n{nelec}/qv{qx}_{qy}_{qz}-vq{vq:.4f}/ni-e125-qa*-thr1.0d-10/opt-sj/opt.out'
    
    files = glob.glob(pattern)
    
    data = []
    for f in files:
        # Extract alpha from path
        path_parts = f.split('/')
        for part in path_parts:
            if part.startswith('ni-e125-qa'):
                alpha_str = part.split('-qa')[1].split('-thr')[0]
                alpha = float(alpha_str)
                break
        else:
            continue
        
        energy, error = parse_opt_out(f)
        if energy is not None:
            data.append((alpha, energy, error))
    
    if not data:
        return None
    
    data = np.array(sorted(data, key=lambda x: x[0]))
    return data


def fit_parabola(alpha, energy, weights=None):
    """
    Fit a parabola E = a*(alpha - alpha_opt)^2 + E_min around its global minimum
    Returns (alpha_opt, E_min, coefficients)
    """
    if weights is None:
        weights = np.ones_like(alpha)
    
    global_min_idx = np.argmin(energy)
    fit_range = slice(0, 3*global_min_idx+1)
    
    # Fit E = a*alpha^2 + b*alpha + c around the global minimum
    coeffs = np.polyfit(alpha[fit_range], energy[fit_range], 2, w=weights[fit_range])
    a, b, c = coeffs
    
    if a <= 0:
        print("  Warning: Parabola fit has a <= 0, minimum may not be valid")
        return None, None, coeffs
    
    alpha_opt = -b / (2*a)
    E_min = c - b**2 / (4*a)
    
    return alpha_opt, E_min, coeffs


def fit_spline(alpha, energy, errors=None):
    """
    Fit a smoothing spline and find minimum.
    """
    if errors is not None:
        weights = 1.0 / (errors + 1e-10)
    else:
        weights = None
    
    spline = UnivariateSpline(alpha, energy, w=weights, s=len(alpha)*0.1)
    
    # Find minimum
    result = minimize_scalar(spline, bounds=(alpha.min(), alpha.max()), method='bounded')
    alpha_opt = result.x
    E_min = spline(alpha_opt)
    
    return alpha_opt, E_min, spline


def plot_alpha_scan(data, fit_results, title, save_path=None):
    """
    Plot E(alpha) with fit.
    """
    alpha, energy, errors = data[:, 0], data[:, 1], data[:, 2]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Data points
    ax.errorbar(alpha, energy, yerr=errors, fmt='o', capsize=3, label='VMC data')
    
    # Parabolic fit
    if fit_results['parabola']['alpha_opt'] is not None:
        alpha_fine = np.linspace(alpha.min(), alpha.max(), 100)
        coeffs = fit_results['parabola']['coeffs']
        E_fit = np.polyval(coeffs, alpha_fine)
        ax.plot(alpha_fine, E_fit, '--', label=f"Parabola: α_opt={fit_results['parabola']['alpha_opt']:.4f}")
        ax.axvline(fit_results['parabola']['alpha_opt'], color='r', linestyle=':', alpha=0.5)
    
    # Spline fit
    if fit_results['spline']['spline'] is not None:
        alpha_fine = np.linspace(alpha.min(), alpha.max(), 100)
        E_spline = fit_results['spline']['spline'](alpha_fine)
        ax.plot(alpha_fine, E_spline, '-', label=f"Spline: α_opt={fit_results['spline']['alpha_opt']:.4f}")
    
    ax.set_xlabel('α (trial wavefunction parameter)')
    ax.set_ylabel('E (Ha)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved plot to {save_path}")
    
    plt.close()
    return fig


def analyze_single(rs, nelec, qidx, vq, base_path='runs', plot=True, output_dir='alpha_analysis'):
    """
    Analyze alpha optimization for a single (rs, nelec, qidx, vq) point.
    """
    print(f"\nAnalyzing rs={rs}, N={nelec}, q=({qidx[0]},{qidx[1]},{qidx[2]}), vq={vq}")
    
    data = collect_alpha_data(rs, nelec, qidx, vq, base_path)
    
    if data is None or len(data) < 3:
        print("  Insufficient data points for fitting")
        return None
    
    alpha, energy, errors = data[:, 0], data[:, 1], data[:, 2]
    print(f"  Found {len(data)} alpha values: {alpha}")
    
    # Fit parabola
    weights = 1.0 / (errors + 1e-10)
    alpha_opt_para, E_min_para, coeffs = fit_parabola(alpha, energy, weights)
    
    # Fit spline
    alpha_opt_spline, E_min_spline, spline = fit_spline(alpha, energy, errors)
    
    fit_results = {
        'parabola': {'alpha_opt': alpha_opt_para, 'E_min': E_min_para, 'coeffs': coeffs},
        'spline': {'alpha_opt': alpha_opt_spline, 'E_min': E_min_spline, 'spline': spline}
    }
    
    if alpha_opt_para is not None:
        print(f"  Parabola fit: α_opt = {alpha_opt_para:.4f}, E_min = {E_min_para:.6f}")
    print(f"  Spline fit:   α_opt = {alpha_opt_spline:.4f}, E_min = {E_min_spline:.6f}")
    
    if plot:
        os.makedirs(output_dir, exist_ok=True)
        title = f'E(α) for rs={rs}, N={nelec}, q=({qidx[0]},{qidx[1]},{qidx[2]}), vq={vq}'
        save_path = os.path.join(output_dir, f'alpha_rs{rs}_q{qidx[0]}{qidx[1]}{qidx[2]}_vq{vq:.4f}.png')
        plot_alpha_scan(data, fit_results, title, save_path)
    
    return {
        'rs': rs, 'nelec': nelec, 'qidx': qidx, 'vq': vq,
        'data': data, 'fit_results': fit_results
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze alpha optimization results')
    parser.add_argument('--rs', type=float, default=20.0, help='rs value')
    parser.add_argument('--nelec', type=int, default=162, help='Number of electrons')
    parser.add_argument('--qidx', type=int, nargs=3, default=[1, 0, 0], help='q-vector indices')
    parser.add_argument('--vq', type=float, default=0.001, help='vq value')
    parser.add_argument('--base_path', type=str, default='runs', help='Base path for runs')
    parser.add_argument('--output_dir', type=str, default='alpha_analysis', help='Output directory')
    parser.add_argument('--all', action='store_true', help='Analyze all available data')
    
    args = parser.parse_args()
    
    if args.all:
        # Find all unique combinations
        pattern = f'{args.base_path}/rs*-n*/qv*-vq*/ni-e125-qa*-thr1.0d-10/opt-sj/opt.out'
        files = glob.glob(pattern)
        
        # Extract unique (rs, nelec, qidx, vq) combinations
        combinations = set()
        for f in files:
            parts = f.split('/')
            for p in parts:
                if p.startswith('rs'):
                    rs_nelec = p.split('-n')
                    rs = float(rs_nelec[0][2:])
                    nelec = int(rs_nelec[1])
                elif p.startswith('qv'):
                    qv_vq = p.split('-vq')
                    qidx = tuple(map(int, qv_vq[0][2:].split('_')))
                    vq = float(qv_vq[1])
            combinations.add((rs, nelec, qidx, vq))
        
        results = []
        for rs, nelec, qidx, vq in sorted(combinations):
            result = analyze_single(rs, nelec, list(qidx), vq, args.base_path, 
                                   plot=True, output_dir=args.output_dir)
            if result:
                results.append(result)
        
        # Save summary
        if results:
            summary_file = os.path.join(args.output_dir, 'alpha_opt_summary.txt')
            with open(summary_file, 'w') as f:
                f.write("# rs  nelec  qx qy qz  vq  alpha_opt_para  alpha_opt_spline\n")
                for r in results:
                    qx, qy, qz = r['qidx']
                    alpha_para = r['fit_results']['parabola']['alpha_opt']
                    alpha_spline = r['fit_results']['spline']['alpha_opt']
                    alpha_para_str = f"{alpha_para:.4f}" if alpha_para else "N/A"
                    f.write(f"{r['rs']:.1f}  {r['nelec']}  {qx} {qy} {qz}  {r['vq']:.4f}  "
                           f"{alpha_para_str}  {alpha_spline:.4f}\n")
            print(f"\nSummary saved to {summary_file}")
    else:
        analyze_single(args.rs, args.nelec, args.qidx, args.vq, 
                      args.base_path, plot=True, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
