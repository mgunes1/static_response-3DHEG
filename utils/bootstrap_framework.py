"""
Generic bootstrap error estimation for curve_fit models.
Supports arbitrary number of parameters and forms.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import Callable, Dict, Tuple, Any, Optional, List


class BootstrapFitter:
    """
    Generic bootstrap parameter estimation.
    
    Works with any model function and any number of parameters.
    """
    
    def __init__(
        self,
        model_func: Callable,
        param_names: List[str],
        bounds: Tuple[List[float], List[float]],
        maxfev: int = 5000
    ):
        """
        Initialize bootstrap fitter.
        
        Args:
            model_func: Function like f(x, *params, **kwargs) -> y
                       Must support being called as model_func(x, p1, p2, ..., **context)
            param_names: Names of parameters to fit (e.g., ['B', 'n'])
            bounds: (lower_bounds, upper_bounds) lists for all parameters
            maxfev: Max evaluations per fit
        """
        self.model = model_func
        self.param_names = param_names
        self.bounds = bounds
        self.maxfev = maxfev
        self.n_params = len(param_names)
        
        if len(bounds[0]) != self.n_params or len(bounds[1]) != self.n_params:
            raise ValueError(f"bounds must have {self.n_params} entries, got {len(bounds[0])}, {len(bounds[1])}")
    
    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dy: np.ndarray,
        p0: List[float],
        context: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single fit to data.
        
        Args:
            x: Independent variable (e.g., q)
            y: Dependent variable (e.g., G)
            dy: Uncertainties on y
            p0: Initial guess for parameters
            context: Dict of kwargs to pass to model (e.g., {'rs': 5.0})
        
        Returns:
            popt: Best-fit parameters
            pcov: Covariance matrix
        """
        context = context or {}
        
        # Wrapper that passes context as kwargs
        def wrapped_model(x, *params):
            return self.model(x, *params, **context)
        
        popt, pcov = curve_fit(
            wrapped_model,
            x, y,
            p0=p0,
            sigma=dy,
            absolute_sigma=True,
            bounds=self.bounds,
            maxfev=self.maxfev
        )
        return popt, pcov
    
    def bootstrap(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dy: np.ndarray,
        popt: np.ndarray,
        context: Dict[str, Any] = None,
        n_bootstrap: int = 1000,
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Bootstrap errors on all parameters.
        
        Resamples (x, y, dy) jointly with replacement, refits, collects parameter distributions.
        
        Args:
            x, y, dy: Data
            popt: Best-fit parameters (used as p0 for each bootstrap fit)
            context: Dict of kwargs for model
            n_bootstrap: Number of bootstrap samples
            verbose: Print warnings about failed fits
        
        Returns:
            Dict with keys:
                '{param_name}': array of parameter values across bootstrap samples
                'n_success': number of successful fits
                'n_failed': number of failed fits
        """
        context = context or {}
        n_data = len(x)
        
        # Initialize storage
        boot_samples = {name: [] for name in self.param_names}
        n_failed = 0
        
        for _ in range(n_bootstrap):
            # Resample data points jointly
            idx = np.random.choice(n_data, size=n_data, replace=True)
            x_boot = x[idx]
            y_boot = y[idx]
            dy_boot = dy[idx]
            
            try:
                popt_boot, _ = self.fit(x_boot, y_boot, dy_boot, popt, context)
                
                # Check validity (prevent inf/nan)
                if not np.all(np.isfinite(popt_boot)):
                    n_failed += 1
                    continue
                
                # Store
                for i, name in enumerate(self.param_names):
                    boot_samples[name].append(popt_boot[i])
                    
            except (RuntimeError, ValueError):
                n_failed += 1
        
        # Convert to arrays
        for name in self.param_names:
            boot_samples[name] = np.array(boot_samples[name])
        
        n_success = len(boot_samples[self.param_names[0]])
        boot_samples['n_success'] = n_success
        boot_samples['n_failed'] = n_failed
        
        if n_success == 0:
            raise RuntimeError(f"All {n_bootstrap} bootstrap fits failed")
        
        if verbose and n_failed > 0.05 * n_bootstrap:
            print(f"  ⚠️  {n_failed}/{n_bootstrap} fits failed ({100*n_failed/n_bootstrap:.1f}%) "
                  f"— kept {n_success} successful")
        
        return boot_samples
    
    def error_summary(
        self,
        boot_samples: Dict[str, np.ndarray],
        method: str = 'std'
    ) -> Dict[str, float]:
        """
        Compute parameter errors from bootstrap distribution.
        
        Args:
            boot_samples: Output from bootstrap()
            method: 'std' (standard deviation) or 'percentile' (16th-84th percentile)
        
        Returns:
            Dict mapping param_name -> error_estimate
        """
        errors = {}
        
        for name in self.param_names:
            samples = boot_samples[name]
            
            if method == 'std':
                errors[name] = np.std(samples)
            elif method == 'percentile':
                p16, p84 = np.percentile(samples, [16, 84])
                errors[name] = (p84 - p16) / 2
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return errors


class FitResults:
    """Container for fit results with bootstrap errors."""
    
    def __init__(self):
        self.params = {}      # rs -> {param_name -> value}
        self.errors = {}      # rs -> {param_name -> error}
        self.covariance = {}  # rs -> covariance matrix
        self.chi2r = {}       # rs -> reduced chi-squared
        self.boot_samples = {}  # rs -> full bootstrap distributions (optional)
    
    def add(
        self,
        key: str,
        popt: np.ndarray,
        pcov: np.ndarray,
        param_names: List[str],
        chi2r: float,
        errors: Dict[str, float],
        boot_samples: Dict[str, np.ndarray] = None
    ):
        """Add fit results for one key (e.g., one rs value)."""
        self.params[key] = {name: popt[i] for i, name in enumerate(param_names)}
        self.errors[key] = errors
        self.covariance[key] = pcov
        self.chi2r[key] = chi2r
        if boot_samples is not None:
            self.boot_samples[key] = boot_samples
    
    def table(self, keys: List[str], param_names: List[str]) -> str:
        """Pretty-print results as a table."""
        lines = []
        
        # Header
        header = f"{'key':>6}"
        for pname in param_names:
            header += f"  {pname:>12}  d{pname:>10}"
        header += f"  {'χ²/dof':>7}"
        lines.append(header)
        lines.append("=" * len(header))
        
        # Rows
        for key in keys:
            row = f"{key:>6.1f}" if isinstance(key, (int, float)) else f"{str(key):>6}"
            for pname in param_names:
                val = self.params[key][pname]
                err = self.errors[key][pname]
                row += f"  {val:>12.6f}  {err:>10.3e}"
            row += f"  {self.chi2r[key]:>7.2f}"
            lines.append(row)
        
        return "\n".join(lines)


# ============================================================================
# Example usage with the original Moroni form
# ============================================================================

def fit_moroni_all_rs(
    data_G: Dict[float, Dict],
    rs_list: List[float],
    PARAMS_MORONI: Dict,
    PARAMS_N: Dict,
    form_Moroni_allfree: Callable,
    B_model: Callable,
    get_n: Callable,
    n_bootstrap: int = 1000,
    error_method: str = 'std'
) -> FitResults:
    """
    Fit Moroni form across all rs values with bootstrap errors.
    
    Args:
        data_G: Dict[rs] -> {'qlist': ..., 'G': ..., 'dG': ...}
        rs_list: List of rs values to fit
        PARAMS_MORONI, PARAMS_N: Parameter dicts
        form_Moroni_allfree: Model function f(q, B, n, rs=...)
        B_model, get_n: Functions to compute initial guesses
        n_bootstrap: Number of bootstrap iterations
        error_method: 'std' or 'percentile'
    
    Returns:
        FitResults object with all results
    """
    
    # Initialize fitter
    fitter = BootstrapFitter(
        model_func=form_Moroni_allfree,
        param_names=['B', 'n'],
        bounds=([0.1, 0.1], [3.0, 10.0]),
        maxfev=5000
    )
    
    results = FitResults()
    
    for rs in rs_list:
        print(f"\nrs = {rs:.1f}")
        
        # Get data
        d = data_G[rs]
        qlist, G, dG = d['qlist'], d['G'], d['dG']
        nq = len(qlist)
        
        # Initial guesses
        B_pub = B_model(rs, *PARAMS_MORONI.values())
        n_0 = get_n(rs, *PARAMS_N)
        p0 = [B_pub, n_0]
        
        print(f"  Initial guess: B={B_pub:.6f}, n={n_0:.4f}")
        
        # Main fit
        try:
            popt, pcov = fitter.fit(
                qlist, G, dG, p0,
                context={'rs': rs}
            )
        except Exception as e:
            print(f"  ❌ Fit failed: {e}")
            continue
        
        B_val, n_val = popt
        
        # Bootstrap errors
        boot_samples = fitter.bootstrap(
            qlist, G, dG, popt,
            context={'rs': rs},
            n_bootstrap=n_bootstrap,
            verbose=True
        )
        
        errors = fitter.error_summary(boot_samples, method=error_method)
        
        # Chi-squared
        model_vals = form_Moroni_allfree(qlist, B_val, n_val, rs=rs)
        resid = (model_vals - G) / dG
        chi2r = np.sum(resid**2) / (nq - 2)
        
        # Store
        results.add(
            key=rs,
            popt=popt,
            pcov=pcov,
            param_names=['B', 'n'],
            chi2r=chi2r,
            errors=errors,
            boot_samples=boot_samples
        )
        
        print(f"  ✓ B = {B_val:.6f} ± {errors['B']:.3e}")
        print(f"  ✓ n = {n_val:.4f} ± {errors['n']:.3e}")
        print(f"  ✓ χ²/dof = {chi2r:.2f}")
    
    return results


# ============================================================================
# Usage with a different form (arbitrary parameters)
# ============================================================================

def fit_custom_form(
    data_x: np.ndarray,
    data_y: np.ndarray,
    data_dy: np.ndarray,
    custom_model: Callable,
    param_names: List[str],
    p0: List[float],
    bounds: Tuple[List[float], List[float]],
    context: Dict[str, Any] = None,
    n_bootstrap: int = 1000
) -> Tuple[FitResults, Dict]:
    """
    One-off fit to custom model with bootstrap errors.
    
    Example:
        def my_model(x, a, b, c, scale=1.0):
            return a * np.exp(-b * x) + c * scale
        
        results, boot = fit_custom_form(
            x_data, y_data, dy_data,
            my_model,
            param_names=['a', 'b', 'c'],
            p0=[1.0, 0.1, 0.5],
            bounds=([0, 0, 0], [10, 1, 10]),
            context={'scale': 2.0}
        )
    """
    fitter = BootstrapFitter(
        model_func=custom_model,
        param_names=param_names,
        bounds=bounds
    )
    
    context = context or {}
    
    # Main fit
    popt, pcov = fitter.fit(data_x, data_y, data_dy, p0, context)
    
    # Bootstrap
    boot_samples = fitter.bootstrap(
        data_x, data_y, data_dy, popt,
        context=context,
        n_bootstrap=n_bootstrap
    )
    
    errors = fitter.error_summary(boot_samples)
    
    # Package results
    results = FitResults()
    results.add('fit', popt, pcov, param_names, chi2r=0, errors=errors)
    
    return results, boot_samples


# ============================================================================
# Diagnostic: plot bootstrap distributions
# ============================================================================

def plot_bootstrap_distributions(
    boot_samples: Dict[str, np.ndarray],
    param_names: List[str],
    figsize: Tuple[int, int] = (12, 4)
):
    """
    Plot histograms of bootstrap parameter distributions.
    
    Helps diagnose if fit is robust or if distribution is multimodal/skewed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return
    
    fig, axes = plt.subplots(1, len(param_names), figsize=figsize)
    if len(param_names) == 1:
        axes = [axes]
    
    for i, name in enumerate(param_names):
        samples = boot_samples[name]
        axes[i].hist(samples, bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_xlabel(name)
        axes[i].set_ylabel('Count')
        axes[i].axvline(np.median(samples), color='r', linestyle='--', label='Median')
        axes[i].legend()
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    # Example: fit a simple power law with bootstrap
    np.random.seed(42)
    x = np.linspace(1, 10, 50)
    y_true = 2.0 * x**(-0.5)
    y = y_true + np.random.normal(0, 0.1, size=len(x))
    dy = 0.1 * np.ones_like(y)
    
    def power_law(x, A, n):
        return A * x**n
    
    fitter = BootstrapFitter(
        power_law,
        param_names=['A', 'n'],
        bounds=([0.1, -2], [10, 0.5])
    )
    
    popt, pcov = fitter.fit(x, y, dy, p0=[2.0, -0.5])
    boot = fitter.bootstrap(x, y, dy, popt, n_bootstrap=1000)
    errors = fitter.error_summary(boot)
    
    print(f"A = {popt[0]:.4f} ± {errors['A']:.4f}")
    print(f"n = {popt[1]:.4f} ± {errors['n']:.4f}")
    print(f"Bootstrap successes: {boot['n_success']}/{boot['n_success'] + boot['n_failed']}")
