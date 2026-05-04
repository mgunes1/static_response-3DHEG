# Plan: vq-range + alpha-scan grid revision (April 13 2026)

## Background

Empirical findings from rs=50, q=[2,2,1]:
- Old vql~1e-5 gave SNR~0.007 (undetectable); new [0.001,...,0.004] gives SNR~800,
  same chi(q). EF-scaling prescription was wrong.
- alpha_opt shifts upward with vq → centering grid on guess_alpha2 misses minimum
  at large vq. Need wide uniform grid.
- chi2_red gate (>5 → failed) incorrectly rejects fits with well-determined minima.
  High chi2_red reflects quartic deviations at large vq, not a bad minimum.
- rescan_widen is meaningless once grid is already wide and uniform.
- vq=0 produces trivial E(alpha): alpha×vq=0 always; exclude.

Resolved parameter choices:
- Alpha grid: 10 uniform points in [0.05, 1.50]; ALPHA_HI capped at 1.5
- vql: `np.arange(0.0005, 0.0025+1e-9, 0.0005)` = [0.0005, 0.001, 0.0015, 0.002, 0.0025]
  (user manages per-rs manually; this is the rs=50 default)
- chi2_red gate: removed entirely
- rescan_widen: removed entirely

## Files to edit

- `v5.0/workflow/alpha_scan.py`
- `v5.0/workflow/snakefile`

---

## Phase 1 — `alpha_scan.py` changes

**Files:** `v5.0/workflow/alpha_scan.py`

### 1a. Add `make_alpha_grid_uniform`

Add after the existing `make_alpha_grid` function:
```python
ALPHA_UNIFORM_LO = 0.05
ALPHA_UNIFORM_HI = 1.50
ALPHA_UNIFORM_N  = 10

def make_alpha_grid_uniform(n: int = ALPHA_UNIFORM_N,
                             lo: float = ALPHA_UNIFORM_LO,
                             hi: float = ALPHA_UNIFORM_HI) -> list:
    """Wide uniform alpha grid, no centering on heuristic."""
    raw = np.linspace(lo, hi, n)
    out = []; seen = set()
    for a in raw:
        s = ALPHA_FMT % float(np.clip(a, ALPHA_LO, ALPHA_HI))
        if s in seen: continue
        seen.add(s); out.append(s)
    return out
```

### 1b. Remove chi2_red gate from `select_alpha` (~lines 366–373)

Delete the block:
```python
    if fit.chi2_red > chi2_max:
        return AlphaSelection(
            **base, status='failed',
            reason=f'parabola fit poor: chi2_red={fit.chi2_red:.2f} > {chi2_max}',
        )
```
Keep `chi2_red` stored in JSON (diagnostic only).
Remove `CHI2_MAX_DEFAULT = 5.0` constant.
Remove `chi2_max` param from `select_alpha` signature (or keep but unused).

### 1c. Remove `rescan_widen` logic from `select_alpha` (~lines 321–338)

The SNR gate currently triggers `rescan_widen` when SNR < snr_min.
With a wide uniform grid SNR should always be adequate; if still low,
fail directly instead of widening.

Replace the rescan_widen branch with a direct fail:
```python
    if snr < snr_min:
        return AlphaSelection(
            **base, status='failed',
            reason=f'low SNR ({snr:.2f} < {snr_min}): parabola undetectable',
        )
```
Remove `next_alphas` from the SNR gate path.
Also remove handling of `status='rescan_widen'` in the checkpoint
(snakefile checkpoint logic that calls `select_alpha` with iteration>0).

### 1d. Update `_selftest` part 3

Change the selftest grid to use `make_alpha_grid_uniform` instead of
`make_alpha_grid(center=0.3, halfwidth=0.25)`, so it exercises the new function.

---

## Phase 2 — `snakefile` changes

**Files:** `v5.0/workflow/snakefile`

### 2a. Change vql definition (~line 100)

```python
# was: vql = np.arange(0,0.005,0.001)
vql = np.arange(0.0005, 0.0025 + 1e-9, 0.0005)  # [0.0005,...,0.0025], 5 pts
```

### 2b. Change `alpha_grid_for` to use uniform grid (~lines 115–127)

```python
SCAN_NPOINTS_INIT = 10   # was 5
# Remove: SCAN_HALFWIDTH_INIT

def alpha_grid_for(rs, nelec, qidx):
    key = (float(rs), int(nelec), tuple(qidx))
    if key not in _alpha_grid_cache:
        _alpha_grid_cache[key] = alpha_scan.make_alpha_grid_uniform(
            n=SCAN_NPOINTS_INIT
        )
    return _alpha_grid_cache[key]
```

Note: grid is now (rs,Ne,q)-independent, so `_alpha_grid_cache` is
effectively a constant (kept for API compatibility).

### 2c. Remove rescan_widen handling from checkpoint body

In `checkpoint alpha_select` run block: remove the `status=='rescan_widen'`
branch (only `rescan_shift` remains as a rescan path).

---

## Phase 3 — Verification

1. `snakemake -n` dry run: confirm vq=0 absent, 10 alpha points per (q,vq),
   grid is [0.050,...,1.500].
2. Re-run rs=50, q=[2,2,1] with new code.
3. Check all `alpha_selected.json`: status=ok, alpha_opt in [0.05,1.50],
   chi2_red present but not causing failures.
4. Compare extracted chi(q) to previous run — should agree within error bars.
