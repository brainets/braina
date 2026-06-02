---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Single-trial DFC → condition GCMI → exact permutations → group WfStats

This notebook walks through, step by step, a group analysis on **10 rodents**, two
experimental conditions (**PP** and **UP**, 4 trials each → 8 trials per rat):

1. **Dynamic Functional Connectivity (DFC)** between every pair of brain regions,
   single-trial, on sliding windows of `dt = 150` samples with `step = 1`.
2. **Mutual information (GCMI)** between the single-trial DFC and the discrete
   condition regressor `y` (0 = PP, 1 = UP), continuous–discrete flavour.
3. The **exact permutation null**: every distinct 4/4 relabelling of the 8 trials
   = C(8,4) = **70** permutations.
4. Pool true + permuted MI across rats and run the group statistics workflow
   **`WfStats`** (random-effects, cluster correction).

> **Interpretation note.** MI is estimated from only 8 trials, so per-window
> estimates are noisy and positively biased. The exact permutation test + RFX
> group statistics are what control for that — treat the per-rat MI as an effect
> size and read significance from `WfStats`.

<!-- #region -->
## Setup

If you run this outside the repo environment, install the dependency:

```bash
pip install "frites" "numpy<2.0" xarray netcdf4
```
<!-- #endregion -->

```python
import numpy as np
import xarray as xr
from itertools import combinations

from frites.conn import conn_dfc
from frites.estimator import GCMIEstimator
from frites.workflow import WfStats
```

## Step 0 — Configuration

The 8 trials per rat are ordered `[PP, PP, PP, PP, UP, UP, UP, UP]` so they line
up with the regressor `Y = [0,0,0,0, 1,1,1,1]`.

```python
RATS = [f"rat-{i:02d}" for i in range(1, 11)]   # 10 rodents
CONDITIONS = ["PP", "UP"]                        # PP -> y=0, UP -> y=1
N_TRIALS_PER_COND = 4                            # 4 trials / condition
DT = 150                                         # window length, in samples
STEP = 1                                         # window step, in samples

# regressor: 4 PP trials (0) then 4 UP trials (1)
Y = np.array([0] * N_TRIALS_PER_COND + [1] * N_TRIALS_PER_COND)
N_TRIALS = Y.size                                # 8

# continuous-discrete GCMI estimator (copula-normalised, bias-corrected)
ESTIMATOR = GCMIEstimator(mi_type="cd", copnorm=True, biascorrect=True,
                          verbose=False)

OUT_PREFIX = "dfc_mi_wfstats"                    # output file prefix
print(f"{len(RATS)} rats, {N_TRIALS} trials/rat, regressor Y = {Y}")
```

## Step 1 — Load one rat's data

**Replace `load_rat_data` with your real loader.** It must return `x` of shape
`(n_trials=8, n_roi, n_times)`, with trials ordered PP-then-UP, plus the region
names and the time vector. As shipped it simulates data so the notebook runs
end-to-end (an UP-only ROI0–ROI1 coupling is injected to create a real effect).

```python
def load_rat_data(rat):
    """Return (x, roi, times) for one rat.

    x     : (n_trials=8, n_roi, n_times), trials ordered [PP*4, UP*4]
    roi   : list of region names, length n_roi
    times : time vector, length n_times
    """
    # ---- simulated stand-in so the notebook runs out of the box ----------
    rng = np.random.RandomState(abs(hash(rat)) % (2 ** 32))
    n_roi, n_times = 4, 600
    roi = [f"ROI{j}" for j in range(n_roi)]
    times = np.arange(n_times) / 1000.0          # e.g. 1 kHz sampling

    x = rng.randn(N_TRIALS, n_roi, n_times)
    sl = slice(200, 450)
    x[4:, 1, sl] += 0.8 * x[4:, 0, sl]           # UP trials only -> drives MI
    return x, roi, times
    # ----------------------------------------------------------------------


# quick look at one rat
x0, roi0, times0 = load_rat_data(RATS[0])
print("x:", x0.shape, "| roi:", roi0, "| n_times:", times0.size)
```

## Step 2 — Sliding windows

`conn_dfc` indexes the time vector with `[start, stop]` **inclusive**, so a
`DT`-sample window is `[start, start + DT - 1]` (using `start + DT` throws an
out-of-bounds error).

```python
def build_windows(n_times):
    starts = np.arange(0, n_times - DT + 1, STEP)
    return np.c_[starts, starts + DT - 1]


win_sample = build_windows(times0.size)
print(f"{win_sample.shape[0]} windows of {DT} samples, step {STEP}")
print("first / last window:", win_sample[0], win_sample[-1])
```

## Step 3 — Single-trial DFC for one rat

`conn_dfc` returns an xarray with dims `(trials, roi, times)`, where `roi` holds
the region-**pair** names and `times` the window centres.

```python
dfc = conn_dfc(x0, win_sample=win_sample, times=times0, roi=roi0,
               n_jobs=1, verbose=False)
print("DFC dims:", dfc.dims, "shape:", dfc.shape)
print("pairs:", [str(r) for r in dfc["roi"].values])
dfc
```

## Step 4 — MI between DFC and the condition regressor

GCMI (continuous–discrete) between each pair × window DFC time series (across
the 8 trials) and the discrete labels. The estimator wants `x` shaped
`(n_var, n_mv, n_samples)`; here `n_var = pairs × windows`, `n_mv = 1`,
`n_samples = trials`.

```python
def dfc_to_mi(dfc_vals, labels):
    """GCMI(cd) -> (n_pairs, n_win) from DFC (n_trials, n_pairs, n_win)."""
    n_trials, n_pairs, n_win = dfc_vals.shape
    x = dfc_vals.transpose(1, 2, 0).reshape(n_pairs * n_win, 1, n_trials)
    mi = ESTIMATOR.estimate(x, labels.astype(float))   # (1, n_pairs*n_win)
    return mi.reshape(n_pairs, n_win)


mi0 = dfc_to_mi(dfc.values, Y)
print("true MI shape (pairs, windows):", mi0.shape)
```

## Step 5 — The exact permutation set

Every distinct relabelling of the 8 trials into a 4/4 split. Because labels are
exchangeable within a condition, this is C(8,4) = **70** (not 8!).

```python
def exact_permutations():
    perms = []
    for ones in combinations(range(N_TRIALS), N_TRIALS_PER_COND):
        lab = np.zeros(N_TRIALS, dtype=int)
        lab[list(ones)] = 1
        perms.append(lab)
    return np.asarray(perms)


perms_labels = exact_permutations()
n_perm = perms_labels.shape[0]
print(f"{n_perm} permutations, e.g. {perms_labels[0]} ... {perms_labels[-1]}")
```

## Step 6 — Loop over rats: accumulate effect + permutations

For every rat: DFC → true MI → 70 permuted MIs. We collect per-rat arrays and
capture the pair names / window times once.

```python
effect_per_rat = []      # each: (n_pairs, n_win)
perms_per_rat = []       # each: (n_perm, n_pairs, n_win)
pair_names = win_times = None

for rat in RATS:
    x, roi, times = load_rat_data(rat)
    assert x.shape[0] == N_TRIALS, f"{rat}: expected {N_TRIALS} trials"

    ws = build_windows(x.shape[-1])
    dfc = conn_dfc(x, win_sample=ws, times=times, roi=roi,
                   n_jobs=1, verbose=False)
    dfc_vals = dfc.values                                  # (8, n_pairs, n_win)

    effect = dfc_to_mi(dfc_vals, Y)                        # (n_pairs, n_win)
    perms = np.stack([dfc_to_mi(dfc_vals, perms_labels[p])
                      for p in range(n_perm)], axis=0)     # (70, n_pairs, n_win)

    effect_per_rat.append(effect)
    perms_per_rat.append(perms)

    if pair_names is None:
        pair_names = [str(r) for r in dfc["roi"].values]
        win_times = dfc["times"].values
    print(f"  {rat}: DFC {dfc_vals.shape} -> MI {effect.shape}")
```

## Step 7 — Pool across rats and shape inputs for `WfStats`

`WfStats.fit` wants a **list with one entry per pair**:
`effect[pair]` of shape `(n_subjects, n_times)` and
`perms[pair]` of shape `(n_perm, n_subjects, n_times)`.

```python
eff = np.stack(effect_per_rat, axis=1)        # (n_pairs, n_subjects, n_win)
prm = np.stack(perms_per_rat, axis=2)         # (n_perm, n_pairs, n_subjects, n_win)
n_pairs = eff.shape[0]
print("pooled effect:", eff.shape, "| pooled perms:", prm.shape)

effect_list = [eff[p] for p in range(n_pairs)]
perms_list = [prm[:, p] for p in range(n_pairs)]
```

## Step 8 — Group statistics with `WfStats`

Random-effects inference with cluster-based multiple-comparison correction.
Returns `(n_windows, n_pairs)` p- and t-values.

```python
wf = WfStats(verbose=False)
pvalues, tvalues = wf.fit(effect_list, perms_list, inference="rfx",
                          mcp="cluster", tail=1)
pvalues, tvalues = np.asarray(pvalues), np.asarray(tvalues)
print("pvalues:", pvalues.shape, "| tvalues:", tvalues.shape)
print("significant (p<0.05) pair x window points:", int((pvalues < 0.05).sum()))
```

## Step 9 — Save labelled results

```python
coords = {"times": win_times, "roi": pair_names}
dims = ("times", "roi")

xr.DataArray(eff.mean(1).T, coords=coords, dims=dims,
             name="mi").to_netcdf(f"{OUT_PREFIX}_mi_mean.nc")
xr.DataArray(pvalues, coords=coords, dims=dims,
             name="pvalues").to_netcdf(f"{OUT_PREFIX}_pvalues.nc")
xr.DataArray(tvalues, coords=coords, dims=dims,
             name="tvalues").to_netcdf(f"{OUT_PREFIX}_tvalues.nc")
print(f"Saved {OUT_PREFIX}_{{mi_mean,pvalues,tvalues}}.nc")
```

## Step 10 — Quick visualisation (optional)

Mean MI across rats and the significance mask for each region pair.

```python
import matplotlib.pyplot as plt

mi_mean = eff.mean(1)                     # (n_pairs, n_win)
fig, ax = plt.subplots(figsize=(9, 4))
for p, name in enumerate(pair_names):
    ax.plot(win_times, mi_mean[p], label=name)
    sig = pvalues[:, p] < 0.05
    ax.fill_between(win_times, 0, mi_mean[p], where=sig, alpha=0.2)
ax.set(xlabel="time (window centre)", ylabel="MI (bits)",
       title="Mean DFC↔condition MI across rats (shaded = p<0.05)")
ax.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()
```
