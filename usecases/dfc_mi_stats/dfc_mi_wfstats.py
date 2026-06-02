# /// script
# dependencies = ["frites", "numpy<2.0", "xarray", "netcdf4"]
# ///
"""
Single-trial DFC -> condition GCMI -> exact permutations -> group WfStats
========================================================================

Pipeline (per rat, then pooled across the 10 rats):

1. Dynamic Functional Connectivity (`frites.conn.conn_dfc`) between every pair
   of brain regions, computed at the single-trial level on sliding windows of
   ``dt = 150`` samples with ``step = 1`` sample. The 8 trials per rat are the
   4 PP trials followed by the 4 UP trials.

2. Mutual information between the single-trial DFC (continuous "brain activity"
   for each pair / window) and the discrete condition regressor
   ``y = [0,0,0,0, 1,1,1,1]`` (0 = PP, 1 = UP), estimated with the Gaussian
   Copula MI (GCMI), continuous-discrete flavour (`mi_type='cd'`).

3. The *exact* permutation null: every distinct relabelling of the 8 trials
   into a 4/4 split. With a balanced 4/4 design that is C(8,4) = 70
   permutations (labels are exchangeable within a class, so 8! over-counts).

4. The true MI and the 70 permuted MIs are stored for every rat and then fed
   into the group-level statistics workflow `frites.workflow.WfStats`
   (random-effects, cluster-based correction).

NOTE on interpretation: MI here is estimated from only 8 trials, so per-window
estimates are noisy and positively biased. The exact permutation test + RFX
group statistics are exactly what control for that bias / chance level, which
is why the raw per-rat MI is treated only as an effect size and significance
comes from WfStats.

Replace `load_rat_data` with your real loader. As shipped it simulates data so
the script runs end-to-end: `uv run usecases/dfc_mi_stats/dfc_mi_wfstats.py`
"""
import numpy as np
import xarray as xr
from itertools import combinations

from frites.conn import conn_dfc
from frites.estimator import GCMIEstimator
from frites.workflow import WfStats

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
RATS = [f"rat-{i:02d}" for i in range(1, 11)]   # 10 rodents
CONDITIONS = ["PP", "UP"]                        # PP -> y=0, UP -> y=1
N_TRIALS_PER_COND = 4                            # 4 trials / condition
DT = 150                                         # window length, in samples
STEP = 1                                         # window step, in samples

# regressor: 4 PP trials (0) then 4 UP trials (1)
Y = np.array([0] * N_TRIALS_PER_COND + [1] * N_TRIALS_PER_COND)
N_TRIALS = Y.size                                # 8

# the continuous-discrete GCMI estimator (copula-normalised, bias-corrected)
ESTIMATOR = GCMIEstimator(mi_type="cd", copnorm=True, biascorrect=True,
                          verbose=False)

OUT_PREFIX = "dfc_mi_wfstats"                    # output file prefix


# ----------------------------------------------------------------------------
# Data loading  --  REPLACE THIS with your real data
# ----------------------------------------------------------------------------
def load_rat_data(rat):
    """Return one rat's data ready for conn_dfc.

    Returns
    -------
    x : np.ndarray, shape (n_trials=8, n_roi, n_times)
        Trials ordered as [PP, PP, PP, PP, UP, UP, UP, UP] so that they line up
        with the global ``Y`` regressor.
    roi : list of str        # brain region names, length n_roi
    times : np.ndarray       # time vector, length n_times
    """
    # ---- simulated stand-in so the script runs out of the box -------------
    rng = np.random.RandomState(abs(hash(rat)) % (2 ** 32))
    n_roi, n_times = 4, 600
    roi = [f"ROI{j}" for j in range(n_roi)]
    times = np.arange(n_times) / 1000.0          # e.g. 1 kHz sampling

    x = rng.randn(N_TRIALS, n_roi, n_times)
    # inject a condition difference in the ROI0-ROI1 coupling during a window
    sl = slice(200, 450)
    x[4:, 1, sl] += 0.8 * x[4:, 0, sl]           # UP trials only -> drives MI
    return x, roi, times
    # ------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def build_windows(n_times):
    """Sliding windows of length DT (samples), stepping by STEP.

    conn_dfc indexes the time vector with [start, stop] *inclusive*, so a
    DT-sample window is [start, start + DT - 1].
    """
    starts = np.arange(0, n_times - DT + 1, STEP)
    return np.c_[starts, starts + DT - 1]


def dfc_to_mi(dfc_vals, labels):
    """GCMI(cd) between single-trial DFC and a discrete label vector.

    Parameters
    ----------
    dfc_vals : (n_trials, n_pairs, n_win) array
    labels   : (n_trials,) discrete regressor

    Returns
    -------
    mi : (n_pairs, n_win) array
    """
    n_trials, n_pairs, n_win = dfc_vals.shape
    # estimator wants x as (n_var, n_mv, n_samples); here n_var = pairs*windows,
    # n_mv = 1 (univariate), n_samples = trials.
    x = dfc_vals.transpose(1, 2, 0).reshape(n_pairs * n_win, 1, n_trials)
    mi = ESTIMATOR.estimate(x, labels.astype(float))      # (1, n_pairs*n_win)
    return mi.reshape(n_pairs, n_win)


def exact_permutations():
    """All distinct 4/4 relabellings of the 8 trials -> C(8,4) = 70 vectors."""
    perms = []
    for ones in combinations(range(N_TRIALS), N_TRIALS_PER_COND):
        lab = np.zeros(N_TRIALS, dtype=int)
        lab[list(ones)] = 1
        perms.append(lab)
    return np.asarray(perms)                     # (70, 8)


# ----------------------------------------------------------------------------
# Main: loop over rats, accumulate effect + permutations
# ----------------------------------------------------------------------------
def main():
    perms_labels = exact_permutations()
    n_perm = perms_labels.shape[0]
    print(f"Exact permutation set: {n_perm} relabellings (C(8,4))")

    effect_per_rat = []      # each: (n_pairs, n_win)
    perms_per_rat = []       # each: (n_perm, n_pairs, n_win)
    pair_names = win_times = None

    for rat in RATS:
        x, roi, times = load_rat_data(rat)
        assert x.shape[0] == N_TRIALS, f"{rat}: expected {N_TRIALS} trials"

        win_sample = build_windows(x.shape[-1])

        # ---- 1. single-trial DFC: (trials, pairs, windows) ----------------
        dfc = conn_dfc(x, win_sample=win_sample, times=times, roi=roi,
                       n_jobs=1, verbose=False)
        dfc_vals = dfc.values                          # (8, n_pairs, n_win)

        # ---- 2. true MI between DFC and the condition regressor -----------
        effect = dfc_to_mi(dfc_vals, Y)                # (n_pairs, n_win)

        # ---- 3. permuted MI for all 70 relabellings -----------------------
        perms = np.stack([dfc_to_mi(dfc_vals, perms_labels[p])
                          for p in range(n_perm)], axis=0)  # (70, n_pairs, n_win)

        effect_per_rat.append(effect)
        perms_per_rat.append(perms)

        if pair_names is None:                         # capture coords once
            pair_names = [str(r) for r in dfc["roi"].values]
            win_times = dfc["times"].values
        print(f"  {rat}: DFC {dfc_vals.shape} -> MI {effect.shape}")

    # ---- 4. pool across rats -> WfStats list inputs -----------------------
    # effects:  (n_pairs, n_subjects, n_win)
    eff = np.stack(effect_per_rat, axis=1)
    # perms:    (n_perm, n_pairs, n_subjects, n_win)
    prm = np.stack(perms_per_rat, axis=2)
    n_pairs = eff.shape[0]
    print(f"\nPooled effect {eff.shape}, perms {prm.shape}")

    # WfStats wants a list (one entry per pair/ROI):
    #   effect[pair] : (n_subjects, n_times)
    #   perms[pair]  : (n_perm, n_subjects, n_times)
    effect_list = [eff[p] for p in range(n_pairs)]
    perms_list = [prm[:, p] for p in range(n_pairs)]

    wf = WfStats(verbose=False)
    pvalues, tvalues = wf.fit(effect_list, perms_list, inference="rfx",
                              mcp="cluster", tail=1)
    pvalues, tvalues = np.asarray(pvalues), np.asarray(tvalues)  # (n_win, n_pairs)
    print(f"WfStats done -> pvalues {pvalues.shape}, tvalues {tvalues.shape}")

    # ---- save everything as labelled xarrays ------------------------------
    coords = {"times": win_times, "roi": pair_names}
    dims = ("times", "roi")
    xr.DataArray(eff.mean(1).T, coords=coords, dims=dims,
                 name="mi").to_netcdf(f"{OUT_PREFIX}_mi_mean.nc")
    xr.DataArray(pvalues, coords=coords, dims=dims,
                 name="pvalues").to_netcdf(f"{OUT_PREFIX}_pvalues.nc")
    xr.DataArray(tvalues, coords=coords, dims=dims,
                 name="tvalues").to_netcdf(f"{OUT_PREFIX}_tvalues.nc")

    n_sig = int((pvalues < 0.05).sum())
    print(f"\nSaved {OUT_PREFIX}_{{mi_mean,pvalues,tvalues}}.nc")
    print(f"Significant (p<0.05) pair x window points: {n_sig}")


if __name__ == "__main__":
    main()
