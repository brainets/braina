# `frites_conn_pid` and `frites_conn_ii` - what a pair tells about a target y

Both take the data `(n_epochs, n_roi, n_times)` and a per-trial variable `y`
(stimulus, choice, model variable...). Both use the shared GCMI estimator
and are computed across trials, per time point, for every ROI pair. Output
dims `(roi, times)` with `roi` = pairs `A-B`.

## `frites_conn_pid` - partial information decomposition

Decomposes `I([X, Y]; y)` into four non-negative terms, written to four
files: `_infotot`, `_unique` (per node), `_redundancy` (both carry the same
information about y) and `_synergy` (only available jointly).

- `mi_type` - `'cc'` for continuous y, `'cd'` for categorical y.
- `dt` - number of successive time points pooled into each estimate (1 =
  none; >1 smooths and stabilises at the cost of time resolution).

Reference: Williams & Beer 2010 (arXiv:1004.2515).
API: https://brainets.github.io/frites/api/generated/frites.conn.conn_pid.html

## `frites_conn_ii` - interaction information

The single-number relative: `II = I([X,Y]; y) - I(X; y) - I(Y; y)`.
**Positive = synergy-dominated, negative = redundancy-dominated.** Cheaper
than PID when only the net balance matters. Same `mi_type` and `dt`.

Reference: McGill 1954; application to learning in Combrisson et al. 2024,
eLife - PDF:
https://github.com/brainets/braina/blob/main/papers/Combrisson_elife_2024_interaction_information_learning.pdf
API: https://brainets.github.io/frites/api/generated/frites.conn.conn_ii.html

## Statistics

Neither returns p-values. Build a null by shuffling `y` across trials, rerun,
stack `(n_perm, ...)` and use `frites_wf_stats`; or, for the plain
"does region X encode y" question, use `frites_wf_mi` which has the
permutation test built in.
