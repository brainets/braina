# Ranking multiplets, bootstrap CIs, and choosing the estimator

## `hoi_get_nbest_mult` - which multiplets matter, by name

Reads the `.nc` output of any `hoi_*` tool (coordinates `multiplets`,
`order`, `multiplet_names`) and calls `hoi.utils.get_nbest_mult` with
`orders=`, `multiplets=`, `names=`. CSV columns: `index, order, hoi,
multiplet, names`.

- Returns the **`n_best` most positive and the `n_best` most negative**
  values (positive first, up to `2 * n_best` rows). Not an absolute-value
  ranking. Map the sign to redundancy/synergy with the table in `SKILL.md`.
- `minsize` / `maxsize` filter by order first (e.g. `minsize=3` to skip
  pairs, whose O-information is always 0).
- Needs `.nc` input; `.npy` outputs lose the coordinates and are refused.
- Names come from the feature coordinate of the HOI input `.nc`; with
  `.npy` input the `multiplet` column still lists feature indices in input
  column order. With a target `y`, index `n_features` is labelled `y`.

API: https://brainets.github.io/hoi/api/generated/hoi.utils.get_nbest_mult.html

## Bootstrap confidence intervals (`n_boots`, `ci_percentiles`, `random_state`)

Mirrors HOI's own example (`examples/hoi/statistics/plot_bootstrapping.py`):
resample the samples with replacement `n_boots` times, refit with
`fit(samples=...)`, take percentiles (default `[5, 95]`). Written to
`<output>_ci.nc` with dims `(ci, multiplets, variables)`, `ci = low, high`;
the tool reports how many multiplets have a CI excluding 0.

This quantifies **uncertainty of the estimate**, not significance against
chance: a CI excluding 0 is suggestive, but the estimator is biased for
finite samples, so a permutation test (`frites_wf_stats`, see the
`workflows` skill) is still the right tool for "is this above chance".
Cost = `n_boots` full fits; start with 20-50.

## `samples_path`

A `.npy` of integer sample indices restricts the fit to those samples (one
condition, one bootstrap draw, a train split...). Combine with `n_boots` to
bootstrap within a condition.

## `method` - entropy estimator (`hoi/core/entropies.py`)

- `'gc'` (default) Gaussian copula: rank-normalise then Gaussian entropy.
  Robust to monotonic nonlinearities, fast, biased for small n; the same
  assumption family as Frites' GCMI.
- `'gauss'`: plain Gaussian, i.e. assumes the raw data are Gaussian.
- `'binning'`: histogram estimator; the data must be **discretised
  beforehand** (integers / few levels), otherwise results are meaningless.
- `'knn'`: Kozachenko-Leonenko k-nearest-neighbour estimator; non-parametric,
  slower, better for genuinely non-Gaussian continuous data.
- `'kernel'`: kernel density estimator; non-parametric, sensitive to
  bandwidth, slow for many samples.
Compare two estimators on the same data before trusting a sign near 0.
