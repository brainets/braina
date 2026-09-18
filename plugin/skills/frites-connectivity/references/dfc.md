# `frites_conn_dfc` - dynamic functional connectivity

Undirected, time-resolved coupling between ROI pairs, per trial
(`frites.conn.conn_dfc`). Output dims `(trials, roi, times)` with `roi` =
pairs `A-B` and `times` = one value per window (window centre).

## Parameters

- `win_sample` - `[[start, stop], ...]` in samples, **inclusive** bounds (a
  stop index equal to `n_times` is out of range; use `n_times - 1`). `None`
  = one window over the whole epoch. Equivalent to
  `frites.conn.define_windows` when scripting directly.
- `agg_ch` - when several channels belong to the same ROI name, `True`
  aggregates them into one multivariate estimate per ROI pair instead of
  computing every channel pair.
- `estimator` - dependency measure:
  - `'gcmi'` (default) Gaussian-copula mutual information;
  - `'pearson'`, `'spearman'` - linear / rank correlation (signed);
  - `'dcorr'` - distance correlation (captures some nonlinear dependence);
  - `'binmi'` - binning-based MI (few bins; for discrete-ish data).
- `n_jobs` - parallel pairs.

## Interpretation

- GCMI values are in bits and non-negative; correlation estimators are
  signed in [-1, 1]. Say which one was used.
- Shorter windows = better time resolution, noisier estimate; typical LFP
  windows are 50-200 ms.
- For group statistics on the trial-averaged DFC use `frites_wf_stats`; for
  instantaneous comodulation with built-in statistics use
  `frites_wf_conn_comod` instead of `dfc` + stats.

API: https://brainets.github.io/frites/api/generated/frites.conn.conn_dfc.html
