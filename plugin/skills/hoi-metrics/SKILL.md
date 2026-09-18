---
name: hoi-metrics
description: Use once a HOI (higher-order interactions) tool has been chosen (oinfo, gradient_oinfo, infotopo, redundancy_mmi, synergy_mmi, rsi, dtc, tc, sinfo, infotot, transfer_entropy, dotot, redundancy_phiid, atoms_phiid, get_nbest_mult) and the question is about how it actually works, what a parameter does, how to interpret its sign, which estimator or bootstrap settings to pick, or CPU/GPU. Grounded in the real HOI source (hoi/metrics, hoi/core, hoi/utils), not just the MCP wrapper docstrings. Not for choosing which tool to use in the first place - see the orientation skill for that.
---

# HOI metrics

Covers the 15 higher-order-interaction tools braina exposes. Per-metric
detail lives in `references/` next to this file
(`${CLAUDE_PLUGIN_ROOT}/skills/hoi-metrics/references/`); **read the matching
file before answering a parameter-level question.**

| Tools | Reference file |
|---|---|
| `hoi_oinfo`, `hoi_gradient_oinfo`, `hoi_infotopo`, `hoi_dtc`, `hoi_tc`, `hoi_sinfo` | `references/static-metrics.md` |
| `hoi_redundancy_mmi`, `hoi_synergy_mmi`, `hoi_rsi`, `hoi_infotot` | `references/static-metrics.md` (target-y section) |
| `hoi_transfer_entropy`, `hoi_dotot`, `hoi_redundancy_phiid`, `hoi_atoms_phiid` | `references/dynamic-metrics.md` |
| `hoi_get_nbest_mult`, bootstrap CIs (`n_boots`), `samples_path`, `method` | `references/nbest-bootstrap-estimators.md` |
| CPU vs GPU, JAX memory on shared machines | `references/compute-backend.md` |

## Input and output conventions

- Static metrics: `(n_samples, n_features)` or `(n_samples, n_features,
  n_variables)`; the analysis is over **multiplets** (all subsets of
  features between `minsize` and `maxsize`), not pairs. Dynamic metrics
  need `(n_samples, n_features, n_times)` and a lag `tau`.
- Give a `.nc` input whose feature dimension has a coordinate (e.g. `roi`):
  the output then carries `multiplet_names` such as `A / C / D`. HOI itself
  rejects xarray, so the wrapper converts to numpy and keeps the names.
- Every `hoi_*` tool writes a `.nc` with dims `(multiplets, variables)` and
  coordinates `multiplets` (feature indices, e.g. `0,2,3`), `order`
  (multiplet size) and `multiplet_names`. Save as `.nc`; `.npy` drops this
  metadata and `hoi_get_nbest_mult` refuses it.
- When a target `y` is given to a metric that accepts it, HOI appends `y`
  as an extra feature: multiplets containing index `n_features` are the
  task-related ones and are labelled `... / y`.
- Cost grows combinatorially with `n_features` and `maxsize`; cap `maxsize`
  (3-5) on more than ~10 features.

## Sign conventions differ between metrics

The single most common misinterpretation. Check before reporting "synergy"
or "redundancy":

| Metric | Positive means | Negative means |
|---|---|---|
| `oinfo`, `gradient_oinfo`, `infotopo`, `dotot` | Redundancy | Synergy |
| `rsi` | **Synergy** | **Redundancy** |
| `redundancy_mmi`, `synergy_mmi`, `redundancy_phiid`, `atoms_phiid` | one-directional quantities (>= 0 in principle) | small negative = estimation noise |
| `dtc`, `tc`, `sinfo`, `infotot`, `transfer_entropy` | magnitudes (>= 0), no redundancy/synergy sign | idem |

`rsi` is the **opposite** of `oinfo`. `hoi_get_nbest_mult` reports the most
positive and the most negative multiplets separately - say which half is
synergy for the metric at hand.

## Estimator

Every metric's `fit()` takes `method` - exposed by all wrappers:
`'gc'` (Gaussian copula, default), `'gauss'`, `'binning'` (data must be
discretised first), `'knn'`, `'kernel'`. The default shares the
Gaussian-copula assumption of Frites' `covgc`/`te`/`pid`: robust to monotonic
nonlinearities, not fully non-parametric. Details and when to switch:
`references/nbest-bootstrap-estimators.md`.

## Uncertainty and significance

- `n_boots > 0` on any static metric writes `<output>_ci.nc` with bootstrap
  percentiles of the estimate (uncertainty of the value).
- p-values against chance need a permutation null fed to `frites_wf_stats`;
  the `workflows` skill has the recipe. The two answer different questions.

## Examples

Relative to `${CLAUDE_PLUGIN_ROOT}`: `examples/hoi/metrics/plot_oinfo.py`,
`plot_infotopo.py`, `plot_syn_red_mmi.py`, `plot_rsi.py`,
`plot_syn_phiID.py`; bootstrap: `examples/hoi/statistics/plot_bootstrapping.py`.
No dedicated example for `dtc`, `gradient_oinfo`, `tc`, `sinfo`, `dotot`.
