---
name: workflows
description: Use once a statistical workflow tool has been chosen (frites_wf_stats, frites_wf_mi, frites_wf_conn_comod, frites_wf_mi_combine) or the question is about how permutation testing, cluster correction, bootstrapping, or fixed/random-effect inference actually works under the hood - for either Frites connectivity or HOI metrics. Grounded in the real Frites and HOI source (frites/workflow, frites/stats, hoi/utils), not just the MCP wrapper docstrings. Not for choosing which tool to use in the first place - see the orientation skill for that.
---

# Statistical workflows for Frites and HOI

Covers the four workflow tools - `frites_wf_stats`, `frites_wf_mi`,
`frites_wf_conn_comod`, `frites_wf_mi_combine` - and how the same
permutation machinery applies to HOI output. How the test works internally
(null construction, rfx t-test, cluster correction) is in
`references/permutation-internals.md` next to this file; read it when the
question is "why" rather than "which parameter".

Neither the bare `frites_conn_*` tools nor the `hoi_*` tools compute
significance. `frites_wf_stats` is a generic non-parametric engine: it only
needs an effect and a matching permutation null, whatever produced them.

## `frites_wf_stats` - statistics on a precomputed effect

- Inputs: effect `(n_roi, n_subjects, n_times)` and permutations `(n_perm,
  n_roi, n_subjects, n_times)` (`.npy` or `.nc`; a `.nc` with `roi`/`times`
  coordinates gives labelled outputs). "roi" may be any unit of test:
  pairs, multiplets, frequencies.
- `inference` (`rfx` | `ffx`), `mcp` (`cluster` | `maxstat` | `fdr` |
  `bonferroni`), `tail` (`-1`, `0`, `1`), `cluster_th` (`None` = auto).
- Writes `<prefix>_pvalues.nc` and, for `rfx`, `<prefix>_tvalues.nc`, both
  `(times, roi)`.
- Not exposed: `cluster_alpha`, `ttested`, `rfx_sigma`, `rfx_center`
  (library defaults 0.05, False, 0.001, False).
API: https://brainets.github.io/frites/api/generated/frites.workflow.WfStats.html

## `frites_wf_mi` - MI about y with built-in statistics

- Data: one file per subject (`data_path=[...]`, `y_path=[...]`) or a `.nc`
  with a `subject` dimension; single file = single subject. `rfx` requires
  at least two subjects (refused otherwise, as Frites asserts).
- `mi_type`: `cc` continuous y, `cd` categorical y, `ccd` conditional (y
  continuous, z categorical - z not exposed via MCP).
- `inference`, `n_perm`, `mcp` (`cluster` default; `maxstat`, `fdr`,
  `bonferroni`, `nostat`), `cluster_th` (`None` auto or `'tfce'`),
  `cluster_alpha`, `random_state` (reproducible permutations), `estimator`
  (`default` GCMI, or `pearson`/`spearman`/`dcorr`/`binmi` for `cc`),
  `n_jobs`.
- `save_workflow=True` pickles the fitted workflow to `<prefix>_wf.pkl` for
  `frites_wf_mi_combine`.
- Outputs `<prefix>_mi.nc` and `<prefix>_pvalues.nc`, `(times, roi)`.
API: https://brainets.github.io/frites/api/generated/frites.workflow.WfMi.html

## `frites_wf_conn_comod` - pairwise comodulation with statistics

Same data conventions and statistical options as `frites_wf_mi` (no `y`):
instantaneous MI-based coupling for every ROI pair (`roi` = `A-B`), tested
against trial-swapped permutations.
API: https://brainets.github.io/frites/api/generated/frites.workflow.WfConnComod.html

## `frites_wf_mi_combine` - contrast two MI workflows

`WfMiCombine(wf_1, wf_2)`: subtracts effect and permutations (`wf_1 - wf_2`)
and re-runs the statistics, e.g. "is information about the stimulus larger
in condition A than B". Both workflows must be fitted on the same ROIs,
times, `inference` and `n_perm`, and saved with `save_workflow=True`.
Options: `mcp`, `cluster_th`, `cluster_alpha`.
API: https://brainets.github.io/frites/api/generated/frites.workflow.WfMiCombine.html

## Applying this to HOI output

`hoi_*` outputs are `(n_mult, n_variables)`; `n_mult` plays the role of
`n_roi`. Recipe for a p-value:

1. **Build the null yourself.** Metrics with `y` (`rsi`, `redundancy_mmi`,
   `synergy_mmi`, `gradient_oinfo`, `infotot`): shuffle `y` across samples
   and rerun the same `hoi_*` call `n_perm` times (save the shuffled `y` as
   `.npy`, or use `samples_path`). Metrics without `y` (`oinfo`, `infotopo`,
   `dtc`, `tc`, `sinfo`): shuffle one feature's sample order (breaking its
   dependency with the rest) and rerun.
2. **Stack** the permuted results into `(n_perm, n_mult, n_subjects, 1)` and
   the real effect into `(n_mult, n_subjects, 1)`; save as `.npy` and call
   `frites_wf_stats`.
3. **Choose `inference` deliberately**: `rfx` only if the subject axis really
   holds one estimate per subject; a single dataset has no population to
   generalise to - use `ffx`. `mcp='maxstat'` or `'fdr'` fit one-value-per-
   multiplet outputs better than `cluster`, which assumes contiguity.

## Bootstrap confidence intervals (HOI's own approach)

`n_boots > 0` on any static `hoi_*` tool resamples samples with replacement
and writes `<output>_ci.nc` with `ci_percentiles` (default `[5, 95]`). This
answers **how uncertain is the estimate**, not **is it above chance**; the
two are complementary, not interchangeable. Details:
`hoi-metrics/references/nbest-bootstrap-estimators.md`.

## Examples

`${CLAUDE_PLUGIN_ROOT}/examples/frites/statistics/plot_wf_mi_stats_compare_ffx.py`
and `..._rfx.py` (ffx vs rfx), `examples/frites/mi/plot_wf_mi_combine.py`
(WfMiCombine), `examples/hoi/statistics/plot_bootstrapping.py` (bootstrap).
There is no example of permutation-testing HOI output through
`frites_wf_stats`; the recipe above is not yet demonstrated in the repo.
