---
description: Use once a Frites workflow tool has been chosen (frites_wf_stats, frites_wf_mi, frites_wf_conn_comod) or the question is about how permutation testing, cluster correction, or fixed/random-effect inference actually works under the hood. Grounded in the real Frites source (frites/workflow, frites/stats), not just the MCP wrapper docstrings. Not for choosing which tool to use in the first place — see the orientation skill for that.
tags: [frites, statistics, permutation-testing, cluster-correction, workflows]
---

# Frites workflows and statistics

Covers the 3 workflow tools braina exposes — `frites_wf_stats`,
`frites_wf_mi`, `frites_wf_conn_comod` — plus `WfMiCombine`, a fourth
workflow that exists in Frites (`frites/workflow/wf_mi_combine.py`) but
**is not currently wrapped by any MCP tool**.

## How the permutation test actually works

All statistics here are non-parametric and permutation-based
(`frites/stats/`). The logic is the same regardless of which workflow
produced the effect:

1. **Build a null distribution.** For `WfMi`/`WfConnComod`, this means
   repeatedly shuffling the regressor `y` (or swapping trials) *within
   each subject* (`frites.stats.permute_mi_vector` /
   `permute_mi_trials`) and recomputing the same MI/connectivity metric
   on each shuffle — `n_perm` times. Shuffling within-subject keeps the
   permutation exchangeable under the null while respecting the
   study's structure. For `frites_wf_stats` used standalone, the caller
   is responsible for supplying this null themselves (e.g. from
   `frites_sim_ar`-style surrogates, or by permuting whatever produced
   the original effect).
2. **Fixed vs. random effect (`inference`)** changes *how* the true
   effect and the null are compared:
   - `rfx` — runs a **one-sample t-test across subjects**
     (`frites.stats.rfx_ttest`) on both the true effect and every
     permutation, with a small regularization (`sigma`, default
     `0.001`, a "hat" adjustment preventing near-zero variance from
     inflating t-values). This is what makes `rfx` generalize beyond the
     specific subjects tested — it's explicitly testing whether the
     *population* mean differs from zero.
   - `ffx` — skips that across-subject t-test and compares the pooled
     effect directly against the pooled permutation null. No claim about
     the population is made, only about the specific data tested.
3. **Multiple-comparisons correction (`mcp`)** is applied to the
   comparison between true effect and null:
   - `'cluster'` (`frites.stats.cluster_correction_mcp`) — first picks a
     cluster-forming threshold (`cluster_th`; if left `None`, it's
     auto-set as the `1 - cluster_alpha` percentile of the permutation
     distribution itself, e.g. the 95th percentile for the default
     `cluster_alpha=0.05`), groups contiguous above-threshold points into
     clusters, then tests each cluster's mass/size against the null
     distribution of cluster masses built from the permutations. This is
     why it's the most powerful option when a true effect is expected to
     be contiguous in time — the correction is done once per cluster, not
     once per time point.
   - `'maxstat'`/`'fdr'`/`'bonferroni'` (`frites.stats.testwise_correction_mcp`)
     — correct at *every individual test* (every time point / ROI /
     frequency) independently. The library's own docstring is blunt about
     the cost: "usually suffers from a low statistical power."
   - `'nostat'` / `None` — skip correction entirely. Exists in the real
     API but not something to reach for when the result needs to survive
     scrutiny — only useful for quick-look exploration.
4. **`tail`** picks which side of the null distribution counts:
   `1` = true effect greater than the null, `-1` = smaller, `0` = either
   direction (two-tailed).

## `frites_wf_stats` — standalone statistical workflow

The general-purpose entry point: takes a precomputed effect and a
precomputed permutation null and runs the machinery above. This is the
tool to reach for after any `frites_conn_*` or `hoi_*` call, since none
of those compute p-values on their own.

Exposed: `inference`, `mcp`, `tail`, `cluster_th`. Not exposed:
`cluster_alpha` (library default `0.05` — see how `cluster_th=None` is
resolved above), `ttested` (whether the input is already t-tested),
`rfx_sigma`/`rfx_center` (the regularization and re-centering knobs for
the rfx t-test).

API reference: https://brainets.github.io/frites/api/generated/frites.workflow.WfStats.html

## `frites_wf_mi` — mutual information workflow

Computes local mutual information between the data and a regressor `y`
*and* runs the permutation test above in a single call — the direct tool
for "does this region encode Y" questions with built-in group statistics.

- `mi_type` — `'cc'` (continuous `y`), `'cd'` (categorical `y`), or
  `'ccd'` (conditional — a third option not available on the plain
  `frites_conn_*` tools, which only support `'cc'`/`'cd'`).
- `inference`, `n_perm` — as described above.

Hardcoded by the wrapper (real `WfMi.fit()` accepts these but the MCP
tool doesn't expose them): `mcp='cluster'`, `cluster_th=None` (auto),
`cluster_alpha=0.05`. If a genuinely different correction method is
needed for a mutual-information workflow, that currently requires
scripting against Frites directly rather than through this MCP tool.

API reference: https://brainets.github.io/frites/api/generated/frites.workflow.WfMi.html

## `frites_wf_conn_comod` — connectivity comodulation workflow

Same idea as `WfMi` but for pairwise connectivity comodulation across
ROIs rather than a data-vs-`y` relationship — instantaneous MI-based
coupling between every ROI pair, with the same built-in permutation
statistics.

- `inference`, `n_perm` — as described above.

Same hardcoding caveat as `frites_wf_mi`: `mcp` is always `'cluster'`
through this wrapper.

API reference: https://brainets.github.io/frites/api/generated/frites.workflow.WfConnComod.html

## `WfMiCombine` — not currently exposed via MCP

Combines two already-fitted `WfMi` workflows by subtracting their effect
and permutations (`wf_1 − wf_2`), then lets the combined difference be
tested for significance — the typical use case is "fit MI for stimulus A
vs. baseline, fit MI for stimulus B vs. baseline, then test whether A's
effect is larger than B's." There is no `frites_wf_mi_combine` MCP tool;
this would need direct Frites scripting outside braina today.

API reference: https://brainets.github.io/frites/api/generated/frites.workflow.WfMiCombine.html

## Examples

See the `orientation` skill's Step 5 table for the validated example
script per tool. `frites_wf_stats`'s `ffx`-vs-`rfx` comparison is
specifically demonstrated in
`examples/frites/statistics/plot_wf_mi_stats_compare_ffx.py` and
`..._rfx.py`.
