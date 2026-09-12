---
description: Use once a statistical workflow tool has been chosen (frites_wf_stats, frites_wf_mi, frites_wf_conn_comod) or the question is about how permutation testing, cluster correction, bootstrapping, or fixed/random-effect inference actually works under the hood — for either Frites connectivity or HOI metrics. Grounded in the real Frites and HOI source (frites/workflow, frites/stats, hoi/utils), not just the MCP wrapper docstrings. Not for choosing which tool to use in the first place — see the orientation skill for that.
tags: [frites, hoi, statistics, permutation-testing, cluster-correction, bootstrapping, workflows]
---

# Statistical workflows for Frites and HOI

Covers the 3 Frites workflow tools braina exposes — `frites_wf_stats`,
`frites_wf_mi`, `frites_wf_conn_comod` — plus `WfMiCombine`, a fourth
workflow that exists in Frites (`frites/workflow/wf_mi_combine.py`) but
**is not currently wrapped by any MCP tool**. It also covers how the same
statistical machinery applies to HOI output, and HOI's own idiomatic
alternative (bootstrap confidence intervals).

**Neither toolbox's `hoi_*` tools compute significance on their own** —
same situation as the bare `frites_conn_*` tools. `frites_wf_stats` is a
generic non-parametric engine: it doesn't know or care whether the
`effect`/`perms` arrays it's given came from Frites or HOI, only that
their shapes match its (n_roi-like-axis, n_subjects, ..., n_times)
convention. That's what makes it reusable across both toolboxes — see
"Applying this to HOI output" below.

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

## Applying this to HOI output

`frites_wf_stats`'s `_prepare_stats_input` step splits a 3D/4D array on
its first non-permutation axis and treats each slice as one independent
test — that's literally "one test per ROI" for Frites, but the same
mechanism works for "one test per multiplet" for HOI: an `hoi_*` tool's
`(n_mult, n_variables)` output plays the same structural role as Frites'
`(n_roi, n_subjects, n_times)`, with `n_mult` standing in for `n_roi`.

To actually get a p-value on an `hoi_*` result:

1. **Build the null yourself** — braina's `hoi_*` wrappers don't do this
   automatically, unlike `frites_wf_mi`/`frites_wf_conn_comod`. The
   right shuffle depends on the metric:
   - Metrics that take a `y` (`redundancy_mmi`, `synergy_mmi`, `rsi`,
     `gradient_oinfo`) — shuffle `y` and rerun the same `hoi_*` call,
     `n_perm` times. Directly analogous to `permute_mi_vector` in Frites.
   - Metrics without `y` (`oinfo`, `infotopo`, `dtc`) — shuffle one
     feature's sample order (breaking its higher-order dependency with
     the rest) and rerun, `n_perm` times.
2. **Stack the results** into `(n_perm, n_mult, ...)` and feed alongside
   the real (unshuffled) `(n_mult, ...)` effect into `frites_wf_stats`.
3. **Pick `inference` deliberately.** If there's genuinely one HOI
   estimate per subject in the `n_variables` axis, `rfx` is appropriate,
   exactly as with Frites. If there's only a single dataset/subject
   (the common case for an HOI analysis), there's no population to
   generalize to — use `ffx` rather than defaulting to `rfx` out of habit.

## HOI's own approach: bootstrap confidence intervals

HOI's own examples (`examples/hoi/statistics/plot_bootstrapping.py`)
don't use permutation testing at all — they use **bootstrapping**:
resample the samples *with replacement* (`sklearn.utils.resample`) and
recompute the same HOI metric `n_boots` times via the real fit()
signature's `samples` parameter (`model.fit(method="gc",
samples=samples, minsize=3)`), then take the `[5, 95]%` percentile of the
bootstrap distribution as a confidence interval around the estimate.

This answers a **different question** than permutation testing: bootstrap
CIs quantify *how uncertain the estimate itself is*; permutation p-values
quantify *whether the estimate is different from chance*. They're
complementary, not interchangeable — don't substitute one for the other.

**Not exposed via any MCP tool**: `samples` exists on every HOI metric's
real `fit()` but none of braina's `hoi_*` wrappers expose it, so
bootstrapping currently requires scripting directly against HOI rather
than going through braina's MCP tools.

## Examples

See the `orientation` skill's Step 5 table for the validated example
script per Frites workflow tool. `frites_wf_stats`'s `ffx`-vs-`rfx`
comparison is specifically demonstrated in
`examples/frites/statistics/plot_wf_mi_stats_compare_ffx.py` and
`..._rfx.py`. For HOI's bootstrap approach, see
`examples/hoi/statistics/plot_bootstrapping.py` — there is no example of
permutation-testing HOI output via `frites_wf_stats` (the recipe above is
not yet demonstrated anywhere in the repo).
