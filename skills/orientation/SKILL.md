---
description: Use when the user has electrophysiological/neuroimaging data (fMRI, MEG, EEG, LFP, MUA) and an analysis question but hasn't settled on a method or a statistical approach — e.g. "how do I check if region A drives region B", "is this a connectivity or a higher-order interaction question", "which braina tool should I use", "how do I test this for significance", "fixed vs random effect", "which correction for multiple comparisons". Triages the question against the Frites and HOI toolboxes and against inference/correction choices, returning a ranked shortlist of candidates with tradeoffs. Not for users who already know which tool and stats approach they want.
tags: [triage, orientation, frites, hoi, connectivity, higher-order-interactions, statistics]
---

# Braina orientation: picking the right tool

This skill is a decision aid, not a reference manual. Its job is to turn a
plain-language question into a **ranked shortlist of candidate MCP tools**,
each with a one-line reason and a one-line tradeoff — never a single
forced answer. Once the user picks a direction, hand off to the relevant
topic skill (`frites-connectivity`, `frites-workflows`, `hoi-metrics`) for
parameter-level detail.

If the data hasn't been inspected yet, run `inspect_data` first — the
answer to "how many signals, how many trials, is there a `y` variable"
often settles half the triage below on its own.

## Step 0 — no real data yet?

If the user is prototyping a pipeline or teaching/learning before they have
real recordings, suggest `frites_sim_ar` to generate synthetic AR data with
a known ground-truth interaction structure, so the chosen method can be
sanity-checked before touching real data.

## Step 1 — how many signals are being asked about, jointly?

- **Exactly two** (or many pairs analyzed one pair at a time) → Frites,
  go to Step 2.
- **Three or more, analyzed as one joint group** (the question is about
  the group's structure, not any single pair) → HOI, go to Step 3.
- **Not sure yet / exploratory** → recommend starting pairwise with Frites
  to build intuition on a couple of region pairs, or, if there are many
  regions and no clear pair in mind, use `hoi_get_nbest_mult` first to
  search for which subsets of regions look most informative before
  committing to a deeper analysis.

## Step 2 — Frites: pairwise questions

Ask (or infer from context) which of these the user actually means —
they sound similar but answer different questions:

### 2a. "Does A drive/influence B over time?" (directionality)

1. **`frites_conn_covgc`** — covariance-based Granger causality. The
   established default for directed influence in LFP/iEEG (used in
   Brovelli et al. 2004 PNAS, 2015 J Neurosci). Time-resolved, well
   validated. By default (`method='gc'`) it copula-normalizes the data
   first, so it's robust to monotonic nonlinearities, not limited to
   strictly linear/Gaussian data (that stricter assumption only applies
   if `method='gauss'` is chosen explicitly). *Tradeoff:* heavier to
   compute than a plain correlation; still assumes the dependency
   structure itself is well described by a Gaussian copula.
2. **`frites_conn_te`** — transfer entropy. Scans a range of delays
   directly rather than needing a single window/lag choice like `covgc`.
   *Tradeoff:* uses the same Gaussian-copula estimator family as `covgc`
   under the hood (see the `frites-connectivity` skill), so despite
   transfer entropy being a model-free concept, this particular estimate
   isn't fully nonparametric; it's also noisier and needs more
   trials/data, and the delay parameters (`max_delay`, `min_delay`,
   `step_delay`) require some tuning.
3. **`frites_conn_fit`** — feature-specific information transfer. Use
   instead of the above when the real question is *which stimulus/task
   feature* is being transferred from A to B, not just whether influence
   exists. *Tradeoff:* more specialized, needs a well-defined feature `y`.

### 2b. "How coupled are A and B over time?" (no direction implied)

1. **`frites_conn_dfc`** — dynamic functional connectivity over sliding
   windows. Good general-purpose default for time-resolved undirected
   coupling. *Tradeoff:* window choice (`win_sample`) trades time
   resolution against stability of the estimate.
2. **`frites_conn_spec`** — spectral connectivity (e.g. coherence). Use
   when the coupling is expected to live in a specific frequency band
   (oscillatory synchrony). *Tradeoff:* needs a frequency range decision
   up front; less informative if the coupling isn't oscillatory.
3. **`frites_conn_ccf`** — cross-correlation. Fastest, simplest, good
   first pass. *Tradeoff:* least nuanced — no frequency or conditioning
   information.

### 2c. "Does A's activity carry information about an external variable Y?" (stimulus, condition, behavior)

1. **`frites_wf_mi`** — mutual information workflow with built-in
   group-level permutation statistics. The direct answer to this question
   when there are multiple subjects/trials. *Tradeoff:* heavier
   (`n_perm` permutations); needs `mi_type` set correctly for the data
   type (continuous vs. categorical `y`).
2. **`frites_conn_ii`** — interaction information. Use if the real
   question is how A's relationship with Y *changes* depending on a third
   variable, rather than a plain group-level MI test. *Tradeoff:* a single
   summary number, no built-in group stats (pair with `frites_wf_stats`).

### 2d. "What do A and B jointly tell you about a target Y — redundant, unique, or synergistic?"

1. **`frites_conn_pid`** — partial information decomposition. Directly
   answers this with separate unique/redundant/synergistic terms. The
   most interpretable option when the decomposition itself is the point.
   *Tradeoff:* three outputs to interpret instead of one; needs enough
   data to estimate each term reliably.
2. **`frites_conn_ii`** — cheaper single-number summary (synergy minus
   redundancy) if the full three-way breakdown isn't needed.

Once a tool from 2a–2d has produced an effect, go to **Step 4** to decide
how to test it for significance across subjects/trials.

## Step 3 — HOI: joint questions over three or more signals

### 3a. "Overall, is this group of regions more redundant or synergistic?"

1. **`hoi_oinfo`** — O-information. The standard single-number summary:
   positive = redundancy-dominated, negative = synergy-dominated. Good
   default starting point for any higher-order question.
2. **`hoi_gradient_oinfo`** — same idea, but tracks how O-info changes as
   regions are added one at a time. Use when the goal is to find *which*
   region tips the group from redundant to synergistic (or vice versa).
   *Tradeoff:* more expensive, more output to interpret than plain `oinfo`.

### 3b. "I want redundancy and synergy as two separate numbers, not their difference"

1. **`hoi_redundancy_mmi`** + **`hoi_synergy_mmi`** — explicit
   decomposition via minimum-MI. Use together when both quantities matter
   individually (O-info alone only gives their difference).
2. **`hoi_rsi`** — redundancy-synergy index. A single normalized balance
   score if a simpler summary than two raw quantities is preferred.

### 3c. "I want the full picture across all interaction orders, not just one order"

1. **`hoi_infotopo`** — information topology across pairs, triplets, …,
   n-tuples. The richest output. *Tradeoff:* combinatorial cost grows
   fast with `maxsize`; only worth it when the multi-scale structure
   itself is the research question.

### 3d. "Which subset of regions is worth analyzing at all?"

1. **`hoi_get_nbest_mult`** — searches for the most informative multiplets
   from a HOI metric's output. Use this *before* committing to a deep
   analysis on a large set of regions, to prioritize which subsets to
   examine with the tools above.

### 3e. "I want one number for how much the whole group shares in common"

1. **`hoi_dtc`** — dual total correlation. A total-correlation-style
   summary of shared information across the whole group, independent of
   the synergy/redundancy framing above.

HOI metrics don't compute p-values on their own. Go to **Step 4** once
there's an effect (real vs. a null built from shuffled/permuted data) to
test for significance across subjects or trials.

## Step 4 — what type of statistical analysis?

This step is the same regardless of whether the effect came from Frites
or HOI. Two situations:

- **Already using `frites_wf_mi` or `frites_wf_conn_comod`** — statistics
  are built in. Only `inference` and `n_perm` need to be chosen (see
  below); there's no separate stats call to make.
- **Using any other tool** (`frites_conn_*`, any `hoi_*`) — these return a
  raw effect with no p-values. Build a null distribution (e.g. shuffle
  trial labels / circularly shift time series / recompute the same metric
  on permuted data) and feed the real effect + permutations into
  **`frites_wf_stats`**.

### 4a. Fixed-effect or random-effect? (`inference`)

1. **`rfx`** (random effect) — the usual choice for group studies with
   multiple subjects. Treats subjects as a random sample from a
   population, so conclusions generalize *beyond* the subjects tested.
2. **`ffx`** (fixed effect) — pools all trials/subjects together;
   conclusions apply only to the specific subjects tested. More sensitive
   with few subjects/trials, but doesn't generalize. Use for single-subject
   analyses or small, non-representative cohorts (e.g. a handful of iEEG
   patients) where population-level generalization isn't the goal.

### 4b. Multiple-comparisons correction (`mcp`)

1. **`cluster`** — cluster-based permutation correction. Best default for
   time-resolved neural signals, where a true effect is expected to be
   contiguous in time (and/or space/frequency) rather than scattered.
   Most powerful option *when that contiguity assumption holds*.
   *Tradeoff:* needs a cluster-forming threshold (`cluster_th`, can be left
   `None` for an automatic default); less appropriate if effects aren't
   expected to cluster.
2. **`maxstat`** — max-statistic correction. No contiguity assumption, so
   it's the safer choice for single time-point comparisons or non-temporal
   outputs (e.g. one HOI value per multiplet). *Tradeoff:* noticeably more
   conservative than `cluster` → lower power.
3. **`fdr`** — false discovery rate. Controls the *expected proportion* of
   false positives rather than the family-wise error rate. Reasonable when
   testing many largely independent comparisons (many ROI pairs, many
   multiplets) and some false positives are tolerable. *Tradeoff:* weaker
   guarantee than `cluster`/`maxstat`/`bonferroni`, and less appropriate
   when tests are strongly correlated (e.g. neighboring time points).
4. **`bonferroni`** — classic, very conservative family-wise correction.
   Use when the number of comparisons is small and the simplest, most
   defensible correction is preferred over statistical power.
   *Tradeoff:* lowest power of the four, especially as the number of tests
   grows.

### 4c. Direction of the test (`tail`)

- **`tail=0`** (two-tailed) — the safer default when there's no strong
  prior on the direction of the effect (increase *or* decrease).
- **`tail=1`** — one-tailed, testing only for an increase (e.g. "does
  connectivity go up"). Use only when the hypothesis is genuinely
  directional — it doubles sensitivity in that direction but is blind to
  an effect in the opposite direction.
- **`tail=-1`** — one-tailed, testing only for a decrease.

### 4d. How many permutations? (`n_perm`, workflow tools only)

More permutations → more precise p-values (the smallest achievable
p-value is `1 / (n_perm + 1)`) at the cost of compute time. `n_perm=1000`
is a reasonable default; increase it if the correction method (e.g.
`bonferroni` over many comparisons) demands a stricter significance
threshold than 1000 permutations can resolve.

## Step 5 — worked example for the chosen tool

Before writing new analysis code, check whether a validated example already
covers the chosen tool — reuse its API usage rather than reconstructing it
from memory. Paths are relative to `${CLAUDE_PLUGIN_ROOT}`.

| Tool | Example |
|---|---|
| `frites_conn_covgc` | `examples/frites/conn/plot_covgc.py` |
| `frites_conn_dfc` | `examples/frites/conn/plot_dfc.py` |
| `frites_conn_pid` | `examples/frites/conn/plot_pid.py` |
| `frites_conn_ii` | `examples/frites/conn/plot_ii.py` |
| `frites_conn_fit` | `examples/frites/conn/plot_fit.py` |
| `frites_conn_ccf` | `examples/frites/conn/plot_ccf.py` |
| `frites_wf_conn_comod` | `examples/frites/conn/plot_conn.py` |
| `frites_wf_mi` | `examples/frites/mi/plot_wf_mi_cc.py` (also `plot_wf_mi_cd.py`, `plot_wf_mi_ccd.py` for the other `mi_type`s) |
| `frites_wf_stats` (`ffx` vs `rfx`) | `examples/frites/statistics/plot_wf_mi_stats_compare_ffx.py`, `..._rfx.py` |
| `frites_sim_ar` | `examples/frites/simulations/plot_ground_truth.py`, `examples/frites/armodel/plot_ar_pairwise.py` |
| `hoi_oinfo` | `examples/hoi/metrics/plot_oinfo.py` |
| `hoi_infotopo` | `examples/hoi/metrics/plot_infotopo.py` |
| `hoi_redundancy_mmi` / `hoi_synergy_mmi` | `examples/hoi/metrics/plot_syn_red_mmi.py` |
| `hoi_rsi` | `examples/hoi/metrics/plot_rsi.py` |
| `hoi_get_nbest_mult` | not standalone — used inside `plot_oinfo.py` / `plot_infotopo.py` |

**No dedicated example yet** for `frites_conn_te`, `frites_conn_spec`,
`hoi_dtc`, and `hoi_gradient_oinfo` — for these, rely on the MCP tool's
docstring and the underlying Frites/HOI API docs directly, and say so
rather than presenting an improvised call with unwarranted confidence.

## After the shortlist

Once the user picks a tool, switch to the matching topic skill for
parameter details and worked examples:
`frites-connectivity`, `frites-workflows`, or `hoi-metrics`.

## Further reading

For theoretical background beyond what this skill or the topic skills
cover — full derivations, glossaries, the broader design philosophy of
each toolbox:

- Frites: https://brainets.github.io/frites/overview/index.html
- HOI: https://brainets.github.io/hoi/
