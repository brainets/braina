---
name: orientation
description: Use when the user has electrophysiological/neuroimaging data (fMRI, MEG, EEG, LFP, MUA) and an analysis question but hasn't settled on a method or a statistical approach - e.g. "how do I check if region A drives region B", "is this a connectivity or a higher-order interaction question", "which braina tool should I use", "how do I get my data into braina", "how do I test this for significance", "fixed vs random effect", "which correction for multiple comparisons". Triages the question against the Frites and HOI toolboxes and against inference/correction choices, returning a ranked shortlist of candidates with tradeoffs. Not for users who already know which tool and stats approach they want.
---

# Braina orientation: picking the right tool

A decision aid, not a reference manual. Turn a plain-language question into
a **ranked shortlist of candidate MCP tools**, each with a one-line reason
and a one-line tradeoff - never a single forced answer. Once the user picks
a direction, hand off to the topic skill (`frites-connectivity`,
`hoi-metrics`, `workflows`) for parameter-level detail. Worked examples per
tool: `references/examples.md` next to this file.

## Step 0 - get the data in, then look at it

- Not `.nc`/`.npy` yet (MNE `.fif` epochs, MATLAB `.mat`, `.csv`)? Use
  `convert_to_nc`; it also attaches `roi` names, `times`/`sfreq` to a bare
  `.npy`. Coordinates must be named `roi` and `times` for Frites to keep
  labels - the converter does that.
- Run `inspect_data` first: "how many signals, how many trials, is there a
  y variable, is there a time axis" settles half the triage.
- No real data yet? `frites_sim_ar` (with `random_state`) gives synthetic
  data with known directed interactions to sanity-check a pipeline.
- After any tool: `plot_result` writes a PNG; look at it before
  interpreting numbers.

## Step 1 - how many signals are asked about, jointly?

- **Two** (or many pairs, one at a time) -> Frites, Step 2.
- **Three or more as one group** (the group's structure is the question)
  -> HOI, Step 3.
- **Unsure / exploratory** -> start pairwise on a couple of pairs, or run
  `hoi_oinfo` over all regions with `maxsize=3` and rank with
  `hoi_get_nbest_mult` (needs `.nc` output) to see which triplets stand out.

## Step 2 - Frites: pairwise questions

### 2a. "Does A drive B?" (directed)
1. **`frites_conn_covgc`** - covariance-based Granger causality, per trial,
   time-resolved; the established default (Brovelli 2004, 2015). Default
   `method='gc'` copula-normalises (robust to monotonic nonlinearities).
   *Tradeoff:* needs `dt`/`lag`/`t0` choices; still a Gaussian-copula
   dependency assumption. Follow with `frites_conn_net` for A->B minus B->A.
2. **`frites_conn_te`** - transfer entropy scanned over delays, across
   trials. *Tradeoff:* same estimator family (not fully non-parametric),
   noisier, needs many trials.
3. **`frites_conn_fit`** - feature-specific information transfer: *which
   task feature* is transferred. *Tradeoff:* needs a well-defined `y`.

### 2b. "How coupled are A and B?" (undirected)
1. **`frites_conn_dfc`** - windowed coupling per trial (GCMI, or
   `estimator='pearson'|'spearman'|'dcorr'`). *Tradeoff:* window length
   trades time resolution for stability.
2. **`frites_conn_spec`** - coherence / PLV per frequency, when the coupling
   is oscillatory. *Tradeoff:* frequency choice up front; needs `sfreq`.
3. **`frites_conn_ccf`** - cross-correlation, the quick lag estimate.
   *Tradeoff:* least nuanced.
For coupling **with built-in group statistics**, `frites_wf_conn_comod`.

### 2c. "Does A carry information about an external variable y?"
1. **`frites_wf_mi`** - MI between data and `y` with permutation statistics
   built in; multi-subject (one file per subject, `inference='rfx'`).
   *Tradeoff:* `mi_type` must match `y` (`cc` continuous, `cd` categorical,
   `ccd` conditional); `n_perm` costs time.
2. **`frites_wf_mi_combine`** - contrast two fitted `frites_wf_mi` runs
   (saved with `save_workflow=True`), e.g. "is the effect larger in
   condition A than B".

### 2d. "What do A and B jointly tell about y: redundant, unique, synergistic?"
1. **`frites_conn_pid`** - full decomposition (four outputs).
2. **`frites_conn_ii`** - one number, synergy minus redundancy.
Both need a permutation null + `frites_wf_stats` for p-values.

## Step 3 - HOI: joint questions over three or more signals

### 3a. "Is this group more redundant or synergistic?"
1. **`hoi_oinfo`** - O-information: `> 0` redundancy, `< 0` synergy. The
   default starting point (`minsize=3`).
2. **`hoi_gradient_oinfo`** (needs `y`) - which region tips the balance.
3. **`hoi_sinfo`** / **`hoi_tc`** / **`hoi_dtc`** - *how much* dependence
   there is, without the redundancy/synergy sign.

### 3b. "Redundancy and synergy about y as two numbers"
1. **`hoi_redundancy_mmi`** + **`hoi_synergy_mmi`** (+ `hoi_infotot` for the
   total they decompose).
2. **`hoi_rsi`** - one balance number; **sign is opposite to `oinfo`**
   (positive = synergy).

### 3c. "The full picture across all interaction orders"
1. **`hoi_infotopo`** - all orders from pairs up; combinatorial cost.

### 3d. "Which subset of regions stands out?"
1. **`hoi_get_nbest_mult`** on the `.nc` output of any metric: most positive
   and most negative multiplets, with region names.

### 3e. "Does the group's past predict its future?" (time-lagged HOI)
1. **`hoi_dotot`** - dynamic O-information (needs 3D input and a lag `tau`).
2. **`hoi_transfer_entropy`**, **`hoi_redundancy_phiid`**,
   **`hoi_atoms_phiid`** - pairwise lagged information flow and its phiID
   atoms. For trial-resolved directed connectivity prefer Step 2a.

HOI metrics carry no p-values; `n_boots` gives bootstrap CIs (uncertainty),
Step 4 gives significance.

## Step 4 - what type of statistical analysis?

- **`frites_wf_mi` / `frites_wf_conn_comod`**: statistics built in; choose
  `inference`, `mcp`, `n_perm`, `random_state` deliberately.
- **Any other tool** (`frites_conn_*`, `hoi_*`): build a null (shuffle trial
  labels / `y` / one signal's samples, recompute) and give effect +
  permutations to **`frites_wf_stats`**.

### 4a. `inference`
1. **`rfx`** - random effect, t-test across subjects; generalises to the
   population. Needs >= 2 subjects (the tools refuse otherwise).
2. **`ffx`** - fixed effect, pools everything; only speaks about these
   subjects/trials. Use for single-subject or a few iEEG patients.

### 4b. `mcp` (multiple comparisons)
1. **`cluster`** - for effects contiguous in time/frequency; most powerful
   when that holds (`cluster_th=None` auto, or `'tfce'`).
2. **`maxstat`** - no contiguity assumption; safe for one-value-per-multiplet
   HOI outputs; conservative.
3. **`fdr`** - many largely independent tests, some false positives
   tolerable.
4. **`bonferroni`** - few tests, maximal defensibility, lowest power.

### 4c. `tail`
`0` two-tailed (default when no directional prior), `1` increase only,
`-1` decrease only.

### 4d. `n_perm`
Smallest p-value is `1/(n_perm+1)`; 1000 is the usual default, more for
strict corrections.

## After the shortlist

Hand off to `frites-connectivity`, `hoi-metrics` or `workflows`. Theory:
Frites https://brainets.github.io/frites/overview/index.html, HOI
https://brainets.github.io/hoi/.
