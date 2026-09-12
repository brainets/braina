---
description: Use once a Frites connectivity tool has been chosen (covgc, dfc, pid, ii, te, fit, spec, ccf) and the question is about how it actually works, what a parameter does, how to interpret its output, or which parameter values to pick. Grounded in the real Frites source (frites/conn, frites/core), not just the MCP wrapper docstrings. Not for choosing which tool to use in the first place — see the orientation skill for that.
tags: [frites, connectivity, granger-causality, mutual-information, transfer-entropy, pid]
---

# Frites connectivity measures

Covers the 8 pairwise connectivity tools braina exposes:
`frites_conn_covgc`, `frites_conn_dfc`, `frites_conn_pid`, `frites_conn_ii`,
`frites_conn_te`, `frites_conn_fit`, `frites_conn_spec`, `frites_conn_ccf`.

All data conventions from `CLAUDE.md` apply: `(n_epochs, n_roi, n_times)`
input, `.nc` when metadata (ROI names, `sfreq`) needs to survive.

## The shared estimator core

Five of the eight tools — **`covgc`** (when `method='gc'`), **`te`**,
**`ii`**, **`pid`**, and **`fit`** — are not five different estimators.
They all reduce to the *same* computation: rank/copula-normalize the data
(`frites.core.copnorm_nd`), then compute Gaussian-Copula (Conditional)
Mutual Information (`frites.core.gcmi_nd.cmi_nd_ggg` / `mi_nd_gg`). What
differs between them is **which samples get fed into that CMI** — past vs.
future of X and Y for Granger causality, X/Y/stimulus for interaction
information, and so on — not the underlying estimator.

Practical implications of this:
- They share the same core assumption: after copula-normalizing (a
  monotonic rank transform), the *dependency structure* is well described
  by a Gaussian copula. This is more flexible than assuming the raw data
  is linearly/Gaussian related (that's what `covgc`'s `method='gauss'`
  does instead), but it is **not** a fully assumption-free/nonparametric
  estimate — a real nonlinearity that isn't captured by a monotonic
  transform plus Gaussian dependency will still be missed.
- **`dfc`** is the outlier among the information-theoretic tools: it
  doesn't call `frites.core` directly, it goes through a `GCMIEstimator`
  abstraction (`frites/estimator/`) that defaults to the same GCMI
  computation but is designed to accept other estimator types (not
  exposed via the MCP wrapper — always GCMI here).
- **`spec`** and **`ccf`** don't use information theory at all — `spec`
  does wavelet/multitaper time-frequency decomposition then coherence/PLV/
  cross-spectrum (`frites/conn/conn_tf.py`), and `ccf` does classical
  autocorrelation (`frites/utils/preproc._acf`). Reach for these when the
  question is about oscillatory synchrony or raw temporal lag, not
  information sharing.

## `frites_conn_covgc` — covariance-based Granger causality

Directed, time-resolved influence between X and Y, computed per trial.
Returns `(n_epochs, n_pairs, n_windows, 3)`: index `0` = X→Y, `1` = Y→X,
`2` = instantaneous coupling (X·Y).

- `dt` — window length (samples) over which the covariance is estimated.
- `lag` — how far into the past to look (samples).
- `t0` — one GC estimate is produced per entry (window centers/starts).
- `method` — `'gauss'` assumes the raw time-points are linearly/Gaussian
  related (classic linear Granger causality); `'gc'` copula-normalizes
  first (see shared core above), more robust to monotonic nonlinearities.
- `conditional` — if `True`, conditions each direction's estimate on the
  past of *other* sources too (multivariate GC), not just the pair.

Reference: Brovelli et al. 2015 (`papers/Brovelli_jneurosci_2015_...`).
Not exposed by the MCP wrapper: `norm` (normalized GC), `gcrn`, `n_jobs`
(hardcoded to 1 — no parallelism across pairs via this tool).
API reference: https://brainets.github.io/frites/api/generated/frites.conn.conn_covgc.html

## `frites_conn_dfc` — dynamic functional connectivity

Undirected, time-resolved coupling between ROI pairs, by default via GCMI
(see shared core above), optionally over sliding windows.

- `win_sample` — `[[start, stop], ...]` windows (build with
  `frites.conn.define_windows` if scripting directly against Frites,
  outside the MCP tool).
- `agg_ch` — if a region has multiple channels/contacts, `True` aggregates
  them into one multivariate estimate per region pair instead of
  computing every channel-pair separately.

Not exposed: `estimator` (always defaults to GCMI through this tool).
API reference: https://brainets.github.io/frites/api/generated/frites.conn.conn_dfc.html

## `frites_conn_pid` — partial information decomposition

For each pair (X, Y) and a target `y`, decomposes the information the
pair carries about `y` into: total information, each node's **unique**
contribution, **redundant** information (both carry the same info), and
**synergistic** information (only available jointly). Returns four files
(`_infotot`, `_unique`, `_redundancy`, `_synergy`).

- `mi_type` — `'cc'` if `y` is a continuous regressor, `'cd'` if `y` is
  categorical.

Reference: Williams & Beer 2010. Not exposed: `gcrn` (library default
`True`), `dt` (library default `1`).
API reference: https://brainets.github.io/frites/api/generated/frites.conn.conn_pid.html

## `frites_conn_ii` — interaction information

A cheaper single-number relative of PID: `II = I([X,Y]; S) − I(X;S) −
I(Y;S)`. **Positive → synergy-dominated, negative → redundancy-dominated.**
Use this instead of `pid` when only the net balance is needed, not the
full four-way breakdown.

- `mi_type`, `dt` — same meaning as in `pid`.

Reference: McGill 1954.
API reference: https://brainets.github.io/frites/api/generated/frites.conn.conn_ii.html

## `frites_conn_te` — transfer entropy

Across-trial transfer entropy from each source to each target, scanning a
range of delays. As transfer entropy, conceptually it doesn't assume
linearity the way `covgc(method='gauss')` does — but note the estimator
used here is the same Gaussian-Copula CMI described above, so it inherits
that estimator's Gaussian-copula-dependency assumption rather than being
a fully nonparametric (e.g. binning/kNN-based) TE estimate.

- `min_delay`, `max_delay`, `step_delay` — the range and resolution of
  delays scanned (samples). The wrapper averages across delays (matches
  the library default `return_delays=False`).

Not exposed: `sfreq` (only meaningful with `return_delays`, not used
here), `gcrn` (library default `True`).
API reference: https://brainets.github.io/frites/api/generated/frites.conn.conn_te.html

## `frites_conn_fit` — feature-specific information transfer

Like `te`, but asks specifically how much information *about feature y*
is transferred from source to target, rather than raw directed influence.
Same GCMI/copula estimator core.

- `mi_type` — `'cc'`/`'cd'` for `y`.
- `max_delay` — in seconds if the data carries an `sfreq` attribute
  (the wrapper reads `data.attrs['sfreq']` automatically for `.nc`
  input), otherwise in samples.
- `net` — if `True`, computes the net transfer (subtracting the
  reverse-direction contribution).

Reference: Celotto et al. 2023 (`papers/Celotto_neurips_2023_...`).
API reference: https://brainets.github.io/frites/api/generated/frites.conn.conn_fit.html

## `frites_conn_spec` — spectral connectivity

Wavelet- or multitaper-based time-frequency decomposition, then a
per-frequency, per-time coupling metric. No information theory involved.

- `freqs` — required, central frequencies of interest.
- `metric` — `'coh'` (coherence), `'plv'` (phase-locking value), `'sxy'`
  (cross-spectrum).
- `mode` — `'morlet'` (wavelets) or `'multitaper'`.
- `sm_times`/`sm_freqs`, `n_cycles` — smoothing / time-frequency
  trade-off knobs; more smoothing trades time/frequency resolution for a
  more stable estimate.

Needs a sampling frequency: pass `.nc` input with `attrs['sfreq']` set —
`.npy` input has no `sfreq` metadata and the underlying decomposition will
fail without it.
API reference: https://brainets.github.io/frites/api/generated/frites.conn.conn_spec.html

## `frites_conn_ccf` — cross-correlation function

Full-lag single-trial cross-correlation between ROI pairs — the cheapest
way to find a temporal lag between two time series without any
frequency-domain assumptions. No information theory, no smoothing knobs,
no parameters beyond the data itself: it always returns the full lag
range (there is no cropping parameter — if only a narrow lag window
matters, crop the output afterward).

Interpretation: peak at a **negative** lag means the target should be
shifted *toward* the source (source leads); peak at a **positive** lag
means the target leads.
API reference: https://brainets.github.io/frites/api/generated/frites.conn.conn_ccf.html

## Examples

See the `orientation` skill's Step 5 table for the validated example
script per tool (`${CLAUDE_PLUGIN_ROOT}/examples/frites/conn/...`) —
`te` and `spec` currently have no dedicated example; rely on this skill
and the MCP tool's docstring instead of improvising a call.
