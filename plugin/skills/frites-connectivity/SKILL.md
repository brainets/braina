---
name: frites-connectivity
description: Use once a Frites connectivity tool has been chosen (covgc, dfc, pid, ii, te, fit, spec, ccf, or the reshape/net post-processing) and the question is about how it actually works, what a parameter does, how to interpret its output, or which parameter values to pick. Grounded in the real Frites source (frites/conn, frites/core, frites/estimator), not just the MCP wrapper docstrings. Not for choosing which tool to use in the first place - see the orientation skill for that.
---

# Frites connectivity measures

Covers the 8 pairwise connectivity tools braina exposes -
`frites_conn_covgc`, `frites_conn_dfc`, `frites_conn_pid`, `frites_conn_ii`,
`frites_conn_te`, `frites_conn_fit`, `frites_conn_spec`, `frites_conn_ccf` -
and the two post-processing tools `frites_conn_reshape` and `frites_conn_net`.

Per-tool detail (every parameter, output layout, interpretation, references)
lives in `references/` next to this file
(`${CLAUDE_PLUGIN_ROOT}/skills/frites-connectivity/references/`). **Read the
matching reference file before answering a parameter-level question**; this
page only holds what applies to all of them.

| Tool | Question it answers | Reference file |
|---|---|---|
| `frites_conn_covgc` | Does A drive B (directed, time-resolved, per trial)? | `references/covgc.md` |
| `frites_conn_te` | Directed influence scanned over a range of delays | `references/te-fit.md` |
| `frites_conn_fit` | How much information *about feature y* flows A->B | `references/te-fit.md` |
| `frites_conn_dfc` | Undirected coupling over (sliding) windows, per trial | `references/dfc.md` |
| `frites_conn_spec` | Coherence / PLV / cross-spectrum per frequency and time | `references/spec-ccf.md` |
| `frites_conn_ccf` | Raw temporal lag between two signals | `references/spec-ccf.md` |
| `frites_conn_pid` | Unique / redundant / synergistic information of a pair about y | `references/pid-ii.md` |
| `frites_conn_ii` | Net synergy-minus-redundancy of a pair about y | `references/pid-ii.md` |
| `frites_conn_reshape`, `frites_conn_net`, `plot_result` | Turning pair-wise outputs into matrices, net flow, figures | `references/post-processing.md` |

## Data conventions (apply to every tool)

- Input `(n_epochs, n_roi, n_times)`. Use a `.nc` file whose coordinates are
  named exactly `roi` and `times` and whose `attrs['sfreq']` is set: the
  wrappers pass those names to Frites, so ROI labels and a time axis in
  seconds survive into the output. A `.npy` input (or other coordinate
  names) gets `roi_0, roi_1, ...` and a 1 Hz axis instead - use
  `convert_to_nc` first.
- Pair labels in outputs: undirected tools give `A-B`, directed tools give
  `A->B` (covgc gives `A-B` plus a `direction` coordinate `x->y`, `y->x`,
  `x.y`).
- Every tool returns a text summary (shape, dims, coords, min/mean/max) of
  the file it wrote; use it before deciding whether to `inspect_data`.
- `n_jobs` (parallel pairs) is exposed on covgc, dfc, te, spec and ccf;
  default 1.

## The shared estimator core

Five tools - `covgc` (with `method='gc'`), `te`, `ii`, `pid` and `fit` - are
one estimator fed different samples: the data are rank/copula-normalised
(`frites.core.copnorm_nd`) and a Gaussian-Copula (conditional) mutual
information is computed (`frites.core.gcmi_nd`). What differs is *which*
samples enter the CMI (past/future of X and Y for Granger causality, X/Y/y
for interaction information, ...).

- Shared assumption: after a monotonic rank transform, the dependency is
  well described by a Gaussian copula. More flexible than assuming linear /
  Gaussian raw data (`covgc method='gauss'`), but **not** fully
  non-parametric - a nonlinearity that is not a monotonic transform of a
  Gaussian dependency is still missed. State this when reporting "transfer
  entropy" results.
- `dfc` goes through the `frites.estimator` abstraction instead; the wrapper
  exposes `estimator='gcmi' | 'pearson' | 'spearman' | 'dcorr' | 'binmi'`.
- `spec` and `ccf` use no information theory (wavelet/multitaper
  time-frequency decomposition; classical cross-correlation).

## Statistics

None of these tools returns p-values. For group inference feed their output
(effect + a permutation null) to `frites_wf_stats`, or use `frites_wf_mi` /
`frites_wf_conn_comod` which compute effect and statistics together - see the
`workflows` skill.

## Examples

Validated example scripts, relative to `${CLAUDE_PLUGIN_ROOT}`:
`examples/frites/conn/plot_covgc.py`, `plot_dfc.py`, `plot_pid.py`,
`plot_ii.py`, `plot_fit.py`, `plot_ccf.py`, `plot_conn.py`. No dedicated
example exists for `te` and `spec`; rely on the reference files and the
tool docstrings rather than improvising a call.
