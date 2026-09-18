# `frites_conn_covgc` - covariance-based Granger causality

Directed, time-resolved influence between X and Y, computed **per trial**
(`frites.conn.conn_covgc`). Output `.nc` dims: `(trials, roi, times,
direction)` with `roi` = pairs `A-B`, `times` = one value per `t0` window and
`direction` = `x->y`, `y->x`, `x.y` (instantaneous coupling).

## Parameters

- `dt` - window length in samples over which the covariance is estimated.
  Longer = more stable, less time-resolved.
- `lag` - model order: how many past samples enter the prediction.
- `step` - spacing (samples) between the `lag` past samples (1 = consecutive).
- `t0` - list of window starts in samples; one GC estimate per entry. Build
  it e.g. as `list(range(start, stop, step_in_samples))`.
- `method` - `'gc'` (default) copula-normalises the data first, robust to
  monotonic nonlinearities; `'gauss'` assumes the raw data are Gaussian /
  linearly related (classical linear Granger causality).
- `conditional` - `True` conditions each direction on the past of *all other*
  ROIs (multivariate / conditional GC), which removes indirect influences via
  a third region. Needs more data; slower.
- `norm` - `True` returns normalised GC (fraction of predictable variance,
  bounded), easier to compare across pairs.
- `n_jobs` - parallel pairs.

## Interpretation

- Compare `x->y` against `y->x` at each window; `frites_conn_net` gives the
  difference directly (positive = X drives Y more than Y drives X).
- `x.y` is instantaneous (zero-lag) coupling, often shared input or volume
  conduction; report it separately, not as "causality".
- Trial-level output means you can build a null by shuffling trials or use
  `frites_wf_stats` with an rfx t-test across subjects on the trial mean.
- Window centre in time = `t0 + dt/2` (already converted to seconds in the
  `times` coordinate when the input carried `times`).

## References

- Brovelli et al. 2004, PNAS, "Beta oscillations in a large-scale
  sensorimotor cortical network: directional influences revealed by Granger
  causality" - PDF in the braina repo:
  https://github.com/brainets/braina/blob/main/papers/Brovelli_pnas_2004_Granger_causality_LFP.pdf
- Brovelli et al. 2015, J Neurosci, covariance-based GC used here - PDF:
  https://github.com/brainets/braina/blob/main/papers/Brovelli_jneurosci_2015_covariance_Granger_causality.pdf
- Ding et al. 2006, Granger causality review - PDF:
  https://github.com/brainets/braina/blob/main/papers/Ding_2006_Granger_causality.pdf
- API: https://brainets.github.io/frites/api/generated/frites.conn.conn_covgc.html

In a clone of the repository these PDFs are under `papers/` and can be read
with the `read_pdf` tool; the installed plugin does not ship them.
