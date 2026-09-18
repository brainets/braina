# `frites_conn_te` and `frites_conn_fit` - directed, delay-scanning measures

Both are computed **across trials** (not per trial), for every directed pair
`A->B`, and use the shared Gaussian-copula CMI estimator. Output dims
`(roi, times)` with `roi` = `A->B` labels.

## `frites_conn_te` - transfer entropy

`TE(X->Y) = I(Y_t ; X_{t-d} | Y_{t-d})`, averaged over the scanned delays.

- `min_delay`, `max_delay`, `step_delay` - delay range and resolution in
  **samples**. The output time axis starts at `max_delay` (earlier points
  cannot be conditioned).
- `n_jobs` - parallel pairs.
- `sfreq` from `attrs['sfreq']` is forwarded automatically so the time axis
  is in seconds.

Despite transfer entropy being a model-free concept, this estimate inherits
the Gaussian-copula assumption; it is not a binning/kNN TE. Compared with
`covgc` it needs no `dt`/`lag` choice and no per-trial windows, but it is
noisier and needs many trials.

API: https://brainets.github.io/frites/api/generated/frites.conn.conn_te.html

## `frites_conn_fit` - feature-specific information transfer

How much information **about the feature y** is transferred from A to B:
`FIT` isolates the part of `TE` that concerns y (Celotto et al. 2023).

- `mi_type` - `'cc'` / `'cd'` for y.
- `max_delay` - in **seconds** when the input carries `sfreq` (the wrapper
  reads `attrs['sfreq']`), otherwise in samples.
- `net` - `True` returns `FIT(A->B) - FIT(B->A)`.

Reference: Celotto et al. 2023, NeurIPS, "An information-theoretic
quantification of the content of communication between brain regions" - PDF:
https://github.com/brainets/braina/blob/main/papers/Celotto_neurips_2023_Feature_Specific_Information_Transfer.pdf
API: https://brainets.github.io/frites/api/generated/frites.conn.conn_fit.html

## After the fact

`frites_conn_net` turns `A->B` / `B->A` into a net flow; `frites_conn_reshape`
(`directed=True`) gives a `sources x targets x times` matrix.
