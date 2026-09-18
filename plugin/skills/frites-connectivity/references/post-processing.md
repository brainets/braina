# Post-processing connectivity outputs

## `frites_conn_reshape` - pairs to matrix

Turns the raveled `roi` dimension (`A-B` or `A->B` labels) into
`(sources, targets, ...)`, the layout needed for a connectivity matrix plot
or for exporting to a graph library (e.g. XGI / networkx).

- `directed=True` for covgc, te, fit (covgc's `direction` coordinate is
  ravelled into `A->B` / `B->A` first); `directed=False` for dfc, ccf, ii,
  PID components, comod outputs.
- `net=True` (directed only) fills the matrix with `A->B - B->A`.
- `mean_trials=True` averages the `trials` dimension first (default);
  `fill_diagonal` sets the diagonal (NaN by default).

## `frites_conn_net` - net directed flow

For every pair returns `A->B - B->A`, keeping `times` (and `trials` unless
`mean_trials=True`). Positive = A drives B more than the reverse. Output
`roi` labels are undirected pairs `A-B`; the attributes `net_source` /
`net_target` record which direction is positive.

## `plot_result` - look at it

`plot_result(data_path, output_path.png)` draws lines over time per pair
(splitting covgc's `direction`), time-frequency panels for `spec` output, a
`sources x targets` image for reshaped matrices, or bars for HOI multiplets.
`roi=[...]` selects pairs, `mean_dims=[...]` averages extra dimensions. Read
the PNG afterwards to check the result before interpreting numbers.

## Group statistics

`frites_wf_stats` expects `(n_roi, n_subjects, n_times)` effects and
`(n_perm, n_roi, n_subjects, n_times)` permutations; the `workflows` skill
explains how to build those from single-subject connectivity outputs.
