# Static HOI metrics

All take `(n_samples, n_features[, n_variables])`, `minsize`, `maxsize`,
`method`, `samples_path`, and the bootstrap options (`n_boots`,
`ci_percentiles`, `random_state`). Sign conventions: see the table in
`SKILL.md`.

## Group-structure metrics (y optional)

- **`hoi_oinfo`** - O-information `Omega = TC - DTC`. The standard
  single-number balance: `> 0` redundancy-dominated, `< 0`
  synergy-dominated. Identically 0 for pairs (start at `minsize=3`). With
  `y_path` you get the task-related variant (multiplets that include `y`).
  Rosas et al. 2019.
  https://brainets.github.io/hoi/api/generated/hoi.metrics.Oinfo.html
- **`hoi_gradient_oinfo`** (y required) - `dOmega_i = Omega(X) - Omega(X
  without i)`: which feature tips the group towards redundancy or synergy.
  Same sign as `oinfo`. Scagliarini et al. 2023.
  https://brainets.github.io/hoi/api/generated/hoi.metrics.GradientOinfo.html
- **`hoi_infotopo`** - topological information `I_k` across every order from
  `minsize` (default 1: entropies) to `maxsize`. Richest and most expensive
  output; same sign convention as `oinfo`. Baudot et al. 2019.
  https://brainets.github.io/hoi/api/generated/hoi.metrics.InfoTopo.html
- **`hoi_tc`** - total correlation `sum H(X_i) - H(X)`: overall departure
  from independence, `>= 0`.
  https://brainets.github.io/hoi/api/generated/hoi.metrics.TC.html
- **`hoi_dtc`** - dual total correlation `H(X) - sum H(X_i | X_-i)`: shared
  information sensitive to both redundancy and synergy, `>= 0`, no sign to
  read. Te Sun 1978.
  https://brainets.github.io/hoi/api/generated/hoi.metrics.DTC.html
- **`hoi_sinfo`** - S-information `TC + DTC`: total strength of dependence,
  the "how much" companion of `oinfo`'s "which kind".
  https://brainets.github.io/hoi/api/generated/hoi.metrics.Sinfo.html

## Target-y metrics (y required): what the group tells about y

- **`hoi_infotot`** - `I(multiplet; y)`, the total information about the
  target (`minsize=1` gives single-feature MI). This is what the next two
  decompose.
  https://brainets.github.io/hoi/api/generated/hoi.metrics.InfoTot.html
- **`hoi_redundancy_mmi`** / **`hoi_synergy_mmi`** - redundancy and synergy
  as two separate quantities under the minimum-mutual-information (MMI)
  assumption: redundancy = `min_i I(X_i; y)`, synergy = `I(X; y) - max over
  sub-multiplets`. Use both when the magnitudes matter, not only their
  balance.
  https://brainets.github.io/hoi/api/generated/hoi.metrics.RedundancyMMI.html
- **`hoi_rsi`** - redundancy-synergy index `I(X; y) - sum_i I(X_i; y)`.
  **Positive = synergy, negative = redundancy** (opposite of `oinfo`).
  Chechik et al. 2001; Timme et al. 2014.
  https://brainets.github.io/hoi/api/generated/hoi.metrics.RSI.html

## Theory

HOI's theoretical background document (PDF in the braina repo):
https://github.com/brainets/braina/blob/main/papers/Theoretical_background_hoi_documentation.pdf
HOI paper: Neri et al. 2024, JOSS -
https://github.com/brainets/braina/blob/main/papers/Neri_joss_2024_high_order_interactions_toolbox.pdf
Application to MEG: Combrisson et al. 2025, Nat Commun -
https://github.com/brainets/braina/blob/main/papers/Combrisson_natcomms_2025_higher_order_interactions_synergy_meg.pdf
(In a repo clone these are under `papers/`, readable with `read_pdf`.)
