# Dynamic (time-lagged) HOI metrics

Input **must be 3D** `(n_samples, n_features, n_times)`: e.g. trials x ROIs
x time. The wrapper refuses 2D input. `tau` is the lag in samples along the
last axis; the output `variables` dimension is the time axis (length
`n_times`, as returned by HOI). These metrics have no bootstrap option in
braina.

- **`hoi_transfer_entropy`** - pairwise `TE(i->j) = I(X_j,t ; X_i,t-tau |
  X_j,t-tau)`, `>= 0`, ordered pairs `i,j` (both directions). HOI's version;
  for the trial-resolved Frites estimator with delay scanning use
  `frites_conn_te`.
  https://brainets.github.io/hoi/api/generated/hoi.metrics.TransferEntropy.html
- **`hoi_dotot`** - total dynamic O-information (Stramaglia et al. 2021):
  the lagged counterpart of `oinfo` for multiplets of size >= 3 (`minsize`
  default 3). Positive = redundant dynamics, negative = synergistic
  dynamics.
  https://brainets.github.io/hoi/api/generated/hoi.metrics.DOtot.html
- **`hoi_redundancy_phiid`** - redundancy atom of the Integrated Information
  Decomposition (phiID, Mediano et al.): information about the multiplet's
  future carried redundantly by each member's past. `>= 0`.
  https://brainets.github.io/hoi/api/generated/hoi.metrics.RedundancyphiID.html
- **`hoi_atoms_phiid`** - pairwise phiID atoms; default `atoms=['sts']`
  (synergy in the past -> synergy in the future). Other atoms: `'rtr'`
  (redundancy -> redundancy), `'xtx'`, `'yty'` (self-transfer), etc. as
  listed in the HOI API.
  https://brainets.github.io/hoi/api/generated/hoi.metrics.AtomsPhiID.html

Example: `${CLAUDE_PLUGIN_ROOT}/examples/hoi/metrics/plot_syn_phiID.py`.
