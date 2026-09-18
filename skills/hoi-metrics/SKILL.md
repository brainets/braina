---
description: Use once a HOI (higher-order interactions) tool has been chosen (oinfo, gradient_oinfo, infotopo, redundancy_mmi, synergy_mmi, rsi, dtc, get_nbest_mult) and the question is about how it actually works, what a parameter does, how to interpret its sign, or which parameter values to pick. Grounded in the real HOI source (hoi/metrics, hoi/core, hoi/utils), not just the MCP wrapper docstrings. Not for choosing which tool to use in the first place — see the orientation skill for that.
tags: [hoi, higher-order-interactions, o-information, synergy, redundancy, dtc]
---

# HOI metrics

Covers the 8 higher-order-interaction tools braina exposes:
`hoi_oinfo`, `hoi_gradient_oinfo`, `hoi_infotopo`, `hoi_redundancy_mmi`,
`hoi_synergy_mmi`, `hoi_rsi`, `hoi_dtc`, `hoi_get_nbest_mult`.

Input convention: `(n_samples, n_features)` or `(n_samples, n_features,
n_variables)` — this is a joint analysis over 3+ features at once
(multiplets), not pairwise like Frites connectivity.

## The shared estimator core

Every HOI metric's `fit()` takes a `method` argument selecting how
entropy is estimated (`hoi/core/entropies.py`:
`'gc' | 'gauss' | 'binning' | 'histogram' | 'knn' | 'kernel'`), and it
defaults to **`'gc'`** (Gaussian Copula) everywhere. The MCP wrapper never
exposes `method`, so every `hoi_*` tool in braina always runs the
Gaussian-Copula estimator — the same family of assumption used by
`covgc`/`te`/`ii`/`pid`/`fit` in `frites-connectivity`: robust to monotonic
nonlinearities after rank-normalization, but still assuming the
dependency structure is well described by a Gaussian copula, not fully
nonparametric.

## Compute backend: HOI runs on CPU by default in this repo

HOI is built on JAX, which can run on CPU or GPU — but **braina's own
dependency declarations don't ask for GPU support**. Both
`check_env.py` and `mcp/braina_mcp.py` pin plain `"jax"`/`"jaxlib"` with
no CUDA extra, and that's what `uv run` installs. Plain `jax`/`jaxlib`
from PyPI are CPU-only wheels, so every `hoi_*` tool runs on CPU today
regardless of what hardware is available — this isn't a JAX limitation,
it's simply not requested by braina's dependency list.

- **Check what's actually active**: `python -c "import jax;
  print(jax.devices())"`. `[CpuDevice(id=0)]` means CPU; a `Gpu`/`Cuda`
  device listed means GPU is active.
- **To actually use a GPU**, the inline PEP 723 dependency block needs
  the CUDA extra matching the machine's CUDA version, e.g.
  `"jax[cuda12]"` instead of plain `"jax"` — see
  https://jax.readthedocs.io/en/latest/installation.html for the exact
  extra to use (it depends on the local CUDA version, so don't guess a
  version without checking the machine first). HOI's own install docs
  point to the same JAX page rather than giving GPU-specific commands:
  https://brainets.github.io/hoi/install.html.
- **When it matters**: GPU acceleration helps most on large problems —
  many features and/or a large `maxsize` (the combinatorial explosion in
  multiplet count is where JAX's vectorization pays off). For a handful
  of ROIs with `maxsize` capped low, CPU is often fast enough that
  switching isn't worth the setup effort.
- **If running on a shared cluster**: JAX/XLA preallocates most of a
  GPU's memory by default on first use, which can starve other jobs on
  the same GPU. Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` (or
  `XLA_PYTHON_CLIENT_MEM_FRACTION` to cap it) before running if sharing
  a GPU with other processes.

## ⚠️ Sign conventions are not consistent across metrics

This is the single most common source of misinterpretation. **Check which
convention applies before reporting a result as "synergy" or
"redundancy":**

| Metric | Positive means | Negative means |
|---|---|---|
| `oinfo` | Redundancy | Synergy |
| `gradient_oinfo` | Redundancy | Synergy |
| `infotopo` | Redundancy | Synergy |
| `rsi` | **Synergy** | **Redundancy** |
| `redundancy_mmi`, `synergy_mmi` | n/a — each is already a one-directional quantity, not a signed balance | |
| `dtc` | n/a — magnitude of total shared (redundant *and* synergistic) information, not a signed balance | |

`rsi`'s convention is the **opposite** of `oinfo`/`gradient_oinfo`/
`infotopo`. Flipping this sign by habit when moving between metrics is
the most likely mistake here — call it out explicitly whenever comparing
results across metrics.

## `hoi_oinfo` — O-information

`Ω(Xⁿ) = TC(Xⁿ) − DTC(Xⁿ)`. The standard single-number summary of whether
a group of variables is net redundant (`Ω > 0`) or net synergistic
(`Ω < 0`). `y` is optional — pass it to get the *task-related* O-info
(how the group's redundancy/synergy balance relates to an external
variable) instead of the group's unconditional structure.

Reference: Rosas et al. 2019.
API reference: https://brainets.github.io/hoi/api/generated/hoi.metrics.Oinfo.html

## `hoi_gradient_oinfo` — first-order gradient of O-information

`∂ᵢΩ(Xⁿ) = Ω(Xⁿ) − Ω(Xⁿ₋ᵢ)` — how much the O-information changes when
variable `i` is added to (or removed from) the group. Same sign
convention as `oinfo`. Use this to find *which* region is responsible for
tipping a group from redundant to synergistic (or vice versa), rather
than just knowing the group's net balance. `y` is required here (the
gradient is computed with respect to the task-related O-info).

Reference: Scagliarini et al. 2023.
API reference: https://brainets.github.io/hoi/api/generated/hoi.metrics.GradientOinfo.html

## `hoi_infotopo` — topological information

`Iₖ(X₁;...;Xₖ)`, the multivariate mutual information computed across
every interaction order from `minsize` to `maxsize` at once — the richest
output of the three (no single "net balance" number, a decomposition
across orders). Same sign convention as `oinfo`. No `y` parameter exposed
via this tool. Cost grows combinatorially with `maxsize`.

Reference: Baudot et al. 2019; Tapia et al. 2018.
API reference: https://brainets.github.io/hoi/api/generated/hoi.metrics.InfoTopo.html

## `hoi_redundancy_mmi` / `hoi_synergy_mmi` — explicit redundancy and synergy

Minimum-Mutual-Information-based estimates of redundancy and synergy as
**two separate, non-negative-in-spirit quantities**, rather than a single
signed difference. Use both together when the two numbers matter
individually — `oinfo`/`rsi` only give their difference (net balance),
not their individual magnitudes. `y` is required for both.

API reference: https://brainets.github.io/hoi/api/generated/hoi.metrics.RedundancyMMI.html,
https://brainets.github.io/hoi/api/generated/hoi.metrics.SynergyMMI.html

## `hoi_rsi` — Redundancy-Synergy Index

`RSI(S;Y) = I(S;Y) − Σᵢ I(xᵢ;Y)`: the information the whole set `S`
carries about `Y` beyond what its individual elements carry on their own.
**Sign convention is the opposite of `oinfo`**: positive → synergy,
negative → redundancy (see the table above). `y` is required.

References: Chechik et al. 2001; Timme et al. 2014.
API reference: https://brainets.github.io/hoi/api/generated/hoi.metrics.RSI.html

## `hoi_dtc` — Dual Total Correlation

`DTC(Xⁿ) = H(Xⁿ) − Σⱼ H(Xⱼ|X₋ⱼⁿ)`. A total-correlation-style measure of
information shared across the whole group — sensitive to **both**
redundancy and synergy without separating them, so don't read a sign into
it the way you would `oinfo`. `y` is optional (task-related variant, as
in `oinfo`).

Reference: Te Sun 1978.
API reference: https://brainets.github.io/hoi/api/generated/hoi.metrics.DTC.html

## `hoi_get_nbest_mult` — ranking multiplets and naming the regions

`hoi.utils.get_nbest_mult()` needs, besides the values, the multiplet
metadata (`model.multiplets`, `model.order`) that only exists on the fitted
model in memory. Every `hoi_*` wrapper therefore writes that metadata into
its `.nc` output as coordinates: `multiplets` (feature indices, e.g.
`"0,2,3"`), `order` (multiplet size) and, when the input `.nc` had a
coordinate on its feature dimension (e.g. `roi`), `multiplet_names` (e.g.
`"A / C / D"`). `hoi_get_nbest_mult` reads those coordinates back and calls
the real `get_nbest_mult` with `orders=`, `multiplets=` and `names=`, so the
CSV it writes has the columns `index, order, hoi, multiplet, names`.

How the ranking works (this is HOI's own convention, not braina's):
- It returns the **`n_best` most positive values and the `n_best` most
  negative values**, positive first — up to `2 * n_best` rows. It is *not*
  a ranking by absolute value.
- Whether "positive" means redundancy or synergy depends on the metric
  (see the sign-convention section above): for `oinfo` positive =
  redundancy, for `rsi` positive = synergy. Always state which half of the
  table is which.
- `minsize` / `maxsize` filter by order before ranking (e.g. `minsize=3` to
  ignore pairs; note that O-information is identically 0 for pairs).

Gotchas:
- `.npy` outputs drop the coordinates; the tool refuses them with an
  explicit error. Re-run the `hoi_*` tool with a `.nc` `output_path`.
- Region names only appear if the HOI input was a `.nc` with a coordinate
  on its feature (second) dimension. With `.npy` input the `multiplet`
  column still gives the feature indices, in the input's column order.
- HOI does not accept xarray directly; the wrappers convert `.nc` input to
  numpy and keep the names themselves.

API reference: https://brainets.github.io/hoi/api/generated/hoi.utils.get_nbest_mult.html

## Examples

See the `orientation` skill's Step 5 table for the validated example
script per tool (`${CLAUDE_PLUGIN_ROOT}/examples/hoi/metrics/...`) —
`dtc` and `gradient_oinfo` currently have no dedicated example; rely on
this skill and the MCP tool's docstring instead of improvising a call.
