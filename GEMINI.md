# Project: braina (Brain Interaction Analysis)

## 1. Project Context & Purpose
- **Goal:** Analyzing complex neural interactions using Frites, HOI, and XGI toolboxes.
- **Focus:** Information Theoretical Analysis of electrophysiological data (fMRI, MEG, EEG, LFP, MUA).
- **Organization:** Institut de Neurosciences de la Timone (BraiNets), Marseille, France.
- **Novice Mode:** The user may be a novice in this domain. Explain mathematical concepts (entropy, mutual information, O-information, Granger causality) simply and verify all proposed code before recommending it.

## 2. Commands

```bash
# Verify environment (all core dependencies)
uv run check_env.py

# Run the verification test suite for Frites + HOI
uv run mcp/verify_libs.py

# Launch the MCP server standalone
uv run mcp/braina_mcp.py

# Run any example script (all use uv inline dependencies)
uv run examples/frites/conn/ex_conn_covgc.py
uv run examples/hoi/metrics/ex_oinfo.py
```

All scripts use `uv` with PEP 723 inline script metadata (`# /// script` blocks) for dependency management.

## 3. Project Structure
- **`/mcp`**: MCP server (`braina_mcp.py`) exposing 30+ tools for Frites/HOI analysis.
- **`/examples`**: ~50 Python example scripts organized by library:
  - `examples/frites/` — AR models, connectivity, mutual information, statistics, simulations.
  - `examples/hoi/` — Information theory, HOI metrics, tutorials, statistics.
- **`/papers`**: Research papers forming the theoretical foundation.
- **`/tutorials`**: Hands-on learning materials:
  - `multivariate_information_theory_frites_hoi_xgi/` — Integrating frites, hoi, and xgi.
  - `seeg_ebrains_frites/` — SEEG data analysis with frites.
- **`/usecases`**: Real-world analysis scenarios.

## 4. MCP Server Tools (`mcp/braina_mcp.py`)

Tool categories:
- **Data I/O**: `inspect_data`, `read_pdf`, `load_data`, `save_data`
- **Frites connectivity**: `frites_conn_covgc`, `frites_conn_dfc`, `frites_conn_pid`, `frites_conn_ii`, `frites_conn_te`, `frites_conn_fit`, `frites_conn_spec`, `frites_conn_ccf`
- **Frites workflows**: `frites_wf_stats` (WfStats), `frites_wf_mi` (WfMi), `frites_wf_conn_comod` (WfConnComod)
- **Frites simulation**: `frites_sim_ar` (StimSpecAR)
- **HOI metrics**: `hoi_oinfo`, `hoi_gradient_oinfo`, `hoi_infotopo`, `hoi_redundancy_mmi`, `hoi_synergy_mmi`, `hoi_rsi`, `hoi_dtc`, `hoi_get_nbest_mult`

## 5. Core Libraries
- **Frites** (v0.4.0+): Single-trial functional connectivity. Key modules: `frites.conn`, `frites.simulations.StimSpecAR`, `frites.workflow` (WfMi, WfStats, WfConnComod).
  - GitHub: `brainets/frites`
- **HOI** (v0.2.0+): Higher-Order Interactions analysis. Key metrics: Oinfo, GradientOinfo, InfoTopo, RedundancyMMI, SynergyMMI, RSI, DTC.
  - GitHub: `brainets/hoi`
- **XGI** (v0.7.0+): Hypergraph and higher-order network analysis.
  - GitHub: `xgi-org/xgi`

## 6. Source of Truth & Expertise
- **Examples first:** Always consult `/examples` for usage patterns before writing new code.
- **MCP tools as reference:** Read `mcp/braina_mcp.py` for canonical function signatures and parameter conventions.
- **External Documentation:** Use the `context7` tool to prioritize official API patterns for Frites and HOI.
- **Source Code:** Use the `github` MCP tool to read `brainets/frites`, `brainets/hoi`, and `xgi-org/xgi` for specific implementation details.
- **Research:** Use `Google Search` to find the latest JAX or neuroimaging papers to suggest novel solutions.

## 7. Key Data Conventions
- Frites connectivity input: `(n_epochs, n_roi, n_times)` numpy array or xarray DataArray.
- HOI input: `(n_samples, n_features)` or `(n_samples, n_features, n_variables)`.
- Use `.npy` for raw arrays, `.nc` (NetCDF/xarray) when metadata (ROI names, time coords, `sfreq`) must be preserved.
- Sampling frequency stored in `xarray.DataArray.attrs['sfreq']`.
- **Numpy must be `<2.0`** (required by current Frites/HOI versions).

## 8. Reliability & Testing Rules
- **Verification First:** Never suggest code without explaining *why* it matches the toolbox documentation.
- **Test before recommending:** When writing analysis scripts, create small dummy data (numpy/jax) and verify the function runs without errors before presenting to the user.
- **Environment:** Use `uv` for dependency management when writing standalone scripts.

## 9. Coding Style
- Follow PEP 8 guidelines.
- Prefer **JAX** for high-performance math components in the `HOI` sub-modules.
- Use `xarray` or `MNE` structures when dealing with Frites-style connectivity.
- Keep scripts self-contained with clear imports and docstrings.
