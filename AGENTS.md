# Project: braina (Brain Interaction Analysis)

## 1. Project Context & Purpose
- **Goal:** Analyzing complex neural interactions using Frites, HOI, and XGI toolboxes.
- **Focus:** Information Theoretical Analysis of electrophysiological data (fMRI, MEG, EEG, LFP, MUA).
- **Organization:** Institut de Neurosciences de la Timone (BraiNets), Marseille, France.
- **Novice Mode:** The user may be a novice in this domain. Explain mathematical concepts (entropy, mutual information, O-information, Granger causality) simply and verify all proposed code before recommending it.

## 2. Project Structure
- **`/mcp`**: MCP server (`braina_mcp.py`) exposing 30+ tools for Frites/HOI analysis. Registered as an MCP server for Codex.
- **`/examples`**: ~50 Python example scripts organized by library:
  - `examples/frites/` — AR models, connectivity, mutual information, statistics, simulations.
  - `examples/hoi/` — Information theory, HOI metrics, tutorials, statistics.
- **`/papers`**: Research papers forming the theoretical foundation.
- **`/tutorials`**: Hands-on learning materials:
  - `multivariate_information_theory_frites_hoi_xgi/` — Integrating frites, hoi, and xgi.
  - `seeg_ebrains_frites/` — SEEG data analysis with frites.
- **`/usecases`**: Real-world analysis scenarios (BrainHack 2026, Granger causality, master's thesis).

## 3. Core Libraries
- **Frites** (v0.4.0+): Single-trial functional connectivity and information-theoretical analysis.
  - Key modules: `frites.conn` (covgc, dfc, pid, ii, te, fit, ccf, spec), `frites.simulations` (StimSpecAR), `frites.workflow` (WfMi, WfStats, WfConnComod).
  - GitHub: `brainets/frites`
- **HOI** (v0.2.0+): Higher-Order Interactions analysis.
  - Key metrics: Oinfo, GradientOinfo, InfoTopo, RedundancyMMI, SynergyMMI, RSI, DTC.
  - GitHub: `brainets/hoi`
- **XGI** (v0.7.0+): Hypergraph and higher-order network analysis.
  - GitHub: `xgi-org/xgi`

## 4. Source of Truth & Expertise
- **Examples first:** Always consult `/examples` for usage patterns before writing new code.
- **MCP tools as reference:** Read `mcp/braina_mcp.py` for canonical function signatures and parameter conventions.
- **Source code on GitHub:** For implementation details, read `brainets/frites` and `brainets/hoi` source code.
- **Papers:** Read PDFs in `/papers` for theoretical background when needed.

## 5. Reliability & Testing Rules
- **Verification first:** Never suggest code without explaining *why* it matches the toolbox API.
- **Test before recommending:** When writing analysis scripts, create small dummy data (numpy/jax) and verify the function runs without errors before presenting to the user.
- **Data formats:** Use `.npy` for raw arrays and `.nc` (NetCDF/xarray) when metadata (ROI names, time coordinates) must be preserved.
- **Dependencies:** Use `uv` for dependency management when writing standalone scripts.

## 6. Coding Style
- Follow PEP 8 guidelines.
- Prefer **JAX** for high-performance math in HOI sub-modules.
- Use `xarray` or `MNE` structures for Frites-style connectivity data.
- Keep scripts self-contained with clear imports and docstrings.

## 7. Key Data Conventions
- Neural data shape: typically `(n_epochs, n_roi, n_times)` for Frites connectivity functions.
- HOI data shape: typically `(n_samples, n_features)` or `(n_samples, n_features, n_variables)`.
- Time vectors and ROI labels should be stored as xarray coordinates when using `.nc` format.
- Sampling frequency (`sfreq`) is often stored in xarray `.attrs`.
- Numpy must be `<2.0` (required by current Frites/HOI versions).

## 8. Commands
```bash
uv run check_env.py        # Verify environment
uv run mcp/verify_libs.py  # Run Frites + HOI test suite
uv run mcp/braina_mcp.py   # Launch MCP server standalone
uv run examples/frites/conn/ex_conn_covgc.py  # Run any example
```
