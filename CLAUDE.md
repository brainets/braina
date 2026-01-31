# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Braina is an AI-agent-assisted framework for analyzing complex neural interactions using information-theoretical measures on electrophysiological data (fMRI, MEG, EEG, LFP, MUA). Built at the Institut de Neurosciences de la Timone (BraiNets), Marseille.

The user may be a novice in computational neuroscience. Explain mathematical concepts (entropy, mutual information, O-information, Granger causality) simply and verify all proposed code before recommending it.

## Commands

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

All scripts use `uv` with PEP 723 inline script metadata (`# /// script` blocks) for dependency management — no virtualenv setup needed.

## Architecture

### MCP Server (`mcp/braina_mcp.py`)

The central integration layer. A FastMCP server exposing 30+ tools that wrap Frites and HOI library functions with standardized file-based I/O. Each tool takes file paths as input (`.npy` or `.nc`), calls the underlying library function, and saves results. This is registered as a Claude Code MCP server (`braina`).

Tool categories:
- **Data I/O**: `inspect_data`, `read_pdf`, `load_data`, `save_data`
- **Frites connectivity**: `frites_conn_covgc`, `frites_conn_dfc`, `frites_conn_pid`, `frites_conn_ii`, `frites_conn_te`, `frites_conn_fit`, `frites_conn_spec`, `frites_conn_ccf`
- **Frites workflows**: `frites_wf_stats` (WfStats), `frites_wf_mi` (WfMi), `frites_wf_conn_comod` (WfConnComod)
- **Frites simulation**: `frites_sim_ar` (StimSpecAR)
- **HOI metrics**: `hoi_oinfo`, `hoi_gradient_oinfo`, `hoi_infotopo`, `hoi_redundancy_mmi`, `hoi_synergy_mmi`, `hoi_rsi`, `hoi_dtc`, `hoi_get_nbest_mult`

### Core Libraries

- **Frites** (`brainets/frites`): Single-trial functional connectivity. Key modules: `frites.conn`, `frites.simulations.StimSpecAR`, `frites.workflow` (WfMi, WfStats, WfConnComod), `frites.dataset.DatasetEphy`.
- **HOI** (`brainets/hoi`): Higher-Order Interactions. Key classes: `Oinfo`, `GradientOinfo`, `InfoTopo`, `RedundancyMMI`, `SynergyMMI`, `RSI`, `DTC`. Uses JAX for GPU acceleration.
- **XGI** (`xgi-org/xgi`): Hypergraph/higher-order network structures.

### Data Conventions

- Frites connectivity input: `(n_epochs, n_roi, n_times)` numpy array or xarray DataArray.
- HOI input: `(n_samples, n_features)` or `(n_samples, n_features, n_variables)`.
- Use `.npy` for raw arrays, `.nc` (NetCDF/xarray) when metadata (ROI names, time coords, `sfreq`) must be preserved.
- Sampling frequency stored in `xarray.DataArray.attrs['sfreq']`.

## Working in this Repo

- Consult `/examples` for usage patterns before writing new analysis code.
- Read `mcp/braina_mcp.py` for canonical function signatures and parameter conventions.
- For library internals, read source on GitHub via `gh api` or `gh browse` against `brainets/frites` and `brainets/hoi`.
- When writing analysis scripts, test with small dummy data first (see `mcp/verify_libs.py` for patterns).
- Prefer JAX for high-performance math in HOI contexts; use xarray/MNE structures for Frites connectivity.
- Numpy must be `<2.0` (required by current Frites/HOI versions).
