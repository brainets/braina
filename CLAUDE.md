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
uv run plugin/mcp/verify_libs.py

# Run the pytest suite over all MCP tools (synthetic data, ~30 s)
uv run tests/run_tests.py

# End-to-end smoke test of the installation (what /braina:check runs)
uv run plugin/scripts/smoke_test.py

# Launch the MCP server standalone
uv run plugin/mcp/braina_mcp.py

# Refresh the server lockfile after editing its dependency block
uv lock --script plugin/mcp/braina_mcp.py

# Behavioural evals of the skills
claude plugin eval plugin

# Run any example script (all use uv inline dependencies)
uv run plugin/examples/frites/conn/plot_covgc.py
uv run plugin/examples/hoi/metrics/plot_oinfo.py
```

All scripts use `uv` with PEP 723 inline script metadata (`# /// script` blocks) for dependency management — no virtualenv setup needed.

## Architecture

### Repository layout

Everything the plugin installs lives under `plugin/` (manifest, MCP server + lockfile, skills, examples, hooks, scripts, evals); the marketplace points at it (`.claude-plugin/marketplace.json`, `source: ./plugin`). Papers, tutorials, use cases, tests and CI stay at the root and are not installed.

### MCP Server (`plugin/mcp/braina_mcp.py`)

The central integration layer. A FastMCP server exposing 34 tools that wrap Frites and HOI library functions with standardized file-based I/O. Each tool takes file paths as input (`.npy` or `.nc`), calls the underlying library function, saves results, and returns a text summary (shape, coords, min/mean/max). Exceptions are caught and returned as `Error in <tool>: ...` strings. The server is declared in `plugin/.claude-plugin/plugin.json` under `mcpServers`; `plugin/hooks/hooks.json` pre-warms its uv environment at session start. This is registered as a Claude Code MCP server (`braina`).

`load_data`/`save_data` are internal I/O helpers (not exposed as MCP tools) that handle the `.npy` vs `.nc` dispatch for every tool.

Tool categories:
- **Data I/O**: `inspect_data`, `convert_to_nc` (.fif/.mat/.csv/.npy → labelled .nc), `plot_result` (PNG), `read_pdf`
- **Frites connectivity**: `frites_conn_covgc`, `frites_conn_dfc`, `frites_conn_pid`, `frites_conn_ii`, `frites_conn_te`, `frites_conn_fit`, `frites_conn_spec`, `frites_conn_ccf`; post-processing `frites_conn_reshape` (pairs → sources×targets), `frites_conn_net` (A→B minus B→A)
- **Frites workflows**: `frites_wf_stats` (WfStats), `frites_wf_mi` (WfMi, multi-subject, `save_workflow`), `frites_wf_mi_combine` (WfMiCombine from pickles), `frites_wf_conn_comod` (WfConnComod)
- **Frites simulation**: `frites_sim_ar` (StimSpecAR, `random_state`)
- **HOI static metrics** (all with `method`, `samples_path`, bootstrap `n_boots`): `hoi_oinfo`, `hoi_gradient_oinfo`, `hoi_infotopo`, `hoi_redundancy_mmi`, `hoi_synergy_mmi`, `hoi_rsi`, `hoi_dtc`, `hoi_tc`, `hoi_sinfo`, `hoi_infotot`
- **HOI dynamic metrics** (3D input, lag `tau`): `hoi_transfer_entropy`, `hoi_dotot`, `hoi_redundancy_phiid`, `hoi_atoms_phiid`
- **HOI utilities**: `hoi_get_nbest_mult`

Skills: `orientation`, `frites-connectivity`, `hoi-metrics`, `workflows`, `check` — each a short `SKILL.md` plus `references/*.md` read on demand. When adding or changing a tool, update the matching reference file, the pytest suite (`tests/test_mcp_tools.py`, which asserts the tool count) and `CHANGELOG.md`.

### Core Libraries

- **Frites** (`brainets/frites`): Single-trial functional connectivity. Key modules: `frites.conn`, `frites.simulations.StimSpecAR`, `frites.workflow` (WfMi, WfStats, WfConnComod), `frites.dataset.DatasetEphy`.
- **HOI** (`brainets/hoi`): Higher-Order Interactions. Key classes: `Oinfo`, `GradientOinfo`, `InfoTopo`, `RedundancyMMI`, `SynergyMMI`, `RSI`, `DTC`. Uses JAX for GPU acceleration.
- **XGI** (`xgi-org/xgi`): Hypergraph/higher-order network structures.

### Data Conventions

- Frites connectivity input: `(n_epochs, n_roi, n_times)` numpy array or xarray DataArray.
- HOI input: `(n_samples, n_features)` or `(n_samples, n_features, n_variables)`.
- Use `.npy` for raw arrays, `.nc` (NetCDF/xarray) when metadata (ROI names, time coords, `sfreq`) must be preserved.
- Sampling frequency stored in `xarray.DataArray.attrs['sfreq']`.
- Frites only reads ROI names and times from a DataArray when the coordinates are named exactly `roi` and `times` (the wrappers pass these names explicitly via `_xr_kw`); other names fall back to `roi_0, roi_1, ...` and a 1 Hz axis.
- Workflow tools (`frites_wf_mi`, `frites_wf_conn_comod`) accept a list of files (one per subject) or a `.nc` with a `subject` dimension; `inference='rfx'` requires at least 2 subjects.
- HOI tools save `.nc` outputs with `multiplets`, `order` and (when the input had a named feature coordinate) `multiplet_names` coords; `hoi_get_nbest_mult` requires that `.nc` output.

## Working in this Repo

- Consult `plugin/examples` for usage patterns before writing new analysis code.
- Read `plugin/mcp/braina_mcp.py` for canonical function signatures and parameter conventions.
- For library internals, read source on GitHub via `gh api` or `gh browse` against `brainets/frites` and `brainets/hoi`.
- When writing analysis scripts, test with small dummy data first (see `plugin/mcp/verify_libs.py` for patterns).
- Prefer JAX for high-performance math in HOI contexts; use xarray/MNE structures for Frites connectivity.
- Papers are cited in the skills by their GitHub URL under `papers/`; the installed plugin does not ship them.
- Frites must be `>=0.4.5`; earlier versions crash on import under NumPy 2. HOI has never restricted the NumPy version. With Frites `>=0.4.5`, NumPy 2 is supported.
