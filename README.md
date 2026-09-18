<p align="center">
  <img src="docs/braina_logo.png" alt="Braina logo" width="360"/>
</p>

<h1 align="center">Braina</h1>
<p align="center"><b>A Claude Code plugin for brain interaction analysis.</b></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Claude%20Code-plugin-8A63D2.svg" alt="Claude Code plugin">
</p>

Braina turns [Claude Code](https://claude.com/claude-code) into a working
partner for analyzing complex neural interactions in electrophysiological
data (fMRI, MEG, EEG, LFP, MUA) — using information-theoretical measures
from two toolboxes built by the [BraiNets](https://github.com/brainets) team
at the Institut de Neurosciences de la Timone:

- **[Frites](https://github.com/brainets/frites)** — Framework for Information Theoretical analysis of Electrophysiological data and Statistics (Granger causality, transfer entropy, PID, dynamic FC,  mutual-information workflows, permutation-based statistics and group-level analysis).
- **[HOI](https://github.com/brainets/hoi)** — Multivariate Information Theoretical tools for higher-order interaction analysis
  (O-information, synergy, redundancy, RSI, DTC, InfoTopo), GPU-capable via
  JAX.

It ships as an MCP server (34 tools wrapping Frites and HOI, plus data
conversion and plotting) and 5 skills that know how to pick the right tool,
explain what it actually computes, run the statistics correctly and check
the installation — grounded in the real library source, not just tool
docstrings.

## Quick start

```bash
npm install -g @anthropic-ai/claude-code
claude plugin marketplace add brainets/braina
claude plugin install braina@braina-plugins
```

That's it — no need to clone the repo or run from inside it. The first
session pre-warms the Python environment in the background (about 150 MB of
wheels, once); if the `braina` MCP server shows "Connection closed" in that
very first session, run `/braina:check` or simply start a new session. Then,
in any Claude Code session:

> **You:** I have LFP recordings from two regions during a working-memory
> task (`data.nc`). Does region A drive region B?
>
> **Claude:** That's a directionality question — here's a ranked shortlist:
> **covariance-based Granger causality** (the established default, robust to
> monotonic nonlinearities), **transfer entropy** (scans a range of delays
> directly), or **feature-specific information transfer** (if you care which
> task feature is being transferred). Which fits best?
>
> **You:** Let's go with Granger causality.
>
> **Claude:** Running `frites_conn_covgc`... Done. X→Y is stronger than Y→X
> in the 200–500 ms window after stimulus onset. Want to test that for
> significance across your subjects next?

## Skills

| Skill | What it does |
|---|---|
| `orientation` | Triages a plain-language question against Frites and HOI, returning a ranked shortlist of tools with tradeoffs — including which statistical test fits. |
| `frites-connectivity` | Explains each Frites connectivity tool (Granger causality, transfer entropy, PID, coherence, cross-correlation, ...) once one's been picked. |
| `hoi-metrics` | Explains HOI's higher-order metrics (O-info, synergy, redundancy, DTC, ...) — including sign-convention gotchas and CPU/GPU notes. |
| `workflows` | Explains permutation testing, cluster correction, bootstrapping, and fixed/random-effect inference, for Frites *and* HOI output. |
| `check` | `/braina:check` — verifies the installation: versions, JAX backend (CPU/GPU), MCP connection, end-to-end smoke test. |

Each skill is a short `SKILL.md` plus `references/` files that Claude reads
on demand, so only the relevant detail enters the context.

## MCP tools

<details>
<summary>34 tools wrapping Frites and HOI (click to expand)</summary>

| Category | Tools |
|---|---|
| Data I/O | `inspect_data`, `convert_to_nc` (MNE `.fif`, `.mat`, `.csv`, `.npy` → labelled `.nc`), `plot_result` (PNG), `read_pdf` |
| Frites connectivity | `frites_conn_covgc`, `frites_conn_dfc`, `frites_conn_pid`, `frites_conn_ii`, `frites_conn_te`, `frites_conn_fit`, `frites_conn_spec`, `frites_conn_ccf`, `frites_conn_reshape`, `frites_conn_net` |
| Frites workflows | `frites_wf_mi` (multi-subject), `frites_wf_mi_combine`, `frites_wf_stats`, `frites_wf_conn_comod` |
| Frites simulation | `frites_sim_ar` |
| HOI metrics | `hoi_oinfo`, `hoi_gradient_oinfo`, `hoi_infotopo`, `hoi_redundancy_mmi`, `hoi_synergy_mmi`, `hoi_rsi`, `hoi_dtc`, `hoi_tc`, `hoi_sinfo`, `hoi_infotot`, `hoi_transfer_entropy`, `hoi_dotot`, `hoi_redundancy_phiid`, `hoi_atoms_phiid`, `hoi_get_nbest_mult` |

Each tool wraps a Frites or HOI function with file-based I/O (`.npy` or
`.nc`) and returns a summary of the result (shape, coordinates, min/mean/max).
Use `.nc` input with `roi` and `times` coordinates so that outputs keep ROI
names and a time axis in seconds; HOI outputs saved as `.nc` keep the
multiplet metadata that `hoi_get_nbest_mult` needs. Every HOI metric accepts
`method` (estimator) and bootstrap confidence intervals (`n_boots`). See
`plugin/mcp/braina_mcp.py` for exact signatures.

</details>

## Repo contents

Beyond the plugin itself, this repo also carries the reference material the
skills point to:

- **`plugin/examples/`** — ~50 self-contained scripts (Frites + HOI), runnable with
  `uv run plugin/examples/frites/conn/plot_covgc.py` (shipped with the plugin,
  the skills point to them).
- **`tutorials/`** — longer walkthroughs, including a full SEEG analysis
  pipeline and a Frites+HOI+XGI integration notebook.
- **`usecases/`** — end-to-end analysis scenarios (AR simulation, dynamic FC,
  higher-order interaction detection, Granger causality).
- **`papers/`** — the theoretical background behind each method.

<details>
<summary>Developing on braina directly (instead of installing the plugin)</summary>

```bash
git clone https://github.com/brainets/braina.git
cd braina

# Verify the environment and run the test suite
uv run check_env.py
uv run plugin/mcp/verify_libs.py
uv run tests/run_tests.py            # pytest over all 34 MCP tools
uv run plugin/scripts/smoke_test.py  # what /braina:check runs

# Register the MCP server (one-time setup)
claude mcp add braina -- uv run plugin/mcp/braina_mcp.py

claude
```

`plugin/mcp/braina_mcp.py.lock` pins the server's dependencies (`uv lock
--script plugin/mcp/braina_mcp.py` to refresh after changing them).
Behavioural checks of the skills live in `plugin/evals/` and run with
`claude plugin eval plugin`. CI (`.github/workflows/ci.yml`) runs the
environment check, the library verification and the test suite.

The MCP server is declared in `plugin/.claude-plugin/plugin.json`
(`mcpServers`, using `${CLAUDE_PLUGIN_ROOT}`), which only applies to a plugin
install — a plain clone does not auto-register the server, hence the manual
`claude mcp add` step above. The marketplace installs only the `plugin/`
directory (~2 MB); papers, tutorials and use cases stay in the repo.

Reads `CLAUDE.md` for project context.

</details>

<details>
<summary>Project structure</summary>

```
braina/
├── .claude-plugin/marketplace.json   # Self-hosted marketplace → source: ./plugin
├── plugin/                           # Everything the plugin installs
│   ├── .claude-plugin/plugin.json    # Manifest (incl. MCP server declaration)
│   ├── mcp/
│   │   ├── braina_mcp.py             # MCP server — 34 tools wrapping Frites & HOI
│   │   ├── braina_mcp.py.lock        # uv lockfile for the server script
│   │   └── verify_libs.py            # Library-level verification
│   ├── skills/                       # orientation, frites-connectivity, hoi-metrics, workflows, check
│   │   └── <skill>/SKILL.md + references/
│   ├── examples/                     # ~50 example scripts (frites/, hoi/)
│   ├── hooks/hooks.json              # SessionStart: pre-warm the uv environment
│   ├── scripts/                      # warmup.sh, smoke_test.py
│   └── evals/                        # `claude plugin eval` cases
├── tests/                            # pytest suite (uv run tests/run_tests.py)
├── .github/workflows/ci.yml
├── tutorials/  usecases/  papers/  docs/
├── CHANGELOG.md
├── CLAUDE.md                         # Project context for Claude Code
└── check_env.py                      # Environment verification
```

</details>

## License

BSD 3-Clause License. See [LICENSE](LICENSE).
