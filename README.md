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

- **[Frites](https://github.com/brainets/frites)** — Framework for Information Theoretical analysis of Electrophysiological data and Statistics (Granger causality, transfer entropy, PID, dynamic FC,  mutual-information workflows).
- **[HOI](https://github.com/brainets/hoi)** — Multivariate Information Theoretical tools for higher-order interaction analysis
  (O-information, synergy, redundancy, RSI, DTC, InfoTopo), GPU-capable via
  JAX.

It ships as an MCP server (30+ tools wrapping Frites and HOI) plus 4 skills
that know how to pick the right tool, explain what it actually computes, and
run the statistics correctly — grounded in the real library source, not just
tool docstrings.

## Quick start

```bash
npm install -g @anthropic-ai/claude-code
claude plugin marketplace add brainets/braina
claude plugin install braina@braina-plugins
```

That's it — no need to clone the repo or run from inside it. Then, in any
Claude Code session:

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

## MCP tools

<details>
<summary>30+ tools wrapping Frites and HOI (click to expand)</summary>

| Category | Tools |
|---|---|
| Data I/O | `inspect_data`, `read_pdf` |
| Frites connectivity | `frites_conn_covgc`, `frites_conn_dfc`, `frites_conn_pid`, `frites_conn_ii`, `frites_conn_te`, `frites_conn_fit`, `frites_conn_spec`, `frites_conn_ccf` |
| Frites workflows | `frites_wf_mi`, `frites_wf_stats`, `frites_wf_conn_comod` |
| Frites simulation | `frites_sim_ar` |
| HOI metrics | `hoi_oinfo`, `hoi_gradient_oinfo`, `hoi_infotopo`, `hoi_redundancy_mmi`, `hoi_synergy_mmi`, `hoi_rsi`, `hoi_dtc`, `hoi_get_nbest_mult` |

Each tool wraps a Frites or HOI function with file-based I/O (`.npy` or
`.nc`). See `mcp/braina_mcp.py` for exact signatures.

</details>

## Repo contents

Beyond the plugin itself, this repo also carries the reference material the
skills point to:

- **`examples/`** — ~50 self-contained scripts (Frites + HOI), runnable with
  `uv run examples/frites/conn/plot_covgc.py`.
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

# Verify the environment
uv run check_env.py
uv run mcp/verify_libs.py

# Register the MCP server (one-time setup)
claude mcp add braina -- uv run mcp/braina_mcp.py

claude
```

`.mcp.json` at the project root is used for the plugin packaging (it
references `${CLAUDE_PLUGIN_ROOT}`, which only resolves inside a plugin
install) — it does **not** auto-register the server for a plain clone, so
the manual `claude mcp add` step above is still required here. This means
`claude mcp list` will show a harmless warning about `braina` being defined
in both `project` scope (from `.mcp.json`, left unresolved outside a plugin
install) and `local` scope (from the command above) — the local one is what
actually connects, and the warning can be ignored. **Don't run
`claude mcp remove braina -s project`** to silence it — that rewrites the
committed `.mcp.json` itself (emptying it), not just local config.

Reads `CLAUDE.md` for project context.

</details>

<details>
<summary>Project structure</summary>

```
braina/
├── .claude-plugin/
│   ├── plugin.json         # Plugin manifest
│   └── marketplace.json    # Self-hosted marketplace
├── .mcp.json                # MCP server declaration (plugin use)
├── skills/                  # orientation, frites-connectivity, hoi-metrics, workflows
├── mcp/
│   ├── braina_mcp.py        # MCP server — 30+ tools wrapping Frites & HOI
│   └── verify_libs.py       # Test suite for all wrapped functions
├── examples/                # ~50 example scripts (frites/, hoi/)
├── tutorials/
├── usecases/
├── papers/
├── CLAUDE.md                 # Project context for Claude Code
└── check_env.py              # Environment verification
```

</details>

## License

BSD 3-Clause License. See [LICENSE](LICENSE).
