<p align="center">
  <img src="docs/braina_logo.png" alt="Braina logo" width="300"/>
</p>

# AI agent for Brain Interaction Analysis (Braina)

Braina is a framework that turns AI coding agents into experts in computational neuroscience. It provides an MCP server, curated examples, tutorials, and research papers so that AI agents (Gemini CLI or Claude Code) can analyze complex neural interactions using information-theoretical measures.

Built by the [BraiNets](https://github.com/brainets) team at the [Institut de Neurosciences de la Timone](https://www.int.univ-amu.fr/), Marseille, France.

## Core Toolboxes

Braina integrates three Python libraries for brain interaction analysis:

- **[Frites](https://github.com/brainets/frites)** — Single-trial functional connectivity and information-theoretical analysis (Granger causality, transfer entropy, PID, DFC, mutual information workflows).
- **[HOI](https://github.com/brainets/hoi)** — Higher-Order Interactions using JAX (O-information, synergy, redundancy, RSI, DTC, InfoTopo).

These tools operate on electrophysiological data: fMRI, MEG, EEG, LFP, and MUA multivariate time series.

## Installation

### Prerequisites

- **Python 3.10+**
- **[uv](https://docs.astral.sh/uv/)** — used for dependency management. All scripts use PEP 723 inline metadata, so no virtualenv setup is needed.
- **Node.js** (for Gemini CLI and npx-based MCP servers)

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Clone the repository

```bash
git clone https://github.com/brainets/braina.git
cd braina
```

### Verify the environment

```bash
uv run check_env.py        # Check core dependencies (frites, hoi, xgi, numpy, xarray, mne, jax)
uv run mcp/verify_libs.py  # Run the test suite for Frites + HOI functions
```

## Setting up an AI Agent

Braina supports two AI coding agents. You can use either or both.

### Option A: Gemini CLI

1. Install Gemini CLI:

```bash
npm install -g @google/gemini-cli
```

2. Create `~/.gemini/settings.json` (or `~/.config/gemini-cli/settings.json`):

```json
{
  "mcpServers": {
    "braina": {
      "command": "uv",
      "args": ["run", "/path/to/braina/mcp/braina_mcp.py"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_GITHUB_PAT"
      }
    },
    "python-executor": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-everything"]
    },
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp"]
    }
  }
}
```

Replace `/path/to/braina/` with the absolute path to this repository and `YOUR_GITHUB_PAT` with your [GitHub Personal Access Token](https://github.com/settings/tokens).

The agent reads `GEMINI.md` for project context. For details on each MCP server, see [gemini-cli-setup.md](gemini-cli-setup.md).

3. Launch from the repo directory:

```bash
gemini
```

### Option B: Claude Code

1. Install Claude Code:

```bash
npm install -g @anthropic-ai/claude-code
```

2. Register the Braina MCP server (run from the repo directory):

```bash
claude mcp add braina -- uv run mcp/braina_mcp.py
```

The agent reads `CLAUDE.md` for project context. Claude Code can also call Frites/HOI functions directly via Python scripts.

3. Launch from the repo directory:

```bash
claude
```

## Project Structure

```
braina/
├── mcp/
│   ├── braina_mcp.py      # MCP server — 30+ tools wrapping Frites & HOI
│   ├── verify_libs.py     # Test suite for all wrapped functions
│   └── __init__.py
├── examples/
│   ├── frites/            # ~30 example scripts
│   │   ├── conn/          # Connectivity metrics (covgc, dfc, spec, ccf, ...)
│   │   ├── mi/            # Mutual information analysis
│   │   ├── simulations/   # AR model data simulation
│   │   ├── statistics/    # Statistical testing
│   │   └── ...
│   └── hoi/               # ~20 example scripts
│       ├── metrics/       # O-info, synergy, redundancy, RSI, DTC
│       ├── it/            # Information theory fundamentals
│       └── ...
├── tutorials/
│   ├── multivariate_information_theory_frites_hoi_xgi/
│   └── seeg_ebrains_frites/
├── usecases/              # Real-world analysis scenarios
│   ├── brainhack_26/      # BrainHack 2026 challenges
│   ├── hoi/               # HOI redundancy/synergy detection
│   ├── granger/           # Granger Causality analysis
│   └── master_td/         # Master's thesis directed topics
├── papers/                # Research papers (theoretical foundation)
├── GEMINI.md              # Project context for Gemini CLI
├── CLAUDE.md              # Project context for Claude Code
├── check_env.py           # Environment verification
└── gemini-cli-setup.md    # Detailed MCP server setup guide
```

### MCP Server (`mcp/braina_mcp.py`)

The central component. A [FastMCP](https://github.com/modelcontextprotocol/python-sdk) server that exposes 30+ tools over the standard MCP stdio transport. Each tool wraps a Frites or HOI function with file-based I/O (`.npy` or `.nc` files). Tool categories:

| Category | Tools |
|---|---|
| Data I/O | `inspect_data`, `read_pdf` |
| Frites connectivity | `frites_conn_covgc`, `frites_conn_dfc`, `frites_conn_pid`, `frites_conn_ii`, `frites_conn_te`, `frites_conn_fit`, `frites_conn_spec`, `frites_conn_ccf` |
| Frites workflows | `frites_wf_mi`, `frites_wf_stats`, `frites_wf_conn_comod` |
| Frites simulation | `frites_sim_ar` |
| HOI metrics | `hoi_oinfo`, `hoi_gradient_oinfo`, `hoi_infotopo`, `hoi_redundancy_mmi`, `hoi_synergy_mmi`, `hoi_rsi`, `hoi_dtc`, `hoi_get_nbest_mult` |

### Examples

~50 self-contained Python scripts demonstrating Frites and HOI usage. Each script uses PEP 723 inline dependencies and can be run with `uv run`:

```bash
uv run examples/frites/conn/ex_conn_covgc.py
uv run examples/hoi/metrics/ex_oinfo.py
```

### Tutorials

- **`multivariate_information_theory_frites_hoi_xgi/`** — Integration of frites, hoi, and xgi for multivariate information theory analysis. Based on Giovanni Petri's [practical tutorial](https://github.com/lordgrilo/cnww-hoi).
- **`seeg_ebrains_frites/`** — Analyzing SEEG data with frites. Dataset: Lachaux, J.-P., Rheims, S., Chatard, B., Dupin, M., & Bertrand, O. (2023). Human Intracranial Database (release-5). EBRAINS. https://doi.org/10.25493/FCPJ-NZ

### Use Cases

Real-world analysis prompts and solutions that demonstrate end-to-end workflows: AR model simulation, dynamic functional connectivity, higher-order interaction detection, and Granger causality analysis.

## License

BSD 3-Clause License. See [LICENSE](LICENSE).
