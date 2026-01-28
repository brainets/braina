<p align="center">
  <img src="docs/braina_logo.png" alt="Braina logo" width="300"/>
</p>

# AI agent for Brain Interaction Analysis (Braina)

This project is dedicated to the development of an AI agent for the analysis of complex neural interactions using a combination of toolboxes: `frites`, `hoi`, and `xgi`. It serves as a comprehensive resource for researchers and students interested in applying information-theoretical measures for the analysis of functional interactions from electrophysiological data, such as fMRI, MEG, EEG, LFP, MUA multivariate time series. 

## Getting Started

To set up the Braina AI agent, you need to install the `gemini-cli` and configure the Model Context Protocol (MCP) servers.

### 1. Installing gemini-cli

The `gemini-cli` can be installed via npm. Ensure you have Node.js installed, then run:

```bash
npm install -g @google/gemini-cli
```

For the latest installation instructions, visit the [official documentation](https://github.com/google-gemini/gemini-cli).

### 2. Creating the `settings.json` File

The Gemini CLI requires a `settings.json` file to manage credentials and MCP servers. This file should be placed in your Gemini CLI configuration directory (typically `~/.config/gemini-cli/settings.json`).

Create the file with the following structure:

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

#### Key Configurations:

*   **Braina MCP Server:** This custom server (`braina/mcp/braina_mcp.py`) exposes the core functionality of `frites` and `hoi` directly to the agent. **Replace `/path/to/braina/` with the absolute path to this project repository.**
*   **GitHub MCP Tool:** Enables the agent to read source code and documentation directly from GitHub. Replace `YOUR_GITHUB_PAT` with your [GitHub Personal Access Token](https://github.com/settings/tokens).
*   **Python Executor Tool:** Provides a secure environment for the agent to execute and verify Python code locally.
*   **Context7 MCP Server:** Used to fetch up-to-date documentation for various libraries.

For more detailed information on each tool and server, refer to [gemini-cli-setup.md](gemini-cli-setup.md).

### 3. Suggested gemini-cli Extensions

A recommended extension for the Gemini CLI is the **Prompt Library**, which contains professionally crafted prompts for common development and analysis tasks.

## Project Structure

Here is an overview of the project's directory structure:

```
/
├───papers/
│   ├─── ... (Research papers on information theory and neural analysis)
└───tutorials/
    ├───multivariate_information_theory_frites_hoi_xgi/
    │   ├─── ... (Tutorials on integrating frites, hoi, and xgi)
    └───seeg_ebrains_frites/
        ├─── ... (Tutorials on using frites with SEEG data from ebrains)

```

### Papers

The `papers/` directory contains a collection of key research papers that form the theoretical foundation of the analysis techniques used in this project.

### Tutorials

The `tutorials/` directory provides practical examples and hands-on tutorials to guide you through the process of analyzing neural data.

*   **`multivariate_information_theory_frites_hoi_xgi/`**: This tutorial focuses on the integration of `frites`, `hoi`, and `xgi` for performing multivariate information theory analysis. Copied from Giovanni Petri's "Multivariate Information Theory: A Practical Tutorial with HOI, Frites, and XGI" https://github.com/lordgrilo/cnww-hoi
*   **`seeg_ebrains_frites/`**: This tutorial demonstrates how to use the `frites` toolbox to analyze SEEG data, with examples from the ebrains platform. Dataset from Lachaux, J.-P., Rheims, S., Chatard, B., Dupin, M., & Bertrand, O. (2023). Human Intracranial Database (release-5) [Data set]. EBRAINS. https://doi.org/10.25493/FCPJ-NZ
*   **`example_learning_toolbox.md`**: This document contains a guided tour of Frites and HOI generated using the gemini-cli Prompt Library Extension 
