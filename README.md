# AI agent for Brain Interaction Analysis (Braina)

This project is dedicated to the development of an AI agent for the analysis of complex neural interactions using a combination of toolboxes: `frites`, `hoi`, and `xgi`. It serves as a comprehensive resource for researchers and students interested in applying information-theoretical measures for the analysis of functional interactions from electrophysiological data, such as fMRI, MEG, EEG, LFP, MUA multivariate time series. 

## Getting Started

To create the expert AI agent, you will need to have the `gemini-cli` installed with Model Context Protocol (MCP) servers.

### Installing gemini-cli

You can find the latest installation instructions for `gemini-cli` in the official documentation:

*   [https://github.com/google/gemini-cli](https://github.com/google/gemini-cli)

## Gemini CLI and AI Agent Setup

This project can be extended to include AI agents that leverage large language models (LLMs), such as Gemini. To enable these agents, you need to configure the Gemini CLI's Model Context Protocol (MCP) server.

The core of this configuration involves:

1.  **Installing the Gemini CLI:** Ensure you have the `gemini-cli` installed as per the instructions in the "Getting Started" section.
2.  **Setting up MPC Tools:** Refer to the `gemini-cli-setup.md` file for detailed instructions on how to configure essential MPC tools like the GitHub MCP Tool and the Python Executor Tool. These tools enable the Gemini agent to interact with various aspects of your project.
3.  **Configuring `settings.json`:** A `settings.json` file is used to store sensitive information and custom paths, such as your GitHub Personal Access Token (PAT) and the path to your Python interpreter. Instructions for creating and configuring this file are also provided in `gemini-cli-setup.md`.

By properly setting up these components, you can empower AI agents within this project to assist with tasks, code generation, analysis, and more, leveraging the capabilities of the Gemini large language models.

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
