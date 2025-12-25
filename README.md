# Braina: Brain Interaction Analysis

This project is dedicated to the analysis of complex neural interactions using a combination of powerful toolboxes: `frites`, `hoi`, and `xgi`. It serves as a comprehensive resource for researchers and students interested in applying information-theoretical measures to electrophysiological data.

## Getting Started

To interact with this project and its associated notebooks and scripts, you will need to have the `gemini-cli` installed.

### Installing gemini-cli

You can find the latest installation instructions for `gemini-cli` in the official documentation:

*   [https://github.com/google/gemini-cli](https://github.com/google/gemini-cli)

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
        ├─── ... (Tutorials on using frites with SEEG data from EBRAINS)

```

### Papers

The `papers/` directory contains a collection of key research papers that form the theoretical foundation of the information theoretical analysis techniques used in this project. 

### Tutorials

The `tutorials/` directory provides practical Jupyter Notebooks with examples and hands-on tutorials to guide you through the process of analyzing neural data.

*   **`multivariate_information_theory_frites_hoi_xgi/`**: This tutorial focuses on the integration of `frites`, `hoi`, and `xgi` for performing multivariate information theory analysis. Copied from Giovanni Petri's "A Practical Tutorial with HOI, Frites, and XGI" https://lordgrilo.github.io/cnww-hoi/
*   **`seeg_ebrains_frites/`**: This tutorial demonstrates how to use the `frites` toolbox to analyze SEEG data, with examples from the EBRAINS platform. Dataset is taken from Lachaux, J.-P., Rheims, S., Chatard, B., Dupin, M., & Bertrand, O. (2023). Human Intracranial Database (release-5) [Data set]. EBRAINS. https://doi.org/10.25493/FCPJ-NZ
