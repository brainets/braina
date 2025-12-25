# Project: braina (Brain Interaction Analysis)

## 1. Project Context & Purpose
- **Goal:** Analyzing complex neural interactions using Frites, HOI, and XGI toolboxes.
- **Focus:** Information Theoretical Analysis of electrophysiological data.
- **Novice Mode:** I am a novice in this domain. Explain mathematical concepts (entropy, O-information) simply and verify all proposed code before recommending it.

## 2. Project Structure
- **`/papers`**: Contains key research papers forming the theoretical basis of the analysis.
- **`/tutorials/multivariate_information_theory_frites_hoi_xgi`**: A tutorial integrating `frites`, `hoi`, and `xgi` for multivariate information theory analysis.
- **`/tutorials/seeg_ebrains_frites`**: A tutorial on using `frites` with SEEG data, possibly from the ebrains platform.

## 3. Source of Truth & Expertise
- **External Documentation:** Use the `context7` tool to prioritize official API patterns for Frites and HOI.
- **Source Code:** Use the `github` MCP tool to read `brainets/frites`, `brainets/hoi`, and `xgi-org/xgi` for specific implementation details.
- **Research:** Use `Google Search` to find the latest JAX or Neuro-imaging papers to suggest novel solutions.

## 4. Reliability & Testing Rules
- **Verification First:** Never suggest code without explaining *why* it matches the toolbox documentation.
- **Auto-Test Requirement:** Before finalizing any `.py` script, you MUST use the `python-executor` tool to:
    1. Create dummy data (numpy/jax).
    2. Run the proposed function.
    3. Check for dimension errors or performance bottlenecks.
- **Environment:** Use `uv` for dependency management whenever writing standalone scripts.

## 5. Coding Style
- Follow PEP 8 guidelines.
- Prefer **JAX** for high-performance math components in the `HOI` sub-modules.
- Use `xarray` or `MNE` structures when dealing with Frites-style connectivity.
