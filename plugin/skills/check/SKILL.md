---
name: check
description: Verify that the braina plugin works - environment, Frites/HOI versions, JAX backend (CPU or GPU), MCP server connection, and an end-to-end smoke test of the tools. Use when the user asks to check, verify, test or debug the braina installation, when a braina MCP tool is unavailable or the server shows CONNECTION_CLOSED, or right after installing or updating the plugin.
---

# Braina self-check

Run these steps in order and report the outcome plainly.

## 1. Smoke test (environment + tools)

```bash
uv run "${CLAUDE_PLUGIN_ROOT}/scripts/smoke_test.py"
```

- Prints the Frites / HOI / NumPy versions and the JAX backend, then runs
  `frites_sim_ar` -> `frites_conn_covgc` -> `frites_conn_net`, `hoi_oinfo`
  -> `hoi_get_nbest_mult`, and `plot_result` on simulated data.
- Ends with `SMOKE TEST PASSED` (exit 0) or `SMOKE TEST FAILED: [...]`.
- The **first** run after an install downloads ~150 MB of wheels and can
  take a few minutes; that is expected. Later runs take seconds.

## 2. If the braina MCP server did not connect

A `CONNECTION_CLOSED` / "Connection closed" status for the `braina` server
at session start almost always means the cold start exceeded Claude Code's
30 s MCP timeout while `uv` was still downloading dependencies. Fix:

```bash
uv sync --script "${CLAUDE_PLUGIN_ROOT}/mcp/braina_mcp.py"
```

then reconnect with `/mcp` (or restart the session). The plugin's
SessionStart hook runs this sync in the background automatically, so the
second session normally works without intervention. If `uv` itself is
missing: https://docs.astral.sh/uv/getting-started/installation/

## 3. Interpret the JAX line

- `JAX backend: CPU` - the default; HOI tools run on CPU. Fine for a
  handful of ROIs and `maxsize` <= 4-5.
- `JAX backend: GPU` - only happens if the user installed a CUDA-enabled
  `jax` in the environment `uv` resolves. Braina's own dependency block
  requests plain (CPU) `jax`; see the `hoi-metrics` skill, section
  "Compute backend", before promising GPU speed-ups.

## 4. Optional: full test suite (development checkout only)

From a clone of the braina repository:

```bash
uv run tests/run_tests.py
```

runs the pytest suite that exercises all 34 MCP tools on synthetic data.

## Reporting

State: versions, backend, whether the smoke test passed, and - if it did
not - the first failing tool and its error line verbatim. Do not describe
the installation as working until the smoke test has actually passed.
