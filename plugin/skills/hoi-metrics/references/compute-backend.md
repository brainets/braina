# Compute backend: HOI runs on CPU by default in braina

HOI is built on JAX. Braina's dependency block (`plugin/mcp/braina_mcp.py`)
requests plain `jax`/`jaxlib`, which are CPU-only wheels, so every `hoi_*`
tool runs on CPU regardless of the hardware. This is a packaging choice, not
a JAX limitation.

- **Check what is active**: run the `/braina:check` skill (the smoke test
  prints `JAX backend: CPU|GPU`) or `python -c "import jax;
  print(jax.devices())"`.
- **Using a GPU** requires installing the CUDA-matching extra, e.g.
  `jax[cuda12]`, in the environment `uv` resolves for the server (edit the
  dependency block or pre-install into the cached environment). See
  https://jax.readthedocs.io/en/latest/installation.html and
  https://brainets.github.io/hoi/install.html; do not guess the CUDA version
  - check the machine first.
- **When it matters**: many features and/or large `maxsize` (the combinatorial
  multiplet count is where JAX's vectorisation pays off). For a handful of
  ROIs with `maxsize <= 4`, CPU finishes in seconds.
- **Shared GPU machines**: JAX preallocates most GPU memory on first use;
  set `XLA_PYTHON_CLIENT_PREALLOCATE=false` or
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.3` before starting Claude Code.
- A harmless `Unable to initialize backend 'tpu'` message can appear in logs.
