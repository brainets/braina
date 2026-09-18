---
type: llm
---

PASS if the assistant says that braina's HOI tools run on CPU by default because the plugin's dependency declaration requests plain (CPU-only) jax/jaxlib, that using a GPU requires a manual step (installing the CUDA-matching jax extra), mentions how to check the active backend (jax.devices() or the braina check skill), and gives a realistic sense of when a GPU matters (many features / large maxsize).
FAIL if it promises automatic GPU acceleration or says GPUs cannot be used at all.
