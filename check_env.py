# /// script
# dependencies = ["frites>=0.4.5", "hoi", "xgi", "numpy", "xarray", "jax", "jaxlib", "mne"]
# ///
"""Quick environment check: imports, versions and the JAX backend.

For an end-to-end check of the MCP tools, run
`uv run plugin/scripts/smoke_test.py` (or the /braina:check skill).
"""
import frites
import hoi
import xgi
import numpy as np
import xarray as xr
import mne
import jax
print(f"frites version: {frites.__version__}")
print(f"hoi version: {hoi.__version__}")
print(f"xgi version: {xgi.__version__}")
print(f"numpy version: {np.__version__}")
print(f"xarray version: {xr.__version__}")
print(f"mne version: {mne.__version__}")
print(f"jax version: {jax.__version__} | devices: {jax.devices()}")
print("All packages imported successfully!")
