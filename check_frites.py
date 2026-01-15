# /// script
# dependencies = ["frites", "numpy<2.0.0", "xarray", "mne"]
# ///
import frites
import numpy as np
print(f"frites version: {frites.__version__}")
print(f"numpy version: {np.__version__}")
print("frites imported successfully with numpy < 2.0")
