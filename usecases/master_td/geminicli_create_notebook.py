# /// script
# dependencies = ["frites", "numpy", "matplotlib", "nbformat", "xarray"]
# ///
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Cell 1: Header
md1 = """# Tutorial: Autoregressive Simulation and Dynamic Functional Connectivity

This tutorial demonstrates how to simulate autoregressive (AR) data using `frites` and estimate Dynamic Functional Connectivity (DFC).
"""

# Cell 2: Setup
code2 = """import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from frites import set_mpl_style
from frites.simulations import StimSpecAR
from frites.conn import conn_dfc, define_windows

set_mpl_style()"""

# Cell 3: AR Model Section
md3 = """## 3. Auto-regressive model

We simulate a pairwise AR model where a source signal $X$ drives a target signal $Y$. The signals are designed to oscillate at 40Hz.

### 3a & 3b. Single Stimulus Simulation
We generate 100 trials with a single stimulus type.
"""

# Cell 4: Single Stimulus Code
code4 = """# Simulation parameters
ar_type = 'osc_40'
n_epochs = 100
n_stim = 1

# Initialize and fit the AR model
ss = StimSpecAR()
ar = ss.fit(ar_type=ar_type, n_epochs=n_epochs, n_stim=n_stim, random_state=42)

# Plot the model structure
plt.figure(figsize=(6, 4))
ss.plot_model()
plt.title("AR Model Structure (X -> Y)")
plt.show()

# Plot the data and causal coupling
plt.figure(figsize=(10, 8))
ss.plot(cmap='bwr')
plt.suptitle('Single Stimulus Simulation (40Hz)', y=1.02)
plt.tight_layout()
plt.show()

# Extract and plot time series for a single trial
times = ar.times.data
trial_idx = 0

plt.figure(figsize=(12, 4))
plt.plot(times, ar.isel(trials=trial_idx, roi=0), label='Source X')
plt.plot(times, ar.isel(trials=trial_idx, roi=1), label='Target Y', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title(f'Single Trial Time Series (Trial {trial_idx})')
plt.legend()
plt.show()"""

# Cell 5: Multiple Stimuli
md5 = """### 3c. Multiple Stimuli Simulation
We generate 100 trials with three different stimuli.
"""

# Cell 6: Multiple Stimuli Code
code6 = """n_stim_3 = 3

ss_3 = StimSpecAR()
ar_3 = ss_3.fit(ar_type=ar_type, n_epochs=n_epochs, n_stim=n_stim_3, random_state=42)

plt.figure(figsize=(10, 10))
ss_3.plot(cmap='bwr')
plt.suptitle('Three Stimuli Simulation', y=1.02)
plt.tight_layout()
plt.show()"""

# Cell 7: Equations
md7 = """### 3d. AR Model Equation

Based on the `StimSpecAR` source code for `osc_40`, the data is generated using the following equations:

$$
X_t = \eta_{1,t} + 0.55 X_{t-1} - 0.8 X_{t-2}
$$

$$
Y_t = \eta_{2,t} + 0.35 Y_{t-1} - 0.5 Y_{t-2} + C_{2,t} (0.5 X_{t-1})
$$

Where:
*   $X_t, Y_t$ are the signal values at time $t$.
*   $\eta_{1,t}, \eta_{2,t}$ are independent Gaussian noise processes.
*   $C_{2,t}$ is the time-varying coupling strength, modulated by the stimulus (a Gaussian profile over time).
"""

# Cell 8: DFC Section
md8 = """## 4. Single-trial time-resolved (dynamic) Functional Connectivity

We estimate the Dynamic Functional Connectivity (DFC) between X and Y using the data from the single-stimulus simulation (3b).

### 4a. Compute DFC
We use a sliding window approach with `window_length = 0.75s` and `win_step = 0.02s`.
"""

# Cell 9: DFC Code
code9 = """# Define parameters
slwin_len = 0.75    # Window length in seconds
slwin_step = 0.02   # Step size in seconds

# Define sliding windows
win_sample = define_windows(times, slwin_len=slwin_len, slwin_step=slwin_step)[0]

# Compute DFC
dfc = conn_dfc(ar, win_sample, times=times, roi=ar.roi.data, n_jobs=1)

# Plot average DFC across trials
dfc_mean = dfc.mean('trials')

plt.figure(figsize=(10, 6))
for roi_pair in dfc_mean.roi.data:
    plt.plot(dfc_mean.times, dfc_mean.sel(roi=roi_pair), label=roi_pair)

plt.title("Average Dynamic Functional Connectivity (X-Y)")
plt.xlabel("Time (s)")
plt.ylabel("Mutual Information (MI)")
plt.legend()
plt.show()"""

# Cell 10: Window Analysis
md10 = """### 4b. Analyze Window Length Impact
We test different window lengths to observe the trade-off between temporal resolution and estimation stability.
"""

# Cell 11: Window Analysis Code
code11 = """window_lengths = [0.1, 0.5, 0.75, 1.0]
plt.figure(figsize=(12, 8))

for wl in window_lengths:
    win_s = define_windows(times, slwin_len=wl, slwin_step=slwin_step)[0]
    
    # Compute DFC
    dfc_test = conn_dfc(ar, win_s, times=times, roi=ar.roi.data, verbose=False)
    
    # Plot average (assuming the first pair is the one of interest)
    pair_of_interest = dfc_test.roi.data[0] 
    
    plt.plot(dfc_test.mean('trials').times, 
             dfc_test.mean('trials').sel(roi=pair_of_interest), 
             label=f'Win Len = {wl}s')

plt.title("Effect of Window Length on DFC Estimation")
plt.xlabel("Time (s)")
plt.ylabel("Mutual Information")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Interpretation:
# Shorter windows (e.g., 0.1s) provide higher temporal resolution but noisier estimates.
# Longer windows (e.g., 1.0s) provide smoother estimates but smooth out temporal dynamics (smearing).
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(md1),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(md3),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_markdown_cell(md5),
    nbf.v4.new_code_cell(code6),
    nbf.v4.new_markdown_cell(md7),
    nbf.v4.new_markdown_cell(md8),
    nbf.v4.new_code_cell(code9),
    nbf.v4.new_markdown_cell(md10),
    nbf.v4.new_code_cell(code11)
]

with open('tutorial_ar_dfc.ipynb', 'w') as f:
    nbf.write(nb, f)
