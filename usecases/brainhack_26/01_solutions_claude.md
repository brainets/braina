# Brain Interaction Analysis: AR Models and Dynamic Functional Connectivity

This tutorial demonstrates how to simulate autoregressive (AR) models and
estimate dynamic functional connectivity (DFC) using the Frites toolbox.

```python
import numpy as np
import matplotlib.pyplot as plt
from frites.simulations import StimSpecAR
from frites.conn import conn_dfc, define_windows
from frites import set_mpl_style

set_mpl_style()
```

## Problem 3: Auto-regressive model

### 3a & 3b: 40Hz pairwise AR model — single stimulus, 100 trials

We simulate a pairwise AR model where a source X drives a target Y with
oscillatory coupling at 40Hz. With a single stimulus (`n_stim=1`), all 100
trials share the same coupling profile.

```python
ar_type = 'osc_40'
n_epochs = 100
n_stim = 1

ss_1stim = StimSpecAR()
ar_1stim = ss_1stim.fit(ar_type=ar_type, n_epochs=n_epochs, n_stim=n_stim,
                        random_state=0)

# Plot single-trial data (top) and causal coupling over time (bottom)
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Single-trial time series for X and Y (trial 0)
times = ar_1stim.times.values
axes[0].plot(times, ar_1stim[0, 0, :].values, label='X (source)')
axes[0].plot(times, ar_1stim[0, 1, :].values, label='Y (target)', alpha=0.7)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Single trial time series (trial 0)')
axes[0].legend()

# Causal coupling profile used by the simulation
axes[1].plot(times, ss_1stim._cou[0, :], color='C3')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Coupling strength')
axes[1].set_title('Causal coupling X → Y over time (single stimulus)')

plt.tight_layout()
plt.show()
```

```python
# Built-in summary plot (all trials, time-series + coupling)
plt.figure(figsize=(10, 8))
ss_1stim.plot(cmap='bwr')
plt.tight_layout()
plt.show()
```

### 3c: Three stimuli, 100 trials

With three stimuli, each stimulus modulates the X→Y coupling differently.
The total number of trials is `n_epochs × n_stim = 300`.

```python
n_stim_3 = 3

ss_3stim = StimSpecAR()
ar_3stim = ss_3stim.fit(ar_type=ar_type, n_epochs=n_epochs, n_stim=n_stim_3,
                        random_state=0)

# Plot single-trial data and coupling for each stimulus
plt.figure(figsize=(10, 10))
ss_3stim.plot(cmap='bwr')
plt.tight_layout()
plt.show()
```

### 3d: Equations of the osc_40 AR model

The `osc_40` model is a bivariate second-order autoregressive process.
The coefficients are taken from the Frites source
(`frites/simulations/sim_ar.py`, lines 135-138):

$$x_t = 0.55\, x_{t-1} \;-\; 0.8\, x_{t-2} \;+\; \eta_{1,t}$$

$$y_t = 0.35\, y_{t-1} \;-\; 0.5\, y_{t-2} \;+\; c(t)\;\cdot\;0.5\, x_{t-1} \;+\; \eta_{2,t}$$

where:

- $x_t$: source signal at time $t$
- $y_t$: target signal at time $t$
- $\eta_{1,t}, \eta_{2,t} \sim \mathcal{N}(0, 0.05)$: independent white noise
- $c(t)$: time-varying, stimulus-dependent coupling strength (Gaussian bump).
  Its amplitude differs across stimuli, producing stimulus-specific causal
  connectivity from X to Y.

The coefficients `[0.55, -0.8]` produce a damped oscillation near 40Hz at the
default sampling rate of 200Hz. The coupling is unidirectional: X drives Y
through the term $0.5\,x_{t-1}$ scaled by $c(t)$, while Y has no influence
on X.

## Problem 4: Single-trial dynamic Functional Connectivity

We use the single-stimulus data from 3b (`ar_1stim`) to estimate DFC between
X and Y using mutual information on sliding windows.

### 4a: DFC with window_length = 0.75s, step = 0.02s

```python
data = ar_1stim
times = data.times.values
roi = data.roi.values

# Define sliding windows
slwin_len = 0.75   # seconds
slwin_step = 0.02  # seconds
win_sample = define_windows(times, slwin_len=slwin_len,
                            slwin_step=slwin_step)[0]
times_win = times[win_sample].mean(1)

# Compute single-trial DFC
dfc = conn_dfc(data, win_sample, times=times, roi=roi, n_jobs=1)

# Plot average DFC across trials
plt.figure(figsize=(10, 5))
dfc.mean('trials').plot.line(x='times', hue='roi')
plt.title(f'Mean DFC across trials (window = {slwin_len}s, step = {slwin_step}s)')
plt.xlabel('Time (s)')
plt.ylabel('DFC (Mutual Information)')
plt.show()
```

### 4b: Effect of different window lengths

```python
window_lengths = [0.1, 0.25, 0.5, 1.0]

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)

for ax, win_len in zip(axes.ravel(), window_lengths):
    ws = define_windows(times, slwin_len=win_len, slwin_step=slwin_step)[0]
    dfc_w = conn_dfc(data, ws, times=times, roi=roi, n_jobs=1)

    dfc_w.mean('trials').plot.line(x='times', hue='roi', ax=ax, add_legend=False)
    ax.set_title(f'Window = {win_len}s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('DFC (MI)')

# Add a single shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.suptitle('Impact of window length on DFC estimation', fontsize=14)
plt.tight_layout()
plt.show()
```

**Interpretation:**

- **Short windows (0.1s):** High temporal resolution but noisy estimates,
  because fewer time points are available to compute mutual information within
  each window.
- **Medium windows (0.25–0.5s):** Good compromise between temporal precision
  and estimation stability.
- **Long windows (1.0s):** Smooth, stable connectivity estimates but temporal
  dynamics are smeared — onset and offset of coupling become harder to resolve.
- The 0.75s window in 4a captures the stimulus-driven X→Y coupling while
  keeping estimation variance reasonable.
