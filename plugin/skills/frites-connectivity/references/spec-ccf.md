# `frites_conn_spec` and `frites_conn_ccf` - no information theory involved

## `frites_conn_spec` - spectral connectivity

Wavelet (`mode='morlet'`) or multitaper time-frequency decomposition, then a
coupling metric per frequency, time and pair, per trial. Output dims
`(trials, roi, freqs, times)`.

- `freqs` - central frequencies (Hz), required.
- `metric` - `'coh'` coherence (0-1), `'plv'` phase-locking value (0-1),
  `'sxy'` complex cross-spectrum.
- `sm_times` (seconds) and `sm_freqs` (number of frequency bins) - smoothing
  of the single-trial estimate; more smoothing = more stable, coarser.
- `n_cycles` - wavelet length in cycles; more cycles = better frequency
  resolution, worse time resolution. The wavelet must fit in the epoch:
  `n_cycles / f` seconds must be shorter than the signal, or Frites raises
  "wavelet is longer than the signal".
- `n_jobs` - parallel pairs.

Needs a sampling frequency: a `.nc` input with a `times` coordinate (Frites
infers `sfreq` from it) or `attrs['sfreq']`. The wrapper refuses input with
neither. Frequencies must be below Nyquist (`sfreq / 2`).

API: https://brainets.github.io/frites/api/generated/frites.conn.conn_spec.html

## `frites_conn_ccf` - cross-correlation function

Single-trial cross-correlation at every lag between ROI pairs. Output dims
`(trials, roi, times)` where `times` is the **lag axis in samples**
(negative to positive).

- `normalized` - `True` (default) gives values in [-1, 1].
- `n_jobs` - parallel pairs.

Interpretation: a peak at a **negative** lag means the source leads the
target; at a **positive** lag the target leads. There is no lag-cropping
parameter - select the lag window on the output afterwards.

API: https://brainets.github.io/frites/api/generated/frites.conn.conn_ccf.html
