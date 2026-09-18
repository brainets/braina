# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mcp<2",   # mcp 2.x renamed FastMCP to MCPServer and moved the import path
#   "frites>=0.4.5",   # 0.4.4 and earlier crash on import under numpy>=2.0
#   "hoi",
#   "numpy",
#   "xarray",
#   "pandas",
#   "netcdf4",
#   "h5netcdf",
#   "scipy",       # .mat conversion
#   "mne",         # .fif conversion
#   "matplotlib",  # plot_result
#   "PyMuPDF"   # added for PDF reading
# ]
# ///

import functools
import json
import os
import pickle

import numpy as np
import xarray as xr
import pandas as pd
from mcp.server.fastmcp import FastMCP
import frites.conn
import frites.workflow
import frites.stats
import frites.dataset
import frites.simulations
import frites.estimator
import hoi.metrics
import hoi.utils
import fitz  # PyMuPDF

# Initialize FastMCP server. Named `server` (not `mcp`) so that tool
# parameters called `mcp` (Frites' multiple-comparisons-correction argument)
# never shadow it.
server = FastMCP("braina_mcp")

# --- Helper Functions for I/O ---

def load_data(path: str):
    """Loads data from .npy or .nc files (fully into memory, file closed)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if path.endswith('.npy'):
        return np.load(path)
    elif path.endswith('.nc'):
        return xr.load_dataarray(path)
    else:
        raise ValueError(f"Unsupported file format for {path}. Use .npy or .nc")


def _clean_attrs(attrs: dict) -> dict:
    """Make xarray attrs NetCDF-serializable (drop None, JSON-encode nested)."""
    out = {}
    for k, v in attrs.items():
        if v is None:
            continue
        if isinstance(v, (bool, np.bool_)):
            out[k] = int(v)
        elif isinstance(v, (str, int, float, np.integer, np.floating)):
            out[k] = v
        elif isinstance(v, np.ndarray) and v.ndim <= 1 and (np.issubdtype(v.dtype, np.number) or v.dtype.kind in "US"):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(i, (str, int, float)) for i in v):
            out[k] = list(v)
        else:
            try:
                out[k] = json.dumps(v, default=str)
            except Exception:
                out[k] = str(v)
    return out


def save_data(data, path: str):
    """Saves data to .npy or .nc files."""
    if path.endswith('.npy'):
        if isinstance(data, xr.DataArray):
            np.save(path, data.values)
        else:
            np.save(path, np.asarray(data))
    elif path.endswith('.nc'):
        if not isinstance(data, xr.DataArray):
            # Attempt basic conversion if no coords provided, specific wrappers handles this better
            data = xr.DataArray(np.asarray(data))
        data = data.copy()
        data.attrs = _clean_attrs(data.attrs)
        data.to_netcdf(path)
    else:
        raise ValueError(f"Unsupported export format for {path}. Use .npy or .nc")
    return path


def _summarize(data, label: str = "Output") -> str:
    """Human-readable summary of an array/DataArray: shape, dims, coords, stats."""
    lines = []
    arr = data.values if isinstance(data, xr.DataArray) else np.asarray(data)
    lines.append(f"{label}: shape={arr.shape}, dtype={arr.dtype}")
    if isinstance(data, xr.DataArray):
        lines.append(f"  dims: {data.dims}")
        for name, coord in data.coords.items():
            vals = coord.values
            if vals.ndim == 0:
                lines.append(f"  coord {name}: {vals}")
                continue
            if len(vals) <= 6:
                preview = ", ".join(str(v) for v in vals)
            else:
                preview = (", ".join(str(v) for v in vals[:3]) + ", ..., "
                           + ", ".join(str(v) for v in vals[-2:]))
            lines.append(f"  coord {name} ({len(vals)}): [{preview}]")
        if data.attrs:
            lines.append(f"  attrs: {dict(data.attrs)}")
    if np.issubdtype(arr.dtype, np.number) and arr.size:
        finite = np.isfinite(arr)
        n_bad = int(arr.size - finite.sum())
        if finite.any():
            vals = arr[finite]
            lines.append(f"  min={vals.min():.4g}, mean={vals.mean():.4g}, max={vals.max():.4g}"
                         + (f", non-finite={n_bad}" if n_bad else ""))
        else:
            lines.append("  all values non-finite")
    return "\n".join(lines)


def tool(fn):
    """Register `fn` as an MCP tool; any exception becomes a readable error string."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - surfaced to the agent as text
            return f"Error in {fn.__name__}: {type(e).__name__}: {e}"
    return server.tool()(wrapper)


_ESTIMATORS = {
    "gcmi": lambda: frites.estimator.GCMIEstimator(mi_type='cc'),
    "pearson": lambda: frites.estimator.CorrEstimator(method='pearson'),
    "spearman": lambda: frites.estimator.CorrEstimator(method='spearman'),
    "dcorr": lambda: frites.estimator.DcorrEstimator(),
    "binmi": lambda: frites.estimator.BinMIEstimator(mi_type='cc'),
}


def _estimator(name):
    """Map a short name to a Frites estimator instance (None -> library default)."""
    if name is None or name == "default":
        return None
    if name not in _ESTIMATORS:
        raise ValueError(f"Unknown estimator '{name}'. Choose from {sorted(_ESTIMATORS)}.")
    return _ESTIMATORS[name]()


def _sfreq_of(data):
    return data.attrs.get('sfreq') if isinstance(data, xr.DataArray) else None


def _xr_kw(data):
    """
    Tell Frites which coordinates hold the ROI names and the time axis.

    Frites only reads `roi`/`times` from a DataArray when their coordinate
    names are passed explicitly (see frites.dataset.SubjectEphy); otherwise it
    silently falls back to `roi_0, roi_1, ...` and a 1 Hz time axis.
    """
    kw = {}
    if isinstance(data, xr.DataArray):
        for name in ("roi", "times"):
            if name in data.coords:
                kw[name] = name
    return kw

# --- MCP Tool: PDF Reader ---

@tool
def read_pdf(path: str) -> str:
    """
    Read a PDF file and return its text content.

    Parameters
    ----------
    path : str
        Path to a PDF file (absolute or relative to the MCP working directory).

    Returns
    -------
    str
        Text content of the PDF or error message.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text if text else f"No text found in {path}"

# ---MCP Tool for I/O ---

@tool
def inspect_data(path: str) -> str:
    """
    Inspect a neurophysiological data file (.npy or .nc).

    Returns shape, dtype, dimensions, coordinates (with previews), attributes
    (e.g. `sfreq`) and basic value statistics (min/mean/max, non-finite count).
    """
    data = load_data(path)
    return _summarize(data, label=f"File {path}")

# --- Frites Connectivity Wrappers ---

@tool
def frites_conn_covgc(data_path: str, output_path: str, dt: int, lag: int, t0: list[float], step: int = 1, method: str = 'gc', conditional: bool = False, norm: bool = False, n_jobs: int = 1) -> str:
    """
    Single-trial covariance-based Granger Causality for gaussian variables.

    This function computes the (conditional) covariance-based Granger Causality
    (covgc) for each trial.

    Parameters
    ----------
    data_path : str
        Path to input data (npy or nc). Shape: (n_epochs, n_roi, n_times)
    output_path : str
        Path to save the output data. Use .nc extension to preserve metadata (ROI names, times).
    dt : int
        Duration of the time window for covariance correlation in samples
    lag : int
        Number of samples for the lag within each trial
    t0 : list
        Array of zero time in samples of length (n_window,)
    step : int | 1
        Number of samples stepping in the past for the lag within each trial
    method : {'gauss', 'gc'}
        Method for the estimation of the covgc. Use either 'gauss' which
        assumes that the time-points are normally distributed or 'gc' in order
        to use the gaussian-copula.
    conditional : bool | False
        If True, the conditional Granger Causality is computed i.e the past is
        also conditioned by the past of other sources.
    norm : bool | False
        If True, return the normalized Granger causality (0 = no causal
        influence, 1 = full predictability).
    n_jobs : int | 1
        Number of parallel jobs (-1 = all cores).
    """
    data = load_data(data_path)
    t0_arr = np.array(t0)

    gc = frites.conn.conn_covgc(data, dt=dt, lag=lag, t0=t0_arr, step=step, method=method, conditional=conditional, norm=norm, n_jobs=n_jobs, **_xr_kw(data))

    save_data(gc, output_path)
    return f"CovGC computed and saved to {output_path}.\n{_summarize(gc)}"

@tool
def frites_conn_dfc(data_path: str, output_path: str, win_sample: list[list[int]] | None = None, agg_ch: bool = False, estimator: str = 'gcmi', n_jobs: int = 1) -> str:
    """
    Single trial Dynamic Functional Connectivity.

    This function computes the pairwise Dynamic Functional Connectivity (DFC)
    by estimating the statistical dependencies between time-series (possibly on
    sliding windows) and at the single-trial level using a measure of
    information.

    Parameters
    ----------
    data_path : str
        Path to input data (npy or nc). Shape: (n_epochs, n_roi, n_times)
    output_path : str
        Path to save the output data. Use .nc extension to preserve metadata (ROI names, times).
    win_sample : list | None
        List of [start, stop] indices for windows. If None, uses entire time.
    agg_ch : bool | False
        In case there are multiple electrodes, channels, contacts or sources
        inside a brain region, specify how the data has to be aggregated.
    estimator : {'gcmi', 'pearson', 'spearman', 'dcorr', 'binmi'}
        Dependency measure: Gaussian-copula MI (default), Pearson or Spearman
        correlation, distance correlation, or binning-based MI.
    n_jobs : int | 1
        Number of parallel jobs (-1 = all cores).
    """
    data = load_data(data_path)

    win_sample_arr = np.array(win_sample) if win_sample is not None else None

    dfc = frites.conn.conn_dfc(data, win_sample=win_sample_arr, agg_ch=agg_ch, estimator=_estimator(estimator), n_jobs=n_jobs, **_xr_kw(data))

    save_data(dfc, output_path)
    return f"DFC computed and saved to {output_path}.\n{_summarize(dfc)}"

@tool
def frites_conn_pid(data_path: str, y_path: str, output_path_prefix: str, mi_type: str = 'cc', dt: int = 1) -> str:
    """
    Compute the Partial Information Decomposition on connectivity pairs.

    This function can be used to untangle how the information about a stimulus
    is carried inside a brain network.

    Parameters
    ----------
    data_path : str
        Path to input data. Shape: (n_epochs, n_roi, n_times)
    y_path : str
        Path to behavior/stimulus data (y), one value per epoch.
    output_path_prefix : str
        Prefix for output files: generates _infotot.nc, _unique.nc,
        _redundancy.nc and _synergy.nc (.nc to preserve metadata).
    mi_type : {'cc', 'cd'}
        Mutual information type. 'cc' (continuous-continuous) or 'cd' (continuous-discrete).
    dt : int | 1
        Number of successive time points to consider when computing MI.
    """
    data = load_data(data_path)
    y = load_data(y_path)

    infotot, unique, redundancy, synergy = frites.conn.conn_pid(data, y, mi_type=mi_type, dt=dt, **_xr_kw(data))

    outputs = {"infotot": infotot, "unique": unique, "redundancy": redundancy, "synergy": synergy}
    lines = [f"PID components saved with prefix {output_path_prefix}:"]
    for name, arr in outputs.items():
        path = f"{output_path_prefix}_{name}.nc"
        save_data(arr, path)
        lines.append(_summarize(arr, label=f"  {name} -> {path}"))
    return "\n".join(lines)

@tool
def frites_conn_ii(data_path: str, y_path: str, output_path: str, mi_type: str = 'cc', dt: int = 1) -> str:
    """
    Interaction Information on connectivity pairs and behavioral variable.

    This function can be used to investigate if pairs of brain regions (or
    recordings) are mainly carrying the same information, i.e. redundant
    information about a variable of the task (e.g. stimulus, outcome,
    behavioral models) or complementary information, i.e. synergistic.

    Parameters
    ----------
    data_path : str
        Path to data.
    y_path : str
        Path to behavior variable.
    output_path : str
        Path to save output. Use .nc extension to preserve metadata (ROI names, times).
    mi_type : {'cc', 'cd'}
        Mutual information type.
    dt : int
        Number of successive time points to consider when computing MI.
    """
    data = load_data(data_path)
    y = load_data(y_path)

    ii = frites.conn.conn_ii(data, y, mi_type=mi_type, dt=dt, **_xr_kw(data))

    save_data(ii, output_path)
    return f"Interaction Information saved to {output_path}.\n{_summarize(ii)}"

@tool
def frites_conn_te(data_path: str, output_path: str, max_delay: int = 30, min_delay: int = 0, step_delay: int = 1, n_jobs: int = 1) -> str:
    """
    Compute the across-trials transfer entropy (TE).

    Parameters
    ----------
    data_path : str
        Path to data. If a .nc file carries `attrs['sfreq']`, it is forwarded
        to Frites so the output time axis is expressed in seconds.
    output_path : str
        Path to save output. Use .nc extension to preserve metadata (ROI names, times).
    max_delay : int
        Number of time points defining where to stop looking at in the past.
    min_delay : int
        Start delay.
    step_delay : int
        Step between delays.
    n_jobs : int | 1
        Number of parallel jobs (-1 = all cores).
    """
    data = load_data(data_path)

    te = frites.conn.conn_te(data, max_delay=max_delay, min_delay=min_delay, step_delay=step_delay, sfreq=_sfreq_of(data), n_jobs=n_jobs, **_xr_kw(data))

    save_data(te, output_path)
    return f"Transfer Entropy saved to {output_path}.\n{_summarize(te)}"

@tool
def frites_conn_fit(data_path: str, y_path: str, output_path: str, mi_type: str = 'cc', max_delay: float = 0.3, net: bool = False) -> str:
    """
    Feature-specific information transfer.

    Parameters
    ----------
    data_path : str
        Path to data.
    y_path : str
        Path to feature (y).
    output_path : str
        Path to save output. Use .nc extension to preserve metadata (ROI names, times).
    mi_type : {'cc', 'cd'}
        Mutual information type.
    max_delay : float
        Maximum delay for past conditioning in seconds (if sfreq exists) or samples.
    net : bool
        If True, compute net transfer.
    """
    data = load_data(data_path)
    y = load_data(y_path)

    fit = frites.conn.conn_fit(data, y, mi_type=mi_type, max_delay=max_delay, net=net, sfreq=_sfreq_of(data), **_xr_kw(data))

    save_data(fit, output_path)
    return f"FIT saved to {output_path}.\n{_summarize(fit)}"


@tool
def frites_conn_spec(data_path: str, output_path: str, freqs: list[float], metric: str = 'coh', sm_times: float = 0.5, sm_freqs: int = 1, mode: str = 'morlet', n_cycles: float = 7.0, n_jobs: int = 1) -> str:
    """
    Wavelet-based single-trial time-resolved spectral connectivity.

    Parameters
    ----------
    data_path : str
        Path to input data. Needs a sampling frequency: use a .nc file with a
        `times` coordinate (Frites infers sfreq from it) or `attrs['sfreq']`.
    output_path : str
        Path to save output. Use .nc extension to preserve metadata (ROI names, times).
    freqs : list
        Array of central frequencies.
    metric : 'coh' | 'plv' | 'sxy'
        Connectivity metric.
    sm_times : float
        Temporal smoothing in seconds.
    sm_freqs : int
        Frequency smoothing.
    mode : 'morlet' | 'multitaper'
    n_cycles : float
        Number of cycles.
    n_jobs : int | 1
        Number of parallel jobs (-1 = all cores).
    """
    data = load_data(data_path)
    freqs_arr = np.array(freqs)

    sfreq = _sfreq_of(data)
    has_times = isinstance(data, xr.DataArray) and 'times' in data.coords
    if sfreq is None and not has_times:
        raise ValueError("conn_spec needs a sampling frequency: provide a .nc file with a `times` "
                         "coordinate or attrs['sfreq'] set (Frites infers sfreq from `times`).")

    conn = frites.conn.conn_spec(
        data, freqs=freqs_arr, metric=metric, sfreq=sfreq,
        sm_times=sm_times, sm_freqs=sm_freqs, mode=mode,
        n_cycles=n_cycles, n_jobs=n_jobs, **_xr_kw(data)
    )

    save_data(conn, output_path)
    return f"Spectral connectivity ({metric}) saved to {output_path}.\n{_summarize(conn)}"

@tool
def frites_conn_ccf(data_path: str, output_path: str, normalized: bool = True, n_jobs: int = 1) -> str:
    """
    Single trial Cross-Correlation Function.

    Parameters
    ----------
    data_path : str
        Path to input data.
    output_path : str
        Path to save output. Use .nc extension to preserve metadata (ROI names, times).
    normalized : bool | True
        Normalize the cross-correlation (values in [-1, 1]).
    n_jobs : int | 1
        Number of parallel jobs (-1 = all cores).
    """
    data = load_data(data_path)

    ccf = frites.conn.conn_ccf(data, normalized=normalized, n_jobs=n_jobs, **_xr_kw(data))

    save_data(ccf, output_path)
    return f"CCF saved to {output_path}.\n{_summarize(ccf)}"

@tool
def frites_sim_ar(output_path: str, ar_type: str = 'hga', n_epochs: int = 100, n_times: int = 300, n_stim: int = 3, sf: float = 200., dt: int = 50, n_std: int = 3, stim_onset: int = 100, random_state: int | None = None) -> str:
    """
    Simulate Autoregressive (AR) Model data (StimSpecAR).

    Parameters
    ----------
    output_path : str
        Path to save simulated data. Use .nc extension to preserve metadata
        (roi names, times, sfreq, stimulus per trial).
    ar_type : 'hga' | 'osc_20' | 'osc_40' | 'ding_2' | 'ding_3_direct' | 'ding_3_indirect' | 'ding_5'
        Type of AR model (see frites.simulations.StimSpecAR).
    n_epochs : int
        Number of trials per stimulus.
    n_times : int
        Number of time points.
    n_stim : int
        Number of stimulus conditions.
    sf : float | 200.
        Sampling frequency (Hz).
    dt : int | 50
        Duration of the stimulus-driven modulation, in samples.
    n_std : int | 3
        Number of standard deviations for the stimulus amplitude.
    stim_onset : int | 100
        Stimulus onset, in samples.
    random_state : int | None
        Seed for reproducible simulations.
    """
    ar_model = frites.simulations.StimSpecAR()
    data = ar_model.fit(ar_type=ar_type, n_epochs=n_epochs, n_times=n_times, n_stim=n_stim, sf=sf, dt=dt, n_std=n_std, stim_onset=stim_onset, random_state=random_state)
    data.attrs['sfreq'] = float(sf)

    save_data(data, output_path)
    return f"Simulated AR data ({ar_type}) saved to {output_path}.\n{_summarize(data)}"

# --- Frites Workflow Wrappers ---

def _prepare_stats_input(data):
    """
    Frites stats functions often expect a list of arrays (one per ROI).
    If input is a single array (n_roi, n_subjects, n_times), convert to list.
    """
    if isinstance(data, list):
        return data
    if isinstance(data, xr.DataArray):
        if 'roi' in data.dims:
            # Split by ROI
            roi_dim = data.get_axis_num('roi')
            # Move ROI dim to 0 for iteration
            data_np = data.values
            data_np = np.moveaxis(data_np, roi_dim, 0)
            return [data_np[i] for i in range(data_np.shape[0])]
        else:
            # Assume single ROI
            return [data.values]
    if isinstance(data, np.ndarray):
        if data.ndim == 3: # (n_roi, n_subjects, n_times) assumption
             return [data[i] for i in range(data.shape[0])]
        elif data.ndim == 2: # (n_subjects, n_times)
             return [data]
    return data


def _stats_to_xr(arr, effect):
    """Wrap a WfStats (n_times, n_roi) numpy output in a DataArray, reusing
    the `times`/`roi` coordinates of the effect input when they match."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        return xr.DataArray(arr)
    coords = {}
    if isinstance(effect, xr.DataArray):
        for dim, size in zip(("times", "roi"), arr.shape):
            if dim in effect.coords and effect.sizes.get(dim) == size:
                coords[dim] = effect.coords[dim].values
    return xr.DataArray(arr, dims=("times", "roi"), coords=coords)


@tool
def frites_wf_stats(effect_path: str, perms_path: str, output_path_prefix: str, inference: str = 'rfx', mcp: str = 'cluster', tail: int = 1, cluster_th: float | None = None) -> str:
    """
    Run the statistical workflow (WfStats).

    Parameters
    ----------
    effect_path : str
        Path to true effect data (npy or nc). Expected shape (n_roi, n_subjects, n_times) or list logic.
    perms_path : str
        Path to permutation data (npy or nc). Expected shape (n_perm, n_roi, n_subjects, n_times).
    output_path_prefix : str
        Prefix for output files (_pvalues.nc, _tvalues.nc), each of shape
        (n_times, n_roi).
    inference : 'ffx' | 'rfx'
    mcp : 'cluster' | 'maxstat' | 'fdr' | 'bonferroni'
    tail : -1 | 0 | 1
    cluster_th : float | None
        Threshold for cluster forming. If None, auto-inferred (TFCE not supported via simple wrapper yet).
    """
    effect = load_data(effect_path)
    perms = load_data(perms_path)

    # Prepare list input for WfStats
    effect_list = _prepare_stats_input(effect)
    perms_list = _prepare_stats_input(perms)

    # WfStats expects perms as a list of (n_perm, n_subjects, n_times) per ROI.
    # If perms loaded is (n_perm, n_roi, n_subjects, n_times), split by ROI (dim 1).
    if isinstance(perms, (xr.DataArray, np.ndarray)):
        p_data = perms.values if isinstance(perms, xr.DataArray) else perms
        if p_data.ndim == 4: # (n_perm, n_roi, n_subjects, n_times)
             perms_list = [p_data[:, i, :, :] for i in range(p_data.shape[1])]
        elif p_data.ndim == 3: # (n_perm, n_subjects, n_times) - Single ROI
             perms_list = [p_data]

    wf = frites.workflow.WfStats(verbose=False)
    pvalues, tvalues = wf.fit(effect_list, perms_list, inference=inference, mcp=mcp, tail=tail, cluster_th=cluster_th)

    pv = _stats_to_xr(pvalues, effect)
    pv_path = f"{output_path_prefix}_pvalues.nc"
    save_data(pv, pv_path)
    lines = [f"Stats computed (inference={inference}, mcp={mcp}, tail={tail}).",
             _summarize(pv, label=f"p-values -> {pv_path}")]
    if tvalues is not None:
        tv = _stats_to_xr(tvalues, effect)
        tv_path = f"{output_path_prefix}_tvalues.nc"
        save_data(tv, tv_path)
        lines.append(_summarize(tv, label=f"t-values -> {tv_path}"))
    return "\n".join(lines)


def _split_subjects(obj):
    """Split a DataArray with a `subject` dimension into a per-subject list."""
    if isinstance(obj, xr.DataArray) and 'subject' in obj.dims:
        return [obj.isel(subject=i, drop=True) for i in range(obj.sizes['subject'])]
    return [obj]


def _load_subjects(data_path, y_path=None):
    """
    Load one dataset per subject for the Frites workflows.

    `data_path` (and `y_path`) may be a single path or a list of paths, one per
    subject. A single .nc file with a `subject` dimension is split along it.
    Returns (list_of_data, list_of_y_or_None, roi_list_or_None, times_or_None).
    """
    paths = [data_path] if isinstance(data_path, str) else list(data_path)
    xs = []
    for p in paths:
        xs.extend(_split_subjects(load_data(p)))

    ys = None
    if y_path is not None:
        ypaths = [y_path] if isinstance(y_path, str) else list(y_path)
        ys = []
        for p in ypaths:
            ys.extend(_split_subjects(load_data(p)))
        if len(ys) != len(xs):
            raise ValueError(
                f"Got {len(xs)} data file(s)/subject(s) but {len(ys)} y file(s)/subject(s). "
                "Pass one y per subject (list of paths, or a .nc with a `subject` dim).")
        ys = [np.asarray(y) for y in ys]

    rois = [x.coords['roi'].values if isinstance(x, xr.DataArray) and 'roi' in x.coords else None
            for x in xs]
    if all(r is None for r in rois):
        rois = None
    elif any(r is None for r in rois):
        raise ValueError("Either all subjects or none must carry a `roi` coordinate.")

    x0 = xs[0]
    times = x0.coords['times'].values if isinstance(x0, xr.DataArray) and 'times' in x0.coords else None
    xs = [np.asarray(x) for x in xs]
    return xs, ys, rois, times


def _check_inference(inference, n_subjects):
    if inference == 'rfx' and n_subjects < 2:
        raise ValueError(
            f"inference='rfx' (random effect, t-test across subjects) needs at least 2 subjects "
            f"but {n_subjects} was given. Pass one file per subject as a list in data_path "
            "(and y_path), or a .nc with a `subject` dimension, or use inference='ffx'.")


@tool
def frites_wf_mi(data_path: str | list[str], y_path: str | list[str], output_path_prefix: str, mi_type: str = 'cc', inference: str = 'rfx', n_perm: int = 1000, n_jobs: int = 1, mcp: str = 'cluster', cluster_th: float | None = None, cluster_alpha: float = 0.05, random_state: int | None = None, estimator: str = 'default', save_workflow: bool = False) -> str:
    """
    Workflow of local mutual-information and statistics (WfMi).

    Parameters
    ----------
    data_path : str | list[str]
        Path(s) to electrophysiological data, shape (n_epochs, n_roi, n_times)
        per subject. Give a list with one file per subject for group-level
        (rfx) inference, or a single .nc file with a `subject` dimension.
    y_path : str | list[str]
        Path(s) to the regressor (y), one value per epoch, same number of
        files/subjects as data_path.
    output_path_prefix : str
        Prefix for output (_mi.nc, _pvalues.nc).
    mi_type : 'cc' | 'cd' | 'ccd'
    inference : 'ffx' | 'rfx'
        'rfx' requires >= 2 subjects.
    n_perm : int
        Number of permutations.
    n_jobs : int | 1
        Number of parallel jobs (-1 = all cores).
    mcp : 'cluster' | 'maxstat' | 'fdr' | 'bonferroni' | 'nostat' | None
        Multiple-comparisons correction.
    cluster_th : float | None
        Cluster-forming threshold (None = inferred from the permutations;
        'tfce' for threshold-free cluster enhancement).
    cluster_alpha : float | 0.05
        Significance level used to infer the cluster-forming threshold.
    random_state : int | None
        Seed for reproducible permutations.
    estimator : {'default', 'gcmi', 'pearson', 'spearman', 'dcorr', 'binmi'}
        Information estimator ('default' = Frites' GCMI for the given mi_type;
        the correlation estimators only apply to mi_type='cc').
    save_workflow : bool | False
        Also pickle the fitted workflow to <prefix>_wf.pkl so that two
        workflows can later be combined with frites_wf_mi_combine.
    """
    xs, ys, rois, times = _load_subjects(data_path, y_path)
    _check_inference(inference, len(xs))

    ds = frites.dataset.DatasetEphy(xs, y=ys, times=times, roi=rois, verbose=False)

    wf = frites.workflow.WfMi(mi_type=mi_type, inference=inference, estimator=_estimator(estimator), verbose=False)
    mi, pvalues = wf.fit(ds, mcp=mcp, n_perm=n_perm, cluster_th=cluster_th, cluster_alpha=cluster_alpha, n_jobs=n_jobs, random_state=random_state)

    mi_path, pv_path = f"{output_path_prefix}_mi.nc", f"{output_path_prefix}_pvalues.nc"
    save_data(mi, mi_path)
    save_data(pvalues, pv_path)
    lines = [
        f"WfMi completed on {len(xs)} subject(s) (mi_type={mi_type}, inference={inference}, mcp={mcp}, n_perm={n_perm}).",
        _summarize(mi, label=f"MI -> {mi_path}"),
        _summarize(pvalues, label=f"p-values -> {pv_path}"),
    ]
    if save_workflow:
        wf_path = f"{output_path_prefix}_wf.pkl"
        with open(wf_path, "wb") as f:
            pickle.dump(wf, f)
        lines.append(f"Fitted workflow pickled -> {wf_path} (for frites_wf_mi_combine)")
    return "\n".join(lines)


@tool
def frites_wf_mi_combine(wf_path_1: str, wf_path_2: str, output_path_prefix: str, mcp: str = 'cluster', cluster_th: float | None = None, cluster_alpha: float = 0.05, n_perm: int | None = None) -> str:
    """
    Combine two fitted WfMi workflows (wf_1 - wf_2) and re-run the statistics (WfMiCombine).

    Typical use: wf_1 = MI about a stimulus contrast in condition A, wf_2 =
    the same in condition B; the combined workflow tests whether mi_1 > mi_2
    using the difference of effect sizes and of their permutations.

    Parameters
    ----------
    wf_path_1, wf_path_2 : str
        Pickled workflows written by frites_wf_mi with save_workflow=True.
        Both must have been fitted on the same ROIs/times with the same
        inference and number of permutations.
    output_path_prefix : str
        Prefix for output (_mi.nc, _pvalues.nc).
    mcp : 'cluster' | 'maxstat' | 'fdr' | 'bonferroni' | 'nostat' | None
    cluster_th : float | None
    cluster_alpha : float | 0.05
    n_perm : int | None
        Ignored (the permutations of the two fitted workflows are reused).
    """
    with open(wf_path_1, "rb") as f:
        wf_1 = pickle.load(f)
    with open(wf_path_2, "rb") as f:
        wf_2 = pickle.load(f)
    wf = frites.workflow.WfMiCombine(wf_1, wf_2, verbose=False)
    mi, pvalues = wf.fit(mcp=mcp, cluster_th=cluster_th, cluster_alpha=cluster_alpha)

    mi_path, pv_path = f"{output_path_prefix}_mi.nc", f"{output_path_prefix}_pvalues.nc"
    save_data(mi, mi_path)
    save_data(pvalues, pv_path)
    return "\n".join([
        f"WfMiCombine completed (wf_1 - wf_2, mcp={mcp}).",
        _summarize(mi, label=f"MI difference -> {mi_path}"),
        _summarize(pvalues, label=f"p-values -> {pv_path}"),
    ])

@tool
def frites_wf_conn_comod(data_path: str | list[str], output_path_prefix: str, inference: str = 'rfx', n_perm: int = 1000, n_jobs: int = 1, mcp: str = 'cluster', cluster_th: float | None = None, cluster_alpha: float = 0.05, random_state: int | None = None, estimator: str = 'default') -> str:
    """
    Workflow of instantaneous pairwise comodulations and statistics (WfConnComod).

    Parameters
    ----------
    data_path : str | list[str]
        Path(s) to data, shape (n_epochs, n_roi, n_times) per subject. Give a
        list with one file per subject for group-level (rfx) inference, or a
        single .nc file with a `subject` dimension.
    output_path_prefix : str
        Prefix for output (_mi.nc, _pvalues.nc).
    inference : 'ffx' | 'rfx'
        'rfx' requires >= 2 subjects.
    n_perm : int
        Number of permutations.
    n_jobs : int | 1
        Number of parallel jobs (-1 = all cores).
    mcp : 'cluster' | 'maxstat' | 'fdr' | 'bonferroni' | 'nostat' | None
        Multiple-comparisons correction.
    cluster_th : float | None
        Cluster-forming threshold (None = inferred; 'tfce' supported).
    cluster_alpha : float | 0.05
    random_state : int | None
        Seed for reproducible permutations.
    estimator : {'default', 'gcmi', 'pearson', 'spearman', 'dcorr', 'binmi'}
        Dependency estimator ('default' = GCMI).
    """
    xs, _, rois, times = _load_subjects(data_path)
    _check_inference(inference, len(xs))

    ds = frites.dataset.DatasetEphy(xs, times=times, roi=rois, verbose=False)

    wf = frites.workflow.WfConnComod(inference=inference, estimator=_estimator(estimator), verbose=False)
    mi, pvalues = wf.fit(ds, mcp=mcp, n_perm=n_perm, cluster_th=cluster_th, cluster_alpha=cluster_alpha, n_jobs=n_jobs, random_state=random_state)

    mi_path, pv_path = f"{output_path_prefix}_mi.nc", f"{output_path_prefix}_pvalues.nc"
    save_data(mi, mi_path)
    save_data(pvalues, pv_path)

    return "\n".join([
        f"WfConnComod completed on {len(xs)} subject(s) (inference={inference}, n_perm={n_perm}).",
        _summarize(mi, label=f"MI -> {mi_path}"),
        _summarize(pvalues, label=f"p-values -> {pv_path}"),
    ])


# --- Frites connectivity post-processing ---

def _pair_sep(da):
    """Separator used in the `roi` pair names: '->' for directed, '-' otherwise."""
    return '->' if any('->' in str(r) for r in da['roi'].values) else '-'

@tool
def frites_conn_reshape(conn_path: str, output_path: str, directed: bool, net: bool = False, mean_trials: bool = True, fill_diagonal: float | None = None) -> str:
    """
    Reshape a raveled connectivity array ('A-B' / 'A->B' pairs along `roi`)
    into a (sources, targets, ...) matrix — the natural form for plotting a
    connectivity matrix or exporting to a graph library.

    Parameters
    ----------
    conn_path : str
        Path to a .nc connectivity output (frites_conn_covgc, _dfc, _te,
        _fit, _ccf, _ii, PID components, frites_wf_conn_comod MI/p-values).
    output_path : str
        Path to save the reshaped .nc.
    directed : bool
        True for directed measures (covgc, te, fit); False for undirected
        ones (dfc, ccf, ii, pid, comod).
    net : bool | False
        Directed only: return the net flow (A->B minus B->A) instead of the
        full matrix.
    mean_trials : bool | True
        Average over the `trials` dimension first, if present.
    fill_diagonal : float | None
        Value for the diagonal (None = NaN).
    """
    da = load_data(conn_path)
    if not isinstance(da, xr.DataArray) or 'roi' not in da.dims:
        raise ValueError("conn_reshape needs a .nc connectivity array with a `roi` dimension.")
    if mean_trials and 'trials' in da.dims:
        da = da.mean('trials')
    if directed:
        if 'direction' in da.dims:  # covgc-style (roi, times, direction) -> 'A->B' pairs
            da = frites.conn.conn_ravel_directed(da)
        out = frites.conn.conn_reshape_directed(da, net=net, sep=_pair_sep(da), fill_diagonal=fill_diagonal)
    else:
        out = frites.conn.conn_reshape_undirected(da, sep=_pair_sep(da), fill_diagonal=fill_diagonal)
    save_data(out, output_path)
    return f"Reshaped connectivity ({'net ' if net else ''}{'directed' if directed else 'undirected'}) saved to {output_path}.\n{_summarize(out)}"

@tool
def frites_conn_net(conn_path: str, output_path: str, mean_trials: bool = False) -> str:
    """
    Net directed connectivity: for every pair, (A->B) minus (B->A).

    Positive values mean A drives B more than B drives A. Works on the
    raveled .nc output of frites_conn_covgc, frites_conn_te or
    frites_conn_fit, keeping the `times` (and `trials`) dimensions.

    Parameters
    ----------
    conn_path : str
        Path to a directed .nc connectivity output.
    output_path : str
        Path to save the net connectivity (.nc).
    mean_trials : bool | False
        Average over the `trials` dimension first, if present.
    """
    da = load_data(conn_path)
    if not isinstance(da, xr.DataArray) or 'roi' not in da.dims:
        raise ValueError("conn_net needs a .nc connectivity array with a `roi` dimension.")
    if mean_trials and 'trials' in da.dims:
        da = da.mean('trials')
    if 'direction' in da.dims:  # covgc-style -> 'A->B' pairs
        da = frites.conn.conn_ravel_directed(da)
    out = frites.conn.conn_net(da, sep=_pair_sep(da))
    save_data(out, output_path)
    return f"Net directed connectivity saved to {output_path}.\n{_summarize(out)}"


# --- HOI Wrappers ---

def _hoi_input(data_path, y_path=None):
    """
    Load HOI input as numpy (HOI does not accept xarray) and recover feature
    names from the second dimension's coordinate when the input is a .nc file.
    Returns (x, y_or_None, feature_names_or_None, variable_names_or_None).
    """
    x = load_data(data_path)
    names, var_names = None, None
    if isinstance(x, xr.DataArray):
        if x.ndim >= 2 and x.dims[1] in x.coords:
            names = [str(v) for v in x.coords[x.dims[1]].values]
        if x.ndim == 3 and x.dims[2] in x.coords:
            var_names = x.coords[x.dims[2]].values
        x = x.values
    x = np.asarray(x)
    y = None
    if y_path:
        y = np.asarray(load_data(y_path))
    return x, y, names, var_names


def _hoi_to_xr(result, model, names=None, var_names=None, minsize=None, maxsize=None):
    """
    Wrap a fitted HOI result (n_mult, n_variables) in a DataArray that keeps
    the multiplet metadata (`model.multiplets`, `model.order`) so that
    `hoi_get_nbest_mult` can resolve values back to ROI combinations.
    """
    arr = np.asarray(result)
    if arr.ndim == 1:
        arr = arr[:, None]
    mults = np.asarray(model.multiplets)
    orders = np.asarray(model.order)
    mult_ids = [",".join(str(int(i)) for i in m if i >= 0) for m in mults]
    coords = {
        "multiplets": mult_ids,
        "order": ("multiplets", orders),
    }
    if names is not None:
        # With a target y, HOI appends y as an extra feature (index n_features).
        def _label(i):
            i = int(i)
            return names[i] if i < len(names) else ("y" if i == len(names) else f"y{i - len(names)}")
        coords["multiplet_names"] = ("multiplets", [
            " / ".join(_label(i) for i in m if i >= 0) for m in mults])
    if var_names is not None and len(var_names) == arr.shape[1]:
        coords["variables"] = var_names
    attrs = {
        "metric": type(model).__name__,
        "minsize": int(orders.min()) if minsize is None else int(minsize),
        "maxsize": int(orders.max()) if maxsize is None else int(maxsize),
    }
    if names is not None:
        attrs["feature_names"] = json.dumps(list(names))
    return xr.DataArray(arr, dims=("multiplets", "variables"), coords=coords, attrs=attrs)


def _load_samples(samples_path):
    if samples_path is None:
        return None
    samples = np.asarray(load_data(samples_path)).ravel()
    if not np.issubdtype(samples.dtype, np.integer):
        raise ValueError("samples_path must hold integer sample indices.")
    return samples


def _bootstrap_ci(model, fit_kw, n_samples, n_boots, ci, random_state):
    """Resample samples with replacement and return (low, high) percentiles."""
    rng = np.random.default_rng(random_state)
    boots = []
    for _ in range(n_boots):
        idx = rng.integers(0, n_samples, n_samples)
        boots.append(np.asarray(model.fit(samples=idx, **fit_kw)))
    boots = np.stack(boots)  # (n_boots, n_mult, n_var)
    lo, hi = np.percentile(boots, ci, axis=0)
    return lo, hi


def _run_hoi(model_cls, data_path, output_path, y_path=None, minsize=2, maxsize=None, y_required=False,
             method='gc', samples_path=None, n_boots=0, ci_percentiles=(5., 95.), random_state=0,
             fit_extra=None, y_allowed=True):
    """Shared driver for every hoi_* metric tool."""
    x, y, names, var_names = _hoi_input(data_path, y_path if y_allowed else None)
    if y_required and y is None:
        raise ValueError(f"{model_cls.__name__} requires a target variable: pass y_path.")
    if not y_allowed:
        model = model_cls(x)
    else:
        model = model_cls(x, y=y) if not y_required else model_cls(x, y)
    fit_kw = dict(minsize=minsize, maxsize=maxsize, method=method)
    if fit_extra:
        fit_kw.update(fit_extra)
    samples = _load_samples(samples_path)
    result = model.fit(samples=samples, **fit_kw)
    da = _hoi_to_xr(result, model, names=names, var_names=var_names, minsize=minsize, maxsize=maxsize)
    da.attrs["method"] = method
    save_data(da, output_path)
    lines = [f"{model_cls.__name__} (method={method}) saved to {output_path}.", _summarize(da)]
    if not output_path.endswith(".nc"):
        lines.append("Note: .npy output drops the multiplet metadata; use .nc to be able to run hoi_get_nbest_mult.")
    if n_boots and n_boots > 0:
        n_samples = x.shape[0] if samples is None else len(samples)
        lo, hi = _bootstrap_ci(model, fit_kw, n_samples, n_boots, list(ci_percentiles), random_state)
        ci = xr.concat([da.copy(data=lo), da.copy(data=hi)], dim="ci").assign_coords(ci=["low", "high"])
        ci.attrs.update({"n_boots": int(n_boots), "percentiles": list(map(float, ci_percentiles))})
        stem = output_path[:-3] if output_path.endswith(".nc") else output_path.rsplit(".", 1)[0]
        ci_path = f"{stem}_ci.nc"
        save_data(ci, ci_path)
        excl = (lo > 0) | (hi < 0)
        lines.append(f"Bootstrap CI ({n_boots} resamples, percentiles {list(ci_percentiles)}) -> {ci_path}; "
                     f"{int(excl.sum())}/{excl.size} multiplet values have a CI excluding 0.")
    return "\n".join(lines)


@tool
def hoi_oinfo(data_path: str, output_path: str, y_path: str | None = None, minsize: int = 2, maxsize: int | None = None, method: str = 'gc', samples_path: str | None = None, n_boots: int = 0, ci_percentiles: list[float] = [5., 95.], random_state: int = 0) -> str:
    """
    O-information.

    The O-information is defined as the difference between the total
    correlation (TC) minus the dual total correlation (DTC).
    Positive values reflect redundancy, negative values reflect synergy.

    Parameters
    ----------
    data_path : str
        Path to input data (n_samples, n_features, [n_variables]). A .nc file
        whose feature dimension has a coordinate (e.g. ROI names) lets the
        output carry readable multiplet names.
    output_path : str
        Path to save output. Use .nc to keep the multiplet metadata
        (`multiplets`, `order`, `multiplet_names` coords) needed by
        hoi_get_nbest_mult.
    y_path : str | None
        Path to task-related feature.
    minsize : int
        Minimum size of multiplets.
    maxsize : int | None
        Maximum size of multiplets.
    method : {'gc', 'gauss', 'binning', 'knn', 'kernel'}
        Entropy estimator: Gaussian copula (default), plain Gaussian, binning
        (data must already be discretized), k-nearest neighbours, or kernel.
    samples_path : str | None
        Optional .npy of integer sample indices to restrict the fit to (e.g.
        a condition, or one bootstrap draw).
    n_boots : int | 0
        If > 0, bootstrap the estimate by resampling samples with replacement
        this many times and write a `<output>_ci.nc` with the low/high
        percentiles (dims: ci, multiplets, variables).
    ci_percentiles : list[float] | [5, 95]
        Percentiles for the bootstrap confidence interval.
    random_state : int | 0
        Seed for the bootstrap resampling.
    """
    return _run_hoi(hoi.metrics.Oinfo, data_path, output_path, y_path, minsize, maxsize, method=method, samples_path=samples_path, n_boots=n_boots, ci_percentiles=ci_percentiles, random_state=random_state)

@tool
def hoi_gradient_oinfo(data_path: str, y_path: str, output_path: str, minsize: int = 2, maxsize: int | None = None, method: str = 'gc', samples_path: str | None = None, n_boots: int = 0, ci_percentiles: list[float] = [5., 95.], random_state: int = 0) -> str:
    """
    First order Gradient O-information.

    Parameters
    ----------
    data_path : str
        Path to input data (n_samples, n_features, [n_variables]).
    y_path : str
        Path to target variable.
    output_path : str
        Path to save output. Use .nc to keep the multiplet metadata needed by
        hoi_get_nbest_mult.
    minsize : int
        Minimum size of multiplets.
    maxsize : int | None
        Maximum size of multiplets.
    method : {'gc', 'gauss', 'binning', 'knn', 'kernel'}
        Entropy estimator: Gaussian copula (default), plain Gaussian, binning
        (data must already be discretized), k-nearest neighbours, or kernel.
    samples_path : str | None
        Optional .npy of integer sample indices to restrict the fit to (e.g.
        a condition, or one bootstrap draw).
    n_boots : int | 0
        If > 0, bootstrap the estimate by resampling samples with replacement
        this many times and write a `<output>_ci.nc` with the low/high
        percentiles (dims: ci, multiplets, variables).
    ci_percentiles : list[float] | [5, 95]
        Percentiles for the bootstrap confidence interval.
    random_state : int | 0
        Seed for the bootstrap resampling.
    """
    return _run_hoi(hoi.metrics.GradientOinfo, data_path, output_path, y_path, minsize, maxsize, y_required=True, method=method, samples_path=samples_path, n_boots=n_boots, ci_percentiles=ci_percentiles, random_state=random_state)

@tool
def hoi_infotopo(data_path: str, output_path: str, minsize: int = 1, maxsize: int | None = None, method: str = 'gc', samples_path: str | None = None, n_boots: int = 0, ci_percentiles: list[float] = [5., 95.], random_state: int = 0) -> str:
    """
    Topological Information.

    The multivariate mutual information Ik quantify the variability/randomness
    and the statistical dependences between variables.

    Parameters
    ----------
    data_path : str
        Path to input data.
    output_path : str
        Path to save output. Use .nc to keep the multiplet metadata needed by
        hoi_get_nbest_mult.
    minsize : int
        Minimum multiplet size.
    maxsize : int | None
        Maximum multiplet size.
    method : {'gc', 'gauss', 'binning', 'knn', 'kernel'}
        Entropy estimator: Gaussian copula (default), plain Gaussian, binning
        (data must already be discretized), k-nearest neighbours, or kernel.
    samples_path : str | None
        Optional .npy of integer sample indices to restrict the fit to (e.g.
        a condition, or one bootstrap draw).
    n_boots : int | 0
        If > 0, bootstrap the estimate by resampling samples with replacement
        this many times and write a `<output>_ci.nc` with the low/high
        percentiles (dims: ci, multiplets, variables).
    ci_percentiles : list[float] | [5, 95]
        Percentiles for the bootstrap confidence interval.
    random_state : int | 0
        Seed for the bootstrap resampling.
    """
    return _run_hoi(hoi.metrics.InfoTopo, data_path, output_path, None, minsize, maxsize, method=method, samples_path=samples_path, n_boots=n_boots, ci_percentiles=ci_percentiles, random_state=random_state)

@tool
def hoi_redundancy_mmi(data_path: str, y_path: str, output_path: str, minsize: int = 2, maxsize: int | None = None, method: str = 'gc', samples_path: str | None = None, n_boots: int = 0, ci_percentiles: list[float] = [5., 95.], random_state: int = 0) -> str:
    """
    Redundancy estimated using the Minimum Mutual Information.

    Parameters
    ----------
    data_path : str
        Path to input data.
    y_path : str
        Path to feature (y).
    output_path : str
        Path to save output. Use .nc to keep the multiplet metadata needed by
        hoi_get_nbest_mult.
    minsize : int
        Minimum size of multiplets.
    maxsize : int | None
        Maximum size of multiplets.
    method : {'gc', 'gauss', 'binning', 'knn', 'kernel'}
        Entropy estimator: Gaussian copula (default), plain Gaussian, binning
        (data must already be discretized), k-nearest neighbours, or kernel.
    samples_path : str | None
        Optional .npy of integer sample indices to restrict the fit to (e.g.
        a condition, or one bootstrap draw).
    n_boots : int | 0
        If > 0, bootstrap the estimate by resampling samples with replacement
        this many times and write a `<output>_ci.nc` with the low/high
        percentiles (dims: ci, multiplets, variables).
    ci_percentiles : list[float] | [5, 95]
        Percentiles for the bootstrap confidence interval.
    random_state : int | 0
        Seed for the bootstrap resampling.
    """
    return _run_hoi(hoi.metrics.RedundancyMMI, data_path, output_path, y_path, minsize, maxsize, y_required=True, method=method, samples_path=samples_path, n_boots=n_boots, ci_percentiles=ci_percentiles, random_state=random_state)

@tool
def hoi_synergy_mmi(data_path: str, y_path: str, output_path: str, minsize: int = 2, maxsize: int | None = None, method: str = 'gc', samples_path: str | None = None, n_boots: int = 0, ci_percentiles: list[float] = [5., 95.], random_state: int = 0) -> str:
    """
    Synergy estimated using the Minimum Mutual Information.

    Parameters
    ----------
    data_path : str
        Path to input data.
    y_path : str
        Path to feature (y).
    output_path : str
        Path to save output. Use .nc to keep the multiplet metadata needed by
        hoi_get_nbest_mult.
    minsize : int
        Minimum size of multiplets.
    maxsize : int | None
        Maximum size of multiplets.
    method : {'gc', 'gauss', 'binning', 'knn', 'kernel'}
        Entropy estimator: Gaussian copula (default), plain Gaussian, binning
        (data must already be discretized), k-nearest neighbours, or kernel.
    samples_path : str | None
        Optional .npy of integer sample indices to restrict the fit to (e.g.
        a condition, or one bootstrap draw).
    n_boots : int | 0
        If > 0, bootstrap the estimate by resampling samples with replacement
        this many times and write a `<output>_ci.nc` with the low/high
        percentiles (dims: ci, multiplets, variables).
    ci_percentiles : list[float] | [5, 95]
        Percentiles for the bootstrap confidence interval.
    random_state : int | 0
        Seed for the bootstrap resampling.
    """
    return _run_hoi(hoi.metrics.SynergyMMI, data_path, output_path, y_path, minsize, maxsize, y_required=True, method=method, samples_path=samples_path, n_boots=n_boots, ci_percentiles=ci_percentiles, random_state=random_state)

@tool
def hoi_rsi(data_path: str, y_path: str, output_path: str, minsize: int = 2, maxsize: int | None = None, method: str = 'gc', samples_path: str | None = None, n_boots: int = 0, ci_percentiles: list[float] = [5., 95.], random_state: int = 0) -> str:
    """
    Redundancy-Synergy Index (RSI).

    RSI is positive for synergy and negative for redundancy.

    Parameters
    ----------
    data_path : str
        Path to input data.
    y_path : str
        Path to feature (y).
    output_path : str
        Path to save output. Use .nc to keep the multiplet metadata needed by
        hoi_get_nbest_mult.
    minsize : int
        Minimum size of multiplets.
    maxsize : int | None
        Maximum size of multiplets.
    method : {'gc', 'gauss', 'binning', 'knn', 'kernel'}
        Entropy estimator: Gaussian copula (default), plain Gaussian, binning
        (data must already be discretized), k-nearest neighbours, or kernel.
    samples_path : str | None
        Optional .npy of integer sample indices to restrict the fit to (e.g.
        a condition, or one bootstrap draw).
    n_boots : int | 0
        If > 0, bootstrap the estimate by resampling samples with replacement
        this many times and write a `<output>_ci.nc` with the low/high
        percentiles (dims: ci, multiplets, variables).
    ci_percentiles : list[float] | [5, 95]
        Percentiles for the bootstrap confidence interval.
    random_state : int | 0
        Seed for the bootstrap resampling.
    """
    return _run_hoi(hoi.metrics.RSI, data_path, output_path, y_path, minsize, maxsize, y_required=True, method=method, samples_path=samples_path, n_boots=n_boots, ci_percentiles=ci_percentiles, random_state=random_state)

@tool
def hoi_dtc(data_path: str, output_path: str, y_path: str | None = None, minsize: int = 2, maxsize: int | None = None, method: str = 'gc', samples_path: str | None = None, n_boots: int = 0, ci_percentiles: list[float] = [5., 95.], random_state: int = 0) -> str:
    """
    Dual Total Correlation (DTC).

    Parameters
    ----------
    data_path : str
        Path to input data.
    output_path : str
        Path to save output. Use .nc to keep the multiplet metadata needed by
        hoi_get_nbest_mult.
    y_path : str | None
        Path to task-related feature.
    minsize : int
        Minimum size of multiplets.
    maxsize : int | None
        Maximum size of multiplets.
    method : {'gc', 'gauss', 'binning', 'knn', 'kernel'}
        Entropy estimator: Gaussian copula (default), plain Gaussian, binning
        (data must already be discretized), k-nearest neighbours, or kernel.
    samples_path : str | None
        Optional .npy of integer sample indices to restrict the fit to (e.g.
        a condition, or one bootstrap draw).
    n_boots : int | 0
        If > 0, bootstrap the estimate by resampling samples with replacement
        this many times and write a `<output>_ci.nc` with the low/high
        percentiles (dims: ci, multiplets, variables).
    ci_percentiles : list[float] | [5, 95]
        Percentiles for the bootstrap confidence interval.
    random_state : int | 0
        Seed for the bootstrap resampling.
    """
    return _run_hoi(hoi.metrics.DTC, data_path, output_path, y_path, minsize, maxsize, method=method, samples_path=samples_path, n_boots=n_boots, ci_percentiles=ci_percentiles, random_state=random_state)

@tool
def hoi_tc(data_path: str, output_path: str, y_path: str | None = None, minsize: int = 2, maxsize: int | None = None, method: str = 'gc', samples_path: str | None = None, n_boots: int = 0, ci_percentiles: list[float] = [5., 95.], random_state: int = 0) -> str:
    """
    Total Correlation (TC).

    TC = sum of marginal entropies minus the joint entropy: how far the
    variables are, collectively, from being independent. Always >= 0.
    O-information = TC - DTC.

    Parameters
    ----------
    data_path : str
        Path to input data (n_samples, n_features, [n_variables]).
    output_path : str
        Path to save output (.nc to keep multiplet metadata).
    y_path : str | None
        Optional task-related variable.
    minsize, maxsize : int
        Multiplet size range.
    method : {'gc', 'gauss', 'binning', 'knn', 'kernel'}
        Entropy estimator: Gaussian copula (default), plain Gaussian, binning
        (data must already be discretized), k-nearest neighbours, or kernel.
    samples_path : str | None
        Optional .npy of integer sample indices to restrict the fit to (e.g.
        a condition, or one bootstrap draw).
    n_boots : int | 0
        If > 0, bootstrap the estimate by resampling samples with replacement
        this many times and write a `<output>_ci.nc` with the low/high
        percentiles (dims: ci, multiplets, variables).
    ci_percentiles : list[float] | [5, 95]
        Percentiles for the bootstrap confidence interval.
    random_state : int | 0
        Seed for the bootstrap resampling.
    """
    return _run_hoi(hoi.metrics.TC, data_path, output_path, y_path, minsize, maxsize, method=method, samples_path=samples_path, n_boots=n_boots, ci_percentiles=ci_percentiles, random_state=random_state)

@tool
def hoi_sinfo(data_path: str, output_path: str, y_path: str | None = None, minsize: int = 2, maxsize: int | None = None, method: str = 'gc', samples_path: str | None = None, n_boots: int = 0, ci_percentiles: list[float] = [5., 95.], random_state: int = 0) -> str:
    """
    S-information (Sinfo) = TC + DTC.

    Total amount of shared and joint dependence in a multiplet (the
    "strength" companion of the O-information, which is TC - DTC).

    Parameters
    ----------
    data_path : str
        Path to input data (n_samples, n_features, [n_variables]).
    output_path : str
        Path to save output (.nc to keep multiplet metadata).
    y_path : str | None
        Optional task-related variable.
    minsize, maxsize : int
        Multiplet size range.
    method : {'gc', 'gauss', 'binning', 'knn', 'kernel'}
        Entropy estimator: Gaussian copula (default), plain Gaussian, binning
        (data must already be discretized), k-nearest neighbours, or kernel.
    samples_path : str | None
        Optional .npy of integer sample indices to restrict the fit to (e.g.
        a condition, or one bootstrap draw).
    n_boots : int | 0
        If > 0, bootstrap the estimate by resampling samples with replacement
        this many times and write a `<output>_ci.nc` with the low/high
        percentiles (dims: ci, multiplets, variables).
    ci_percentiles : list[float] | [5, 95]
        Percentiles for the bootstrap confidence interval.
    random_state : int | 0
        Seed for the bootstrap resampling.
    """
    return _run_hoi(hoi.metrics.Sinfo, data_path, output_path, y_path, minsize, maxsize, method=method, samples_path=samples_path, n_boots=n_boots, ci_percentiles=ci_percentiles, random_state=random_state)

@tool
def hoi_infotot(data_path: str, y_path: str, output_path: str, minsize: int = 1, maxsize: int | None = None, method: str = 'gc', samples_path: str | None = None, n_boots: int = 0, ci_percentiles: list[float] = [5., 95.], random_state: int = 0) -> str:
    """
    Total information I(multiplet; y): the mutual information between each
    multiplet of features (taken jointly) and the target y. This is the
    quantity that redundancy_mmi / synergy_mmi decompose.

    Parameters
    ----------
    data_path : str
        Path to input data (n_samples, n_features, [n_variables]).
    y_path : str
        Path to the target variable (n_samples,).
    output_path : str
        Path to save output (.nc to keep multiplet metadata).
    minsize, maxsize : int
        Multiplet size range (minsize=1 gives the single-feature MI too).
    method : {'gc', 'gauss', 'binning', 'knn', 'kernel'}
        Entropy estimator: Gaussian copula (default), plain Gaussian, binning
        (data must already be discretized), k-nearest neighbours, or kernel.
    samples_path : str | None
        Optional .npy of integer sample indices to restrict the fit to (e.g.
        a condition, or one bootstrap draw).
    n_boots : int | 0
        If > 0, bootstrap the estimate by resampling samples with replacement
        this many times and write a `<output>_ci.nc` with the low/high
        percentiles (dims: ci, multiplets, variables).
    ci_percentiles : list[float] | [5, 95]
        Percentiles for the bootstrap confidence interval.
    random_state : int | 0
        Seed for the bootstrap resampling.
    """
    return _run_hoi(hoi.metrics.InfoTot, data_path, output_path, y_path, minsize, maxsize, y_required=True, method=method, samples_path=samples_path, n_boots=n_boots, ci_percentiles=ci_percentiles, random_state=random_state)

def _run_hoi_dyn(model_cls, data_path, output_path, tau, direction_axis, minsize, maxsize, method, fit_extra=None):
    """Driver for HOI's dynamical metrics (input (n_samples, n_features, n_times); lag `tau`)."""
    x, _, names, var_names = _hoi_input(data_path, None)
    if x.ndim != 3:
        raise ValueError(f"{model_cls.__name__} needs a 3D input (n_samples, n_features, n_times); got shape {x.shape}.")
    model = model_cls(x)
    fit_kw = dict(minsize=minsize, maxsize=maxsize, tau=tau, direction_axis=direction_axis, method=method)
    if fit_extra:
        fit_kw.update(fit_extra)
    result = model.fit(**fit_kw)
    da = _hoi_to_xr(result, model, names=names, var_names=None, minsize=minsize, maxsize=maxsize)
    da.attrs.update({"method": method, "tau": int(tau), "direction_axis": int(direction_axis)})
    save_data(da, output_path)
    return f"{model_cls.__name__} (tau={tau}, method={method}) saved to {output_path}.\n{_summarize(da)}"

@tool
def hoi_transfer_entropy(data_path: str, output_path: str, tau: int = 1, minsize: int = 2, maxsize: int = 2, method: str = 'gc') -> str:
    """
    Pairwise transfer entropy in a dynamical system (HOI implementation).

    For each pair (i, j) it measures how much the past of i (lagged by `tau`
    along the last axis) predicts the present of j beyond j's own past.
    Input must be 3D: (n_samples, n_features, n_times), i.e. trials x ROIs x
    time. For the trial-level, time-resolved Frites version see
    frites_conn_te.

    Parameters
    ----------
    data_path : str
        Path to 3D input data.
    output_path : str
        Path to save output (.nc to keep multiplet metadata).
    tau : int | 1
        Lag in samples along the last axis.
    minsize, maxsize : int | 2
        Pairwise only (order 2).
    method : str | 'gc'
        Entropy estimator.
    """
    return _run_hoi_dyn(hoi.metrics.TransferEntropy, data_path, output_path, tau, 0, minsize, maxsize, method)

@tool
def hoi_dotot(data_path: str, output_path: str, tau: int = 1, minsize: int = 3, maxsize: int | None = None, method: str = 'gc') -> str:
    """
    Total dynamic O-information (dOtot): the dynamical, lagged counterpart of
    the O-information. Positive = redundancy-dominated dynamics, negative =
    synergy-dominated. Input must be 3D: (n_samples, n_features, n_times).

    Parameters
    ----------
    data_path : str
        Path to 3D input data.
    output_path : str
        Path to save output (.nc to keep multiplet metadata).
    tau : int | 1
        Lag in samples along the last axis.
    minsize, maxsize : int
        Multiplet size range (minimum 3).
    method : str | 'gc'
        Entropy estimator.
    """
    return _run_hoi_dyn(hoi.metrics.DOtot, data_path, output_path, tau, 0, minsize, maxsize, method)

@tool
def hoi_redundancy_phiid(data_path: str, output_path: str, tau: int = 1, minsize: int = 2, maxsize: int | None = None, method: str = 'gc') -> str:
    """
    Redundancy from the Integrated Information Decomposition (phiID):
    information about the future of a multiplet that is redundantly carried
    by the past of each of its members (lag `tau`). Input must be 3D:
    (n_samples, n_features, n_times).

    Parameters
    ----------
    data_path : str
        Path to 3D input data.
    output_path : str
        Path to save output (.nc to keep multiplet metadata).
    tau : int | 1
        Lag in samples along the last axis.
    minsize, maxsize : int
        Multiplet size range.
    method : str | 'gc'
        Entropy estimator.
    """
    return _run_hoi_dyn(hoi.metrics.RedundancyphiID, data_path, output_path, tau, 0, minsize, maxsize, method)

@tool
def hoi_atoms_phiid(data_path: str, output_path: str, atoms: list[str] = ['sts'], tau: int = 1, method: str = 'gc') -> str:
    """
    Pairwise phiID atoms (Integrated Information Decomposition). By default
    returns the synergy->synergy atom 'sts' (information carried
    synergistically by the pair's past about the pair's future). Input must
    be 3D: (n_samples, n_features, n_times).

    Parameters
    ----------
    data_path : str
        Path to 3D input data.
    output_path : str
        Path to save output (.nc to keep multiplet metadata).
    atoms : list[str] | ['sts']
        phiID atoms to compute (e.g. 'rtr', 'sts', 'xtx', 'yty'; see
        hoi.metrics.AtomsPhiID).
    tau : int | 1
        Lag in samples along the last axis.
    method : str | 'gc'
        Entropy estimator.
    """
    return _run_hoi_dyn(hoi.metrics.AtomsPhiID, data_path, output_path, tau, 0, 2, 2, method, fit_extra={"atoms": list(atoms)})

@tool
def hoi_get_nbest_mult(hoi_path: str, output_path: str, n_best: int = 5, minsize: int | None = None, maxsize: int | None = None) -> str:
    """
    Get the n best multiplets from HOI results, resolved to feature/ROI names.

    Works on the .nc output of any hoi_* tool (which stores the `multiplets`
    and `order` coordinates). Following hoi.utils.get_nbest_mult, it returns
    the `n_best` most positive values AND the `n_best` most negative values
    (up to 2*n_best rows), positive first. Whether "positive" means redundancy
    or synergy depends on the metric's sign convention (see the hoi-metrics
    skill). CSV columns: index, order, hoi, multiplet, names.

    Parameters
    ----------
    hoi_path : str
        Path to a .nc file produced by a hoi_* tool.
    output_path : str
        Path to save the ranking (csv).
    n_best : int
        Number of multiplets to return.
    minsize : int | None
        Only consider multiplets of at least this order.
    maxsize : int | None
        Only consider multiplets of at most this order.
    """
    da = load_data(hoi_path)
    if not isinstance(da, xr.DataArray) or 'order' not in da.coords or 'multiplets' not in da.coords:
        raise ValueError(
            f"{hoi_path} has no multiplet metadata. Re-run the hoi_* tool with a .nc output_path "
            "(the .npy format drops the `multiplets`/`order` coordinates).")

    orders = np.asarray(da['order'].values, dtype=int)
    mult_lists = [[int(i) for i in str(s).split(",") if i != ""] for s in da['multiplets'].values]
    width = max(len(m) for m in mult_lists)
    multiplets = np.array([m + [-1] * (width - len(m)) for m in mult_lists], dtype=int)
    names = json.loads(da.attrs['feature_names']) if 'feature_names' in da.attrs else None

    df = hoi.utils.get_nbest_mult(
        da.values, orders=orders, multiplets=multiplets,
        n_best=n_best, minsize=minsize, maxsize=maxsize, names=names)
    df.to_csv(output_path, index=False)

    n_pos, n_neg = int((df["hoi"] >= 0).sum()), int((df["hoi"] < 0).sum())
    return (f"{len(df)} multiplets saved to {output_path} ({n_pos} most positive, {n_neg} most negative; "
            f"n_best={n_best} per sign):\n{df.to_string(index=False)}")


# --- Data conversion and plotting ---

@tool
def convert_to_nc(input_path: str, output_path: str, roi: list[str] | None = None, times: list[float] | None = None, sfreq: float | None = None, t0: float = 0.0, dims: list[str] | None = None, var_name: str | None = None) -> str:
    """
    Convert data to the .nc (NetCDF/xarray) format braina tools expect, with
    `roi`/`times`/`sfreq` metadata. Supported inputs:

    - `.fif` MNE Epochs (`-epo.fif`): channels -> roi, times and sfreq from
      the file, event codes stored as a `trials` coordinate.
    - `.mat` MATLAB (v7 or earlier): pass `var_name` if the file holds
      several variables.
    - `.csv` (rows = samples, columns = features; a non-numeric header row
      becomes the `roi` names) — the natural HOI input.
    - `.npy` / `.nc`: relabel an existing array (attach roi, times, sfreq).

    Parameters
    ----------
    input_path : str
        Input file.
    output_path : str
        Output .nc path.
    roi : list[str] | None
        Names for the ROI/feature dimension (overrides names in the file).
    times : list[float] | None
        Explicit time vector (seconds). If omitted and `sfreq` is given,
        times = t0 + arange(n_times) / sfreq.
    sfreq : float | None
        Sampling frequency (Hz), stored in attrs['sfreq'].
    t0 : float | 0.0
        Time of the first sample when building `times` from `sfreq`.
    dims : list[str] | None
        Dimension names. Defaults: 3D -> ['trials', 'roi', 'times'];
        2D -> ['samples', 'roi'] (HOI layout); 1D -> ['trials'].
    var_name : str | None
        Variable to read from a .mat file.
    """
    import pandas as pd  # noqa: F811 (already imported; explicit for clarity)
    attrs, coords = {}, {}
    ext = os.path.splitext(input_path)[1].lower()
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")
    if ext == ".fif":
        import mne
        epochs = mne.read_epochs(input_path, preload=True, verbose=False)
        arr = epochs.get_data(copy=True)
        file_roi = list(epochs.ch_names)
        file_times = epochs.times
        attrs["sfreq"] = float(epochs.info["sfreq"])
        coords["trials"] = epochs.events[:, 2]
        dims = dims or ["trials", "roi", "times"]
    elif ext == ".mat":
        from scipy.io import loadmat
        try:
            mat = loadmat(input_path)
        except NotImplementedError as e:
            raise ValueError("MATLAB v7.3 (HDF5) files are not supported; save with '-v7' in MATLAB.") from e
        keys = [k for k in mat if not k.startswith("__")]
        if var_name is None:
            if len(keys) != 1:
                raise ValueError(f"Several variables in {input_path}: {keys}. Pass var_name.")
            var_name = keys[0]
        if var_name not in mat:
            raise ValueError(f"Variable '{var_name}' not in {input_path}; available: {keys}")
        arr = np.asarray(mat[var_name]).squeeze()
        file_roi, file_times = None, None
    elif ext in (".csv", ".tsv", ".txt"):
        df = pd.read_csv(input_path, sep=None, engine="python")
        numeric_header = all(_is_number(c) for c in df.columns)
        if numeric_header:
            df = pd.read_csv(input_path, sep=None, engine="python", header=None)
            file_roi = None
        else:
            file_roi = [str(c) for c in df.columns]
        arr = df.to_numpy(dtype=float)
        file_times = None
        dims = dims or ["samples", "roi"]
    elif ext in (".npy", ".nc"):
        src = load_data(input_path)
        if isinstance(src, xr.DataArray):
            attrs.update(src.attrs)
            dims = dims or list(src.dims)
            for k, v in src.coords.items():
                if v.ndim:
                    coords[k] = v.values
            arr = src.values
        else:
            arr = np.asarray(src)
        file_roi, file_times = None, None
    else:
        raise ValueError(f"Unsupported input format '{ext}'. Use .fif, .mat, .csv, .npy or .nc")

    if dims is None:
        dims = {3: ["trials", "roi", "times"], 2: ["samples", "roi"], 1: ["trials"]}.get(arr.ndim)
        if dims is None:
            raise ValueError(f"Cannot guess dimension names for a {arr.ndim}D array; pass dims.")
    if len(dims) != arr.ndim:
        raise ValueError(f"dims {dims} does not match array of shape {arr.shape}.")

    roi = roi if roi is not None else file_roi
    if roi is not None and "roi" in dims:
        if len(roi) != arr.shape[dims.index("roi")]:
            raise ValueError(f"{len(roi)} roi names for {arr.shape[dims.index('roi')]} channels.")
        coords["roi"] = [str(r) for r in roi]
    if sfreq is not None:
        attrs["sfreq"] = float(sfreq)
    if "times" in dims:
        n_times = arr.shape[dims.index("times")]
        if times is not None:
            times = np.asarray(times, dtype=float)
        elif file_times is not None:
            times = np.asarray(file_times, dtype=float)
        elif attrs.get("sfreq"):
            times = t0 + np.arange(n_times) / float(attrs["sfreq"])
        if times is not None:
            if len(times) != n_times:
                raise ValueError(f"{len(times)} time points for {n_times} samples.")
            coords["times"] = times
            if "sfreq" not in attrs and len(times) > 1:
                attrs["sfreq"] = float(1. / np.mean(np.diff(times)))
    coords = {k: v for k, v in coords.items() if k in dims}
    da = xr.DataArray(arr, dims=dims, coords=coords, attrs=attrs)
    save_data(da, output_path)
    warn = ""
    if "roi" in dims and "roi" not in coords:
        warn += "\nWarning: no roi names -> Frites will label them roi_0, roi_1, ... (pass roi=[...])."
    if "times" in dims and "times" not in coords:
        warn += "\nWarning: no time axis -> Frites will assume 1 Hz (pass sfreq= or times=)."
    return f"Converted {input_path} -> {output_path}.\n{_summarize(da)}{warn}"


def _is_number(v):
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


@tool
def plot_result(data_path: str, output_path: str, roi: list[str] | None = None, mean_dims: list[str] | None = None, title: str | None = None, max_items: int = 30) -> str:
    """
    Plot a braina result file as a PNG so it can be inspected visually.

    What gets drawn depends on the dimensions left after averaging:
    - `roi` x `times` (connectivity, MI, p-values): one line per ROI/pair
      (covgc-style outputs also split by `direction`).
    - `roi` x `freqs` x `times` (spectral connectivity): one time-frequency
      image per ROI pair.
    - `multiplets` (HOI outputs): horizontal bars, red = positive, blue =
      negative, labelled with multiplet names when available, limited to
      the `max_items` largest |values|.
    - square `sources` x `targets` (frites_conn_reshape): a matrix image.
    Any `trials` dimension is averaged first. Plain .npy arrays get a line
    (1D) or image (2D) plot.

    Parameters
    ----------
    data_path : str
        Result file (.nc or .npy).
    output_path : str
        PNG path.
    roi : list[str] | None
        Subset of roi labels (pairs) to plot.
    mean_dims : list[str] | None
        Extra dimensions to average over before plotting (e.g. ['freqs']).
    title : str | None
    max_items : int | 30
        Maximum number of lines/bars/panels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    da = load_data(data_path)
    if not isinstance(da, xr.DataArray):
        da = xr.DataArray(np.asarray(da))
    if 'trials' in da.dims:
        da = da.mean('trials')
    for d in (mean_dims or []):
        if d in da.dims:
            da = da.mean(d)
    if roi is not None and 'roi' in da.dims:
        da = da.sel(roi=[r for r in roi if r in da['roi'].values])
    da = da.squeeze(drop=True)
    desc = ""

    if 'multiplets' in da.dims:
        vals = da.values if da.ndim == 1 else da.isel({d: 0 for d in da.dims if d != 'multiplets'}).values
        labels = da['multiplet_names'].values if 'multiplet_names' in da.coords else da['multiplets'].values
        idx = np.argsort(np.abs(vals))[::-1][:max_items][::-1]
        fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(idx) + 1)))
        colors = ['tab:red' if v >= 0 else 'tab:blue' for v in vals[idx]]
        ax.barh(np.arange(len(idx)), vals[idx], color=colors)
        ax.set_yticks(np.arange(len(idx)))
        ax.set_yticklabels([str(labels[i]) for i in idx], fontsize=8)
        ax.axvline(0, color='k', lw=0.8)
        ax.set_xlabel(da.attrs.get('metric', 'value'))
        desc = f"{len(idx)} multiplets (largest |value|), red=positive, blue=negative"
    elif {'sources', 'targets'} <= set(da.dims):
        mat = da if da.ndim == 2 else da.mean([d for d in da.dims if d not in ('sources', 'targets')])
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(mat.transpose('sources', 'targets').values, cmap='viridis', aspect='auto')
        ax.set_xticks(range(mat.sizes['targets'])); ax.set_xticklabels([str(v) for v in mat['targets'].values], rotation=90, fontsize=8)
        ax.set_yticks(range(mat.sizes['sources'])); ax.set_yticklabels([str(v) for v in mat['sources'].values], fontsize=8)
        ax.set_xlabel('targets'); ax.set_ylabel('sources'); fig.colorbar(im, ax=ax)
        desc = "sources x targets matrix"
    elif 'freqs' in da.dims and 'times' in da.dims:
        others = [d for d in da.dims if d not in ('freqs', 'times')]
        panels = [(None, da)] if not others else [(str(v), da.sel({others[0]: v})) for v in da[others[0]].values[:max_items]]
        n = len(panels); ncols = min(3, n); nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
        for ax, (lab, p) in zip(axes.ravel(), panels):
            p = p.squeeze(drop=True).transpose('freqs', 'times')
            im = ax.pcolormesh(p['times'].values, p['freqs'].values, p.values, shading='auto', cmap='viridis')
            ax.set_title(lab or ''); ax.set_xlabel('time (s)'); ax.set_ylabel('freq (Hz)'); fig.colorbar(im, ax=ax)
        for ax in axes.ravel()[n:]:
            ax.axis('off')
        desc = f"{n} time-frequency panel(s)"
    elif 'times' in da.dims:
        x = da['times'].values if 'times' in da.coords else np.arange(da.sizes['times'])
        others = [d for d in da.dims if d != 'times']
        fig, ax = plt.subplots(figsize=(9, 4.5))
        n_lines = 0
        if not others:
            ax.plot(x, da.values); n_lines = 1
        else:
            stacked = da.stack(line=others) if len(others) > 1 else da.rename({others[0]: 'line'})
            for k in range(min(stacked.sizes['line'], max_items)):
                line = stacked.isel(line=k)
                lab = line['line'].values.item() if 'line' in line.coords else k
                lab = " | ".join(map(str, lab)) if isinstance(lab, tuple) else str(lab)
                ax.plot(x, line.values, label=lab, lw=1.2); n_lines += 1
            if n_lines <= 15:
                ax.legend(fontsize=8, ncol=2)
        ax.axhline(0, color='k', lw=0.6, alpha=0.5)
        ax.set_xlabel('time (s)' if 'times' in da.coords else 'sample')
        ax.set_ylabel(da.name or da.attrs.get('type', 'value'))
        desc = f"{n_lines} line(s) over time"
    elif da.ndim == 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(da.values, aspect='auto', cmap='viridis'); fig.colorbar(im, ax=ax)
        ax.set_xlabel(da.dims[1]); ax.set_ylabel(da.dims[0]); desc = "2D image"
    elif da.ndim == 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        xs_ = da[da.dims[0]].values if da.dims[0] in da.coords else np.arange(da.size)
        ax.plot(xs_, da.values); ax.set_xlabel(da.dims[0]); desc = "1D line"
    else:
        raise ValueError(f"Don't know how to plot dims {da.dims}; use mean_dims to reduce them.")

    fig.suptitle(title or os.path.basename(data_path))
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return f"Plot saved to {output_path} ({desc}; dims plotted: {da.dims}). Open the PNG to inspect it."

if __name__ == "__main__":
    server.run()
