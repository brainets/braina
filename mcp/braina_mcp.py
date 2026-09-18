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
#   "PyMuPDF"   # added for PDF reading
# ]
# ///

import functools
import json
import os

import numpy as np
import xarray as xr
import pandas as pd
from mcp.server.fastmcp import FastMCP
import frites.conn
import frites.workflow
import frites.stats
import frites.dataset
import frites.simulations
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
        elif isinstance(v, (str, int, float, np.integer, np.floating, np.ndarray)):
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
def frites_conn_covgc(data_path: str, output_path: str, dt: int, lag: int, t0: list[float], step: int = 1, method: str = 'gc', conditional: bool = False) -> str:
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
    """
    data = load_data(data_path)
    t0_arr = np.array(t0)

    gc = frites.conn.conn_covgc(data, dt=dt, lag=lag, t0=t0_arr, step=step, method=method, conditional=conditional, n_jobs=1, **_xr_kw(data))

    save_data(gc, output_path)
    return f"CovGC computed and saved to {output_path}.\n{_summarize(gc)}"

@tool
def frites_conn_dfc(data_path: str, output_path: str, win_sample: list[list[int]] | None = None, agg_ch: bool = False) -> str:
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
    """
    data = load_data(data_path)

    win_sample_arr = np.array(win_sample) if win_sample is not None else None

    dfc = frites.conn.conn_dfc(data, win_sample=win_sample_arr, agg_ch=agg_ch, n_jobs=1, **_xr_kw(data))

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
def frites_conn_te(data_path: str, output_path: str, max_delay: int = 30, min_delay: int = 0, step_delay: int = 1) -> str:
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
    """
    data = load_data(data_path)

    te = frites.conn.conn_te(data, max_delay=max_delay, min_delay=min_delay, step_delay=step_delay, sfreq=_sfreq_of(data), n_jobs=1, **_xr_kw(data))

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
def frites_conn_spec(data_path: str, output_path: str, freqs: list[float], metric: str = 'coh', sm_times: float = 0.5, sm_freqs: int = 1, mode: str = 'morlet', n_cycles: float = 7.0) -> str:
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
        n_cycles=n_cycles, n_jobs=1, **_xr_kw(data)
    )

    save_data(conn, output_path)
    return f"Spectral connectivity ({metric}) saved to {output_path}.\n{_summarize(conn)}"

@tool
def frites_conn_ccf(data_path: str, output_path: str) -> str:
    """
    Single trial Cross-Correlation Function.

    Parameters
    ----------
    data_path : str
        Path to input data.
    output_path : str
        Path to save output. Use .nc extension to preserve metadata (ROI names, times).
    """
    data = load_data(data_path)

    ccf = frites.conn.conn_ccf(data, n_jobs=1, **_xr_kw(data))

    save_data(ccf, output_path)
    return f"CCF saved to {output_path}.\n{_summarize(ccf)}"

@tool
def frites_sim_ar(output_path: str, ar_type: str = 'hga', n_epochs: int = 100, n_times: int = 300, n_stim: int = 3) -> str:
    """
    Simulate Autoregressive (AR) Model data.

    Parameters
    ----------
    output_path : str
        Path to save simulated data. Use .nc extension to preserve metadata.
    ar_type : 'hga' | 'osc_20' | 'osc_40' | 'ding_2'
    n_epochs : int
    n_times : int
    n_stim : int
    """
    ar_model = frites.simulations.StimSpecAR()
    data = ar_model.fit(ar_type=ar_type, n_epochs=n_epochs, n_times=n_times, n_stim=n_stim)

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
def frites_wf_mi(data_path: str | list[str], y_path: str | list[str], output_path_prefix: str, mi_type: str = 'cc', inference: str = 'rfx', n_perm: int = 1000, n_jobs: int = 1) -> str:
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
    """
    xs, ys, rois, times = _load_subjects(data_path, y_path)
    _check_inference(inference, len(xs))

    ds = frites.dataset.DatasetEphy(xs, y=ys, times=times, roi=rois, verbose=False)

    wf = frites.workflow.WfMi(mi_type=mi_type, inference=inference, verbose=False)
    mi, pvalues = wf.fit(ds, n_perm=n_perm, n_jobs=n_jobs)

    mi_path, pv_path = f"{output_path_prefix}_mi.nc", f"{output_path_prefix}_pvalues.nc"
    save_data(mi, mi_path)
    save_data(pvalues, pv_path)

    return "\n".join([
        f"WfMi completed on {len(xs)} subject(s) (mi_type={mi_type}, inference={inference}, n_perm={n_perm}).",
        _summarize(mi, label=f"MI -> {mi_path}"),
        _summarize(pvalues, label=f"p-values -> {pv_path}"),
    ])

@tool
def frites_wf_conn_comod(data_path: str | list[str], output_path_prefix: str, inference: str = 'rfx', n_perm: int = 1000, n_jobs: int = 1) -> str:
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
    """
    xs, _, rois, times = _load_subjects(data_path)
    _check_inference(inference, len(xs))

    ds = frites.dataset.DatasetEphy(xs, times=times, roi=rois, verbose=False)

    wf = frites.workflow.WfConnComod(inference=inference, verbose=False)
    mi, pvalues = wf.fit(ds, n_perm=n_perm, n_jobs=n_jobs)

    mi_path, pv_path = f"{output_path_prefix}_mi.nc", f"{output_path_prefix}_pvalues.nc"
    save_data(mi, mi_path)
    save_data(pvalues, pv_path)

    return "\n".join([
        f"WfConnComod completed on {len(xs)} subject(s) (inference={inference}, n_perm={n_perm}).",
        _summarize(mi, label=f"MI -> {mi_path}"),
        _summarize(pvalues, label=f"p-values -> {pv_path}"),
    ])


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
        coords["multiplet_names"] = ("multiplets", [
            " / ".join(names[int(i)] for i in m if i >= 0) for m in mults])
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


def _run_hoi(model_cls, data_path, output_path, y_path=None, minsize=2, maxsize=None, y_required=False):
    x, y, names, var_names = _hoi_input(data_path, y_path)
    if y_required and y is None:
        raise ValueError(f"{model_cls.__name__} requires a target variable: pass y_path.")
    model = model_cls(x, y=y) if not y_required else model_cls(x, y)
    result = model.fit(minsize=minsize, maxsize=maxsize)
    da = _hoi_to_xr(result, model, names=names, var_names=var_names, minsize=minsize, maxsize=maxsize)
    save_data(da, output_path)
    note = "" if output_path.endswith(".nc") else (
        "\nNote: .npy output drops the multiplet metadata; use .nc to be able to run hoi_get_nbest_mult.")
    return f"{model_cls.__name__} saved to {output_path}.\n{_summarize(da)}{note}"


@tool
def hoi_oinfo(data_path: str, output_path: str, y_path: str | None = None, minsize: int = 2, maxsize: int | None = None) -> str:
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
    """
    return _run_hoi(hoi.metrics.Oinfo, data_path, output_path, y_path, minsize, maxsize)

@tool
def hoi_gradient_oinfo(data_path: str, y_path: str, output_path: str, minsize: int = 2, maxsize: int | None = None) -> str:
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
    """
    return _run_hoi(hoi.metrics.GradientOinfo, data_path, output_path, y_path, minsize, maxsize, y_required=True)

@tool
def hoi_infotopo(data_path: str, output_path: str, minsize: int = 1, maxsize: int | None = None) -> str:
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
    """
    return _run_hoi(hoi.metrics.InfoTopo, data_path, output_path, None, minsize, maxsize)

@tool
def hoi_redundancy_mmi(data_path: str, y_path: str, output_path: str, minsize: int = 2, maxsize: int | None = None) -> str:
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
    """
    return _run_hoi(hoi.metrics.RedundancyMMI, data_path, output_path, y_path, minsize, maxsize, y_required=True)

@tool
def hoi_synergy_mmi(data_path: str, y_path: str, output_path: str, minsize: int = 2, maxsize: int | None = None) -> str:
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
    """
    return _run_hoi(hoi.metrics.SynergyMMI, data_path, output_path, y_path, minsize, maxsize, y_required=True)

@tool
def hoi_rsi(data_path: str, y_path: str, output_path: str, minsize: int = 2, maxsize: int | None = None) -> str:
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
    """
    return _run_hoi(hoi.metrics.RSI, data_path, output_path, y_path, minsize, maxsize, y_required=True)

@tool
def hoi_dtc(data_path: str, output_path: str, y_path: str | None = None, minsize: int = 2, maxsize: int | None = None) -> str:
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
    """
    return _run_hoi(hoi.metrics.DTC, data_path, output_path, y_path, minsize, maxsize)

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

if __name__ == "__main__":
    server.run()
