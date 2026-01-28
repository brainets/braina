# /// script
# dependencies = [
#   "mcp",
#   "frites",
#   "hoi",
#   "numpy<2.0",
#   "xarray",
#   "pandas",
#   "netcdf4",
#   "h5netcdf",
#   "PyMuPDF"   # added for PDF reading
# ]
# ///

import os
import numpy as np
import xarray as xr
import pandas as pd
from mcp.server.fastmcp import FastMCP
import frites.conn
import frites.workflow
import frites.stats
import frites.dataset
import hoi.metrics
import fitz  # PyMuPDF

# Initialize FastMCP server
mcp = FastMCP("braina_mcp")

# --- Helper Functions for I/O ---

def load_data(path: str):
    """Loads data from .npy or .nc files."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    if path.endswith('.npy'):
        return np.load(path)
    elif path.endswith('.nc'):
        return xr.open_dataarray(path)
    else:
        raise ValueError(f"Unsupported file format for {path}. Use .npy or .nc")

def save_data(data, path: str):
    """Saves data to .npy or .nc files."""
    if path.endswith('.npy'):
        if isinstance(data, xr.DataArray):
            np.save(path, data.values)
        else:
            np.save(path, data)
    elif path.endswith('.nc'):
        if not isinstance(data, xr.DataArray):
            # Attempt basic conversion if no coords provided, specific wrappers handles this better
            data = xr.DataArray(data) 
        data.to_netcdf(path)
    else:
        raise ValueError(f"Unsupported export format for {path}. Use .npy or .nc")
    return path

# --- MCP Tool: PDF Reader ---

@mcp.tool()
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
        return f"Error: File not found: {path}"
    try:
        doc = fitz.open(path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text if text else f"No text found in {path}"
    except Exception as e:
        return f"Error reading PDF {path}: {str(e)}"

# ---MCP Tool for I/O ---

@mcp.tool()
def inspect_data(path: str) -> str:
    """
    Inspect a neurophysiological data file (.npy or .nc).

    Returns metadata about the file including shape, data type, and
    coordinates/dimensions if available (NetCDF).
    """
    try:
        data = load_data(path)
        info = []
        info.append(f"File: {path}")
        info.append(f"Type: {type(data)}")
        info.append(f"Shape: {data.shape}")
        info.append(f"Dtype: {data.dtype}")

        if isinstance(data, (xr.DataArray, xr.Dataset)):
            info.append(f"Dimensions: {data.dims}")
            if hasattr(data, 'coords'):
                info.append("Coordinates:")
                for coord_name, coord_val in data.coords.items():
                    vals_preview = coord_val.values[:3] if len(coord_val) > 0 else []
                    info.append(f"  - {coord_name}: {coord_val.values.shape} values (e.g., {vals_preview}...)")
            if hasattr(data, 'attrs'):
                 info.append(f"Attributes: {list(data.attrs.keys())}")

        return "\n".join(info)
    except Exception as e:
        return f"Error inspecting file: {str(e)}"

# --- Frites Connectivity Wrappers ---

@mcp.tool()
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
    # Ensure t0 is numpy array
    t0_arr = np.array(t0)
    
    # Run Frites function
    gc = frites.conn.conn_covgc(data, dt=dt, lag=lag, t0=t0_arr, step=step, method=method, conditional=conditional, n_jobs=1)
    
    save_data(gc, output_path)
    return f"CovGC computed and saved to {output_path}. Shape: {gc.shape}"

@mcp.tool()
def frites_conn_dfc(data_path: str, output_path: str, win_sample: list[list[int]] = None, agg_ch: bool = False) -> str:
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
    
    dfc = frites.conn.conn_dfc(data, win_sample=win_sample_arr, agg_ch=agg_ch, n_jobs=1)
    
    save_data(dfc, output_path)
    return f"DFC computed and saved to {output_path}. Shape: {dfc.shape}"

@mcp.tool()
def frites_conn_pid(data_path: str, y_path: str, output_path_prefix: str, mi_type: str = 'cc', max_delay: float = 0.3) -> str:
    """
    Compute the Partial Information Decomposition on connectivity pairs.

    This function can be used to untangle how the information about a stimulus
    is carried inside a brain network.

    Parameters
    ----------
    data_path : str
        Path to input data.
    y_path : str
        Path to behavior/stimulus data (y).
    output_path_prefix : str
        Prefix for output files (will generate _unique.nc, _redundancy.nc, etc.). Generates .nc files to preserve metadata.
    mi_type : {'cc', 'cd'}
        Mutual information type. 'cc' (continuous-continuous) or 'cd' (continuous-discrete).
    """
    data = load_data(data_path)
    y = load_data(y_path)
    
    # Run PID
    infotot, unique, redundancy, synergy = frites.conn.conn_pid(data, y, mi_type=mi_type, n_jobs=1)
    
    # Save all outputs
    save_data(infotot, f"{output_path_prefix}_infotot.nc")
    save_data(unique, f"{output_path_prefix}_unique.nc")
    save_data(redundancy, f"{output_path_prefix}_redundancy.nc")
    save_data(synergy, f"{output_path_prefix}_synergy.nc")
    
    return f"PID components saved with prefix {output_path_prefix}"

@mcp.tool()
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
    
    ii = frites.conn.conn_ii(data, y, mi_type=mi_type, dt=dt, n_jobs=1)
    
    save_data(ii, output_path)
    return f"Interaction Information saved to {output_path}"

@mcp.tool()
def frites_conn_te(data_path: str, output_path: str, max_delay: int = 30, min_delay: int = 0, step_delay: int = 1) -> str:
    """
    Compute the across-trials transfer entropy (TE).

    Parameters
    ----------
    data_path : str
        Path to data.
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
    
    te = frites.conn.conn_te(data, max_delay=max_delay, min_delay=min_delay, step_delay=step_delay, n_jobs=1)
    
    save_data(te, output_path)
    return f"Transfer Entropy saved to {output_path}"

@mcp.tool()
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
    
    # Try to extract sfreq from attrs if available in xarray, else might treat max_delay as samples if int
    sfreq = None
    if isinstance(data, xr.DataArray):
        sfreq = data.attrs.get('sfreq')

    fit = frites.conn.conn_fit(data, y, mi_type=mi_type, max_delay=max_delay, net=net, sfreq=sfreq, n_jobs=1)
    
    save_data(fit, output_path)
    return f"FIT saved to {output_path}"


@mcp.tool()
def frites_conn_spec(data_path: str, output_path: str, freqs: list[float], metric: str = 'coh', sm_times: float = 0.5, sm_freqs: int = 1, mode: str = 'morlet', n_cycles: float = 7.0) -> str:
    """
    Wavelet-based single-trial time-resolved spectral connectivity.

    Parameters
    ----------
    data_path : str
        Path to input data.
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
    
    # Check if data has sfreq attr, otherwise try to infer or error
    sfreq = None
    if isinstance(data, xr.DataArray):
        sfreq = data.attrs.get('sfreq')
    
    conn = frites.conn.conn_spec(
        data, freqs=freqs_arr, metric=metric, sfreq=sfreq, 
        sm_times=sm_times, sm_freqs=sm_freqs, mode=mode, 
        n_cycles=n_cycles, n_jobs=1
    )
    
    save_data(conn, output_path)
    return f"Spectral connectivity ({metric}) saved to {output_path}"

@mcp.tool()
def frites_conn_ccf(data_path: str, output_path: str, max_delay: int = 30) -> str:
    """
    Single trial Cross-Correlation Function.

    Parameters
    ----------
    data_path : str
        Path to input data.
    output_path : str
        Path to save output. Use .nc extension to preserve metadata (ROI names, times).
    max_delay : int
        Note: Standard conn_ccf computes full lags. If cropping is needed, handle post-hoc.
    """
    data = load_data(data_path)
    
    ccf = frites.conn.conn_ccf(data, n_jobs=1)
    
    save_data(ccf, output_path)
    return f"CCF saved to {output_path}"

@mcp.tool()
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
    return f"Simulated AR data ({ar_type}) saved to {output_path}"

@mcp.tool()
def hoi_gradient_oinfo(data_path: str, y_path: str, output_path: str, minsize: int = 2, maxsize: int = None) -> str:
    """
    First order Gradient O-information.

    Parameters
    ----------
    data_path : str
        Path to input data.
    y_path : str
        Path to target variable.
    output_path : str
        Path to save output. Use .nc extension to preserve metadata (ROI names).
    """
    x = load_data(data_path)
    y = load_data(y_path)
    
    model = hoi.metrics.GradientOinfo(x, y=y)
    result = model.fit(minsize=minsize, maxsize=maxsize)
    
    save_data(result, output_path)
    return f"Gradient O-info saved to {output_path}"

@mcp.tool()
def hoi_get_nbest_mult(hoi_path: str, output_path: str, n_best: int = 5, minsize: int = None, maxsize: int = None) -> str:
    """
    Get the n best multiplets from HOI results.

    Parameters
    ----------
    hoi_path : str
        Path to HOI results (npy/nc).
    output_path : str
        Path to save dataframe (csv).
    n_best : int
    """
    hoi_res = load_data(hoi_path)
    
    # We need model info (orders, multiplets) to fully reconstruct.
    # If hoi_res is just values, we might be limited.
    # However, hoi.utils.get_nbest_mult usually takes a model object or explicit orders/multiplets.
    # For simplicity, if we saved just the array, we lost metadata.
    # Ideally, we should save/load the whole object, but for MCP passing paths is better.
    # If the user just ran a tool, they might not have the model object.
    # Constraint: simple wrapper might not work without metadata.
    # BUT, if we assume standard usage where we just want top values:
    
    # Check if we can infer or if we need to pass order/mults.
    # Actually, we can't easily without the model.
    # Let's assume the user uses this immediately after fitting, but we don't persist state.
    
    # ALTERNATIVE: Return the indices and values.
    # Or, if we assume the user provides orders/mults separately (complex).
    
    # Simplified approach: Return top N values and their indices in the array.
    # Construct a simple dataframe.
    
    vals = np.array(hoi_res).ravel()
    indices = np.argsort(vals)[::-1] # Descending
    
    top_n_indices = indices[:n_best]
    top_n_vals = vals[top_n_indices]
    
    df = pd.DataFrame({'index': top_n_indices, 'value': top_n_vals})
    df.to_csv(output_path, index=False)
    
    return f"Top {n_best} multiplets indices/values saved to {output_path}"

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

@mcp.tool()
def frites_wf_stats(effect_path: str, perms_path: str, output_path_prefix: str, inference: str = 'rfx', mcp: str = 'cluster', tail: int = 1, cluster_th: float = None) -> str:
    """
    Run the statistical workflow (WfStats).

    Parameters
    ----------
    effect_path : str
        Path to true effect data (npy or nc). Expected shape (n_roi, n_subjects, n_times) or list logic.
    perms_path : str
        Path to permutation data (npy or nc). Expected shape (n_perm, n_roi, n_subjects, n_times).
    output_path_prefix : str
        Prefix for output files (_pvalues.nc, _tvalues.nc).
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
    
    # For perms, _prepare_stats_input might slice incorrectly if dim order differs.
    # WfStats expects perms list of shape (n_perm, n_subjects, n_times) per ROI
    # If perms loaded is (n_perm, n_roi, n_subjects, n_times), we need to split by n_roi (dim 1)
    # Let's handle perms specifically if it's 4D
    if isinstance(perms, (xr.DataArray, np.ndarray)):
        p_data = perms.values if isinstance(perms, xr.DataArray) else perms
        if p_data.ndim == 4: # (n_perm, n_roi, n_subjects, n_times)
             perms_list = [p_data[:, i, :, :] for i in range(p_data.shape[1])]
        elif p_data.ndim == 3: # (n_perm, n_subjects, n_times) - Single ROI
             perms_list = [p_data]

    wf = frites.workflow.WfStats(verbose=False)
    pvalues, tvalues = wf.fit(effect_list, perms_list, inference=inference, mcp=mcp, tail=tail, cluster_th=cluster_th)
    
    save_data(pvalues, f"{output_path_prefix}_pvalues.npy")
    if tvalues is not None:
        save_data(tvalues, f"{output_path_prefix}_tvalues.npy")
        return f"Stats computed. P-values and T-values saved with prefix {output_path_prefix}"
    return f"Stats computed. P-values saved with prefix {output_path_prefix}"

@mcp.tool()
def frites_wf_mi(data_path: str, y_path: str, output_path_prefix: str, mi_type: str = 'cc', inference: str = 'rfx', n_perm: int = 1000, n_jobs: int = 1) -> str:
    """
    Workflow of local mutual-information and statistics (WfMi).

    Parameters
    ----------
    data_path : str
        Path to electrophysiological data.
    y_path : str
        Path to regressor (y).
    output_path_prefix : str
        Prefix for output (_mi.nc, _pvalues.nc).
    mi_type : 'cc' | 'cd' | 'ccd'
    inference : 'ffx' | 'rfx'
    n_perm : int
    """
    data = load_data(data_path)
    y = load_data(y_path)
    
    # Create DatasetEphy
    # Try to infer times/roi from xarray if available
    times = data.coords['times'].values if isinstance(data, xr.DataArray) and 'times' in data.coords else None
    roi = data.coords['roi'].values if isinstance(data, xr.DataArray) and 'roi' in data.coords else None
    
    ds = frites.dataset.DatasetEphy(data, y=y, times=times, roi=roi, verbose=False)
    
    wf = frites.workflow.WfMi(mi_type=mi_type, inference=inference, verbose=False)
    mi, pvalues = wf.fit(ds, n_perm=n_perm, n_jobs=n_jobs)
    
    save_data(mi, f"{output_path_prefix}_mi.nc")
    save_data(pvalues, f"{output_path_prefix}_pvalues.nc")
    
    return f"WfMi completed. MI and p-values saved with prefix {output_path_prefix}"

@mcp.tool()
def frites_wf_conn_comod(data_path: str, output_path_prefix: str, inference: str = 'rfx', n_perm: int = 1000, n_jobs: int = 1) -> str:
    """
    Workflow of instantaneous pairwise comodulations and statistics (WfConnComod).

    Parameters
    ----------
    data_path : str
        Path to data.
    output_path_prefix : str
        Prefix for output (_mi.nc, _pvalues.nc).
    inference : 'ffx' | 'rfx'
    n_perm : int
    """
    data = load_data(data_path)
    
    # Create DatasetEphy
    times = data.coords['times'].values if isinstance(data, xr.DataArray) and 'times' in data.coords else None
    roi = data.coords['roi'].values if isinstance(data, xr.DataArray) and 'roi' in data.coords else None
    
    ds = frites.dataset.DatasetEphy(data, times=times, roi=roi, verbose=False)
    
    wf = frites.workflow.WfConnComod(inference=inference, verbose=False)
    mi, pvalues = wf.fit(ds, n_perm=n_perm, n_jobs=n_jobs)
    
    save_data(mi, f"{output_path_prefix}_mi.nc")
    save_data(pvalues, f"{output_path_prefix}_pvalues.nc")
    
    return f"WfConnComod completed. MI and p-values saved with prefix {output_path_prefix}"


# --- HOI Wrappers ---

@mcp.tool()
def hoi_oinfo(data_path: str, output_path: str, y_path: str = None, minsize: int = 2, maxsize: int = None) -> str:
    """
    O-information.

    The O-information is defined as the difference between the total
    correlation (TC) minus the dual total correlation (DTC).
    Positive values reflect redundancy, negative values reflect synergy.

    Parameters
    ----------
    data_path : str
        Path to input data (n_samples, n_features, [n_variables]).
    output_path : str
        Path to save output. Use .nc extension to preserve metadata (ROI names).
    y_path : str | None
        Path to task-related feature.
    minsize : int
        Minimum size of multiplets.
    maxsize : int | None
        Maximum size of multiplets.
    """
    x = load_data(data_path)
    y = load_data(y_path) if y_path else None
    
    model = hoi.metrics.Oinfo(x, y=y)
    result = model.fit(minsize=minsize, maxsize=maxsize)
    
    save_data(result, output_path)
    return f"O-info saved to {output_path}"

@mcp.tool()
def hoi_infotopo(data_path: str, output_path: str, minsize: int = 1, maxsize: int = None) -> str:
    """
    Topological Information.

    The multivariate mutual information Ik quantify the variability/randomness 
    and the statistical dependences between variables.

    Parameters
    ----------
    data_path : str
        Path to input data.
    output_path : str
        Path to save output. Use .nc extension to preserve metadata (ROI names).
    minsize : int
        Minimum multiplet size.
    maxsize : int | None
        Maximum multiplet size.
    """
    x = load_data(data_path)
    
    model = hoi.metrics.InfoTopo(x)
    result = model.fit(minsize=minsize, maxsize=maxsize)
    
    save_data(result, output_path)
    return f"InfoTopo saved to {output_path}"

@mcp.tool()
def hoi_redundancy_mmi(data_path: str, y_path: str, output_path: str, minsize: int = 2, maxsize: int = None) -> str:
    """
    Redundancy estimated using the Minimum Mutual Information.

    Parameters
    ----------
    data_path : str
        Path to input data.
    y_path : str
        Path to feature (y).
    output_path : str
        Path to save output. Use .nc extension to preserve metadata (ROI names).
    """
    x = load_data(data_path)
    y = load_data(y_path)
    
    model = hoi.metrics.RedundancyMMI(x, y)
    result = model.fit(minsize=minsize, maxsize=maxsize)
    
    save_data(result, output_path)
    return f"RedundancyMMI saved to {output_path}"

@mcp.tool()
def hoi_synergy_mmi(data_path: str, y_path: str, output_path: str, minsize: int = 2, maxsize: int = None) -> str:
    """
    Synergy estimated using the Minimum Mutual Information.

    Parameters
    ----------
    data_path : str
        Path to input data.
    y_path : str
        Path to feature (y).
    output_path : str
        Path to save output. Use .nc extension to preserve metadata (ROI names).
    """
    x = load_data(data_path)
    y = load_data(y_path)
    
    model = hoi.metrics.SynergyMMI(x, y)
    result = model.fit(minsize=minsize, maxsize=maxsize)
    
    save_data(result, output_path)
    return f"SynergyMMI saved to {output_path}"

@mcp.tool()
def hoi_rsi(data_path: str, y_path: str, output_path: str, minsize: int = 2, maxsize: int = None) -> str:
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
        Path to save output. Use .nc extension to preserve metadata (ROI names).
    """
    x = load_data(data_path)
    y = load_data(y_path)
    
    model = hoi.metrics.RSI(x, y)
    result = model.fit(minsize=minsize, maxsize=maxsize)
    
    save_data(result, output_path)
    return f"RSI saved to {output_path}"

@mcp.tool()
def hoi_dtc(data_path: str, output_path: str, y_path: str = None, minsize: int = 2, maxsize: int = None) -> str:
    """
    Dual Total Correlation (DTC).

    Parameters
    ----------
    data_path : str
        Path to input data.
    output_path : str
        Path to save output. Use .nc extension to preserve metadata (ROI names).
    """
    x = load_data(data_path)
    y = load_data(y_path) if y_path else None
    
    model = hoi.metrics.DTC(x, y=y)
    result = model.fit(minsize=minsize, maxsize=maxsize)
    
    save_data(result, output_path)
    return f"DTC saved to {output_path}"

if __name__ == "__main__":
    mcp.run()
