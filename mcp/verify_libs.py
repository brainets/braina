# /// script
# dependencies = [
#   "frites",
#   "hoi",
#   "numpy<2.0",
#   "xarray",
#   "pandas"
# ]
# ///

import numpy as np
import xarray as xr
import frites.conn
import frites.workflow
import frites.dataset
import frites.simulations
from hoi.metrics import Oinfo, GradientOinfo
from hoi.utils import get_nbest_mult

def test_frites_conn():
    print("Testing Frites Connectivity...")
    # Simulate (n_epochs, n_roi, n_times)
    n_epochs, n_roi, n_times = 10, 3, 50
    data = np.random.rand(n_epochs, n_roi, n_times)
    
    # Test conn_covgc
    dt = 10
    lag = 2
    t0 = [20]
    
    gc = frites.conn.conn_covgc(data, dt=dt, lag=lag, t0=t0, n_jobs=1, verbose=False)
    print(f"  conn_covgc success. Output shape: {gc.shape}")

def test_frites_extra():
    print("Testing Frites Extra (Spec, CCF, Sim)...")
    
    # 1. Simulation
    ar = frites.simulations.StimSpecAR(verbose=False)
    data = ar.fit(n_epochs=10, n_times=1000, n_stim=2, random_state=0)
    sfreq = 200.0 # default in StimSpecAR
    
    print(f"  StimSpecAR success. Shape: {data.shape}")
    
    # 2. Spectral Connectivity
    freqs = np.array([10, 20, 30])
    conn = frites.conn.conn_spec(data, freqs=freqs, metric='coh', sfreq=sfreq, n_jobs=1, verbose=False)
    print(f"  conn_spec success. Shape: {conn.shape}")
    
    # 3. CCF
    ccf = frites.conn.conn_ccf(data, n_jobs=1, verbose=False)
    print(f"  conn_ccf success. Shape: {ccf.shape}")

def test_hoi():
    print("Testing HOI...")
    # Simulate (n_samples, n_features)
    n_samples, n_features = 50, 5
    x = np.random.rand(n_samples, n_features)
    
    model = Oinfo(x, verbose=False)
    # Fit O-info
    hoi_val = model.fit(minsize=3, maxsize=3)
    print(f"  Oinfo success. Output shape: {hoi_val.shape}")

def test_hoi_extra():
    print("Testing HOI Extra (Gradient, nbest)...")
    n_samples, n_features = 50, 4
    x = np.random.rand(n_samples, n_features)
    y = np.random.rand(n_samples) # Target for Gradient
    
    # Gradient O-info
    model = GradientOinfo(x, y, verbose=False)
    hoi_val = model.fit(minsize=2, maxsize=3)
    print(f"  GradientOinfo success. Shape: {hoi_val.shape}")
    
    # get_nbest_mult
    df = get_nbest_mult(hoi_val, model=model, n_best=3)
    print(f"  get_nbest_mult success. Rows: {len(df)}")

def test_frites_workflows():
    print("Testing Frites Workflows...")
    n_subjects = 3
    n_epochs = 10
    n_times = 20
    n_roi = 2
    n_perm = 5
    
    # 1. WfStats
    print("  Testing WfStats...")
    effect = [np.random.rand(n_subjects, n_times) for _ in range(n_roi)]
    perms = [np.random.rand(n_perm, n_subjects, n_times) for _ in range(n_roi)]
    
    wf_stats = frites.workflow.WfStats(verbose=False)
    pv, tv = wf_stats.fit(effect, perms, inference='rfx', mcp='cluster', cluster_th=None)
    print(f"  WfStats success. P-values shape: {pv.shape}")

    # 2. WfMi & WfConnComod
    print("  Testing WfMi and WfConnComod...")
    data_list = [np.random.rand(n_epochs, n_roi, n_times) for _ in range(n_subjects)]
    y_list = [np.random.rand(n_epochs) for _ in range(n_subjects)]
    roi_names = np.array([f"roi_{i}" for i in range(n_roi)])
    roi_input = [roi_names] * n_subjects
    times = np.arange(n_times) / 64.
    
    ds = frites.dataset.DatasetEphy(data_list, y=y_list, roi=roi_input, times=times, verbose=False)
    
    wf_mi = frites.workflow.WfMi(mi_type='cc', inference='rfx', verbose=False)
    mi, pv_mi = wf_mi.fit(ds, n_perm=n_perm, n_jobs=1)
    print(f"  WfMi success. MI shape: {mi.shape}")
    
    wf_conn = frites.workflow.WfConnComod(inference='rfx', verbose=False)
    mi_c, pv_c = wf_conn.fit(ds, n_perm=n_perm, n_jobs=1)
    print(f"  WfConnComod success. MI shape: {mi_c.shape}")

if __name__ == "__main__":
    try:
        test_frites_conn()
        test_frites_extra()
        test_hoi()
        test_hoi_extra()
        test_frites_workflows()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        exit(1)
