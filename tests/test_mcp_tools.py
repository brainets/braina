"""End-to-end tests of every braina MCP tool on tiny synthetic data.

The tools are called as plain Python functions (the `@tool` decorator returns
the function it registered), so no MCP transport is involved. Run with
`uv run tests/run_tests.py` so that the server's dependencies are available.
"""
import asyncio
import importlib.util
import os
import warnings

import numpy as np
import pytest
import xarray as xr

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(ROOT, "plugin", "mcp", "braina_mcp.py")

N_EP, N_ROI, N_T, SFREQ = 15, 3, 300, 100.0


@pytest.fixture(scope="session")
def m():
    spec = importlib.util.spec_from_file_location("braina_mcp", SERVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def d(tmp_path_factory):
    """Directory with reusable synthetic inputs."""
    d = tmp_path_factory.mktemp("braina")
    rng = np.random.default_rng(0)
    x = xr.DataArray(rng.standard_normal((N_EP, N_ROI, N_T)), dims=("trials", "roi", "times"),
                     coords={"roi": ["A", "B", "C"], "times": np.arange(N_T) / SFREQ - 1.0},
                     attrs={"sfreq": SFREQ})
    x.to_netcdf(d / "x.nc")
    np.save(d / "x.npy", x.values)
    np.save(d / "y.npy", rng.standard_normal(N_EP))
    for s in range(3):
        xr.DataArray(rng.standard_normal((N_EP, N_ROI, 60)), dims=("trials", "roi", "times"),
                     coords={"roi": ["A", "B", "C"], "times": np.arange(60) / SFREQ}).to_netcdf(d / f"s{s}.nc")
        np.save(d / f"y{s}.npy", rng.standard_normal(N_EP))
        np.save(d / f"z{s}.npy", rng.standard_normal(N_EP))
    h = xr.DataArray(rng.standard_normal((120, 4)), dims=("samples", "roi"), coords={"roi": list("ABCD")})
    h.to_netcdf(d / "h.nc")
    np.save(d / "h.npy", h.values)
    np.save(d / "hy.npy", rng.standard_normal(120))
    xr.DataArray(rng.standard_normal((150, 3, 30)), dims=("samples", "roi", "times"),
                 coords={"roi": list("ABC")}).to_netcdf(d / "d3.nc")
    return d


def ok(out):
    assert not out.startswith("Error"), out
    return out


def subs(d):
    return [str(d / f"s{s}.nc") for s in range(3)], [str(d / f"y{s}.npy") for s in range(3)]


# ----------------------------------------------------------------- registry

def test_tools_registered(m):
    tools = asyncio.run(m.server.list_tools())
    names = {t.name for t in tools}
    assert len(tools) == 34
    for n in ["inspect_data", "read_pdf", "convert_to_nc", "plot_result", "frites_conn_covgc",
              "frites_wf_mi", "frites_wf_mi_combine", "frites_conn_reshape", "frites_conn_net",
              "hoi_oinfo", "hoi_get_nbest_mult", "hoi_tc", "hoi_transfer_entropy"]:
        assert n in names
    wf = next(t for t in tools if t.name == "frites_wf_mi")
    assert "anyOf" in wf.inputSchema["properties"]["data_path"]  # str | list[str]


def test_errors_are_returned_not_raised(m, d):
    out = m.frites_conn_ccf(str(d / "missing.nc"), str(d / "o.nc"))
    assert out.startswith("Error in frites_conn_ccf: FileNotFoundError")


# ----------------------------------------------------------------- I/O

def test_inspect_data(m, d):
    out = ok(m.inspect_data(str(d / "x.nc")))
    assert "dims: ('trials', 'roi', 'times')" in out and "sfreq" in out and "min=" in out


def test_read_pdf(m):
    pdf = os.path.join(ROOT, "papers", "Combrisson_joss_2022_frites_toolbox.pdf")
    if not os.path.exists(pdf):
        pytest.skip("papers/ not present")
    assert "Frites" in ok(m.read_pdf(pdf))[:200]


# ----------------------------------------------------------------- simulation

def test_sim_ar_reproducible_with_sfreq(m, d):
    ok(m.frites_sim_ar(str(d / "ar1.nc"), n_epochs=10, n_times=200, random_state=3))
    ok(m.frites_sim_ar(str(d / "ar2.nc"), n_epochs=10, n_times=200, random_state=3))
    a, b = xr.load_dataarray(d / "ar1.nc"), xr.load_dataarray(d / "ar2.nc")
    assert np.allclose(a.values, b.values)
    assert a.attrs["sfreq"] == 200.0 and a.dims == ("trials", "roi", "times")


# ----------------------------------------------------------------- connectivity

def test_covgc_keeps_metadata(m, d):
    ok(m.frites_conn_covgc(str(d / "x.nc"), str(d / "gc.nc"), dt=50, lag=2, t0=[100, 200], norm=True))
    gc = xr.load_dataarray(d / "gc.nc")
    assert list(gc["roi"].values) == ["A-B", "A-C", "B-C"]
    assert gc.dims == ("trials", "roi", "times", "direction")
    assert abs(gc.attrs["sfreq"] - SFREQ) < 1e-6
    assert gc["times"].values[0] < 10  # seconds, not samples


def test_covgc_npy_falls_back_to_default_labels(m, d):
    ok(m.frites_conn_covgc(str(d / "x.npy"), str(d / "gc2.nc"), dt=50, lag=2, t0=[100]))
    assert list(xr.load_dataarray(d / "gc2.nc")["roi"].values)[0] == "roi_0-roi_1"


@pytest.mark.parametrize("estimator", ["gcmi", "pearson"])
def test_dfc(m, d, estimator):
    ok(m.frites_conn_dfc(str(d / "x.nc"), str(d / f"dfc_{estimator}.nc"),
                         win_sample=[[0, 149], [150, 299]], estimator=estimator))
    dfc = xr.load_dataarray(d / f"dfc_{estimator}.nc")
    assert dfc.shape == (N_EP, 3, 2) and list(dfc["roi"].values)[0] == "A-B"


def test_dfc_unknown_estimator(m, d):
    assert "Unknown estimator" in m.frites_conn_dfc(str(d / "x.nc"), str(d / "bad.nc"), estimator="foo")


def test_pid_ii(m, d):
    ok(m.frites_conn_pid(str(d / "x.nc"), str(d / "y.npy"), str(d / "pid")))
    for c in ["infotot", "unique", "redundancy", "synergy"]:
        assert (d / f"pid_{c}.nc").exists()
    ok(m.frites_conn_ii(str(d / "x.nc"), str(d / "y.npy"), str(d / "ii.nc")))
    assert xr.load_dataarray(d / "ii.nc").dims == ("roi", "times")


def test_te_fit_ccf(m, d):
    ok(m.frites_conn_te(str(d / "x.nc"), str(d / "te.nc"), max_delay=5))
    te = xr.load_dataarray(d / "te.nc")
    assert list(te["roi"].values)[0] == "A->B" and te.attrs["sfreq"] == SFREQ
    ok(m.frites_conn_fit(str(d / "x.nc"), str(d / "y.npy"), str(d / "fit.nc"), max_delay=0.05))
    ok(m.frites_conn_ccf(str(d / "x.nc"), str(d / "ccf.nc"), normalized=False))


def test_spec(m, d):
    ok(m.frites_conn_spec(str(d / "x.nc"), str(d / "spec.nc"), freqs=[10, 20]))
    assert xr.load_dataarray(d / "spec.nc").dims == ("trials", "roi", "freqs", "times")
    out = m.frites_conn_spec(str(d / "y.npy"), str(d / "bad.nc"), freqs=[10])
    assert "sampling frequency" in out


def test_reshape_and_net(m, d):
    ok(m.frites_conn_covgc(str(d / "x.nc"), str(d / "gc.nc"), dt=50, lag=2, t0=[100, 200]))
    ok(m.frites_conn_te(str(d / "x.nc"), str(d / "te.nc"), max_delay=5))
    ok(m.frites_conn_dfc(str(d / "x.nc"), str(d / "dfc.nc")))
    ok(m.frites_conn_reshape(str(d / "gc.nc"), str(d / "gcmat.nc"), directed=True))
    mat = xr.load_dataarray(d / "gcmat.nc")
    assert mat.dims == ("sources", "targets", "times") and list(mat["sources"].values) == ["A", "B", "C"]
    ok(m.frites_conn_reshape(str(d / "te.nc"), str(d / "temat.nc"), directed=True))
    assert xr.load_dataarray(d / "temat.nc").shape[:2] == (3, 3)
    ok(m.frites_conn_reshape(str(d / "dfc.nc"), str(d / "dfcmat.nc"), directed=False))
    ok(m.frites_conn_net(str(d / "gc.nc"), str(d / "gcnet.nc")))
    assert xr.load_dataarray(d / "gcnet.nc").dims == ("trials", "roi", "times")
    ok(m.frites_conn_net(str(d / "te.nc"), str(d / "tenet.nc")))


# ----------------------------------------------------------------- workflows

def test_wf_mi_rfx_requires_two_subjects(m, d):
    out = m.frites_wf_mi(str(d / "x.nc"), str(d / "y.npy"), str(d / "w"), n_perm=5)
    assert "inference='rfx'" in out and "at least 2 subjects" in out
    ok(m.frites_wf_mi(str(d / "x.nc"), str(d / "y.npy"), str(d / "w"), inference="ffx", n_perm=5))


def test_wf_mi_multi_subject_and_combine(m, d):
    S, Y = subs(d)
    Z = [str(d / f"z{s}.npy") for s in range(3)]
    out = ok(m.frites_wf_mi(S, Y, str(d / "w1"), n_perm=10, mcp="fdr", random_state=0,
                            estimator="pearson", save_workflow=True))
    assert "3 subject(s)" in out and (d / "w1_wf.pkl").exists()
    mi = xr.load_dataarray(d / "w1_mi.nc")
    assert mi.dims == ("times", "roi") and list(mi["roi"].values) == ["A", "B", "C"]
    ok(m.frites_wf_mi(S, Z, str(d / "w2"), n_perm=10, mcp="fdr", random_state=0,
                      estimator="pearson", save_workflow=True))
    ok(m.frites_wf_mi_combine(str(d / "w1_wf.pkl"), str(d / "w2_wf.pkl"), str(d / "wc"), mcp="fdr"))
    assert xr.load_dataarray(d / "wc_pvalues.nc").shape == (60, 3)


def test_wf_mi_subject_dim_and_mismatch(m, d):
    S, Y = subs(d)
    big = xr.concat([xr.load_dataarray(p) for p in S], dim="subject")
    big.to_netcdf(d / "all.nc")
    xr.DataArray(np.stack([np.load(p) for p in Y]), dims=("subject", "trials")).to_netcdf(d / "yall.nc")
    assert "3 subject(s)" in ok(m.frites_wf_mi(str(d / "all.nc"), str(d / "yall.nc"), str(d / "wa"), n_perm=5))
    assert "Got 2 data" in m.frites_wf_mi(S[:2], Y[0], str(d / "bad"))


def test_wf_conn_comod(m, d):
    S, _ = subs(d)
    ok(m.frites_wf_conn_comod(S, str(d / "cm"), n_perm=5, mcp="maxstat", random_state=1))
    assert xr.load_dataarray(d / "cm_mi.nc").dims == ("times", "roi")


def test_wf_stats(m, d):
    rng = np.random.default_rng(1)
    eff = xr.DataArray(rng.standard_normal((2, 4, 40)), dims=("roi", "subjects", "times"),
                       coords={"roi": ["A", "B"], "times": np.arange(40) / SFREQ})
    eff.to_netcdf(d / "eff.nc")
    np.save(d / "perm.npy", rng.standard_normal((10, 2, 4, 40)))
    ok(m.frites_wf_stats(str(d / "eff.nc"), str(d / "perm.npy"), str(d / "st")))
    pv = xr.load_dataarray(d / "st_pvalues.nc")
    assert pv.dims == ("times", "roi") and list(pv["roi"].values) == ["A", "B"]
    assert (d / "st_tvalues.nc").exists()


# ----------------------------------------------------------------- HOI

def test_hoi_metadata_and_nbest(m, d):
    ok(m.hoi_oinfo(str(d / "h.nc"), str(d / "oinfo.nc"), maxsize=3))
    da = xr.load_dataarray(d / "oinfo.nc")
    assert da.dims == ("multiplets", "variables") and da.shape == (10, 1)
    assert list(da["order"].values) == [2] * 6 + [3] * 4
    assert str(da["multiplet_names"].values[-1]) == "B / C / D"
    out = ok(m.hoi_get_nbest_mult(str(d / "oinfo.nc"), str(d / "best.csv"), n_best=2))
    assert "most positive" in out and "A /" in out
    ok(m.hoi_oinfo(str(d / "h.nc"), str(d / "oinfo.npy"), maxsize=3))
    assert "no multiplet metadata" in m.hoi_get_nbest_mult(str(d / "oinfo.npy"), str(d / "b.csv"))


def test_hoi_npy_input_has_indices_only(m, d):
    ok(m.hoi_oinfo(str(d / "h.npy"), str(d / "oinfo2.nc"), maxsize=3))
    da = xr.load_dataarray(d / "oinfo2.nc")
    assert "multiplet_names" not in da.coords and str(da["multiplets"].values[0]) == "0,1"


@pytest.mark.parametrize("tool,needs_y", [
    ("hoi_gradient_oinfo", True), ("hoi_redundancy_mmi", True), ("hoi_synergy_mmi", True),
    ("hoi_rsi", True), ("hoi_infotot", True), ("hoi_dtc", False), ("hoi_tc", False),
    ("hoi_sinfo", False), ("hoi_infotopo", None),
])
def test_hoi_metrics(m, d, tool, needs_y):
    fn = getattr(m, tool)
    out = str(d / f"{tool}.nc")
    if needs_y is True:
        ok(fn(str(d / "h.nc"), str(d / "hy.npy"), out, maxsize=3))
    elif needs_y is False:
        ok(fn(str(d / "h.nc"), out, y_path=str(d / "hy.npy"), maxsize=3))
    else:
        ok(fn(str(d / "h.nc"), out, maxsize=3))
    assert "order" in xr.load_dataarray(out).coords


def test_hoi_method_samples_bootstrap(m, d):
    ok(m.hoi_oinfo(str(d / "h.nc"), str(d / "o_knn.nc"), maxsize=3, method="knn"))
    np.save(d / "idx.npy", np.arange(0, 120, 2))
    ok(m.hoi_oinfo(str(d / "h.nc"), str(d / "o_s.nc"), maxsize=3, samples_path=str(d / "idx.npy")))
    out = ok(m.hoi_rsi(str(d / "h.nc"), str(d / "hy.npy"), str(d / "rsi.nc"), maxsize=3, n_boots=4))
    assert "Bootstrap CI" in out
    ci = xr.load_dataarray(d / "rsi_ci.nc")
    assert ci.dims == ("ci", "multiplets", "variables") and list(ci["ci"].values) == ["low", "high"]
    assert (ci.sel(ci="low") <= ci.sel(ci="high")).all()


def test_hoi_dynamic(m, d):
    ok(m.hoi_transfer_entropy(str(d / "d3.nc"), str(d / "hte.nc")))
    assert xr.load_dataarray(d / "hte.nc").shape == (6, 30)
    ok(m.hoi_dotot(str(d / "d3.nc"), str(d / "dotot.nc")))
    ok(m.hoi_redundancy_phiid(str(d / "d3.nc"), str(d / "rphi.nc")))
    ok(m.hoi_atoms_phiid(str(d / "d3.nc"), str(d / "aphi.nc"), atoms=["sts", "rtr"]))
    assert "needs a 3D input" in m.hoi_transfer_entropy(str(d / "h.nc"), str(d / "bad.nc"))


# ----------------------------------------------------------------- conversion / plotting

def test_convert_csv_mat_npy_fif(m, d):
    import pandas as pd
    import scipy.io
    import mne
    rng = np.random.default_rng(2)
    pd.DataFrame(rng.standard_normal((50, 3)), columns=["V1", "V2", "V3"]).to_csv(d / "d.csv", index=False)
    ok(m.convert_to_nc(str(d / "d.csv"), str(d / "csv.nc")))
    c = xr.load_dataarray(d / "csv.nc")
    assert c.dims == ("samples", "roi") and list(c["roi"].values) == ["V1", "V2", "V3"]
    scipy.io.savemat(d / "d.mat", {"lfp": rng.standard_normal((8, 2, 100)), "other": np.arange(3)})
    assert "Pass var_name" in m.convert_to_nc(str(d / "d.mat"), str(d / "mat.nc"))
    ok(m.convert_to_nc(str(d / "d.mat"), str(d / "mat.nc"), var_name="lfp", roi=["r1", "r2"], sfreq=250., t0=-0.2))
    mt = xr.load_dataarray(d / "mat.nc")
    assert mt.attrs["sfreq"] == 250. and abs(mt["times"].values[0] + 0.2) < 1e-9
    np.save(d / "raw.npy", rng.standard_normal((8, 2, 100)))
    out = ok(m.convert_to_nc(str(d / "raw.npy"), str(d / "raw.nc")))
    assert "Warning: no roi names" in out and "Warning: no time axis" in out
    info = mne.create_info(["Fz", "Cz", "Pz"], 100., "eeg")
    mne.EpochsArray(rng.standard_normal((6, 3, 50)), info, tmin=-0.1, verbose=False).save(d / "t-epo.fif", overwrite=True, verbose=False)
    ok(m.convert_to_nc(str(d / "t-epo.fif"), str(d / "fif.nc")))
    f = xr.load_dataarray(d / "fif.nc")
    assert list(f["roi"].values) == ["Fz", "Cz", "Pz"] and f.attrs["sfreq"] == 100. and abs(f["times"].values[0] + 0.1) < 1e-9


def test_plot_result(m, d):
    ok(m.frites_conn_covgc(str(d / "x.nc"), str(d / "gc.nc"), dt=50, lag=2, t0=[100, 200]))
    ok(m.frites_conn_spec(str(d / "x.nc"), str(d / "spec.nc"), freqs=[10, 20]))
    ok(m.hoi_oinfo(str(d / "h.nc"), str(d / "oinfo.nc"), maxsize=3))
    ok(m.frites_conn_reshape(str(d / "gc.nc"), str(d / "gcmat.nc"), directed=True))
    for src, kw in [("gc.nc", {}), ("spec.nc", {}), ("spec.nc", {"mean_dims": ["freqs"]}),
                    ("oinfo.nc", {}), ("gcmat.nc", {}), ("x.nc", {"roi": ["A"]})]:
        png = d / (src.replace(".nc", "") + str(len(kw)) + ".png")
        ok(m.plot_result(str(d / src), str(png), **kw))
        assert png.exists() and png.stat().st_size > 1000
    np.save(d / "m2.npy", np.random.rand(5, 7))
    ok(m.plot_result(str(d / "m2.npy"), str(d / "m2.png")))
