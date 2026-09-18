# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mcp<2",
#   "frites>=0.4.5",
#   "hoi",
#   "numpy",
#   "xarray",
#   "pandas",
#   "netcdf4",
#   "h5netcdf",
#   "scipy",
#   "mne",
#   "matplotlib",
#   "PyMuPDF",
# ]
# ///
"""End-to-end smoke test of the braina installation.

Usage:  uv run <plugin root>/scripts/smoke_test.py

Reports library versions and the JAX backend, then runs one Frites and one
HOI tool through the MCP wrappers on simulated data. Exit code 0 = healthy.
"""
import importlib.util
import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

here = os.path.dirname(os.path.abspath(__file__))
server_path = os.path.join(os.path.dirname(here), "mcp", "braina_mcp.py")


def main() -> int:
    import numpy as np
    import frites, hoi, jax, xarray

    print(f"frites {frites.__version__} | hoi {hoi.__version__} | numpy {np.__version__} | xarray {xarray.__version__}")
    devices = jax.devices()
    kind = "GPU" if any(d.platform != "cpu" for d in devices) else "CPU"
    print(f"JAX backend: {kind} ({', '.join(str(d) for d in devices)})")

    spec = importlib.util.spec_from_file_location("braina_mcp", server_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    failures = []
    with tempfile.TemporaryDirectory() as d:
        p = lambda f: os.path.join(d, f)  # noqa: E731
        steps = [
            ("frites_sim_ar", lambda: m.frites_sim_ar(p("ar.nc"), n_epochs=20, n_times=200, random_state=0)),
            ("frites_conn_covgc", lambda: m.frites_conn_covgc(p("ar.nc"), p("gc.nc"), dt=30, lag=2, t0=[80, 120])),
            ("frites_conn_net", lambda: m.frites_conn_net(p("gc.nc"), p("net.nc"), mean_trials=True)),
            ("hoi_oinfo", lambda: m.hoi_oinfo(p("h.nc"), p("oinfo.nc"), maxsize=3)),
            ("hoi_get_nbest_mult", lambda: m.hoi_get_nbest_mult(p("oinfo.nc"), p("best.csv"), n_best=2)),
            ("plot_result", lambda: m.plot_result(p("gc.nc"), p("gc.png"))),
        ]
        # HOI input: 4 named features
        xarray.DataArray(np.random.default_rng(0).standard_normal((100, 4)), dims=("samples", "roi"),
                         coords={"roi": list("ABCD")}).to_netcdf(p("h.nc"))
        for name, fn in steps:
            out = fn()
            status = "FAIL" if out.startswith("Error") else "ok"
            print(f"  [{status}] {name}: {out.splitlines()[0][:90]}")
            if status == "FAIL":
                failures.append(name)
    if failures:
        print(f"SMOKE TEST FAILED: {failures}")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
