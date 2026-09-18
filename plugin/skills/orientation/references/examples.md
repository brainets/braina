# Validated example scripts per tool

Paths are relative to `${CLAUDE_PLUGIN_ROOT}` (the installed plugin ships
`examples/`). Reuse their API usage instead of reconstructing calls from
memory; run with `uv run <path>` (PEP 723 inline dependencies).

| Tool | Example |
|---|---|
| `frites_conn_covgc` | `examples/frites/conn/plot_covgc.py`; conditional GC: `examples/frites/armodel/plot_ar_condcovgc.py` |
| `frites_conn_dfc` | `examples/frites/conn/plot_dfc.py`; estimators: `examples/frites/estimators/plot_est_comparison.py` |
| `frites_conn_pid` | `examples/frites/conn/plot_pid.py` |
| `frites_conn_ii` | `examples/frites/conn/plot_ii.py` |
| `frites_conn_fit` | `examples/frites/conn/plot_fit.py` |
| `frites_conn_ccf` | `examples/frites/conn/plot_ccf.py` |
| `frites_conn_reshape` / `frites_conn_net` | used inside `examples/frites/conn/plot_covgc.py` |
| `frites_wf_conn_comod` | `examples/frites/conn/plot_conn.py` |
| `frites_wf_mi` | `examples/frites/mi/plot_wf_mi_cc.py`, `plot_wf_mi_cd.py`, `plot_wf_mi_ccd.py` |
| `frites_wf_mi_combine` | `examples/frites/mi/plot_wf_mi_combine.py` |
| `frites_wf_stats` (`ffx` vs `rfx`) | `examples/frites/statistics/plot_wf_mi_stats_compare_ffx.py`, `..._rfx.py` |
| `frites_sim_ar` | `examples/frites/simulations/plot_ground_truth.py`, `examples/frites/armodel/plot_ar_pairwise.py` |
| `convert_to_nc` (MNE / xarray input) | `examples/frites/dataset/plot_dataset_mne.py`, `plot_dataset_xarray.py` |
| `hoi_oinfo` | `examples/hoi/metrics/plot_oinfo.py` |
| `hoi_infotopo` | `examples/hoi/metrics/plot_infotopo.py` |
| `hoi_redundancy_mmi` / `hoi_synergy_mmi` | `examples/hoi/metrics/plot_syn_red_mmi.py` |
| `hoi_rsi` | `examples/hoi/metrics/plot_rsi.py` |
| `hoi_atoms_phiid` | `examples/hoi/metrics/plot_syn_phiID.py` |
| bootstrap (`n_boots`) | `examples/hoi/statistics/plot_bootstrapping.py` |
| `hoi_get_nbest_mult` | used inside `plot_oinfo.py` / `plot_infotopo.py` |

No dedicated example for `frites_conn_te`, `frites_conn_spec`, `hoi_dtc`,
`hoi_tc`, `hoi_sinfo`, `hoi_infotot`, `hoi_gradient_oinfo`, `hoi_dotot`,
`hoi_transfer_entropy`, `hoi_redundancy_phiid`: rely on the topic skill's
reference file and the tool docstring, and say so rather than presenting an
improvised call with unwarranted confidence.
