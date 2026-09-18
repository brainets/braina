# Changelog

## 1.1.0 - 2026-09-18

### Fixed
- MCP server is declared in `plugin/.claude-plugin/plugin.json` (`mcpServers`);
  the root `.mcp.json`, which failed to resolve `${CLAUDE_PLUGIN_ROOT}` in a
  plain clone, is gone.
- The `mcp/` directory no longer shadows the `mcp` SDK package.
- Frites connectivity tools now pass the `roi`/`times` coordinate names to
  Frites, so `.nc` input really keeps ROI names, a time axis in seconds and
  `sfreq` (previously silently replaced by `roi_0, roi_1, ...` and 1 Hz).
- HOI tools convert `.nc` input to numpy (HOI rejects xarray) and keep the
  multiplet metadata (`multiplets`, `order`, `multiplet_names`) in their
  `.nc` output; `hoi_get_nbest_mult` resolves region names from it.
- `frites_wf_stats` writes `.nc` with `times`/`roi` coords; `frites_conn_pid`
  takes `dt` instead of a nonexistent `max_delay`; `frites_conn_te` forwards
  `sfreq`; random-effect inference with fewer than two subjects is refused.
- All tools catch exceptions and return `Error in <tool>: ...` strings, and
  return a summary (shape, coords, min/mean/max) of what they wrote.

### Added
- Multi-subject workflows: `frites_wf_mi` / `frites_wf_conn_comod` accept one
  file per subject or a `.nc` with a `subject` dimension, plus `mcp`,
  `cluster_th`, `cluster_alpha`, `random_state`, `estimator`.
- `frites_wf_mi_combine` (WfMiCombine) via `save_workflow=True` pickles.
- `frites_conn_reshape` (pairs -> sources x targets matrix) and
  `frites_conn_net` (A->B minus B->A).
- Extra Frites parameters: `norm`, `n_jobs` (covgc), `estimator` (dfc),
  `normalized` (ccf), `n_jobs` everywhere; `frites_sim_ar` gains `sf`, `dt`,
  `n_std`, `stim_onset`, `random_state` and stores `sfreq`.
- HOI: `method` (gc / gauss / binning / knn / kernel), `samples_path`, and
  bootstrap confidence intervals (`n_boots`, `ci_percentiles`,
  `random_state`, written to `<output>_ci.nc`) on every metric; new metrics
  `hoi_tc`, `hoi_sinfo`, `hoi_infotot`, `hoi_transfer_entropy`, `hoi_dotot`,
  `hoi_redundancy_phiid`, `hoi_atoms_phiid`.
- `convert_to_nc` (MNE `.fif` epochs, `.mat`, `.csv`, relabelling `.npy`) and
  `plot_result` (PNG of any braina output).
- `/braina:check` skill with `scripts/smoke_test.py`; SessionStart hook that
  pre-warms the `uv` environment; `uv` lockfile for the server script.
- pytest suite (`tests/`), GitHub Actions CI, `claude plugin eval` cases.

### Changed
- Repository layout: everything the plugin installs now lives under
  `plugin/` (manifest, MCP server, skills, examples, hooks, evals); papers,
  tutorials and use cases stay at the root and are no longer copied on
  install (68 MB -> ~2 MB).
- Skills restructured for progressive disclosure: short `SKILL.md` plus
  per-tool `references/` files; limitations fixed in 1.1.0 removed from
  the text; papers cited by DOI/URL.
- Tool count corrected (22 in 1.0.0, 34 in 1.1.0).

## 1.0.0 - 2026-09-13

Initial release: MCP server wrapping Frites and HOI (22 tools) and four
skills (orientation, frites-connectivity, hoi-metrics, workflows).
