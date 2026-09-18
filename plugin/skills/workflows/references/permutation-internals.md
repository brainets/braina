# How the permutation test works (frites/stats)

All statistics are non-parametric and permutation-based. The logic is the
same whichever workflow produced the effect.

1. **Null distribution.** `WfMi` / `WfConnComod` shuffle the regressor `y`
   (or swap trials) **within each subject** (`frites.stats.permute_mi_vector`
   / `permute_mi_trials`) and recompute the metric `n_perm` times.
   Within-subject shuffling keeps exchangeability under the null while
   respecting the study structure. `random_state` seeds this. For
   `frites_wf_stats` the caller supplies the null.
2. **`inference`.**
   - `rfx`: a one-sample t-test **across subjects** (`frites.stats.rfx_ttest`)
     on the true effect and on every permutation, with a small "hat"
     regularisation `sigma=0.001` that stops near-zero variance from
     inflating t-values. This is what makes conclusions population-level:
     the test is whether the population mean differs from zero. Needs >= 2
     subjects (Frites asserts `n_subjects > 1`).
   - `ffx`: no across-subject test; the pooled effect is compared with the
     pooled permutation null. Claims hold only for the tested data.
3. **`mcp`.**
   - `cluster` (`frites.stats.cluster_correction_mcp`): pick a
     cluster-forming threshold - `cluster_th=None` uses the
     `1 - cluster_alpha` percentile of the permutation distribution (95th
     for `cluster_alpha=0.05`); `'tfce'` uses threshold-free cluster
     enhancement - group contiguous supra-threshold points into clusters,
     and compare each cluster's mass with the null distribution of maximum
     cluster masses. One correction per cluster, not per time point: the
     most powerful choice when true effects are contiguous.
   - `maxstat` / `fdr` / `bonferroni` (`frites.stats.testwise_correction_mcp`):
     correct every test (time point, ROI, frequency) independently. Frites'
     own docstring: "usually suffers from a low statistical power".
   - `nostat` / `None`: no correction - exploration only.
4. **`tail`.** `1`: effect greater than null; `-1`: smaller; `0`: either.
5. **p-value resolution.** The smallest p-value is `1 / (n_perm + 1)`; with
   `n_perm=1000` that is 0.001, so a Bonferroni threshold below that cannot
   be reached - raise `n_perm` instead of reading "p = 0".

References: Combrisson et al. 2022, JOSS (Frites) -
https://github.com/brainets/braina/blob/main/papers/Combrisson_joss_2022_frites_toolbox.pdf;
Ince et al. 2017, Hum Brain Mapp (Gaussian-copula MI) -
https://github.com/brainets/braina/blob/main/papers/Ince_hbm_2017_gaussian_copula_information_theory.pdf.
(In a repo clone: `papers/`, readable with `read_pdf`.)
