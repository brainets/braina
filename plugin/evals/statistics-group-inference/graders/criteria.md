---
type: llm
---

PASS if the assistant (1) distinguishes fixed-effect from random-effect inference in plain language and explains why random-effect generalises to the population while fixed-effect only speaks about the tested subjects, (2) presents more than one multiple-comparisons correction (cluster, max-statistic, FDR, Bonferroni) with tradeoffs instead of silently picking one, and (3) points to the braina statistics tools (frites_wf_stats, or the built-in statistics of frites_wf_mi / frites_wf_conn_comod) with the permutation-based null they rely on.
FAIL if it recommends a parametric t-test with no correction, picks one correction without alternatives, or does not mention fixed vs random effects.
