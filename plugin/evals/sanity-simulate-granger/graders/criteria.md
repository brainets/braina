---
type: llm
---

PASS if the assistant actually called the braina simulation tool (frites_sim_ar) and then the Granger causality tool (frites_conn_covgc) rather than only describing what it would do, reported where the output files were saved, and gave a plain-language interpretation of which region drives which (and roughly when, relative to stimulus onset).
FAIL if no tool was called, if a tool error was ignored or glossed over, or if the answer is a generic explanation of Granger causality without results from the run.
