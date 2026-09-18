---
type: llm
---

PASS if the assistant saves the HOI result as a .nc file, uses hoi_get_nbest_mult (or the multiplet_names coordinate of the output) to report actual region-name combinations such as "PFC / MT / V1" rather than bare row indices, and states whether the top values are positive (redundancy) or negative (synergy) for O-information, noting that the tool returns the most positive and the most negative multiplets separately.
FAIL if it reports only numeric indices, mislabels the sign convention, or does not run the tools.
