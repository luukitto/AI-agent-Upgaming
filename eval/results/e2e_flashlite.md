## End-to-end evaluation (LLM gemini-3.5-flash-lite, judge gemini-3.5-flash-lite)

Single-shot ran on all 60 questions; the agentic run was stopped after 15 questions (categories: multi_hop, simple). The paired table compares both configs on exactly those 15 questions.

### Single-shot, all questions

| config | n | correctness | keyword_hit | citation_prec | citation_valid | abstain_acc | false_abstain | R@5 | MRR | latency_ms | llm_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single-shot | 60 | 0.864 | 0.787 | 0.973 | 1.0 | 0.667 | 0.111 | 0.873 | 0.874 | 18532.8 | 1 | 1827 |

### Paired: single-shot vs agentic on the same 15 questions

| config | n | correctness | keyword_hit | citation_prec | citation_valid | abstain_acc | false_abstain | R@5 | MRR | latency_ms | llm_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agentic | 15 | 0.967 | 1.0 | 0.978 | 1.0 | nan | 0.0 | 0.9 | 0.823 | 46283.733 | 2.4 | 4044.133 |
| single-shot | 15 | 0.833 | 0.867 | 1.0 | 1.0 | nan | 0.133 | 0.867 | 0.813 | 14716.333 | 1 | 1771.933 |

### Single-shot by category

| config | category | n | correctness | keyword_hit | citation_prec | citation_valid | abstain_acc | false_abstain | R@5 | MRR | latency_ms | llm_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single-shot | contradiction | 4 | 0.875 | 1.0 | 1.0 | 1.0 | nan | 0.0 | 1.0 | 0.875 | 29865.5 | 1 | 2420.25 |
| single-shot | hotpot_bridge | 14 | 0.857 | 0.571 | 1.0 | 1.0 | nan | 0.286 | 0.786 | 0.857 | 18196.214 | 1 | 1611.714 |
| single-shot | hotpot_comparison | 6 | 0.8 | 0.667 | 0.944 | 1.0 | nan | 0.0 | 1.0 | 1.0 | 14234.167 | 1 | 1574.667 |
| single-shot | injection | 2 | 1.0 | 1.0 | 1.0 | 1.0 | nan | 0.0 | 1.0 | 0.75 | 15140 | 1 | 1936.5 |
| single-shot | insufficient | 6 | 0.833 | nan | nan | 1.0 | 0.667 | nan | nan | nan | 23435.333 | 1 | 1788.167 |
| single-shot | multi_hop | 13 | 0.808 | 0.808 | 0.923 | 1.0 | nan | 0.077 | 0.782 | 0.923 | 19524.154 | 1 | 2108.462 |
| single-shot | postmortem | 3 | 1.0 | 1.0 | 1.0 | 1.0 | nan | 0.0 | 1.0 | 1.0 | 19209 | 1 | 1732 |
| single-shot | simple | 12 | 0.917 | 0.917 | 1.0 | 1.0 | nan | 0.083 | 0.917 | 0.767 | 14168.417 | 1 | 1726.583 |

### Paired by category

| config | category | n | correctness | keyword_hit | citation_prec | citation_valid | abstain_acc | false_abstain | R@5 | MRR | latency_ms | llm_calls | tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agentic | multi_hop | 3 | 0.833 | 1.0 | 0.889 | 1.0 | nan | 0.0 | 0.833 | 1.0 | 69256.333 | 3 | 7961 |
| agentic | simple | 12 | 1.0 | 1.0 | 1.0 | 1.0 | nan | 0.0 | 0.917 | 0.779 | 40540.583 | 2.25 | 3064.917 |
| single-shot | multi_hop | 3 | 0.5 | 0.667 | 1.0 | 1.0 | nan | 0.333 | 0.667 | 1.0 | 16908 | 1 | 1953.333 |
| single-shot | simple | 12 | 0.917 | 0.917 | 1.0 | 1.0 | nan | 0.083 | 0.917 | 0.767 | 14168.417 | 1 | 1726.583 |
