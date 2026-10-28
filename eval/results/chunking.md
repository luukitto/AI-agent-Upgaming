## Chunking comparison (chunk_size=400, overlap=80, hybrid retrieval)

### Chunking: overall

| config | n | R@1 | R@3 | R@5 | R@10 | MRR | latency_ms |
|---|---|---|---|---|---|---|---|
| fixed-400 | 54 | 0.546 | 0.827 | 0.889 | 0.944 | 0.912 | 31.87 |
| sentence-400 | 54 | 0.556 | 0.827 | 0.883 | 0.948 | 0.913 | 10.259 |

### Chunking: by category

| config | category | n | R@1 | R@3 | R@5 | R@10 | MRR | latency_ms |
|---|---|---|---|---|---|---|---|---|
| fixed-400 | contradiction | 4 | 0.5 | 0.875 | 1.0 | 1.0 | 1.0 | 30.25 |
| fixed-400 | hotpot_bridge | 14 | 0.429 | 0.714 | 0.786 | 0.929 | 0.929 | 33.214 |
| fixed-400 | hotpot_comparison | 6 | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 | 28.5 |
| fixed-400 | injection | 2 | 0.5 | 1.0 | 1.0 | 1.0 | 0.667 | 27.5 |
| fixed-400 | multi_hop | 13 | 0.5 | 0.782 | 0.846 | 0.846 | 0.962 | 34.385 |
| fixed-400 | postmortem | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 34.667 |
| fixed-400 | simple | 12 | 0.667 | 0.833 | 0.917 | 1.0 | 0.783 | 29.833 |
| sentence-400 | contradiction | 4 | 0.5 | 0.75 | 1.0 | 1.0 | 1.0 | 11 |
| sentence-400 | hotpot_bridge | 14 | 0.429 | 0.714 | 0.75 | 0.929 | 0.929 | 6.429 |
| sentence-400 | hotpot_comparison | 6 | 0.5 | 1.0 | 1.0 | 1.0 | 1.0 | 6.833 |
| sentence-400 | injection | 2 | 0.5 | 1.0 | 1.0 | 1.0 | 0.667 | 11.5 |
| sentence-400 | multi_hop | 13 | 0.462 | 0.744 | 0.859 | 0.859 | 0.923 | 13.154 |
| sentence-400 | postmortem | 3 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 14 |
| sentence-400 | simple | 12 | 0.75 | 0.917 | 0.917 | 1.0 | 0.831 | 11.917 |
