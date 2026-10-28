# Work-in-progress notes (delete before submission)

## Setup on a new machine
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env          # then put your Google AI Studio key in LLM_API_KEY
python -m scripts.build_corpus  # re-downloads HotpotQA -> data/corpus/hotpot_*.jsonl (git-ignored)
```

## Done
- Parsers (pdf/docx/md/txt/html/csv/json), chunkers (fixed vs sentence), dedup (sha256 exact + simhash near-dup)
- Qdrant store, own BM25, RRF fusion, MMR, hybrid retriever
- LLM client (retries/backoff, usage, disk cache), tool-calling investigator agent, FastAPI app with SSE
- Corpus: ~55k HotpotQA paragraphs, 244 danluu post-mortems, synthetic NovaPay docs with planted contradictions
- Embeddings switched from local fastembed (too slow: ~15 docs/s) to Gemini API `gemini-embedding-001` (768-d)
- Chat models: `gemini-2.5-*` return 404 on the new key -> defaults now `gemini-3.5-flash` / `gemini-3.5-flash-lite` (verified working)

## Blocker: embedding quota
Free tier = 100 embed requests/min, and each text in a batch counts as one request
(~100 texts/min, so ~11 h for ~67k chunks, plus a daily cap). Options:
1. Enable billing on the Google project (paid tier is fast; roughly ~$1-2 for the whole corpus), or
2. Index a smaller dense subset (would weaken the 50k requirement).

## Next steps
1. Resolve embedding quota, then: `python -m scripts.ingest_corpus --limit 2000` (smoke), then full run
2. Verify tool calling works end-to-end with gemini-3.5-flash
3. Eval harness (40+ questions, Recall@K, MRR, correctness, citation accuracy, 2 configs) + report
4. UI, deployment (HF Spaces + Qdrant Cloud), README, AI usage report, tests
