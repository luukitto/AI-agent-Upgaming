# Work-in-progress notes (delete before submission)

## Setup on a new machine
See README §4. Corpus files are git-ignored: run `python -m scripts.build_corpus` first.

## Done
- Everything in the scaffold commit (ingestion, Qdrant, own BM25/RRF/MMR, LLM client, agent, API)
- Web UI (`app/static/index.html`): investigate with live SSE progress, search with per-retriever ranks, document CRUD
- 36 offline tests (`pytest -q`), with fixes for the chunker overlap and lost verify-trace bugs they found
- Eval: 60 questions (`eval/questions.jsonl`) + harness (`eval/run_eval.py`: retrieval / e2e / chunking)
- Dockerfile for HF Spaces, README, EVAL_REPORT skeleton, AI_USAGE draft

## Remaining (needs API key / accounts)
1. `.env` with `LLM_API_KEY`. Embedding quota decision: enable billing (~$1 total) or wait several days on the free tier
2. Qdrant Cloud cluster → set `QDRANT_URL`/`QDRANT_API_KEY` → `python -m scripts.ingest_corpus` (straight to the cloud)
3. Smoke-test: `uvicorn app.api.main:app`, ask the NovaPay example question in the UI
4. `python -m eval.run_eval retrieval`, `chunking`, `e2e` → fill in EVAL_REPORT §4 with the real numbers and analysis
5. HF Space (Docker) + secrets → push → put the URL in README
6. Finish AI_USAGE.md TODOs, delete this file
