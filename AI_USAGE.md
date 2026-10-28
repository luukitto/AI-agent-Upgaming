# AI Usage Report

The brief allows AI tools and asks which ones were used, what they helped with, and what was decided or changed
independently. This report answers those three questions, plus how the AI-generated code was checked.

## Tools used

| Tool | Used for |
|---|---|
| Claude Code (Claude Opus), CLI + PyCharm | planning the architecture against the brief; writing first drafts of modules, the test suite, the eval harness, the UI and the documentation; running benchmarks, tests and evals; debugging API and quota problems |

No other AI assistants were used.

## What AI helped with

- **Planning.** Turned the brief into a requirements checklist and a build plan (ingestion → retrieval → agent → eval → UI → deployment).
- **Initial scaffold** (commit `397cb3b`): parsers for 7 formats, both chunkers, SHA-256 + SimHash dedup, Qdrant store,
  own BM25 / RRF / MMR, LLM client with retries and caching, tool-calling investigator agent, FastAPI app with SSE.
- **Data.** The reproducible HotpotQA subset (first 6,000 validation questions → 54,962 paragraphs), the post-mortem
  parser, and the synthetic NovaPay corpus with planted contradictions, duplicates, missing facts and a prompt injection.
- **Tests** with fakes (in-memory Qdrant, hashing embedder, scripted LLM). Writing them surfaced two real bugs, both fixed:
  - `FixedSizeChunker` started the overlap in the middle of a word;
  - the agent's `verify` trace step was appended to a list that pydantic had already copied, so it never appeared in the result.
- **Debugging integration problems:**
  - Gemini 3 tool calling failed with HTTP 400 on every turn after the first tool call. The fix was to echo the tool
    call's `extra_content` (thought signature) back to the model.
  - Profiling showed local embedding at ~15 docs/s; sorting batches by length made it ~2.5× faster.
  - The Gemini free tier counts every text in an embedding batch as a separate request.
- **Evaluation harness** (60 questions, retrieval / end-to-end / chunking runs), README, deployment files, and the script that copies the local index to Qdrant Cloud.

## What I decided / changed myself

- **Scope and constraints.** One-day deadline, every mandatory requirement first, bonus items only where cheap
  (agentic RAG, streaming, caching, prompt-injection tests).
- **LLM provider.** Chose Google Gemini (free tier, native tool calling). The `gemini-2.5-*` models returned 404 on
  my key, so after testing which models were available I switched to `gemini-3.5-flash` (agent) and
  `gemini-3.5-flash-lite` (decomposition, judge).
- **API keys.** My first key turned out to be an OpenRouter key with no credits. I replaced it with a Google AI Studio
  key and kept all keys in `.env`, which is git-ignored, never in code or chat.
- **Embeddings.** I first rejected the local model because it was too slow and moved to `gemini-embedding-001`. After
  measuring the free-tier quota (100 texts/min, so ~11 hours for the corpus, plus a daily cap), I went back to the
  local `bge-small-en-v1.5` with the speed fix. The Gemini embedder stays selectable with `EMBED_PROVIDER`.
- **Hosting.** Hugging Face Docker Spaces turned out to require a paid PRO plan, so I changed the deployment target
  to a free host instead of paying.
- **Workflow.** Set up the GitHub repository and moved development between two computers through it, using `HANDOFF.md` to carry the state over.

## How I verified AI-generated code

- **Automated tests.** `pytest -q` runs 37 offline tests covering the parsers, CRUD, dedup, BM25/RRF/MMR, the agent
  loop, citation checks, LLM failure handling, admin protection and SSE. They pass on both machines.
- **Measured instead of assumed.** Embedding throughput, API quotas and model availability were checked with small
  scripts before any design choice depended on them; several assumptions turned out to be wrong (see above).
- **Evaluation.** Retrieval, chunking and end-to-end runs on the 60-question set; results and failure analysis are in [EVAL_REPORT.md](EVAL_REPORT.md).
- **Manual checks.** Ran the NovaPay example question end to end and checked the cited chunks against the source files in `data/synthetic/`.
