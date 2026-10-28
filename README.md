---
title: AI Investigation Engine
sdk: docker
app_port: 7860
---

# AI Investigation Engine

An agentic RAG system that answers investigation questions across many documents. It retrieves evidence with
hybrid search (dense vectors plus its own BM25 implementation), plans multi-step questions, calls tools to dig
further, and cites the chunks behind every claim. It also reports contradictions between sources, and it
declines to answer when the evidence is missing.

> Example: *"Which system change caused the payment processing delays in March, which services were affected and
> which documents confirm it?"*. The engine finds the incident report (PDF), the change log (CSV), and the
> architecture doc (MD). It answers **CHG-2031**, lists the three affected services, cites the chunks, and flags
> that the Q1 review (HTML) names a *different* root cause.

- **Live demo:** `<add Hugging Face Space URL>`
- **Evaluation report:** [EVAL_REPORT.md](EVAL_REPORT.md) · **AI usage report:** [AI_USAGE.md](AI_USAGE.md)

---

## 1. Architecture

```
                       ┌──────────────────────── FastAPI (app/api) ─────────────────────────┐
  Browser UI ──HTTP──▶ │ /api/documents (CRUD)  /api/search  /api/ask  /api/ask/stream (SSE) │
 (app/static)          └───────┬──────────────────────┬──────────────────────┬──────────────┘
                               │                      │                      │
                    IngestionService          HybridRetriever           Investigator (agent)
                    parse → dedup → chunk     dense ─┐                  1 decompose (fast LLM)
                    → embed → upsert          BM25  ─┼▶ RRF ▶ MMR       2 seed retrieval per sub-question
                               │              (own)  ┘                  3 tool loop: search_documents /
                               │                      │                   get_document_context / submit_answer
                               ▼                      ▼                 4 verify citations vs evidence log
                  ┌──────── Qdrant ────────┐   in-memory inverted              │
                  │ chunks: 384-d vectors  │   index (pickled, rebuilt         ▼
                  │   + text/metadata      │   from Qdrant on start)   Gemini chat (OpenAI-compatible API)
                  │ documents: registry    │                           retries · backoff · disk cache
                  └────────────────────────┘
```

| Layer | Module | Notes |
|---|---|---|
| Parsing | `app/ingestion/parsers.py` | PDF, DOCX, MD (front matter), TXT, HTML, JSON, CSV → text + metadata (title, author, pages, dates, …) |
| Chunking | `app/ingestion/chunking.py` | `fixed` (char window + overlap) vs `sentence` (heading-aware, packs whole sentences) |
| Dedup | `app/ingestion/dedup.py` | SHA-256 of normalised text (exact) + 64-bit SimHash over 3-shingles (near-dup, Hamming ≤ 3) |
| CRUD | `app/ingestion/pipeline.py` | `doc_id = corpus:source`; re-upload = new version (old chunks removed from both indexes); delete |
| Vector DB | `app/store/vector_store.py` | Qdrant: embedded on-disk locally, Qdrant Cloud in production (same code) |
| Embeddings | `app/embeddings.py` | default: local `BAAI/bge-small-en-v1.5` (384-d, ONNX on CPU, query instruction prefix); optional: `gemini-embedding-001` (768-d, task types). Query LRU cache |
| Keyword search | `app/retrieval/bm25.py` | **own** Okapi BM25 with an incremental inverted index (add/remove per chunk) |
| Fusion / re-rank | `app/retrieval/fusion.py` | **own** Reciprocal Rank Fusion + Maximal Marginal Relevance |
| Agent | `app/agent/` | tool calling, decomposition, citation verification, injection flagging |
| LLM | `app/llm/client.py` | any OpenAI-compatible endpoint; tenacity retries on 429/5xx/timeouts, token accounting, response cache |
| Eval | `eval/` | 60 questions, retrieval + end-to-end + chunking comparisons |

### RAG pipeline, step by step
1. **Decompose.** The fast model splits the question into 1–4 self-contained sub-queries. If this step fails, the pipeline continues with the original question only.
2. **Seed retrieval.** Hybrid search runs for the question and each sub-query, and all hits go into an `EvidenceLog`.
3. **Agent loop** (max 6 steps). The main model sees the evidence wrapped in `<chunk id=…>` tags and can call:
   - `search_documents(query, doc_ids?)`, for follow-up hops (e.g. once it has found a change id, look up its owner),
   - `get_document_context(chunk_id, window)`, to read neighbouring chunks,
   - `submit_answer(status, answer, citations[], contradictions[], confidence)`.

   If the model answers in prose, it is asked to use the tool. If it hits the step limit, `submit_answer` is forced with `tool_choice=required`.
4. **Verify.** Every cited chunk id is checked against the evidence log. Fabricated ids are marked invalid, and an
   "answered" result without a single valid citation is downgraded to `insufficient_evidence`.
5. **Abstain.** The prompt requires `insufficient_evidence` when the facts are absent. The grounding guard in step 4 enforces it.
6. **Prompt-injection defence.** Chunk text is labelled as untrusted data, and chunks that match injection patterns get the
   attribute `suspicious="possible prompt injection"`. The synthetic corpus contains a planted injection, and the eval tests for it.

## 2. Technical decisions

| Decision | Why |
|---|---|
| **Qdrant** | Free 1 GB cloud tier. The same client API works embedded (dev/tests use `:memory:`) and in the cloud. Payload filters support `doc_ids` scoping. |
| **Own BM25 instead of Qdrant sparse vectors** | The task requires our own ranking component. An in-process inverted index supports add/remove per chunk, so CRUD keeps both indexes in sync. ~65k chunks fit in memory easily. |
| **RRF for fusion** | Rank-based, so BM25 scores (unbounded) and cosine scores need no normalisation. It is robust and has one parameter (`k=60`). |
| **MMR re-rank** | Multi-document questions need *different* sources. MMR removes near-duplicate chunks (e.g. the PDF and the Confluence export of the same report). |
| **Sentence chunker by default** | Keeps facts like "CHG-2031 reduced the pool from 200 to 20" inside one chunk and prefixes the section heading. Compared against fixed-size chunks in the eval. |
| **Gemini via the OpenAI-compatible API** | Free tier, native tool calling, and the provider can be swapped by changing `LLM_BASE_URL`/`LLM_MODEL`. A fast model handles decomposition and judging; the main model runs the agent. |
| **Local embeddings (bge-small-en-v1.5)** | The Gemini free tier caps embedding requests per day, and every text counts as a request, so the ~65k chunks would take weeks. A local ONNX model has no quota or cost and embeds the whole corpus in about an hour on a 6-core laptop CPU (length-sorted batches are ~2.5× faster). A query embeds in ~15 ms on the same CPU. BGE is asymmetric: queries get an instruction prefix and documents don't. It beat the faster all-MiniLM-L6-v2 on retrieval benchmarks, and the Gemini embedder remains selectable with `EMBED_PROVIDER=gemini`. |
| **Citation verification in code, not in the prompt** | LLMs invent ids. Checking them against what was actually retrieved is cheap and makes the citation-accuracy metric meaningful. |
| **No LangChain/LlamaIndex** | The whole pipeline is ~2k lines and every step can be explained and debugged directly. |

## 3. Dataset (≥ 50,000 records)

`python -m scripts.build_corpus` builds a reproducible corpus. It uses fixed splits and fixed order, with no random sampling.

| Source | Records | Purpose |
|---|---|---|
| HotpotQA (distractor, validation): all context paragraphs of the first 6,000 questions | **54,962** paragraphs | scale + multi-hop Wikipedia questions with gold supporting titles |
| danluu/post-mortems | 244 incident summaries | real incident data |
| Synthetic "NovaPay" corpus (`scripts/make_synthetic.py`) | 12 files in 7 formats | investigation scenario with planted multi-hop chains, contradictions, an exact and a near duplicate, a prompt injection and deliberately missing facts |

The HotpotQA subset is the first *N* questions of the official validation split together with *all* their context
paragraphs (gold and distractors). This keeps every eval question answerable from the index, while the distractors
make retrieval non-trivial.

## 4. Running locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # set LLM_API_KEY (Google AI Studio)
python -m scripts.build_corpus                           # download + build corpus files
python -m scripts.ingest_corpus --limit 2000             # smoke index (fast)
python -m scripts.ingest_corpus                          # full index (~65k chunks)
uvicorn app.api.main:app --reload                        # http://localhost:8000
pytest -q                                                # 37 tests, no API key needed
```

Evaluation: `python -m eval.make_questions`, then `python -m eval.run_eval retrieval|e2e|chunking` (see EVAL_REPORT.md).

### API

| Method | Path | |
|---|---|---|
| GET | `/api/stats` | counts, models, formats |
| GET | `/api/documents?corpus=&offset=` | paginated list |
| GET | `/api/documents/{doc_id}` | document + chunks |
| POST | `/api/documents` (multipart `files`) | add; returns created / updated / duplicate / near-duplicate info |
| PUT | `/api/documents/{doc_id}` | replace content (new version) |
| DELETE | `/api/documents/{doc_id}` | delete (dataset docs need `X-Admin-Token`) |
| POST | `/api/search` `{query, mode: dense\|bm25\|hybrid, top_k, use_mmr}` | retrieval only, with per-retriever ranks |
| POST | `/api/ask` `{question, agentic, mode}` | full investigation |
| POST | `/api/ask/stream` | same, as Server-Sent Events: `plan`, `search`, `tool_call`, `answer`, `error` |

## 5. Deployment

```
GitHub repo ──push──▶ Hugging Face Space (Docker, free CPU) ──HTTPS──▶ Qdrant Cloud (free 1 GB cluster)
                          │  secrets: LLM_API_KEY, QDRANT_URL,               ▲
                          │           QDRANT_API_KEY, ADMIN_TOKEN            │ one-time: run scripts.ingest_corpus
                          └──HTTPS──▶ Gemini API                              │ locally with QDRANT_URL set
```

1. Create a free Qdrant Cloud cluster and copy its URL and API key.
2. Put `QDRANT_URL`/`QDRANT_API_KEY` in your local `.env` and either run `python -m scripts.ingest_corpus` (embeds and writes straight to the cloud), or, if you already built the local index, run `python -m scripts.copy_to_cloud`, which copies the points with their vectors in a few minutes instead of re-embedding.
3. Create a Hugging Face Space (SDK: Docker) and add the four values above as **Secrets**. Push this repo to the Space; the README front matter configures it.

**Persistence.** Qdrant Cloud is the source of truth, so the container is stateless. On restart the BM25 index is
rebuilt from Qdrant (a scroll over all chunks, roughly a minute for 65k), and uploaded documents survive restarts.

**Secrets.** Keys come only from environment variables or Space secrets. `.env` is git-ignored, the embeddings key
is sent in a header (never in URLs or logs), and dataset documents can only be changed or deleted with `ADMIN_TOKEN`.

**Free-tier limits and costs.**

| Resource | Limit | Mitigation |
|---|---|---|
| HF Space (free CPU) | sleeps after ~48 h idle; cold start ≈ 1–2 min (BM25 rebuild) | documented; first request after sleep is slow |
| Qdrant Cloud free | 1 GB RAM/disk | 65k × 384-d float32 ≈ 100 MB vectors + ~60 MB payload; vectors `on_disk` |
| Gemini free tier | per-minute and per-day request caps | per-IP rate limit (10 questions/min), retries with backoff, LLM response cache, query-embedding cache |
| Embedding the corpus | ≈ 29 M characters, ~65k chunks | local model: $0, ≈ 1 h one-time on a laptop CPU; the model (~130 MB) is baked into the Docker image |
| Cost per question | agentic: ~3–6 LLM calls, ~6–15k tokens; single-shot: 1 call | `agentic=false` toggle in UI/API; measured per config in EVAL_REPORT.md |

## 6. Tests

`pytest -q` runs 37 tests offline. They use an in-memory Qdrant, a hashing embedder and a scripted LLM, so the
real chunkers, dedup, BM25, RRF, MMR, agent loop, citation checks, API routes and eval metrics run without network
access. They cover parsers for all 7 formats, CRUD and versioning, exact and near duplicates, fabricated-citation
detection, the step limit, LLM failure handling, tool argument errors, injection flagging, admin protection and SSE.

## 7. Project layout

```
app/        api/ agent/ ingestion/ retrieval/ store/ llm/ embeddings.py config.py container.py static/
scripts/    build_corpus.py make_synthetic.py ingest_corpus.py
eval/       make_questions.py questions.jsonl run_eval.py results/
tests/      conftest.py test_algorithms.py test_pipeline.py test_eval.py
```
