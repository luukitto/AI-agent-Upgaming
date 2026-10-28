# AI Usage Report

> Draft. Complete the sections marked **TODO (author)** in your own words before submission.

## Tools used

| Tool | Used for |
|---|---|
| Claude Code (Claude Opus) | scaffolding the project structure; writing first drafts of modules, the test suite, the eval harness, the UI and the documentation; finding bugs while writing tests |
| **TODO (author)** | any other assistants (ChatGPT, Copilot, …) |

## What AI helped with

- Initial scaffold: parsers, chunkers, dedup, Qdrant store, BM25/RRF/MMR, LLM client, agent, API (commit `397cb3b`).
- Test suite with fakes (in-memory Qdrant, hashing embedder, scripted LLM). Writing it surfaced two real bugs, both since fixed:
  - `FixedSizeChunker` started the overlap in the middle of a word;
  - the agent's `verify` trace step was appended to a list that pydantic had already copied, so it never appeared in the result.
- Evaluation harness, question set, README, deployment files.

## What I decided / changed myself

**TODO (author).** For example: choice of datasets and the NovaPay scenario; switching from local fastembed to
Gemini embeddings after measuring ~15 docs/s; model choice after the `gemini-2.5-*` 404s; reviewing and
correcting generated code; the design of the planted contradictions; eval analysis.

## How I verified AI-generated code

**TODO (author).** For example: read every module, ran the test suite, ran the eval, manually tried the UI with
the example question, checked the citations against the source files.
