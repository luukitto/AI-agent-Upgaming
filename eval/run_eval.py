"""Automated evaluation.

  python -m eval.run_eval retrieval            # no LLM: bm25 vs dense vs hybrid vs hybrid+MMR
  python -m eval.run_eval e2e [--limit N]      # LLM: single-shot RAG vs agentic RAG (+ LLM judge)
  python -m eval.run_eval chunking             # fixed vs sentence chunking on a small in-memory sub-corpus

Metrics (doc-level, a "gold group" is satisfied by any one of its doc ids):
  Recall@K       fraction of gold groups among the first K distinct retrieved documents
  MRR            1 / rank of the first retrieved gold document
  correctness    LLM judge score (1 / 0.5 / 0) vs the reference answer
  keyword_hit    fraction of required answer regexes present (deterministic, judge-free)
  citation_prec  share of valid citations that point to a gold document
  citation_valid share of citations that were actually retrieved (non-fabricated)
  abstain_acc    on unanswerable questions: share answered with insufficient_evidence
  false_abstain  on answerable questions: share wrongly answered with insufficient_evidence

Results -> eval/results/<mode>.json (per question) and <mode>.md (summary tables).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

from app.config import ROOT_DIR

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
KS = (1, 3, 5, 10)


# ---------------- metrics ----------------
def distinct_docs(chunk_ids: list[str]) -> list[str]:
    out: list[str] = []
    for cid in chunk_ids:
        d = cid.rpartition("#")[0]
        if d not in out:
            out.append(d)
    return out


def recall_at_k(docs: list[str], gold: list[list[str]], k: int) -> float:
    top = set(docs[:k])
    return sum(any(d in top for d in g) for g in gold) / len(gold)


def mrr(docs: list[str], gold: list[list[str]]) -> float:
    golds = {d for g in gold for d in g}
    return next((1 / i for i, d in enumerate(docs, 1) if d in golds), 0.0)


def keyword_hit(answer: str, patterns: list[str]) -> float:
    if not patterns:
        return 1.0
    norm = lambda s: re.sub(r"\s+", " ", s.lower())  # noqa: E731
    return sum(bool(re.search(p.lower(), norm(answer))) for p in patterns) / len(patterns)


def mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return round(statistics.mean(xs), 3) if xs else float("nan")


def load_questions(limit: int | None = None, categories: set[str] | None = None) -> list[dict]:
    rows = [json.loads(line) for line in open(HERE / "questions.jsonl", encoding="utf-8")]
    if categories:
        rows = [r for r in rows if r["category"] in categories]
    return rows[:limit] if limit else rows


def table(rows: list[dict], cols: list[str], title: str) -> str:
    out = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def by_category(per_q: list[dict], metric_cols: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for r in per_q:
        groups[(r["config"], r["category"])].append(r)
    return [{"config": cfg, "category": cat, "n": len(rs), **{m: mean(r.get(m) for r in rs) for m in metric_cols}}
            for (cfg, cat), rs in sorted(groups.items())]


def save(mode: str, per_q: list[dict], md: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{mode}.json").write_text(json.dumps(per_q, indent=1, ensure_ascii=False), encoding="utf-8")
    (RESULTS / f"{mode}.md").write_text(md, encoding="utf-8")
    print(md)


# ---------------- retrieval-only ----------------
RETRIEVAL_CONFIGS = {
    "bm25": dict(mode="bm25", use_mmr=False),
    "dense": dict(mode="dense", use_mmr=False),
    "hybrid": dict(mode="hybrid", use_mmr=False),
    "hybrid+mmr": dict(mode="hybrid", use_mmr=True),
}


def retrieval_metrics(retriever, questions: list[dict], configs: dict[str, dict], top_k: int = 10) -> list[dict]:
    from app.retrieval.hybrid import RetrievalConfig, default_config
    per_q = []
    for name, overrides in configs.items():
        cfg = RetrievalConfig(**{**default_config().__dict__, "top_k": top_k, **overrides})
        for q in questions:
            if not q["gold"]:
                continue  # unanswerable: no gold documents to retrieve
            t = time.time()
            docs = distinct_docs([h.chunk_id for h in retriever.search(q["question"], cfg)])
            per_q.append({"config": name, "id": q["id"], "category": q["category"],
                          **{f"R@{k}": recall_at_k(docs, q["gold"], k) for k in KS},
                          "MRR": mrr(docs, q["gold"]), "latency_ms": int((time.time() - t) * 1000),
                          "docs": docs[:10]})
    return per_q


def retrieval_summary(per_q: list[dict], title: str) -> str:
    cols = [f"R@{k}" for k in KS] + ["MRR", "latency_ms"]
    overall = by_category([{**r, "category": "ALL"} for r in per_q], cols)
    return (table(overall, ["config", "n", *cols], f"{title}: overall")
            + "\n" + table(by_category(per_q, cols), ["config", "category", "n", *cols], f"{title}: by category"))


def run_retrieval(args) -> None:
    from app.container import get_container
    c = get_container()
    per_q = retrieval_metrics(c.retriever, load_questions(args.limit), RETRIEVAL_CONFIGS)
    md = f"## Retrieval evaluation ({c.store.count_documents()} docs, {c.store.count_chunks()} chunks)\n\n"
    save("retrieval", per_q, md + retrieval_summary(per_q, "Retrieval"))


# ---------------- end-to-end ----------------
def judge(llm, model: str, q: dict, res) -> dict:
    from app.agent.prompts import JUDGE_PROMPT
    from app.llm.client import LLMError, parse_json_loose
    try:
        r = llm.chat(model=model, temperature=0.0, response_format={"type": "json_object"},
                     messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                         question=q["question"], reference=q["reference"], answer=res.answer, status=res.status)}])
        out = parse_json_loose(r["content"])
        return {"score": float(out.get("score", 0)), "reason": str(out.get("reason", ""))[:300]}
    except (LLMError, ValueError, TypeError, AttributeError) as e:
        return {"score": None, "reason": f"judge failed: {e}"}


def e2e_row(q: dict, res, verdict: dict, config: str) -> dict:
    unanswerable = q["reference"].startswith("INSUFFICIENT")
    abstained = res.status == "insufficient_evidence"
    golds = {d for g in q["gold"] for d in g}
    valid = [c for c in res.citations if c.valid]
    docs = distinct_docs(res.retrieved_chunk_ids)
    return {
        "config": config, "id": q["id"], "category": q["category"], "status": res.status,
        "correctness": verdict["score"], "judge_reason": verdict["reason"],
        "keyword_hit": None if unanswerable else keyword_hit(res.answer, q["keywords"]),
        "abstain_acc": (1.0 if abstained else 0.0) if unanswerable else None,
        "false_abstain": None if unanswerable else (1.0 if abstained else 0.0),
        "citation_valid": (len(valid) / len(res.citations)) if res.citations else None,
        "citation_prec": (sum(c.doc_id in golds for c in valid) / len(valid)) if (valid and golds) else None,
        "R@5": recall_at_k(docs, q["gold"], 5) if q["gold"] else None,
        "R@all": recall_at_k(docs, q["gold"], len(docs)) if q["gold"] else None,
        "MRR": mrr(docs, q["gold"]) if q["gold"] else None,
        "contradictions": len(res.contradictions), "latency_ms": res.latency_ms,
        "llm_calls": res.usage.get("llm_calls", 0),
        "tokens": res.usage.get("prompt_tokens", 0) + res.usage.get("completion_tokens", 0),
        "answer": res.answer, "citations": [c.chunk_id for c in res.citations],
    }


def run_e2e(args) -> None:
    from app.agent.investigator import AgentConfig
    from app.container import get_container
    from app.llm.client import get_llm
    c = get_container()
    llm, settings = get_llm(), c.settings
    configs = {
        "single-shot": AgentConfig(agentic=False, decompose=False),
        "agentic": AgentConfig(agentic=True, decompose=True, max_steps=settings.agent_max_steps),
    }
    configs = {k: v for k, v in configs.items() if not args.configs or k in args.configs.split(",")}
    questions = load_questions(args.limit, set(args.categories.split(",")) if args.categories else None)
    # finished questions are appended to a progress file, so a run stopped by quota resumes where it left off
    # (with the latencies measured the first time, not cache-hit latencies)
    RESULTS.mkdir(exist_ok=True)
    progress = RESULTS / f"e2e{'_' + args.tag if args.tag else ''}.progress.jsonl"
    done = {(r["config"], r["id"]): r for r in map(json.loads, progress.open(encoding="utf-8"))} \
        if progress.exists() else {}
    per_q = []
    for name, acfg in configs.items():
        for i, q in enumerate(questions, 1):
            if (name, q["id"]) in done:
                per_q.append(done[(name, q["id"])])
                continue
            res = c.investigator.investigate(q["question"], acfg)
            if res.status == "error":
                # an outage/quota error is not a wrong answer: stop, keep partial results, re-run later
                print(f"LLM error on {q['id']}: {res.answer}\nStopping; re-run the same command to resume.")
                return save_e2e(per_q, llm, settings, args)
            row = e2e_row(q, res, judge(llm, settings.llm_fast_model, q, res), name)
            per_q.append(row)
            with progress.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{name} {i}/{len(questions)}] {q['id']} {row['status']} correct={row['correctness']} "
                  f"{row['latency_ms']}ms", flush=True)
            time.sleep(args.sleep)  # free-tier RPM friendliness

    save_e2e(per_q, llm, settings, args)


def save_e2e(per_q, llm, settings, args) -> None:
    cols = ["correctness", "keyword_hit", "citation_prec", "citation_valid", "abstain_acc", "false_abstain",
            "R@5", "MRR", "latency_ms", "llm_calls", "tokens"]
    overall = by_category([{**r, "category": "ALL"} for r in per_q], cols)
    md = (f"## End-to-end evaluation (LLM {settings.llm_model}, judge {settings.llm_fast_model})\n\n"
          + table(overall, ["config", "n", *cols], "Overall") + "\n"
          + table(by_category(per_q, cols), ["config", "category", "n", *cols], "By category")
          + f"\nTotal LLM usage incl. judge: {llm.usage.as_dict()}\n")
    save("e2e" + (f"_{args.tag}" if args.tag else ""), per_q, md)


# ---------------- chunking comparison ----------------
def run_chunking(args) -> None:
    """Index the same sub-corpus twice (in-memory Qdrant), once per chunker, and compare retrieval.

    Sub-corpus = all NovaPay files + post-mortems + the HotpotQA context paragraphs (gold + distractors) of the
    eval questions. Small enough to embed on a free tier, and every question's evidence is present.
    """
    from qdrant_client import QdrantClient

    from app.embeddings import get_embedder
    from app.ingestion.chunking import get_chunker
    from app.ingestion.pipeline import IngestionService, make_doc_id
    from app.models import ParsedDocument
    from app.retrieval.hybrid import HybridRetriever
    from app.store.vector_store import VectorStore
    from scripts.ingest_corpus import postmortem_docs

    questions = load_questions(args.limit)
    gold = {d for q in questions for g in q["gold"] for d in g if d.startswith("hotpot:")}
    # hotpot_paragraphs.jsonl is written in question order, so its head holds the context paragraphs
    # (gold + distractors) of the first questions, which are exactly the eval's hotpot questions
    hotpot = []
    with open(ROOT_DIR / "data/corpus/hotpot_paragraphs.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            p = json.loads(line)
            if i < 12 * len(gold) or make_doc_id("hotpot", p["title"]) in gold:
                hotpot.append(ParsedDocument(title=p["title"], text=p["text"], source=p["title"], doc_type="wiki",
                                             metadata={"dataset": "hotpot_qa"}))

    embedder = get_embedder()
    per_q = []
    for strategy in ("fixed", "sentence"):
        store = VectorStore(QdrantClient(":memory:"), "chunks", "documents", embedder.dim)
        retriever = HybridRetriever(store, embedder)
        ing = IngestionService(store, embedder, retriever, get_chunker(strategy, args.chunk_size, args.chunk_overlap))
        for p in sorted((ROOT_DIR / "data" / "synthetic").iterdir()):
            ing.ingest_file(p.read_bytes(), p.name, corpus="novapay")
        ing.bulk_ingest(postmortem_docs(), corpus="postmortems")
        ing.bulk_ingest(iter(hotpot), corpus="hotpot")
        print(f"{strategy}: {store.count_documents()} docs, {store.count_chunks()} chunks", flush=True)
        for r in retrieval_metrics(retriever, questions, {"hybrid": dict(mode="hybrid", use_mmr=False)}):
            per_q.append({**r, "config": f"{strategy}-{args.chunk_size}"})
    md = (f"## Chunking comparison (chunk_size={args.chunk_size}, overlap={args.chunk_overlap}, hybrid retrieval)\n\n"
          + retrieval_summary(per_q, "Chunking"))
    save("chunking", per_q, md)


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["retrieval", "e2e", "chunking"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--configs", help="e2e: comma list of single-shot,agentic")
    ap.add_argument("--categories", help="comma list of question categories")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--chunk-size", type=int, default=400)
    ap.add_argument("--chunk-overlap", type=int, default=80)
    args = ap.parse_args()
    {"retrieval": run_retrieval, "e2e": run_e2e, "chunking": run_chunking}[args.mode](args)


if __name__ == "__main__":
    main()
