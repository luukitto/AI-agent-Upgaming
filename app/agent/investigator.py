"""Agentic RAG investigation loop.

Pipeline per question:
  1. Decompose   - cheap model splits the question into sub-queries (multi-hop planning).
  2. Seed        - hybrid retrieval for the question + every sub-query (multi-query retrieval).
  3. Agent loop  - main model iterates with tool calling (search_documents / get_document_context)
                   until it calls submit_answer or hits the step limit.
  4. Verify      - citations are checked against the evidence log; fabricated ids are dropped; an
                   "answered" result without any valid citation is downgraded to insufficient_evidence.

`agentic=False` gives a single-shot RAG baseline (steps 2 + one forced submit_answer call), used in eval.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Callable

from app.agent.prompts import AGENT_SYSTEM, DECOMPOSE_PROMPT, FINALIZE_NUDGE
from app.agent.tools import TOOL_SCHEMAS, EvidenceLog, ToolExecutor, format_chunks
from app.config import get_settings
from app.llm.client import LLMClient, LLMError, parse_json_loose
from app.models import Citation, Contradiction, InvestigationResult
from app.retrieval.hybrid import HybridRetriever, RetrievalConfig, default_config
from app.store.vector_store import VectorStore

log = logging.getLogger(__name__)
Event = Callable[[str, dict], None]
SUBMIT_ONLY = [t for t in TOOL_SCHEMAS if t["function"]["name"] == "submit_answer"]


@dataclass
class AgentConfig:
    agentic: bool = True
    decompose: bool = True
    max_steps: int = 6
    model: str | None = None
    fast_model: str | None = None
    seed_k: int = 6


class Investigator:
    def __init__(self, llm: LLMClient, retriever: HybridRetriever, store: VectorStore):
        self.llm = llm
        self.retriever = retriever
        self.store = store

    def investigate(self, question: str, agent_cfg: AgentConfig | None = None,
                    retr_cfg: RetrievalConfig | None = None, on_event: Event | None = None) -> InvestigationResult:
        s = get_settings()
        acfg = agent_cfg or AgentConfig(max_steps=s.agent_max_steps)
        rcfg = retr_cfg or default_config()
        model = acfg.model or s.llm_model
        fast_model = acfg.fast_model or s.llm_fast_model
        emit = on_event or (lambda *_: None)
        t0 = time.time()
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "llm_calls": 0}
        trace: list[dict] = []
        evidence = EvidenceLog()
        tools = ToolExecutor(self.retriever, self.store, rcfg, evidence)

        def call(**kw) -> dict:
            r = self.llm.chat(**kw)
            usage["llm_calls"] += 1
            for k in ("prompt_tokens", "completion_tokens"):
                usage[k] += r.get("usage", {}).get(k, 0)
            return r

        subs: list[str] = []

        def result(**kw) -> InvestigationResult:
            return InvestigationResult(question=question, trace=trace, usage=usage, sub_questions=subs,
                                       retrieved_chunk_ids=list(evidence.order),
                                       latency_ms=int((time.time() - t0) * 1000), **kw)

        try:
            # 1. decomposition
            if acfg.decompose:
                subs.extend(self._decompose(question, fast_model, call))
                trace.append({"step": "decompose", "sub_questions": subs})
                emit("plan", {"sub_questions": subs})

            # 2. seed retrieval (question + sub-questions), smaller k per query to fit context
            seed_cfg = RetrievalConfig(**{**rcfg.__dict__, "top_k": acfg.seed_k})
            for q in [question, *subs]:
                hits = self.retriever.search(q, seed_cfg)
                new = evidence.add(hits)
                trace.append({"step": "search", "query": q, "results": [h.chunk_id for h in hits]})
                emit("search", {"query": q, "results": [{"chunk_id": h.chunk_id, "title": h.title} for h in new]})

            seed_text = format_chunks([evidence.chunks[c] for c in evidence.order])
            user_msg = (f"Question: {question}\n"
                        + (f"Planned sub-questions: {json.dumps(subs)}\n" if subs else "")
                        + f"\nInitial evidence:\n{seed_text}")
            messages: list[dict] = [{"role": "system", "content": AGENT_SYSTEM},
                                    {"role": "user", "content": user_msg}]

            # 3. agent loop
            final_args = None
            steps = acfg.max_steps if acfg.agentic else 0
            for step in range(steps):
                resp = call(model=model, messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto")
                if not resp["tool_calls"]:
                    # model answered in prose; ask it to use the structured tool
                    messages.append({"role": "assistant", "content": resp["content"] or "(empty)"})
                    messages.append({"role": "user", "content": "Call submit_answer with your answer and citations."})
                    continue
                messages.append(_assistant_msg(resp))
                for tc in resp["tool_calls"]:
                    if tc["name"] == "submit_answer":
                        final_args = tc["arguments"]
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "received"})
                        continue
                    emit("tool_call", {"name": tc["name"], "arguments": tc["arguments"]})
                    before = len(evidence.order)
                    out = tools.run(tc["name"], tc["arguments"])
                    trace.append({"step": "tool", "name": tc["name"], "arguments": tc["arguments"],
                                  "new_chunks": evidence.order[before:]})
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
                if final_args is not None:
                    break

            # forced finalisation (also the whole generation step for the non-agentic baseline)
            if final_args is None:
                messages.append({"role": "user", "content": FINALIZE_NUDGE})
                resp = call(model=model, messages=messages, tools=SUBMIT_ONLY, tool_choice="required")
                if resp["tool_calls"]:
                    final_args = resp["tool_calls"][0]["arguments"]
                elif resp["content"]:
                    final_args = resp["content"]  # last resort: JSON in plain text
                else:
                    raise LLMError("model did not produce a final answer")

            # 4. verification
            res = self._finalize(final_args, evidence, result)
            # res.trace is a copy (pydantic validation), so append to it, not to `trace`
            res.trace.append({"step": "verify", "valid_citations": sum(c.valid for c in res.citations),
                          "invalid_citations": sum(not c.valid for c in res.citations)})
            emit("answer", res.model_dump())
            return res
        except LLMError as e:
            log.warning("investigation failed: %s", e)
            res = result(answer=f"The language model is currently unavailable: {e}", status="error")
            emit("error", {"message": str(e)})
            return res

    def _decompose(self, question: str, model: str, call) -> list[str]:
        try:
            r = call(model=model, messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(question=question)}],
                     temperature=0.0, response_format={"type": "json_object"})
            subs = parse_json_loose(r["content"]).get("sub_questions", [])
            return [s for s in subs if isinstance(s, str) and s.strip() and s.strip() != question][:4]
        except (LLMError, ValueError, AttributeError) as e:
            log.info("decomposition skipped: %s", e)  # degrade gracefully: plain retrieval still works
            return []

    def _finalize(self, raw: str | dict, evidence: EvidenceLog, result) -> InvestigationResult:
        try:
            args = raw if isinstance(raw, dict) else parse_json_loose(raw)
        except ValueError:
            args = {"status": "answered", "answer": str(raw), "citations": []}

        citations, seen = [], set()
        for c in args.get("citations") or []:
            cid = c.get("chunk_id", "") if isinstance(c, dict) else str(c)
            if not cid or cid in seen:
                continue
            seen.add(cid)
            hit = evidence.chunks.get(cid)
            quote = (c.get("quote") or "") if isinstance(c, dict) else ""
            citations.append(Citation(chunk_id=cid, doc_id=hit.doc_id if hit else cid.rpartition("#")[0],
                                      title=hit.title if hit else "", quote=quote[:300], valid=hit is not None))
        contradictions = [Contradiction(description=c.get("description", ""),
                                        chunk_ids=[i for i in c.get("chunk_ids", []) if i in evidence.chunks])
                          for c in (args.get("contradictions") or []) if isinstance(c, dict) and c.get("description")]

        status = args.get("status", "answered")
        if status not in ("answered", "insufficient_evidence"):
            status = "answered"
        answer = str(args.get("answer", "")).strip()
        valid = [c for c in citations if c.valid]
        if status == "answered" and not valid:
            # grounding guard: no verifiable evidence -> do not present as an answer
            status = "insufficient_evidence"
            answer = ("I could not verify this answer against the indexed documents, so I am not giving one. "
                      f"(Unverified draft: {answer[:300]})")
        try:
            confidence = float(args.get("confidence", 0.5 if status == "answered" else 0.0))
        except (TypeError, ValueError):
            confidence = 0.5
        return result(answer=answer, status=status, confidence=max(0.0, min(1.0, confidence)),
                      citations=citations, contradictions=contradictions)


def _assistant_msg(resp: dict) -> dict:
    return {"role": "assistant", "content": resp["content"] or None,
            "tool_calls": [{"id": tc["id"], "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                           for tc in resp["tool_calls"]]}
