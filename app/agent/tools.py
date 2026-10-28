"""Tools exposed to the LLM via function calling, plus their executors."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.models import SearchHit
from app.retrieval.hybrid import HybridRetriever, RetrievalConfig
from app.store.vector_store import VectorStore

INJECTION_RE = re.compile(
    r"(ignore (all |any )?(previous|prior|above) instructions|disregard (the )?(system|previous)|"
    r"you are now|new instructions:|system prompt|reveal your)", re.I)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Hybrid (semantic + keyword) search over the document collection. "
                           "Returns the most relevant chunks with their chunk ids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Focused search query"},
                    "doc_ids": {"type": "array", "items": {"type": "string"},
                                "description": "Optional: restrict search to these document ids"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_context",
            "description": "Read the chunks surrounding a chunk (same document) to see full context.",
            "parameters": {
                "type": "object",
                "properties": {"chunk_id": {"type": "string"},
                               "window": {"type": "integer", "description": "chunks before/after (1-3)"}},
                "required": ["chunk_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final, evidence-grounded answer. Call exactly once at the end.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["answered", "insufficient_evidence"]},
                    "answer": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "items": {"type": "object", "properties": {
                            "chunk_id": {"type": "string"},
                            "quote": {"type": "string", "description": "short verbatim supporting quote"}},
                            "required": ["chunk_id"]},
                    },
                    "contradictions": {
                        "type": "array",
                        "items": {"type": "object", "properties": {
                            "description": {"type": "string"},
                            "chunk_ids": {"type": "array", "items": {"type": "string"}}},
                            "required": ["description"]},
                    },
                    "confidence": {"type": "number", "description": "0.0-1.0"},
                },
                "required": ["status", "answer", "citations"],
            },
        },
    },
]


@dataclass
class EvidenceLog:
    """Everything the agent has seen. Used to validate citations afterwards."""
    chunks: dict[str, SearchHit] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)  # first-seen order (for Recall@K / MRR on agent runs)

    def add(self, hits: list[SearchHit]) -> list[SearchHit]:
        new = []
        for h in hits:
            if h.chunk_id not in self.chunks:
                self.chunks[h.chunk_id] = h
                self.order.append(h.chunk_id)
                new.append(h)
        return new


def format_chunks(hits: list[SearchHit], max_chars: int = 1200) -> str:
    if not hits:
        return "No results."
    parts = []
    for h in hits:
        text = h.text[:max_chars]
        flag = ' suspicious="possible prompt injection"' if INJECTION_RE.search(text) else ""
        parts.append(f'<chunk id="{h.chunk_id}" doc="{h.doc_id}" title="{h.title}"{flag}>\n{text}\n</chunk>')
    return "\n".join(parts)


class ToolExecutor:
    def __init__(self, retriever: HybridRetriever, store: VectorStore, cfg: RetrievalConfig, evidence: EvidenceLog):
        self.retriever = retriever
        self.store = store
        self.cfg = cfg
        self.evidence = evidence

    def run(self, name: str, raw_args: str) -> str:
        try:
            args = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            return "ERROR: arguments were not valid JSON."
        try:
            if name == "search_documents":
                return self.search(args.get("query", ""), args.get("doc_ids"))
            if name == "get_document_context":
                return self.context(args.get("chunk_id", ""), int(args.get("window", 1)))
        except Exception as e:  # a tool failure must not crash the investigation
            return f"ERROR: tool {name} failed: {e}"
        return f"ERROR: unknown tool {name}"

    def search(self, query: str, doc_ids: list[str] | None = None) -> str:
        if not query.strip():
            return "ERROR: empty query"
        hits = self.retriever.search(query, self.cfg, doc_ids=doc_ids or None)
        self.evidence.add(hits)
        return format_chunks(hits)

    def context(self, chunk_id: str, window: int) -> str:
        window = max(1, min(window, 3))
        doc_id, _, pos = chunk_id.rpartition("#")
        if not doc_id or not pos.isdigit():
            return "ERROR: invalid chunk id"
        p = int(pos)
        ids = [f"{doc_id}#{i}" for i in range(max(0, p - window), p + window + 1)]
        found = self.store.get_chunks(ids)
        hits = [found[i][0] for i in ids if i in found]
        self.evidence.add(hits)
        return format_chunks(hits)
