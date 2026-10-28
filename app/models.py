"""Domain models shared across layers."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ParsedDocument(BaseModel):
    """Output of a parser: plain text plus extracted metadata."""
    title: str
    text: str
    source: str                      # file name / dataset id
    doc_type: str                    # pdf, docx, md, ...
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    position: int
    title: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentRecord(BaseModel):
    doc_id: str
    title: str
    source: str
    doc_type: str
    content_hash: str
    simhash: int
    num_chunks: int
    version: int = 1
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    debug: dict[str, Any] = Field(default_factory=dict)  # per-retriever ranks


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    quote: str = ""
    valid: bool = True               # chunk was actually retrieved during investigation


class Contradiction(BaseModel):
    description: str
    chunk_ids: list[str] = Field(default_factory=list)


class InvestigationResult(BaseModel):
    question: str
    answer: str
    status: str                      # "answered" | "insufficient_evidence" | "error"
    confidence: float = 0.0
    citations: list[Citation] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    sub_questions: list[str] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)  # tool calls, for transparency
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: int = 0
