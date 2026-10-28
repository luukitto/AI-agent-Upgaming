"""Two chunking strategies, compared in the evaluation:

1. FixedSizeChunker   - sliding character window with overlap. Simple, predictable sizes,
                        but cuts sentences/facts in half.
2. SentenceChunker    - structure-aware: splits on headings/paragraphs, then packs whole
                        sentences up to the size budget, with a one-sentence overlap.
                        Keeps facts intact and prefixes each chunk with its section heading.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")


class Chunker(ABC):
    name: str

    def __init__(self, chunk_size: int = 800, overlap: int = 120):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    @abstractmethod
    def split(self, text: str) -> list[str]: ...


class FixedSizeChunker(Chunker):
    name = "fixed"

    def split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        chunks, start, step = [], 0, self.chunk_size - self.overlap
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            # avoid cutting a word: back off to the last whitespace if possible
            if end < len(text):
                ws = text.rfind(" ", start + step // 2, end)
                if ws != -1:
                    end = ws
            chunks.append(text[start:end].strip())
            if end >= len(text):
                break
            start = max(end - self.overlap, start + 1)
            # don't start the overlap mid-word either: move forward to the next word boundary
            if not text[start - 1].isspace():
                ws = text.find(" ", start, end)
                if ws != -1:
                    start = ws + 1
        return [c for c in chunks if c]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]


class SentenceChunker(Chunker):
    name = "sentence"

    def _sections(self, text: str) -> list[tuple[str, str]]:
        """Split text into (heading, body) sections using markdown-style headings."""
        sections: list[tuple[str, list[str]]] = [("", [])]
        for line in text.splitlines():
            m = HEADING_RE.match(line.strip())
            if m:
                sections.append((m.group(1).strip(), []))
            else:
                sections[-1][1].append(line)
        return [(h, "\n".join(body).strip()) for h, body in sections if "\n".join(body).strip()]

    def split(self, text: str) -> list[str]:
        chunks: list[str] = []
        for heading, body in self._sections(text):
            prefix = f"[{heading}] " if heading else ""
            budget = self.chunk_size - len(prefix)
            units: list[str] = []
            for para in re.split(r"\n\s*\n", body):
                for line in para.splitlines():  # list items / table rows are their own units
                    units.extend(split_sentences(line))
            current: list[str] = []
            length = 0
            for sent in units:
                # a single over-long sentence falls back to fixed splitting
                if len(sent) > budget:
                    if current:
                        chunks.append(prefix + " ".join(current))
                        current, length = [], 0
                    chunks.extend(prefix + p for p in FixedSizeChunker(budget, self.overlap).split(sent))
                    continue
                if length + len(sent) + 1 > budget and current:
                    chunks.append(prefix + " ".join(current))
                    # one-sentence overlap keeps cross-sentence context
                    tail = current[-1] if len(current[-1]) <= self.overlap * 2 else ""
                    current = [tail] if tail else []
                    length = len(tail)
                current.append(sent)
                length += len(sent) + 1
            if current:
                chunks.append(prefix + " ".join(current))
        return chunks


def get_chunker(strategy: str, chunk_size: int, overlap: int) -> Chunker:
    if strategy == "fixed":
        return FixedSizeChunker(chunk_size, overlap)
    if strategy == "sentence":
        return SentenceChunker(chunk_size, overlap)
    raise ValueError(f"Unknown chunk strategy: {strategy}")
