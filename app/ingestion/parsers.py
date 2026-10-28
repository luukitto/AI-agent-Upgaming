"""File-format parsers. Each parser turns raw bytes into a ParsedDocument (text + metadata).

Supported: .pdf, .docx, .md, .txt, .html/.htm, .json, .jsonl, .csv
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Callable

from app.models import ParsedDocument

DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")


class UnsupportedFormatError(ValueError):
    pass


def _extract_dates(text: str) -> list[str]:
    return sorted(set(DATE_RE.findall(text)))[:20]


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return line.lstrip("#").strip()
        if line:
            return line[:120]
    return None


def parse_pdf(data: bytes, name: str) -> ParsedDocument:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n\n".join(pages)
    info = reader.metadata or {}
    title = (info.get("/Title") if info else None) or _first_heading(text) or name
    return ParsedDocument(
        title=str(title), text=text, source=name, doc_type="pdf",
        metadata={"pages": len(pages), "author": str(info.get("/Author", "")) if info else ""},
    )


def parse_docx(data: bytes, name: str) -> ParsedDocument:
    import docx

    d = docx.Document(io.BytesIO(data))
    parts = []
    for p in d.paragraphs:
        if not p.text.strip():
            continue
        # keep headings recognisable for the sentence chunker
        prefix = "# " if p.style is not None and p.style.name.lower().startswith("heading") else ""
        parts.append(prefix + p.text)
    for table in d.tables:
        for row in table.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    text = "\n".join(parts)
    props = d.core_properties
    return ParsedDocument(
        title=props.title or _first_heading(text) or name, text=text, source=name, doc_type="docx",
        metadata={"author": props.author or ""},
    )


def parse_markdown(data: bytes, name: str) -> ParsedDocument:
    text = data.decode("utf-8", errors="replace")
    meta: dict = {}
    # optional YAML-ish front matter: key: value lines between --- markers
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            for line in text[3:end].splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            text = text[end + 4:].lstrip()
    title = meta.pop("title", None) or _first_heading(text) or name
    return ParsedDocument(title=title, text=text, source=name, doc_type="md", metadata=meta)


def parse_text(data: bytes, name: str) -> ParsedDocument:
    text = data.decode("utf-8", errors="replace")
    return ParsedDocument(title=_first_heading(text) or name, text=text, source=name, doc_type="txt")


def parse_html(data: bytes, name: str) -> ParsedDocument:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else None
    text = soup.get_text("\n", strip=True)
    return ParsedDocument(title=title or _first_heading(text) or name, text=text, source=name, doc_type="html")


def parse_json(data: bytes, name: str) -> ParsedDocument:
    obj = json.loads(data.decode("utf-8"))
    records = obj if isinstance(obj, list) else [obj]
    lines = []
    for r in records:
        if isinstance(r, dict):
            lines.append("; ".join(f"{k}: {v}" for k, v in r.items()))
        else:
            lines.append(str(r))
    text = "\n".join(lines)
    title = obj.get("title", name) if isinstance(obj, dict) else name
    return ParsedDocument(title=str(title), text=text, source=name, doc_type="json",
                          metadata={"records": len(records)})


def parse_csv(data: bytes, name: str) -> ParsedDocument:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8", errors="replace")))
    rows = ["; ".join(f"{k}: {v}" for k, v in row.items()) for row in reader]
    return ParsedDocument(title=name, text="\n".join(rows), source=name, doc_type="csv",
                          metadata={"rows": len(rows), "columns": reader.fieldnames or []})


PARSERS: dict[str, Callable[[bytes, str], ParsedDocument]] = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".txt": parse_text,
    ".html": parse_html,
    ".htm": parse_html,
    ".json": parse_json,
    ".csv": parse_csv,
}


def parse_file(data: bytes, name: str) -> ParsedDocument:
    ext = Path(name).suffix.lower()
    parser = PARSERS.get(ext)
    if parser is None:
        raise UnsupportedFormatError(f"Unsupported file type '{ext}'. Supported: {sorted(PARSERS)}")
    doc = parser(data, name)
    doc.text = normalize_text(doc.text)
    doc.metadata.setdefault("dates", _extract_dates(doc.text))
    doc.metadata["char_count"] = len(doc.text)
    return doc


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
