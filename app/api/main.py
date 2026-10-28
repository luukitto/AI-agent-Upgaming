"""FastAPI application: REST API + SSE streaming + static UI."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent.investigator import AgentConfig
from app.config import get_settings
from app.container import Container, get_container
from app.ingestion.parsers import PARSERS, UnsupportedFormatError
from app.llm.client import LLMError
from app.retrieval.hybrid import RetrievalConfig, default_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    t = time.time()
    get_container()  # warm up: load embedding model, BM25 index, duplicate index
    log.info("startup complete in %.1fs", time.time() - t)
    yield


app = FastAPI(title="AI Investigation Engine", version="1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------- simple per-IP rate limit for LLM endpoints (protects the free API quota) ----------------
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def rate_limit(request: Request, limit: int = 10, window_s: int = 60) -> None:
    ip = request.client.host if request.client else "?"
    now = time.time()
    with _hits_lock:
        q = _hits[ip]
        while q and now - q[0] > window_s:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(429, f"Rate limit: max {limit} questions per {window_s}s")
        q.append(now)


def require_admin_for(doc_id: str, token: str | None) -> None:
    """Uploaded ('user:') docs are editable by anyone; dataset docs need the admin token."""
    admin = get_settings().admin_token
    if doc_id.startswith("user:"):
        return
    if not admin or token != admin:
        raise HTTPException(403, "Modifying dataset documents requires X-Admin-Token")


# ---------------- schemas ----------------
class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    mode: str = Field(default="hybrid", pattern="^(dense|bm25|hybrid)$")
    top_k: int = Field(default=8, ge=1, le=50)
    use_mmr: bool = True


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    agentic: bool = True
    mode: str = Field(default="hybrid", pattern="^(dense|bm25|hybrid)$")


def _retr_cfg(mode: str, top_k: int | None = None, use_mmr: bool | None = None) -> RetrievalConfig:
    cfg = default_config()
    cfg.mode = mode
    if top_k:
        cfg.top_k = top_k
    if use_mmr is not None:
        cfg.use_mmr = use_mmr
    return cfg


# ---------------- routes ----------------
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/stats")
def stats(c: Container = Depends(get_container)):
    s = c.settings
    return {"documents": c.store.count_documents(), "chunks": c.store.count_chunks(),
            "bm25_terms": len(c.retriever.bm25.postings), "embed_model": s.embed_model,
            "llm_model": s.llm_model, "fast_model": s.llm_fast_model, "chunk_strategy": s.chunk_strategy,
            "vector_db": "qdrant-cloud" if s.qdrant_url else "qdrant-local",
            "supported_formats": sorted(PARSERS)}


@app.get("/api/documents")
def list_documents(limit: int = 50, offset: str | None = None, corpus: str | None = None,
                   c: Container = Depends(get_container)):
    docs, nxt = c.store.list_documents(min(limit, 200), offset, corpus)
    return {"documents": [d.model_dump() for d in docs], "next_offset": nxt}


@app.get("/api/documents/{doc_id:path}")
def get_document(doc_id: str, c: Container = Depends(get_container)):
    rec = c.store.get_document(doc_id)
    if rec is None:
        raise HTTPException(404, "document not found")
    return {"document": rec.model_dump(), "chunks": [h.model_dump() for h in c.store.doc_chunks(doc_id)]}


async def _read_upload(f: UploadFile) -> bytes:
    data = await f.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{f.filename}: file too large (max 10 MB)")
    return data


@app.post("/api/documents")
async def upload_documents(request: Request, files: list[UploadFile] = File(...),
                           c: Container = Depends(get_container)):
    rate_limit(request)  # uploads spend embedding quota too
    results = []
    for f in files:
        data = await _read_upload(f)
        try:
            res = await asyncio.to_thread(c.ingestion.ingest_file, data, f.filename or "upload.txt", "user")
        except UnsupportedFormatError as e:
            res = {"status": "rejected", "source": f.filename, "reason": str(e)}
        except Exception as e:  # parser failure on a corrupt file etc.
            log.exception("ingest failed")
            res = {"status": "error", "source": f.filename, "reason": f"could not process file: {e}"}
        results.append(res)
    return {"results": results}


@app.put("/api/documents/{doc_id:path}")
async def update_document(doc_id: str, file: UploadFile = File(...), x_admin_token: str | None = Header(None),
                          c: Container = Depends(get_container)):
    require_admin_for(doc_id, x_admin_token)
    rec = c.store.get_document(doc_id)
    if rec is None:
        raise HTTPException(404, "document not found")
    data = await _read_upload(file)
    corpus, _, _ = doc_id.partition(":")
    try:
        # keep the original source name so the doc_id stays the same -> versioned update
        return await asyncio.to_thread(c.ingestion.ingest_file, data, rec.source, corpus)
    except UnsupportedFormatError as e:
        raise HTTPException(415, str(e))


@app.delete("/api/documents/{doc_id:path}")
def delete_document(doc_id: str, x_admin_token: str | None = Header(None), c: Container = Depends(get_container)):
    require_admin_for(doc_id, x_admin_token)
    if not c.ingestion.delete(doc_id):
        raise HTTPException(404, "document not found")
    return {"deleted": doc_id}


@app.post("/api/search")
def search(req: SearchRequest, c: Container = Depends(get_container)):
    t = time.time()
    hits = c.retriever.search(req.query, _retr_cfg(req.mode, req.top_k, req.use_mmr))
    return {"hits": [h.model_dump() for h in hits], "latency_ms": int((time.time() - t) * 1000)}


@app.post("/api/ask")
def ask(req: AskRequest, request: Request, c: Container = Depends(get_container)):
    rate_limit(request)
    try:
        inv = c.investigator
    except LLMError as e:
        raise HTTPException(503, str(e))
    return inv.investigate(req.question, AgentConfig(agentic=req.agentic, max_steps=c.settings.agent_max_steps),
                           _retr_cfg(req.mode)).model_dump()


@app.post("/api/ask/stream")
async def ask_stream(req: AskRequest, request: Request, c: Container = Depends(get_container)):
    """Server-Sent Events: plan -> search -> tool_call ... -> answer, as the investigation progresses."""
    rate_limit(request)
    try:
        inv = c.investigator
    except LLMError as e:
        raise HTTPException(503, str(e))
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_event(kind: str, data: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (kind, data))

    def worker():
        try:
            inv.investigate(req.question, AgentConfig(agentic=req.agentic, max_steps=c.settings.agent_max_steps),
                            _retr_cfg(req.mode), on_event=on_event)
        except Exception as e:
            log.exception("stream worker failed")
            on_event("error", {"message": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=worker, daemon=True).start()

    async def gen():
        while True:
            item = await queue.get()
            if item is None:
                break
            kind, data = item
            yield f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
