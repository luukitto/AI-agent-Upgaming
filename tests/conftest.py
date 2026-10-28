"""Test doubles: in-memory Qdrant, a deterministic hashing embedder and a scripted LLM.

No network and no API key needed; the real algorithms (chunking, dedup, BM25, RRF, MMR, agent loop,
citation verification, API) all run unchanged on top of these fakes.
"""
import json
import os
import tempfile
import zlib
from pathlib import Path

# must happen before app.config is imported anywhere: keep tests away from the real data dir / keys
_tmp = tempfile.mkdtemp(prefix="inv-test-")
os.environ.update({"BM25_CACHE_PATH": str(Path(_tmp) / "bm25.pkl"), "LLM_API_KEY": "", "EMBED_API_KEY": "",
                   "QDRANT_URL": "", "ADMIN_TOKEN": "secret"})

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402

from app.ingestion.chunking import SentenceChunker  # noqa: E402
from app.ingestion.pipeline import IngestionService  # noqa: E402
from app.llm.client import LLMError  # noqa: E402
from app.retrieval.bm25 import tokenize  # noqa: E402
from app.retrieval.hybrid import HybridRetriever  # noqa: E402
from app.store.vector_store import VectorStore  # noqa: E402

DIM = 64
SYNTHETIC = Path(__file__).resolve().parent.parent / "data" / "synthetic"


class FakeEmbedder:
    """Bag-of-words feature hashing -> unit vector. Similar wording => similar vectors."""
    dim = DIM
    model = "fake-hashing"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for tok in tokenize(text):
            v[zlib.crc32(tok.encode()) % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v + 1 / np.sqrt(DIM)

    def embed_documents(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, DIM), np.float32)

    def embed_query(self, text):
        return self._vec(text)


class ScriptedLLM:
    """Returns pre-baked assistant messages in order; records the requests it received."""

    def __init__(self, responses=()):
        self.responses = list(responses)
        self.requests = []

    def chat(self, **kw):
        self.requests.append(kw)
        if not self.responses:
            raise LLMError("script exhausted")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return {"content": "", "tool_calls": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}, **r}


def tool_call(name, args, id_="t1"):
    return {"tool_calls": [{"id": id_, "name": name, "arguments": json.dumps(args)}]}


class FakeContainer:
    def __init__(self):
        from app.config import get_settings
        self.settings = get_settings()
        self.store = VectorStore(QdrantClient(":memory:"), "chunks", "documents", DIM)
        self.embedder = FakeEmbedder()
        self.retriever = HybridRetriever(self.store, self.embedder)
        self.ingestion = IngestionService(self.store, self.embedder, self.retriever, SentenceChunker(800, 120))
        self.llm = ScriptedLLM()

    @property
    def investigator(self):
        from app.agent.investigator import Investigator
        return Investigator(self.llm, self.retriever, self.store)


@pytest.fixture
def container():
    return FakeContainer()


@pytest.fixture
def novapay(container):
    """Container with the synthetic NovaPay corpus indexed."""
    for p in sorted(SYNTHETIC.iterdir()):
        container.ingestion.ingest_file(p.read_bytes(), p.name, corpus="novapay")
    return container
