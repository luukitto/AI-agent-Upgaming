"""Unit tests for the self-implemented algorithms: chunking, dedup, BM25, RRF, MMR, parsers."""
import numpy as np
import pytest

from app.ingestion.chunking import FixedSizeChunker, SentenceChunker
from app.ingestion.dedup import DuplicateIndex, content_hash, hamming, simhash
from app.ingestion.parsers import UnsupportedFormatError, parse_file
from app.llm.client import parse_json_loose
from app.retrieval.bm25 import BM25Index
from app.retrieval.fusion import mmr, reciprocal_rank_fusion
from tests.conftest import SYNTHETIC

TEXT = ("# Root cause\nChange CHG-2031 reduced the pool. Load was low on Monday. "
        "At the peak requests queued and timed out.\n\n# Impact\nledger-service was delayed by 2 hours.")


def test_fixed_chunker_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = FixedSizeChunker(200, 50).split(text)
    assert all(len(c) <= 200 for c in chunks)
    assert len(chunks) > 1
    assert chunks[1].split()[0] in chunks[0]  # overlap
    assert set(text.split()) == {w for c in chunks for w in c.split()}  # nothing lost


def test_sentence_chunker_keeps_sentences_and_headings():
    chunks = SentenceChunker(80, 20).split(TEXT)
    assert all(c.startswith(("[Root cause]", "[Impact]")) for c in chunks)
    assert any("Change CHG-2031 reduced the pool." in c for c in chunks)  # sentence not cut
    assert any("ledger-service was delayed by 2 hours." in c for c in chunks)


def test_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        FixedSizeChunker(100, 100)


def test_exact_and_near_duplicates():
    a = (SYNTHETIC / "INC-0312_confluence_export.txt").read_text()
    b = a + "\nExported again by wiki-bot."
    assert content_hash(a) == content_hash(a.upper())  # normalised
    assert hamming(simhash(a), simhash(b)) <= 3
    other = (SYNTHETIC / "INC-0325_false_declines.md").read_text()
    assert hamming(simhash(a), simhash(other)) > 10

    idx = DuplicateIndex()
    idx.add("d1", content_hash(a), simhash(a))
    assert idx.find_exact(content_hash(a)) == "d1"
    assert idx.find_near(simhash(b))[0] == "d1"
    assert idx.find_near(simhash(b), exclude="d1") is None
    idx.remove("d1")
    assert idx.find_exact(content_hash(a)) is None


def test_bm25_ranking_and_removal():
    idx = BM25Index()
    idx.add("a#0", "a", "payment gateway connection pool exhausted")
    idx.add("b#0", "b", "fraud model declined legitimate payments")
    idx.add("c#0", "c", "weather report for tbilisi")
    res = idx.search("connection pool", k=3)
    assert res[0][0] == "a#0" and len(res) == 1
    assert {c for c, _ in idx.search("payments")} == {"a#0", "b#0"}  # stemming: payments ~ payment
    assert [c for c, _ in idx.search("payment", doc_ids={"b"})] == ["b#0"]
    idx.remove_doc("a")
    assert idx.search("connection pool") == []
    assert len(idx) == 2


def test_bm25_persistence(tmp_path):
    idx = BM25Index()
    idx.add("a#0", "a", "ledger reconciliation failed")
    idx.save(tmp_path / "b.pkl")
    loaded = BM25Index.load(tmp_path / "b.pkl")
    assert loaded.search("reconciliation")[0][0] == "a#0"
    loaded.add("b#0", "b", "new term")  # defaultdict restored
    assert loaded.search("term")[0][0] == "b#0"


def test_rrf():
    fused = reciprocal_rank_fusion({"dense": ["x", "y", "z"], "bm25": ["y", "w"]}, k=60)
    assert fused[0][0] == "y"
    assert fused[0][2] == {"dense": 2, "bm25": 1}
    assert {f[0] for f in fused} == {"x", "y", "z", "w"}
    assert reciprocal_rank_fusion({"dense": ["x"], "bm25": ["w"]}, weights={"bm25": 2.0})[0][0] == "w"


def test_mmr_prefers_diversity():
    q = np.array([1, 0, 0], dtype=np.float32)
    cands = np.array([[1, 0, 0], [1, 0.01, 0], [0.7, 0.7, 0], [0, 0, 1]], dtype=np.float32)
    rel = np.array([1.0, 0.99, 0.95, 0.0])
    assert mmr(q, cands, rel, k=2, lam=0.5) == [0, 2]  # skips the near-identical item
    assert mmr(q, cands, rel, k=2, lam=1.0) == [0, 1]  # pure relevance
    assert mmr(q, cands[:0], rel[:0], k=2) == []


@pytest.mark.parametrize("name,doc_type,needle", [
    ("INC-0312_payment_delays.pdf", "pdf", "CHG-2031"),
    ("INC-0318_ledger_reconciliation.docx", "docx", "migration 0047"),
    ("INC-0325_false_declines.md", "md", "v7.2"),
    ("oncall_handbook.txt", "txt", "db_pool_wait_ms"),
    ("q1_reliability_review.html", "html", "TLS certificate rotation"),
    ("slo_policy.json", "json", "payment-gateway"),
    ("change_log.csv", "csv", "CHG-2045"),
])
def test_parsers(name, doc_type, needle):
    doc = parse_file((SYNTHETIC / name).read_bytes(), name)
    assert doc.doc_type == doc_type
    assert needle in doc.text
    assert doc.title and doc.metadata["char_count"] == len(doc.text)


def test_parser_metadata():
    md = parse_file((SYNTHETIC / "INC-0325_false_declines.md").read_bytes(), "x.md")
    assert md.title.startswith("Incident Report INC-0325") and md.metadata["severity"] == "SEV-2"
    assert "2026-03-19" in md.metadata["dates"]
    with pytest.raises(UnsupportedFormatError):
        parse_file(b"x", "a.exe")


def test_parse_json_loose():
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_loose('Sure! {"a": [1]} hope it helps') == {"a": [1]}
    with pytest.raises(ValueError):
        parse_json_loose("no json here")
