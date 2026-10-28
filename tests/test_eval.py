"""The eval harness itself: metric math and a full offline retrieval-eval run over the fake stack."""
from eval.run_eval import (RETRIEVAL_CONFIGS, distinct_docs, keyword_hit, load_questions, mrr, recall_at_k,
                           retrieval_metrics)


def test_metric_math():
    docs = distinct_docs(["a#0", "a#1", "b#0", "c#2"])
    assert docs == ["a", "b", "c"]
    gold = [["x", "b"], ["c"]]  # group 1 satisfied by x OR b
    assert recall_at_k(docs, gold, 1) == 0 and recall_at_k(docs, gold, 2) == 0.5 and recall_at_k(docs, gold, 3) == 1
    assert mrr(docs, gold) == 0.5 and mrr(["z"], gold) == 0
    assert keyword_hit("Root cause: CHG-2031, see  CHG-2034", [r"chg-2031", r"chg-2034", r"chg-9"]) == 2 / 3


def test_question_set_shape():
    qs = load_questions()
    cats = {q["category"] for q in qs}
    assert len(qs) >= 40
    assert {"simple", "multi_hop", "contradiction", "insufficient", "injection"} <= cats
    assert all(q["gold"] == [] for q in qs if q["reference"].startswith("INSUFFICIENT"))
    assert len({q["id"] for q in qs}) == len(qs)


def test_retrieval_eval_runs_on_novapay(novapay):
    qs = [q for q in load_questions() if q["id"].startswith("np-")]
    rows = retrieval_metrics(novapay.retriever, qs, RETRIEVAL_CONFIGS)
    assert {r["config"] for r in rows} == set(RETRIEVAL_CONFIGS)
    bm25 = [r for r in rows if r["config"] == "bm25"]
    # keyword search alone already finds most NovaPay evidence (ids like CHG-2031 are exact tokens)
    assert sum(r["R@10"] for r in bm25) / len(bm25) > 0.6
