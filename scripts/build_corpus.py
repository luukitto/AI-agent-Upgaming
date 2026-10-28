"""Build the reproducible corpus.

1. HotpotQA (distractor, validation split): first N questions (default 6000) -> all their context
   paragraphs (54,998 unique) become documents. Deterministic: fixed split + fixed order, no sampling.
2. danluu/post-mortems: each bullet of the README becomes a short incident record.
3. Synthetic "NovaPay" incident corpus (see make_synthetic.py) in 7 file formats, with planted
   multi-hop links, contradictions, a near-duplicate and a prompt-injection line.

Outputs:
  data/corpus/hotpot_paragraphs.jsonl   {title, text, sentences}
  data/corpus/postmortems.jsonl         {title, text, url}
  data/corpus/hotpot_questions.jsonl    {id, question, answer, type, level, gold_titles, supporting}
  data/synthetic/*                      raw files
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ROOT_DIR  # noqa: E402

CORPUS = ROOT_DIR / "data" / "corpus"
POSTMORTEMS_URL = "https://raw.githubusercontent.com/danluu/post-mortems/master/README.md"


def build_hotpot(n_questions: int) -> None:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet",
                           repo_type="dataset", local_dir=ROOT_DIR / "data" / "raw")
    rows = pq.read_table(path).to_pylist()[:n_questions]
    paragraphs: dict[str, dict] = {}
    with open(CORPUS / "hotpot_questions.jsonl", "w", encoding="utf-8") as fq:
        for r in rows:
            ctx = dict(zip(r["context"]["title"], r["context"]["sentences"]))
            for title, sents in ctx.items():
                # first occurrence of a title wins (titles are Wikipedia article names -> unique)
                paragraphs.setdefault(title, {"title": title, "text": "".join(sents).strip(), "sentences": sents})
            sf = r["supporting_facts"]
            supporting = [{"title": t, "sentence": ctx[t][i].strip()}
                          for t, i in zip(sf["title"], sf["sent_id"]) if t in ctx and i < len(ctx[t])]
            fq.write(json.dumps({"id": r["id"], "question": r["question"], "answer": r["answer"],
                                 "type": r["type"], "level": r["level"],
                                 "gold_titles": sorted(set(sf["title"])), "supporting": supporting}) + "\n")
    with open(CORPUS / "hotpot_paragraphs.jsonl", "w", encoding="utf-8") as fp:
        for p in paragraphs.values():
            if p["text"]:
                fp.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"hotpot: {len(rows)} questions, {len(paragraphs)} unique paragraphs")


def build_postmortems() -> None:
    md = urllib.request.urlopen(POSTMORTEMS_URL, timeout=30).read().decode("utf-8")
    section, n = "", 0
    with open(CORPUS / "postmortems.jsonl", "w", encoding="utf-8") as f:
        for line in md.splitlines():
            if line.startswith("## "):
                section = line[3:].strip()
            m = re.match(r"^\s*(?:[-*]\s+)?\[(.+?)\]\((\S+?)\)\.?\s*(.*)$", line)
            if m and m.group(3):
                org, url, desc = m.groups()
                f.write(json.dumps({"title": f"{org} post-mortem ({section})", "text": f"{org}: {desc}",
                                    "url": url, "category": section}) + "\n")
                n += 1
    print(f"postmortems: {n} records")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-questions", type=int, default=6000)
    args = ap.parse_args()
    CORPUS.mkdir(parents=True, exist_ok=True)
    build_hotpot(args.n_questions)
    try:
        build_postmortems()
    except Exception as e:  # network optional
        print(f"postmortems skipped: {e}")
    from scripts.make_synthetic import main as make_synthetic
    make_synthetic()
