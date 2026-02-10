from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

from stackdup.utils.jsonl import read_jsonl, write_json
from stackdup.utils.metrics import compute_retrieval_metrics


def _require_pyserini():
    try:
        from pyserini.search.lucene import LuceneSearcher  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Pyserini is not installed. Install optional deps with: "
            ".venv/bin/pip install -r requirements-pyserini.txt"
        ) from e


def main() -> None:
    _require_pyserini()
    ap = argparse.ArgumentParser(description="Evaluate Lucene BM25 via Pyserini (optional baseline).")
    ap.add_argument("--index", required=True, help="Path to lucene index directory")
    ap.add_argument("--queries", required=True, help="test_queries.jsonl")
    ap.add_argument("--qrels", required=True, help="test_qrels.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(args.index)
    searcher.set_bm25(k1=0.9, b=0.4)

    queries: Dict[str, str] = {}
    for r in read_jsonl(args.queries):
        queries[str(r["query_id"])] = str(r["text"])

    qrels: Dict[str, set[str]] = {}
    for r in read_jsonl(args.qrels):
        qrels.setdefault(str(r["query_id"]), set()).add(str(r["doc_id"]))

    t0 = time.perf_counter()
    rankings: Dict[str, List[str]] = {}
    for qid, qtext in queries.items():
        hits = searcher.search(qtext, k=int(args.topk))
        rankings[qid] = [str(h.docid) for h in hits]
    t1 = time.perf_counter()

    metrics = compute_retrieval_metrics(
        rankings=rankings,
        qrels=qrels,
        recall_ks=(1, 5, 10, 50),
        mrr_k=10,
        ndcg_k=10,
    )

    result = {
        "name": "bm25_pyserini_lucene",
        "kind": "baseline",
        "metrics": metrics,
        "timing": {"rank_seconds": t1 - t0, "n_queries": len(queries), "topk": int(args.topk)},
    }
    write_json(Path(args.out), result)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
