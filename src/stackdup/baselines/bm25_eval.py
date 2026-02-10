from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

from rank_bm25 import BM25Okapi

from stackdup.baselines.common import load_retrieval_jsonl
from stackdup.utils.config import load_yaml
from stackdup.utils.jsonl import write_json
from stackdup.utils.metrics import compute_retrieval_metrics


def _tokenize(s: str) -> List[str]:
    return [t for t in s.lower().split() if t]


def main() -> None:
    ap = argparse.ArgumentParser(description="Small-scale BM25 baseline evaluation (pure Python).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg.get("paths") or cfg.get("data") or {}
    data = load_retrieval_jsonl(paths["corpus"], paths["test_queries"], paths["test_qrels"])

    tok_corpus = [_tokenize(t) for t in data.corpus_texts]
    bm25 = BM25Okapi(tok_corpus)

    qids = sorted(data.queries.keys())
    t0 = time.perf_counter()
    rankings: Dict[str, List[str]] = {}
    for qid in qids:
        scores = bm25.get_scores(_tokenize(data.queries[qid]))
        # Top-k indices
        import numpy as np

        scores = np.asarray(scores)
        topk = min(int(args.topk), scores.shape[0])
        idx = np.argpartition(-scores, kth=topk - 1)[:topk]
        idx = idx[scores[idx].argsort()[::-1]]
        rankings[qid] = [data.corpus_ids[i] for i in idx.tolist()]
    t1 = time.perf_counter()

    metrics = compute_retrieval_metrics(
        rankings=rankings,
        qrels=data.qrels,
        recall_ks=(1, 5, 10, 50),
        mrr_k=10,
        ndcg_k=10,
    )
    result = {
        "name": "bm25_python",
        "kind": "baseline",
        "metrics": metrics,
        "timing": {"rank_seconds": t1 - t0, "n_queries": len(qids), "topk": int(args.topk)},
        "notes": "This is a small-scale BM25 implementation for smoke runs. For the plan's Lucene BM25, use the optional Pyserini baseline.",
    }
    write_json(Path(args.out), result)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
