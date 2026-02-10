from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from stackdup.baselines.common import load_retrieval_jsonl
from stackdup.utils.config import load_yaml
from stackdup.utils.jsonl import write_json
from stackdup.utils.metrics import compute_retrieval_metrics


def tfidf_rank(corpus_texts: List[str], query_texts: List[str], topk: int) -> List[List[int]]:
    vec = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=200000,
        ngram_range=(1, 2),
    )
    X = vec.fit_transform(corpus_texts)
    Q = vec.transform(query_texts)
    sims = cosine_similarity(Q, X)  # shape [nq, nd]

    # Argpartition for speed; refine ordering.
    import numpy as np

    top_idx = np.argpartition(-sims, kth=min(topk, sims.shape[1] - 1), axis=1)[:, :topk]
    out: List[List[int]] = []
    for i in range(top_idx.shape[0]):
        idx = top_idx[i]
        scores = sims[i, idx]
        order = idx[scores.argsort()[::-1]]
        out.append(order.tolist())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="TF-IDF + cosine sparse baseline evaluation.")
    ap.add_argument("--config", required=True, help="YAML config containing paths.* to corpus/queries/qrels")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    paths = cfg.get("paths") or cfg.get("data") or {}
    data = load_retrieval_jsonl(paths["corpus"], paths["test_queries"], paths["test_qrels"])

    qids = sorted(data.queries.keys())
    query_texts = [data.queries[qid] for qid in qids]
    t0 = time.perf_counter()
    idx_rankings = tfidf_rank(data.corpus_texts, query_texts, topk=int(args.topk))
    t1 = time.perf_counter()

    rankings: Dict[str, List[str]] = {}
    for qid, idxs in zip(qids, idx_rankings):
        rankings[qid] = [data.corpus_ids[j] for j in idxs]

    metrics = compute_retrieval_metrics(
        rankings=rankings,
        qrels=data.qrels,
        recall_ks=(1, 5, 10, 50),
        mrr_k=10,
        ndcg_k=10,
    )
    result = {
        "name": "tfidf_cosine",
        "kind": "baseline",
        "metrics": metrics,
        "timing": {"rank_seconds": t1 - t0, "n_queries": len(qids), "topk": int(args.topk)},
    }
    write_json(Path(args.out), result)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
