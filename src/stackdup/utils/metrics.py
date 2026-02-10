from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


def _dcg(rels: Sequence[int], k: int) -> float:
    s = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        if rel <= 0:
            continue
        s += (2.0 ** rel - 1.0) / math.log2(i + 1.0)
    return s


def recall_at_k(ranked_doc_ids: List[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = 0
    for d in ranked_doc_ids[:k]:
        if d in relevant:
            hit = 1
            break
    return float(hit)


def mrr_at_k(ranked_doc_ids: List[str], relevant: set[str], k: int) -> float:
    for i, d in enumerate(ranked_doc_ids[:k], start=1):
        if d in relevant:
            return 1.0 / float(i)
    return 0.0


def ndcg_at_k(ranked_doc_ids: List[str], relevant: set[str], k: int) -> float:
    # Binary relevance.
    rels = [1 if d in relevant else 0 for d in ranked_doc_ids[:k]]
    dcg = _dcg(rels, k)
    ideal = _dcg(sorted(rels, reverse=True), k)
    return 0.0 if ideal == 0.0 else (dcg / ideal)


def compute_retrieval_metrics(
    rankings: Mapping[str, List[str]],
    qrels: Mapping[str, set[str]],
    recall_ks: Sequence[int],
    mrr_k: int,
    ndcg_k: int,
) -> Dict[str, float]:
    n = 0
    recall_sums = {k: 0.0 for k in recall_ks}
    mrr_sum = 0.0
    ndcg_sum = 0.0

    for qid, ranked in rankings.items():
        rel = qrels.get(qid, set())
        if not rel:
            continue
        n += 1
        for k in recall_ks:
            recall_sums[k] += recall_at_k(ranked, rel, k)
        mrr_sum += mrr_at_k(ranked, rel, mrr_k)
        ndcg_sum += ndcg_at_k(ranked, rel, ndcg_k)

    if n == 0:
        return {f"recall@{k}": 0.0 for k in recall_ks} | {"mrr@10": 0.0, "ndcg@10": 0.0, "n_eval": 0}

    out: Dict[str, float] = {f"recall@{k}": recall_sums[k] / n for k in recall_ks}
    out["mrr@10"] = mrr_sum / n
    out["ndcg@10"] = ndcg_sum / n
    out["n_eval"] = float(n)
    return out
