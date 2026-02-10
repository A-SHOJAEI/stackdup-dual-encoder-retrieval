from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from stackdup.baselines.common import load_retrieval_jsonl
from stackdup.modeling.encoder import EncoderConfig, TransformerEncoder
from stackdup.utils.config import load_yaml
from stackdup.utils.jsonl import write_json
from stackdup.utils.metrics import compute_retrieval_metrics


def _load_checkpoint(model: TransformerEncoder, ckpt_path: Path) -> Dict:
    obj = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(obj["model"], strict=True)
    return obj


@torch.no_grad()
def _encode_all(
    encoder: TransformerEncoder,
    tokenizer,
    texts: List[str],
    max_length: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    encoder.eval()
    out: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="encode", unit="batch"):
        batch = texts[i : i + batch_size]
        toks = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        emb = encoder(toks["input_ids"], toks["attention_mask"]).detach().cpu().numpy()
        out.append(emb)
    return np.vstack(out).astype(np.float32)


def _rank_with_faiss(corpus_emb: np.ndarray, query_emb: np.ndarray, topk: int) -> np.ndarray:
    import faiss

    index = faiss.IndexFlatIP(corpus_emb.shape[1])
    index.add(corpus_emb)
    scores, idx = index.search(query_emb, topk)
    return idx


def _rank_bruteforce(corpus_emb: np.ndarray, query_emb: np.ndarray, topk: int) -> np.ndarray:
    sims = query_emb @ corpus_emb.T  # [nq, nd]
    idx = np.argpartition(-sims, kth=min(topk, sims.shape[1] - 1), axis=1)[:, :topk]
    # refine ordering per query
    out = np.zeros_like(idx)
    for i in range(idx.shape[0]):
        cand = idx[i]
        order = cand[np.asarray(sims[i, cand]).argsort()[::-1]]
        out[i] = order
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a bi-encoder checkpoint on retrieval metrics and write JSON results.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path; defaults to runs/.../best.pt or last.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    run = cfg.get("run", {})
    model_cfg = cfg["model"]
    eval_cfg = cfg.get("eval", {})
    data_cfg = cfg["data"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_dir = Path(run.get("out_dir", "runs/unnamed"))
    ckpt = Path(args.checkpoint) if args.checkpoint else (out_dir / "best.pt")
    if not ckpt.exists():
        ckpt = out_dir / "last.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(model_cfg["name_or_path"]), use_fast=True)
    encoder = TransformerEncoder(
        EncoderConfig(
            name_or_path=str(model_cfg["name_or_path"]),
            pooling=str(model_cfg.get("pooling", "mean")),
            normalize=bool(model_cfg.get("normalize", True)),
        )
    ).to(device)
    ckpt_obj = _load_checkpoint(encoder, ckpt)

    rd = load_retrieval_jsonl(data_cfg["corpus"], data_cfg["test_queries"], data_cfg["test_qrels"])
    qids = sorted(rd.queries.keys())
    query_texts = [rd.queries[qid] for qid in qids]

    batch_size = int(eval_cfg.get("batch_size", 64))
    max_length = int(model_cfg.get("max_length", 128))
    topks = [int(k) for k in eval_cfg.get("topk", [1, 5, 10, 50])]
    topk = max(topks)

    t0 = time.perf_counter()
    corpus_emb = _encode_all(encoder, tokenizer, rd.corpus_texts, max_length=max_length, batch_size=batch_size, device=device)
    t1 = time.perf_counter()
    query_emb = _encode_all(encoder, tokenizer, query_texts, max_length=max_length, batch_size=batch_size, device=device)
    t2 = time.perf_counter()

    use_faiss = bool(eval_cfg.get("use_faiss", True))
    try:
        idx = _rank_with_faiss(corpus_emb, query_emb, topk=topk) if use_faiss else _rank_bruteforce(corpus_emb, query_emb, topk=topk)
        rank_kind = "faiss" if use_faiss else "bruteforce"
    except Exception:
        idx = _rank_bruteforce(corpus_emb, query_emb, topk=topk)
        rank_kind = "bruteforce_fallback"

    t3 = time.perf_counter()

    rankings: Dict[str, List[str]] = {}
    for qi, qid in enumerate(qids):
        rankings[qid] = [rd.corpus_ids[j] for j in idx[qi].tolist()]

    metrics = compute_retrieval_metrics(
        rankings=rankings,
        qrels=rd.qrels,
        recall_ks=topks,
        mrr_k=int(eval_cfg.get("mrr_k", 10)),
        ndcg_k=int(eval_cfg.get("ndcg_k", 10)),
    )

    result = {
        "name": str(run.get("name", out_dir.name)),
        "kind": "biencoder",
        "checkpoint": str(ckpt),
        "config_path": str(args.config),
        "metrics": metrics,
        "timing": {
            "encode_corpus_seconds": t1 - t0,
            "encode_queries_seconds": t2 - t1,
            "rank_seconds": t3 - t2,
            "rank_kind": rank_kind,
            "n_corpus": len(rd.corpus_ids),
            "n_queries": len(qids),
            "topk": topk,
        },
        "device": device,
        "checkpoint_meta": {"step": ckpt_obj.get("step"), "meta": ckpt_obj.get("meta")},
    }
    write_json(out_path, result)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
