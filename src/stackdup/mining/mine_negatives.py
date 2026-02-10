from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from transformers import AutoTokenizer

from stackdup.modeling.encoder import EncoderConfig, TransformerEncoder
from stackdup.utils.config import load_yaml
from stackdup.utils.jsonl import read_jsonl, write_jsonl


def _tokenize_bm25(s: str) -> List[str]:
    return [t for t in s.lower().split() if t]


def mine_bm25(
    corpus: List[Tuple[str, str]],
    pairs: List[dict],
    k: int,
) -> List[dict]:
    corpus_ids = [cid for cid, _ in corpus]
    corpus_texts = [txt for _, txt in corpus]
    tok_corpus = [_tokenize_bm25(t) for t in corpus_texts]
    bm25 = BM25Okapi(tok_corpus)

    out: List[dict] = []
    for ex in tqdm(pairs, desc="BM25 mining", unit="ex"):
        qid = str(ex["query_id"])
        qtext = str(ex["query_text"])
        pos_id = str(ex["pos_id"])
        scores = np.asarray(bm25.get_scores(_tokenize_bm25(qtext)))
        topk = min(int(k) + 20, scores.shape[0])
        idx = np.argpartition(-scores, kth=topk - 1)[:topk]
        idx = idx[scores[idx].argsort()[::-1]]
        negs: List[Tuple[str, str]] = []
        for j in idx.tolist():
            did = corpus_ids[j]
            if did == pos_id:
                continue
            negs.append((did, corpus_texts[j]))
            if len(negs) >= int(k):
                break
        out.append(
            {
                "query_id": qid,
                "pos_id": pos_id,
                "neg_ids": [d for d, _ in negs],
                "neg_texts": [t for _, t in negs],
            }
        )
    return out


@torch.no_grad()
def mine_ann(
    corpus: List[Tuple[str, str]],
    pairs: List[dict],
    k: int,
    model_name_or_path: str,
    pooling: str,
    normalize: bool,
    max_length: int,
    batch_size: int = 64,
    device: str = "cpu",
    use_faiss: bool = True,
) -> List[dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    enc = TransformerEncoder(EncoderConfig(name_or_path=model_name_or_path, pooling=pooling, normalize=normalize))
    enc.eval().to(device)

    corpus_ids = [cid for cid, _ in corpus]
    corpus_texts = [txt for _, txt in corpus]

    def _encode_texts(texts: List[str]) -> np.ndarray:
        embs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            toks = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            toks = {k: v.to(device) for k, v in toks.items()}
            e = enc(toks["input_ids"], toks["attention_mask"]).detach().cpu().numpy()
            embs.append(e)
        return np.vstack(embs)

    t0 = time.perf_counter()
    corpus_emb = _encode_texts(corpus_texts).astype(np.float32)
    t1 = time.perf_counter()

    if use_faiss:
        import faiss

        index = faiss.IndexFlatIP(corpus_emb.shape[1])
        index.add(corpus_emb)
        build_s = time.perf_counter() - t1
    else:
        index = None
        build_s = 0.0

    out: List[dict] = []
    for ex in tqdm(pairs, desc="ANN mining", unit="ex"):
        qid = str(ex["query_id"])
        qtext = str(ex["query_text"])
        pos_id = str(ex["pos_id"])
        qemb = _encode_texts([qtext]).astype(np.float32)
        if index is not None:
            scores, idx = index.search(qemb, int(k) + 10)
            idx = idx[0].tolist()
        else:
            sims = (qemb @ corpus_emb.T)[0]
            idx = np.argpartition(-sims, kth=min(len(sims) - 1, int(k) + 9))[: int(k) + 10]
            idx = idx[np.asarray(sims)[idx].argsort()[::-1]].tolist()

        negs: List[Tuple[str, str]] = []
        for j in idx:
            did = corpus_ids[j]
            if did == pos_id:
                continue
            negs.append((did, corpus_texts[j]))
            if len(negs) >= int(k):
                break
        out.append(
            {
                "query_id": qid,
                "pos_id": pos_id,
                "neg_ids": [d for d, _ in negs],
                "neg_texts": [t for _, t in negs],
            }
        )
    return out


def _load_corpus(corpus_path: Path) -> List[Tuple[str, str]]:
    return [(str(r["doc_id"]), str(r["text"])) for r in read_jsonl(corpus_path)]


def _load_pairs(pairs_path: Path, limit: Optional[int] = None) -> List[dict]:
    out: List[dict] = []
    for r in read_jsonl(pairs_path):
        out.append(r)
        if limit is not None and len(out) >= limit:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Mine hard negatives (BM25 or ANN/FAISS) for training pairs.")
    ap.add_argument("--config", required=True, help="Bi-encoder YAML config (uses data.* and mining.*)")
    ap.add_argument("--mode", choices=["bm25", "ann"], default=None)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--out", dest="out_file", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of pairs (debug)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data = cfg["data"]
    mining = cfg.get("mining", {})
    mode = args.mode or mining.get("mode", "bm25")
    k = int(args.k or mining.get("k", 50))
    out_file = Path(args.out_file or mining.get("out_file", "data/mined_negatives.jsonl"))

    corpus = _load_corpus(Path(data["corpus"]))
    pairs = _load_pairs(Path(data["train_pairs"]), limit=args.limit)

    if mode == "bm25":
        mined = mine_bm25(corpus=corpus, pairs=pairs, k=k)
    else:
        model = cfg["model"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mined = mine_ann(
            corpus=corpus,
            pairs=pairs,
            k=k,
            model_name_or_path=str(model["name_or_path"]),
            pooling=str(model.get("pooling", "mean")),
            normalize=bool(model.get("normalize", True)),
            max_length=int(model.get("max_length", 128)),
            device=device,
            use_faiss=bool(cfg.get("eval", {}).get("use_faiss", True)),
        )

    write_jsonl(out_file, mined)
    print(f"Wrote mined negatives to {out_file}")


if __name__ == "__main__":
    main()
