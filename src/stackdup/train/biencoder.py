from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from stackdup.mining.mine_negatives import mine_bm25
from stackdup.modeling.encoder import EncoderConfig, TransformerEncoder
from stackdup.train.collate import BiEncoderCollator, Batch
from stackdup.train.dataset import JsonlPairsDataset
from stackdup.utils.config import ensure_dir, load_yaml
from stackdup.utils.jsonl import read_jsonl, write_json
from stackdup.utils.repro import set_reproducibility


def _load_mined_negatives(path: Path) -> Dict[str, List[str]]:
    m: Dict[str, List[str]] = {}
    if not path.exists():
        return m
    for r in read_jsonl(path):
        qid = str(r["query_id"])
        neg_texts = r.get("neg_texts") or []
        m[qid] = [str(t) for t in neg_texts]
    return m


def _save_checkpoint(path: Path, model: torch.nn.Module, optim: Any, sched: Any, step: int, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": sched.state_dict() if sched is not None else None,
            "step": step,
            "meta": meta,
        },
        path,
    )


@torch.no_grad()
def _eval_val_loss(model: TransformerEncoder, loader: DataLoader, temp: float, device: str) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        q = {k: v.to(device) for k, v in batch.query.items()}
        p = {k: v.to(device) for k, v in batch.pos.items()}
        qemb = model(q["input_ids"], q["attention_mask"])
        pemb = model(p["input_ids"], p["attention_mask"])
        logits = (qemb @ pemb.t()) / temp
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        total += float(loss.item()) * logits.size(0)
        n += logits.size(0)
    return total / max(1, n)


def _mined_loss(
    qemb: torch.Tensor,
    pemb_pos: torch.Tensor,
    neg_emb: torch.Tensor,
    neg_counts: List[int],
    temp: float,
) -> torch.Tensor:
    # neg_emb: [sum(negs), D]
    # Build per-example candidate matrix [B, 1+K, D] with variable K.
    bsz, dim = qemb.shape
    max_k = max(neg_counts) if neg_counts else 0
    if max_k == 0:
        return torch.tensor(0.0, device=qemb.device, dtype=qemb.dtype)

    cand = torch.zeros((bsz, 1 + max_k, dim), device=qemb.device, dtype=qemb.dtype)
    cand[:, 0, :] = pemb_pos

    offset = 0
    for i, k in enumerate(neg_counts):
        if k <= 0:
            continue
        take = min(k, max_k)
        cand[i, 1 : 1 + take, :] = neg_emb[offset : offset + take]
        offset += k

    logits = torch.einsum("bd,bkd->bk", qemb, cand) / temp
    labels = torch.zeros((bsz,), device=qemb.device, dtype=torch.long)
    return F.cross_entropy(logits, labels)


def _load_corpus_for_mining(corpus_path: Path) -> List[Tuple[str, str]]:
    return [(str(r["doc_id"]), str(r["text"])) for r in read_jsonl(corpus_path)]


def _load_pairs_in_memory(pairs_path: Path, limit: Optional[int] = None) -> List[dict]:
    out: List[dict] = []
    for r in read_jsonl(pairs_path):
        out.append(r)
        if limit is not None and len(out) >= limit:
            break
    return out


def _refresh_bm25_mining(cfg: Dict[str, Any]) -> None:
    data = cfg["data"]
    mining = cfg.get("mining", {})
    out_file = Path(mining.get("out_file", "data/mined_negatives.jsonl"))
    k = int(mining.get("k", 50))
    corpus = _load_corpus_for_mining(Path(data["corpus"]))
    pairs = _load_pairs_in_memory(Path(data["train_pairs"]))
    mined = mine_bm25(corpus=corpus, pairs=pairs, k=k)
    from stackdup.utils.jsonl import write_jsonl

    write_jsonl(out_file, mined)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a dual-encoder bi-encoder retriever (InfoNCE with optional mined negatives).")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    run = cfg["run"]
    out_dir = ensure_dir(run["out_dir"])
    ensure_dir(out_dir / "checkpoints")

    repro = set_reproducibility(seed=int(run.get("seed", 123)), deterministic=bool(run.get("deterministic", False)))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.get("mining", {}).get("enabled") and cfg.get("mining", {}).get("refresh_before_training"):
        mode = str(cfg["mining"].get("mode", "bm25"))
        if mode != "bm25":
            raise NotImplementedError("refresh_before_training currently supports mode=bm25 only. Use mining CLI for ANN mining.")
        _refresh_bm25_mining(cfg)

    mined_neg = {}
    if cfg.get("mining", {}).get("enabled"):
        mined_neg = _load_mined_negatives(Path(cfg["mining"].get("out_file", "data/mined_negatives.jsonl")))

    model_cfg = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(str(model_cfg["name_or_path"]), use_fast=True)
    model = TransformerEncoder(
        EncoderConfig(
            name_or_path=str(model_cfg["name_or_path"]),
            pooling=str(model_cfg.get("pooling", "mean")),
            normalize=bool(model_cfg.get("normalize", True)),
        )
    ).to(device)

    train_cfg = cfg["train"]
    temp = float(train_cfg.get("temperature", 0.07))
    max_negs = int(cfg.get("mining", {}).get("k", 0)) if cfg.get("mining", {}).get("enabled") else 0

    ds_train = JsonlPairsDataset(
        pairs_path=Path(cfg["data"]["train_pairs"]),
        mined_neg_by_qid=mined_neg if mined_neg else None,
        shuffle_buffer_size=1000,
        seed=int(run.get("seed", 123)),
    )
    ds_val = JsonlPairsDataset(pairs_path=Path(cfg["data"]["val_pairs"]), mined_neg_by_qid=None)

    collate = BiEncoderCollator(tokenizer, max_length=int(model_cfg.get("max_length", 128)), max_negs=max_negs)

    dl_train = DataLoader(
        ds_train,
        batch_size=int(train_cfg.get("batch_size", 32)),
        collate_fn=collate,
        num_workers=0,
    )
    dl_val = DataLoader(ds_val, batch_size=int(train_cfg.get("batch_size", 32)), collate_fn=collate, num_workers=0)

    optim = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    # Steps are unknown with IterableDataset; we use a simple fixed scheduler per epoch based on a capped step count.
    # For production-scale training, prefer a fixed max_steps and stop condition.
    approx_steps_per_epoch = int(train_cfg.get("approx_steps_per_epoch", 200))
    epochs = int(train_cfg.get("epochs", 1))
    total_steps = max(1, approx_steps_per_epoch * epochs)
    warmup = int(float(train_cfg.get("warmup_ratio", 0.0)) * total_steps)
    sched = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup, num_training_steps=total_steps)

    mp = bool(train_cfg.get("mixed_precision", False)) and (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=mp)

    meta = {
        "config_path": str(args.config),
        "config": cfg,
        "repro": repro.to_dict(),
        "device": device,
        "started_at_unix": time.time(),
    }
    write_json(out_dir / "run_meta.json", meta)

    best_val = float("inf")
    step = 0
    log_every = int(train_cfg.get("log_every", 50))
    eval_every = int(train_cfg.get("eval_every", 500))
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    mined_weight = float(train_cfg.get("mined_neg_weight", 1.0))

    model.train()
    for epoch in range(epochs):
        pbar = tqdm(dl_train, total=approx_steps_per_epoch, desc=f"train epoch {epoch+1}/{epochs}")
        for batch in pbar:
            if step >= (epoch + 1) * approx_steps_per_epoch:
                break
            step += 1

            q = {k: v.to(device) for k, v in batch.query.items()}
            p = {k: v.to(device) for k, v in batch.pos.items()}

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mp):
                qemb = model(q["input_ids"], q["attention_mask"])
                pemb = model(p["input_ids"], p["attention_mask"])
                logits = (qemb @ pemb.t()) / temp
                labels = torch.arange(logits.size(0), device=logits.device)
                loss_inbatch = F.cross_entropy(logits, labels)

                loss = loss_inbatch
                loss_mined = None
                if batch.neg is not None and batch.neg_counts is not None and any(c > 0 for c in batch.neg_counts):
                    n = {k: v.to(device) for k, v in batch.neg.items()}
                    neg_emb = model(n["input_ids"], n["attention_mask"])
                    mined = _mined_loss(qemb, pemb, neg_emb, batch.neg_counts, temp)
                    loss = loss + mined_weight * mined
                    loss_mined = mined

            optim.zero_grad(set_to_none=True)
            if mp:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optim.step()
            sched.step()

            if step % log_every == 0:
                msg = f"loss={loss.item():.4f} inbatch={loss_inbatch.item():.4f}"
                if loss_mined is not None:
                    msg += f" mined={loss_mined.item():.4f}"
                pbar.set_postfix_str(msg)

            if step % eval_every == 0:
                val_loss = _eval_val_loss(model, dl_val, temp=temp, device=device)
                write_json(out_dir / "val_metrics.json", {"step": step, "val_loss": val_loss})
                ckpt_path = out_dir / "checkpoints" / f"step_{step}.pt"
                _save_checkpoint(ckpt_path, model, optim, sched, step, {"epoch": epoch})
                if bool(train_cfg.get("save_best", True)) and val_loss < best_val:
                    best_val = val_loss
                    _save_checkpoint(out_dir / "best.pt", model, optim, sched, step, {"epoch": epoch, "val_loss": val_loss})

    # Always write a last checkpoint for convenience.
    _save_checkpoint(out_dir / "last.pt", model, optim, sched, step, {"epoch": epochs - 1})
    print(f"Finished training. out_dir={out_dir}")


if __name__ == "__main__":
    main()
