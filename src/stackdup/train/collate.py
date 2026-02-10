from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerBase

from stackdup.train.dataset import PairExample


@dataclass(frozen=True)
class Batch:
    query_ids: List[str]
    pos_ids: List[str]
    query: Dict[str, torch.Tensor]
    pos: Dict[str, torch.Tensor]
    neg: Optional[Dict[str, torch.Tensor]]
    neg_counts: Optional[List[int]]


class BiEncoderCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int, max_negs: int = 0):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.max_negs = int(max_negs)

    def __call__(self, examples: List[PairExample]) -> Batch:
        q_texts = [ex.query_text for ex in examples]
        p_texts = [ex.pos_text for ex in examples]

        q = self.tokenizer(
            q_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        p = self.tokenizer(
            p_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        neg_texts_flat: List[str] = []
        neg_counts: List[int] = []
        for ex in examples:
            negs = ex.neg_texts or []
            if self.max_negs > 0:
                negs = negs[: self.max_negs]
            neg_counts.append(len(negs))
            neg_texts_flat.extend(negs)

        if neg_texts_flat:
            n = self.tokenizer(
                neg_texts_flat,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            n = None

        return Batch(
            query_ids=[ex.query_id for ex in examples],
            pos_ids=[ex.pos_id for ex in examples],
            query=q,
            pos=p,
            neg=n,
            neg_counts=neg_counts if neg_texts_flat else None,
        )
