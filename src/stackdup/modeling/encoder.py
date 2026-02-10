from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


Pooling = Literal["cls", "mean"]


@dataclass(frozen=True)
class EncoderConfig:
    name_or_path: str
    pooling: Pooling
    normalize: bool


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = AutoModel.from_pretrained(cfg.name_or_path)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = out.last_hidden_state  # [B, T, H]

        if self.cfg.pooling == "cls":
            emb = last_hidden[:, 0, :]
        elif self.cfg.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)  # [B, T, 1]
            denom = mask.sum(dim=1).clamp_min(1.0)
            emb = (last_hidden * mask).sum(dim=1) / denom
        else:
            raise ValueError(f"Unknown pooling: {self.cfg.pooling}")

        if self.cfg.normalize:
            emb = F.normalize(emb, p=2, dim=-1)
        return emb

