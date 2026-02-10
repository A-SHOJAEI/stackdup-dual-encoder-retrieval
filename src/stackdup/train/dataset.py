from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset


@dataclass(frozen=True)
class PairExample:
    query_id: str
    pos_id: str
    query_text: str
    pos_text: str
    neg_texts: Optional[List[str]] = None


class JsonlPairsDataset(IterableDataset):
    """
    Streaming dataset that optionally does deterministic buffer shuffling for large JSONL files.
    """

    def __init__(
        self,
        pairs_path: str | Path,
        mined_neg_by_qid: Optional[Dict[str, List[str]]] = None,
        shuffle_buffer_size: int = 0,
        seed: int = 0,
    ):
        super().__init__()
        self.pairs_path = Path(pairs_path)
        self.mined_neg_by_qid = mined_neg_by_qid or {}
        self.shuffle_buffer_size = int(shuffle_buffer_size)
        self.seed = int(seed)

    def _iter_raw(self) -> Iterator[PairExample]:
        with self.pairs_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                qid = str(r["query_id"])
                yield PairExample(
                    query_id=qid,
                    pos_id=str(r["pos_id"]),
                    query_text=str(r["query_text"]),
                    pos_text=str(r["pos_text"]),
                    neg_texts=self.mined_neg_by_qid.get(qid),
                )

    def __iter__(self) -> Iterator[PairExample]:
        it = self._iter_raw()
        if self.shuffle_buffer_size <= 0:
            yield from it
            return

        rng = random.Random(self.seed)
        buf: List[PairExample] = []
        for ex in it:
            buf.append(ex)
            if len(buf) >= self.shuffle_buffer_size:
                idx = rng.randrange(len(buf))
                yield buf.pop(idx)
        while buf:
            idx = rng.randrange(len(buf))
            yield buf.pop(idx)
