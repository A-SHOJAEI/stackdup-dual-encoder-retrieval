from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {p}, got {type(data)}")
    return data


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class Paths:
    corpus: Path
    train_pairs: Path
    val_pairs: Path
    test_queries: Path
    test_qrels: Path


def parse_smoke_paths(cfg: Dict[str, Any]) -> Paths:
    data = cfg.get("data") or cfg.get("paths") or {}
    try:
        return Paths(
            corpus=Path(data["corpus"]),
            train_pairs=Path(data["train_pairs"]),
            val_pairs=Path(data["val_pairs"]),
            test_queries=Path(data["test_queries"]),
            test_qrels=Path(data["test_qrels"]),
        )
    except KeyError as e:
        raise KeyError(f"Missing config key: {e}") from e
