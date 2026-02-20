"""Generate synthetic Stack Overflow duplicate-question retrieval data for smoke tests."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

from stackdup.utils.config import load_yaml
from stackdup.utils.jsonl import write_jsonl


_TOPICS = [
    "python", "javascript", "java", "c++", "html", "css", "sql",
    "react", "node", "django", "flask", "pandas", "numpy", "git",
    "docker", "kubernetes", "linux", "bash", "regex", "api",
]

_VERBS = [
    "how to", "why does", "can I", "what is the best way to",
    "error when", "problem with", "issue with", "cannot",
    "how do I", "is there a way to",
]

_OBJECTS = [
    "sort a list", "read a file", "connect to database", "parse JSON",
    "handle exceptions", "use async", "create a class", "iterate over",
    "filter results", "deploy to production", "write unit tests",
    "configure logging", "install packages", "set up environment",
    "merge branches", "resolve conflicts", "optimize performance",
    "fix memory leak", "debug error", "implement authentication",
]


def _random_text(rng: random.Random, vocab_size: int, style: str) -> str:
    topic = rng.choice(_TOPICS)
    verb = rng.choice(_VERBS)
    obj = rng.choice(_OBJECTS)
    if style == "title_body":
        title = f"{verb} {obj} in {topic}"
        body_words = [rng.choice(_TOPICS + _OBJECTS[:vocab_size % len(_OBJECTS)]) for _ in range(rng.randint(5, 15))]
        return f"{title}\n{' '.join(body_words)}"
    return f"{verb} {obj} in {topic}"


def _make_duplicate(rng: random.Random, text: str) -> str:
    """Create a paraphrased duplicate by shuffling words and changing phrasing."""
    lines = text.split("\n")
    title = lines[0] if lines else text
    words = title.split()
    if len(words) > 3:
        i, j = rng.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)


def generate(cfg_path: str | Path) -> None:
    cfg = load_yaml(cfg_path)
    ds = cfg.get("dataset", cfg)
    out_dir = Path(ds["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(ds.get("seed", 123))
    n_docs = int(ds.get("n_docs", 300))
    n_queries = int(ds.get("n_queries", 120))
    vocab_size = int(ds.get("vocab_size", 200))
    dup_per_query = int(ds.get("dup_per_query", 1))
    style = str(ds.get("text_style", "title_body"))

    rng = random.Random(seed)

    # Generate corpus
    corpus: List[Dict[str, Any]] = []
    for i in range(n_docs):
        corpus.append({
            "doc_id": f"d{i:05d}",
            "text": _random_text(rng, vocab_size, style),
        })

    # Generate queries + qrels + pairs
    train_pairs: List[Dict[str, Any]] = []
    val_pairs: List[Dict[str, Any]] = []
    test_queries: List[Dict[str, Any]] = []
    test_qrels: List[Dict[str, Any]] = []

    n_train = int(n_queries * 0.6)
    n_val = int(n_queries * 0.2)

    for qi in range(n_queries):
        # Each query is a "duplicate" of a random doc
        pos_docs = rng.sample(corpus, min(dup_per_query, len(corpus)))
        query_text = _make_duplicate(rng, pos_docs[0]["text"])
        qid = f"q{qi:05d}"

        for pos_doc in pos_docs:
            pair = {
                "query_id": qid,
                "pos_id": pos_doc["doc_id"],
                "query_text": query_text,
                "pos_text": pos_doc["text"],
            }
            if qi < n_train:
                train_pairs.append(pair)
            elif qi < n_train + n_val:
                val_pairs.append(pair)
            else:
                test_queries.append({"query_id": qid, "text": query_text})
                test_qrels.append({"query_id": qid, "doc_id": pos_doc["doc_id"], "relevance": 1})

    # Derive paths from config or defaults
    paths = cfg.get("paths", {})
    write_jsonl(Path(paths.get("corpus", out_dir / "corpus.jsonl")), corpus)
    write_jsonl(Path(paths.get("train_pairs", out_dir / "train_pairs.jsonl")), train_pairs)
    write_jsonl(Path(paths.get("val_pairs", out_dir / "val_pairs.jsonl")), val_pairs)
    write_jsonl(Path(paths.get("test_queries", out_dir / "test_queries.jsonl")), test_queries)
    write_jsonl(Path(paths.get("test_qrels", out_dir / "test_qrels.jsonl")), test_qrels)

    print(f"Generated synthetic data: {len(corpus)} docs, {len(train_pairs)} train, "
          f"{len(val_pairs)} val, {len(test_queries)} test queries in {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    generate(args.config)


if __name__ == "__main__":
    main()
