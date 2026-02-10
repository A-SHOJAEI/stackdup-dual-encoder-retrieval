from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Set, Tuple

from stackdup.utils.jsonl import read_jsonl


@dataclass(frozen=True)
class RetrievalData:
    corpus_ids: List[str]
    corpus_texts: List[str]
    queries: Dict[str, str]
    qrels: Dict[str, Set[str]]


def load_retrieval_jsonl(
    corpus_path: str | Path, queries_path: str | Path, qrels_path: str | Path
) -> RetrievalData:
    corpus_ids: List[str] = []
    corpus_texts: List[str] = []
    for r in read_jsonl(corpus_path):
        corpus_ids.append(str(r["doc_id"]))
        corpus_texts.append(str(r["text"]))

    queries: Dict[str, str] = {}
    for r in read_jsonl(queries_path):
        queries[str(r["query_id"])] = str(r["text"])

    qrels: Dict[str, Set[str]] = {}
    for r in read_jsonl(qrels_path):
        qid = str(r["query_id"])
        did = str(r["doc_id"])
        qrels.setdefault(qid, set()).add(did)

    return RetrievalData(corpus_ids=corpus_ids, corpus_texts=corpus_texts, queries=queries, qrels=qrels)
