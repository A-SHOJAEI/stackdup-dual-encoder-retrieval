from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

from stackdup.utils.jsonl import read_jsonl


def _require_pyserini():
    try:
        import pyserini  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Pyserini is not installed. Install optional deps with: "
            ".venv/bin/pip install -r requirements-pyserini.txt"
        ) from e


def _write_pyserini_json_docs(corpus_jsonl: Path, docs_dir: Path) -> None:
    docs_dir.mkdir(parents=True, exist_ok=True)
    out_path = docs_dir / "docs.json"
    with out_path.open("w", encoding="utf-8") as f:
        for r in read_jsonl(corpus_jsonl):
            # Pyserini expects {"id": "...", "contents": "..."} in JSON collection.
            doc = {"id": str(r["doc_id"]), "contents": str(r["text"])}
            f.write(json.dumps(doc, ensure_ascii=True) + "\n")


def main() -> None:
    _require_pyserini()
    ap = argparse.ArgumentParser(description="Build a Lucene BM25 index via Pyserini (optional baseline).")
    ap.add_argument("--corpus", required=True, help="corpus.jsonl (doc_id/text)")
    ap.add_argument("--out", required=True, help="Index output directory")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    docs_dir = out_dir / "pyserini_json_collection"
    _write_pyserini_json_docs(corpus, docs_dir)

    from pyserini.index.lucene import IndexReader  # noqa: F401

    # Use the official CLI via module to reduce API churn.
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(docs_dir),
        "--index",
        str(out_dir / "lucene-index"),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(int(args.threads)),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    subprocess.run(cmd, check=True)
    print(f"Wrote Pyserini index to {out_dir/'lucene-index'}")


if __name__ == "__main__":
    main()
