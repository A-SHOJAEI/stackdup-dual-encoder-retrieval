# Experiments Mapping to the Plan

This repo implements the following plan items with runnable configs and scripts.

## Dual-Encoder Baseline (Plan: hard-negative mined dual-encoder)

- Config: `configs/smoke_baseline.yaml`
- Objective: InfoNCE with in-batch negatives plus **BM25-mined hard negatives** (mined once before training for the smoke run).
- Entry point: `python -m stackdup.train.biencoder --config configs/smoke_baseline.yaml`

## Ablation (Plan: No hard negatives)

This is implemented exactly as listed in the plan:

- Config: `configs/smoke_no_hardneg.yaml`
- Difference vs baseline: `mining.enabled: false` (training uses **in-batch negatives only**)
- Entry point: `python -m stackdup.train.biencoder --config configs/smoke_no_hardneg.yaml`

## Sparse Baselines (Plan: BM25, TF-IDF)

Smoke baselines (no Java, fast):
- TF-IDF + cosine: `python -m stackdup.baselines.tfidf_eval --config configs/smoke_data.yaml --out artifacts/tfidf_results.json`
- BM25 (pure Python): `python -m stackdup.baselines.bm25_eval --config configs/smoke_data.yaml --out artifacts/bm25_results.json`

Plan primary baseline (Lucene BM25 via Pyserini), optional:
- Index: `python -m stackdup.baselines.pyserini_index --corpus data/splits/corpus.jsonl --out indices/bm25_pyserini`
- Eval: `python -m stackdup.baselines.pyserini_eval --index indices/bm25_pyserini/lucene-index --queries data/splits/test_queries.jsonl --qrels data/splits/test_qrels.jsonl --out artifacts/bm25_pyserini_results.json`

Pyserini requires additional dependencies: `requirements-pyserini.txt`.
