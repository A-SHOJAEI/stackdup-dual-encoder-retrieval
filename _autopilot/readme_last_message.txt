Overwrote `README.md` with a repo-specific, code-faithful description of:

- The exact problem framing (duplicate detection as retrieval) and what’s implemented in smoke vs full-dump pipelines
- Dataset provenance for both the synthetic smoke dataset and the Stack Overflow Stack Exchange dump (including verification via Archive.org metadata)
- Methodology (bi-encoder architecture, InfoNCE + optional mined-negatives loss, BM25/ANN mining, FAISS ranking, metric definitions)
- Baselines and the no-hard-negatives ablation (tied to `configs/smoke_baseline.yaml` and `configs/smoke_no_hardneg.yaml`)
- Exact results copied from `artifacts/report.md` plus the delta block from `artifacts/results.json`, with explicit pointers to where they appear
- Repro commands mapped to the actual `Makefile` targets and CLIs, plus clear limitations and concrete next research steps