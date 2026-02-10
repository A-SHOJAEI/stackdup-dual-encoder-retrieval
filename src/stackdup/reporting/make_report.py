from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from stackdup.utils.jsonl import read_json


def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _metric_row(e: Dict[str, Any]) -> Dict[str, Any]:
    m = e.get("metrics", {}) or {}
    return {
        "name": e.get("name", "unknown"),
        "kind": e.get("kind", "unknown"),
        "recall@1": m.get("recall@1", 0.0),
        "recall@5": m.get("recall@5", 0.0),
        "recall@10": m.get("recall@10", 0.0),
        "recall@50": m.get("recall@50", 0.0),
        "mrr@10": m.get("mrr@10", 0.0),
        "ndcg@10": m.get("ndcg@10", 0.0),
        "n_eval": m.get("n_eval", 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Render artifacts/report.md from artifacts/results.json.")
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = read_json(args.results)
    experiments: List[Dict[str, Any]] = results.get("experiments", [])
    rows = [_metric_row(e) for e in experiments]

    # Sort: baselines first, then biencoder.
    def _key(r):
        kind = str(r["kind"])
        return (0 if kind == "baseline" else 1, str(r["name"]))

    rows.sort(key=_key)

    md = []
    md.append("# Retrieval Results\n")
    md.append("This report is generated from `artifacts/results.json`.\n")

    headers = ["name", "kind", "recall@1", "recall@5", "recall@10", "recall@50", "mrr@10", "ndcg@10", "n_eval"]
    md.append("| " + " | ".join(headers) + " |")
    md.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        md.append("| " + " | ".join(_fmt(r[h]) for h in headers) + " |")

    comps = results.get("comparisons", {}) or {}
    if comps:
        md.append("\n## Comparisons\n")
        for name, delta in comps.items():
            md.append(f"### {name}\n")
            md.append("```json")
            import json

            md.append(json.dumps(delta, indent=2, sort_keys=True))
            md.append("```")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
