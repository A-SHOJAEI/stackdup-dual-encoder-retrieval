from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from stackdup.utils.jsonl import read_json, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect multiple result JSON files into artifacts/results.json.")
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    experiments: List[Dict[str, Any]] = []
    for p in args.inputs:
        experiments.append(read_json(p))

    # Convenience comparisons: if we have a biencoder baseline + its no-hardneg ablation, compute deltas.
    by_name = {e.get("name"): e for e in experiments}
    comparisons: Dict[str, Any] = {}

    # Heuristic: use names containing "baseline" and "no_hardneg" or "no_hardneg".
    base = next((e for e in experiments if "baseline" in str(e.get("name", "")).lower()), None)
    abl = next((e for e in experiments if "no_hardneg" in str(e.get("name", "")).lower()), None)
    if base and abl and base.get("metrics") and abl.get("metrics"):
        deltas = {}
        for k, v in base["metrics"].items():
            if k in abl["metrics"] and isinstance(v, (int, float)) and isinstance(abl["metrics"][k], (int, float)):
                deltas[k] = float(v) - float(abl["metrics"][k])
        comparisons["biencoder_baseline_minus_no_hardneg"] = deltas

    out = {"experiments": experiments, "comparisons": comparisons}
    write_json(Path(args.out), out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
