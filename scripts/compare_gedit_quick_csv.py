#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def load_rows(path: Path, language: str) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if language != "all" and row.get("instruction_language") != language:
                continue
            try:
                semantics = float(row["sementics_score"])
                quality = float(row["quality_score"])
            except (KeyError, TypeError, ValueError):
                continue
            rows[row["key"]] = {
                **row,
                "semantics": semantics,
                "quality": quality,
                "overall": math.sqrt(semantics * quality),
            }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two GEdit score CSVs on matching keys.")
    parser.add_argument("--baseline-csv", required=True, type=Path)
    parser.add_argument("--candidate-csv", required=True, type=Path)
    parser.add_argument("--baseline-name", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--language", default="all")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    baseline = load_rows(args.baseline_csv, args.language)
    candidate = load_rows(args.candidate_csv, args.language)
    keys = sorted(set(baseline) & set(candidate))
    if not keys:
        raise SystemExit("No matching keys to compare.")

    deltas = [float(candidate[key]["overall"]) - float(baseline[key]["overall"]) for key in keys]
    result = {
        "baseline": args.baseline_name,
        "candidate": args.candidate_name,
        "group": args.group,
        "language": args.language,
        "count": len(keys),
        "baseline_semantics": statistics.mean(float(baseline[key]["semantics"]) for key in keys),
        "candidate_semantics": statistics.mean(float(candidate[key]["semantics"]) for key in keys),
        "delta_semantics": statistics.mean(
            float(candidate[key]["semantics"]) - float(baseline[key]["semantics"]) for key in keys
        ),
        "baseline_quality": statistics.mean(float(baseline[key]["quality"]) for key in keys),
        "candidate_quality": statistics.mean(float(candidate[key]["quality"]) for key in keys),
        "delta_quality": statistics.mean(
            float(candidate[key]["quality"]) - float(baseline[key]["quality"]) for key in keys
        ),
        "baseline_overall": statistics.mean(float(baseline[key]["overall"]) for key in keys),
        "candidate_overall": statistics.mean(float(candidate[key]["overall"]) for key in keys),
        "delta_overall": statistics.mean(deltas),
        "wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "losses": sum(delta < 0 for delta in deltas),
        "per_key": [
            {
                "key": key,
                "baseline_semantics": baseline[key]["semantics"],
                "candidate_semantics": candidate[key]["semantics"],
                "delta_semantics": float(candidate[key]["semantics"]) - float(baseline[key]["semantics"]),
                "baseline_quality": baseline[key]["quality"],
                "candidate_quality": candidate[key]["quality"],
                "delta_quality": float(candidate[key]["quality"]) - float(baseline[key]["quality"]),
                "baseline_overall": baseline[key]["overall"],
                "candidate_overall": candidate[key]["overall"],
                "delta_overall": float(candidate[key]["overall"]) - float(baseline[key]["overall"]),
                "instruction": candidate[key].get("instruction", ""),
            }
            for key in keys
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_key"}, indent=2))


if __name__ == "__main__":
    main()
