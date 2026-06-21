#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_COMPONENTS = (
    "rubric_forbidden_after_absent",
    "rubric_edit_success",
    "rubric_required_after",
    "rubric_preservation",
    "rubric_cepr_raw_reward",
    "cepr_raw_reward",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def _component_scores(row: dict[str, Any]) -> dict[str, Any]:
    scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
    components = scores.get("component_scores") if isinstance(scores.get("component_scores"), dict) else None
    if components is not None:
        return components
    nested_scores = row.get("scores", {})
    if isinstance(nested_scores, dict) and isinstance(nested_scores.get("component_scores"), dict):
        return nested_scores["component_scores"]
    return {}


def _edit_type(row: dict[str, Any]) -> str:
    structured = row.get("structured_edit") if isinstance(row.get("structured_edit"), dict) else {}
    return str(structured.get("edit_type") or row.get("family") or "unknown")


def _status(row: dict[str, Any]) -> str:
    return str(row.get("candidate_status") or row.get("training_status") or row.get("status") or "unknown")


def _float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def summarize(path: Path, min_forbidden: float, min_success: float) -> None:
    rows = _load_jsonl(path)
    print(f"\n### {path}")
    print(f"rows={len(rows)}")
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[_edit_type(row)].append(row)

    for edit_type, group in sorted(by_type.items(), key=lambda item: (-len(item[1]), item[0])):
        statuses = Counter(_status(row) for row in group)
        metrics: dict[str, list[float]] = {name: [] for name in DEFAULT_COMPONENTS}
        low_forbidden = 0
        low_success = 0
        for row in group:
            components = _component_scores(row)
            forbidden = _float(components.get("rubric_forbidden_after_absent"))
            success = _float(components.get("rubric_edit_success"))
            if forbidden is not None and forbidden < min_forbidden:
                low_forbidden += 1
            if success is not None and success < min_success:
                low_success += 1
            for name in DEFAULT_COMPONENTS:
                value = _float(components.get(name))
                if value is not None:
                    metrics[name].append(value)

        metric_text = " ".join(
            f"{name}={mean(values):.4f}" for name, values in metrics.items() if values
        )
        print(
            f"{edit_type:22s} total={len(group):4d} "
            f"statuses={dict(statuses)} "
            f"low_forbidden<{min_forbidden:g}={low_forbidden:4d} "
            f"low_success<{min_success:g}={low_success:4d} "
            f"{metric_text}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit self-evolution SFT manifests for contract-violating training rows."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="train_manifest.jsonl or train_weights.jsonl paths")
    parser.add_argument("--min-forbidden", type=float, default=0.30)
    parser.add_argument("--min-success", type=float, default=0.40)
    args = parser.parse_args()
    for path in args.paths:
        summarize(path, min_forbidden=args.min_forbidden, min_success=args.min_success)


if __name__ == "__main__":
    main()
