from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _primary_family(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return str(record.get("primary_family") or metadata.get("primary_family") or "unknown")


def _edit_families(record: dict[str, Any]) -> set[str]:
    metadata = record.get("metadata") or {}
    raw = record.get("edit_families") or metadata.get("edit_families") or []
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",")]
    return {str(item) for item in raw if item}


def _score(record: dict[str, Any]) -> float:
    metadata = record.get("metadata") or {}
    value = record.get("source_selection_score", metadata.get("source_selection_score", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_object_manifest(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    records = _read_jsonl(args.input)
    object_primary = [record for record in records if _primary_family(record) == "object"]
    object_secondary = [
        record
        for record in records
        if _primary_family(record) != "object" and "object" in _edit_families(record)
    ]

    object_primary.sort(key=_score, reverse=True)
    object_secondary.sort(key=_score, reverse=True)
    if args.shuffle_ties:
        # Stable score buckets keep high-quality records first while avoiding repeated image-order artifacts.
        def shuffle_equal_scores(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            output: list[dict[str, Any]] = []
            bucket: list[dict[str, Any]] = []
            last_score: float | None = None
            for item in items:
                score = round(_score(item), 6)
                if last_score is not None and score != last_score:
                    rng.shuffle(bucket)
                    output.extend(bucket)
                    bucket = []
                bucket.append(item)
                last_score = score
            rng.shuffle(bucket)
            output.extend(bucket)
            return output

        object_primary = shuffle_equal_scores(object_primary)
        object_secondary = shuffle_equal_scores(object_secondary)

    selected = object_primary[: args.max_primary]
    if args.include_secondary and len(selected) < args.max_records:
        selected.extend(object_secondary[: args.max_records - len(selected)])
    selected = selected[: args.max_records]

    edit_type_cycle = [item.strip() for item in args.edit_type_cycle.split(",") if item.strip()]
    if not edit_type_cycle:
        edit_type_cycle = ["object_removal", "object_replacement", "object_addition"]

    for index, record in enumerate(selected):
        scheduled_edit_type = edit_type_cycle[index % len(edit_type_cycle)]
        metadata = dict(record.get("metadata") or {})
        metadata["experiment_focus"] = (
            "object-primary self-evolution: removal/replacement/addition only; "
            "reject global color/style/background-only edits"
        )
        metadata["target_edit_types"] = ["object_removal", "object_replacement", "object_addition"]
        metadata["scheduled_edit_type"] = scheduled_edit_type
        metadata["avoid_edit_types"] = ["global_adjustment", "color_change", "style_transfer", "background_change"]
        metadata["object_primary_manifest"] = True
        record["metadata"] = metadata

    _write_jsonl(selected, args.output)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "max_records": args.max_records,
        "max_primary": args.max_primary,
        "include_secondary": args.include_secondary,
        "input_count": len(records),
        "object_primary_available": len(object_primary),
        "object_secondary_available": len(object_secondary),
        "selected_count": len(selected),
        "selected_primary_counts": Counter(_primary_family(record) for record in selected),
        "scheduled_edit_type_counts": Counter(
            (record.get("metadata") or {}).get("scheduled_edit_type", "unknown") for record in selected
        ),
    }
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an unlabeled manifest whose first records are true object-primary images.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/unlabeled/selected/magicbrush_all_images_moe/manifest_object_removal_replacement_focus_rounds256.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/unlabeled/selected/magicbrush_all_images_moe/manifest_object_primary_focus_rounds96.jsonl"),
    )
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=96)
    parser.add_argument("--max-primary", type=int, default=96)
    parser.add_argument("--include-secondary", action="store_true")
    parser.add_argument("--shuffle-ties", action="store_true")
    parser.add_argument(
        "--edit-type-cycle",
        default="object_removal,object_replacement,object_addition",
        help="Comma-separated target edit_type cycle assigned to selected records.",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    summary = build_object_manifest(args)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
