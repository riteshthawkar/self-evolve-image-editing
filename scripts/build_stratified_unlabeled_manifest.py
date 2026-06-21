from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_EDIT_TYPE_CYCLE = [
    "object_removal",
    "object_replacement",
    "object_addition",
    "attribute_change",
    "color_change",
    "material_change",
    "spatial_move",
    "background_change",
    "style_transfer",
    "local_enhancement",
]

STRATA_AWARE_CYCLES = {
    "object": [
        "object_removal",
        "object_replacement",
        "object_addition",
        "attribute_change",
        "material_change",
        "spatial_move",
    ],
    "color": ["color_change", "attribute_change", "material_change", "local_enhancement"],
    "background": ["background_change", "style_transfer", "local_enhancement", "spatial_move"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def metadata_value(record: dict[str, Any], key: str, default: Any = None) -> Any:
    metadata = record.get("metadata") or {}
    if key in record:
        return record[key]
    current: Any = metadata
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def parse_weight_map(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    output: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split("=", 1)
        output[key.strip()] = float(value)
    return output


def parse_str_list(raw: str | None, default: list[str]) -> list[str]:
    if raw is None:
        return list(default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or list(default)


def finite_score(record: dict[str, Any], key: str) -> float:
    value = metadata_value(record, key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def stable_id(record: dict[str, Any]) -> str:
    return str(record.get("image") or record.get("image_path") or record.get("key"))


def largest_remainder(labels: list[str], weights: dict[str, float], total_count: int) -> dict[str, int]:
    if not labels or total_count <= 0:
        return {}
    total_weight = sum(max(0.0, weights.get(label, 0.0)) for label in labels)
    if total_weight <= 0.0:
        base = total_count // len(labels)
        quotas = {label: base for label in labels}
        for label in labels[: total_count - base * len(labels)]:
            quotas[label] += 1
        return quotas
    raw = {label: total_count * max(0.0, weights.get(label, 0.0)) / total_weight for label in labels}
    quotas = {label: int(raw[label]) for label in labels}
    remaining = total_count - sum(quotas.values())
    order = sorted(labels, key=lambda label: (raw[label] - quotas[label], weights.get(label, 0.0)), reverse=True)
    for label in order[:remaining]:
        quotas[label] += 1
    return quotas


def scheduled_edit_type(label: str, index: int, args: argparse.Namespace) -> str:
    global_cycle = parse_str_list(args.edit_type_cycle, DEFAULT_EDIT_TYPE_CYCLE)
    if args.schedule_mode == "global_cycle":
        cycle = global_cycle
    else:
        cycle = STRATA_AWARE_CYCLES.get(label, global_cycle)
    return cycle[index % len(cycle)]


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    records: list[dict[str, Any]] = []
    for input_path in args.input:
        for record in read_jsonl(input_path):
            copied = dict(record)
            metadata = dict(copied.get("metadata") or {})
            metadata.setdefault("source_manifest", str(input_path))
            copied["metadata"] = metadata
            records.append(copied)
    input_records = len(records)

    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        key = stable_id(record)
        previous = deduped.get(key)
        if previous is None or finite_score(record, args.score_key) > finite_score(previous, args.score_key):
            deduped[key] = record
    records = list(deduped.values())

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = Counter()
    include_labels = set(parse_str_list(args.include_labels, [])) if args.include_labels else set()
    for record in records:
        label = str(metadata_value(record, args.metadata_key, "unknown") or "unknown").strip() or "unknown"
        if include_labels and label not in include_labels:
            rejected["excluded_label"] += 1
            continue
        if finite_score(record, args.score_key) < args.min_score:
            rejected["low_score"] += 1
            continue
        buckets[label].append(record)

    for label, items in buckets.items():
        rng.shuffle(items)
        items.sort(key=lambda record: finite_score(record, args.score_key), reverse=True)

    target_counts = parse_weight_map(args.target_counts)
    labels = [label for label in target_counts if label in buckets]
    labels.extend(label for label in sorted(buckets) if label not in labels)
    if target_counts:
        quotas = {label: int(target_counts.get(label, 0)) for label in labels}
        remaining = max(0, args.max_records - sum(quotas.values()))
        if remaining:
            filler = largest_remainder(labels, {label: 1.0 for label in labels}, remaining)
            for label, value in filler.items():
                quotas[label] = quotas.get(label, 0) + value
    else:
        fractions = parse_weight_map(args.target_fractions)
        if not fractions:
            fractions = {label: 1.0 for label in labels}
        quotas = largest_remainder(labels, fractions, args.max_records)

    selected: list[dict[str, Any]] = []
    selected_counts = Counter()
    for label in labels:
        quota = max(0, quotas.get(label, 0))
        for record in buckets.get(label, [])[:quota]:
            copied = dict(record)
            metadata = dict(copied.get("metadata") or {})
            scheduled = scheduled_edit_type(label, selected_counts[label], args)
            metadata["primary_family"] = label
            metadata["scheduled_edit_type"] = scheduled
            metadata["target_edit_types"] = parse_str_list(args.edit_type_cycle, DEFAULT_EDIT_TYPE_CYCLE)
            metadata["stratified_manifest"] = True
            metadata["stratified_manifest_metadata_key"] = args.metadata_key
            copied["metadata"] = metadata
            selected.append(copied)
            selected_counts[label] += 1

    if len(selected) < args.max_records:
        used = {stable_id(record) for record in selected}
        fallback = [record for label in labels for record in buckets.get(label, []) if stable_id(record) not in used]
        rng.shuffle(fallback)
        fallback.sort(key=lambda record: finite_score(record, args.score_key), reverse=True)
        for record in fallback[: args.max_records - len(selected)]:
            label = str(metadata_value(record, args.metadata_key, "unknown") or "unknown").strip() or "unknown"
            copied = dict(record)
            metadata = dict(copied.get("metadata") or {})
            metadata["primary_family"] = label
            metadata["scheduled_edit_type"] = scheduled_edit_type(label, selected_counts[label], args)
            metadata["target_edit_types"] = parse_str_list(args.edit_type_cycle, DEFAULT_EDIT_TYPE_CYCLE)
            metadata["stratified_manifest"] = True
            metadata["stratified_manifest_backfill"] = True
            copied["metadata"] = metadata
            selected.append(copied)
            selected_counts[label] += 1

    rng.shuffle(selected)
    selected = selected[: args.max_records]
    summary = {
        "inputs": [str(path) for path in args.input],
        "output": str(args.output),
        "input_records": input_records,
        "deduped_records": len(records),
        "selected_records": len(selected),
        "metadata_key": args.metadata_key,
        "score_key": args.score_key,
        "min_score": args.min_score,
        "available_per_label": {label: len(items) for label, items in sorted(buckets.items())},
        "selected_per_label": dict(sorted(Counter(str(metadata_value(record, args.metadata_key, "unknown") or "unknown") for record in selected).items())),
        "scheduled_edit_type_counts": dict(sorted(Counter((record.get("metadata") or {}).get("scheduled_edit_type", "unknown") for record in selected).items())),
        "rejected": dict(rejected),
        "seed": args.seed,
        "schedule_mode": args.schedule_mode,
    }
    return selected, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stratified unlabeled self-evolution manifest.")
    parser.add_argument("--input", type=Path, action="append", required=True, help="Input JSONL manifest. Can be repeated.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=1024)
    parser.add_argument("--metadata-key", default="primary_family")
    parser.add_argument("--include-labels", default="object,color,background")
    parser.add_argument("--target-fractions", default="object=0.50,color=0.25,background=0.25")
    parser.add_argument("--target-counts", default=None)
    parser.add_argument("--score-key", default="source_selection_score")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--schedule-mode", choices=["strata_aware", "global_cycle"], default="strata_aware")
    parser.add_argument("--edit-type-cycle", default=",".join(DEFAULT_EDIT_TYPE_CYCLE))
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    manifest, summary = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in manifest:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
