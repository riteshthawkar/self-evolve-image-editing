from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from qwen_edit_project.utils.config import save_json
from qwen_edit_project.utils.paths import ensure_dir, resolve_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return path


def get_nested(record: dict[str, Any], dotted_key: str, default: str = "unknown") -> str:
    current: Any = record
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    if current is None:
        return default
    return str(current)


def balanced_order(records: list[dict[str, Any]], stratify_key: str, seed: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rng = random.Random(seed)
    for record in records:
        groups[get_nested(record, stratify_key)].append(record)
    for group in groups.values():
        rng.shuffle(group)

    ordered: list[dict[str, Any]] = []
    keys = sorted(groups)
    while keys:
        next_keys = []
        for key in keys:
            group = groups[key]
            if group:
                ordered.append(group.pop())
            if group:
                next_keys.append(key)
        keys = next_keys
    return ordered


def split_manifest(
    input_path: Path,
    output_dir: Path,
    pilot_count: int,
    main_count: int,
    heldout_count: int,
    seed: int,
    stratify_key: str,
) -> dict[str, Any]:
    records = read_jsonl(input_path)
    ordered = balanced_order(records, stratify_key=stratify_key, seed=seed)
    pilot = ordered[:pilot_count]
    main_start = pilot_count
    main = ordered[main_start : main_start + main_count]
    heldout_start = main_start + main_count
    heldout = ordered[heldout_start : heldout_start + heldout_count]
    remainder = ordered[heldout_start + heldout_count :]

    ensure_dir(output_dir)
    paths = {
        "pilot": write_jsonl(pilot, output_dir / "pilot_manifest.jsonl"),
        "main": write_jsonl(main, output_dir / "main_manifest.jsonl"),
        "heldout": write_jsonl(heldout, output_dir / "heldout_manifest.jsonl"),
        "remainder": write_jsonl(remainder, output_dir / "remainder_manifest.jsonl"),
    }
    summary = {
        "input_manifest": str(input_path),
        "output_dir": str(output_dir),
        "seed": seed,
        "stratify_key": stratify_key,
        "counts": {
            "input": len(records),
            "pilot": len(pilot),
            "main": len(main),
            "heldout": len(heldout),
            "remainder": len(remainder),
        },
        "paths": {name: str(path) for name, path in paths.items()},
    }
    save_json(summary, output_dir / "split_summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic source-manifest splits for remote experiments.")
    parser.add_argument("--input", required=True, help="Selected source-image manifest JSONL.")
    parser.add_argument("--output-dir", default="data/unlabeled/splits")
    parser.add_argument("--pilot-count", type=int, default=128)
    parser.add_argument("--main-count", type=int, default=1024)
    parser.add_argument("--heldout-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--stratify-key", default="metadata.primary_family")
    args = parser.parse_args()

    input_path = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    if input_path is None or not input_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {args.input}")
    if output_dir is None:
        raise ValueError("Could not resolve output directory")

    summary = split_manifest(
        input_path=input_path,
        output_dir=output_dir,
        pilot_count=args.pilot_count,
        main_count=args.main_count,
        heldout_count=args.heldout_count,
        seed=args.seed,
        stratify_key=args.stratify_key,
    )
    print(json.dumps(summary["counts"], indent=2))
    print(f"Splits written to {output_dir}")


if __name__ == "__main__":
    main()
