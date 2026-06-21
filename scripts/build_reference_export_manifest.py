#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"Expected JSON object rows in {path}")
                rows.append(row)
    return rows


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    tmp.replace(path)


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def clean_key(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return cleaned[:120] if cleaned else "row"


def row_family(row: dict[str, Any]) -> str:
    structured = row.get("structured_edit") if isinstance(row.get("structured_edit"), dict) else {}
    return str(row.get("family") or row.get("edit_type") or structured.get("edit_type") or "unknown")


def existing_path(raw: Any) -> bool:
    return bool(raw) and Path(str(raw)).exists()


def selected_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter[str]]:
    rng = random.Random(args.seed)
    include_families = {item.strip() for item in args.include_families.split(",") if item.strip()}
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected: Counter[str] = Counter()
    for path in args.input_manifest:
        for row in load_jsonl(path):
            family = row_family(row)
            if include_families and family not in include_families:
                rejected["family_filtered"] += 1
                continue
            if not str(row.get(args.prompt_key) or "").strip():
                rejected["missing_prompt"] += 1
                continue
            if not existing_path(row.get(args.source_image_key)):
                rejected["missing_source_image"] += 1
                continue
            if args.require_candidate and not existing_path(row.get(args.candidate_image_key)):
                rejected["missing_candidate_image"] += 1
                continue
            buckets[family].append(row)

    chosen: list[dict[str, Any]] = []
    for family, rows in sorted(buckets.items()):
        rng.shuffle(rows)
        chosen.extend(rows[: args.per_family_limit])
    rng.shuffle(chosen)
    if args.max_rows >= 0:
        chosen = chosen[: args.max_rows]
    return chosen, rejected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a custom export JSON and indexed manifest for non-eval base-reference outputs."
    )
    parser.add_argument("--input-manifest", action="append", type=Path, required=True)
    parser.add_argument("--edit-json-output", type=Path, required=True)
    parser.add_argument("--indexed-manifest-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--reference-image-root", type=Path, required=True)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--source-image-key", default="edit_image")
    parser.add_argument("--candidate-image-key", default="chosen_image")
    parser.add_argument(
        "--include-families",
        default="object_removal,object_replacement,object_addition,background,color,color_change,background_change,global_adjustment,style_transfer",
    )
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--per-family-limit", type=int, default=64)
    parser.add_argument("--require-candidate", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rows, rejected = selected_rows(args)
    edit_json: dict[str, dict[str, str]] = {}
    indexed_rows: list[dict[str, Any]] = []
    used_keys: set[str] = set()
    for index, row in enumerate(rows):
        base_key = clean_key(str(row.get("record_key") or row.get("group_id") or f"row_{index:06d}"))
        key = base_key
        suffix = 1
        while key in used_keys:
            suffix += 1
            key = f"{base_key}_{suffix}"
        used_keys.add(key)
        prompt = str(row.get(args.prompt_key)).strip()
        family = row_family(row)
        source_image = str(row.get(args.source_image_key))
        edit_json[key] = {"id": source_image, "prompt": prompt, "edit_type": family}
        indexed = dict(row)
        indexed["baseline_key"] = key
        indexed["baseline_image"] = str(args.reference_image_root / f"{key}.png")
        indexed["baseline_reference_source"] = "qwen_base_export"
        indexed_rows.append(indexed)

    write_json(edit_json, args.edit_json_output)
    write_jsonl(indexed_rows, args.indexed_manifest_output)
    summary = {
        "edit_json_output": str(args.edit_json_output),
        "indexed_manifest_output": str(args.indexed_manifest_output),
        "reference_image_root": str(args.reference_image_root),
        "rows": len(indexed_rows),
        "per_family": Counter(row_family(row) for row in indexed_rows),
        "rejected": rejected,
        "seed": args.seed,
    }
    write_json(summary, args.summary or args.indexed_manifest_output.with_suffix(".summary.json"))
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
