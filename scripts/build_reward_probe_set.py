#!/usr/bin/env python3
"""Build a small, edit-type-balanced probe set of real (source, edited, instruction) triplets.

This is offline analysis tooling for validating the internal rubric CEPR reward
without running the editor. It streams a public editing dataset (AnyEdit by
default, which has the richest hard-edit coverage), keeps only edit types that
map onto our structured taxonomy, and caps the number of pairs per type so the
probe set is balanced across easy and hard edits.

The output is a directory with:
    source/<key>.jpg     - the original image
    edited/<key>.jpg     - the ground-truth edited image
    manifest.jsonl       - one row per pair with instruction + mapped edit type

No GPU is required. Run with the base or qedit python; both have `datasets`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

# AnyEdit raw edit_type -> our structured taxonomy edit_type.
# NOTE: only types that map onto a canonical EDIT_TYPES bucket (edit_schema.py)
# are kept. AnyEdit's `action_change` has no canonical taxonomy bucket and the
# live curriculum never emits action edits, so it is intentionally excluded;
# including it produced a phantom type that silently decayed to
# `local_enhancement` and skewed the probe set away from the live distribution.
ANYEDIT_TYPE_MAP: dict[str, str] = {
    "remove": "object_removal",
    "replace": "object_replacement",
    "add": "object_addition",
    "color_alter": "color_change",
    "material_alter": "material_change",
    "material_change": "material_change",
    "appearance_alter": "attribute_change",
    "background_change": "background_change",
    "tone_transfer": "style_transfer",
    "tune_transfer": "style_transfer",
}

DEFAULT_TARGET_TYPES = [
    "object_removal",
    "object_replacement",
    "object_addition",
    "color_change",
    "attribute_change",
    "background_change",
    "style_transfer",
]

# AnyEdit has 383 parquet shards grouped into contiguous edit-type blocks.
# Streaming sequentially from shard 0 only yields `visual_depth` (an excluded
# conditioned type), so we target the shards that actually hold each taxonomy
# type. Probed empirically against Bin1117/AnyEdit. Multiple shards per type
# give headroom when a single shard does not contain enough usable pairs.
NUM_ANYEDIT_SHARDS = 383
TYPE_SHARD_MAP: dict[str, list[int]] = {
    "object_removal": [165, 175],
    "object_replacement": [190, 200],
    "object_addition": [220, 240, 255],
    "color_change": [110, 130, 150],
    "background_change": [285, 330],
    "attribute_change": [375, 382],
    "style_transfer": [40],
}


def shard_file(index: int) -> str:
    return f"data/train-{index:05d}-of-{NUM_ANYEDIT_SHARDS:05d}.parquet"


def stable_key(image_id: str, raw_type: str) -> str:
    digest = hashlib.sha256(f"{raw_type}:{image_id}".encode("utf-8")).hexdigest()[:12]
    return f"{raw_type}_{digest}"


def resize_cap(image: Image.Image, max_side: int) -> Image.Image:
    image = image.convert("RGB")
    if max_side <= 0:
        return image
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    if scale < 1.0:
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
    return image


def map_edit_type(raw: str) -> str | None:
    key = str(raw or "").strip().lower()
    if key in ANYEDIT_TYPE_MAP:
        return ANYEDIT_TYPE_MAP[key]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Bin1117/AnyEdit")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="data/probe/anyedit_pairs")
    parser.add_argument("--max-per-type", type=int, default=40)
    parser.add_argument("--max-rows", type=int, default=200000,
                        help="Safety cap on streamed rows before stopping.")
    parser.add_argument("--max-side", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--target-types", nargs="*", default=DEFAULT_TARGET_TYPES)
    parser.add_argument("--progress-every", type=int, default=2000)
    args = parser.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.out)
    source_dir = out_dir / "source"
    edited_dir = out_dir / "edited"
    source_dir.mkdir(parents=True, exist_ok=True)
    edited_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    target_types = [t for t in args.target_types if t in TYPE_SHARD_MAP]
    missing = [t for t in args.target_types if t not in TYPE_SHARD_MAP]
    if missing:
        print(f"WARNING: no shard mapping for {missing}; skipping.", flush=True)
    kept: Counter[str] = Counter()
    seen_raw: Counter[str] = Counter()
    rows_streamed = 0

    manifest_handle = manifest_path.open("w", encoding="utf-8")
    try:
        for mapped_type in target_types:
            shards = TYPE_SHARD_MAP[mapped_type]
            print(f"=== {mapped_type}: shards {shards} (target {args.max_per_type}) ===", flush=True)
            for shard_index in shards:
                if kept[mapped_type] >= args.max_per_type:
                    break
                data_file = shard_file(shard_index)
                ds = load_dataset(
                    args.repo_id,
                    data_files={"train": data_file},
                    split="train",
                    streaming=True,
                )
                for row in ds:
                    rows_streamed += 1
                    raw_type = row.get("edit_type")
                    seen_raw[str(raw_type)] += 1
                    mapped = map_edit_type(raw_type)
                    if mapped != mapped_type:
                        continue
                    if kept[mapped_type] >= args.max_per_type:
                        break

                    source = row.get("image_file")
                    edited = row.get("edited_file")
                    instruction = (row.get("edit_instruction") or "").strip()
                    if not isinstance(source, Image.Image) or not isinstance(edited, Image.Image):
                        continue
                    if not instruction:
                        continue

                    key = stable_key(str(row.get("image_id", rows_streamed)), str(raw_type))
                    source_path = source_dir / f"{key}.jpg"
                    edited_path = edited_dir / f"{key}.jpg"
                    try:
                        resize_cap(source, args.max_side).save(source_path, "JPEG", quality=args.jpeg_quality)
                        resize_cap(edited, args.max_side).save(edited_path, "JPEG", quality=args.jpeg_quality)
                    except Exception as exc:  # pragma: no cover - corrupt sample guard
                        print(f"skip {key}: {exc}", flush=True)
                        continue

                    record: dict[str, Any] = {
                        "key": key,
                        "edit_type_raw": str(raw_type),
                        "edit_type": mapped,
                        "instruction": instruction,
                        "source_caption": (row.get("input") or "").strip(),
                        "target_caption": (row.get("output") or "").strip(),
                        "source_path": str(source_path),
                        "edited_path": str(edited_path),
                        "source_dataset": args.repo_id,
                        "shard": shard_index,
                    }
                    manifest_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                    manifest_handle.flush()
                    kept[mapped_type] += 1
                    if kept[mapped_type] % 10 == 0 or kept[mapped_type] == args.max_per_type:
                        print(f"  {mapped_type}: {kept[mapped_type]}/{args.max_per_type}", flush=True)
                    if rows_streamed >= args.max_rows:
                        break
                if rows_streamed >= args.max_rows:
                    print("Hit max-rows safety cap.", flush=True)
                    break
    finally:
        manifest_handle.close()

    summary = {
        "repo_id": args.repo_id,
        "rows_streamed": rows_streamed,
        "kept_total": int(sum(kept.values())),
        "kept_by_type": dict(kept),
        "max_per_type": args.max_per_type,
        "target_types": list(target_types),
        "type_shard_map": {t: TYPE_SHARD_MAP[t] for t in target_types},
        "top_raw_types_seen": dict(seen_raw.most_common(20)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
