#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests
from PIL import Image


PRESETS: dict[str, dict[str, Any]] = {
    "hq_edit": {
        "repo_id": "UCSC-VLAA/HQ-Edit",
        "split": "train",
        "source_fields": [
            "input_image",
            "source_image",
            "original_image",
            "src_image",
            "image_file",
            "source",
            "input",
            "image",
        ],
    },
    "ultraedit": {
        "repo_id": "BleachNick/UltraEdit",
        "split": "FreeForm_0",
        "source_fields": [
            "input_image",
            "source_image",
            "original_image",
            "src_image",
            "image_file",
            "source",
            "input",
            "image",
        ],
    },
    "anyedit": {
        "repo_id": "Bin1117/AnyEdit",
        "split": "train",
        "source_fields": [
            "input_image",
            "source_image",
            "original_image",
            "src_image",
            "image_file",
            "source",
            "input",
            "image",
        ],
    },
    "gpt_image_edit_1_5m": {
        "repo_id": "UCSC-VLAA/GPT-Image-Edit-1.5M",
        "split": "train",
        "source_fields": [
            "input_image",
            "source_image",
            "original_image",
            "src_image",
            "image_file",
            "source",
            "input",
            "image",
        ],
    },
}

CAPTION_FIELDS = [
    "caption",
    "source_caption",
    "input_caption",
    "input",
    "prompt",
    "edit_prompt",
    "instruction",
    "edit_instruction",
    "edit",
    "text",
]

INSTRUCTION_FIELDS = [
    "instruction",
    "edit_instruction",
    "edit_prompt",
    "edit",
    "prompt",
    "input",
    "text",
    "task_instruction",
]

EDIT_TYPE_FIELDS = [
    "edit_type",
    "task",
    "category",
    "type",
    "edit_category",
]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def image_digest(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return hashlib.sha256(buffer.getvalue()).hexdigest()[:16]


def safe_name(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:80] or "dataset"


def nested_get(row: dict[str, Any], key: str) -> Any:
    current: Any = row
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def first_present(row: dict[str, Any], fields: list[str]) -> tuple[str | None, Any]:
    for field in fields:
        value = nested_get(row, field) if "." in field else row.get(field)
        if value is not None:
            return field, value
    return None, None


def scalar_text(row: dict[str, Any], fields: list[str]) -> str | None:
    _field, value = first_present(row, fields)
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return str(value)
    return None


def image_from_bytes(data: bytes) -> Image.Image | None:
    if not data:
        return None
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


def image_from_string(value: str, timeout: float) -> Image.Image | None:
    value = value.strip()
    if not value:
        return None
    if value.startswith(("http://", "https://")):
        response = requests.get(value, timeout=timeout)
        response.raise_for_status()
        return image_from_bytes(response.content)
    path = Path(value)
    if path.exists():
        return Image.open(path).convert("RGB")
    return None


def image_from_value(value: Any, timeout: float) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, bytes):
        return image_from_bytes(value)
    if isinstance(value, str):
        return image_from_string(value, timeout=timeout)
    if isinstance(value, dict):
        if isinstance(value.get("bytes"), bytes):
            image = image_from_bytes(value["bytes"])
            if image is not None:
                return image
        for key in ("path", "url", "image", "file"):
            raw = value.get(key)
            if isinstance(raw, str):
                image = image_from_string(raw, timeout=timeout)
                if image is not None:
                    return image
        if isinstance(value.get("array"), list):
            return None
    return None


def resolve_dataset_spec(raw: str) -> dict[str, Any]:
    if raw in PRESETS:
        return {"name": raw, **PRESETS[raw]}
    parts = raw.split(":")
    if len(parts) == 1:
        return {"name": safe_name(parts[0]), "repo_id": parts[0], "split": "train", "source_fields": PRESETS["hq_edit"]["source_fields"]}
    if len(parts) == 2:
        return {"name": safe_name(parts[0]), "repo_id": parts[0], "split": parts[1], "source_fields": PRESETS["hq_edit"]["source_fields"]}
    return {
        "name": safe_name(parts[0]),
        "repo_id": parts[0],
        "config": parts[1] or None,
        "split": parts[2] or "train",
        "source_fields": PRESETS["hq_edit"]["source_fields"],
    }


def load_stream(spec: dict[str, Any]):
    from datasets import get_dataset_split_names, load_dataset

    kwargs = {
        "split": spec.get("split", "train"),
        "streaming": True,
    }
    config = spec.get("config")
    try:
        if config:
            return load_dataset(spec["repo_id"], config, **kwargs)
        return load_dataset(spec["repo_id"], **kwargs)
    except ValueError as exc:
        if "Bad split" not in str(exc):
            raise
        split_names = get_dataset_split_names(spec["repo_id"], config) if config else get_dataset_split_names(spec["repo_id"])
        if not split_names:
            raise
        spec["split"] = split_names[0]
        kwargs["split"] = split_names[0]
        print(
            f"Requested split was unavailable for {spec['repo_id']}; falling back to split={split_names[0]}",
            flush=True,
        )
        if config:
            return load_dataset(spec["repo_id"], config, **kwargs)
        return load_dataset(spec["repo_id"], **kwargs)


def row_schema(row: dict[str, Any]) -> dict[str, str]:
    schema = {}
    for key, value in row.items():
        if isinstance(value, Image.Image):
            schema[key] = "PIL.Image"
        elif isinstance(value, dict):
            schema[key] = f"dict:{','.join(sorted(value)[:8])}"
        elif isinstance(value, list):
            schema[key] = f"list[{len(value)}]"
        else:
            schema[key] = type(value).__name__
    return schema


def passes_image_prefilter(image: Image.Image, args: argparse.Namespace) -> tuple[bool, str | None]:
    width, height = image.size
    short_side = min(width, height)
    long_side = max(width, height)
    if short_side < args.min_short_side:
        return False, "short_side_too_small"
    if long_side / max(short_side, 1) > args.max_aspect_ratio:
        return False, "extreme_aspect_ratio"
    return True, None


def save_image(image: Image.Image, output_dir: Path, key: str, max_side: int, quality: int) -> Path:
    image = image.convert("RGB")
    if max_side > 0:
        width, height = image.size
        scale = min(1.0, max_side / max(width, height))
        if scale < 1.0:
            image = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
    path = output_dir / f"{key}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="JPEG", quality=quality, optimize=True)
    return path


def collect_dataset(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    dataset_name = spec["name"]
    raw_dir = args.output_root / dataset_name / "images"
    manifest_path = args.output_root / dataset_name / "manifest.jsonl"
    rejected_path = args.output_root / dataset_name / "rejected.jsonl"
    schema_path = args.output_root / dataset_name / "schema.json"
    summary_path = args.output_root / dataset_name / "summary.json"

    existing = read_jsonl(manifest_path) if args.resume else []
    existing_ids = {str((row.get("metadata") or {}).get("dataset_row_id")) for row in existing}
    existing_hashes = {str((row.get("metadata") or {}).get("image_sha256_16")) for row in existing}
    selected = list(existing)
    rejected_count = 0
    scanned = 0
    started = time.time()
    stream = load_stream(spec)
    source_fields = args.source_field or list(spec.get("source_fields") or PRESETS["hq_edit"]["source_fields"])

    first_schema_written = schema_path.exists()
    for row_index, row in enumerate(stream):
        if args.max_rows_scanned and scanned >= args.max_rows_scanned:
            break
        if len(selected) >= args.max_images:
            break
        scanned += 1
        dataset_row_id = f"{spec['repo_id']}::{spec.get('config') or ''}::{spec.get('split', 'train')}::{row_index}"
        if dataset_row_id in existing_ids:
            continue
        if not first_schema_written:
            schema_path.parent.mkdir(parents=True, exist_ok=True)
            with schema_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "dataset": spec,
                        "schema": row_schema(row),
                        "source_field_priority": source_fields,
                    },
                    handle,
                    indent=2,
                    ensure_ascii=True,
                )
                handle.write("\n")
            first_schema_written = True

        source_field, raw_image = first_present(row, source_fields)
        if raw_image is None:
            append_jsonl(rejected_path, {"dataset_row_id": dataset_row_id, "reason": "missing_source_field"})
            rejected_count += 1
            continue
        try:
            image = image_from_value(raw_image, timeout=args.request_timeout)
        except Exception as exc:
            append_jsonl(
                rejected_path,
                {"dataset_row_id": dataset_row_id, "reason": "image_decode_error", "error": repr(exc)[:300]},
            )
            rejected_count += 1
            continue
        if image is None:
            append_jsonl(rejected_path, {"dataset_row_id": dataset_row_id, "reason": "unsupported_source_value"})
            rejected_count += 1
            continue
        ok, reason = passes_image_prefilter(image, args)
        if not ok:
            append_jsonl(
                rejected_path,
                {
                    "dataset_row_id": dataset_row_id,
                    "reason": reason,
                    "width": image.size[0],
                    "height": image.size[1],
                },
            )
            rejected_count += 1
            continue
        digest = image_digest(image)
        if digest in existing_hashes:
            append_jsonl(rejected_path, {"dataset_row_id": dataset_row_id, "reason": "duplicate_image_hash"})
            rejected_count += 1
            continue
        key = f"{dataset_name}_{row_index:08d}_{digest}"
        output_path = save_image(image, raw_dir, key, max_side=args.max_saved_side, quality=args.jpeg_quality)
        rel_path = output_path.relative_to(args.repo_root)
        caption = scalar_text(row, CAPTION_FIELDS)
        instruction = scalar_text(row, INSTRUCTION_FIELDS)
        edit_type = scalar_text(row, EDIT_TYPE_FIELDS)
        manifest_row = {
            "key": key,
            "image": str(rel_path),
            "caption": caption or instruction or "",
            "metadata": {
                "source_dataset": dataset_name,
                "source_repo_id": spec["repo_id"],
                "source_config": spec.get("config"),
                "source_split": spec.get("split", "train"),
                "dataset_row_id": dataset_row_id,
                "source_image_field": source_field,
                "source_instruction": instruction,
                "source_edit_type": edit_type,
                "image_sha256_16": digest,
                "width": image.size[0],
                "height": image.size[1],
                "editing_dataset_source": True,
            },
        }
        append_jsonl(manifest_path, manifest_row)
        selected.append(manifest_row)
        existing_ids.add(dataset_row_id)
        existing_hashes.add(digest)
        if args.progress_every and len(selected) % args.progress_every == 0:
            elapsed = max(time.time() - started, 1.0)
            print(
                f"{dataset_name}: selected={len(selected)} scanned={scanned} "
                f"rejected={rejected_count} rate={scanned / elapsed:.2f} rows/s",
                flush=True,
            )

    summary = {
        "dataset": spec,
        "selected": len(selected),
        "new_selected": max(0, len(selected) - len(existing)),
        "scanned_this_run": scanned,
        "rejected_this_run": rejected_count,
        "manifest_jsonl": str(manifest_path),
        "rejected_jsonl": str(rejected_path),
        "schema_json": str(schema_path),
        "raw_images_dir": str(raw_dir),
        "max_images": args.max_images,
        "max_rows_scanned": args.max_rows_scanned,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download bounded source/raw images from public image-editing datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help=(
            "Dataset preset or HF spec. Presets: hq_edit, ultraedit, anyedit, gpt_image_edit_1_5m. "
            "Custom format: repo_id or repo_id:split or repo_id:config:split."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/unlabeled/raw/editing_datasets"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--max-rows-scanned", type=int, default=5000)
    parser.add_argument("--source-field", action="append", default=None, help="Override source image field priority.")
    parser.add_argument("--min-short-side", type=int, default=384)
    parser.add_argument("--max-aspect-ratio", type=float, default=2.5)
    parser.add_argument("--max-saved-side", type=int, default=1400)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--request-timeout", type=float, default=20.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_root = (args.repo_root / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    return args


def main() -> None:
    args = build_parser()
    summaries = []
    for raw in args.dataset:
        spec = resolve_dataset_spec(raw)
        print(f"Collecting {spec['name']} from {spec['repo_id']} split={spec.get('split', 'train')}", flush=True)
        summaries.append(collect_dataset(spec, args))
    combined_path = args.output_root / "download_summary.json"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with combined_path.open("w", encoding="utf-8") as handle:
        json.dump({"summaries": summaries}, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    print(json.dumps({"summaries": summaries}, indent=2, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
