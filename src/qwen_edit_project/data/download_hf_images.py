from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image

from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import ensure_dir, relative_to_repo, resolve_path


def _safe_key(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned[:160] or "item"


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def _read_existing_keys(metadata_path: Path) -> set[str]:
    if not metadata_path.exists():
        return set()
    keys = set()
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            keys.add(str(item["key"]))
    return keys


def _caption_from_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        strings = []
        for item in value:
            if isinstance(item, str):
                strings.append(item)
            elif isinstance(item, dict) and "caption" in item:
                strings.append(str(item["caption"]))
        return strings[0].strip() if strings else None
    if isinstance(value, dict):
        for key in ("caption", "text", "raw", "sentence"):
            if key in value and value[key]:
                return str(value[key]).strip()
    return str(value).strip() or None


def _open_image_from_value(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        for key in ("path", "file_name", "filename"):
            if value.get(key):
                return Image.open(value[key]).convert("RGB")
        for key in ("url", "coco_url", "flickr_url"):
            if value.get(key):
                with urllib.request.urlopen(value[key], timeout=30) as response:
                    return Image.open(io.BytesIO(response.read())).convert("RGB")
    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            with urllib.request.urlopen(value, timeout=30) as response:
                return Image.open(io.BytesIO(response.read())).convert("RGB")
        return Image.open(value).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)!r}")


def _resize_for_storage(image: Image.Image, max_long_side: int | None) -> Image.Image:
    if not max_long_side or max(image.size) <= max_long_side:
        return image
    resized = image.copy()
    resized.thumbnail((max_long_side, max_long_side), Image.Resampling.LANCZOS)
    return resized


def _save_image(image: Image.Image, path: Path, image_format: str, quality: int) -> None:
    ensure_dir(path.parent)
    kwargs: dict[str, Any] = {}
    if image_format.lower() in {"jpg", "jpeg", "webp"}:
        kwargs["quality"] = quality
    if image_format.lower() in {"jpg", "jpeg"}:
        kwargs["optimize"] = True
    image.save(path, **kwargs)


def _load_dataset_from_config(config: dict[str, Any]):
    from datasets import load_dataset

    dataset_cfg = config["dataset"]
    kwargs: dict[str, Any] = {
        "path": dataset_cfg["path"],
        "split": dataset_cfg.get("split", "train"),
        "streaming": bool(dataset_cfg.get("streaming", False)),
        "trust_remote_code": bool(dataset_cfg.get("trust_remote_code", False)),
    }
    if dataset_cfg.get("name"):
        kwargs["name"] = dataset_cfg["name"]
    if dataset_cfg.get("data_dir"):
        kwargs["data_dir"] = dataset_cfg["data_dir"]
    if dataset_cfg.get("cache_dir"):
        kwargs["cache_dir"] = str(resolve_path(dataset_cfg["cache_dir"]))
    return load_dataset(**kwargs)


def _iter_dataset(config: dict[str, Any]):
    dataset = _load_dataset_from_config(config)
    dataset_cfg = config["dataset"]
    seed = int(config.get("runtime", {}).get("seed", 123))
    shuffle_buffer = int(dataset_cfg.get("shuffle_buffer", 0) or 0)
    if shuffle_buffer > 0 and bool(dataset_cfg.get("streaming", False)) and hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    elif bool(dataset_cfg.get("shuffle", False)) and hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=seed)
    return dataset


def download_hf_images(config: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    output_cfg = config["output"]
    filters = config.get("filters", {})
    image_column = dataset_cfg.get("image_column", "image")
    caption_column = dataset_cfg.get("caption_column")
    id_column = dataset_cfg.get("id_column")
    skip = int(dataset_cfg.get("skip", 0) or 0)
    target_count = limit if limit is not None else int(dataset_cfg.get("limit", 0) or 0)

    output_dir = ensure_dir(resolve_path(output_cfg.get("images_dir", "data/unlabeled/raw")))
    metadata_path = resolve_path(output_cfg.get("metadata_jsonl", "data/unlabeled/raw/metadata.jsonl"))
    summary_path = resolve_path(output_cfg.get("summary_json", "data/unlabeled/raw/download_summary.json"))
    if metadata_path is None or summary_path is None:
        raise ValueError("output metadata and summary paths must resolve")
    ensure_dir(metadata_path.parent)

    image_format = output_cfg.get("image_format", "jpg").lower()
    suffix = "jpg" if image_format == "jpeg" else image_format
    quality = int(output_cfg.get("quality", 92))
    max_long_side = int(output_cfg.get("max_long_side", 1536) or 0) or None
    min_short_side = int(filters.get("min_short_side", 384))
    max_aspect_ratio = float(filters.get("max_aspect_ratio", 2.8))
    resume = bool(output_cfg.get("resume", True))
    progress_every = int(output_cfg.get("progress_every", 100))
    existing_keys = _read_existing_keys(metadata_path) if resume else set()

    seen = 0
    saved = 0
    skipped = 0
    errors = 0
    records: list[dict[str, Any]] = []
    dataset = _iter_dataset(config)
    metadata_handle = metadata_path.open("a" if resume else "w", encoding="utf-8")
    try:
        for row_index, row in enumerate(dataset):
            if row_index < skip:
                continue
            if target_count and saved >= target_count:
                break
            seen += 1
            try:
                raw_key = str(row.get(id_column, row_index)) if id_column else str(row_index + skip)
                key = f"{_safe_key(raw_key)}_{_short_hash(dataset_cfg['path'] + ':' + raw_key)}"
                if key in existing_keys:
                    skipped += 1
                    continue
                image = _open_image_from_value(row[image_column])
                width, height = image.size
                aspect_ratio = max(width, height) / max(min(width, height), 1)
                if min(width, height) < min_short_side or aspect_ratio > max_aspect_ratio:
                    skipped += 1
                    continue
                image = _resize_for_storage(image, max_long_side)
                image_path = output_dir / f"{key}.{suffix}"
                _save_image(image, image_path, image_format=image_format, quality=quality)
                caption = _caption_from_value(row.get(caption_column)) if caption_column else None
                record = {
                    "key": key,
                    "image": relative_to_repo(image_path),
                    "caption": caption,
                    "source": {
                        "dataset": dataset_cfg["path"],
                        "name": dataset_cfg.get("name"),
                        "split": dataset_cfg.get("split", "train"),
                        "row_index": row_index,
                        "raw_id": raw_key,
                    },
                    "width": image.width,
                    "height": image.height,
                }
                metadata_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                metadata_handle.flush()
                records.append(record)
                existing_keys.add(key)
                saved += 1
                if progress_every > 0 and saved % progress_every == 0:
                    print(f"Saved {saved} images after scanning {seen} rows.")
            except Exception as exc:
                errors += 1
                if errors <= 10:
                    print(f"Skipping row {row_index}: {exc}")
                continue
    finally:
        metadata_handle.close()

    summary = {
        "config_path": config.get("_config_path"),
        "dataset": dataset_cfg,
        "images_dir": str(output_dir),
        "metadata_jsonl": str(metadata_path),
        "seen_rows": seen,
        "saved": saved,
        "skipped": skipped,
        "errors": errors,
        "limit": target_count,
    }
    save_json(summary, summary_path)
    print(f"Saved {saved} images to {output_dir}")
    print(f"Metadata: {metadata_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a bounded HF image pool for source-image filtering.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], help="Override config using dotted.key=value")
    parser.add_argument("--seed-sample", type=int, default=None, help="Randomly sample N metadata records into sample_metadata.jsonl.")
    return parser


def _write_seed_sample(metadata_jsonl: str, count: int, seed: int) -> Path:
    path = resolve_path(metadata_jsonl)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Metadata JSONL not found: {metadata_jsonl}")
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    rng = random.Random(seed)
    sample = records if len(records) <= count else rng.sample(records, count)
    sample_path = path.with_name("sample_metadata.jsonl")
    with sample_path.open("w", encoding="utf-8") as handle:
        for item in sample:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")
    return sample_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)
    summary = download_hf_images(config, limit=args.limit)
    if args.seed_sample:
        seed = int(config.get("runtime", {}).get("seed", 123))
        sample_path = _write_seed_sample(summary["metadata_jsonl"], args.seed_sample, seed=seed)
        print(f"Sample metadata: {sample_path}")


if __name__ == "__main__":
    main()
