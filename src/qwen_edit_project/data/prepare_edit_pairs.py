from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageStat

from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import ensure_dir, relative_to_repo, resolve_path


def safe_key(value: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return key[:180] or "item"


def open_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            from io import BytesIO

            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        for key in ("path", "file_name", "filename"):
            if value.get(key):
                return Image.open(value[key]).convert("RGB")
    if isinstance(value, str):
        return Image.open(value).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)!r}")


def resize_for_storage(image: Image.Image, max_long_side: int | None) -> Image.Image:
    if not max_long_side or max(image.size) <= max_long_side:
        return image
    resized = image.copy()
    resized.thumbnail((max_long_side, max_long_side), Image.Resampling.LANCZOS)
    return resized


def save_image(image: Image.Image, path: Path, image_format: str = "jpg", quality: int = 92) -> None:
    ensure_dir(path.parent)
    kwargs: dict[str, Any] = {}
    if image_format.lower() in {"jpg", "jpeg", "webp"}:
        kwargs["quality"] = quality
    if image_format.lower() in {"jpg", "jpeg"}:
        kwargs["optimize"] = True
    image.save(path, **kwargs)


def image_stats(image: Image.Image) -> dict[str, float]:
    resized = image.convert("RGB").resize((256, 256))
    gray = resized.convert("L")
    hsv = resized.convert("HSV")
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=1.5))
    high_freq = ImageChops.difference(gray, blurred)
    luminance = ImageStat.Stat(gray)
    saturation = ImageStat.Stat(hsv.split()[1])
    edges = ImageStat.Stat(gray.filter(ImageFilter.FIND_EDGES)).mean[0] / 255.0
    histogram = np.asarray(gray.histogram(), dtype=np.float64)
    probs = histogram[histogram > 0] / max(histogram.sum(), 1.0)
    entropy = float(-(probs * np.log2(probs)).sum() / 8.0)
    return {
        "width": float(image.width),
        "height": float(image.height),
        "short_side": float(min(image.width, image.height)),
        "aspect_ratio": float(max(image.width, image.height) / max(min(image.width, image.height), 1)),
        "luminance_mean": float(luminance.mean[0] / 255.0),
        "luminance_std": float(luminance.stddev[0] / 255.0),
        "saturation_mean": float(saturation.mean[0] / 255.0),
        "sharpness": float(ImageStat.Stat(high_freq).stddev[0] / 255.0),
        "edge_density": float(edges),
        "entropy": entropy,
    }


def normalized_difference(source: Image.Image, target: Image.Image) -> dict[str, float]:
    source_small = source.convert("RGB").resize((256, 256))
    target_small = target.convert("RGB").resize((256, 256))
    diff = ImageChops.difference(source_small, target_small).convert("L")
    values = np.asarray(diff, dtype=np.float32) / 255.0
    changed = values > 0.08
    changed_fraction = float(changed.mean())
    diff_mean = float(values.mean())
    if changed.any():
        ys, xs = np.where(changed)
        bbox_area = float(((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) / (256 * 256))
    else:
        bbox_area = 0.0
    return {
        "diff_mean": diff_mean,
        "changed_fraction": changed_fraction,
        "change_bbox_fraction": bbox_area,
        "preservation_fraction": 1.0 - changed_fraction,
    }


def instruction_score(instruction: str) -> tuple[float, list[str]]:
    text = instruction.strip()
    lowered = text.lower()
    reasons = []
    if len(text.split()) < 3:
        reasons.append("instruction_too_short")
    if len(text) > 220:
        reasons.append("instruction_too_long")
    vague_terms = {"something", "anything", "better", "nice", "good", "improve it", "make it cool"}
    if any(term in lowered for term in vague_terms):
        reasons.append("vague_instruction")
    action_terms = {
        "add",
        "remove",
        "replace",
        "change",
        "make",
        "turn",
        "convert",
        "adjust",
        "move",
        "color",
        "style",
        "background",
        "object",
        "text",
    }
    action_score = 1.0 if any(term in lowered for term in action_terms) else 0.55
    length_score = min(1.0, max(0.0, (len(text.split()) - 2) / 8.0))
    return 0.65 * action_score + 0.35 * length_score, reasons


def quality_score(stats: dict[str, float]) -> float:
    short_side = min(1.0, max(0.0, (stats["short_side"] - 256) / 768))
    aspect = 1.0 if stats["aspect_ratio"] <= 2.4 else max(0.0, 1.0 - (stats["aspect_ratio"] - 2.4) / 1.2)
    exposure = 1.0 - min(abs(stats["luminance_mean"] - 0.50) / 0.45, 1.0)
    contrast = min(1.0, stats["luminance_std"] / 0.22)
    sharpness = min(1.0, stats["sharpness"] / 0.08)
    entropy = min(1.0, stats["entropy"] / 0.75)
    return 0.20 * short_side + 0.15 * aspect + 0.15 * exposure + 0.15 * contrast + 0.20 * sharpness + 0.15 * entropy


def pair_score(source: Image.Image, target: Image.Image, instruction: str, thresholds: dict[str, Any]) -> dict[str, Any]:
    src_stats = image_stats(source)
    tgt_stats = image_stats(target)
    diff_stats = normalized_difference(source, target)
    instr_score, instruction_reasons = instruction_score(instruction)
    src_quality = quality_score(src_stats)
    tgt_quality = quality_score(tgt_stats)

    changed_fraction = diff_stats["changed_fraction"]
    preservation_fraction = diff_stats["preservation_fraction"]
    edit_visibility = min(1.0, changed_fraction / max(float(thresholds.get("target_changed_fraction", 0.08)), 1e-6))
    preservation_score = min(1.0, preservation_fraction / max(float(thresholds.get("min_preservation_fraction", 0.45)), 1e-6))
    locality_score = 1.0 - max(0.0, changed_fraction - float(thresholds.get("soft_max_changed_fraction", 0.55))) / 0.45
    locality_score = max(0.0, min(1.0, locality_score))
    score = (
        0.20 * src_quality
        + 0.15 * tgt_quality
        + 0.20 * instr_score
        + 0.20 * edit_visibility
        + 0.15 * preservation_score
        + 0.10 * locality_score
    )

    reject_reasons = list(instruction_reasons)
    if src_stats["short_side"] < float(thresholds.get("min_short_side", 384)):
        reject_reasons.append("source_too_small")
    if tgt_stats["short_side"] < float(thresholds.get("min_short_side", 384)):
        reject_reasons.append("target_too_small")
    if max(src_stats["aspect_ratio"], tgt_stats["aspect_ratio"]) > float(thresholds.get("max_aspect_ratio", 2.6)):
        reject_reasons.append("extreme_aspect_ratio")
    if src_quality < float(thresholds.get("min_source_quality", 0.45)):
        reject_reasons.append("low_source_quality")
    if tgt_quality < float(thresholds.get("min_target_quality", 0.40)):
        reject_reasons.append("low_target_quality")
    if changed_fraction < float(thresholds.get("min_changed_fraction", 0.03)):
        reject_reasons.append("edit_too_small_or_invisible")
    if changed_fraction > float(thresholds.get("max_changed_fraction", 0.75)):
        reject_reasons.append("edit_changes_too_much")
    if preservation_fraction < float(thresholds.get("min_preservation_fraction", 0.45)):
        reject_reasons.append("low_preservation")
    if instr_score < float(thresholds.get("min_instruction_score", 0.50)):
        reject_reasons.append("weak_instruction")
    if score < float(thresholds.get("min_total_score", 0.55)):
        reject_reasons.append("low_total_score")

    return {
        "score": score,
        "accepted": not reject_reasons,
        "reject_reasons": sorted(set(reject_reasons)),
        "source_quality": src_quality,
        "target_quality": tgt_quality,
        "instruction_score": instr_score,
        "source_stats": src_stats,
        "target_stats": tgt_stats,
        "diff": diff_stats,
    }


def load_hf_dataset(config: dict[str, Any]):
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
    if dataset_cfg.get("cache_dir"):
        kwargs["cache_dir"] = str(resolve_path(dataset_cfg["cache_dir"]))
    dataset = load_dataset(**kwargs)
    if bool(dataset_cfg.get("shuffle", False)) and hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=int(config.get("runtime", {}).get("seed", 123)))
    return dataset


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def prepare_edit_pairs(config: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    columns = config["columns"]
    output = config["output"]
    filters = config.get("filters", {})
    storage = config.get("storage", {})

    output_root = ensure_dir(resolve_path(output["root_dir"]))
    source_dir = ensure_dir(output_root / "source")
    target_dir = ensure_dir(output_root / "target")
    all_records_path = resolve_path(output["all_records_jsonl"])
    selected_records_path = resolve_path(output["selected_records_jsonl"])
    rejected_records_path = resolve_path(output["rejected_records_jsonl"])
    manifest_path = resolve_path(output["manifest_json"])
    summary_path = resolve_path(output["summary_json"])
    if None in {all_records_path, selected_records_path, rejected_records_path, manifest_path, summary_path}:
        raise ValueError("All output paths must resolve")

    image_format = storage.get("image_format", "jpg").lower()
    suffix = "jpg" if image_format == "jpeg" else image_format
    quality = int(storage.get("quality", 92))
    max_long_side = int(storage.get("max_long_side", 1536) or 0) or None
    target_count = limit if limit is not None else int(dataset_cfg.get("limit", 0) or 0)
    max_selected = int(output.get("max_selected", 0) or 0)
    progress_every = int(output.get("progress_every", 100))
    resume = bool(output.get("resume", True))
    stream_records = bool(output.get("stream_records", True))

    dataset = load_hf_dataset(config)
    all_records = read_jsonl(all_records_path) if resume else []
    selected: list[dict[str, Any]] = [record for record in all_records if record.get("accepted")]
    rejected: list[dict[str, Any]] = [record for record in all_records if not record.get("accepted")]
    processed_keys = {str(record["key"]) for record in all_records if record.get("key")}
    if resume and processed_keys:
        print(f"Resuming edit-pair filtering with {len(processed_keys)} existing records.", flush=True)

    seen = 0
    errors = 0
    record_handle = all_records_path.open("a" if resume else "w", encoding="utf-8") if stream_records else None
    try:
        for row_index, row in enumerate(dataset):
            if target_count and seen >= target_count:
                break
            seen += 1
            try:
                raw_id = str(row.get(columns.get("id", ""), row_index)) if columns.get("id") else str(row_index)
                key_parts = [str(dataset_cfg.get("tag", dataset_cfg["path"])), raw_id]
                if columns.get("turn"):
                    key_parts.append(str(row.get(columns["turn"], "0")))
                key = safe_key("__".join(key_parts))
                if key in processed_keys:
                    continue
                instruction = str(row[columns["instruction"]]).strip()
                source = resize_for_storage(open_image(row[columns["source_image"]]), max_long_side)
                target = resize_for_storage(open_image(row[columns["target_image"]]), max_long_side)
                source_path = source_dir / f"{key}_source.{suffix}"
                target_path = target_dir / f"{key}_target.{suffix}"
                save_image(source, source_path, image_format=image_format, quality=quality)
                save_image(target, target_path, image_format=image_format, quality=quality)
                score_data = pair_score(source, target, instruction, filters)
                record = {
                    "key": key,
                    "source_image": relative_to_repo(source_path),
                    "target_image": relative_to_repo(target_path),
                    "instruction": instruction,
                    "dataset": dataset_cfg["path"],
                    "split": dataset_cfg.get("split", "train"),
                    "row_index": row_index,
                    "score": score_data["score"],
                    "accepted": score_data["accepted"],
                    "reject_reasons": score_data["reject_reasons"],
                    "metrics": {
                        k: v for k, v in score_data.items() if k not in {"accepted", "reject_reasons"}
                    },
                }
                all_records.append(record)
                processed_keys.add(key)
                if record_handle is not None:
                    record_handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                    record_handle.flush()
                if record["accepted"]:
                    selected.append(record)
                else:
                    rejected.append(record)
                if progress_every > 0 and seen % progress_every == 0:
                    print(
                        f"Processed {seen}; total_records={len(all_records)} "
                        f"selected={len(selected)} rejected={len(rejected)}",
                        flush=True,
                    )
            except Exception as exc:
                errors += 1
                if errors <= 20:
                    print(f"Skipping row {row_index}: {exc}", flush=True)
    finally:
        if record_handle is not None:
            record_handle.close()

    selected = sorted(selected, key=lambda item: item["score"], reverse=True)
    if max_selected:
        rejected.extend({**item, "accepted": False, "reject_reasons": ["selection_budget_exceeded"]} for item in selected[max_selected:])
        selected = selected[:max_selected]
    manifest = [
        {
            "prompt": item["instruction"],
            "image": item["target_image"],
            "edit_image": item["source_image"],
        }
        for item in selected
    ]

    write_jsonl(all_records, all_records_path)
    write_jsonl(selected, selected_records_path)
    write_jsonl(rejected, rejected_records_path)
    save_json(manifest, manifest_path)
    summary = {
        "dataset": dataset_cfg,
        "seen": seen,
        "selected": len(selected),
        "rejected": len(rejected),
        "errors": errors,
        "manifest_json": str(manifest_path),
        "selected_records_jsonl": str(selected_records_path),
        "reject_reasons": Counter(reason for item in rejected for reason in item.get("reject_reasons", [])).most_common(),
    }
    save_json(summary, summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and filter source-target instruction edit pairs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], help="Override config using dotted.key=value")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)
    prepare_edit_pairs(config, limit=args.limit)


if __name__ == "__main__":
    main()
