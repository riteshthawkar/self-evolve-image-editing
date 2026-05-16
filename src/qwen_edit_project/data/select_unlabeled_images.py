from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageStat

from qwen_edit_project.self_evolve.data import IMAGE_EXTENSIONS
from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import ensure_dir, relative_to_repo, resolve_path


EDIT_FAMILIES = {"exposure", "color", "contrast", "tone", "style", "object", "background", "local"}


@dataclass
class ImageStats:
    width: int
    height: int
    aspect_ratio: float
    megapixels: float
    luminance_mean: float
    luminance_std: float
    saturation_mean: float
    edge_density: float
    sharpness: float
    entropy: float
    average_hash: str


@dataclass
class VLMJudgment:
    caption: str | None
    quality_score: float
    natural_image_score: float
    editable_content_score: float
    object_region_clarity: float
    preservation_potential: float
    clutter_penalty: float
    text_watermark_penalty: float
    edit_families: list[str]
    reject_reasons: list[str]
    raw_response: str | None = None


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _score_range(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 1.0
    if value < low:
        return _clamp(value / max(low, 1e-6))
    return _clamp(1.0 - (value - high) / max(high, 1e-6))


def _average_hash(image: Image.Image, hash_size: int = 8) -> str:
    gray = image.convert("L").resize((hash_size, hash_size))
    values = np.asarray(gray, dtype=np.float32)
    mean = values.mean()
    bits = (values > mean).flatten()
    return "".join("1" if bit else "0" for bit in bits)


def _hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right))


def _entropy(gray: Image.Image) -> float:
    histogram = np.asarray(gray.histogram(), dtype=np.float64)
    total = histogram.sum()
    if total <= 0:
        return 0.0
    probabilities = histogram[histogram > 0] / total
    return float(-(probabilities * np.log2(probabilities)).sum() / 8.0)


def compute_image_stats(path: Path, analysis_size: int = 256) -> ImageStats:
    with Image.open(path) as handle:
        image = handle.convert("RGB")
    width, height = image.size
    resized = image.resize((analysis_size, analysis_size))
    gray = resized.convert("L")
    hsv = resized.convert("HSV")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=1.5))
    high_freq = ImageChops.difference(gray, blurred)

    luminance_stat = ImageStat.Stat(gray)
    saturation_stat = ImageStat.Stat(hsv.split()[1])
    edge_mean = ImageStat.Stat(edges).mean[0] / 255.0
    high_freq_std = ImageStat.Stat(high_freq).stddev[0] / 255.0

    return ImageStats(
        width=width,
        height=height,
        aspect_ratio=max(width, height) / max(min(width, height), 1),
        megapixels=(width * height) / 1_000_000.0,
        luminance_mean=luminance_stat.mean[0] / 255.0,
        luminance_std=luminance_stat.stddev[0] / 255.0,
        saturation_mean=saturation_stat.mean[0] / 255.0,
        edge_density=edge_mean,
        sharpness=high_freq_std,
        entropy=_entropy(gray),
        average_hash=_average_hash(resized),
    )


def _stats_to_scores(stats: ImageStats) -> dict[str, float]:
    resolution_score = _clamp((min(stats.width, stats.height) - 384) / 640.0)
    aspect_score = _score_range(stats.aspect_ratio, 1.0, 2.2)
    exposure_score = _score_range(stats.luminance_mean, 0.18, 0.82)
    contrast_score = _score_range(stats.luminance_std, 0.12, 0.38)
    saturation_score = _score_range(stats.saturation_mean, 0.08, 0.70)
    sharpness_score = _score_range(stats.sharpness, 0.018, 0.18)
    structure_score = _score_range(stats.edge_density, 0.03, 0.30)
    entropy_score = _score_range(stats.entropy, 0.45, 0.92)
    return {
        "resolution_score": resolution_score,
        "aspect_score": aspect_score,
        "exposure_score": exposure_score,
        "contrast_score": contrast_score,
        "saturation_score": saturation_score,
        "sharpness_score": sharpness_score,
        "structure_score": structure_score,
        "entropy_score": entropy_score,
        "technical_quality_score": (
            0.20 * resolution_score
            + 0.15 * aspect_score
            + 0.15 * exposure_score
            + 0.15 * contrast_score
            + 0.15 * sharpness_score
            + 0.10 * structure_score
            + 0.10 * entropy_score
        ),
    }


class HeuristicVLMScorer:
    def score(self, image_path: Path, stats: ImageStats) -> VLMJudgment:
        scores = _stats_to_scores(stats)
        families = ["exposure", "contrast"]
        if stats.saturation_mean > 0.08:
            families.extend(["color", "tone"])
        if stats.edge_density > 0.045:
            families.extend(["style", "local"])
        if stats.entropy > 0.58 and stats.edge_density > 0.065:
            families.extend(["object", "background"])
        families = sorted(set(families))

        clutter_penalty = _clamp((stats.edge_density - 0.25) / 0.20)
        text_watermark_penalty = 0.0
        if stats.edge_density > 0.32 and stats.entropy > 0.82:
            text_watermark_penalty = 0.35

        reject_reasons: list[str] = []
        if scores["technical_quality_score"] < 0.40:
            reject_reasons.append("low_technical_quality")
        if not families:
            reject_reasons.append("no_clear_edit_family")
        if text_watermark_penalty > 0.25:
            reject_reasons.append("possible_dense_text_or_watermark")

        return VLMJudgment(
            caption=f"Unlabeled image with {stats.width}x{stats.height} resolution.",
            quality_score=scores["technical_quality_score"],
            natural_image_score=0.75,
            editable_content_score=_clamp(0.25 + 0.12 * len(families)),
            object_region_clarity=_clamp(0.55 * scores["structure_score"] + 0.45 * scores["entropy_score"]),
            preservation_potential=_clamp(0.55 * scores["sharpness_score"] + 0.45 * scores["structure_score"] - 0.25 * clutter_penalty),
            clutter_penalty=clutter_penalty,
            text_watermark_penalty=text_watermark_penalty,
            edit_families=families,
            reject_reasons=reject_reasons,
        )


class QwenOpenVLMScorer:
    def __init__(self, config: dict[str, Any]):
        self.model_id = config.get("model_id", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.device = config.get("device", "auto")
        self.torch_dtype = config.get("torch_dtype", "auto")
        self.max_new_tokens = int(config.get("max_new_tokens", 512))
        self.temperature = float(config.get("temperature", 0.0))
        self.model = None
        self.processor = None

    def _ensure_model(self):
        if self.model is not None and self.processor is not None:
            return self.model, self.processor
        import torch
        from transformers import AutoProcessor

        dtype = "auto"
        if self.torch_dtype == "float16":
            dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            dtype = torch.bfloat16

        device_map = "auto" if self.device == "auto" else {"": self.device}
        load_kwargs = {"device_map": device_map}
        if dtype != "auto":
            load_kwargs["torch_dtype"] = dtype
        else:
            load_kwargs["torch_dtype"] = "auto"

        model_id_lower = self.model_id.lower()
        if "qwen3-vl" in model_id_lower:
            try:
                from transformers import Qwen3VLForConditionalGeneration

                self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_id, **load_kwargs)
            except ImportError as exc:
                raise ImportError(
                    "Qwen3-VL requires a recent transformers build with Qwen3VLForConditionalGeneration. "
                    "Upgrade with: pip install --upgrade transformers accelerate qwen-vl-utils"
                ) from exc
        else:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration

                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, **load_kwargs)
            except ImportError:
                from transformers import AutoModelForImageTextToText

                self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        return self.model, self.processor

    @staticmethod
    def _prompt() -> str:
        families = ", ".join(sorted(EDIT_FAMILIES))
        return (
            "You are selecting source images for instruction-guided image editing self-training. "
            "Judge whether this image is useful for generating verifiable edits. Return only JSON with: "
            "caption, quality_score, natural_image_score, editable_content_score, object_region_clarity, "
            "preservation_potential, clutter_penalty, text_watermark_penalty, edit_families, reject_reasons. "
            "All scores must be numbers from 0 to 1. edit_families must use only these labels: "
            f"{families}. Prefer images with clear editable content and enough unchanged structure to preserve."
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"VLM response did not contain JSON: {text[:200]}")
        return json.loads(match.group(0))

    def score(self, image_path: Path, stats: ImageStats) -> VLMJudgment:
        model, processor = self._ensure_model()
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._prompt()},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        inputs = inputs.to(model.device)
        generate_kwargs = {"max_new_tokens": self.max_new_tokens}
        if self.temperature > 0.0:
            generate_kwargs.update({"do_sample": True, "temperature": self.temperature})
        generated_ids = model.generate(**inputs, **generate_kwargs)
        trimmed = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
        response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        data = self._parse_json(response)
        edit_families = [str(item) for item in data.get("edit_families", []) if str(item) in EDIT_FAMILIES]
        return VLMJudgment(
            caption=data.get("caption"),
            quality_score=_clamp(float(data.get("quality_score", 0.0))),
            natural_image_score=_clamp(float(data.get("natural_image_score", 0.0))),
            editable_content_score=_clamp(float(data.get("editable_content_score", 0.0))),
            object_region_clarity=_clamp(float(data.get("object_region_clarity", 0.0))),
            preservation_potential=_clamp(float(data.get("preservation_potential", 0.0))),
            clutter_penalty=_clamp(float(data.get("clutter_penalty", 0.0))),
            text_watermark_penalty=_clamp(float(data.get("text_watermark_penalty", 0.0))),
            edit_families=edit_families,
            reject_reasons=[str(item) for item in data.get("reject_reasons", [])],
            raw_response=response,
        )


def build_vlm_scorer(config: dict[str, Any]):
    backend = config.get("backend", "heuristic")
    if backend == "heuristic":
        return HeuristicVLMScorer()
    if backend in {"qwen2_5_vl", "qwen_vl"}:
        return QwenOpenVLMScorer(config)
    raise ValueError(f"Unsupported VLM scoring backend: {backend}")


def discover_images(config: dict[str, Any], limit: int | None = None) -> list[Path]:
    source = config.get("source", "directory")
    if source == "directory":
        images_dir = resolve_path(config["images_dir"])
        if images_dir is None or not images_dir.exists():
            raise FileNotFoundError(f"images_dir not found: {config['images_dir']}")
        paths = sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        return paths[:limit] if limit is not None else paths
    if source == "jsonl":
        manifest_path = resolve_path(config["manifest_jsonl"])
        if manifest_path is None or not manifest_path.exists():
            raise FileNotFoundError(f"manifest_jsonl not found: {config['manifest_jsonl']}")
        paths = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                image_path = resolve_path(item["image"])
                if image_path is not None:
                    paths.append(image_path)
        return paths[:limit] if limit is not None else paths
    raise ValueError(f"Unsupported source selector input: {source}")


def image_key(image_path: Path, images_dir: Path | None) -> str:
    if images_dir is not None:
        try:
            return str(image_path.relative_to(images_dir).with_suffix("")).replace("/", "__")
        except ValueError:
            pass
    return image_path.stem


def combine_score(stats_scores: dict[str, float], vlm: VLMJudgment, weights: dict[str, float]) -> float:
    values = {
        "technical_quality": stats_scores["technical_quality_score"],
        "vlm_quality": vlm.quality_score,
        "natural_image": vlm.natural_image_score,
        "editable_content": vlm.editable_content_score,
        "object_region_clarity": vlm.object_region_clarity,
        "preservation_potential": vlm.preservation_potential,
        "edit_family_coverage": _clamp(len(vlm.edit_families) / 4.0),
        "clutter_penalty": vlm.clutter_penalty,
        "text_watermark_penalty": vlm.text_watermark_penalty,
    }
    positive = sum(values[name] * float(weight) for name, weight in weights.items() if not name.endswith("_penalty"))
    penalties = sum(values[name] * abs(float(weight)) for name, weight in weights.items() if name.endswith("_penalty"))
    normalizer = sum(abs(float(weight)) for weight in weights.values()) or 1.0
    return _clamp((positive - penalties) / normalizer)


def _asdict_stats(stats: ImageStats) -> dict[str, Any]:
    return {
        "width": stats.width,
        "height": stats.height,
        "aspect_ratio": stats.aspect_ratio,
        "megapixels": stats.megapixels,
        "luminance_mean": stats.luminance_mean,
        "luminance_std": stats.luminance_std,
        "saturation_mean": stats.saturation_mean,
        "edge_density": stats.edge_density,
        "sharpness": stats.sharpness,
        "entropy": stats.entropy,
        "average_hash": stats.average_hash,
    }


def _asdict_vlm(vlm: VLMJudgment) -> dict[str, Any]:
    return {
        "caption": vlm.caption,
        "quality_score": vlm.quality_score,
        "natural_image_score": vlm.natural_image_score,
        "editable_content_score": vlm.editable_content_score,
        "object_region_clarity": vlm.object_region_clarity,
        "preservation_potential": vlm.preservation_potential,
        "clutter_penalty": vlm.clutter_penalty,
        "text_watermark_penalty": vlm.text_watermark_penalty,
        "edit_families": vlm.edit_families,
        "reject_reasons": vlm.reject_reasons,
        "raw_response": vlm.raw_response,
    }


def write_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")


def _materialize_selected(selected: list[dict[str, Any]], output_cfg: dict[str, Any]) -> None:
    mode = output_cfg.get("materialize_mode", "none")
    if mode == "none":
        return
    out_dir = resolve_path(output_cfg.get("selected_images_dir"))
    if out_dir is None:
        raise ValueError("output.selected_images_dir is required when materialize_mode is copy or symlink")
    ensure_dir(out_dir)
    for item in selected:
        src = resolve_path(item["image"])
        if src is None:
            continue
        suffix = src.suffix.lower()
        dst = out_dir / f"{item['key']}{suffix}"
        if dst.exists():
            continue
        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "symlink":
            dst.symlink_to(src.resolve())
        else:
            raise ValueError(f"Unsupported materialize_mode: {mode}")


def select_images(config: dict[str, Any], limit: int | None = None) -> dict[str, Any]:
    input_cfg = config["input"]
    images_dir = resolve_path(input_cfg.get("images_dir"))
    image_paths = discover_images(input_cfg, limit=limit)
    scorer = build_vlm_scorer(config.get("vlm", {}))
    selection_cfg = config.get("selection", {})
    thresholds = selection_cfg.get("thresholds", {})
    weights = selection_cfg.get(
        "weights",
        {
            "technical_quality": 0.20,
            "vlm_quality": 0.15,
            "natural_image": 0.10,
            "editable_content": 0.20,
            "object_region_clarity": 0.15,
            "preservation_potential": 0.15,
            "edit_family_coverage": 0.10,
            "clutter_penalty": 0.08,
            "text_watermark_penalty": 0.12,
        },
    )

    scored: list[dict[str, Any]] = []
    for image_path in image_paths:
        stats = compute_image_stats(image_path)
        stats_scores = _stats_to_scores(stats)
        vlm = scorer.score(image_path, stats)
        score = combine_score(stats_scores, vlm, weights)
        reject_reasons = list(vlm.reject_reasons)
        if min(stats.width, stats.height) < int(thresholds.get("min_short_side", 384)):
            reject_reasons.append("short_side_too_small")
        if stats.aspect_ratio > float(thresholds.get("max_aspect_ratio", 2.5)):
            reject_reasons.append("extreme_aspect_ratio")
        if stats_scores["technical_quality_score"] < float(thresholds.get("min_technical_quality", 0.35)):
            reject_reasons.append("low_technical_quality")
        if vlm.editable_content_score < float(thresholds.get("min_editable_content", 0.35)):
            reject_reasons.append("low_editable_content")
        if vlm.preservation_potential < float(thresholds.get("min_preservation_potential", 0.30)):
            reject_reasons.append("low_preservation_potential")
        if len(vlm.edit_families) < int(thresholds.get("min_edit_families", 1)):
            reject_reasons.append("insufficient_edit_family_coverage")
        if score < float(thresholds.get("min_total_score", 0.45)):
            reject_reasons.append("low_total_score")
        scored.append(
            {
                "key": image_key(image_path, images_dir),
                "image": relative_to_repo(image_path),
                "caption": vlm.caption,
                "score": score,
                "accepted": not reject_reasons,
                "reject_reasons": sorted(set(reject_reasons)),
                "edit_families": vlm.edit_families,
                "primary_family": vlm.edit_families[0] if vlm.edit_families else "unknown",
                "stats": _asdict_stats(stats),
                "stats_scores": stats_scores,
                "vlm": _asdict_vlm(vlm),
            }
        )

    candidates = sorted((item for item in scored if item["accepted"]), key=lambda item: item["score"], reverse=True)
    rejected = [item for item in scored if not item["accepted"]]
    max_selected = int(selection_cfg.get("max_selected", 0) or 0)
    min_hash_distance = int(selection_cfg.get("min_average_hash_distance", 6))
    max_family_fraction = float(selection_cfg.get("max_family_fraction", 0.45))

    selected: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    selected_hashes: list[str] = []
    for item in candidates:
        if max_selected and len(selected) >= max_selected:
            rejected.append({**item, "accepted": False, "reject_reasons": ["selection_budget_exceeded"]})
            continue
        if any(_hamming(item["stats"]["average_hash"], prior) < min_hash_distance for prior in selected_hashes):
            rejected.append({**item, "accepted": False, "reject_reasons": ["near_duplicate"]})
            continue
        family = item["primary_family"]
        family_limit = max(1, math.ceil(max_family_fraction * max(max_selected or len(candidates), 1)))
        if family_counts.get(family, 0) >= family_limit:
            rejected.append({**item, "accepted": False, "reject_reasons": ["family_quota_exceeded"]})
            continue
        selected.append(item)
        selected_hashes.append(item["stats"]["average_hash"])
        family_counts[family] = family_counts.get(family, 0) + 1

    output_cfg = config["output"]
    selected_manifest = resolve_path(output_cfg["selected_manifest_jsonl"])
    rejected_manifest = resolve_path(output_cfg["rejected_manifest_jsonl"])
    score_jsonl = resolve_path(output_cfg["score_jsonl"])
    summary_path = resolve_path(output_cfg["summary_json"])
    if selected_manifest is None or rejected_manifest is None or score_jsonl is None or summary_path is None:
        raise ValueError("All output paths must be configured")

    manifest_records = [
        {
            "key": item["key"],
            "image": item["image"],
            "caption": item["caption"],
            "metadata": {
                "source_selection_score": item["score"],
                "edit_families": item["edit_families"],
                "primary_family": item["primary_family"],
            },
        }
        for item in selected
    ]
    write_jsonl(manifest_records, selected_manifest)
    write_jsonl(rejected, rejected_manifest)
    write_jsonl(scored, score_jsonl)
    _materialize_selected(manifest_records, output_cfg)

    summary = {
        "config_path": config.get("_config_path"),
        "images_seen": len(scored),
        "selected": len(selected),
        "rejected": len(rejected),
        "selected_manifest_jsonl": str(selected_manifest),
        "rejected_manifest_jsonl": str(rejected_manifest),
        "score_jsonl": str(score_jsonl),
        "family_counts": family_counts,
        "thresholds": thresholds,
        "weights": weights,
    }
    save_json(summary, summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Select high-editability source images for self-evolving image editing.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--set", action="append", default=[], help="Override config using dotted.key=value")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    summary = select_images(config, limit=args.limit)
    print(f"Selected {summary['selected']} / {summary['images_seen']} images.")
    print(f"Manifest: {summary['selected_manifest_jsonl']}")


if __name__ == "__main__":
    main()
