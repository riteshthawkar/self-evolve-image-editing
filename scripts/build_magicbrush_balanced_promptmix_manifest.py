from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_magicbrush_object_manifest import (
    _classify_instruction_with_reason,
    _csv_set,
    _load_jsonl,
    _quality_ok,
)


BACKGROUND_TERMS = {
    "background",
    "backdrop",
    "environment",
    "scene",
    "setting",
    "sky",
    "beach",
    "forest",
    "mountain",
    "landscape",
    "room",
    "wall",
    "floor",
    "field",
    "street",
    "road",
}

STYLE_TERMS = {
    "style",
    "painting",
    "sketch",
    "drawing",
    "cartoon",
    "anime",
    "watercolor",
    "watercolour",
    "oil painting",
    "mosaic",
    "blueprint",
    "woodblock",
    "ukiyo",
    "steampunk",
    "pixel art",
    "comic",
    "illustration",
}

ADJUST_TERMS = {
    "color",
    "colour",
    "texture",
    "material",
    "bright",
    "brightness",
    "darker",
    "lighter",
    "saturation",
    "contrast",
    "blur",
    "sharp",
    "size",
    "larger",
    "smaller",
    "transparent",
    "opacity",
}

EXTRACT_TERMS = {
    "extract",
    "isolate",
    "cut out",
    "separate",
    "remove the background",
    "transparent background",
}

ACTION_TERMS = {
    "raise",
    "lift",
    "lower",
    "turn",
    "look",
    "stand",
    "sit",
    "open",
    "close",
    "hold",
    "wear",
    "smile",
    "pose",
}


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _normalized_instruction(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def classify_balanced_instruction(text: str) -> tuple[str | None, str]:
    normalized = _normalized_instruction(text)
    padded = f" {normalized} "

    if _contains_any(normalized, EXTRACT_TERMS):
        return "extract", "classified_extract"

    if _contains_any(normalized, STYLE_TERMS):
        return "style", "classified_style"

    if _contains_any(normalized, BACKGROUND_TERMS) and re.search(
        r"\b(change|replace|turn|convert|transform|blur|remove)\b", normalized
    ):
        return "background", "classified_background"

    object_type, object_reason = _classify_instruction_with_reason(text, clean_object_only=True)
    if object_type is not None:
        return object_type, object_reason

    if _contains_any(normalized, ADJUST_TERMS) and re.search(
        r"\b(change|make|turn|adjust|increase|decrease|blur|sharpen)\b", normalized
    ):
        return "adjust", "classified_adjust"

    if (
        re.search(r"\b(make|have|let)\b", normalized)
        and _contains_any(padded, {f" {term} " for term in ACTION_TERMS})
    ) or re.search(r"\b(raise|lift|lower|open|close|wear|hold|smile)\b", normalized):
        return "action", "classified_action"

    object_type, object_reason = _classify_instruction_with_reason(text, clean_object_only=False)
    if object_type is not None:
        return object_type, f"fallback_{object_reason}"

    return None, "unclassified"


def contract_prompt(instruction: str, edit_type: str) -> str:
    instruction = re.sub(r"\s+", " ", instruction).strip()
    contracts = {
        "object_removal": (
            "Completely remove the requested object or objects; no visible part of the removed object should remain. "
            "Fill the region naturally and keep unrelated content, layout, lighting, and viewpoint unchanged."
        ),
        "object_replacement": (
            "Fully replace the requested source object with the target object; no visible part of the original object "
            "should remain. Keep location, scale, lighting, viewpoint, and unrelated content unchanged."
        ),
        "object_addition": (
            "Add the requested object naturally at the requested location. Keep all existing unrelated content, layout, "
            "lighting, and viewpoint unchanged."
        ),
        "background": (
            "Change only the requested background or scene elements. Preserve the foreground subjects, their identity, "
            "layout, lighting consistency, and viewpoint."
        ),
        "style": (
            "Apply the requested visual style while preserving the objects, composition, identities, layout, and readable text."
        ),
        "adjust": (
            "Apply only the requested attribute change. Preserve unrelated objects, composition, lighting consistency, and text."
        ),
        "action": (
            "Apply the requested pose or action change while preserving identity, scene layout, lighting, and unrelated content."
        ),
        "extract": (
            "Isolate the requested subject cleanly while preserving its shape, texture, and visible details."
        ),
    }
    contract = contracts.get(edit_type, "Keep unrelated content, layout, lighting, viewpoint, and text unchanged.")
    return f"{instruction}{'' if instruction.endswith('.') else '.'} {contract}"


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    include_edit_types = _csv_set(args.include_edit_types)
    prompt_variants = [item.strip() for item in str(args.prompt_variants).split(",") if item.strip()]
    if not prompt_variants:
        prompt_variants = ["plain", "contract"]
    invalid_variants = sorted(set(prompt_variants) - {"plain", "contract"})
    if invalid_variants:
        raise ValueError(f"Unsupported prompt variant(s): {invalid_variants}")

    records = _load_jsonl(args.input)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = Counter()
    reason_counts = Counter()
    for record in records:
        instruction = str(record.get("instruction", "")).strip()
        edit_type, reason = classify_balanced_instruction(instruction)
        reason_counts[reason] += 1
        if edit_type is None:
            rejected[reason] += 1
            continue
        if include_edit_types and edit_type not in include_edit_types:
            rejected["excluded_edit_type"] += 1
            continue
        if not _quality_ok(record, args.min_score, args.max_changed_fraction):
            rejected["quality_or_split_filtered"] += 1
            continue
        buckets[edit_type].append(record)

    selected: list[tuple[str, dict[str, Any]]] = []
    for edit_type, items in sorted(buckets.items()):
        rng.shuffle(items)
        selected.extend((edit_type, record) for record in items[: args.per_type_limit])
    rng.shuffle(selected)
    selected = selected[: args.max_records]

    weights = {}
    for raw in str(args.family_weights).split(","):
        if not raw.strip():
            continue
        family, value = raw.split("=", 1)
        weights[family.strip()] = float(value)

    manifest: list[dict[str, Any]] = []
    for edit_type, record in selected:
        plain_prompt = str(record["instruction"]).strip()
        for variant in prompt_variants:
            prompt = plain_prompt if variant == "plain" else contract_prompt(plain_prompt, edit_type)
            manifest.append(
                {
                    "prompt": prompt,
                    "image": record["target_image"],
                    "edit_image": record["source_image"],
                    "sample_weight": weights.get(edit_type, args.sample_weight),
                    "candidate_status": "supervised_balanced_pair",
                    "family": edit_type,
                    "prompt_variant": variant,
                    "structured_edit": {
                        "edit_type": edit_type,
                        "instruction": prompt,
                        "plain_instruction": plain_prompt,
                    },
                    "source": "magicbrush_train_balanced",
                    "record_key": record.get("key"),
                    "score": record.get("score"),
                }
            )

    replay_count = round(len(manifest) * args.replay_ratio)
    replay_sources = list(dict.fromkeys(record["source_image"] for _, record in selected))
    for index in range(replay_count):
        source_image = replay_sources[index % max(len(replay_sources), 1)]
        manifest.append(
            {
                "prompt": args.replay_prompt,
                "image": source_image,
                "edit_image": source_image,
                "sample_weight": args.replay_weight,
                "candidate_status": "reconstruction_replay",
                "family": "reconstruction_replay",
                "source": "magicbrush_train_balanced",
            }
        )

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(manifest),
        "selected_records": len(selected),
        "variant_rows": len(selected) * len(prompt_variants),
        "replay_rows": replay_count,
        "per_type": Counter(edit_type for edit_type, _ in selected),
        "available_per_type": {key: len(value) for key, value in sorted(buckets.items())},
        "rejected": rejected,
        "reason_counts": reason_counts,
        "include_edit_types": sorted(include_edit_types),
        "prompt_variants": prompt_variants,
        "sample_weight": args.sample_weight,
        "family_weights": weights,
        "replay_ratio": args.replay_ratio,
        "replay_weight": args.replay_weight,
        "min_score": args.min_score,
        "max_changed_fraction": args.max_changed_fraction,
        "seed": args.seed,
    }
    return manifest, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced MagicBrush prompt-mix LoRA manifest.")
    parser.add_argument("--input", type=Path, default=Path("data/edit_pairs/magicbrush_full/selected_records.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/magicbrush_balanced_promptmix_train.jsonl"))
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=896)
    parser.add_argument("--per-type-limit", type=int, default=128)
    parser.add_argument(
        "--include-edit-types",
        default="object_removal,object_replacement,object_addition,background,style,adjust,action,extract",
    )
    parser.add_argument("--prompt-variants", default="plain,contract")
    parser.add_argument("--sample-weight", type=float, default=0.70)
    parser.add_argument(
        "--family-weights",
        default="object_removal=0.80,object_replacement=0.75,object_addition=0.80,style=0.75,extract=0.75,background=0.65,adjust=0.65,action=0.65",
    )
    parser.add_argument("--min-score", type=float, default=0.78)
    parser.add_argument("--max-changed-fraction", type=float, default=0.60)
    parser.add_argument("--replay-ratio", type=float, default=0.75)
    parser.add_argument("--replay-weight", type=float, default=0.50)
    parser.add_argument(
        "--replay-prompt",
        default="Reconstruct the input image exactly. Preserve all content, layout, colors, and text.",
    )
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    manifest, summary = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in manifest:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    summary_path = args.summary or args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
