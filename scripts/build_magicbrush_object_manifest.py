from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
}

REPLACEMENT_OBJECT_NOUN_HINTS = {
    "airplane",
    "animal",
    "apple",
    "bag",
    "basket",
    "bat",
    "bear",
    "bed",
    "bench",
    "bird",
    "bottle",
    "bowl",
    "box",
    "building",
    "bus",
    "cabinet",
    "cake",
    "car",
    "cat",
    "chair",
    "clock",
    "cup",
    "dog",
    "door",
    "flower",
    "food",
    "frisbee",
    "fungus",
    "hat",
    "horse",
    "knife",
    "lamp",
    "laptop",
    "mug",
    "mushroom",
    "phone",
    "pizza",
    "plant",
    "plate",
    "sauce",
    "refrigerator",
    "shelf",
    "shirt",
    "sign",
    "sink",
    "sunflower",
    "table",
    "toilet",
    "train",
    "truck",
    "tv",
    "umbrella",
    "vase",
    "window",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _csv_set(value: str | None) -> set[str]:
    if value is None:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _classify_instruction(text: str) -> str | None:
    edit_type, _ = _classify_instruction_with_reason(text, clean_object_only=False)
    return edit_type


def _replacement_target_text(normalized: str) -> str:
    patterns = [
        r"\breplace\b.+?\bwith\b\s+(.+)$",
        r"\b(?:turn|change|convert|transform)\b.+?\binto\b\s+(.+)$",
        r"\bchange\b.+?\bto\s+(.+)$",
        r"\b(?:add|insert|place|put)\b\s+(.+?)(?:\s+\bin\b|\s+\bon\b|\s+\bat\b|$)",
        r"^(.+?)\s+\binstead of\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return match.group(1)
    return normalized


def _replacement_target_has_object_hint(normalized: str) -> bool:
    target = _replacement_target_text(normalized)
    tokens = set(re.findall(r"[a-z]+", target))
    if not tokens:
        return False
    if "color" in tokens or "colour" in tokens:
        return False
    if tokens.issubset(COLOR_WORDS | {"change", "make", "the", "a", "an", "to", "into", "with", "one"}):
        return False
    return bool(tokens & REPLACEMENT_OBJECT_NOUN_HINTS) or len(tokens - COLOR_WORDS) >= 2


def _classify_instruction_with_reason(text: str, *, clean_object_only: bool) -> tuple[str | None, str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    lowered = f" {normalized} "
    has_remove = any(phrase in lowered for phrase in (" remove ", " delete ", " erase ", " get rid of "))
    has_add = any(phrase in lowered for phrase in (" add ", " insert ", " place ", " put "))
    has_replace_word = " replace " in lowered or " instead of " in lowered
    has_into_change = bool(
        re.search(r"\b(turn|change|convert|transform)\b.+\binto\b", normalized)
        or re.search(r"\bchange\b.+\bto (?:a|an|the)\b", normalized)
    )
    has_remove_then_insert = has_remove and bool(
        re.search(r"\b(remove|delete|erase|get rid of)\b.+\b(add|insert|place|put)\b", normalized)
    )
    has_explicit_replace_with = bool(re.search(r"\breplace\b.+\bwith\b", normalized))
    if clean_object_only and has_remove and has_add:
        return None, "multi_operation_remove_add"
    has_replacement = has_replace_word or has_into_change or (has_remove_then_insert and not clean_object_only) or has_explicit_replace_with

    if has_replacement:
        if clean_object_only and not (
            has_explicit_replace_with
            or has_into_change
            or " instead of " in lowered
            or has_remove_then_insert
        ):
            return None, "ambiguous_replacement_marker"
        if clean_object_only and "background" in lowered:
            return None, "background_replacement"
        if clean_object_only and not _replacement_target_has_object_hint(normalized):
            return None, "replacement_without_object_hint"
        return "object_replacement", "classified_replacement"

    if has_remove:
        if clean_object_only and any(term in lowered for term in (" background ", " sky ", " cloud ", " clouds ")):
            return None, "nonobject_removal"
        return "object_removal", "classified_removal"

    if has_add:
        return "object_addition", "classified_addition"

    return None, "not_object_instruction"


def _quality_ok(record: dict[str, Any], min_score: float, max_changed_fraction: float) -> bool:
    if not bool(record.get("accepted", False)):
        return False
    if str(record.get("split", "")) != "train":
        return False
    if float(record.get("score", 0.0)) < min_score:
        return False
    changed_fraction = (
        (record.get("metrics") or {}).get("diff") or {}
    ).get("changed_fraction")
    if changed_fraction is not None and float(changed_fraction) > max_changed_fraction:
        return False
    return True


def _strict_object_prompt(instruction: str, edit_type: str) -> str:
    instruction = re.sub(r"\s+", " ", instruction).strip()
    if edit_type == "object_removal":
        contract = (
            "Completely remove the requested object or objects; no visible part of the removed object should remain. "
            "Fill the removed area naturally and keep all unrelated content, layout, lighting, and viewpoint unchanged."
        )
    elif edit_type == "object_replacement":
        contract = (
            "Fully replace the requested source object with the requested target object; no visible part of the original "
            "source object should remain. Keep the location, scale, lighting, viewpoint, and unrelated content unchanged."
        )
    else:
        contract = "Keep all unrelated content, layout, lighting, and viewpoint unchanged."
    if instruction.endswith("."):
        return f"{instruction} {contract}"
    return f"{instruction}. {contract}"


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    records = _load_jsonl(args.input)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = Counter()
    include_edit_types = _csv_set(args.include_edit_types)
    prompt_variants = [item.strip() for item in str(args.prompt_variants).split(",") if item.strip()]
    if not prompt_variants:
        prompt_variants = ["strict"] if args.strict_object_contract else ["plain"]
    invalid_variants = sorted(set(prompt_variants) - {"plain", "strict"})
    if invalid_variants:
        raise ValueError(f"Unsupported prompt variant(s): {invalid_variants}")

    for record in records:
        instruction = str(record.get("instruction", "")).strip()
        edit_type, reason = _classify_instruction_with_reason(
            instruction,
            clean_object_only=bool(args.clean_object_only),
        )
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

    manifest: list[dict[str, Any]] = []
    for edit_type, record in selected:
        plain_prompt = str(record["instruction"]).strip()
        for variant in prompt_variants:
            prompt = _strict_object_prompt(plain_prompt, edit_type) if variant == "strict" else plain_prompt
            manifest.append(
                {
                    "prompt": prompt,
                    "image": record["target_image"],
                    "edit_image": record["source_image"],
                    "sample_weight": args.object_weight,
                    "candidate_status": "supervised_object_pair",
                    "family": edit_type,
                    "prompt_variant": variant,
                    "structured_edit": {
                        "edit_type": edit_type,
                        "instruction": prompt,
                        "plain_instruction": plain_prompt,
                    },
                    "source": "magicbrush_train",
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
                "source": "magicbrush_train",
            }
        )

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(manifest),
        "object_rows": len(selected),
        "object_variant_rows": len(selected) * len(prompt_variants),
        "replay_rows": replay_count,
        "per_type": Counter(edit_type for edit_type, _ in selected),
        "available_per_type": {key: len(value) for key, value in sorted(buckets.items())},
        "rejected": rejected,
        "include_edit_types": sorted(include_edit_types),
        "clean_object_only": bool(args.clean_object_only),
        "strict_object_contract": args.strict_object_contract,
        "prompt_variants": prompt_variants,
        "object_weight": args.object_weight,
        "min_score": args.min_score,
        "max_changed_fraction": args.max_changed_fraction,
        "seed": args.seed,
    }
    return manifest, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an object-edit-only MagicBrush training manifest.")
    parser.add_argument("--input", type=Path, default=Path("data/edit_pairs/magicbrush_full/selected_records.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/magicbrush_object_train_512_replay035.jsonl"))
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=512)
    parser.add_argument("--per-type-limit", type=int, default=256)
    parser.add_argument(
        "--include-edit-types",
        default="",
        help="Optional comma-separated edit types to include after instruction classification.",
    )
    parser.add_argument(
        "--clean-object-only",
        action="store_true",
        help="Use strict object-edit heuristics and reject ambiguous replacement/removal instructions.",
    )
    parser.add_argument("--strict-object-contract", action="store_true")
    parser.add_argument(
        "--prompt-variants",
        default="",
        help="Comma-separated prompt variants to emit for each object pair: plain, strict, or both.",
    )
    parser.add_argument("--object-weight", type=float, default=1.0)
    parser.add_argument("--min-score", type=float, default=0.78)
    parser.add_argument("--max-changed-fraction", type=float, default=0.55)
    parser.add_argument("--replay-ratio", type=float, default=0.35)
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
