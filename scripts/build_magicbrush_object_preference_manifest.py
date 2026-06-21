from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_magicbrush_object_manifest import (
    _classify_instruction_with_reason,
    _csv_set,
    _load_jsonl,
    _quality_ok,
    _strict_object_prompt,
)


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    include_edit_types = _csv_set(args.include_edit_types)
    prompt_variants = [item.strip() for item in args.prompt_variants.split(",") if item.strip()]
    if not prompt_variants:
        prompt_variants = ["plain", "strict"]
    invalid = sorted(set(prompt_variants) - {"plain", "strict"})
    if invalid:
        raise ValueError(f"Unsupported prompt variants: {invalid}")

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected = Counter()
    for record in _load_jsonl(args.input):
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
    for edit_type, records in sorted(buckets.items()):
        rng.shuffle(records)
        selected.extend((edit_type, record) for record in records[: args.per_type_limit])
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
                    "chosen_image": record["target_image"],
                    "rejected_image": record["source_image"],
                    "edit_image": record["source_image"],
                    "sample_weight": args.preference_weight,
                    "family": edit_type,
                    "prompt_variant": variant,
                    "preference_source": "magicbrush_target_over_source",
                    "structured_edit": {
                        "edit_type": edit_type,
                        "instruction": prompt,
                        "plain_instruction": plain_prompt,
                    },
                    "record_key": record.get("key"),
                    "score": record.get("score"),
                }
            )

    summary: dict[str, Any] = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(manifest),
        "selected_records": len(selected),
        "variant_rows": len(manifest),
        "per_type": Counter(edit_type for edit_type, _ in selected),
        "available_per_type": {key: len(value) for key, value in sorted(buckets.items())},
        "rejected": rejected,
        "include_edit_types": sorted(include_edit_types),
        "prompt_variants": prompt_variants,
        "preference_weight": args.preference_weight,
        "min_score": args.min_score,
        "max_changed_fraction": args.max_changed_fraction,
        "seed": args.seed,
    }
    return manifest, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MagicBrush object target-over-source preference pairs.")
    parser.add_argument("--input", type=Path, default=Path("data/edit_pairs/magicbrush_full/selected_records.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/magicbrush_object_preference_target_over_source_512.jsonl"),
    )
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=512)
    parser.add_argument("--per-type-limit", type=int, default=192)
    parser.add_argument(
        "--include-edit-types",
        default="object_removal,object_replacement,object_addition",
    )
    parser.add_argument("--clean-object-only", action="store_true")
    parser.add_argument("--prompt-variants", default="plain,strict")
    parser.add_argument("--preference-weight", type=float, default=1.0)
    parser.add_argument("--min-score", type=float, default=0.78)
    parser.add_argument("--max-changed-fraction", type=float, default=0.55)
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
