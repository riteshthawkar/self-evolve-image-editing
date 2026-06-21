from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


OBJECT_FAMILIES = {"object_addition", "object_removal", "object_replacement"}


def _csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _existing_path(path: str) -> bool:
    return bool(path) and Path(path).exists()


def _score_margin(record: dict[str, Any]) -> float:
    return float(record.get("winner_score") or 0.0) - float(record.get("loser_score") or 0.0)


def _sample_balanced(
    records: list[dict[str, Any]],
    *,
    rng: random.Random,
    max_rows: int,
    per_family_limit: int,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(str(record.get("family", "")), []).append(record)
    selected: list[dict[str, Any]] = []
    for family, family_records in sorted(buckets.items()):
        family_records.sort(
            key=lambda item: (
                float(item.get("winner_score") or 0.0),
                _score_margin(item),
            ),
            reverse=True,
        )
        family_cap = min(per_family_limit, len(family_records))
        selected.extend(family_records[:family_cap])
    rng.shuffle(selected)
    return selected[:max_rows]


def build_generated_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter[str]]:
    rng = random.Random(args.seed)
    include_families = _csv_set(args.generated_families)
    generated: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    seen: set[tuple[str, str, str, str]] = set()

    for pattern in args.generated_preferences_glob:
        for path in sorted(Path().glob(pattern)):
            for record in _load_jsonl(path):
                family = str(record.get("family", ""))
                if include_families and family not in include_families:
                    rejected["excluded_family"] += 1
                    continue
                winner_score = float(record.get("winner_score") or 0.0)
                loser_score = float(record.get("loser_score") or 0.0)
                margin = winner_score - loser_score
                if winner_score < args.min_winner_score:
                    rejected["winner_score_below_threshold"] += 1
                    continue
                if margin < args.min_margin:
                    rejected["margin_below_threshold"] += 1
                    continue
                if not _existing_path(str(record.get("winner_image", ""))):
                    rejected["missing_winner_image"] += 1
                    continue
                if not _existing_path(str(record.get("loser_image", ""))):
                    rejected["missing_loser_image"] += 1
                    continue
                if not _existing_path(str(record.get("source_image", ""))):
                    rejected["missing_source_image"] += 1
                    continue
                instruction = str(record.get("instruction", "")).strip()
                if not instruction:
                    rejected["missing_instruction"] += 1
                    continue
                key = (
                    str(record.get("source_image", "")),
                    instruction,
                    str(record.get("winner_image", "")),
                    str(record.get("loser_image", "")),
                )
                if key in seen:
                    rejected["duplicate_pair"] += 1
                    continue
                seen.add(key)
                generated.append(record)

    selected = _sample_balanced(
        generated,
        rng=rng,
        max_rows=args.generated_max_rows,
        per_family_limit=args.generated_per_family_limit,
    )
    rows: list[dict[str, Any]] = []
    for record in selected:
        margin = _score_margin(record)
        sample_weight = min(
            args.generated_max_weight,
            args.generated_weight * (1.0 + args.margin_weight_scale * min(margin, args.margin_weight_clip)),
        )
        rows.append(
            {
                "prompt": str(record["instruction"]).strip(),
                "chosen_image": record["winner_image"],
                "rejected_image": record["loser_image"],
                "edit_image": record["source_image"],
                "sample_weight": sample_weight,
                "family": record.get("family"),
                "prompt_variant": "self_evolve",
                "preference_source": "self_evolve_generated_winner_over_loser",
                "record_key": record.get("record_key"),
                "group_id": record.get("group_id"),
                "operation_id": record.get("operation_id"),
                "structured_edit": record.get("structured_edit"),
                "winner_score": record.get("winner_score"),
                "loser_score": record.get("loser_score"),
                "score_margin": margin,
            }
        )

    if args.generated_repeat > 1 and rows:
        rows = [dict(row) for _ in range(args.generated_repeat) for row in rows]
    rng.shuffle(rows)
    return rows, rejected


def build_manifest(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    target_rows = _load_jsonl(args.target_source_input) if args.target_source_input else []
    rng.shuffle(target_rows)
    if args.target_source_rows >= 0:
        target_rows = target_rows[: args.target_source_rows]
    for row in target_rows:
        row["preference_source"] = row.get("preference_source", "magicbrush_target_over_source")
        row["sample_weight"] = float(row.get("sample_weight", 1.0)) * args.target_source_weight

    generated_rows, generated_rejected = build_generated_rows(args)
    manifest = target_rows + generated_rows
    rng.shuffle(manifest)

    summary: dict[str, Any] = {
        "output": str(args.output),
        "rows": len(manifest),
        "target_source_rows": len(target_rows),
        "generated_rows": len(generated_rows),
        "generated_repeat": args.generated_repeat,
        "per_family": Counter(str(row.get("family", "")) for row in manifest),
        "per_source": Counter(str(row.get("preference_source", "")) for row in manifest),
        "generated_rejected": generated_rejected,
        "min_winner_score": args.min_winner_score,
        "min_margin": args.min_margin,
        "target_source_weight": args.target_source_weight,
        "generated_weight": args.generated_weight,
        "seed": args.seed,
    }
    return manifest, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build pairwise preference manifest mixing target/source and generated winner/loser pairs."
    )
    parser.add_argument(
        "--target-source-input",
        type=Path,
        default=Path("data/manifests/magicbrush_object_preference_target_over_source_512.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--target-source-rows", type=int, default=512)
    parser.add_argument("--target-source-weight", type=float, default=1.0)
    parser.add_argument(
        "--generated-preferences-glob",
        action="append",
        default=["outputs/self_evolve/**/evaluator_preferences.jsonl"],
    )
    parser.add_argument("--generated-families", default="object_removal,object_replacement,object_addition")
    parser.add_argument("--generated-max-rows", type=int, default=256)
    parser.add_argument("--generated-per-family-limit", type=int, default=96)
    parser.add_argument("--generated-repeat", type=int, default=2)
    parser.add_argument("--generated-weight", type=float, default=1.25)
    parser.add_argument("--generated-max-weight", type=float, default=1.75)
    parser.add_argument("--margin-weight-scale", type=float, default=1.0)
    parser.add_argument("--margin-weight-clip", type=float, default=0.2)
    parser.add_argument("--min-winner-score", type=float, default=0.5)
    parser.add_argument("--min-margin", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    manifest, summary = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary_path = args.summary or args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
