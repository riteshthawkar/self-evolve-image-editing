#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


def parse_candidate(raw: str) -> tuple[str, Path, Path]:
    parts = raw.split("=")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Candidates must be LABEL=IMAGE_DIR=AVERAGE_SCORE_JSON")
    label, image_dir, score_json = parts
    if not label:
        raise argparse.ArgumentTypeError("Candidate label cannot be empty")
    return label, Path(image_dir), Path(score_json)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ImgEdit folder by choosing the highest scored candidate per key.")
    parser.add_argument("--edit-json", default="data/processed/benchmark/imgedit/basic_edit.json")
    parser.add_argument("--image-root", default="outputs/benchmark_images/imgedit")
    parser.add_argument("--scores-root", default="outputs/scores/imgedit")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--tie-label", default=None)
    args = parser.parse_args()

    edit_specs = load_json(Path(args.edit_json))
    candidates = []
    labels = set()
    for label, image_dir, score_json in args.candidate:
        if label in labels:
            raise ValueError(f"Duplicate label: {label}")
        labels.add(label)
        if not image_dir.exists():
            raise FileNotFoundError(image_dir)
        if not score_json.exists():
            raise FileNotFoundError(score_json)
        scores = load_json(score_json)
        candidates.append((label, image_dir, score_json, scores))
    tie_label = args.tie_label or candidates[0][0]
    label_order = {label: idx for idx, (label, _, _, _) in enumerate(candidates)}
    if tie_label not in label_order:
        raise ValueError(f"tie label {tie_label!r} is not a candidate")

    output_dir = Path(args.image_root) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {label: 0 for label in label_order}
    records = []
    for key, item in edit_specs.items():
        key = str(key)
        ranked = []
        for label, image_dir, score_json, scores in candidates:
            if key not in scores:
                raise KeyError(f"{key} missing from {score_json}")
            try:
                score = float(scores[key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-numeric score for {key} in {score_json}: {scores[key]!r}") from exc
            tie_bonus = 1 if label == tie_label else 0
            ranked.append((score, tie_bonus, -label_order[label], label, image_dir))
        score, _, _, label, image_dir = max(ranked)
        src = image_dir / f"{key}.png"
        if not src.exists():
            raise FileNotFoundError(src)
        dst = output_dir / f"{key}.png"
        link_or_copy(src, dst)
        counts[label] += 1
        records.append(
            {
                "key": key,
                "edit_type": item.get("edit_type"),
                "choice": label,
                "selection_score": score,
                "source_path": str(src),
                "target_path": str(dst),
            }
        )

    manifest = {
        "model_name": args.model_name,
        "edit_json": args.edit_json,
        "candidates": [
            {"label": label, "image_dir": str(image_dir), "score_json": str(score_json)}
            for label, image_dir, score_json, _ in candidates
        ],
        "tie_label": tie_label,
        "counts": counts,
        "records": records,
    }
    manifest_path = Path(args.scores_root) / f"{args.model_name}_score_selector_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Built {len(records)} selected image(s) in {output_dir}", flush=True)
    print(f"Choice counts: {counts}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
