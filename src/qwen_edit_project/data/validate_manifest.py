from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qwen_edit_project.utils.image_io import can_open_image, sample_items, save_contact_sheet
from qwen_edit_project.utils.paths import resolve_path


def load_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError("Manifest must be a list of objects")
    return data


def _normalize_to_paths(value: Any) -> list[Path]:
    if isinstance(value, list):
        return [resolve_path(str(item)) for item in value if resolve_path(str(item)) is not None]
    resolved = resolve_path(str(value))
    return [resolved] if resolved is not None else []


def validate_manifest(path: Path, preview_count: int = 5) -> tuple[int, list[str]]:
    records = load_manifest(path)
    errors: list[str] = []
    preview_candidates: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        prompt = str(record.get("prompt", "")).strip()
        if not prompt:
            errors.append(f"Record {index}: prompt is empty")
        image_paths = _normalize_to_paths(record.get("image"))
        if len(image_paths) != 1:
            errors.append(f"Record {index}: image must resolve to one path")
            continue
        image_path = image_paths[0]
        if not image_path.exists():
            errors.append(f"Record {index}: image does not exist: {image_path}")
        else:
            ok, reason = can_open_image(image_path)
            if not ok:
                errors.append(f"Record {index}: image unreadable: {image_path} ({reason})")
        edit_paths = _normalize_to_paths(record.get("edit_image"))
        if not edit_paths:
            errors.append(f"Record {index}: edit_image must resolve to one or more paths")
            continue
        for edit_path in edit_paths:
            if not edit_path.exists():
                errors.append(f"Record {index}: edit_image missing: {edit_path}")
                continue
            ok, reason = can_open_image(edit_path)
            if not ok:
                errors.append(f"Record {index}: edit_image unreadable: {edit_path} ({reason})")
        preview_candidates.append(record)

    print(f"Validated {len(records)} records from {path}")
    print(f"Errors: {len(errors)}")
    if preview_candidates:
        sampled = sample_items(preview_candidates, preview_count)
        preview_images = [resolve_path(str(item["image"])) for item in sampled]
        preview_labels = [str(item["prompt"]) for item in sampled]
        preview_images = [item for item in preview_images if item is not None]
        if preview_images:
            contact_sheet_path = resolve_path("outputs/logs/manifest_preview.jpg")
            if contact_sheet_path is not None:
                save_contact_sheet(preview_images, preview_labels[: len(preview_images)], contact_sheet_path)
                print(f"Saved preview contact sheet to {contact_sheet_path}")
    return len(records), errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a DiffSynth training manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--preview-count", type=int, default=5)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    manifest_path = resolve_path(args.manifest)
    if manifest_path is None or not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    _, errors = validate_manifest(manifest_path, preview_count=args.preview_count)
    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
