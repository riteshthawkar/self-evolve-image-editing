from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from qwen_edit_project.utils.config import save_json
from qwen_edit_project.utils.paths import relative_to_repo, resolve_path


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "records" in data and isinstance(data["records"], list):
                return data["records"]
            return list(data.values())
        raise TypeError("JSON input must be a list or object")
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError(f"Unsupported input format: {path}")


def normalize_edit_image(value: Any, separator: str) -> str | list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and separator in value:
        return [chunk.strip() for chunk in value.split(separator) if chunk.strip()]
    return str(value)


def map_record(
    record: dict[str, Any],
    prompt_field: str,
    image_field: str,
    edit_image_field: str,
    separator: str,
) -> dict[str, Any]:
    prompt = str(record[prompt_field]).strip()
    image = record[image_field]
    edit_image = normalize_edit_image(record[edit_image_field], separator)
    return {
        "prompt": prompt,
        "image": image,
        "edit_image": edit_image,
    }


def relativize_manifest_paths(records: list[dict[str, Any]], base_dir: Path) -> list[dict[str, Any]]:
    relativized: list[dict[str, Any]] = []
    for record in records:
        out = dict(record)
        image_path = Path(record["image"])
        if image_path.is_absolute():
            out["image"] = relative_to_repo(image_path)
        else:
            out["image"] = relative_to_repo((base_dir / image_path).resolve())
        edit_image = record["edit_image"]
        if isinstance(edit_image, list):
            out["edit_image"] = [
                relative_to_repo((base_dir / Path(item)).resolve()) if not Path(item).is_absolute() else relative_to_repo(Path(item))
                for item in edit_image
            ]
        else:
            edit_path = Path(edit_image)
            out["edit_image"] = (
                relative_to_repo((base_dir / edit_path).resolve())
                if not edit_path.is_absolute()
                else relative_to_repo(edit_path)
            )
        relativized.append(out)
    return relativized


def build_manifest(args: argparse.Namespace) -> Path:
    input_path = resolve_path(args.input)
    if input_path is None or not input_path.exists():
        raise FileNotFoundError(f"Input records file not found: {args.input}")
    base_dir = resolve_path(args.base_dir) if args.base_dir else input_path.parent
    if base_dir is None:
        raise ValueError("Base directory could not be resolved")
    records = load_records(input_path)
    manifest = [
        map_record(record, args.prompt_field, args.image_field, args.edit_image_field, args.edit_image_separator)
        for record in records
    ]
    if args.relativize_paths:
        manifest = relativize_manifest_paths(manifest, base_dir)
    return save_json(manifest, args.output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a DiffSynth-compatible manifest.")
    parser.add_argument("--input", required=True, help="Input JSON, JSONL, or CSV file.")
    parser.add_argument("--output", required=True, help="Output manifest JSON path.")
    parser.add_argument("--base-dir", default=None, help="Base directory used to resolve relative paths.")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--image-field", default="image")
    parser.add_argument("--edit-image-field", default="edit_image")
    parser.add_argument("--edit-image-separator", default="|")
    parser.add_argument(
        "--relativize-paths",
        action="store_true",
        help="Write paths relative to the repo root instead of preserving the source values.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output_path = build_manifest(args)
    print(f"Wrote manifest to {output_path}")


if __name__ == "__main__":
    main()

