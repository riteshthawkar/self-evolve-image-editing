#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


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


def parse_source(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Sources must be LABEL=IMAGE_DIR")
    label, image_dir = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Source label cannot be empty")
    return label, Path(image_dir)


def parse_route(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Routes must be EDIT_TYPE=SOURCE_LABEL")
    edit_type, label = raw.split("=", 1)
    edit_type = edit_type.strip()
    label = label.strip()
    if not edit_type or not label:
        raise argparse.ArgumentTypeError("Routes must have non-empty edit type and source label")
    return edit_type, label


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ImgEdit folder by routing edit types to source folders.")
    parser.add_argument("--edit-json", default="data/processed/benchmark/imgedit/basic_edit.json")
    parser.add_argument("--image-root", default="outputs/benchmark_images/imgedit")
    parser.add_argument("--scores-root", default="outputs/scores/imgedit")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--source", action="append", type=parse_source, required=True)
    parser.add_argument("--default-label", required=True)
    parser.add_argument("--route", action="append", type=parse_route, default=[])
    args = parser.parse_args()

    edit_specs = load_json(Path(args.edit_json))
    sources = dict(args.source)
    if len(sources) != len(args.source):
        raise ValueError("Source labels must be unique")
    if args.default_label not in sources:
        raise ValueError(f"default label {args.default_label!r} is not a source")
    for label, image_dir in sources.items():
        if not image_dir.exists():
            raise FileNotFoundError(f"Source {label} directory not found: {image_dir}")

    routes = dict(args.route)
    unknown_labels = sorted({label for label in routes.values() if label not in sources})
    if unknown_labels:
        raise ValueError(f"Route references unknown source label(s): {unknown_labels}")
    edit_types = {str(item.get("edit_type")) for item in edit_specs.values()}
    unknown_types = sorted(set(routes) - edit_types)
    if unknown_types:
        raise ValueError(f"Route references unknown edit_type(s): {unknown_types}")

    output_dir = Path(args.image_root) / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {label: 0 for label in sources}
    type_counts: dict[str, dict[str, int]] = {}
    records = []

    for key, item in edit_specs.items():
        key = str(key)
        edit_type = str(item.get("edit_type"))
        label = routes.get(edit_type, args.default_label)
        src = sources[label] / f"{key}.png"
        if not src.exists():
            raise FileNotFoundError(src)
        dst = output_dir / f"{key}.png"
        link_or_copy(src, dst)
        counts[label] += 1
        type_counts.setdefault(edit_type, {})
        type_counts[edit_type][label] = type_counts[edit_type].get(label, 0) + 1
        records.append(
            {
                "key": key,
                "edit_type": edit_type,
                "source": label,
                "source_path": str(src),
                "target_path": str(dst),
            }
        )

    manifest = {
        "model_name": args.model_name,
        "edit_json": args.edit_json,
        "default_label": args.default_label,
        "routes": routes,
        "sources": {label: str(path) for label, path in sources.items()},
        "counts": counts,
        "type_counts": {key: dict(sorted(value.items())) for key, value in sorted(type_counts.items())},
        "records": records,
    }
    manifest_path = Path(args.scores_root) / f"{args.model_name}_type_router_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"Built {len(records)} routed image(s) in {output_dir}", flush=True)
    print(f"Choice counts: {counts}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
