from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

from qwen_edit_project.utils.image_io import black_image, create_image_grid
from qwen_edit_project.utils.qwen_pipeline import render_generation


def extract_primary_image(output: Any) -> Image.Image:
    if hasattr(output, "images"):
        return output.images[0].convert("RGB")
    if isinstance(output, Image.Image):
        return output.convert("RGB")
    raise TypeError(f"Unsupported pipeline output type: {type(output)!r}")


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def resolve_grid_shape(generation: dict[str, Any], sample_count: int) -> tuple[int, int]:
    rows = generation.get("grid_rows")
    columns = generation.get("grid_cols")
    if rows is None and columns is None:
        columns = max(int(math.ceil(math.sqrt(sample_count))), 1)
        rows = max(int(math.ceil(sample_count / columns)), 1)
    elif rows is None:
        columns = max(int(columns), 1)
        rows = max(int(math.ceil(sample_count / columns)), 1)
    elif columns is None:
        rows = max(int(rows), 1)
        columns = max(int(math.ceil(sample_count / rows)), 1)
    else:
        rows = max(int(rows), 1)
        columns = max(int(columns), 1)
    return rows, columns


def build_sample_grid(images: list[Image.Image], generation: dict[str, Any]) -> Image.Image:
    rows, columns = resolve_grid_shape(generation, len(images))
    return create_image_grid(images, rows=rows, columns=columns)


def generate_prompt_samples(
    pipe: Any,
    prompt: str,
    generation: dict[str, Any],
    prompt_index: int,
) -> tuple[list[Image.Image], list[str]]:
    sample_count = int(generation.get("samples_per_prompt", 1))
    seed = int(generation.get("seed", 42))
    seed_stride = int(generation.get("seed_stride", 1000))
    fill_with_black = bool(generation.get("fill_with_black_on_error", False))
    width = int(generation.get("width", 1024))
    height = int(generation.get("height", 1024))

    images: list[Image.Image] = []
    errors: list[str] = []
    for sample_index in range(sample_count):
        sample_generation = dict(generation)
        sample_generation["seed"] = seed + prompt_index * seed_stride + sample_index
        try:
            output = render_generation(pipe, prompt, sample_generation)
            images.append(extract_primary_image(output))
        except Exception as exc:
            if not fill_with_black:
                raise
            errors.append(str(exc))

    if fill_with_black and len(images) < sample_count:
        images.extend(black_image((width, height)) for _ in range(sample_count - len(images)))
    if not images:
        raise RuntimeError("No images were generated for the prompt")
    return images, errors
