from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw

from .paths import ensure_dir


def can_open_image(path: Path) -> tuple[bool, str | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        return True, None
    except Exception as exc:
        return False, str(exc)


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def black_image(size: tuple[int, int]) -> Image.Image:
    return Image.new("RGB", size, color=(0, 0, 0))


def create_image_grid(
    images: Sequence[Image.Image],
    rows: int,
    columns: int,
    fill_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    if not images:
        raise ValueError("At least one image is required to build a grid")
    if rows < 1 or columns < 1:
        raise ValueError("Grid rows and columns must be positive")

    rgb_images = [image.convert("RGB") for image in images]
    cell_width = max(image.width for image in rgb_images)
    cell_height = max(image.height for image in rgb_images)
    grid = Image.new("RGB", (columns * cell_width, rows * cell_height), color=fill_color)

    for index, image in enumerate(rgb_images[: rows * columns]):
        x = (index % columns) * cell_width
        y = (index // columns) * cell_height
        offset_x = x + (cell_width - image.width) // 2
        offset_y = y + (cell_height - image.height) // 2
        grid.paste(image, (offset_x, offset_y))

    return grid


def save_contact_sheet(
    image_paths: Sequence[Path],
    labels: Sequence[str],
    output_path: Path,
    thumb_size: tuple[int, int] = (256, 256),
    columns: int = 2,
) -> Path:
    ensure_dir(output_path.parent)
    count = len(image_paths)
    rows = max((count + columns - 1) // columns, 1)
    label_height = 48
    sheet = Image.new(
        "RGB",
        (columns * thumb_size[0], rows * (thumb_size[1] + label_height)),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(sheet)

    for index, image_path in enumerate(image_paths):
        image = load_rgb(image_path)
        image.thumbnail(thumb_size)
        x = (index % columns) * thumb_size[0]
        y = (index // columns) * (thumb_size[1] + label_height)
        offset_x = x + (thumb_size[0] - image.width) // 2
        offset_y = y + (thumb_size[1] - image.height) // 2
        sheet.paste(image, (offset_x, offset_y))
        draw.text((x + 4, y + thumb_size[1] + 4), labels[index][:80], fill=(0, 0, 0))

    sheet.save(output_path)
    return output_path


def sample_items(items: Sequence[dict], count: int, seed: int = 0) -> list[dict]:
    if len(items) <= count:
        return list(items)
    generator = random.Random(seed)
    return generator.sample(list(items), count)
