from __future__ import annotations

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageStat


def prepare_rgb(image: Image.Image, size: tuple[int, int] = (128, 128)) -> Image.Image:
    return image.convert("RGB").resize(size)


def luminance_mean(image: Image.Image) -> float:
    gray = prepare_rgb(image).convert("L")
    return ImageStat.Stat(gray).mean[0] / 255.0


def luminance_std(image: Image.Image) -> float:
    gray = prepare_rgb(image).convert("L")
    return ImageStat.Stat(gray).stddev[0] / 255.0


def saturation_mean(image: Image.Image) -> float:
    hsv = prepare_rgb(image).convert("HSV")
    return ImageStat.Stat(hsv).mean[1] / 255.0


def warmth_score(image: Image.Image) -> float:
    red, _, blue = prepare_rgb(image).split()
    red_mean = ImageStat.Stat(red).mean[0] / 255.0
    blue_mean = ImageStat.Stat(blue).mean[0] / 255.0
    return red_mean - blue_mean


def changed_fraction(
    original: Image.Image,
    edited: Image.Image,
    diff_threshold: int = 18,
) -> float:
    original_rgb = prepare_rgb(original)
    edited_rgb = prepare_rgb(edited)
    diff = ImageChops.difference(original_rgb, edited_rgb).convert("L")
    histogram = diff.histogram()
    total = sum(histogram)
    changed = sum(histogram[diff_threshold + 1 :])
    if total == 0:
        return 0.0
    return changed / total


def edge_preservation_score(original: Image.Image, edited: Image.Image) -> float:
    orig_edges = prepare_rgb(original).convert("L").filter(ImageFilter.FIND_EDGES)
    edit_edges = prepare_rgb(edited).convert("L").filter(ImageFilter.FIND_EDGES)
    diff = ImageChops.difference(orig_edges, edit_edges)
    edge_delta = ImageStat.Stat(diff).mean[0] / 255.0
    return max(0.0, 1.0 - edge_delta * 2.0)


def mean_absolute_difference(original: Image.Image, edited: Image.Image) -> float:
    original_arr = np.asarray(prepare_rgb(original), dtype=np.float32)
    edited_arr = np.asarray(prepare_rgb(edited), dtype=np.float32)
    return float(np.abs(original_arr - edited_arr).mean() / 255.0)


def diff_region_statistics(
    original: Image.Image,
    edited: Image.Image,
    diff_threshold: int = 18,
) -> dict[str, float]:
    original_arr = np.asarray(prepare_rgb(original), dtype=np.float32)
    edited_arr = np.asarray(prepare_rgb(edited), dtype=np.float32)
    diff = np.abs(original_arr - edited_arr).mean(axis=2) / 255.0
    threshold = diff_threshold / 255.0
    mask = diff > threshold

    changed_fraction_value = float(mask.mean())
    if mask.any():
        inside_mean = float(diff[mask].mean())
        ys, xs = np.where(mask)
        bbox_area = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
        compactness = float(mask.sum() / max(bbox_area, 1.0))
    else:
        inside_mean = 0.0
        compactness = 0.0

    if (~mask).any():
        outside_mean = float(diff[~mask].mean())
    else:
        outside_mean = inside_mean

    total_mean = float(diff.mean())
    precision = float(inside_mean / max(inside_mean + outside_mean, 1e-6))
    return {
        "changed_fraction": changed_fraction_value,
        "inside_mean_delta": inside_mean,
        "outside_mean_delta": outside_mean,
        "total_mean_delta": total_mean,
        "mask_compactness": compactness,
        "mask_precision": precision,
    }
