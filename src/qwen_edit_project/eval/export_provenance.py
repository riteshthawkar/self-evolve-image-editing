from __future__ import annotations

from pathlib import Path
from typing import Any

from qwen_edit_project.utils.config import save_json


def build_edit_export_provenance(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config["model"]
    generation_cfg = config["generation"]
    scoring_cfg = config.get("scoring", {})
    return {
        "model_backend": model_cfg.get("backend", "diffsynth"),
        "base_model": model_cfg.get("base_model"),
        "model_type": model_cfg.get("model_type", "base"),
        "checkpoint_path": model_cfg.get("checkpoint_path"),
        "torch_dtype": model_cfg.get("torch_dtype", "auto"),
        "generation": {
            "num_inference_steps": generation_cfg.get("num_inference_steps"),
            "true_cfg_scale": generation_cfg.get("true_cfg_scale"),
            "guidance_scale": generation_cfg.get("guidance_scale"),
            "negative_prompt": generation_cfg.get("negative_prompt"),
            "num_images_per_prompt": generation_cfg.get("num_images_per_prompt"),
            "preserve_input_resolution": generation_cfg.get("preserve_input_resolution", True),
        },
        "scoring": {
            "backbone": scoring_cfg.get("backbone"),
            "expected_openai_model": scoring_cfg.get("expected_openai_model"),
        },
    }


def validate_resume_provenance(
    *,
    benchmark: str,
    output_root: Path,
    summary_path: Path,
    expected: dict[str, Any],
    no_resume: bool,
    allow_mismatch: bool = False,
) -> None:
    if no_resume or allow_mismatch or not output_root.exists():
        return
    existing_png_count = sum(1 for _ in output_root.rglob("*.png"))
    if existing_png_count == 0:
        return
    if not summary_path.exists():
        raise RuntimeError(
            f"{benchmark} output directory already contains {existing_png_count} PNGs, but no "
            f"provenance summary exists at {summary_path}. Use a new model.model_name, delete the "
            "old output directory, or rerun export with --no-resume to regenerate with the current settings."
        )
    import json

    with summary_path.open("r", encoding="utf-8") as handle:
        previous = json.load(handle)
    previous_provenance = previous.get("export_provenance")
    if previous_provenance != expected:
        save_json(
            {"expected": expected, "previous": previous_provenance},
            summary_path.with_name(summary_path.stem + "_provenance_mismatch.json"),
        )
        raise RuntimeError(
            f"{benchmark} output directory already contains {existing_png_count} PNGs generated with "
            "different or unknown settings. Refusing to silently mix paper-matched outputs with older "
            f"images. Inspect {summary_path.with_name(summary_path.stem + '_provenance_mismatch.json')}, "
            "then use a new model.model_name, delete the old output directory, or pass --no-resume."
        )
