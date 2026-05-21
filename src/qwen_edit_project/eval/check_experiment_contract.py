from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from qwen_edit_project.utils.config import load_yaml_config
from qwen_edit_project.utils.paths import resolve_path


EXPECTED_BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
EXPECTED_EDIT_GENERATION = {
    "num_inference_steps": 40,
    "true_cfg_scale": 4.0,
    "guidance_scale": 1.0,
    "negative_prompt": " ",
    "num_images_per_prompt": 1,
    "preserve_input_resolution": False,
}


def _format_path(path: str | Path) -> str:
    resolved = resolve_path(path)
    return str(resolved if resolved is not None else path)


class ContractError(AssertionError):
    pass


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def require_equal(actual: Any, expected: Any, message: str, errors: list[str]) -> None:
    if actual != expected:
        errors.append(f"{message}: expected {expected!r}, got {actual!r}")


def check_generation(prefix: str, generation: dict[str, Any], errors: list[str]) -> None:
    for key, expected in EXPECTED_EDIT_GENERATION.items():
        require_equal(generation.get(key), expected, f"{prefix}.generation.{key}", errors)


def check_official_edit_model(prefix: str, model: dict[str, Any], errors: list[str]) -> None:
    require_equal(model.get("backend"), "official_diffusers", f"{prefix}.model.backend", errors)
    require_equal(model.get("base_model"), EXPECTED_BASE_MODEL, f"{prefix}.model.base_model", errors)
    require_equal(model.get("model_type"), "base", f"{prefix}.model.model_type", errors)
    require_equal(model.get("torch_dtype"), "bfloat16", f"{prefix}.model.torch_dtype", errors)


def check_eval_config(path: str | Path, errors: list[str]) -> None:
    config = load_yaml_config(path)
    prefix = _format_path(path)
    check_official_edit_model(prefix, config.get("model", {}), errors)
    check_generation(prefix, config.get("generation", {}), errors)
    require_equal(config.get("prompting", {}).get("use_prompt_polish"), False, f"{prefix}.prompting.use_prompt_polish", errors)
    if config.get("benchmark") == "gedit":
        scoring = config.get("scoring", {})
        require_equal(scoring.get("backbone"), "gpt4o", f"{prefix}.scoring.backbone", errors)
        require_equal(scoring.get("expected_openai_model"), "gpt-4.1", f"{prefix}.scoring.expected_openai_model", errors)


def check_self_evolve_config(path: str | Path, errors: list[str]) -> None:
    config = load_yaml_config(path)
    prefix = _format_path(path)
    editor = config.get("editor", {})
    require_equal(editor.get("backend"), "qwen_edit", f"{prefix}.editor.backend", errors)
    check_generation(f"{prefix}.editor", editor.get("generation", {}), errors)
    check_official_edit_model(f"{prefix}.editor", editor.get("model", {}), errors)
    replay_ratio = float(config.get("training", {}).get("reconstruction_replay_ratio", 0.0))
    require(replay_ratio > 0.0, f"{prefix}.training.reconstruction_replay_ratio must be > 0", errors)


def check_train_config(path: str | Path, errors: list[str]) -> None:
    config = load_yaml_config(path)
    prefix = _format_path(path)
    model_paths = config.get("model", {}).get("model_id_with_origin_paths", "")
    require(EXPECTED_BASE_MODEL in model_paths, f"{prefix}.model must start from {EXPECTED_BASE_MODEL}", errors)
    dataset = config.get("dataset", {})
    require_equal(dataset.get("data_file_keys"), "image,edit_image", f"{prefix}.dataset.data_file_keys", errors)
    require_equal(dataset.get("extra_inputs"), "edit_image", f"{prefix}.dataset.extra_inputs", errors)
    require_equal(dataset.get("max_pixels"), 1024 * 1024, f"{prefix}.dataset.max_pixels", errors)
    require_equal(config.get("training", {}).get("height"), None, f"{prefix}.training.height", errors)
    require_equal(config.get("training", {}).get("width"), None, f"{prefix}.training.width", errors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate that experiment configs follow the Qwen paper-matched contract.")
    parser.add_argument("--self-evolve-glob", default="configs/self_evolve/qwen_edit_2509*.yaml")
    args = parser.parse_args()

    errors: list[str] = []
    check_eval_config("configs/eval/gedit.yaml", errors)
    check_eval_config("configs/eval/imgedit.yaml", errors)
    check_train_config("configs/train/lora_2509.yaml", errors)
    check_train_config("configs/train/full_2509.yaml", errors)

    repo_root = resolve_path(".")
    if repo_root is None:
        raise RuntimeError("Could not resolve repository root")
    self_evolve_paths = sorted(repo_root.glob(args.self_evolve_glob))
    require(bool(self_evolve_paths), f"No self-evolve configs matched {args.self_evolve_glob}", errors)
    for path in self_evolve_paths:
        check_self_evolve_config(path, errors)

    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise ContractError(f"Experiment contract check failed:\n{joined}")
    print("Experiment contract check passed.")


if __name__ == "__main__":
    main()
