from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from qwen_edit_project.eval.summarize_scores import summarize_oneig
from qwen_edit_project.utils.commands import run_and_tee
from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import ensure_dir, resolve_path
from qwen_edit_project.utils.runtime import get_python_executable
from qwen_edit_project.utils.run_metadata import base_run_metadata, utc_timestamp

ONEIG_MODULES = {
    "alignment": {
        "module": "scripts.alignment.alignment_score",
        "image_suffix": "",
        "class_items_key": "alignment_class_items",
    },
    "text": {
        "module": "scripts.text.text_score",
        "image_suffix": "text",
        "class_items_key": None,
    },
    "diversity": {
        "module": "scripts.diversity.diversity_score",
        "image_suffix": "",
        "class_items_key": "diversity_class_items",
    },
    "style": {
        "module": "scripts.style.style_score",
        "image_suffix": "anime",
        "class_items_key": None,
    },
    "reasoning": {
        "module": "scripts.reasoning.reasoning_score",
        "image_suffix": "reasoning",
        "class_items_key": None,
    },
}


def csv_snapshot(results_dir: Path) -> set[str]:
    if not results_dir.exists():
        return set()
    return {path.name for path in results_dir.glob("*.csv")}


def copy_new_csvs(results_dir: Path, before: set[str], destination_dir: Path) -> list[str]:
    copied: list[str] = []
    for path in sorted(results_dir.glob("*.csv")):
        if path.name in before:
            continue
        target = destination_dir / path.name
        shutil.copy2(path, target)
        copied.append(str(target))
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OneIG-Bench public scorers.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    model_name = config["model"]["model_name"]
    mode = str(config["dataset"].get("mode", "EN")).upper()
    image_root = resolve_path(config["output"]["image_root"])
    scores_dir = resolve_path(config["output"]["scores_dir"])
    oneig_root = resolve_path(config["scoring"].get("repo_path", "third_party/oneig-bench"))
    if image_root is None or scores_dir is None or oneig_root is None:
        raise ValueError("OneIG-Bench paths must resolve")

    image_dir = image_root / mode.lower()
    if not image_dir.exists():
        raise FileNotFoundError(f"OneIG-Bench output directory not found: {image_dir}")

    timestamp = utc_timestamp()
    run_dir = ensure_dir(scores_dir / model_name / mode.lower() / timestamp)
    results_dir = oneig_root / "results"
    python_executable = get_python_executable(config)
    grid = f"{config['generation'].get('grid_cols', 2)},{config['generation'].get('grid_rows', 2)}"
    copied_files: dict[str, list[str]] = {}

    for module_name in config["scoring"].get("modules", list(ONEIG_MODULES)):
        spec = ONEIG_MODULES[module_name]
        before = csv_snapshot(results_dir)
        target_image_dir = image_dir / spec["image_suffix"] if spec["image_suffix"] else image_dir
        command = [
            python_executable,
            "-m",
            spec["module"],
            "--mode",
            mode,
            "--image_dirname",
            str(target_image_dir),
            "--model_names",
            model_name,
            "--image_grid",
            grid,
        ]
        class_items_key = spec["class_items_key"]
        if class_items_key:
            command.extend(["--class_items", *config["scoring"][class_items_key]])
        log_path = resolve_path(f"outputs/logs/oneig_{module_name}_{timestamp}.log")
        return_code = run_and_tee(command, cwd=oneig_root, log_path=log_path)
        if return_code != 0:
            raise SystemExit(return_code)
        copied_files[module_name] = copy_new_csvs(results_dir, before, run_dir)

    save_json(
        {
            **base_run_metadata(),
            "benchmark": "oneig",
            "config_path": config["_config_path"],
            "model_name": model_name,
            "mode": mode,
            "score_dir": str(run_dir),
            "copied_files": copied_files,
            "metrics": summarize_oneig(run_dir, model_name),
        },
        run_dir / f"{model_name}_summary.json",
    )


if __name__ == "__main__":
    main()
