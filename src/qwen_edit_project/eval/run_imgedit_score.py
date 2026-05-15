from __future__ import annotations

import argparse
import os
from pathlib import Path

from qwen_edit_project.eval.summarize_scores import summarize_imgedit
from qwen_edit_project.utils.commands import run_and_tee
from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import ensure_dir, resolve_path
from qwen_edit_project.utils.runtime import get_python_executable
from qwen_edit_project.utils.run_metadata import base_run_metadata, utc_timestamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ImgEdit public scorer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY must be set before scoring ImgEdit")

    model_name = config["model"]["model_name"]
    edited_images_dir = resolve_path(config["output"]["edited_images_dir"])
    scores_dir = resolve_path(config["output"]["scores_dir"])
    if edited_images_dir is None or scores_dir is None:
        raise ValueError("edited_images_dir and scores_dir must resolve")
    result_dir = edited_images_dir / model_name
    if not result_dir.exists():
        raise FileNotFoundError(f"ImgEdit output directory not found: {result_dir}")
    ensure_dir(scores_dir)

    timestamp = utc_timestamp()
    repo_root = resolve_path(".")
    python_executable = get_python_executable(config)
    basic_log = resolve_path(f"outputs/logs/imgedit_score_{timestamp}.log")
    command = [
        python_executable,
        str(resolve_path("third_party/imgedit/Benchmark/Basic/basic_bench.py")),
        "--result_img_folder",
        str(result_dir),
        "--edit_json",
        str(resolve_path(config["dataset"]["edit_json"])),
        "--origin_img_root",
        str(resolve_path(config["dataset"]["origin_img_root"])),
        "--num_processes",
        str(config["scoring"].get("num_processes", 4)),
        "--prompts_json",
        str(resolve_path(config["dataset"]["prompts_json"])),
    ]
    return_code = run_and_tee(command, cwd=repo_root, log_path=basic_log)
    if return_code != 0:
        raise SystemExit(return_code)

    avg_log = resolve_path(f"outputs/logs/imgedit_avg_{timestamp}.log")
    average_score_json = scores_dir / f"{model_name}_average_score.json"
    command = [
        python_executable,
        str(resolve_path("third_party/imgedit/Benchmark/Basic/step1_get_avgscore.py")),
        "--result_json",
        str(result_dir / "result.json"),
        "--average_score_json",
        str(average_score_json),
    ]
    return_code = run_and_tee(command, cwd=repo_root, log_path=avg_log)
    if return_code != 0:
        raise SystemExit(return_code)

    type_log = resolve_path(f"outputs/logs/imgedit_types_{timestamp}.log")
    typescore_json = scores_dir / f"{model_name}_typescore.json"
    command = [
        python_executable,
        str(resolve_path("third_party/imgedit/Benchmark/Basic/step2_typescore.py")),
        "--average_score_json",
        str(average_score_json),
        "--typescore_json",
        str(typescore_json),
        "--basic_edit",
        str(resolve_path(config["dataset"]["edit_json"])),
    ]
    return_code = run_and_tee(command, cwd=repo_root, log_path=type_log)
    if return_code != 0:
        raise SystemExit(return_code)

    save_json(
        {
            **base_run_metadata(),
            "benchmark": "imgedit",
            "config_path": config["_config_path"],
            "model_name": model_name,
            "result_dir": str(result_dir),
            "average_score_json": str(average_score_json),
            "typescore_json": str(typescore_json),
            "logs": [str(basic_log), str(avg_log), str(type_log)],
            "metrics": summarize_imgedit(scores_dir, model_name),
        },
        scores_dir / f"{model_name}_summary.json",
    )


if __name__ == "__main__":
    main()
