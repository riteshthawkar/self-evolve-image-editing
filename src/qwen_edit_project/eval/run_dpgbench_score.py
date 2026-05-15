from __future__ import annotations

import argparse

from qwen_edit_project.eval.summarize_scores import summarize_dpgbench
from qwen_edit_project.utils.commands import run_and_tee
from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import ensure_dir, resolve_path
from qwen_edit_project.utils.runtime import resolve_executable
from qwen_edit_project.utils.run_metadata import base_run_metadata, utc_timestamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DPG-Bench public scorer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    model_name = config["model"]["model_name"]
    image_root = resolve_path(config["output"]["image_root"])
    scores_dir = resolve_path(config["output"]["scores_dir"])
    prompts_csv = resolve_path(config["dataset"]["prompts_csv"])
    ella_root = resolve_path(config["scoring"].get("repo_path", "third_party/ella"))
    if image_root is None or scores_dir is None or prompts_csv is None or ella_root is None:
        raise ValueError("DPG-Bench paths must resolve")

    model_dir = image_root / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"DPG-Bench output directory not found: {model_dir}")
    ensure_dir(scores_dir)

    accelerate_executable = resolve_executable(
        config.get("runtime", {}).get("accelerate_executable"),
        "accelerate",
    )
    timestamp = utc_timestamp()
    results_path = scores_dir / f"{model_name}_results.txt"
    score_log = resolve_path(f"outputs/logs/dpgbench_score_{timestamp}.log")
    command = [
        accelerate_executable,
        "launch",
    ]
    if config["scoring"].get("num_processes") is not None:
        command.extend(["--num_processes", str(config["scoring"]["num_processes"])])
    command.extend(
        [
            str(ella_root / "dpg_bench/compute_dpg_bench.py"),
            "--image-root-path",
            str(model_dir),
            "--resolution",
            str(config["scoring"]["resolution"]),
            "--csv",
            str(prompts_csv),
            "--res-path",
            str(results_path),
            "--pic-num",
            str(config["scoring"].get("pic_num", config["generation"].get("samples_per_prompt", 1))),
            "--vqa-model",
            str(config["scoring"].get("vqa_model", "mplug")),
        ]
    )
    return_code = run_and_tee(command, cwd=ella_root, log_path=score_log)
    if return_code != 0:
        raise SystemExit(return_code)

    save_json(
        {
            **base_run_metadata(),
            "benchmark": "dpgbench",
            "config_path": config["_config_path"],
            "model_name": model_name,
            "result_txt": str(results_path),
            "logs": [str(score_log)],
            "metrics": summarize_dpgbench(results_path),
        },
        scores_dir / f"{model_name}_summary.json",
    )


if __name__ == "__main__":
    main()
