from __future__ import annotations

import argparse

from qwen_edit_project.eval.summarize_scores import summarize_geneval
from qwen_edit_project.utils.commands import run_and_tee
from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import ensure_dir, resolve_path
from qwen_edit_project.utils.runtime import get_python_executable
from qwen_edit_project.utils.run_metadata import base_run_metadata, utc_timestamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GenEval public scorer.")
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
    if image_root is None or scores_dir is None:
        raise ValueError("output.image_root and output.scores_dir must resolve")
    model_dir = image_root / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"GenEval output directory not found: {model_dir}")

    object_detector_root = resolve_path(config["scoring"]["object_detector_root"])
    if object_detector_root is None or not object_detector_root.exists():
        raise FileNotFoundError("GenEval object detector root is missing")

    ensure_dir(scores_dir)
    python_executable = get_python_executable(config)
    geneval_root = resolve_path(config["scoring"].get("repo_path", "third_party/geneval"))
    if geneval_root is None:
        raise ValueError("scoring.repo_path must resolve")

    timestamp = utc_timestamp()
    results_path = scores_dir / f"{model_name}_results.jsonl"
    score_log = resolve_path(f"outputs/logs/geneval_score_{timestamp}.log")
    command = [
        python_executable,
        str(geneval_root / "evaluation/evaluate_images.py"),
        str(model_dir),
        "--outfile",
        str(results_path),
        "--model-path",
        str(object_detector_root),
    ]
    if config["scoring"].get("model_config"):
        command.extend(["--model-config", str(resolve_path(config["scoring"]["model_config"]))])
    if config["scoring"].get("options"):
        command.append("--options")
        command.extend(str(item) for item in config["scoring"]["options"])

    return_code = run_and_tee(command, cwd=geneval_root, log_path=score_log)
    if return_code != 0:
        raise SystemExit(return_code)

    summary_log = resolve_path(f"outputs/logs/geneval_summary_{timestamp}.log")
    summary_command = [
        python_executable,
        str(geneval_root / "evaluation/summary_scores.py"),
        str(results_path),
    ]
    return_code = run_and_tee(summary_command, cwd=geneval_root, log_path=summary_log)
    if return_code != 0:
        raise SystemExit(return_code)

    save_json(
        {
            **base_run_metadata(),
            "benchmark": "geneval",
            "config_path": config["_config_path"],
            "model_name": model_name,
            "result_jsonl": str(results_path),
            "logs": [str(score_log), str(summary_log)],
            "metrics": summarize_geneval(results_path),
        },
        scores_dir / f"{model_name}_summary.json",
    )


if __name__ == "__main__":
    main()
