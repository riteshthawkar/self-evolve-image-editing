from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from qwen_edit_project.eval.summarize_scores import summarize_imgedit
from qwen_edit_project.utils.commands import run_and_tee
from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import ensure_dir, resolve_path
from qwen_edit_project.utils.runtime import get_python_executable
from qwen_edit_project.utils.run_metadata import base_run_metadata, utc_timestamp


def extract_imgedit_average(entry: object) -> float | None:
    if not isinstance(entry, str):
        return None
    scores = []
    for line in entry.splitlines():
        parts = line.strip().split(": ")
        if len(parts) == 2 and parts[1].isdigit():
            scores.append(int(parts[1]))
    if not scores:
        return None
    return round(sum(scores) / len(scores), 2)


def load_json_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def invalid_imgedit_keys(edit_specs: dict, results: dict) -> list[str]:
    invalid = []
    for key in edit_specs:
        if extract_imgedit_average(results.get(key)) is None:
            invalid.append(str(key))
    return invalid


def save_filtered_edit_json(edit_specs: dict, keys: list[str], path: Path) -> None:
    selected = {key: edit_specs[key] for key in keys}
    save_json(selected, path)


def run_basic_scorer(
    *,
    python_executable: str,
    repo_root: Path,
    result_dir: Path,
    edit_json_path: Path,
    origin_img_root: Path,
    prompts_json_path: Path,
    num_processes: int,
    timestamp: str,
    label: str,
    model: str,
    timeout: float,
    max_retries: int,
) -> Path:
    basic_log = resolve_path(f"outputs/logs/imgedit_score_{timestamp}_{label}.log")
    command = [
        python_executable,
        "-m",
        "qwen_edit_project.eval.imgedit_basic_bench",
        "--result_img_folder",
        str(result_dir),
        "--edit_json",
        str(edit_json_path),
        "--origin_img_root",
        str(origin_img_root),
        "--num_processes",
        str(num_processes),
        "--prompts_json",
        str(prompts_json_path),
        "--model",
        model,
        "--timeout",
        str(timeout),
        "--max_retries",
        str(max_retries),
    ]
    return_code = run_and_tee(command, cwd=repo_root, log_path=basic_log)
    if return_code != 0:
        raise SystemExit(return_code)
    return basic_log


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
    edit_json_path = resolve_path(config["dataset"]["edit_json"])
    origin_img_root = resolve_path(config["dataset"]["origin_img_root"])
    prompts_json_path = resolve_path(config["dataset"]["prompts_json"])
    edited_images_dir = resolve_path(config["output"]["edited_images_dir"])
    scores_dir = resolve_path(config["output"]["scores_dir"])
    if edit_json_path is None or origin_img_root is None or prompts_json_path is None:
        raise ValueError("ImgEdit dataset paths must resolve")
    if edited_images_dir is None or scores_dir is None:
        raise ValueError("edited_images_dir and scores_dir must resolve")
    result_dir = edited_images_dir / model_name
    if not result_dir.exists():
        raise FileNotFoundError(f"ImgEdit output directory not found: {result_dir}")
    ensure_dir(scores_dir)

    edit_specs = load_json_dict(edit_json_path)
    missing_images = [key for key in edit_specs if not (result_dir / f"{key}.png").exists()]
    if missing_images:
        raise FileNotFoundError(
            "ImgEdit export is incomplete; refusing to score a partial image set. "
            f"Missing {len(missing_images)} generated PNG(s), first missing keys: {missing_images[:20]}"
        )

    timestamp = utc_timestamp()
    repo_root = resolve_path(".")
    python_executable = get_python_executable(config)
    result_json = result_dir / "result.json"
    logs = []
    results = load_json_dict(result_json)
    invalid_keys = invalid_imgedit_keys(edit_specs, results)
    openai_model = str(config["scoring"].get("openai_model", "gpt-4o"))
    openai_timeout = float(config["scoring"].get("openai_timeout_seconds", 60))
    openai_max_retries = int(config["scoring"].get("openai_max_retries", 2))
    if invalid_keys and len(invalid_keys) == len(edit_specs):
        logs.append(
            str(
                run_basic_scorer(
                    python_executable=python_executable,
                    repo_root=repo_root,
                    result_dir=result_dir,
                    edit_json_path=edit_json_path,
                    origin_img_root=origin_img_root,
                    prompts_json_path=prompts_json_path,
                    num_processes=int(config["scoring"].get("num_processes", 4)),
                    timestamp=timestamp,
                    label="initial",
                    model=openai_model,
                    timeout=openai_timeout,
                    max_retries=openai_max_retries,
                )
            )
        )
        results = load_json_dict(result_json)
        invalid_keys = invalid_imgedit_keys(edit_specs, results)

    max_retry_rounds = int(config["scoring"].get("max_retry_rounds", 5))
    retry_num_processes = int(config["scoring"].get("retry_num_processes", 1))
    retry_sleep_seconds = float(config["scoring"].get("retry_sleep_seconds", 10))
    for retry_index in range(1, max_retry_rounds + 1):
        if not invalid_keys:
            break
        retry_json = scores_dir / f"{model_name}_retry_{retry_index:02d}_edit.json"
        save_filtered_edit_json(edit_specs, invalid_keys, retry_json)
        full_results = dict(results)
        if retry_sleep_seconds > 0:
            time.sleep(retry_sleep_seconds)
        logs.append(
            str(
                run_basic_scorer(
                    python_executable=python_executable,
                    repo_root=repo_root,
                    result_dir=result_dir,
                    edit_json_path=retry_json,
                    origin_img_root=origin_img_root,
                    prompts_json_path=prompts_json_path,
                    num_processes=max(1, retry_num_processes),
                    timestamp=timestamp,
                    label=f"retry_{retry_index:02d}",
                    model=openai_model,
                    timeout=openai_timeout,
                    max_retries=openai_max_retries,
                )
            )
        )
        retry_results = load_json_dict(result_json)
        full_results.update(retry_results)
        save_json(full_results, result_json)
        results = full_results
        invalid_keys = invalid_imgedit_keys(edit_specs, results)

    if invalid_keys and not bool(config["scoring"].get("allow_partial", False)):
        failure_path = scores_dir / f"{model_name}_unscored_keys.json"
        save_json({"unscored_keys": invalid_keys}, failure_path)
        raise RuntimeError(
            f"ImgEdit scoring still has {len(invalid_keys)} unscored key(s) after "
            f"{max_retry_rounds} retry round(s). Wrote {failure_path}. Rerun scoring to retry, "
            "or set scoring.allow_partial=true only for debugging."
        )

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
    logs.append(str(avg_log))

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
    logs.append(str(type_log))

    save_json(
        {
            **base_run_metadata(),
            "benchmark": "imgedit",
            "config_path": config["_config_path"],
            "model_name": model_name,
            "result_dir": str(result_dir),
            "average_score_json": str(average_score_json),
            "typescore_json": str(typescore_json),
            "logs": logs,
            "unscored_keys": invalid_keys,
            "metrics": summarize_imgedit(scores_dir, model_name),
        },
        scores_dir / f"{model_name}_summary.json",
    )


if __name__ == "__main__":
    main()
