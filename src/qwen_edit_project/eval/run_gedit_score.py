from __future__ import annotations

import argparse
import os
from pathlib import Path

from qwen_edit_project.eval.summarize_scores import summarize_gedit
from qwen_edit_project.utils.commands import run_and_tee
from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import resolve_path
from qwen_edit_project.utils.runtime import get_python_executable
from qwen_edit_project.utils.run_metadata import base_run_metadata, utc_timestamp


def ensure_secret_env(secret_path: Path, target_path: Path) -> None:
    target_path.write_text(secret_path.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GEdit public scorer.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    model_name = config["model"]["model_name"]
    edited_images_dir = resolve_path(config["output"]["edited_images_dir"])
    save_dir = resolve_path(config["scoring"]["save_dir"])
    if edited_images_dir is None or save_dir is None:
        raise ValueError("edited_images_dir and save_dir must resolve")
    model_dir = edited_images_dir / model_name / "fullset"
    if not model_dir.exists():
        raise FileNotFoundError(f"GEdit output directory not found: {model_dir}")

    repo_root = resolve_path(".")
    secret_env = os.environ.get("GEDIT_SECRET_ENV_PATH") or config["scoring"].get("scorer_secret_env_path")
    secret_path = resolve_path(secret_env) if secret_env else None
    if secret_path is None or not secret_path.exists():
        raise FileNotFoundError("GEdit scorer secret env file is missing")
    local_secret = repo_root / "secret.env"
    python_executable = get_python_executable(config)
    had_existing_secret = local_secret.exists()
    original_secret = local_secret.read_text(encoding="utf-8") if had_existing_secret else None
    ensure_secret_env(secret_path, local_secret)
    gedit_root = resolve_path("third_party/step1x-edit/GEdit-Bench")
    if gedit_root is None or not gedit_root.exists():
        raise FileNotFoundError("GEdit scorer repo is missing. Run scripts/bootstrap.sh first.")
    scorer_pythonpath = os.pathsep.join(
        [
            str(gedit_root / "viescore"),
            str(gedit_root),
            os.environ.get("PYTHONPATH", ""),
        ]
    )
    scorer_env = {"PYTHONPATH": scorer_pythonpath}

    timestamp = utc_timestamp()
    log_path = resolve_path(f"outputs/logs/gedit_score_{timestamp}.log")
    try:
        command = [
            python_executable,
            str(gedit_root / "run_gedit_score.py"),
            "--model_name",
            model_name,
            "--edited_images_dir",
            str(edited_images_dir),
            "--save_dir",
            str(save_dir),
            "--backbone",
            config["scoring"].get("backbone", "gpt4o"),
        ]
        return_code = run_and_tee(command, cwd=repo_root, log_path=log_path, env=scorer_env)
        if return_code != 0:
            raise SystemExit(return_code)

        stats_log = resolve_path(f"outputs/logs/gedit_stats_{timestamp}.log")
        stats_command = [
            python_executable,
            str(gedit_root / "calculate_statistics.py"),
            "--model_name",
            model_name,
            "--backbone",
            config["scoring"].get("backbone", "gpt4o"),
            "--save_path",
            str(save_dir),
            "--language",
            config["dataset"].get("instruction_language", "all"),
        ]
        return_code = run_and_tee(stats_command, cwd=repo_root, log_path=stats_log, env=scorer_env)
        if return_code != 0:
            raise SystemExit(return_code)
    finally:
        if had_existing_secret and original_secret is not None:
            local_secret.write_text(original_secret, encoding="utf-8")
        elif local_secret.exists():
            local_secret.unlink()

    save_json(
        {
            **base_run_metadata(),
            "benchmark": "gedit",
            "config_path": config["_config_path"],
            "model_name": model_name,
            "score_dir": str(save_dir),
            "score_log": str(log_path),
            "stats_log": str(stats_log),
            "metrics": summarize_gedit(save_dir, model_name, config["scoring"].get("backbone", "gpt4o")),
        },
        save_dir / f"{model_name}_summary.json",
    )


if __name__ == "__main__":
    main()
