from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from qwen_edit_project.utils.commands import run_and_tee, shell_join
from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import ensure_dir, resolve_path
from qwen_edit_project.utils.run_metadata import base_run_metadata, utc_timestamp


def _append_flag(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    command.extend([flag, str(value)])


def build_train_command(config: dict[str, Any]) -> tuple[list[str], Path]:
    runtime = config["runtime"]
    mode = config["mode"]
    working_dir = resolve_path(runtime["working_dir"])
    if working_dir is None:
        raise ValueError("runtime.working_dir is required")
    train_script = working_dir / "examples/qwen_image/model_training/train.py"
    command = [runtime.get("accelerate_executable", "accelerate"), "launch"]
    accelerate_config = runtime.get("accelerate_config_file")
    if accelerate_config:
        command.extend(["--config_file", str(working_dir / accelerate_config)])
    command.append(str(train_script))

    dataset = config["dataset"]
    model = config["model"]
    training = config["training"]
    output = config["output"]

    _append_flag(command, "--dataset_base_path", resolve_path(dataset["dataset_base_path"]))
    _append_flag(command, "--dataset_metadata_path", resolve_path(dataset["dataset_metadata_path"]))
    _append_flag(command, "--data_file_keys", dataset["data_file_keys"])
    _append_flag(command, "--extra_inputs", dataset.get("extra_inputs"))
    _append_flag(command, "--max_pixels", dataset.get("max_pixels"))
    _append_flag(command, "--dataset_repeat", dataset.get("dataset_repeat"))
    _append_flag(command, "--dataset_num_workers", dataset.get("dataset_num_workers"))
    _append_flag(command, "--height", training.get("height"))
    _append_flag(command, "--width", training.get("width"))
    _append_flag(command, "--gradient_accumulation_steps", training.get("gradient_accumulation_steps"))

    _append_flag(command, "--model_id_with_origin_paths", model["model_id_with_origin_paths"])
    _append_flag(command, "--tokenizer_path", model.get("tokenizer_path"))
    _append_flag(command, "--processor_path", model.get("processor_path"))
    _append_flag(command, "--zero_cond_t", model.get("zero_cond_t", False))
    _append_flag(command, "--remove_prefix_in_ckpt", model.get("remove_prefix_in_ckpt"))

    _append_flag(command, "--learning_rate", training.get("learning_rate"))
    _append_flag(command, "--num_epochs", training.get("num_epochs"))
    _append_flag(command, "--use_gradient_checkpointing", training.get("use_gradient_checkpointing", False))
    _append_flag(command, "--find_unused_parameters", training.get("find_unused_parameters", False))
    _append_flag(command, "--output_path", resolve_path(output["output_path"]))

    if mode == "lora":
        lora = config["lora"]
        _append_flag(command, "--lora_base_model", lora["lora_base_model"])
        _append_flag(command, "--lora_target_modules", lora["lora_target_modules"])
        _append_flag(command, "--lora_rank", lora["lora_rank"])
        _append_flag(command, "--lora_checkpoint", lora.get("lora_checkpoint"))
        _append_flag(command, "--preset_lora_path", lora.get("preset_lora_path"))
        _append_flag(command, "--preset_lora_model", lora.get("preset_lora_model"))
    elif mode == "full":
        _append_flag(command, "--trainable_models", training.get("trainable_models"))
    else:
        raise ValueError(f"Unsupported train mode: {mode}")

    if config.get("resume", {}).get("enabled"):
        for value in config["resume"].get("extra_args", []):
            command.append(str(value))

    for value in config.get("extra_args", []):
        command.append(str(value))

    return command, working_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch DiffSynth training from YAML config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--set", action="append", default=[], help="Override config using dotted.key=value")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    command, working_dir = build_train_command(config)
    output = config["output"]
    command_file = resolve_path(output["command_file"])
    log_dir = ensure_dir(resolve_path(output["log_dir"]))
    if command_file is None:
        raise ValueError("output.command_file is required")
    ensure_dir(command_file.parent)
    command_file.write_text(shell_join(command) + "\n", encoding="utf-8")

    timestamp = utc_timestamp()
    log_path = log_dir / f"{config['name']}_{timestamp}.log"
    metadata_path = log_dir / f"{config['name']}_{timestamp}.json"
    metadata = base_run_metadata()
    metadata.update(
        {
            "config_path": config["_config_path"],
            "command": command,
            "command_shell": shell_join(command),
            "working_dir": str(working_dir),
            "log_path": str(log_path),
            "mode": config["mode"],
            "output_path": str(resolve_path(output["output_path"])),
        }
    )
    save_json(metadata, metadata_path)

    print(shell_join(command))
    if args.dry_run or config["runtime"].get("dry_run", False):
        print(f"Dry run. Command file written to {command_file}")
        return

    return_code = run_and_tee(command, cwd=working_dir, log_path=log_path)
    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()

