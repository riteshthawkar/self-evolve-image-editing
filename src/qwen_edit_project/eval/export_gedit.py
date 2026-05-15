from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import ensure_dir, resolve_path
from qwen_edit_project.utils.prompting import polish_prompt
from qwen_edit_project.utils.qwen_pipeline import load_qwen_edit_pipeline, render_edit
from qwen_edit_project.utils.run_metadata import base_run_metadata


def load_gedit_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_cfg = config["dataset"]
    source = dataset_cfg.get("source", "huggingface")
    if source == "huggingface":
        from datasets import load_dataset

        dataset = load_dataset(dataset_cfg["dataset_name"], split=dataset_cfg.get("split", "train"))
    elif source == "disk":
        from datasets import load_from_disk

        local_path = resolve_path(dataset_cfg["local_path"])
        if local_path is None:
            raise ValueError("dataset.local_path is required when source=disk")
        dataset = load_from_disk(str(local_path))
    else:
        raise ValueError(f"Unsupported GEdit dataset source: {source}")

    language_filter = dataset_cfg.get("instruction_language", "all")
    task_filter = dataset_cfg.get("task_type", "all")
    records: list[dict[str, Any]] = []
    for item in dataset:
        if language_filter != "all" and item["instruction_language"] != language_filter:
            continue
        if task_filter != "all" and item["task_type"] != task_filter:
            continue
        records.append(item)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GEdit benchmark images.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    records = load_gedit_records(config)
    if args.limit is not None:
        records = records[: args.limit]

    model_cfg = config["model"]
    model_name = model_cfg["model_name"]
    pipe = load_qwen_edit_pipeline(
        model_id_with_origin_paths=model_cfg["model_id_with_origin_paths"],
        checkpoint_path=model_cfg.get("checkpoint_path"),
        model_type=model_cfg.get("model_type", "base"),
        device=args.device,
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
    )

    output_root = ensure_dir(resolve_path(config["output"]["edited_images_dir"])) / model_name / "fullset"
    generation = dict(config["generation"])
    written = 0
    for item in records:
        prompt = polish_prompt(
            item["instruction"],
            use_prompt_polish=config.get("prompting", {}).get("use_prompt_polish", False),
            image_context=item["input_image_raw"],
        )
        generation["width"], generation["height"] = item["input_image_raw"].size
        output = render_edit(pipe, prompt, [item["input_image_raw"]], generation)
        image = output.images[0] if hasattr(output, "images") else output
        out_dir = ensure_dir(output_root / item["task_type"] / item["instruction_language"])
        image.save(out_dir / f"{item['key']}.png")
        written += 1

    summary = base_run_metadata()
    summary.update(
        {
            "benchmark": "gedit",
            "config_path": config["_config_path"],
            "model_name": model_name,
            "checkpoint_path": model_cfg.get("checkpoint_path"),
            "records_exported": written,
            "output_root": str(output_root),
        }
    )
    summary_path = resolve_path(config["output"]["summary_path"])
    if summary_path is None:
        raise ValueError("output.summary_path must resolve")
    save_json(summary, summary_path.parent / f"{model_name}_summary.json")
    print(f"Exported {written} GEdit images to {output_root}")


if __name__ == "__main__":
    main()
