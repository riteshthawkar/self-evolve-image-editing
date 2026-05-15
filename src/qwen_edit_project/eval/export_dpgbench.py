from __future__ import annotations

import argparse
import csv

from qwen_edit_project.eval.generation_common import build_sample_grid, generate_prompt_samples
from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import ensure_dir, resolve_path
from qwen_edit_project.utils.qwen_pipeline import load_qwen_generation_pipeline
from qwen_edit_project.utils.run_metadata import base_run_metadata


def load_dpg_prompts(csv_path) -> list[dict[str, str]]:
    prompts: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            item_id = row["item_id"].strip()
            if not item_id or item_id in seen_ids:
                continue
            prompts.append({"item_id": item_id, "prompt": row["text"].strip()})
            seen_ids.add(item_id)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DPG-Bench images.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    prompts_csv = resolve_path(config["dataset"]["prompts_csv"])
    if prompts_csv is None:
        raise ValueError("dataset.prompts_csv must resolve")
    prompts = load_dpg_prompts(prompts_csv)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    model_cfg = config["model"]
    pipe = load_qwen_generation_pipeline(
        model_id_with_origin_paths=model_cfg["model_id_with_origin_paths"],
        checkpoint_path=model_cfg.get("checkpoint_path"),
        model_type=model_cfg.get("model_type", "base"),
        device=args.device,
        tokenizer_model_id=model_cfg.get("tokenizer_model_id", model_cfg.get("base_model", "Qwen/Qwen-Image")),
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
    )

    model_name = model_cfg["model_name"]
    output_root = ensure_dir(resolve_path(config["output"]["image_root"])) / model_name
    generation = dict(config["generation"])
    failures: list[dict[str, object]] = []

    for index, item in enumerate(prompts):
        images, errors = generate_prompt_samples(pipe, item["prompt"], generation, prompt_index=index)
        build_sample_grid(images, generation).save(output_root / f"{item['item_id']}.png")
        if errors:
            failures.append({"item_id": item["item_id"], "errors": errors})

    summary = base_run_metadata()
    summary.update(
        {
            "benchmark": "dpgbench",
            "config_path": config["_config_path"],
            "model_name": model_name,
            "checkpoint_path": model_cfg.get("checkpoint_path"),
            "records_exported": len(prompts),
            "samples_per_prompt": generation.get("samples_per_prompt", 1),
            "output_root": str(output_root),
            "failures": failures,
        }
    )
    summary_path = resolve_path(config["output"]["summary_path"])
    if summary_path is None:
        raise ValueError("output.summary_path must resolve")
    save_json(summary, summary_path.parent / f"{model_name}_summary.json")
    print(f"Exported {len(prompts)} DPG-Bench prompts to {output_root}")


if __name__ == "__main__":
    main()
