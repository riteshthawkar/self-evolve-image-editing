from __future__ import annotations

import argparse
import csv
from collections import Counter

from qwen_edit_project.eval.generation_common import build_sample_grid, generate_prompt_samples
from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import ensure_dir, resolve_path
from qwen_edit_project.utils.qwen_pipeline import load_qwen_generation_pipeline
from qwen_edit_project.utils.run_metadata import base_run_metadata

ONEIG_CATEGORY_DIRS = {
    "Anime_Stylization": "anime",
    "Portrait": "human",
    "General_Object": "object",
    "Text_Rendering": "text",
    "Knowledge_Reasoning": "reasoning",
    "Multilingualism": "multilingualism",
}


def load_oneig_records(dataset_cfg: dict[str, object]) -> list[dict[str, str]]:
    mode = str(dataset_cfg.get("mode", "EN")).upper()
    csv_key = "csv_zh" if mode == "ZH" else "csv_en"
    csv_path = resolve_path(str(dataset_cfg[csv_key]))
    if csv_path is None:
        raise ValueError(f"{csv_key} must resolve")

    prompt_field = "prompt_cn" if mode == "ZH" else "prompt_en"
    category_filter = {str(item) for item in dataset_cfg.get("categories", [])}
    records: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            category = row["category"].strip()
            if category_filter and category not in category_filter:
                continue
            prompt = row.get(prompt_field) or row.get("prompt_en")
            if not prompt:
                continue
            records.append(
                {
                    "id": row["id"].strip(),
                    "category": category,
                    "prompt": prompt.strip(),
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Export OneIG-Bench images.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    records = load_oneig_records(config["dataset"])
    if args.limit is not None:
        records = records[: args.limit]

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
    mode = str(config["dataset"].get("mode", "EN")).lower()
    output_root = ensure_dir(ensure_dir(resolve_path(config["output"]["image_root"])) / mode)
    generation = dict(config["generation"])
    failures: list[dict[str, object]] = []
    counts = Counter()

    for index, record in enumerate(records):
        category_dir = ONEIG_CATEGORY_DIRS[record["category"]]
        output_dir = ensure_dir(output_root / category_dir / model_name)
        images, errors = generate_prompt_samples(pipe, record["prompt"], generation, prompt_index=index)
        build_sample_grid(images, generation).save(output_dir / f"{record['id']}.webp", format="WEBP")
        counts[category_dir] += 1
        if errors:
            failures.append({"id": record["id"], "errors": errors})

    summary = base_run_metadata()
    summary.update(
        {
            "benchmark": "oneig",
            "config_path": config["_config_path"],
            "model_name": model_name,
            "checkpoint_path": model_cfg.get("checkpoint_path"),
            "records_exported": len(records),
            "samples_per_prompt": generation.get("samples_per_prompt", 1),
            "output_root": str(output_root),
            "mode": mode,
            "counts_by_category": dict(counts),
            "failures": failures,
        }
    )
    summary_path = resolve_path(config["output"]["summary_path"])
    if summary_path is None:
        raise ValueError("output.summary_path must resolve")
    save_json(summary, summary_path.parent / f"{model_name}_summary.json")
    print(f"Exported {len(records)} OneIG-Bench prompts to {output_root}")


if __name__ == "__main__":
    main()
