from __future__ import annotations

import argparse
import json

from qwen_edit_project.eval.generation_common import (
    build_sample_grid,
    generate_prompt_samples,
    load_jsonl_records,
)
from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import ensure_dir, resolve_path
from qwen_edit_project.utils.qwen_pipeline import load_qwen_generation_pipeline
from qwen_edit_project.utils.run_metadata import base_run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GenEval benchmark images.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    dataset_path = resolve_path(config["dataset"]["metadata_jsonl"])
    if dataset_path is None:
        raise ValueError("dataset.metadata_jsonl must resolve")
    records = load_jsonl_records(dataset_path)
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
    output_root = ensure_dir(ensure_dir(resolve_path(config["output"]["image_root"])) / model_name)
    generation = dict(config["generation"])
    failures: list[dict[str, object]] = []

    for index, record in enumerate(records):
        prompt_dir = ensure_dir(output_root / f"{index:05d}")
        sample_dir = ensure_dir(prompt_dir / "samples")
        images, errors = generate_prompt_samples(pipe, record["prompt"], generation, prompt_index=index)
        for sample_index, image in enumerate(images):
            image.save(sample_dir / f"{sample_index:04d}.png")
        build_sample_grid(images, generation).save(prompt_dir / "grid.png")
        with (prompt_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        if errors:
            failures.append({"prompt_index": index, "errors": errors})

    summary = base_run_metadata()
    summary.update(
        {
            "benchmark": "geneval",
            "config_path": config["_config_path"],
            "model_name": model_name,
            "checkpoint_path": model_cfg.get("checkpoint_path"),
            "records_exported": len(records),
            "samples_per_prompt": generation.get("samples_per_prompt", 1),
            "output_root": str(output_root),
            "failures": failures,
        }
    )
    summary_path = resolve_path(config["output"]["summary_path"])
    if summary_path is None:
        raise ValueError("output.summary_path must resolve")
    save_json(summary, summary_path.parent / f"{model_name}_summary.json")
    print(f"Exported {len(records)} GenEval prompts to {output_root}")


if __name__ == "__main__":
    main()
