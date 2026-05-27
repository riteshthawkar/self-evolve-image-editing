from __future__ import annotations

import argparse
import json

from qwen_edit_project.eval.generation_common import (
    build_sample_grid,
    generate_prompt_samples,
    load_jsonl_records,
)
from qwen_edit_project.eval.export_provenance import (
    build_generation_export_provenance,
    validate_resume_provenance,
    write_export_provenance,
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
    parser.add_argument("--no-resume", action="store_true", help="Regenerate prompts even when complete output files already exist.")
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
    model_name = model_cfg["model_name"]
    output_root = ensure_dir(ensure_dir(resolve_path(config["output"]["image_root"])) / model_name)
    generation = dict(config["generation"])
    sample_count = int(generation.get("samples_per_prompt", 1))
    summary_path = resolve_path(config["output"]["summary_path"])
    if summary_path is None:
        raise ValueError("output.summary_path must resolve")
    actual_summary_path = summary_path.parent / f"{model_name}_summary.json"
    export_provenance = build_generation_export_provenance(config)
    validate_resume_provenance(
        benchmark="GenEval",
        output_root=output_root,
        summary_path=actual_summary_path,
        expected=export_provenance,
        no_resume=args.no_resume,
        existing_file_patterns=("*.png",),
    )
    write_export_provenance(output_root, export_provenance)

    pending_records: list[tuple[int, dict[str, object]]] = []
    skipped = 0
    for index, record in enumerate(records):
        prompt_dir = output_root / f"{index:05d}"
        sample_dir = prompt_dir / "samples"
        complete = (
            (prompt_dir / "grid.png").exists()
            and (prompt_dir / "metadata.jsonl").exists()
            and all((sample_dir / f"{sample_index:04d}.png").exists() for sample_index in range(sample_count))
        )
        if complete and not args.no_resume:
            skipped += 1
        else:
            pending_records.append((index, record))

    pipe = load_qwen_generation_pipeline(
        model_id_with_origin_paths=model_cfg["model_id_with_origin_paths"],
        checkpoint_path=model_cfg.get("checkpoint_path"),
        model_type=model_cfg.get("model_type", "base"),
        device=args.device,
        tokenizer_model_id=model_cfg.get("tokenizer_model_id", model_cfg.get("base_model", "Qwen/Qwen-Image")),
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
    )

    failures: list[dict[str, object]] = []
    written = 0

    for index, record in pending_records:
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
        written += 1

    summary = base_run_metadata()
    summary.update(
        {
            "benchmark": "geneval",
            "config_path": config["_config_path"],
            "model_name": model_name,
            "checkpoint_path": model_cfg.get("checkpoint_path"),
            "records_exported": written,
            "records_skipped_existing": skipped,
            "records_requested": len(records),
            "samples_per_prompt": generation.get("samples_per_prompt", 1),
            "output_root": str(output_root),
            "failures": failures,
            "export_provenance": export_provenance,
        }
    )
    save_json(summary, actual_summary_path)
    print(f"Exported {written} GenEval prompts to {output_root} (skipped {skipped} existing)")


if __name__ == "__main__":
    main()
