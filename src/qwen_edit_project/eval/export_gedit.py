from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.eval.export_provenance import (
    build_edit_export_provenance,
    validate_resume_provenance,
    write_export_provenance,
)
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
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true", help="Regenerate images even when output files already exist.")
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    records = load_gedit_records(config)
    if args.offset:
        records = records[args.offset :]
    if args.limit is not None:
        records = records[: args.limit]

    model_cfg = config["model"]
    model_name = model_cfg["model_name"]
    output_base = ensure_dir(resolve_path(config["output"]["edited_images_dir"]))
    output_root = output_base / model_name / "fullset"
    summary_path = resolve_path(config["output"]["summary_path"])
    if summary_path is None:
        raise ValueError("output.summary_path must resolve")
    actual_summary_path = summary_path.parent / f"{model_name}_summary.json"
    export_provenance = build_edit_export_provenance(config)
    validate_resume_provenance(
        benchmark="GEdit",
        output_root=output_root,
        summary_path=actual_summary_path,
        expected=export_provenance,
        no_resume=args.no_resume,
        allow_mismatch=bool(config["output"].get("allow_resume_mismatch", False)),
    )
    ensure_dir(output_root)
    write_export_provenance(output_root, export_provenance)
    pipe = load_qwen_edit_pipeline(
        model_id_with_origin_paths=model_cfg["model_id_with_origin_paths"],
        checkpoint_path=model_cfg.get("checkpoint_path"),
        model_type=model_cfg.get("model_type", "base"),
        device=args.device,
        processor_model_id=model_cfg.get("processor_model_id", "Qwen/Qwen-Image-Edit"),
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
        backend=model_cfg.get("backend", "diffsynth"),
        base_model=model_cfg.get("base_model"),
        local_files_only=bool(model_cfg.get("local_files_only", False)),
        lora_scale=model_cfg.get("lora_scale"),
    )

    generation = dict(config["generation"])
    preserve_input_resolution = bool(generation.get("preserve_input_resolution", True))
    written = 0
    skipped = 0
    failed = 0
    failures: list[dict[str, str]] = []
    for item in records:
        out_dir = ensure_dir(output_root / item["task_type"] / item["instruction_language"])
        out_path = out_dir / f"{item['key']}.png"
        if out_path.exists() and not args.no_resume:
            skipped += 1
            continue
        prompt = polish_prompt(
            item["instruction"],
            use_prompt_polish=config.get("prompting", {}).get("use_prompt_polish", False),
            image_context=item["input_image_raw"],
        )
        if preserve_input_resolution:
            generation["width"], generation["height"] = item["input_image_raw"].size
        else:
            generation.pop("width", None)
            generation.pop("height", None)
        try:
            output = render_edit(pipe, prompt, [item["input_image_raw"]], generation)
            image = output.images[0] if hasattr(output, "images") else output
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            image.save(tmp_path, format="PNG")
            tmp_path.replace(out_path)
        except Exception as exc:
            failed += 1
            failures.append({"key": str(item["key"]), "error": repr(exc)})
            print(f"Failed GEdit export for {item['key']}: {exc}", flush=True)
            continue
        written += 1
        done = written + skipped + failed
        if done % int(config["output"].get("progress_every", 25)) == 0:
            print(
                f"GEdit export progress: processed={done}/{len(records)} "
                f"written={written} skipped={skipped} failed={failed}",
                flush=True,
            )

    summary = base_run_metadata()
    summary.update(
        {
            "benchmark": "gedit",
            "config_path": config["_config_path"],
            "model_name": model_name,
            "checkpoint_path": model_cfg.get("checkpoint_path"),
            "records_exported": written,
            "records_skipped_existing": skipped,
            "records_failed": failed,
            "records_requested": len(records),
            "output_root": str(output_root),
            "failures": failures,
            "export_provenance": export_provenance,
        }
    )
    save_json(summary, actual_summary_path)
    if failures:
        failure_path = summary_path.parent / f"{model_name}_export_failures.jsonl"
        with failure_path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=True) + "\n")
    print(f"Exported {written} GEdit images to {output_root}")


if __name__ == "__main__":
    main()
