from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import ensure_dir, resolve_path
from qwen_edit_project.utils.prompting import polish_prompt
from qwen_edit_project.utils.qwen_pipeline import load_qwen_edit_pipeline, render_edit
from qwen_edit_project.utils.run_metadata import base_run_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ImgEdit benchmark images.")
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

    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    edit_json_path = resolve_path(dataset_cfg["edit_json"])
    origin_root = resolve_path(dataset_cfg["origin_img_root"])
    if edit_json_path is None or origin_root is None:
        raise ValueError("ImgEdit dataset paths must resolve")
    with edit_json_path.open("r", encoding="utf-8") as handle:
        edit_specs = json.load(handle)

    items = list(edit_specs.items())
    if args.offset:
        items = items[args.offset :]
    if args.limit is not None:
        items = items[: args.limit]

    pipe = load_qwen_edit_pipeline(
        model_id_with_origin_paths=model_cfg["model_id_with_origin_paths"],
        checkpoint_path=model_cfg.get("checkpoint_path"),
        model_type=model_cfg.get("model_type", "base"),
        device=args.device,
        torch_dtype=model_cfg.get("torch_dtype", "auto"),
    )
    output_root = ensure_dir(resolve_path(config["output"]["edited_images_dir"])) / model_cfg["model_name"]

    generation = dict(config["generation"])
    written = 0
    skipped = 0
    failed = 0
    failures: list[dict[str, str]] = []
    for key, item in items:
        out_path = output_root / f"{key}.png"
        if out_path.exists() and not args.no_resume:
            skipped += 1
            continue
        input_image_path = origin_root / item["id"]
        prompt = polish_prompt(
            item["prompt"],
            use_prompt_polish=config.get("prompting", {}).get("use_prompt_polish", False),
        )
        with Image.open(input_image_path) as image:
            generation["width"], generation["height"] = image.size
        try:
            output = render_edit(pipe, prompt, [input_image_path], generation)
            image = output.images[0] if hasattr(output, "images") else output
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            image.save(tmp_path)
            tmp_path.replace(out_path)
        except Exception as exc:
            failed += 1
            failures.append({"key": str(key), "error": repr(exc)})
            print(f"Failed ImgEdit export for {key}: {exc}", flush=True)
            continue
        written += 1
        done = written + skipped + failed
        if done % int(config["output"].get("progress_every", 25)) == 0:
            print(
                f"ImgEdit export progress: processed={done}/{len(items)} "
                f"written={written} skipped={skipped} failed={failed}",
                flush=True,
            )

    summary = base_run_metadata()
    summary.update(
        {
            "benchmark": "imgedit",
            "config_path": config["_config_path"],
            "model_name": model_cfg["model_name"],
            "checkpoint_path": model_cfg.get("checkpoint_path"),
            "records_exported": written,
            "records_skipped_existing": skipped,
            "records_failed": failed,
            "records_requested": len(items),
            "output_root": str(output_root),
            "failures": failures,
        }
    )
    summary_path = resolve_path(config["output"]["summary_path"])
    if summary_path is None:
        raise ValueError("output.summary_path must resolve")
    save_json(summary, summary_path.parent / f"{model_cfg['model_name']}_summary.json")
    if failures:
        failure_path = summary_path.parent / f"{model_cfg['model_name']}_export_failures.jsonl"
        with failure_path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=True) + "\n")
    print(f"Exported {written} ImgEdit images to {output_root}")


if __name__ == "__main__":
    main()
