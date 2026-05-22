from __future__ import annotations

import argparse
from pathlib import Path

from qwen_edit_project.utils.config import load_yaml_config, merge_override, parse_override, save_json
from qwen_edit_project.utils.paths import ensure_dir, resolve_path
from qwen_edit_project.utils.prompting import polish_prompt
from qwen_edit_project.utils.qwen_pipeline import load_qwen_edit_pipeline, render_edit
from qwen_edit_project.utils.run_metadata import base_run_metadata, utc_timestamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run smoke validation against a checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=["lora", "full"], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--set", action="append", default=[])
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    for raw in args.set:
        key, value = parse_override(raw)
        config = merge_override(config, key, value)

    validation = config["validation"]
    edit_images = [resolve_path(path) for path in validation.get("edit_images", [])]
    if not edit_images or any(path is None for path in edit_images):
        raise ValueError("validation.edit_images must contain at least one resolvable image path")

    prompt = polish_prompt(
        validation["prompt"],
        use_prompt_polish=config.get("prompting", {}).get("use_prompt_polish", False),
    )
    generation = {
        "seed": validation.get("seed", 123),
        "num_inference_steps": validation.get("num_inference_steps", 40),
        "width": validation.get("width"),
        "height": validation.get("height"),
        "negative_prompt": validation.get("negative_prompt", " "),
    }
    model = config["model"]
    pipe = load_qwen_edit_pipeline(
        model_id_with_origin_paths=model["model_id_with_origin_paths"],
        checkpoint_path=args.checkpoint,
        model_type=args.mode,
        device=args.device,
        torch_dtype=model.get("torch_dtype", "auto"),
        backend=model.get("backend", "diffsynth"),
        base_model=model.get("pretrained_model_name_or_path") or model.get("base_model"),
    )
    output = render_edit(pipe, prompt, [Path(item) for item in edit_images if item is not None], generation)
    image = output.images[0] if hasattr(output, "images") else output
    output_dir = ensure_dir(resolve_path("outputs/validation"))
    timestamp = utc_timestamp()
    output_name = args.output_name or f"{config['name']}_{args.mode}_{timestamp}.png"
    image_path = output_dir / output_name
    image.save(image_path)

    metadata = base_run_metadata()
    metadata.update(
        {
            "config_path": config["_config_path"],
            "checkpoint_path": str(resolve_path(args.checkpoint)),
            "mode": args.mode,
            "prompt": prompt,
            "seed": generation["seed"],
            "image_path": str(image_path),
            "edit_images": [str(path) for path in edit_images],
        }
    )
    save_json(metadata, image_path.with_suffix(".json"))
    print(f"Saved validation image to {image_path}")


if __name__ == "__main__":
    main()
