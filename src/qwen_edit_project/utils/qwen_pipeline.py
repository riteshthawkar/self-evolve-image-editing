from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from .device import resolve_torch_device, resolve_torch_dtype
from .paths import resolve_path


def _load_diffsynth_modules():
    import torch
    from diffsynth import load_state_dict
    from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline

    return torch, load_state_dict, ModelConfig, QwenImagePipeline


def parse_model_configs(model_id_with_origin_paths: str, model_config_cls):
    configs = []
    for part in model_id_with_origin_paths.split(","):
        item = part.strip()
        if not item:
            continue
        model_id, origin_pattern = item.split(":", 1)
        configs.append(model_config_cls(model_id=model_id, origin_file_pattern=origin_pattern))
    return configs


def load_qwen_edit_pipeline(
    model_id_with_origin_paths: str,
    checkpoint_path: str | None = None,
    model_type: str = "base",
    device: str = "auto",
    processor_model_id: str = "Qwen/Qwen-Image-Edit",
    torch_dtype: str | None = "auto",
):
    torch, load_state_dict, ModelConfig, QwenImagePipeline = _load_diffsynth_modules()
    resolved_device = resolve_torch_device(device)
    resolved_dtype = resolve_torch_dtype(torch, torch_dtype, resolved_device)
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=resolved_dtype,
        device=resolved_device,
        model_configs=parse_model_configs(model_id_with_origin_paths, ModelConfig),
        tokenizer_config=None,
        processor_config=ModelConfig(model_id=processor_model_id, origin_file_pattern="processor/"),
    )

    resolved_checkpoint = resolve_path(checkpoint_path) if checkpoint_path else None
    if model_type == "lora" and resolved_checkpoint is not None:
        pipe.load_lora(pipe.dit, str(resolved_checkpoint))
    elif model_type == "full" and resolved_checkpoint is not None:
        state_dict = load_state_dict(str(resolved_checkpoint))
        pipe.dit.load_state_dict(state_dict)
    return pipe


def load_qwen_generation_pipeline(
    model_id_with_origin_paths: str,
    checkpoint_path: str | None = None,
    model_type: str = "base",
    device: str = "auto",
    tokenizer_model_id: str = "Qwen/Qwen-Image",
    torch_dtype: str | None = "auto",
):
    torch, load_state_dict, ModelConfig, QwenImagePipeline = _load_diffsynth_modules()
    resolved_device = resolve_torch_device(device)
    resolved_dtype = resolve_torch_dtype(torch, torch_dtype, resolved_device)
    pipe = QwenImagePipeline.from_pretrained(
        torch_dtype=resolved_dtype,
        device=resolved_device,
        model_configs=parse_model_configs(model_id_with_origin_paths, ModelConfig),
        tokenizer_config=ModelConfig(model_id=tokenizer_model_id, origin_file_pattern="tokenizer/"),
    )

    resolved_checkpoint = resolve_path(checkpoint_path) if checkpoint_path else None
    if model_type == "lora" and resolved_checkpoint is not None:
        pipe.load_lora(pipe.dit, str(resolved_checkpoint))
    elif model_type == "full" and resolved_checkpoint is not None:
        state_dict = load_state_dict(str(resolved_checkpoint))
        pipe.dit.load_state_dict(state_dict)
    return pipe


def normalize_edit_inputs(edit_images: list[Path | Image.Image]) -> Image.Image | list[Image.Image]:
    loaded: list[Image.Image] = []
    for item in edit_images:
        if isinstance(item, Image.Image):
            loaded.append(item.convert("RGB"))
        else:
            loaded.append(Image.open(item).convert("RGB"))
    if len(loaded) == 1:
        return loaded[0]
    return loaded


def build_generation_kwargs(generation: dict[str, Any]) -> dict[str, Any]:
    kwargs = {
        "seed": generation.get("seed", 42),
        "num_inference_steps": generation.get("num_inference_steps", 28),
        "negative_prompt": generation.get("negative_prompt", " "),
    }
    width = generation.get("width")
    height = generation.get("height")
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height
    if generation.get("guidance_scale") is not None:
        kwargs["guidance_scale"] = generation["guidance_scale"]
    if generation.get("true_cfg_scale") is not None:
        kwargs["true_cfg_scale"] = generation["true_cfg_scale"]
    return kwargs


def _calculate_qwen_edit_resize(target_area: int, ratio: float) -> tuple[int, int]:
    import math

    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


def _resize_for_qwen_edit_understanding(image: Image.Image) -> Image.Image:
    width, height = _calculate_qwen_edit_resize(1024 * 1024, image.size[0] / image.size[1])
    return image.resize((width, height))


def _extract_masked_hidden_states(hidden_states, attention_mask):
    import torch

    bool_mask = attention_mask.bool()
    valid_lengths = bool_mask.sum(dim=1)
    selected = hidden_states[bool_mask]
    return torch.split(selected, valid_lengths.tolist(), dim=0)


def _build_qwen_edit_prompt(prompt: str, num_images: int) -> tuple[str, int]:
    template = (
        "<|im_start|>system\n"
        "Describe the key features of the input image (color, shape, size, texture, objects, background), "
        "then explain how the user's text instruction should alter or modify the image. Generate a new image "
        "that meets the user's requirements while maintaining consistency with the original input where appropriate."
        "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    if num_images == 1:
        user_content = f"<|vision_start|><|image_pad|><|vision_end|>{prompt}"
    else:
        image_slots = "".join(
            f"Picture {index + 1}: <|vision_start|><|image_pad|><|vision_end|>"
            for index in range(num_images)
        )
        user_content = f"{image_slots}{prompt}"
    return template.format(user_content), 64


def _build_qwen_text_prompt(prompt: str) -> tuple[str, int]:
    template = (
        "<|im_start|>system\n"
        "Describe how the requested edit should change the image while preserving unrelated content whenever possible."
        "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
    return template.format(prompt), 0


def _pool_hidden_states(trimmed_states):
    import torch

    max_seq_len = max(state.size(0) for state in trimmed_states)
    attention_mask = torch.stack(
        [
            torch.cat(
                [
                    torch.ones(state.size(0), dtype=torch.long, device=state.device),
                    torch.zeros(max_seq_len - state.size(0), dtype=torch.long, device=state.device),
                ]
            )
            for state in trimmed_states
        ]
    )
    token_embeddings = torch.stack(
        [torch.cat([state, state.new_zeros(max_seq_len - state.size(0), state.size(1))]) for state in trimmed_states]
    )
    pooled_embedding = (
        token_embeddings * attention_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)
    ).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
    return token_embeddings, attention_mask, pooled_embedding


def extract_qwen_edit_understanding_features(
    pipe: Any,
    prompt: str,
    edit_images: list[Path | Image.Image],
) -> dict[str, Any]:
    import torch

    if getattr(pipe, "processor", None) is None or getattr(pipe, "text_encoder", None) is None:
        raise ValueError("Qwen edit understanding features require a pipeline with both processor and text_encoder loaded.")

    images = normalize_edit_inputs(edit_images)
    image_list = images if isinstance(images, list) else [images]
    processor_images = image_list if len(image_list) == 1 else [_resize_for_qwen_edit_understanding(image) for image in image_list]
    processor_input = processor_images[0] if len(processor_images) == 1 else processor_images
    text, drop_idx = _build_qwen_edit_prompt(prompt, len(image_list))
    model_inputs = pipe.processor(text=[text], images=processor_input, padding=True, return_tensors="pt").to(pipe.device)
    hidden_states = pipe.text_encoder(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        pixel_values=model_inputs.pixel_values,
        image_grid_thw=model_inputs.image_grid_thw,
        output_hidden_states=True,
    )[-1]
    split_hidden_states = _extract_masked_hidden_states(hidden_states, model_inputs.attention_mask)
    trimmed_states = [state[drop_idx:] for state in split_hidden_states]
    token_embeddings, attention_mask, pooled_embedding = _pool_hidden_states(trimmed_states)
    token_embeddings = token_embeddings.to(dtype=pipe.torch_dtype, device=pipe.device)
    pooled_embedding = pooled_embedding.to(dtype=pipe.torch_dtype, device=pipe.device)
    return {
        "token_embeddings": token_embeddings,
        "attention_mask": attention_mask,
        "pooled_embedding": pooled_embedding,
        "raw_hidden_states": trimmed_states,
        "image_count": len(image_list),
        "prompt": prompt,
    }


def extract_qwen_text_features(
    pipe: Any,
    prompt: str,
) -> dict[str, Any]:
    import torch

    tokenizer = getattr(pipe, "tokenizer", None)
    if tokenizer is None and getattr(pipe, "processor", None) is not None:
        tokenizer = getattr(pipe.processor, "tokenizer", None)
    if tokenizer is None or getattr(pipe, "text_encoder", None) is None:
        raise ValueError("Qwen text features require a pipeline with both tokenizer or processor.tokenizer and text_encoder.")

    text, drop_idx = _build_qwen_text_prompt(prompt)
    model_inputs = tokenizer([text], padding=True, return_tensors="pt").to(pipe.device)
    hidden_states = pipe.text_encoder(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        output_hidden_states=True,
    )[-1]
    split_hidden_states = _extract_masked_hidden_states(hidden_states, model_inputs.attention_mask)
    trimmed_states = [state[drop_idx:] if state.size(0) > drop_idx else state for state in split_hidden_states]
    token_embeddings, attention_mask, pooled_embedding = _pool_hidden_states(trimmed_states)
    token_embeddings = token_embeddings.to(dtype=pipe.torch_dtype, device=pipe.device)
    pooled_embedding = pooled_embedding.to(dtype=pipe.torch_dtype, device=pipe.device)
    return {
        "token_embeddings": token_embeddings,
        "attention_mask": attention_mask,
        "pooled_embedding": pooled_embedding,
        "raw_hidden_states": trimmed_states,
        "prompt": prompt,
    }


def render_edit(
    pipe: Any,
    prompt: str,
    edit_images: list[Path | Image.Image],
    generation: dict[str, Any],
) -> Image.Image:
    conditioning = normalize_edit_inputs(edit_images)
    kwargs = build_generation_kwargs(generation)
    kwargs["edit_image"] = conditioning
    return pipe(prompt, **kwargs)


def render_generation(
    pipe: Any,
    prompt: str,
    generation: dict[str, Any],
) -> Image.Image:
    return pipe(prompt, **build_generation_kwargs(generation))
