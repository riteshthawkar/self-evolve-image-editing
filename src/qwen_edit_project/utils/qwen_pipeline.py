from __future__ import annotations

import inspect
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
    backend: str = "diffsynth",
    base_model: str | None = None,
    local_files_only: bool = False,
):
    if backend in {"diffusers", "official_diffusers", "qwen_edit_plus"}:
        return load_qwen_edit_plus_pipeline(
            base_model=base_model or _first_model_id(model_id_with_origin_paths),
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            device=device,
            torch_dtype=torch_dtype,
            local_files_only=local_files_only,
        )

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
    pipe._qwen_edit_backend = "diffsynth"
    return pipe


def _first_model_id(model_id_with_origin_paths: str) -> str:
    for part in model_id_with_origin_paths.split(","):
        item = part.strip()
        if item:
            return item.split(":", 1)[0]
    return "Qwen/Qwen-Image-Edit-2509"


def _from_pretrained_with_dtype(pipeline_cls: Any, model_id: str, dtype: Any, local_files_only: bool) -> Any:
    try:
        return pipeline_cls.from_pretrained(
            model_id,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        )
    except TypeError as exc:
        if "torch_dtype" not in str(exc):
            raise
    return pipeline_cls.from_pretrained(
        model_id,
        dtype=dtype,
        local_files_only=local_files_only,
    )


def load_qwen_edit_plus_pipeline(
    base_model: str = "Qwen/Qwen-Image-Edit-2509",
    checkpoint_path: str | None = None,
    model_type: str = "base",
    device: str = "auto",
    torch_dtype: str | None = "auto",
    local_files_only: bool = False,
):
    import torch

    try:
        from diffusers import QwenImageEditPlusPipeline
    except ImportError:
        from diffusers import DiffusionPipeline

        pipeline_cls = DiffusionPipeline
    else:
        pipeline_cls = QwenImageEditPlusPipeline

    resolved_device = resolve_torch_device(device)
    resolved_dtype = resolve_torch_dtype(torch, torch_dtype, resolved_device)
    pipe = _from_pretrained_with_dtype(pipeline_cls, base_model, resolved_dtype, local_files_only)
    if hasattr(pipe, "to"):
        pipe.to(resolved_device)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=None)

    resolved_checkpoint = resolve_path(checkpoint_path) if checkpoint_path else None
    if resolved_checkpoint is not None:
        if model_type == "lora" and hasattr(pipe, "load_lora_weights"):
            pipe.load_lora_weights(str(resolved_checkpoint))
        else:
            raise ValueError(
                "The official Diffusers QwenImageEditPlusPipeline backend is only validated for "
                "base-model evaluation or Diffusers-compatible LoRA weights. Use model.backend=diffsynth "
                "for DiffSynth-trained full checkpoints."
            )
    pipe._qwen_edit_backend = "official_diffusers"
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
    if generation.get("num_images_per_prompt") is not None:
        kwargs["num_images_per_prompt"] = generation["num_images_per_prompt"]
    return kwargs


def _filter_pipeline_kwargs(pipe: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep configs portable across DiffSynth releases with different call signatures."""
    try:
        signature = inspect.signature(pipe.__call__)
    except (TypeError, ValueError):
        return kwargs
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    allowed = set(parameters)
    return {key: value for key, value in kwargs.items() if key in allowed}


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


def _build_qwen_edit_prompt(pipe: Any, prompt: str, num_images: int) -> tuple[str, int]:
    if hasattr(pipe, "prompt_template_encode") and hasattr(pipe, "prompt_template_encode_start_idx"):
        image_slots = "".join(
            f"Picture {index + 1}: <|vision_start|><|image_pad|><|vision_end|>"
            for index in range(num_images)
        )
        return pipe.prompt_template_encode.format(image_slots + prompt), int(pipe.prompt_template_encode_start_idx)

    template = (
        "<|im_start|>system\n"
        "Describe the key features of the input image (color, shape, size, texture, objects, background), "
        "then explain how the user's text instruction should alter or modify the image. Generate a new image "
        "that meets the user's requirements while maintaining consistency with the original input where appropriate."
        "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )
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
    text, drop_idx = _build_qwen_edit_prompt(pipe, prompt, len(image_list))
    device = getattr(pipe, "device", getattr(pipe, "_execution_device", "cpu"))
    dtype = getattr(pipe, "torch_dtype", getattr(getattr(pipe, "text_encoder", None), "dtype", None))
    model_inputs = pipe.processor(text=[text], images=processor_input, padding=True, return_tensors="pt").to(device)
    outputs = pipe.text_encoder(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        pixel_values=model_inputs.pixel_values,
        image_grid_thw=model_inputs.image_grid_thw,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs[-1]
    if isinstance(hidden_states, (tuple, list)):
        hidden_states = hidden_states[-1]
    split_hidden_states = _extract_masked_hidden_states(hidden_states, model_inputs.attention_mask)
    trimmed_states = [state[drop_idx:] for state in split_hidden_states]
    token_embeddings, attention_mask, pooled_embedding = _pool_hidden_states(trimmed_states)
    if dtype is not None:
        token_embeddings = token_embeddings.to(dtype=dtype, device=device)
        pooled_embedding = pooled_embedding.to(dtype=dtype, device=device)
    else:
        token_embeddings = token_embeddings.to(device=device)
        pooled_embedding = pooled_embedding.to(device=device)
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
    device = getattr(pipe, "device", getattr(pipe, "_execution_device", "cpu"))
    dtype = getattr(pipe, "torch_dtype", getattr(getattr(pipe, "text_encoder", None), "dtype", None))
    model_inputs = tokenizer([text], padding=True, return_tensors="pt").to(device)
    outputs = pipe.text_encoder(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        output_hidden_states=True,
    )
    hidden_states = outputs.hidden_states[-1] if hasattr(outputs, "hidden_states") else outputs[-1]
    if isinstance(hidden_states, (tuple, list)):
        hidden_states = hidden_states[-1]
    split_hidden_states = _extract_masked_hidden_states(hidden_states, model_inputs.attention_mask)
    trimmed_states = [state[drop_idx:] if state.size(0) > drop_idx else state for state in split_hidden_states]
    token_embeddings, attention_mask, pooled_embedding = _pool_hidden_states(trimmed_states)
    if dtype is not None:
        token_embeddings = token_embeddings.to(dtype=dtype, device=device)
        pooled_embedding = pooled_embedding.to(dtype=dtype, device=device)
    else:
        token_embeddings = token_embeddings.to(device=device)
        pooled_embedding = pooled_embedding.to(device=device)
    return {
        "token_embeddings": token_embeddings,
        "attention_mask": attention_mask,
        "pooled_embedding": pooled_embedding,
        "raw_hidden_states": trimmed_states,
        "prompt": prompt,
    }


def _image_to_vae_tensor(
    pipe: Any,
    image: Image.Image,
    size: int,
    device: Any,
    dtype: Any,
):
    import numpy as np
    import torch

    image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    processor = getattr(pipe, "image_processor", None)
    if processor is not None and hasattr(processor, "preprocess"):
        try:
            return processor.preprocess(image).to(device=device, dtype=dtype)
        except Exception:
            pass

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(device=device, dtype=dtype)


def extract_qwen_vae_latents(
    pipe: Any,
    image: Image.Image,
    size: int = 512,
):
    """Encode an image with the editor's own VAE for internal preservation rewards."""
    import torch

    vae = getattr(pipe, "vae", None)
    if vae is None or not hasattr(vae, "encode"):
        raise ValueError("Qwen VAE latents require a pipeline with a VAE encoder.")

    device = getattr(pipe, "device", getattr(pipe, "_execution_device", "cpu"))
    dtype = getattr(pipe, "torch_dtype", getattr(vae, "dtype", torch.float32))
    if dtype is None:
        dtype = torch.float32
    pixel_values = _image_to_vae_tensor(pipe, image, size=size, device=device, dtype=dtype)
    with torch.no_grad():
        encoded = vae.encode(pixel_values)

    latent_dist = getattr(encoded, "latent_dist", None)
    if latent_dist is not None:
        if hasattr(latent_dist, "mean"):
            mean = latent_dist.mean
            latents = mean() if callable(mean) else mean
        else:
            latents = latent_dist.mode()
    elif hasattr(encoded, "latents"):
        latents = encoded.latents
    elif isinstance(encoded, (tuple, list)):
        latents = encoded[0]
        nested_dist = getattr(latents, "latent_dist", None)
        if nested_dist is not None:
            if hasattr(nested_dist, "mean"):
                nested_mean = nested_dist.mean
                latents = nested_mean() if callable(nested_mean) else nested_mean
            else:
                latents = nested_dist.mode()
    else:
        latents = encoded

    scaling_factor = getattr(getattr(vae, "config", None), "scaling_factor", None)
    if scaling_factor is not None:
        latents = latents * scaling_factor
    return latents.float()


def render_edit(
    pipe: Any,
    prompt: str,
    edit_images: list[Path | Image.Image],
    generation: dict[str, Any],
) -> Image.Image:
    conditioning = normalize_edit_inputs(edit_images)
    kwargs = build_generation_kwargs(generation)
    backend = getattr(pipe, "_qwen_edit_backend", "diffsynth")
    if backend == "official_diffusers":
        seed = kwargs.pop("seed", None)
        if seed is not None and "generator" not in kwargs:
            import torch

            device = getattr(pipe, "_execution_device", getattr(pipe, "device", "cpu"))
            try:
                kwargs["generator"] = torch.Generator(device=device).manual_seed(int(seed))
            except RuntimeError:
                kwargs["generator"] = torch.Generator(device="cpu").manual_seed(int(seed))
        kwargs["image"] = conditioning
        kwargs["prompt"] = prompt
        return pipe(**_filter_pipeline_kwargs(pipe, kwargs))

    kwargs["edit_image"] = conditioning
    return pipe(prompt, **_filter_pipeline_kwargs(pipe, kwargs))


def render_generation(
    pipe: Any,
    prompt: str,
    generation: dict[str, Any],
) -> Image.Image:
    return pipe(prompt, **_filter_pipeline_kwargs(pipe, build_generation_kwargs(generation)))
