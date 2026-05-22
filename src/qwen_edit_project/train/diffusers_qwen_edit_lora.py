from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import math
import os
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import Qwen2Tokenizer, Qwen2VLProcessor, Qwen2_5_VLForConditionalGeneration

import diffusers
from diffusers import (
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
    offload_models,
)
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module

try:
    from diffusers.training_utils import _collate_lora_metadata
except ImportError:  # pragma: no cover - compatibility with older Diffusers

    def _collate_lora_metadata(_: dict[str, Any]) -> dict[str, Any]:
        return {}


if is_torch_npu_available():  # pragma: no cover - NPU-only runtime path
    torch.npu.config.allow_internal_format = False


logger = get_logger(__name__)

CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


def calculate_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


def retrieve_latents(encoder_output: Any, generator: torch.Generator | None = None, sample_mode: str = "sample"):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents from VAE encoder output")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
    if not isinstance(records, list):
        raise TypeError(f"Training manifest must be a JSON list or JSONL records: {path}")
    return records


def resolve_record_path(base_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base_path / path


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as handle:
        image = exif_transpose(handle)
        if image.mode != "RGB":
            image = image.convert("RGB")
        else:
            image = image.copy()
    return image


class EditManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        base_path: Path,
        image_key: str,
        condition_image_key: str,
        prompt_key: str,
        repeats: int,
    ) -> None:
        records = load_manifest(manifest_path)
        if repeats < 1:
            raise ValueError("--dataset_repeat must be >= 1")
        self.examples: list[dict[str, Any]] = []
        for record in records:
            prompt = record.get(prompt_key)
            target_image = record.get(image_key)
            source_image = record.get(condition_image_key)
            if not prompt or not target_image or not source_image:
                continue
            self.examples.append(
                {
                    "prompt": str(prompt),
                    "target_path": resolve_record_path(base_path, str(target_image)),
                    "source_path": resolve_record_path(base_path, str(source_image)),
                }
            )
        if not self.examples:
            raise ValueError(f"No trainable edit examples found in {manifest_path}")
        missing_paths = []
        for example in self.examples:
            for key in ("target_path", "source_path"):
                if not example[key].exists():
                    missing_paths.append(str(example[key]))
        if missing_paths:
            preview = ", ".join(missing_paths[:5])
            suffix = "" if len(missing_paths) <= 5 else f", ... ({len(missing_paths)} missing total)"
            raise FileNotFoundError(f"Training manifest references missing image paths: {preview}{suffix}")
        self.examples = self.examples * repeats

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


def collate_single_example(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return examples


@dataclass
class PreparedEditBatch:
    model_input: torch.Tensor
    image_latents: torch.Tensor
    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor
    img_shapes: list[list[tuple[int, int, int]]]
    txt_seq_lens: list[int]
    target_height: int
    target_width: int


def choose_training_sizes(args: argparse.Namespace, source_image: Image.Image) -> tuple[int, int, int, int, int, int]:
    if args.preserve_aspect_ratio:
        ratio = source_image.size[0] / source_image.size[1]
        target_width, target_height = calculate_dimensions(args.max_pixels, ratio)
        condition_width, condition_height = calculate_dimensions(args.condition_pixels, ratio)
        return target_width, target_height, target_width, target_height, condition_width, condition_height
    return (
        args.resolution,
        args.resolution,
        args.resolution,
        args.resolution,
        args.condition_resolution,
        args.condition_resolution,
    )


def encode_edit_example(
    args: argparse.Namespace,
    conditioning_pipe: QwenImageEditPlusPipeline,
    vae: AutoencoderKLQwenImage,
    text_encoder: Qwen2_5_VLForConditionalGeneration,
    example: dict[str, Any],
    accelerator: Accelerator,
    weight_dtype: torch.dtype,
    latents_mean: torch.Tensor,
    latents_std_scale: torch.Tensor,
) -> PreparedEditBatch:
    source_image = load_rgb(example["source_path"])
    target_image = load_rgb(example["target_path"])
    target_width, target_height, vae_width, vae_height, condition_width, condition_height = choose_training_sizes(
        args, source_image
    )

    condition_image = conditioning_pipe.image_processor.resize(source_image, condition_height, condition_width)
    target_tensor = conditioning_pipe.image_processor.preprocess(target_image, target_height, target_width).unsqueeze(2)
    source_tensor = conditioning_pipe.image_processor.preprocess(source_image, vae_height, vae_width).unsqueeze(2)

    with torch.no_grad():
        with offload_models(text_encoder, device=accelerator.device, offload=args.offload):
            prompt_embeds, prompt_embeds_mask = conditioning_pipe.encode_prompt(
                image=[condition_image],
                prompt=[example["prompt"]],
                device=accelerator.device,
                max_sequence_length=args.max_sequence_length,
            )

        with offload_models(vae, device=accelerator.device, offload=args.offload):
            target_tensor = target_tensor.to(device=accelerator.device, dtype=vae.dtype)
            source_tensor = source_tensor.to(device=accelerator.device, dtype=vae.dtype)
            target_latents = retrieve_latents(vae.encode(target_tensor), sample_mode="sample")
            source_latents = retrieve_latents(vae.encode(source_tensor), sample_mode="argmax")

    model_input = (target_latents - latents_mean) * latents_std_scale
    model_input = model_input.to(device=accelerator.device, dtype=weight_dtype)
    source_latents = (source_latents - latents_mean) * latents_std_scale
    source_latents = source_latents.to(device=accelerator.device, dtype=weight_dtype)

    image_latent_height, image_latent_width = source_latents.shape[3:]
    image_latents = QwenImageEditPlusPipeline._pack_latents(
        source_latents,
        batch_size=1,
        num_channels_latents=model_input.shape[1],
        height=image_latent_height,
        width=image_latent_width,
    )

    vae_scale_factor = conditioning_pipe.vae_scale_factor
    img_shapes = [
        [
            (1, target_height // vae_scale_factor // 2, target_width // vae_scale_factor // 2),
            (1, vae_height // vae_scale_factor // 2, vae_width // vae_scale_factor // 2),
        ]
    ]
    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

    return PreparedEditBatch(
        model_input=model_input,
        image_latents=image_latents,
        prompt_embeds=prompt_embeds.to(device=accelerator.device, dtype=weight_dtype),
        prompt_embeds_mask=prompt_embeds_mask.to(device=accelerator.device),
        img_shapes=img_shapes,
        txt_seq_lens=[int(value) for value in txt_seq_lens],
        target_height=target_height,
        target_width=target_width,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen-Image-Edit LoRA with native Diffusers components.")
    parser.add_argument("--pretrained_model_name_or_path", default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--dataset_base_path", default=".")
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--image_key", default="image")
    parser.add_argument("--condition_image_key", default="edit_image")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--dataset_repeat", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--condition_resolution", type=int, default=384)
    parser.add_argument("--max_pixels", type=int, default=VAE_IMAGE_SIZE)
    parser.add_argument("--condition_pixels", type=int, default=CONDITION_IMAGE_SIZE)
    parser.add_argument("--preserve_aspect_ratio", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=250)
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--scale_lr", action="store_true")
    parser.add_argument(
        "--lr_scheduler",
        default="constant",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument(
        "--weighting_scheme",
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
    )
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--scheduler_shift", type=float, default=3.0)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_layers", default="to_k,to_q,to_v,to_out.0")
    parser.add_argument("--lora_checkpoint", default=None)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1.0e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1.0e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--mixed_precision", default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--output_dir", default="outputs/checkpoints/Qwen-Image-Edit-2509_lora_diffusers")
    parser.add_argument("--logging_dir", default="logs")
    parser.add_argument("--report_to", default="tensorboard")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--upcast_before_saving", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.train_batch_size != 1:
        raise ValueError("Qwen edit LoRA training currently requires --train_batch_size=1.")
    if args.max_sequence_length > 1024:
        raise ValueError("--max_sequence_length cannot exceed 1024 for Qwen-Image-Edit.")
    return args


def unwrap_model(accelerator: Accelerator, model: torch.nn.Module) -> torch.nn.Module:
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model


def load_lora_into_transformer(transformer: QwenImageTransformer2DModel, checkpoint: str) -> None:
    lora_state_dict = QwenImageEditPlusPipeline.lora_state_dict(checkpoint)
    transformer_state_dict = {
        key.replace("transformer.", ""): value
        for key, value in lora_state_dict.items()
        if key.startswith("transformer.")
    }
    if not transformer_state_dict:
        transformer_state_dict = dict(lora_state_dict)
    transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
    incompatible_keys = set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")
    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
    if unexpected_keys:
        logger.warning("Unexpected LoRA keys while loading %s: %s", checkpoint, unexpected_keys)


def main() -> None:
    args = parse_args()

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError("MPS does not support bf16 mixed precision. Use fp16 or no mixed precision.")

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    report_to = None if args.report_to in {"none", "None", ""} else args.report_to
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        log_with=report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        diffusers.utils.logging.set_verbosity_info()
    else:
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = EditManifestDataset(
        manifest_path=Path(args.dataset_metadata_path),
        base_path=Path(args.dataset_base_path),
        image_key=args.image_key,
        condition_image_key=args.condition_image_key,
        prompt_key=args.prompt_key,
        repeats=args.dataset_repeat,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_single_example,
        num_workers=args.dataloader_num_workers,
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    tokenizer = Qwen2Tokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    processor = Qwen2VLProcessor.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="processor",
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=args.revision,
        shift=args.scheduler_shift,
        local_files_only=args.local_files_only,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        local_files_only=args.local_files_only,
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        torch_dtype=weight_dtype,
        local_files_only=args.local_files_only,
    )
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        local_files_only=args.local_files_only,
    )

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    if args.offload:
        vae.to(dtype=weight_dtype)
        text_encoder.to(dtype=weight_dtype)
    else:
        vae.to(device=accelerator.device, dtype=weight_dtype)
        text_encoder.to(device=accelerator.device, dtype=weight_dtype)
    transformer.to(device=accelerator.device, dtype=weight_dtype)

    conditioning_pipe = QwenImageEditPlusPipeline(
        scheduler=noise_scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        processor=processor,
        transformer=None,
    )

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    target_modules = [layer.strip() for layer in args.lora_layers.split(",") if layer.strip()]
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)
    if args.lora_checkpoint:
        load_lora_into_transformer(transformer, args.lora_checkpoint)

    def save_model_hook(models: list[torch.nn.Module], weights: list[Any], output_dir: str) -> None:
        if not accelerator.is_main_process:
            return
        transformer_lora_layers_to_save = None
        modules_to_save = {}
        for model in models:
            unwrapped = unwrap_model(accelerator, model)
            if isinstance(unwrapped, type(unwrap_model(accelerator, transformer))):
                transformer_lora_layers_to_save = get_peft_model_state_dict(unwrapped)
                modules_to_save["transformer"] = unwrapped
            else:
                raise ValueError(f"Unexpected model during save: {model.__class__}")
            if weights:
                weights.pop()
        QwenImageEditPlusPipeline.save_lora_weights(
            output_dir,
            transformer_lora_layers=transformer_lora_layers_to_save,
            **_collate_lora_metadata(modules_to_save),
        )

    def load_model_hook(models: list[torch.nn.Module], input_dir: str) -> None:
        transformer_to_load = None
        while models:
            model = models.pop()
            unwrapped = unwrap_model(accelerator, model)
            if isinstance(unwrapped, type(unwrap_model(accelerator, transformer))):
                transformer_to_load = unwrapped
            else:
                raise ValueError(f"Unexpected model during load: {model.__class__}")
        if transformer_to_load is None:
            return
        load_lora_into_transformer(transformer_to_load, input_dir)
        if args.mixed_precision == "fp16":
            cast_training_params([transformer_to_load], dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)
    trainable_parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError(
            "No trainable LoRA parameters were created. Check --lora_layers against the Qwen transformer module names."
        )
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable_parameters)
    logger.info("Trainable LoRA parameters = %s", trainable_parameter_count)

    optimizer_name = args.optimizer.lower()
    if optimizer_name == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise ImportError("Install bitsandbytes to use --use_8bit_adam.") from exc
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            [{"params": trainable_parameters}],
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        raise ValueError(f"Unsupported optimizer for Diffusers Qwen edit LoRA training: {args.optimizer}")

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    vae_scale_factor = conditioning_pipe.vae_scale_factor
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(accelerator.device)
    latents_std_scale = (
        1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(accelerator.device)
    )

    if accelerator.is_main_process:
        tracker_config = vars(args).copy()
        accelerator.init_trackers("qwen-edit-diffusers-lora", config=tracker_config)
        with open(Path(args.output_dir) / "training_args.json", "w", encoding="utf-8") as handle:
            json.dump(tracker_config, handle, indent=2, ensure_ascii=True)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running Diffusers-native Qwen edit LoRA training *****")
    logger.info("  Num examples = %s", len(train_dataset))
    logger.info("  Num batches each epoch = %s", len(train_dataloader))
    logger.info("  Num epochs = %s", args.num_train_epochs)
    logger.info("  Instantaneous batch size per device = %s", args.train_batch_size)
    logger.info("  Total train batch size = %s", total_batch_size)
    logger.info("  Total optimization steps = %s", args.max_train_steps)

    global_step = 0
    first_epoch = 0
    resume_step = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = []
            for name in os.listdir(args.output_dir) if os.path.isdir(args.output_dir) else []:
                if not name.startswith("checkpoint-"):
                    continue
                try:
                    dirs.append((int(name.split("-")[-1]), name))
                except ValueError:
                    continue
            dirs = sorted(dirs, key=lambda item: item[0])
            path = dirs[-1][1] if dirs else None
        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting fresh.")
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[-1])
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = (global_step % num_update_steps_per_epoch) * args.gradient_accumulation_steps

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps: torch.Tensor, n_dim: int = 4, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == timestep).nonzero().item() for timestep in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    guidance_embeds = bool(getattr(unwrap_model(accelerator, transformer).config, "guidance_embeds", False))
    if guidance_embeds and args.guidance_scale is None:
        raise ValueError("--guidance_scale is required for guidance-distilled Qwen transformers.")

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        active_dataloader = train_dataloader
        if resume_step:
            accelerator.print(f"Skipping {resume_step} already-processed dataloader batch(es) in epoch {epoch}.")
            if hasattr(accelerator, "skip_first_batches"):
                active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            else:
                active_dataloader = itertools.islice(train_dataloader, resume_step, None)
            resume_step = 0
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(transformer):
                if len(batch) != 1:
                    raise ValueError("Diffusers edit training currently expects exactly one example per batch.")
                prepared = encode_edit_example(
                    args=args,
                    conditioning_pipe=conditioning_pipe,
                    vae=vae,
                    text_encoder=text_encoder,
                    example=batch[0],
                    accelerator=accelerator,
                    weight_dtype=weight_dtype,
                    latents_mean=latents_mean,
                    latents_std_scale=latents_std_scale,
                )

                model_input = prepared.model_input
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = QwenImageEditPlusPipeline._pack_latents(
                    noisy_model_input.permute(0, 2, 1, 3, 4).contiguous(),
                    batch_size=bsz,
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[3],
                    width=model_input.shape[4],
                )
                latent_model_input = torch.cat([packed_noisy_model_input, prepared.image_latents], dim=1)
                guidance = None
                if guidance_embeds:
                    guidance = torch.full([bsz], args.guidance_scale, device=model_input.device, dtype=torch.float32)

                cache_context = getattr(transformer, "cache_context", None)
                context = cache_context("cond") if cache_context is not None else nullcontext()
                with context:
                    model_pred = transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prepared.prompt_embeds,
                        encoder_hidden_states_mask=prepared.prompt_embeds_mask,
                        timestep=timesteps / 1000,
                        img_shapes=prepared.img_shapes,
                        txt_seq_lens=prepared.txt_seq_lens,
                        guidance=guidance,
                        return_dict=False,
                    )[0]
                model_pred = model_pred[:, : packed_noisy_model_input.shape[1]]
                model_pred = QwenImageEditPlusPipeline._unpack_latents(
                    model_pred,
                    prepared.target_height,
                    prepared.target_width,
                    vae_scale_factor,
                )

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = noise - model_input
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    dim=1,
                ).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                free_memory()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process and args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = []
                        for name in os.listdir(args.output_dir):
                            if not name.startswith("checkpoint-"):
                                continue
                            try:
                                checkpoints.append((int(name.split("-")[-1]), name))
                            except ValueError:
                                continue
                        checkpoints = [name for _, name in sorted(checkpoints, key=lambda item: item[0])]
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            for checkpoint in checkpoints[: len(checkpoints) - args.checkpoints_total_limit + 1]:
                                shutil.rmtree(os.path.join(args.output_dir, checkpoint))
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info("Saved state to %s", save_path)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_transformer = unwrap_model(accelerator, transformer)
        if args.upcast_before_saving:
            unwrapped_transformer.to(torch.float32)
        else:
            unwrapped_transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(unwrapped_transformer)
        QwenImageEditPlusPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            **_collate_lora_metadata({"transformer": unwrapped_transformer}),
        )
        logger.info("Saved final Diffusers LoRA weights to %s", args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    main()
