from __future__ import annotations

import math
import random
import statistics
import gc
from typing import Any

from PIL import Image, ImageEnhance, ImageOps

from qwen_edit_project.self_evolve.edit_schema import (
    extract_json_object,
    normalize_structured_edit,
    proposal_definition_from_structured_edit,
    structured_edit_prompt,
)
from qwen_edit_project.self_evolve.image_metrics import (
    changed_fraction,
    diff_region_statistics,
    edge_preservation_score,
    luminance_mean,
    luminance_std,
    mean_absolute_difference,
    saturation_mean,
    warmth_score,
)
from qwen_edit_project.self_evolve.proposal_bank import PROPOSAL_BANK, available_proposals
from qwen_edit_project.self_evolve.types import EditProposal, ProposalDefinition, SolverResult, UnlabeledImageRecord
from qwen_edit_project.utils.prompting import polish_prompt
from qwen_edit_project.utils.qwen_pipeline import (
    extract_qwen_edit_understanding_features,
    extract_qwen_text_features,
    extract_qwen_vae_latents,
    load_qwen_edit_pipeline,
    render_edit,
)


PROPOSAL_BY_ID = {proposal.operation_id: proposal for proposal in PROPOSAL_BANK}
INVERSE_OPERATION_MAP = {
    "brightness_up": "brightness_down",
    "brightness_down": "brightness_up",
    "saturation_up": "saturation_down",
    "saturation_down": "saturation_up",
    "contrast_up": "contrast_down",
    "contrast_down": "contrast_up",
    "warm_tone": "cool_tone",
    "cool_tone": "warm_tone",
}
INTERNAL_ONLY_METRICS = {"internal_prompt_gain", "semantic"}


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _mean_changed_fraction_score(value: float, expected_range: tuple[float, float]) -> float:
    low, high = expected_range
    if low <= value <= high:
        return 1.0
    if value < low:
        return _clamp(1.0 - (low - value) / max(low, 1e-6))
    return _clamp(1.0 - (value - high) / max(1.0 - high, 1e-6))


def _apply_pillow_operation(image: Image.Image, operation_id: str) -> Image.Image:
    if operation_id == "brightness_up":
        return ImageEnhance.Brightness(image).enhance(1.25)
    if operation_id == "brightness_down":
        return ImageEnhance.Brightness(image).enhance(0.75)
    if operation_id == "saturation_up":
        return ImageEnhance.Color(image).enhance(1.35)
    if operation_id == "saturation_down":
        return ImageEnhance.Color(image).enhance(0.60)
    if operation_id == "contrast_up":
        return ImageEnhance.Contrast(image).enhance(1.35)
    if operation_id == "contrast_down":
        return ImageEnhance.Contrast(image).enhance(0.70)
    if operation_id == "warm_tone":
        red, green, blue = image.split()
        red = red.point(lambda value: min(255, int(value * 1.10 + 8)))
        blue = blue.point(lambda value: max(0, int(value * 0.92)))
        return Image.merge("RGB", (red, green, blue))
    if operation_id == "cool_tone":
        red, green, blue = image.split()
        red = red.point(lambda value: max(0, int(value * 0.92)))
        blue = blue.point(lambda value: min(255, int(value * 1.10 + 8)))
        return Image.merge("RGB", (red, green, blue))
    if operation_id == "grayscale":
        return ImageOps.grayscale(image).convert("RGB")
    raise ValueError(f"Unsupported pillow prototype operation: {operation_id}")


def _reverse_proposal(proposal: EditProposal) -> EditProposal | None:
    inverse_operation_id = proposal.definition.inverse_operation_id or INVERSE_OPERATION_MAP.get(
        proposal.definition.operation_id
    )
    if inverse_operation_id is None:
        return None
    inverse_definition = PROPOSAL_BY_ID[inverse_operation_id]
    return EditProposal(
        record_key=proposal.record_key,
        round_index=proposal.round_index,
        proposal_index=proposal.proposal_index,
        definition=inverse_definition,
        difficulty_level=proposal.difficulty_level,
        instruction=inverse_definition.instruction,
    )


class DifficultyController:
    def __init__(self, initial_level: int, min_level: int, max_level: int, promote_at: float, demote_at: float):
        self.level = initial_level
        self.min_level = min_level
        self.max_level = max_level
        self.promote_at = promote_at
        self.demote_at = demote_at

    def update(self, acceptance_rate: float) -> int:
        if acceptance_rate >= self.promote_at and self.level < self.max_level:
            self.level += 1
        elif acceptance_rate <= self.demote_at and self.level > self.min_level:
            self.level -= 1
        return self.level


class ScriptedProposer:
    def __init__(self, config: dict[str, Any]):
        self.families = config.get("families")
        self.operation_ids = config.get("operation_ids")
        self.family_policy = config.get("family_policy", "random")

    @staticmethod
    def _metadata_family_preferences(record: UnlabeledImageRecord) -> tuple[list[str], list[str]]:
        raw_families = record.metadata.get("edit_families") or record.metadata.get("families") or []
        if isinstance(raw_families, str):
            raw_families = [part.strip() for part in raw_families.split(",")]
        primary_family = record.metadata.get("primary_family")
        primary = []
        if isinstance(primary_family, str) and primary_family:
            primary.append(primary_family)
        families = [*primary]
        if isinstance(raw_families, list):
            families.extend(str(item) for item in raw_families if item)
        seen = set()
        ordered = []
        for family in families:
            if family not in seen:
                seen.add(family)
                ordered.append(family)
        return primary, ordered

    def propose(
        self,
        record: UnlabeledImageRecord,
        round_index: int,
        difficulty_level: int,
        proposals_per_image: int,
        seed: int,
    ) -> list[EditProposal]:
        candidates = available_proposals(difficulty_level, families=self.families)
        if self.operation_ids:
            operation_filter = set(self.operation_ids)
            candidates = [candidate for candidate in candidates if candidate.operation_id in operation_filter]
        if not candidates:
            return []
        stable_key_seed = sum(ord(char) for char in record.key)
        rng = random.Random(seed + round_index * 100_003 + stable_key_seed)
        primary_families, metadata_families = self._metadata_family_preferences(record)
        metadata_family_set = set(metadata_families)
        if self.family_policy == "metadata_preferred" and metadata_family_set:
            primary = [candidate for candidate in candidates if candidate.family in set(primary_families)]
            preferred = [
                candidate
                for candidate in candidates
                if candidate.family in metadata_family_set and candidate.family not in set(primary_families)
            ]
            fallback = [candidate for candidate in candidates if candidate.family not in metadata_family_set]
            rng.shuffle(primary)
            rng.shuffle(preferred)
            rng.shuffle(fallback)
            selected = (primary + preferred + fallback)[:proposals_per_image]
        elif proposals_per_image >= len(candidates):
            selected = candidates[:]
            rng.shuffle(selected)
        else:
            selected = rng.sample(candidates, proposals_per_image)
        return [
            EditProposal(
                record_key=record.key,
                round_index=round_index,
                proposal_index=index,
                definition=proposal,
                difficulty_level=difficulty_level,
                instruction=proposal.instruction,
            )
            for index, proposal in enumerate(selected)
        ]


class InternalQwenProposer:
    def propose(
        self,
        record: UnlabeledImageRecord,
        round_index: int,
        difficulty_level: int,
        proposals_per_image: int,
        seed: int,
    ) -> list[EditProposal]:
        raise NotImplementedError(
            "internal_qwen proposer is not available yet. "
            "The control loop is implemented, but the public Qwen edit pipeline does not expose "
            "the internal understanding branch as a standalone proposer API."
        )


class TrainableQwenVLProposer:
    """Round-updatable Qwen-Image-Edit VLM proposer with a scripted bootstrap fallback.

    The proposer emits a structured edit JSON object using the autoregressive VLM/text-encoder
    component from the edit checkpoint. The diffusion editor itself cannot emit text; using the
    edit model's VLM component keeps the proposer tied to the same model family while training a
    separate proposer LoRA that is never used for final editor evaluation.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_name_or_path = str(config.get("model_name_or_path", "Qwen/Qwen-Image-Edit-2509"))
        self.model_subfolder = config.get("model_subfolder", "text_encoder")
        self.processor_subfolder = config.get("processor_subfolder", "processor")
        self.model_class = str(config.get("model_class", "qwen2_5_vl"))
        self.checkpoint_path = config.get("checkpoint_path")
        self.device = config.get("device", "auto")
        self.torch_dtype = config.get("torch_dtype", "auto")
        self.local_files_only = bool(config.get("local_files_only", False))
        self.generation = dict(config.get("generation", {}))
        self.fallback_on_error = bool(config.get("fallback_on_error", True))
        self.scripted_probability = float(config.get("scripted_probability", 0.25))
        fallback_config = dict(config.get("scripted_fallback", {}))
        fallback_config.setdefault("families", config.get("families"))
        fallback_config.setdefault("operation_ids", config.get("operation_ids"))
        fallback_config.setdefault("family_policy", config.get("family_policy", "metadata_preferred"))
        self.scripted = ScriptedProposer(fallback_config)
        self.processor = None
        self.model = None

    def set_checkpoint_path(self, checkpoint_path: str | None) -> None:
        if checkpoint_path == self.checkpoint_path:
            return
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.processor = None

    def model_state(self) -> dict[str, Any]:
        return {
            "backend": self.config.get("backend", "trainable_qwen_image_edit"),
            "model_name_or_path": self.model_name_or_path,
            "model_subfolder": self.model_subfolder,
            "processor_subfolder": self.processor_subfolder,
            "checkpoint_path": self.checkpoint_path,
        }

    @staticmethod
    def _resolve_device(device: str):
        import torch

        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _ensure_model(self):
        if self.model is not None and self.processor is not None:
            return self.model, self.processor
        import torch

        from qwen_edit_project.utils.device import resolve_torch_dtype

        if self.model_class == "qwen2_5_vl":
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        else:
            try:
                from transformers import AutoModelForImageTextToText as ModelClass
            except ImportError:
                try:
                    from transformers import AutoModelForVision2Seq as ModelClass
                except ImportError:
                    from transformers import AutoModelForCausalLM as ModelClass
        from transformers import AutoProcessor

        dtype = resolve_torch_dtype(torch, self.torch_dtype, self._resolve_device(self.device))
        processor_kwargs = {
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
        }
        if self.processor_subfolder:
            processor_kwargs["subfolder"] = self.processor_subfolder
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
            "local_files_only": self.local_files_only,
        }
        if self.model_subfolder:
            model_kwargs["subfolder"] = self.model_subfolder
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path, **processor_kwargs)
        self.model = ModelClass.from_pretrained(self.model_name_or_path, **model_kwargs)
        if self.checkpoint_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.checkpoint_path)
        self.model.to(self._resolve_device(self.device))
        self.model.eval()
        return self.model, self.processor

    def _messages(self, record: UnlabeledImageRecord, difficulty_level: int, proposals_per_image: int) -> list[dict[str, Any]]:
        metadata_hint = ""
        if record.caption:
            metadata_hint += f"\nImage caption: {record.caption}"
        if record.metadata:
            keys = ["primary_family", "edit_families", "objects", "scene", "style"]
            parts = [f"{key}: {record.metadata[key]}" for key in keys if key in record.metadata]
            if parts:
                metadata_hint += "\nMetadata: " + "; ".join(parts)
        prompt = structured_edit_prompt(difficulty_level, proposals_per_image) + metadata_hint
        return [
            {
                "role": "system",
                "content": "You are a research proposer for image-editing self-training.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(record.image_path)},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

    @staticmethod
    def _to_payloads(decoded: str) -> list[dict[str, Any]]:
        parsed = extract_json_object(decoded)
        if parsed is None:
            return []
        proposals = parsed.get("proposals")
        if isinstance(proposals, list):
            return [item for item in proposals if isinstance(item, dict)]
        return [parsed]

    def _generate_payloads(
        self,
        record: UnlabeledImageRecord,
        difficulty_level: int,
        proposals_per_image: int,
    ) -> list[dict[str, Any]]:
        import torch

        model, processor = self._ensure_model()
        messages = self._messages(record, difficulty_level, proposals_per_image)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with Image.open(record.image_path) as image_handle:
            image = image_handle.convert("RGB")
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        max_new_tokens = int(self.generation.get("max_new_tokens", 512))
        temperature = float(self.generation.get("temperature", 0.7))
        top_p = float(self.generation.get("top_p", 0.9))
        do_sample = bool(self.generation.get("do_sample", temperature > 0.0))
        with torch.no_grad():
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            output_ids = model.generate(**generation_kwargs)
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
        return self._to_payloads(decoded)

    def propose(
        self,
        record: UnlabeledImageRecord,
        round_index: int,
        difficulty_level: int,
        proposals_per_image: int,
        seed: int,
    ) -> list[EditProposal]:
        rng = random.Random(seed + round_index * 100_003 + sum(ord(char) for char in record.key))
        if self.scripted_probability > 0 and rng.random() < self.scripted_probability:
            return self.scripted.propose(record, round_index, difficulty_level, proposals_per_image, seed)
        try:
            payloads = self._generate_payloads(record, difficulty_level, proposals_per_image)
        except Exception:
            if not self.fallback_on_error:
                raise
            return self.scripted.propose(record, round_index, difficulty_level, proposals_per_image, seed)

        proposals: list[EditProposal] = []
        for index, payload in enumerate(payloads[:proposals_per_image]):
            instruction = str(payload.get("instruction", "")).strip()
            if not instruction:
                continue
            structured_edit = normalize_structured_edit(payload, instruction=instruction)
            definition = proposal_definition_from_structured_edit(
                structured_edit,
                proposal_index=index,
                difficulty_level=difficulty_level,
            )
            proposals.append(
                EditProposal(
                    record_key=record.key,
                    round_index=round_index,
                    proposal_index=index,
                    definition=definition,
                    difficulty_level=difficulty_level,
                    instruction=definition.instruction,
                    structured_edit=structured_edit,
                )
            )
        if proposals:
            return proposals
        if not self.fallback_on_error:
            return []
        return self.scripted.propose(record, round_index, difficulty_level, proposals_per_image, seed)


class PillowPrototypeEditor:
    def edit_image(self, image: Image.Image, instruction: str, operation_id: str | None = None) -> Image.Image:
        if operation_id is None:
            raise ValueError("PillowPrototypeEditor requires operation_id for deterministic editing.")
        return _apply_pillow_operation(image.convert("RGB"), operation_id)

    def edit(self, record: UnlabeledImageRecord, proposal: EditProposal) -> Image.Image:
        image = Image.open(record.image_path).convert("RGB")
        return self.edit_image(image, proposal.instruction, proposal.definition.operation_id)

    def edit_candidate(self, record: UnlabeledImageRecord, proposal: EditProposal, candidate_index: int, seed: int) -> Image.Image:
        image = Image.open(record.image_path).convert("RGB")
        edited = self.edit_image(image, proposal.instruction, proposal.definition.operation_id)
        if candidate_index == 0:
            return edited
        rng = random.Random(seed + candidate_index * 7919)
        brightness = 0.96 + rng.random() * 0.08
        contrast = 0.96 + rng.random() * 0.08
        edited = ImageEnhance.Brightness(edited).enhance(brightness)
        return ImageEnhance.Contrast(edited).enhance(contrast)


class QwenEditEditor:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.device = config.get("device", "auto")
        self.generation = dict(config.get("generation", {}))
        self.pipeline = None
        self.current_checkpoint_path = config["model"].get("checkpoint_path")

    def set_checkpoint_path(self, checkpoint_path: str | None) -> None:
        if checkpoint_path == self.current_checkpoint_path:
            return
        self.current_checkpoint_path = checkpoint_path
        self.pipeline = None

    def set_model_checkpoint(
        self,
        checkpoint_path: str | None,
        *,
        model_type: str | None = None,
        backend: str | None = None,
    ) -> None:
        """Switch the editor to a checkpoint with an explicitly compatible loader.

        Self-evolve training currently writes checkpoints through DiffSynth. Reloading
        those weights through the official Diffusers base pipeline is not a valid
        research comparison unless the checkpoint is known to be Diffusers-compatible.
        This helper updates the checkpoint, model type, and backend as one atomic
        editor state transition so later rounds cannot accidentally mix them.
        """
        model_cfg = self.config["model"]
        old_state = (
            self.current_checkpoint_path,
            model_cfg.get("model_type", "base"),
            model_cfg.get("backend", "diffsynth"),
        )
        if model_type is not None:
            model_cfg["model_type"] = model_type
        if backend is not None:
            model_cfg["backend"] = backend
        self.current_checkpoint_path = checkpoint_path
        new_state = (
            self.current_checkpoint_path,
            model_cfg.get("model_type", "base"),
            model_cfg.get("backend", "diffsynth"),
        )
        if new_state != old_state:
            self.pipeline = None

    def model_state(self) -> dict[str, Any]:
        model_cfg = self.config.get("model", {})
        return {
            "backend": model_cfg.get("backend", "diffsynth"),
            "model_type": model_cfg.get("model_type", "base"),
            "checkpoint_path": self.current_checkpoint_path,
            "base_model": model_cfg.get("base_model"),
        }

    def _ensure_pipeline(self):
        if self.pipeline is not None:
            return self.pipeline
        model_cfg = self.config["model"]
        self.pipeline = load_qwen_edit_pipeline(
            model_id_with_origin_paths=model_cfg["model_id_with_origin_paths"],
            checkpoint_path=self.current_checkpoint_path,
            model_type=model_cfg.get("model_type", "base"),
            device=self.device,
            processor_model_id=model_cfg.get("processor_model_id", "Qwen/Qwen-Image-Edit"),
            torch_dtype=model_cfg.get("torch_dtype", "auto"),
            backend=model_cfg.get("backend", "diffsynth"),
            base_model=model_cfg.get("base_model"),
            local_files_only=bool(model_cfg.get("local_files_only", False)),
        )
        return self.pipeline

    def _resolved_torch_device(self):
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @staticmethod
    def _empty_cuda_cache() -> None:
        try:
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    @staticmethod
    def _move_module(module: Any, device: Any) -> None:
        if module is not None and hasattr(module, "to"):
            module.to(device)

    def prepare_for_generation(self) -> None:
        """Ensure heavyweight generation modules are on the target device."""
        pipeline = self._ensure_pipeline()
        device = self._resolved_torch_device()
        if hasattr(pipeline, "to"):
            pipeline.to(device)
        for name in ("transformer", "dit", "unet", "text_encoder", "vae"):
            self._move_module(getattr(pipeline, name, None), device)
        self._empty_cuda_cache()

    def prepare_for_internal_scoring(self, scoring_device: str | None = None) -> None:
        """Free generation-only GPU memory before internal CEPR scoring.

        Qwen-Image-Edit generation fits on a large H200, but it leaves almost no free
        memory for extra internal text/VAE reward passes. CEPR scoring only needs the
        editor's text/understanding path and VAE latents, not the diffusion transformer,
        so we temporarily offload the transformer/DiT to CPU.
        """
        pipeline = self._ensure_pipeline()
        if hasattr(pipeline, "to"):
            pipeline.to("cpu")
        self._empty_cuda_cache()
        for name in ("transformer", "dit", "unet"):
            self._move_module(getattr(pipeline, name, None), "cpu")
        device = scoring_device or self._resolved_torch_device()
        for name in ("text_encoder", "vae"):
            self._move_module(getattr(pipeline, name, None), device)
        self._empty_cuda_cache()

    def edit_image(self, image: Image.Image, instruction: str, operation_id: str | None = None) -> Image.Image:
        self.prepare_for_generation()
        pipeline = self._ensure_pipeline()
        generation = dict(self.generation)
        if bool(generation.get("preserve_input_resolution", True)):
            generation["width"], generation["height"] = image.size
        else:
            generation.pop("width", None)
            generation.pop("height", None)
        prompt = polish_prompt(instruction, use_prompt_polish=False, image_context=image)
        output = render_edit(pipeline, prompt, [image.convert("RGB")], generation)
        return output.images[0] if hasattr(output, "images") else output

    def edit(self, record: UnlabeledImageRecord, proposal: EditProposal) -> Image.Image:
        image = Image.open(record.image_path).convert("RGB")
        return self.edit_image(image, proposal.instruction, proposal.definition.operation_id)

    def edit_candidate(self, record: UnlabeledImageRecord, proposal: EditProposal, candidate_index: int, seed: int) -> Image.Image:
        image = Image.open(record.image_path).convert("RGB")
        original_seed = self.generation.get("seed")
        self.generation["seed"] = int(seed + candidate_index * 7919)
        try:
            return self.edit_image(image, proposal.instruction, proposal.definition.operation_id)
        finally:
            if original_seed is None:
                self.generation.pop("seed", None)
            else:
                self.generation["seed"] = original_seed


class StatSolver:
    def __init__(self, config: dict[str, Any]):
        self.global_weight = float(config.get("global_weight", 0.7))
        self.local_weight = float(config.get("local_weight", 0.3))
        self.acceptance_threshold = float(config.get("acceptance_threshold", 0.72))

    def _global_score(self, definition: ProposalDefinition, original: Image.Image, edited: Image.Image) -> tuple[float, dict[str, float]]:
        if definition.metric in INTERNAL_ONLY_METRICS or definition.verifier == "internal":
            return 0.5, {
                "global_proxy_supported": 0.0,
                "internal_metric_required": 1.0,
                "luminance_delta": 0.0,
                "contrast_delta": 0.0,
                "saturation_delta": 0.0,
                "warmth_delta": 0.0,
                "saturation_level": 0.0,
            }

        original_luminance = luminance_mean(original)
        edited_luminance = luminance_mean(edited)
        original_contrast = luminance_std(original)
        edited_contrast = luminance_std(edited)
        original_saturation = saturation_mean(original)
        edited_saturation = saturation_mean(edited)
        original_warmth = warmth_score(original)
        edited_warmth = warmth_score(edited)

        deltas = {
            "luminance_delta": edited_luminance - original_luminance,
            "contrast_delta": edited_contrast - original_contrast,
            "saturation_delta": edited_saturation - original_saturation,
            "warmth_delta": edited_warmth - original_warmth,
            "saturation_level": edited_saturation,
        }

        if definition.metric == "luminance":
            value = deltas["luminance_delta"]
        elif definition.metric == "contrast":
            value = deltas["contrast_delta"]
        elif definition.metric == "saturation":
            value = deltas["saturation_delta"]
        elif definition.metric == "warmth":
            value = deltas["warmth_delta"]
        elif definition.metric == "saturation_level":
            value = deltas["saturation_level"]
        else:
            raise ValueError(f"Unsupported solver metric: {definition.metric}")

        if definition.direction == "increase":
            score = _clamp(value / definition.target)
        elif definition.direction == "decrease":
            score = _clamp((-value) / definition.target)
        elif definition.direction == "at_most":
            score = 1.0 if value <= definition.target else _clamp(1.0 - (value - definition.target) / (1.0 - definition.target))
        else:
            raise ValueError(f"Unsupported solver direction: {definition.direction}")

        return score, {
            "global_proxy_supported": 1.0,
            "internal_metric_required": 0.0,
            "luminance_delta": deltas["luminance_delta"],
            "contrast_delta": deltas["contrast_delta"],
            "saturation_delta": deltas["saturation_delta"],
            "warmth_delta": deltas["warmth_delta"],
            "saturation_level": deltas["saturation_level"],
        }

    def _local_score(self, proposal: EditProposal, original: Image.Image, edited: Image.Image) -> tuple[float, dict[str, float]]:
        area_fraction = changed_fraction(original, edited)
        area_score = _mean_changed_fraction_score(area_fraction, proposal.definition.expected_changed_fraction)
        edge_score = edge_preservation_score(original, edited)
        local_score = 0.5 * area_score + 0.5 * edge_score
        return local_score, {
            "changed_fraction": area_fraction,
            "area_score": area_score,
            "edge_preservation_score": edge_score,
        }

    def score(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        editor: Any | None = None,
    ) -> SolverResult:
        global_score, global_signals = self._global_score(proposal.definition, original, edited)
        local_score, local_signals = self._local_score(proposal, original, edited)
        total_score = self.global_weight * global_score + self.local_weight * local_score
        return SolverResult(
            global_score=global_score,
            local_score=local_score,
            total_score=total_score,
            accepted=total_score >= self.acceptance_threshold,
            component_scores={
                "proxy_global_score": global_score,
                "proxy_local_score": local_score,
                "proxy_total_score": total_score,
            },
            signals={**global_signals, **local_signals},
        )


class MultiSignalSolver(StatSolver):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.proxy_weight = float(config.get("proxy_weight", 1.0))
        self.spatial_weight = float(config.get("spatial_weight", 0.0))
        self.cycle_weight = float(config.get("cycle_weight", 0.0))
        self.internal_weight = float(config.get("internal_weight", 0.0))
        self.cycle_neutral_score = float(config.get("cycle_neutral_score", 0.5))
        self.internal_neutral_score = float(config.get("internal_neutral_score", 0.5))
        self.spatial_inside_scale = float(config.get("spatial_inside_scale", 0.10))
        self.spatial_outside_scale = float(config.get("spatial_outside_scale", 0.05))
        self.cycle_change_scale = float(config.get("cycle_change_scale", 0.20))
        self.cycle_mae_scale = float(config.get("cycle_mae_scale", 0.08))
        self.internal_instruction_weight = float(config.get("internal_instruction_weight", 0.65))
        self.internal_prompt_gain_scale = float(config.get("internal_prompt_gain_scale", 0.08))

    def _spatial_score(self, proposal: EditProposal, original: Image.Image, edited: Image.Image) -> tuple[float, dict[str, float]]:
        stats = diff_region_statistics(original, edited)
        area_score = _mean_changed_fraction_score(stats["changed_fraction"], proposal.definition.expected_changed_fraction)
        inside_support = _clamp(stats["inside_mean_delta"] / max(self.spatial_inside_scale, 1e-6))
        outside_preservation = _clamp(1.0 - stats["outside_mean_delta"] / max(self.spatial_outside_scale, 1e-6))
        compactness = stats["mask_compactness"]
        precision = stats["mask_precision"]
        if proposal.definition.scope == "local":
            spatial_score = 0.25 * area_score + 0.25 * inside_support + 0.30 * outside_preservation + 0.20 * compactness
        else:
            spatial_score = 0.35 * area_score + 0.25 * inside_support + 0.25 * outside_preservation + 0.15 * precision
        return spatial_score, {
            "spatial_area_score": area_score,
            "spatial_inside_support": inside_support,
            "spatial_outside_preservation": outside_preservation,
            "spatial_mask_compactness": compactness,
            "spatial_mask_precision": precision,
            "spatial_changed_fraction": stats["changed_fraction"],
            "spatial_inside_mean_delta": stats["inside_mean_delta"],
            "spatial_outside_mean_delta": stats["outside_mean_delta"],
        }

    def _cycle_score(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        editor: Any | None,
    ) -> tuple[float, dict[str, float]]:
        reverse_proposal = _reverse_proposal(proposal)
        if reverse_proposal is None or editor is None or not hasattr(editor, "edit_image"):
            return self.cycle_neutral_score, {
                "cycle_supported": 0.0,
                "cycle_reconstruction_changed_fraction": 1.0,
                "cycle_reconstruction_edge_score": 0.0,
                "cycle_reconstruction_mae": 1.0,
            }
        try:
            reconstructed = editor.edit_image(edited, reverse_proposal.instruction, reverse_proposal.definition.operation_id)
        except Exception:
            return self.cycle_neutral_score, {
                "cycle_supported": 0.0,
                "cycle_reconstruction_changed_fraction": 1.0,
                "cycle_reconstruction_edge_score": 0.0,
                "cycle_reconstruction_mae": 1.0,
                "cycle_runtime_error": 1.0,
            }
        reconstruction_change = changed_fraction(original, reconstructed)
        reconstruction_edge = edge_preservation_score(original, reconstructed)
        reconstruction_mae = mean_absolute_difference(original, reconstructed)
        change_recovery = _clamp(1.0 - reconstruction_change / max(self.cycle_change_scale, 1e-6))
        mae_recovery = _clamp(1.0 - reconstruction_mae / max(self.cycle_mae_scale, 1e-6))
        cycle_score = 0.4 * change_recovery + 0.3 * reconstruction_edge + 0.3 * mae_recovery
        return cycle_score, {
            "cycle_supported": 1.0,
            "cycle_change_recovery": change_recovery,
            "cycle_reconstruction_changed_fraction": reconstruction_change,
            "cycle_reconstruction_edge_score": reconstruction_edge,
            "cycle_reconstruction_mae": reconstruction_mae,
            "cycle_mae_recovery": mae_recovery,
        }

    def _internal_feature_score(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        editor: Any | None,
    ) -> tuple[float, dict[str, float]]:
        if not isinstance(editor, QwenEditEditor):
            return self.internal_neutral_score, {
                "internal_supported": 0.0,
                "internal_original_text_similarity": 0.0,
                "internal_edited_text_similarity": 0.0,
                "internal_blank_preservation": 0.0,
                "internal_prompt_gain": 0.0,
            }
        try:
            import torch
            import torch.nn.functional as F

            pipe = editor._ensure_pipeline()
            prompt = polish_prompt(proposal.instruction, use_prompt_polish=False, image_context=original)
            with torch.no_grad():
                text_feature = extract_qwen_text_features(pipe, prompt)["pooled_embedding"][0].float()
                original_prompt_feature = extract_qwen_edit_understanding_features(pipe, prompt, [original])["pooled_embedding"][0].float()
                edited_prompt_feature = extract_qwen_edit_understanding_features(pipe, prompt, [edited])["pooled_embedding"][0].float()
                original_blank_feature = extract_qwen_edit_understanding_features(pipe, " ", [original])["pooled_embedding"][0].float()
                edited_blank_feature = extract_qwen_edit_understanding_features(pipe, " ", [edited])["pooled_embedding"][0].float()

            original_text_similarity = 0.5 * (
                1.0 + float(F.cosine_similarity(original_prompt_feature.unsqueeze(0), text_feature.unsqueeze(0)).item())
            )
            edited_text_similarity = 0.5 * (
                1.0 + float(F.cosine_similarity(edited_prompt_feature.unsqueeze(0), text_feature.unsqueeze(0)).item())
            )
            blank_preservation = 0.5 * (
                1.0 + float(F.cosine_similarity(original_blank_feature.unsqueeze(0), edited_blank_feature.unsqueeze(0)).item())
            )
            prompt_gain = edited_text_similarity - original_text_similarity
            instruction_score = _clamp(0.5 + 0.5 * prompt_gain / max(self.internal_prompt_gain_scale, 1e-6))
            internal_score = (
                self.internal_instruction_weight * instruction_score
                + (1.0 - self.internal_instruction_weight) * blank_preservation
            )
            return internal_score, {
                "internal_supported": 1.0,
                "internal_original_text_similarity": original_text_similarity,
                "internal_edited_text_similarity": edited_text_similarity,
                "internal_blank_preservation": blank_preservation,
                "internal_prompt_gain": prompt_gain,
                "internal_instruction_score": instruction_score,
            }
        except Exception:
            return self.internal_neutral_score, {
                "internal_supported": 0.0,
                "internal_original_text_similarity": 0.0,
                "internal_edited_text_similarity": 0.0,
                "internal_blank_preservation": 0.0,
                "internal_prompt_gain": 0.0,
                "internal_runtime_error": 1.0,
            }

    def score(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        editor: Any | None = None,
    ) -> SolverResult:
        proxy_result = super().score(proposal, original, edited, editor=editor)
        weighted_total = self.proxy_weight * proxy_result.total_score
        total_weight = self.proxy_weight
        component_scores = dict(proxy_result.component_scores)
        signals = dict(proxy_result.signals)

        if self.spatial_weight > 0.0:
            spatial_score, spatial_signals = self._spatial_score(proposal, original, edited)
            component_scores["spatial_score"] = spatial_score
            signals.update(spatial_signals)
            weighted_total += self.spatial_weight * spatial_score
            total_weight += self.spatial_weight

        if self.cycle_weight > 0.0:
            cycle_score, cycle_signals = self._cycle_score(proposal, original, edited, editor)
            component_scores["cycle_score"] = cycle_score
            signals.update(cycle_signals)
            weighted_total += self.cycle_weight * cycle_score
            total_weight += self.cycle_weight

        if self.internal_weight > 0.0:
            internal_score, internal_signals = self._internal_feature_score(proposal, original, edited, editor)
            component_scores["internal_qwen_score"] = internal_score
            signals.update(internal_signals)
            weighted_total += self.internal_weight * internal_score
            total_weight += self.internal_weight

        total_score = weighted_total / max(total_weight, 1e-6)
        component_scores["hybrid_total_score"] = total_score
        return SolverResult(
            global_score=proxy_result.global_score,
            local_score=proxy_result.local_score,
            total_score=total_score,
            accepted=total_score >= self.acceptance_threshold,
            component_scores=component_scores,
            signals=signals,
        )


class GenericRelativeSelfRewardEvaluator(MultiSignalSolver):
    """Generic self-evolution baseline without editing-specific delta gates.

    This intentionally mirrors the kind of continuous self-reward transplant that works for
    reasoning-style proposer/solver loops but is under-specified for image editing. It ranks K
    candidates by a scalar self-reward and relative group score, without enforcing preservation,
    counterfactual instruction discrimination, or hard edit-success gates.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.top_m = int(config.get("top_m", 1))
        self.quality_threshold = float(config.get("quality_threshold", self.acceptance_threshold))
        self.rank_self_reward_weight = float(config.get("rank_self_reward_weight", 0.70))
        self.rank_relative_weight = float(config.get("rank_relative_weight", 0.30))

    @staticmethod
    def _weighted_mean(weighted_values: list[tuple[float, float]], default: float = 0.0) -> float:
        total_weight = sum(weight for _, weight in weighted_values if weight > 0.0)
        if total_weight <= 0.0:
            return default
        return sum(value * weight for value, weight in weighted_values if weight > 0.0) / total_weight

    def score_group(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited_candidates: list[Image.Image],
        editor: Any | None = None,
    ) -> list[SolverResult]:
        if not edited_candidates:
            return []

        rows: list[dict[str, Any]] = []
        for candidate_index, edited in enumerate(edited_candidates):
            result = super().score(proposal, original, edited, editor=editor)
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "result": result,
                    "self_reward_score": result.total_score,
                }
            )

        base_scores = [row["self_reward_score"] for row in rows]
        min_base = min(base_scores)
        max_base = max(base_scores)
        value_range = max(max_base - min_base, 1e-6)
        for row in rows:
            relative_score = (
                (row["self_reward_score"] - min_base) / value_range if max_base > min_base else 0.5
            )
            quality_score = self._weighted_mean(
                [
                    (row["self_reward_score"], self.rank_self_reward_weight),
                    (relative_score, self.rank_relative_weight),
                ],
                default=row["self_reward_score"],
            )
            row["relative_score"] = relative_score
            row["quality_score"] = quality_score

        ranked_rows = sorted(rows, key=lambda row: row["quality_score"], reverse=True)
        accepted_candidate_indices = {
            row["candidate_index"]
            for row in ranked_rows[: self.top_m]
            if row["quality_score"] >= self.quality_threshold
        }
        rank_by_candidate = {row["candidate_index"]: rank + 1 for rank, row in enumerate(ranked_rows)}

        outputs: list[SolverResult] = []
        for row in rows:
            result = row["result"]
            candidate_index = row["candidate_index"]
            accepted = candidate_index in accepted_candidate_indices
            component_scores = dict(result.component_scores)
            component_scores.update(
                {
                    "generic_self_reward_score": row["self_reward_score"],
                    "relative_group_score": row["relative_score"],
                    "relative_quality_score": row["quality_score"],
                }
            )
            signals = dict(result.signals)
            signals.update(
                {
                    "candidate_index": float(candidate_index),
                    "group_size": float(len(rows)),
                    "feasible": 1.0,
                    "accepted_by_generic_ranker": 1.0 if accepted else 0.0,
                    "generic_rank": float(rank_by_candidate.get(candidate_index, 0)),
                    "generic_self_reward_threshold": self.quality_threshold,
                    "delta_instruction_gate_used": 0.0,
                    "delta_preservation_gate_used": 0.0,
                    "counterfactual_gate_used": 0.0,
                }
            )
            outputs.append(
                SolverResult(
                    global_score=result.global_score,
                    local_score=result.local_score,
                    total_score=row["quality_score"],
                    accepted=accepted,
                    component_scores=component_scores,
                    signals=signals,
                )
            )
        return outputs


class HardGatedRelativeEvaluator(MultiSignalSolver):
    """Evaluator for the research-grade delta-grounded candidate-ranking path."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.instruction_threshold = float(config.get("instruction_threshold", 0.62))
        self.preservation_threshold = float(config.get("preservation_threshold", 0.68))
        self.quality_threshold = float(config.get("quality_threshold", 0.50))
        self.max_disagreement = float(config.get("max_disagreement", 0.35))
        self.top_m = int(config.get("top_m", 1))
        self.counterfactual_distractors = int(config.get("counterfactual_distractors", 3))
        self.rank_spatial_weight = float(config.get("rank_spatial_weight", 0.35))
        self.rank_counterfactual_weight = float(config.get("rank_counterfactual_weight", 0.30))
        self.rank_relative_weight = float(config.get("rank_relative_weight", 0.25))
        self.rank_cycle_weight = float(config.get("rank_cycle_weight", 0.05))
        self.rank_internal_weight = float(config.get("rank_internal_weight", 0.05))
        self.counterfactual_backend = config.get("counterfactual_backend", "auto")
        self.counterfactual_prompt_gain_scale = float(config.get("counterfactual_prompt_gain_scale", 0.08))
        self.require_internal_when_weighted = bool(config.get("require_internal_when_weighted", False))

    def _distractor_definitions(self, proposal: EditProposal) -> list[ProposalDefinition]:
        chosen: list[ProposalDefinition] = []
        inverse_id = proposal.definition.inverse_operation_id or INVERSE_OPERATION_MAP.get(
            proposal.definition.operation_id
        )
        if inverse_id is not None:
            chosen.append(PROPOSAL_BY_ID[inverse_id])
        for definition in PROPOSAL_BANK:
            if definition.operation_id == proposal.definition.operation_id:
                continue
            if definition in chosen:
                continue
            if definition.family == proposal.definition.family or definition.metric == proposal.definition.metric:
                chosen.append(definition)
        for definition in PROPOSAL_BANK:
            if definition.operation_id != proposal.definition.operation_id and definition not in chosen:
                chosen.append(definition)
        return chosen[: self.counterfactual_distractors]

    def describe_distractors(self, proposal: EditProposal) -> list[dict[str, Any]]:
        return [
            {
                "operation_id": definition.operation_id,
                "family": definition.family,
                "instruction": definition.instruction,
                "verifier": definition.verifier,
            }
            for definition in self._distractor_definitions(proposal)
        ]

    def _qwen_prompt_gain(
        self,
        instruction: str,
        original: Image.Image,
        edited: Image.Image,
        editor: Any | None,
    ) -> float | None:
        if not isinstance(editor, QwenEditEditor):
            return None
        try:
            import torch
            import torch.nn.functional as F

            pipe = editor._ensure_pipeline()
            prompt = polish_prompt(instruction, use_prompt_polish=False, image_context=original)
            with torch.no_grad():
                text_feature = extract_qwen_text_features(pipe, prompt)["pooled_embedding"][0].float()
                original_feature = extract_qwen_edit_understanding_features(
                    pipe, prompt, [original]
                )["pooled_embedding"][0].float()
                edited_feature = extract_qwen_edit_understanding_features(
                    pipe, prompt, [edited]
                )["pooled_embedding"][0].float()
            original_similarity = float(
                F.cosine_similarity(original_feature.unsqueeze(0), text_feature.unsqueeze(0)).item()
            )
            edited_similarity = float(
                F.cosine_similarity(edited_feature.unsqueeze(0), text_feature.unsqueeze(0)).item()
            )
            return edited_similarity - original_similarity
        except Exception:
            return None

    def _proxy_counterfactual_score(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
    ) -> tuple[float, dict[str, float]]:
        true_score, _ = self._global_score(proposal.definition, original, edited)
        distractor_scores: list[float] = []
        for definition in self._distractor_definitions(proposal):
            score, _ = self._global_score(definition, original, edited)
            distractor_scores.append(score)
        max_distractor = max(distractor_scores) if distractor_scores else 0.0
        margin = true_score - max_distractor
        score = _clamp(0.5 + 0.5 * margin)
        return score, {
            "counterfactual_backend_proxy": 1.0,
            "counterfactual_backend_internal": 0.0,
            "counterfactual_true_score": true_score,
            "counterfactual_max_distractor_score": max_distractor,
            "counterfactual_margin": margin,
        }

    def _internal_counterfactual_score(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        editor: Any | None,
    ) -> tuple[float, dict[str, float]] | None:
        true_gain = self._qwen_prompt_gain(proposal.instruction, original, edited, editor)
        if true_gain is None:
            return None
        distractor_gains = []
        for definition in self._distractor_definitions(proposal):
            gain = self._qwen_prompt_gain(definition.instruction, original, edited, editor)
            if gain is not None:
                distractor_gains.append(gain)
        max_distractor_gain = max(distractor_gains) if distractor_gains else 0.0
        margin = true_gain - max_distractor_gain
        score = _clamp(0.5 + 0.5 * margin / max(self.counterfactual_prompt_gain_scale, 1e-6))
        return score, {
            "counterfactual_backend_proxy": 0.0,
            "counterfactual_backend_internal": 1.0,
            "counterfactual_true_score": true_gain,
            "counterfactual_max_distractor_score": max_distractor_gain,
            "counterfactual_margin": margin,
            "counterfactual_distractor_count": float(len(distractor_gains)),
        }

    def _counterfactual_score(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        editor: Any | None = None,
    ) -> tuple[float, dict[str, float]]:
        use_internal = self.counterfactual_backend == "internal" or (
            self.counterfactual_backend == "auto" and proposal.definition.verifier == "internal"
        )
        if use_internal:
            internal_result = self._internal_counterfactual_score(proposal, original, edited, editor)
            if internal_result is not None:
                return internal_result
            if self.counterfactual_backend in {"auto", "internal"}:
                return 0.5, {
                    "counterfactual_backend_proxy": 0.0,
                    "counterfactual_backend_internal": 0.0,
                    "counterfactual_true_score": 0.0,
                    "counterfactual_max_distractor_score": 0.0,
                    "counterfactual_margin": 0.0,
                    "counterfactual_runtime_error": 1.0,
                }
        return self._proxy_counterfactual_score(proposal, original, edited)

    @staticmethod
    def _weighted_mean(weighted_values: list[tuple[float, float]], default: float = 0.0) -> float:
        total_weight = sum(weight for _, weight in weighted_values if weight > 0.0)
        if total_weight <= 0.0:
            return default
        return sum(value * weight for value, weight in weighted_values if weight > 0.0) / total_weight

    @staticmethod
    def _disagreement(values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        return float(statistics.pstdev(values))

    def _gate_scores(self, result: SolverResult) -> tuple[float, float]:
        instruction = result.global_score
        if result.signals.get("global_proxy_supported", 1.0) <= 0.0:
            instruction = result.signals.get("internal_instruction_score", result.global_score)
        elif "internal_instruction_score" in result.signals:
            instruction = 0.70 * result.global_score + 0.30 * result.signals["internal_instruction_score"]
        if (
            self.require_internal_when_weighted
            and (self.internal_weight > 0.0 or self.rank_internal_weight > 0.0)
            and result.signals.get("internal_supported", 0.0) <= 0.0
        ):
            instruction = 0.0
        outside_preservation = result.signals.get("spatial_outside_preservation", result.local_score)
        edge_preservation = result.signals.get("edge_preservation_score", result.local_score)
        preservation = 0.55 * edge_preservation + 0.45 * outside_preservation
        return _clamp(instruction), _clamp(preservation)

    def _rank_base_score(self, result: SolverResult, counterfactual_score: float) -> float:
        spatial = result.component_scores.get("spatial_score", result.local_score)
        cycle = result.component_scores.get("cycle_score", self.cycle_neutral_score)
        internal = result.component_scores.get("internal_qwen_score", self.internal_neutral_score)
        return self._weighted_mean(
            [
                (spatial, self.rank_spatial_weight),
                (counterfactual_score, self.rank_counterfactual_weight),
                (cycle, self.rank_cycle_weight),
                (internal, self.rank_internal_weight),
            ],
            default=result.total_score,
        )

    def score_group(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited_candidates: list[Image.Image],
        editor: Any | None = None,
    ) -> list[SolverResult]:
        if not edited_candidates:
            return []

        rows: list[dict[str, Any]] = []
        for candidate_index, edited in enumerate(edited_candidates):
            result = super().score(proposal, original, edited, editor=editor)
            counterfactual_score, counterfactual_signals = self._counterfactual_score(
                proposal, original, edited, editor=editor
            )
            instruction_score, preservation_score = self._gate_scores(result)
            rank_base_score = self._rank_base_score(result, counterfactual_score)
            feasible = instruction_score >= self.instruction_threshold and preservation_score >= self.preservation_threshold
            rows.append(
                {
                    "candidate_index": candidate_index,
                    "result": result,
                    "counterfactual_score": counterfactual_score,
                    "counterfactual_signals": counterfactual_signals,
                    "instruction_score": instruction_score,
                    "preservation_score": preservation_score,
                    "rank_base_score": rank_base_score,
                    "feasible": feasible,
                }
            )

        base_values = [row["rank_base_score"] for row in rows]
        min_base = min(base_values)
        max_base = max(base_values)
        value_range = max(max_base - min_base, 1e-6)
        for row in rows:
            relative_score = (row["rank_base_score"] - min_base) / value_range if max_base > min_base else 0.5
            result = row["result"]
            spatial = result.component_scores.get("spatial_score", result.local_score)
            cycle = result.component_scores.get("cycle_score", self.cycle_neutral_score)
            internal = result.component_scores.get("internal_qwen_score", self.internal_neutral_score)
            quality_score = self._weighted_mean(
                [
                    (spatial, self.rank_spatial_weight),
                    (row["counterfactual_score"], self.rank_counterfactual_weight),
                    (relative_score, self.rank_relative_weight),
                    (cycle, self.rank_cycle_weight),
                    (internal, self.rank_internal_weight),
                ],
                default=row["rank_base_score"],
            )
            component_values = [
                row["instruction_score"],
                row["preservation_score"],
                spatial,
                row["counterfactual_score"],
                relative_score,
            ]
            if "cycle_score" in result.component_scores:
                component_values.append(cycle)
            if "internal_qwen_score" in result.component_scores:
                component_values.append(internal)
            row["relative_score"] = relative_score
            row["quality_score"] = quality_score
            row["disagreement"] = self._disagreement(component_values)

        feasible_rows = [row for row in rows if row["feasible"]]
        ranked_rows = sorted(feasible_rows, key=lambda row: row["quality_score"], reverse=True)
        accepted_candidate_indices = {
            row["candidate_index"]
            for row in ranked_rows[: self.top_m]
            if row["quality_score"] >= self.quality_threshold and row["disagreement"] <= self.max_disagreement
        }
        rank_by_candidate = {row["candidate_index"]: rank + 1 for rank, row in enumerate(ranked_rows)}

        outputs: list[SolverResult] = []
        for row in rows:
            result = row["result"]
            candidate_index = row["candidate_index"]
            rank = rank_by_candidate.get(candidate_index, 0)
            accepted = candidate_index in accepted_candidate_indices
            component_scores = dict(result.component_scores)
            component_scores.update(
                {
                    "instruction_gate_score": row["instruction_score"],
                    "preservation_gate_score": row["preservation_score"],
                    "counterfactual_score": row["counterfactual_score"],
                    "relative_group_score": row["relative_score"],
                    "rank_base_score": row["rank_base_score"],
                    "relative_quality_score": row["quality_score"],
                    "evaluator_disagreement": row["disagreement"],
                }
            )
            signals = dict(result.signals)
            signals.update(row["counterfactual_signals"])
            signals.update(
                {
                    "candidate_index": float(candidate_index),
                    "group_size": float(len(rows)),
                    "feasible": 1.0 if row["feasible"] else 0.0,
                    "accepted_by_ranker": 1.0 if accepted else 0.0,
                    "feasible_rank": float(rank),
                    "instruction_threshold": self.instruction_threshold,
                    "preservation_threshold": self.preservation_threshold,
                    "quality_threshold": self.quality_threshold,
                    "max_disagreement": self.max_disagreement,
                }
            )
            outputs.append(
                SolverResult(
                    global_score=result.global_score,
                    local_score=result.local_score,
                    total_score=row["quality_score"],
                    accepted=accepted,
                    component_scores=component_scores,
                    signals=signals,
                )
            )
        return outputs


class InternalContrastiveEditPreservationEvaluator(HardGatedRelativeEvaluator):
    """Internal-only CEPR evaluator for the main self-evolving method.

    The reward is intentionally constrained instead of additive:

    R(y | x, c) = sqrt(E(y, x, c) * P(y, x)) when edit, preservation, and
    validity gates pass; otherwise R=0.

    E is an internal contrastive prompt-gain score against counterfactual
    instructions. P is internal source preservation from Qwen semantic features
    and Qwen VAE latent locality. Q is a hard validity gate over edit-region
    plausibility and excessive latent drift. No external VLM, detector, CLIP, or
    OCR model is used.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.edit_threshold = float(config.get("edit_threshold", 0.53))
        self.preservation_threshold = float(config.get("preservation_threshold", 0.60))
        self.validity_threshold = float(config.get("validity_threshold", 0.35))
        self.reward_threshold = float(config.get("reward_threshold", config.get("quality_threshold", 0.52)))
        self.top_m = int(config.get("top_m", 1))
        self.require_internal_components = bool(config.get("require_internal_components", True))
        self.edit_temperature = float(config.get("edit_temperature", 0.04))
        self.true_gain_temperature = float(config.get("true_gain_temperature", 0.04))
        self.min_true_gain = float(config.get("min_true_gain", 0.0))
        self.semantic_preservation_temperature = float(config.get("semantic_preservation_temperature", 0.08))
        self.latent_resolution = int(config.get("latent_resolution", 512))
        self.latent_mask_std_scale = float(config.get("latent_mask_std_scale", 0.75))
        self.latent_mask_min_delta = float(config.get("latent_mask_min_delta", 0.03))
        self.latent_preservation_temperature = float(config.get("latent_preservation_temperature", 0.12))
        self.max_total_latent_delta = float(config.get("max_total_latent_delta", 0.45))
        self.latent_drift_temperature = float(config.get("latent_drift_temperature", 0.15))
        self.taxonomy_enabled = bool(config.get("taxonomy_enabled", True))
        self.taxonomy_required = bool(config.get("taxonomy_required", False))
        self.taxonomy_threshold = float(config.get("taxonomy_threshold", 0.40))
        self.taxonomy_temperature = float(config.get("taxonomy_temperature", 0.06))
        self.taxonomy_source_drop_temperature = float(config.get("taxonomy_source_drop_temperature", 0.06))
        self.taxonomy_distractor_temperature = float(config.get("taxonomy_distractor_temperature", 0.05))
        self.max_dynamic_distractors = int(config.get("max_dynamic_distractors", 4))
        self.empty_cache_per_candidate = bool(config.get("empty_cache_per_candidate", True))
        self.retry_cpu_on_oom = bool(config.get("retry_cpu_on_oom", True))

    @staticmethod
    def _cosine_similarity(value_a: Any, value_b: Any) -> float:
        import torch.nn.functional as F

        return float(F.cosine_similarity(value_a.unsqueeze(0), value_b.unsqueeze(0)).item())

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        message = str(exc).lower()
        return "cuda out of memory" in message or exc.__class__.__name__ == "OutOfMemoryError"

    def _get_internal_pipe(self, editor: Any | None) -> Any:
        if not isinstance(editor, QwenEditEditor):
            raise ValueError("internal_cepr requires QwenEditEditor so rewards can use Qwen internal features.")
        return editor._ensure_pipeline()

    def _cached_text_feature(self, pipe: Any, prompt: str, cache: dict[tuple[str, str], Any]):
        key = ("text", prompt)
        if key not in cache:
            try:
                cache[key] = extract_qwen_text_features(pipe, prompt)["pooled_embedding"][0].float()
            except Exception:
                # Some public Qwen wrappers expose only the multimodal encoder path.
                # A neutral image keeps the reward internal while still yielding a prompt-conditioned anchor.
                neutral_image = Image.new("RGB", (self.latent_resolution, self.latent_resolution), (127, 127, 127))
                cache[key] = extract_qwen_edit_understanding_features(pipe, prompt, [neutral_image])[
                    "pooled_embedding"
                ][0].float()
        return cache[key]

    def _cached_understanding_feature(
        self,
        pipe: Any,
        prompt: str,
        image_label: str,
        image: Image.Image,
        cache: dict[tuple[str, str], Any],
    ):
        key = ("understanding", f"{image_label}|{prompt}")
        if key not in cache:
            cache[key] = extract_qwen_edit_understanding_features(pipe, prompt, [image])["pooled_embedding"][0].float()
        return cache[key]

    def _cached_vae_latents(
        self,
        pipe: Any,
        image_label: str,
        image: Image.Image,
        cache: dict[tuple[str, str], Any],
    ):
        key = ("vae_latents", image_label)
        if key not in cache:
            cache[key] = extract_qwen_vae_latents(pipe, image, size=self.latent_resolution)
        return cache[key]

    def _prompt_gain_cached(
        self,
        pipe: Any,
        instruction: str,
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        cache: dict[tuple[str, str], Any],
    ) -> float:
        prompt = polish_prompt(instruction, use_prompt_polish=False, image_context=original)
        text_feature = self._cached_text_feature(pipe, prompt, cache)
        original_feature = self._cached_understanding_feature(pipe, prompt, "original", original, cache)
        edited_feature = self._cached_understanding_feature(pipe, prompt, f"candidate:{candidate_index}", edited, cache)
        original_similarity = self._cosine_similarity(original_feature, text_feature)
        edited_similarity = self._cosine_similarity(edited_feature, text_feature)
        return edited_similarity - original_similarity

    @staticmethod
    def _join_terms(*values: Any) -> str:
        terms = [str(value).strip() for value in values if value not in (None, "", [])]
        return " ".join(terms).strip()

    def _taxonomy_prompt_bundle(self, proposal: EditProposal) -> dict[str, Any]:
        spec = normalize_structured_edit(
            proposal.structured_edit,
            instruction=proposal.instruction,
            family=proposal.definition.family,
        )
        if not proposal.structured_edit:
            return {"supported": False, "spec": spec}

        edit_type = str(spec.get("edit_type", "local_enhancement"))
        source_object = spec.get("source_object")
        target_object = spec.get("target_object")
        source_attribute = spec.get("source_attribute") or spec.get("source_material") or spec.get("source_style")
        target_attribute = spec.get("target_attribute") or spec.get("target_material") or spec.get("target_style")
        target_location = spec.get("target_location")
        target_region = spec.get("target_region", "target region")

        main_prompts = [spec.get("instruction", proposal.instruction), proposal.instruction]
        target_prompts: list[str] = []
        source_drop_prompts: list[str] = []
        wrong_prompts: list[str] = []

        if edit_type == "object_replacement":
            if target_object:
                target_prompts.append(f"{target_region} contains {target_object}")
                main_prompts.append(f"{source_object or 'the original object'} is replaced by {target_object}")
            if source_object:
                source_drop_prompts.append(f"{target_region} still contains {source_object}")
                wrong_prompts.append(f"{source_object} is still present and unchanged")
            for wrong in ("a dog", "a chair", "a car", "a plant"):
                if target_object and wrong.lower() not in str(target_object).lower():
                    wrong_prompts.append(f"{source_object or 'the object'} is replaced by {wrong}")
        elif edit_type == "object_removal":
            if source_object:
                source_drop_prompts.append(f"{target_region} contains {source_object}")
                main_prompts.append(f"{source_object} has been removed from {target_region}")
            target_prompts.append(f"{target_region} is cleanly filled after object removal")
        elif edit_type == "object_addition":
            if target_object:
                target_prompts.append(f"{target_object} has been added to {target_region}")
                main_prompts.append(f"{target_region} now contains {target_object}")
            wrong_prompts.append(f"nothing new has been added to {target_region}")
        elif edit_type == "spatial_move":
            moved = source_object or "the target object"
            if target_location:
                target_prompts.append(f"{moved} is {target_location}")
                main_prompts.append(f"{moved} has moved to {target_location}")
            if spec.get("source_location"):
                source_drop_prompts.append(f"{moved} remains {spec['source_location']}")
            wrong_prompts.extend([f"{moved} stays in its original position", f"{moved} is in the wrong location"])
        elif edit_type in {"attribute_change", "color_change", "material_change", "style_transfer"}:
            target = self._join_terms(target_attribute, target_object or source_object)
            source = self._join_terms(source_attribute, source_object)
            if target:
                target_prompts.append(f"{target_region} has {target}")
                main_prompts.append(f"{source_object or 'the target'} is changed to {target}")
            if source:
                source_drop_prompts.append(f"{target_region} still has {source}")
            if target_attribute:
                wrong_prompts.append(f"{target_region} does not have {target_attribute}")
        elif edit_type == "background_change":
            if target_attribute or target_object:
                target_prompts.append(f"background changed to {self._join_terms(target_attribute, target_object)}")
            source_drop_prompts.append("the background is unchanged")
        else:
            target_prompts.append(proposal.instruction)

        preserve_prompts = [
            f"unrelated content is preserved: {item}"
            for item in spec.get("preserve", [])[: self.max_dynamic_distractors]
        ]
        unique = lambda items: list(dict.fromkeys(item for item in items if item))
        return {
            "supported": True,
            "spec": spec,
            "main_prompts": unique(main_prompts),
            "target_prompts": unique(target_prompts),
            "source_drop_prompts": unique(source_drop_prompts),
            "wrong_prompts": unique(wrong_prompts)[: self.max_dynamic_distractors],
            "preserve_prompts": unique(preserve_prompts),
        }

    @staticmethod
    def _geometric_mean(values: list[float], default: float = 1.0) -> float:
        filtered = [_clamp(value) for value in values if math.isfinite(float(value))]
        if not filtered:
            return default
        product = 1.0
        for value in filtered:
            product *= max(value, 1e-6)
        return _clamp(product ** (1.0 / len(filtered)))

    def _taxonomy_score(
        self,
        pipe: Any,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        cache: dict[tuple[str, str], Any],
    ) -> tuple[float, dict[str, float]]:
        if not self.taxonomy_enabled:
            return 1.0, {"cepr_taxonomy_supported": 0.0, "cepr_taxonomy_score": 1.0}

        bundle = self._taxonomy_prompt_bundle(proposal)
        if not bundle.get("supported"):
            neutral = 0.0 if self.taxonomy_required else 1.0
            return neutral, {
                "cepr_taxonomy_supported": 0.0,
                "cepr_taxonomy_score": neutral,
                "cepr_taxonomy_required": 1.0 if self.taxonomy_required else 0.0,
            }

        main_gains = [
            self._prompt_gain_cached(pipe, prompt, original, edited, candidate_index, cache)
            for prompt in bundle.get("main_prompts", [])
        ]
        target_gains = [
            self._prompt_gain_cached(pipe, prompt, original, edited, candidate_index, cache)
            for prompt in bundle.get("target_prompts", [])
        ]
        source_drops = [
            -self._prompt_gain_cached(pipe, prompt, original, edited, candidate_index, cache)
            for prompt in bundle.get("source_drop_prompts", [])
        ]
        wrong_gains = [
            self._prompt_gain_cached(pipe, prompt, original, edited, candidate_index, cache)
            for prompt in bundle.get("wrong_prompts", [])
        ]

        main_gain = statistics.mean(main_gains) if main_gains else 0.0
        target_gain = statistics.mean(target_gains) if target_gains else main_gain
        source_drop = statistics.mean(source_drops) if source_drops else 0.0
        max_wrong_gain = max(wrong_gains) if wrong_gains else 0.0
        contrastive_margin = main_gain - max_wrong_gain

        main_score = _sigmoid(main_gain / max(self.taxonomy_temperature, 1e-6))
        target_score = _sigmoid(target_gain / max(self.taxonomy_temperature, 1e-6))
        contrastive_score = (
            _sigmoid(contrastive_margin / max(self.taxonomy_distractor_temperature, 1e-6))
            if wrong_gains
            else 1.0
        )
        components = [main_score, target_score, contrastive_score]
        if source_drops:
            source_drop_score = _sigmoid(source_drop / max(self.taxonomy_source_drop_temperature, 1e-6))
            components.append(source_drop_score)
        else:
            source_drop_score = 1.0

        taxonomy_score = self._geometric_mean(components)
        return taxonomy_score, {
            "cepr_taxonomy_supported": 1.0,
            "cepr_taxonomy_score": taxonomy_score,
            "cepr_taxonomy_required": 1.0 if self.taxonomy_required else 0.0,
            "cepr_taxonomy_main_gain": main_gain,
            "cepr_taxonomy_target_gain": target_gain,
            "cepr_taxonomy_source_drop": source_drop,
            "cepr_taxonomy_max_wrong_gain": max_wrong_gain,
            "cepr_taxonomy_contrastive_margin": contrastive_margin,
            "cepr_taxonomy_main_score": main_score,
            "cepr_taxonomy_target_score": target_score,
            "cepr_taxonomy_source_drop_score": source_drop_score,
            "cepr_taxonomy_contrastive_score": contrastive_score,
        }

    def _edit_specificity(
        self,
        pipe: Any,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        cache: dict[tuple[str, str], Any],
    ) -> tuple[float, dict[str, float]]:
        true_gain = self._prompt_gain_cached(pipe, proposal.instruction, original, edited, candidate_index, cache)
        distractor_gains = [
            self._prompt_gain_cached(pipe, definition.instruction, original, edited, candidate_index, cache)
            for definition in self._distractor_definitions(proposal)
        ]
        max_distractor_gain = max(distractor_gains) if distractor_gains else 0.0
        contrastive_margin = true_gain - max_distractor_gain
        contrastive_score = _sigmoid(contrastive_margin / max(self.edit_temperature, 1e-6))
        absolute_score = _sigmoid((true_gain - self.min_true_gain) / max(self.true_gain_temperature, 1e-6))
        edit_specificity = math.sqrt(max(contrastive_score * absolute_score, 0.0))
        return edit_specificity, {
            "cepr_true_prompt_gain": true_gain,
            "cepr_max_distractor_gain": max_distractor_gain,
            "cepr_contrastive_margin": contrastive_margin,
            "cepr_contrastive_score": contrastive_score,
            "cepr_absolute_edit_score": absolute_score,
            "cepr_distractor_count": float(len(distractor_gains)),
        }

    def _semantic_preservation(
        self,
        pipe: Any,
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        cache: dict[tuple[str, str], Any],
    ) -> tuple[float, dict[str, float]]:
        original_feature = self._cached_understanding_feature(pipe, " ", "original", original, cache)
        edited_feature = self._cached_understanding_feature(pipe, " ", f"candidate:{candidate_index}", edited, cache)
        cosine = self._cosine_similarity(original_feature, edited_feature)
        cosine_01 = 0.5 * (1.0 + cosine)
        semantic_preservation = math.exp(-(1.0 - cosine_01) / max(self.semantic_preservation_temperature, 1e-6))
        return _clamp(semantic_preservation), {
            "cepr_semantic_preservation_cosine": cosine,
            "cepr_semantic_preservation_score": _clamp(semantic_preservation),
        }

    def _latent_locality(
        self,
        pipe: Any,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        cache: dict[tuple[str, str], Any],
    ) -> tuple[float, float, dict[str, float]]:
        import torch

        original_latents = self._cached_vae_latents(pipe, "original", original, cache)
        edited_latents = self._cached_vae_latents(pipe, f"candidate:{candidate_index}", edited, cache)
        delta = (original_latents - edited_latents).abs().mean(dim=1).squeeze(0)
        normalizer = 0.5 * (
            original_latents.abs().mean(dim=1).squeeze(0) + edited_latents.abs().mean(dim=1).squeeze(0)
        )
        relative_delta = delta / normalizer.clamp_min(1e-4)
        mean_delta = float(relative_delta.mean().item())
        std_delta = float(relative_delta.std(unbiased=False).item())
        threshold = max(self.latent_mask_min_delta, mean_delta + self.latent_mask_std_scale * std_delta)
        mask = relative_delta > threshold
        changed_fraction_value = float(mask.float().mean().item())
        if mask.any():
            inside_delta = float(relative_delta[mask].mean().item())
        else:
            inside_delta = 0.0
        if (~mask).any():
            outside_delta = float(relative_delta[~mask].mean().item())
        else:
            outside_delta = mean_delta

        outside_preservation = math.exp(-outside_delta / max(self.latent_preservation_temperature, 1e-6))
        region_score = _mean_changed_fraction_score(changed_fraction_value, proposal.definition.expected_changed_fraction)
        excess_drift = max(0.0, mean_delta - self.max_total_latent_delta)
        drift_score = math.exp(-excess_drift / max(self.latent_drift_temperature, 1e-6))
        validity = math.sqrt(max(region_score * drift_score, 0.0))
        return _clamp(outside_preservation), _clamp(validity), {
            "cepr_latent_outside_preservation": _clamp(outside_preservation),
            "cepr_latent_region_score": region_score,
            "cepr_latent_validity_score": _clamp(validity),
            "cepr_latent_changed_fraction": changed_fraction_value,
            "cepr_latent_inside_delta": inside_delta,
            "cepr_latent_outside_delta": outside_delta,
            "cepr_latent_total_delta": mean_delta,
            "cepr_latent_delta_std": std_delta,
            "cepr_latent_mask_threshold": threshold,
            "cepr_latent_drift_score": _clamp(drift_score),
        }

    def _preservation_and_validity(
        self,
        pipe: Any,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        cache: dict[tuple[str, str], Any],
    ) -> tuple[float, float, dict[str, float]]:
        semantic_preservation, semantic_signals = self._semantic_preservation(
            pipe, original, edited, candidate_index, cache
        )
        latent_preservation, validity, latent_signals = self._latent_locality(
            pipe, proposal, original, edited, candidate_index, cache
        )
        preservation = math.sqrt(max(semantic_preservation * latent_preservation, 0.0))
        signals = {
            **semantic_signals,
            **latent_signals,
            "cepr_preservation_score": _clamp(preservation),
            "cepr_validity_score": _clamp(validity),
        }
        return _clamp(preservation), _clamp(validity), signals

    def _failure_result(self, error_name: str, candidate_index: int, group_size: int) -> SolverResult:
        signals = {
            "candidate_index": float(candidate_index),
            "group_size": float(group_size),
            "feasible": 0.0,
            "accepted_by_ranker": 0.0,
            "feasible_rank": 0.0,
            "cepr_internal_supported": 0.0,
            error_name: 1.0,
            "cepr_edit_threshold": self.edit_threshold,
            "cepr_preservation_threshold": self.preservation_threshold,
            "cepr_validity_threshold": self.validity_threshold,
            "cepr_reward_threshold": self.reward_threshold,
        }
        return SolverResult(
            global_score=0.0,
            local_score=0.0,
            total_score=0.0,
            accepted=False,
            component_scores={
                "cepr_edit_specificity": 0.0,
                "cepr_taxonomy": 0.0,
                "cepr_semantic_edit": 0.0,
                "cepr_preservation": 0.0,
                "cepr_validity": 0.0,
                "cepr_reward": 0.0,
            },
            signals=signals,
        )

    def _candidate_error_row(self, exc: Exception, candidate_index: int, scoring_device: str) -> dict[str, Any]:
        return {
            "candidate_index": candidate_index,
            "edit_specificity": 0.0,
            "taxonomy_score": 0.0,
            "semantic_edit": 0.0,
            "preservation": 0.0,
            "validity": 0.0,
            "reward": 0.0,
            "raw_reward": 0.0,
            "feasible": False,
            "signals": {
                "cepr_internal_supported": 0.0,
                "cepr_candidate_runtime_error": 1.0,
                "cepr_candidate_runtime_error_type": exc.__class__.__name__,
                "cepr_candidate_runtime_error_message": str(exc)[:500],
                "cepr_scoring_device": scoring_device,
            },
        }

    def _score_candidate_row(
        self,
        pipe: Any,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        candidate_index: int,
        scoring_device: str,
    ) -> dict[str, Any]:
        cache: dict[tuple[str, str], Any] = {}
        try:
            edit_specificity, edit_signals = self._edit_specificity(
                pipe, proposal, original, edited, candidate_index, cache
            )
            taxonomy_score, taxonomy_signals = self._taxonomy_score(
                pipe, proposal, original, edited, candidate_index, cache
            )
            preservation, validity, preservation_signals = self._preservation_and_validity(
                pipe, proposal, original, edited, candidate_index, cache
            )
            semantic_edit = (
                math.sqrt(max(edit_specificity * taxonomy_score, 0.0))
                if taxonomy_signals.get("cepr_taxonomy_supported", 0.0) > 0.0 or self.taxonomy_required
                else edit_specificity
            )
            reward = math.sqrt(max(semantic_edit * preservation, 0.0))
            feasible = (
                edit_specificity >= self.edit_threshold
                and (
                    taxonomy_score >= self.taxonomy_threshold
                    or (
                        taxonomy_signals.get("cepr_taxonomy_supported", 0.0) <= 0.0
                        and not self.taxonomy_required
                    )
                )
                and preservation >= self.preservation_threshold
                and validity >= self.validity_threshold
                and reward >= self.reward_threshold
            )
            return {
                "candidate_index": candidate_index,
                "edit_specificity": edit_specificity,
                "taxonomy_score": taxonomy_score,
                "semantic_edit": semantic_edit,
                "preservation": preservation,
                "validity": validity,
                "reward": reward if feasible else 0.0,
                "raw_reward": reward,
                "feasible": feasible,
                "signals": {
                    **edit_signals,
                    **taxonomy_signals,
                    **preservation_signals,
                    "cepr_internal_supported": 1.0,
                    "cepr_scoring_device": scoring_device,
                },
            }
        finally:
            cache.clear()
            if self.empty_cache_per_candidate:
                QwenEditEditor._empty_cuda_cache()

    def score(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited: Image.Image,
        editor: Any | None = None,
    ) -> SolverResult:
        return self.score_group(proposal, original, [edited], editor=editor)[0]

    def score_group(
        self,
        proposal: EditProposal,
        original: Image.Image,
        edited_candidates: list[Image.Image],
        editor: Any | None = None,
    ) -> list[SolverResult]:
        if not edited_candidates:
            return []
        rows: list[dict[str, Any]] = []
        for candidate_index, edited in enumerate(edited_candidates):
            try:
                if hasattr(editor, "prepare_for_internal_scoring"):
                    editor.prepare_for_internal_scoring()
                pipe = self._get_internal_pipe(editor)
                rows.append(self._score_candidate_row(pipe, proposal, original, edited, candidate_index, "cuda"))
            except Exception as exc:
                if self.retry_cpu_on_oom and self._is_cuda_oom(exc) and hasattr(editor, "prepare_for_internal_scoring"):
                    try:
                        QwenEditEditor._empty_cuda_cache()
                        editor.prepare_for_internal_scoring(scoring_device="cpu")
                        pipe = self._get_internal_pipe(editor)
                        row = self._score_candidate_row(pipe, proposal, original, edited, candidate_index, "cpu")
                        row["signals"]["cepr_gpu_oom_recovered"] = 1.0
                        rows.append(row)
                        continue
                    except Exception as cpu_exc:
                        row = self._candidate_error_row(cpu_exc, candidate_index, "cpu")
                        row["signals"]["cepr_gpu_oom_before_cpu_retry"] = 1.0
                        rows.append(row)
                        continue
                rows.append(self._candidate_error_row(exc, candidate_index, "cuda"))

        ranked_rows = sorted(
            [row for row in rows if row["feasible"]],
            key=lambda row: row["reward"],
            reverse=True,
        )
        accepted_candidate_indices = {row["candidate_index"] for row in ranked_rows[: self.top_m]}
        rank_by_candidate = {row["candidate_index"]: rank + 1 for rank, row in enumerate(ranked_rows)}

        outputs: list[SolverResult] = []
        for row in rows:
            candidate_index = row["candidate_index"]
            accepted = candidate_index in accepted_candidate_indices
            rank = rank_by_candidate.get(candidate_index, 0)
            reward = row["reward"]
            signals = dict(row["signals"])
            signals.update(
                {
                    "candidate_index": float(candidate_index),
                    "group_size": float(len(rows)),
                    "feasible": 1.0 if row["feasible"] else 0.0,
                    "accepted_by_ranker": 1.0 if accepted else 0.0,
                    "feasible_rank": float(rank),
                    "cepr_edit_threshold": self.edit_threshold,
                    "cepr_preservation_threshold": self.preservation_threshold,
                    "cepr_validity_threshold": self.validity_threshold,
                    "cepr_reward_threshold": self.reward_threshold,
                }
            )
            component_scores = {
                "cepr_edit_specificity": row["edit_specificity"],
                "cepr_taxonomy": row["taxonomy_score"],
                "cepr_semantic_edit": row["semantic_edit"],
                "cepr_preservation": row["preservation"],
                "cepr_validity": row["validity"],
                "cepr_raw_reward": row["raw_reward"],
                "cepr_reward": reward,
            }
            outputs.append(
                SolverResult(
                    global_score=row["edit_specificity"],
                    local_score=row["preservation"],
                    total_score=reward,
                    accepted=accepted,
                    component_scores=component_scores,
                    signals=signals,
                )
            )
        return outputs


def build_proposer(config: dict[str, Any]):
    backend = config.get("backend", "scripted")
    if backend == "scripted":
        return ScriptedProposer(config)
    if backend == "internal_qwen":
        return InternalQwenProposer()
    if backend in {"trainable_qwen_image_edit", "qwen_image_edit_lora", "trainable_qwen_vl", "qwen_vl_lora"}:
        return TrainableQwenVLProposer(config)
    raise ValueError(f"Unsupported proposer backend: {backend}")


def build_editor(config: dict[str, Any]):
    backend = config.get("backend", "qwen_edit")
    if backend == "pillow_demo":
        return PillowPrototypeEditor()
    if backend == "qwen_edit":
        return QwenEditEditor(config)
    raise ValueError(f"Unsupported editor backend: {backend}")


def build_evaluator(config: dict[str, Any]):
    backend = config.get("backend", "stat")
    if backend == "stat":
        return StatSolver(config)
    if backend == "internal_qwen":
        evaluator_config = dict(config)
        evaluator_config.setdefault("proxy_weight", 0.7)
        evaluator_config.setdefault("internal_weight", 0.3)
        return MultiSignalSolver(evaluator_config)
    if backend == "hybrid":
        return MultiSignalSolver(config)
    if backend in {"generic_relative_self_reward", "evolmm_style"}:
        evaluator_config = dict(config)
        evaluator_config.setdefault("global_weight", 1.0)
        evaluator_config.setdefault("local_weight", 0.0)
        evaluator_config.setdefault("proxy_weight", 1.0)
        evaluator_config.setdefault("spatial_weight", 0.0)
        evaluator_config.setdefault("cycle_weight", 0.0)
        evaluator_config.setdefault("internal_weight", 0.0)
        return GenericRelativeSelfRewardEvaluator(evaluator_config)
    if backend in {"hard_gated_relative", "delta_ranker"}:
        evaluator_config = dict(config)
        evaluator_config.setdefault("spatial_weight", 0.20)
        evaluator_config.setdefault("cycle_weight", 0.0)
        evaluator_config.setdefault("internal_weight", 0.0)
        return HardGatedRelativeEvaluator(evaluator_config)
    if backend in {"internal_cepr", "contrastive_edit_preservation"}:
        evaluator_config = dict(config)
        evaluator_config.setdefault("counterfactual_backend", "internal")
        evaluator_config.setdefault("counterfactual_distractors", 4)
        evaluator_config.setdefault("top_m", 1)
        return InternalContrastiveEditPreservationEvaluator(evaluator_config)
    raise ValueError(f"Unsupported evaluator backend: {backend}")


def build_solver(config: dict[str, Any]):
    """Backward-compatible name for historical configs and scripts."""
    return build_evaluator(config)
