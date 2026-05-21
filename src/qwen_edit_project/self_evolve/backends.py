from __future__ import annotations

import random
import statistics
from typing import Any

from PIL import Image, ImageEnhance, ImageOps

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

    def edit_image(self, image: Image.Image, instruction: str, operation_id: str | None = None) -> Image.Image:
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


def build_proposer(config: dict[str, Any]):
    backend = config.get("backend", "scripted")
    if backend == "scripted":
        return ScriptedProposer(config)
    if backend == "internal_qwen":
        return InternalQwenProposer()
    raise ValueError(f"Unsupported proposer backend: {backend}")


def build_editor(config: dict[str, Any]):
    backend = config.get("backend", "qwen_edit")
    if backend == "pillow_demo":
        return PillowPrototypeEditor()
    if backend == "qwen_edit":
        return QwenEditEditor(config)
    raise ValueError(f"Unsupported editor backend: {backend}")


def build_solver(config: dict[str, Any]):
    backend = config.get("backend", "stat")
    if backend == "stat":
        return StatSolver(config)
    if backend == "internal_qwen":
        solver_config = dict(config)
        solver_config.setdefault("proxy_weight", 0.7)
        solver_config.setdefault("internal_weight", 0.3)
        return MultiSignalSolver(solver_config)
    if backend == "hybrid":
        return MultiSignalSolver(config)
    if backend in {"generic_relative_self_reward", "evolmm_style"}:
        solver_config = dict(config)
        solver_config.setdefault("global_weight", 1.0)
        solver_config.setdefault("local_weight", 0.0)
        solver_config.setdefault("proxy_weight", 1.0)
        solver_config.setdefault("spatial_weight", 0.0)
        solver_config.setdefault("cycle_weight", 0.0)
        solver_config.setdefault("internal_weight", 0.0)
        return GenericRelativeSelfRewardEvaluator(solver_config)
    if backend in {"hard_gated_relative", "delta_ranker"}:
        solver_config = dict(config)
        solver_config.setdefault("spatial_weight", 0.20)
        solver_config.setdefault("cycle_weight", 0.0)
        solver_config.setdefault("internal_weight", 0.0)
        return HardGatedRelativeEvaluator(solver_config)
    raise ValueError(f"Unsupported solver backend: {backend}")
