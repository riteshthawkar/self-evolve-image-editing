from __future__ import annotations

from PIL import Image, ImageDraw

from qwen_edit_project.self_evolve.image_metrics import (
    box_mask_from_boxes,
    masked_region_statistics,
)
from qwen_edit_project.self_evolve.backends import InternalRubricCEPREvaluator
from qwen_edit_project.self_evolve.types import EditProposal, ProposalDefinition


def _base_image() -> Image.Image:
    image = Image.new("RGB", (128, 128), (128, 128, 128))
    draw = ImageDraw.Draw(image)
    draw.rectangle((44, 44, 84, 84), fill=(40, 40, 40))
    return image


def test_masked_region_statistics_separates_target_edit_from_outside_damage() -> None:
    original = _base_image()
    clean_edit = original.copy()
    draw = ImageDraw.Draw(clean_edit)
    draw.rectangle((44, 44, 84, 84), fill=(220, 220, 220))

    damaged_edit = clean_edit.copy()
    draw = ImageDraw.Draw(damaged_edit)
    draw.rectangle((0, 0, 24, 24), fill=(255, 0, 0))

    mask = box_mask_from_boxes(original.size, [(44, 44, 84, 84)], size=(128, 128))
    clean = masked_region_statistics(original, clean_edit, mask)
    damaged = masked_region_statistics(original, damaged_edit, mask)

    assert clean["target_change"] > 0.45
    assert clean["outside_change"] < 0.01
    assert damaged["target_change"] == clean["target_change"]
    assert damaged["outside_change"] > clean["outside_change"]
    assert damaged["outside_changed_fraction"] > clean["outside_changed_fraction"]


def test_conservative_region_gate_rejects_outside_damage() -> None:
    class DummyEvaluator(InternalRubricCEPREvaluator):
        def _detect_object_boxes(self, image, phrase, cache=None):  # noqa: ANN001
            return [{"score": 0.9, "box": (44, 44, 84, 84), "label": phrase}]

    original = _base_image()
    clean_edit = original.copy()
    draw = ImageDraw.Draw(clean_edit)
    draw.rectangle((44, 44, 84, 84), fill=(220, 220, 220))

    damaged_edit = clean_edit.copy()
    draw = ImageDraw.Draw(damaged_edit)
    draw.rectangle((0, 0, 48, 48), fill=(255, 0, 0))

    evaluator = DummyEvaluator(
        {
            "conservative_region_reward_enabled": True,
            "object_detector_enabled": True,
            "conservative_region_edit_types": ["object_removal"],
            "conservative_region_require_mask_edit_types": ["object_removal"],
            "conservative_region_max_outside_change": 0.05,
            "conservative_region_max_outside_changed_fraction": 0.20,
            "conservative_region_min_reward": 0.30,
            "conservative_region_min_outside_preservation": 0.50,
            "conservative_region_min_target_change_score": 0.35,
        }
    )
    spec = {"edit_type": "object_removal", "source_object": "box", "target_region": "box"}

    _, clean_signals = evaluator._conservative_region_score(spec, original, clean_edit)
    _, damaged_signals = evaluator._conservative_region_score(spec, original, damaged_edit)

    assert clean_signals["conservative_region_gate_pass"] == 1.0
    assert damaged_signals["conservative_region_gate_pass"] == 0.0
    assert damaged_signals["conservative_region_reject_reason"] == "outside_change"


def test_object_detector_contract_fail_closed_on_detector_error() -> None:
    class FailingDetectorEvaluator(InternalRubricCEPREvaluator):
        def _detect_object_boxes(self, image, phrase, cache=None):  # noqa: ANN001
            raise RuntimeError("detector unavailable")

    evaluator = FailingDetectorEvaluator(
        {
            "object_detector_enabled": True,
            "object_detector_edit_types": ["object_removal"],
        }
    )
    scores, signals = evaluator._object_detector_contract_score(
        {"edit_type": "object_removal", "source_object": "box"},
        _base_image(),
        _base_image(),
    )

    assert scores["object_detector_contract"] == 0.0
    assert signals["object_detector_supported"] == 1.0
    assert signals["object_detector_error"] == 1.0
    assert signals["object_detector_gate_pass"] == 0.0


def test_score_candidate_row_uses_conservative_region_gate() -> None:
    class DummyFullEvaluator(InternalRubricCEPREvaluator):
        def _edit_specificity(self, pipe, proposal, original, edited, candidate_index, cache):  # noqa: ANN001
            return 0.9, {}

        def _taxonomy_score(self, pipe, proposal, original, edited, candidate_index, cache):  # noqa: ANN001
            return 0.9, {"cepr_taxonomy_supported": 1.0}

        def _preservation_and_validity(self, pipe, proposal, original, edited, candidate_index, cache):  # noqa: ANN001
            return 0.9, 0.9, {}

        def _rubric_score(self, pipe, proposal, original, edited, candidate_index, validity, cache):  # noqa: ANN001
            return {
                "rubric_source_grounded": 0.9,
                "rubric_required_after": 0.9,
                "rubric_forbidden_after_absent": 0.9,
                "rubric_edit_success": 0.9,
                "rubric_preservation": 0.9,
                "rubric_validity": validity,
                "rubric_reward": 0.9,
            }, {"rubric_forbidden_after_supported": 1.0}

        def _object_detector_contract_score(self, spec, original, edited, cache=None):  # noqa: ANN001
            return {"object_detector_contract": 1.0}, {
                "object_detector_supported": 0.0,
                "object_detector_gate_pass": 1.0,
            }

        def _detect_object_boxes(self, image, phrase, cache=None):  # noqa: ANN001
            return [{"score": 0.9, "box": (44, 44, 84, 84), "label": phrase}]

    original = _base_image()
    clean_edit = original.copy()
    draw = ImageDraw.Draw(clean_edit)
    draw.rectangle((44, 44, 84, 84), fill=(220, 220, 220))

    damaged_edit = clean_edit.copy()
    draw = ImageDraw.Draw(damaged_edit)
    draw.rectangle((0, 0, 48, 48), fill=(255, 0, 0))

    proposal = EditProposal(
        record_key="synthetic",
        round_index=0,
        proposal_index=0,
        definition=ProposalDefinition(
            operation_id="synthetic_removal",
            instruction="Remove the box.",
            family="object",
            difficulty=1,
            scope="local",
            metric="synthetic",
            direction="up",
            target=1.0,
            expected_changed_fraction=(0.0, 1.0),
        ),
        difficulty_level=1,
        instruction="Remove the box.",
        structured_edit={"edit_type": "object_removal", "source_object": "box", "target_region": "box"},
    )
    evaluator = DummyFullEvaluator(
        {
            "conservative_region_reward_enabled": True,
            "object_detector_enabled": True,
            "conservative_region_edit_types": ["object_removal"],
            "conservative_region_require_mask_edit_types": ["object_removal"],
            "conservative_region_max_outside_change": 0.05,
            "conservative_region_max_outside_changed_fraction": 0.20,
            "conservative_region_min_reward": 0.30,
            "conservative_region_min_outside_preservation": 0.50,
            "conservative_region_min_target_change_score": 0.35,
            "empty_cache_per_candidate": False,
        }
    )

    clean_row = evaluator._score_candidate_row(None, proposal, original, clean_edit, 0, "cpu")
    damaged_row = evaluator._score_candidate_row(None, proposal, original, damaged_edit, 0, "cpu")

    assert clean_row["feasible"] is True
    assert damaged_row["feasible"] is False
    assert damaged_row["signals"]["rubric_reject_reason"] == "conservative_region"
    assert damaged_row["signals"]["conservative_region_reject_reason"] == "outside_change"
