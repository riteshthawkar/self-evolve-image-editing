"""Correctness invariants for the internal CEPR reward.

These tests lock down the sign and monotonicity of the contrastive
edit-preservation reward so that a future sign flip or distractor regression
fails loudly instead of silently degrading self-evolution.

The Qwen feature extractors are replaced with deterministic fakes that map
registered images and prompts to fixed embedding vectors. This lets us assert
the reward math directly without loading the Qwen-Image-Edit checkpoint.
"""

from __future__ import annotations

import math

import pytest
import torch
from PIL import Image

from qwen_edit_project.self_evolve import backends
from qwen_edit_project.self_evolve.backends import (
    InternalContrastiveEditPreservationEvaluator,
    InternalRubricCEPREvaluator,
    _clamp,
    _mean_changed_fraction_score,
    _sigmoid,
)
from qwen_edit_project.self_evolve.types import EditProposal, ProposalDefinition


def _proposal(
    instruction: str,
    structured_edit: dict | None = None,
    edit_type: str = "local_enhancement",
) -> EditProposal:
    definition = ProposalDefinition(
        operation_id="test_op",
        instruction=instruction,
        family=edit_type,
        difficulty=2,
        scope="local",
        metric="internal_prompt_gain",
        direction="increase",
        target=0.0,
        expected_changed_fraction=(0.05, 0.70),
        verifier="internal_cepr_plus",
    )
    return EditProposal(
        record_key="rec",
        round_index=1,
        proposal_index=0,
        definition=definition,
        difficulty_level=2,
        instruction=instruction,
        structured_edit=structured_edit or {},
    )


def _install_fakes(
    monkeypatch,
    *,
    text_vecs: dict[str, tuple[float, ...]] | None = None,
    img_vecs: dict[int, tuple[float, ...]] | None = None,
    latents: dict[int, torch.Tensor] | None = None,
    default_text: tuple[float, ...] = (1.0, 0.0),
) -> None:
    text_vecs = text_vecs or {}
    img_vecs = img_vecs or {}
    latents = latents or {}

    def fake_text(pipe, prompt):
        vec = text_vecs.get(prompt, default_text)
        return {"pooled_embedding": torch.tensor([list(vec)], dtype=torch.float32)}

    def fake_understanding(pipe, prompt, images):
        image = images[0]
        vec = img_vecs[id(image)]
        return {"pooled_embedding": torch.tensor([list(vec)], dtype=torch.float32)}

    def fake_vae(pipe, image, size=512):
        return latents[id(image)]

    monkeypatch.setattr(backends, "extract_qwen_text_features", fake_text)
    monkeypatch.setattr(backends, "extract_qwen_edit_understanding_features", fake_understanding)
    monkeypatch.setattr(backends, "extract_qwen_vae_latents", fake_vae)


# --------------------------------------------------------------------------- #
# Pure math invariants (no model, no patching).
# --------------------------------------------------------------------------- #


def test_sigmoid_properties() -> None:
    assert _sigmoid(0.0) == pytest.approx(0.5)
    assert _sigmoid(50.0) > 0.99
    assert _sigmoid(-50.0) < 0.01
    assert _sigmoid(1.0) > _sigmoid(0.0) > _sigmoid(-1.0)


def test_clamp_bounds() -> None:
    assert _clamp(-1.0) == 0.0
    assert _clamp(2.0) == 1.0
    assert _clamp(0.3) == pytest.approx(0.3)


def test_mean_changed_fraction_score_band() -> None:
    assert _mean_changed_fraction_score(0.4, (0.3, 0.5)) == 1.0
    below = _mean_changed_fraction_score(0.1, (0.3, 0.5))
    above = _mean_changed_fraction_score(0.9, (0.3, 0.5))
    assert 0.0 <= below < 1.0
    assert 0.0 <= above < 1.0


def test_geometric_mean_failed_component_dominates() -> None:
    geometric_mean = InternalContrastiveEditPreservationEvaluator._geometric_mean
    assert geometric_mean([1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert geometric_mean([0.25, 0.25]) == pytest.approx(0.25, abs=1e-6)
    # A single failed (zero) component must drag the mean toward zero so that one
    # strong signal cannot compensate for a violated constraint.
    assert geometric_mean([0.0, 1.0]) < 0.05


# --------------------------------------------------------------------------- #
# Sign and monotonicity of edit specificity (the core anti-sign-bug lock).
# --------------------------------------------------------------------------- #


def test_edit_specificity_sign_and_monotonicity(monkeypatch) -> None:
    evaluator = InternalContrastiveEditPreservationEvaluator({})
    original = Image.new("RGB", (8, 8), (10, 10, 10))
    edited_on = Image.new("RGB", (8, 8), (20, 20, 20))
    edited_wrong = Image.new("RGB", (8, 8), (30, 30, 30))
    true_instruction = "Make the cat blue."

    _install_fakes(
        monkeypatch,
        text_vecs={true_instruction: (0.0, 1.0)},  # default distractor text = (1, 0)
        img_vecs={
            id(original): (1.0, 0.0),
            id(edited_on): (0.0, 1.0),  # moved toward the instruction direction
            id(edited_wrong): (1.0, -1.0),  # moved away from the instruction direction
        },
        default_text=(1.0, 0.0),
    )

    proposal = _proposal(true_instruction)  # no structured edit -> photometric fallback

    spec_on, sig_on = evaluator._edit_specificity(object(), proposal, original, edited_on, 0, {})
    spec_id, sig_id = evaluator._edit_specificity(object(), proposal, original, original, 1, {})
    spec_wrong, sig_wrong = evaluator._edit_specificity(object(), proposal, original, edited_wrong, 2, {})

    # Identical image must produce exactly zero prompt gain (sign-agnostic anchor).
    assert sig_id["cepr_true_prompt_gain"] == pytest.approx(0.0, abs=1e-6)
    # Sign lock: on-instruction edits have positive gain, wrong edits negative.
    assert sig_on["cepr_true_prompt_gain"] > 0.5
    assert sig_wrong["cepr_true_prompt_gain"] < 0.0
    # Specificity must be monotone in edit correctness.
    assert spec_on > spec_id > spec_wrong
    assert spec_on > 0.8
    assert spec_wrong < 0.2


def test_edit_specificity_uses_distractor_bank(monkeypatch) -> None:
    evaluator = InternalContrastiveEditPreservationEvaluator({})
    original = Image.new("RGB", (8, 8), (10, 10, 10))
    edited = Image.new("RGB", (8, 8), (20, 20, 20))

    _install_fakes(
        monkeypatch,
        img_vecs={id(original): (1.0, 0.0), id(edited): (0.0, 1.0)},
    )

    proposal = _proposal(
        "Replace the cat with a dog.",
        structured_edit={
            "edit_type": "object_replacement",
            "source_object": "cat",
            "target_object": "a dog",
        },
        edit_type="object_replacement",
    )

    _, signals = evaluator._edit_specificity(object(), proposal, original, edited, 0, {})
    # The contrastive term must be computed against a non-empty distractor bank so
    # an edit only scores well when it beats plausible alternative edits.
    assert signals["cepr_distractor_count"] >= 1.0
    assert "cepr_contrastive_margin" in signals
    assert "cepr_max_distractor_gain" in signals


# --------------------------------------------------------------------------- #
# Preservation invariants.
# --------------------------------------------------------------------------- #


def test_semantic_preservation_identical_image_is_full(monkeypatch) -> None:
    evaluator = InternalContrastiveEditPreservationEvaluator({})
    original = Image.new("RGB", (8, 8), (123, 123, 123))

    _install_fakes(monkeypatch, img_vecs={id(original): (0.3, 0.7, 0.1)})

    preservation, signals = evaluator._semantic_preservation(object(), original, original, 0, {})
    assert preservation == pytest.approx(1.0, abs=1e-5)
    assert signals["cepr_semantic_preservation_cosine"] == pytest.approx(1.0, abs=1e-5)


def test_semantic_preservation_decreases_with_dissimilarity(monkeypatch) -> None:
    evaluator = InternalContrastiveEditPreservationEvaluator({})
    original = Image.new("RGB", (8, 8), (0, 0, 0))
    near = Image.new("RGB", (8, 8), (1, 1, 1))
    far = Image.new("RGB", (8, 8), (2, 2, 2))

    _install_fakes(
        monkeypatch,
        img_vecs={
            id(original): (1.0, 0.0),
            id(near): (0.99, 0.14),  # nearly collinear -> high cosine
            id(far): (0.0, 1.0),  # orthogonal -> low cosine
        },
    )

    preservation_near, _ = evaluator._semantic_preservation(object(), original, near, 0, {})
    preservation_far, _ = evaluator._semantic_preservation(object(), original, far, 1, {})
    assert preservation_near > preservation_far


def test_latent_locality_identical_image_is_fully_preserved(monkeypatch) -> None:
    evaluator = InternalContrastiveEditPreservationEvaluator({})
    original = Image.new("RGB", (8, 8), (64, 64, 64))
    latent = torch.randn(1, 4, 8, 8)

    _install_fakes(monkeypatch, latents={id(original): latent})

    outside_preservation, validity, signals = evaluator._latent_locality(
        object(), _proposal("Make it brighter."), original, original, 0, {}
    )
    assert outside_preservation == pytest.approx(1.0, abs=1e-5)
    assert signals["cepr_latent_total_delta"] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(validity)


# --------------------------------------------------------------------------- #
# Distractor bank (contrastive counterfactuals).
# --------------------------------------------------------------------------- #


def test_distractor_definitions_exclude_self_and_are_bounded() -> None:
    evaluator = InternalContrastiveEditPreservationEvaluator({})
    proposal = _proposal(
        "Replace the person with a stuffed animal.",
        structured_edit={
            "edit_type": "object_replacement",
            "source_object": "person",
            "target_object": "a stuffed animal",
        },
        edit_type="object_replacement",
    )

    distractors = evaluator._distractor_definitions(proposal)
    assert distractors
    assert len(distractors) <= evaluator.counterfactual_distractors
    assert all(d.operation_id != proposal.definition.operation_id for d in distractors)


def test_describe_distractors_reports_operations() -> None:
    evaluator = InternalContrastiveEditPreservationEvaluator({})
    proposal = _proposal("Make the sky more dramatic.")
    described = evaluator.describe_distractors(proposal)
    assert described
    assert all("operation_id" in entry and "instruction" in entry for entry in described)


# --------------------------------------------------------------------------- #
# Rubric CEPR decision invariants (anti reward-hacking locks).
#
# These assert the decision algebra of the structured rubric reward using
# deterministic fake Qwen features. The central claim under test is that the
# rubric reward rejects the failure modes the opaque scalar CEPR accepted:
# no-op edits (old object still present) and plausible-but-wrong edits.
#
# Concept axes in a 4-D fake feature space:
#   dims 0,1 -> background / preserved content
#   dim 2    -> the source object ("person")
#   dim 3    -> the requested new object ("stuffed animal")
# --------------------------------------------------------------------------- #

_BG = (1.0, 1.0, 0.0, 0.0)
_PERSON = (0.0, 0.0, 1.0, 0.0)
_STUFFED = (0.0, 0.0, 0.0, 1.0)

_IMG_ORIGINAL = (1.0, 1.0, 1.0, 0.0)   # background + person, no stuffed animal
_IMG_GOOD = (1.0, 1.0, 0.0, 1.0)       # background + stuffed animal, person gone
_IMG_WRONG = (1.0, 0.0, 1.0, 0.0)      # background changed, person remains, no stuffed
_IMG_NO_PERSON = (1.0, 1.0, 0.0, 0.0)  # background only, source object absent


def _install_concept_fakes(monkeypatch, image_vectors) -> None:
    def fake_text(pipe, prompt):
        lowered = prompt.lower()
        if "stuffed" in lowered:
            vec = _STUFFED
        elif "person" in lowered:
            vec = _PERSON
        else:
            vec = _BG
        return {"pooled_embedding": torch.tensor([list(vec)], dtype=torch.float32)}

    def fake_understanding(pipe, prompt, images):
        vec = image_vectors[id(images[0])]
        return {"pooled_embedding": torch.tensor([list(vec)], dtype=torch.float32)}

    monkeypatch.setattr(backends, "extract_qwen_text_features", fake_text)
    monkeypatch.setattr(backends, "extract_qwen_edit_understanding_features", fake_understanding)


def _replacement_proposal() -> EditProposal:
    return _proposal(
        "Replace the person with a stuffed animal.",
        structured_edit={
            "edit_type": "object_replacement",
            "source_object": "person",
            "target_object": "a stuffed animal",
            "required_after": ["a stuffed animal is visible"],
            "forbidden_after": ["a person remains visible"],
            "preserve": ["background"],
        },
        edit_type="object_replacement",
    )


def test_rubric_reward_orders_good_above_noop_and_wrong(monkeypatch) -> None:
    evaluator = InternalRubricCEPREvaluator({})
    original = Image.new("RGB", (8, 8), (10, 10, 10))
    good = Image.new("RGB", (8, 8), (20, 20, 20))
    wrong = Image.new("RGB", (8, 8), (30, 30, 30))

    _install_concept_fakes(
        monkeypatch,
        {
            id(original): _IMG_ORIGINAL,
            id(good): _IMG_GOOD,
            id(wrong): _IMG_WRONG,
        },
    )
    proposal = _replacement_proposal()

    good_scores, _ = evaluator._rubric_score(object(), proposal, original, good, 0, 1.0, {})
    noop_scores, _ = evaluator._rubric_score(object(), proposal, original, original, 1, 1.0, {})
    wrong_scores, _ = evaluator._rubric_score(object(), proposal, original, wrong, 2, 1.0, {})

    # The no-op leaves the person in place; the wrong edit never adds the stuffed
    # animal. Both must score strictly below the correct replacement.
    assert good_scores["rubric_reward"] > noop_scores["rubric_reward"]
    assert good_scores["rubric_reward"] > wrong_scores["rubric_reward"]


def test_rubric_required_after_higher_for_correct_edit(monkeypatch) -> None:
    evaluator = InternalRubricCEPREvaluator({})
    original = Image.new("RGB", (8, 8), (10, 10, 10))
    good = Image.new("RGB", (8, 8), (20, 20, 20))

    _install_concept_fakes(
        monkeypatch,
        {id(original): _IMG_ORIGINAL, id(good): _IMG_GOOD},
    )
    proposal = _replacement_proposal()

    good_scores, _ = evaluator._rubric_score(object(), proposal, original, good, 0, 1.0, {})
    noop_scores, _ = evaluator._rubric_score(object(), proposal, original, original, 1, 1.0, {})
    assert good_scores["rubric_required_after"] > noop_scores["rubric_required_after"]


def test_rubric_penalizes_persistent_forbidden_object(monkeypatch) -> None:
    # A no-op leaves the "person" present, so old-state removal must score low;
    # the correct edit removes the person, so absence must score higher.
    evaluator = InternalRubricCEPREvaluator({})
    original = Image.new("RGB", (8, 8), (10, 10, 10))
    good = Image.new("RGB", (8, 8), (20, 20, 20))

    _install_concept_fakes(
        monkeypatch,
        {id(original): _IMG_ORIGINAL, id(good): _IMG_GOOD},
    )
    proposal = _replacement_proposal()

    good_scores, _ = evaluator._rubric_score(object(), proposal, original, good, 0, 1.0, {})
    noop_scores, _ = evaluator._rubric_score(object(), proposal, original, original, 1, 1.0, {})
    assert (
        good_scores["rubric_forbidden_after_absent"]
        > noop_scores["rubric_forbidden_after_absent"]
    )


def test_rubric_source_grounding_requires_source_object_present(monkeypatch) -> None:
    evaluator = InternalRubricCEPREvaluator({})
    grounded_original = Image.new("RGB", (8, 8), (10, 10, 10))
    ungrounded_original = Image.new("RGB", (8, 8), (40, 40, 40))
    good = Image.new("RGB", (8, 8), (20, 20, 20))

    _install_concept_fakes(
        monkeypatch,
        {
            id(grounded_original): _IMG_ORIGINAL,
            id(ungrounded_original): _IMG_NO_PERSON,
            id(good): _IMG_GOOD,
        },
    )
    proposal = _replacement_proposal()

    grounded, _ = evaluator._rubric_score(object(), proposal, grounded_original, good, 0, 1.0, {})
    ungrounded, _ = evaluator._rubric_score(object(), proposal, ungrounded_original, good, 1, 1.0, {})
    assert grounded["rubric_source_grounded"] > ungrounded["rubric_source_grounded"]


# --------------------------------------------------------------------------- #
# internal_vlm_judge skip_infeasible decision-safety lock.
#
# The expensive VLM judge runs with require_for_feasible=true, so it can only
# ever REMOVE feasibility, never grant it. Skipping the judge on candidates that
# a cheaper gate already rejected must therefore be decision-identical to running
# it on them. These tests lock that invariant so the ~2x speedup cannot silently
# start changing accept/reject outcomes.
# --------------------------------------------------------------------------- #


def _judge_config(skip_infeasible: bool) -> dict:
    return {
        "internal_vlm_judge": {
            "enabled": True,
            "mode": "per_candidate",
            "max_candidates": 20,
            "require_for_feasible": True,
            "fail_open": False,
            "min_score_for_feasible": 0.35,
            "use_unreliable_scores": True,
            "skip_infeasible": skip_infeasible,
        }
    }


def _passing_judge_score() -> dict:
    return {
        "score": 0.9,
        "semantic": 0.9,
        "preservation": 0.9,
        "artifact_free": 0.9,
        "confidence": 0.9,
        "overall": 0.9,
        "instruction_following": 0.9,
        "edit_success": 0.9,
        "target_correctness": 0.9,
        "reason": "ok",
    }


def _install_recording_judge(monkeypatch, evaluator, recorded: list[list[int]]) -> None:
    def fake_run(pipe, proposal, original, candidate_items):
        indices = [int(ci) for ci, _img in candidate_items]
        recorded.append(indices)
        scores = {ci: _passing_judge_score() for ci in indices}
        summary = {
            "best_candidate_index": indices[0] if indices else None,
            "missing_candidate_indices": [],
            "fallback_candidate_indices": [],
            "fallback_errors": {},
        }
        return scores, summary

    monkeypatch.setattr(evaluator, "_run_internal_vlm_judge", fake_run)
    monkeypatch.setattr(evaluator, "_get_internal_pipe", lambda editor=None: object())


def _judge_rows() -> list[dict]:
    # candidate 0 passed the cheaper gates; candidate 1 was already rejected.
    return [
        {
            "candidate_index": 0,
            "feasible": True,
            "reward": 0.6,
            "raw_reward": 0.6,
            "semantic_edit": 0.5,
            "signals": {},
            "component_scores": {},
        },
        {
            "candidate_index": 1,
            "feasible": False,
            "reward": 0.0,
            "raw_reward": 0.1,
            "semantic_edit": 0.2,
            "signals": {"rubric_reject_reason": "conservative_region"},
            "component_scores": {},
        },
    ]


def _judge_proposal() -> EditProposal:
    return _proposal(
        "Brighten the lamp in the corner.",
        structured_edit={"edit_type": "local_enhancement"},
        edit_type="local_enhancement",
    )


def test_skip_infeasible_excludes_rejected_candidate_from_judge(monkeypatch) -> None:
    evaluator = InternalRubricCEPREvaluator(_judge_config(skip_infeasible=True))
    recorded: list[list[int]] = []
    _install_recording_judge(monkeypatch, evaluator, recorded)

    original = Image.new("RGB", (8, 8), (10, 10, 10))
    candidates = [Image.new("RGB", (8, 8), (20, 20, 20)), Image.new("RGB", (8, 8), (30, 30, 30))]
    rows = _judge_rows()

    evaluator._apply_group_judge(_judge_proposal(), original, candidates, rows, editor=None)

    # The judge was invoked only for the feasible candidate.
    assert recorded == [[0]]
    # The already-rejected candidate is left exactly as the cheaper gate left it.
    infeasible = rows[1]
    assert infeasible["feasible"] is False
    assert infeasible["reward"] == 0.0
    assert infeasible["signals"]["rubric_reject_reason"] == "conservative_region"


def test_no_skip_infeasible_judges_every_candidate(monkeypatch) -> None:
    evaluator = InternalRubricCEPREvaluator(_judge_config(skip_infeasible=False))
    recorded: list[list[int]] = []
    _install_recording_judge(monkeypatch, evaluator, recorded)

    original = Image.new("RGB", (8, 8), (10, 10, 10))
    candidates = [Image.new("RGB", (8, 8), (20, 20, 20)), Image.new("RGB", (8, 8), (30, 30, 30))]
    rows = _judge_rows()

    evaluator._apply_group_judge(_judge_proposal(), original, candidates, rows, editor=None)

    # Without the flag the judge sees both candidates (feasible sorted first).
    assert recorded == [[0, 1]]


def test_skip_infeasible_preserves_feasible_decision(monkeypatch) -> None:
    # The feasible candidate must reach an identical accept/reject decision whether
    # or not the judge is skipped on the rejected sibling.
    def run(skip: bool) -> dict:
        evaluator = InternalRubricCEPREvaluator(_judge_config(skip_infeasible=skip))
        _install_recording_judge(monkeypatch, evaluator, [])
        original = Image.new("RGB", (8, 8), (10, 10, 10))
        candidates = [Image.new("RGB", (8, 8), (20, 20, 20)), Image.new("RGB", (8, 8), (30, 30, 30))]
        rows = _judge_rows()
        evaluator._apply_group_judge(_judge_proposal(), original, candidates, rows, editor=None)
        return rows[0]

    skipped = run(True)
    full = run(False)
    assert skipped["feasible"] == full["feasible"]
    assert skipped["reward"] == pytest.approx(full["reward"])
