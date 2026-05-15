from __future__ import annotations

from qwen_edit_project.self_evolve.types import ProposalDefinition


PROPOSAL_BANK: list[ProposalDefinition] = [
    ProposalDefinition(
        operation_id="brightness_up",
        instruction="Make the overall image brighter while preserving the scene structure.",
        family="exposure",
        difficulty=1,
        scope="global",
        metric="luminance",
        direction="increase",
        target=0.08,
        expected_changed_fraction=(0.35, 1.0),
    ),
    ProposalDefinition(
        operation_id="brightness_down",
        instruction="Make the overall image darker while preserving the scene structure.",
        family="exposure",
        difficulty=1,
        scope="global",
        metric="luminance",
        direction="decrease",
        target=0.08,
        expected_changed_fraction=(0.35, 1.0),
    ),
    ProposalDefinition(
        operation_id="saturation_up",
        instruction="Increase the color saturation while keeping the composition unchanged.",
        family="color",
        difficulty=1,
        scope="global",
        metric="saturation",
        direction="increase",
        target=0.08,
        expected_changed_fraction=(0.30, 0.95),
    ),
    ProposalDefinition(
        operation_id="saturation_down",
        instruction="Reduce the color saturation while keeping the composition unchanged.",
        family="color",
        difficulty=1,
        scope="global",
        metric="saturation",
        direction="decrease",
        target=0.08,
        expected_changed_fraction=(0.30, 0.95),
    ),
    ProposalDefinition(
        operation_id="contrast_up",
        instruction="Increase the overall contrast while preserving the original structure.",
        family="contrast",
        difficulty=2,
        scope="global",
        metric="contrast",
        direction="increase",
        target=0.06,
        expected_changed_fraction=(0.25, 0.95),
    ),
    ProposalDefinition(
        operation_id="contrast_down",
        instruction="Soften the image contrast while preserving the original structure.",
        family="contrast",
        difficulty=2,
        scope="global",
        metric="contrast",
        direction="decrease",
        target=0.05,
        expected_changed_fraction=(0.25, 0.95),
    ),
    ProposalDefinition(
        operation_id="warm_tone",
        instruction="Shift the image toward a warmer tone while keeping the content unchanged.",
        family="tone",
        difficulty=2,
        scope="global",
        metric="warmth",
        direction="increase",
        target=0.06,
        expected_changed_fraction=(0.25, 0.90),
    ),
    ProposalDefinition(
        operation_id="cool_tone",
        instruction="Shift the image toward a cooler tone while keeping the content unchanged.",
        family="tone",
        difficulty=2,
        scope="global",
        metric="warmth",
        direction="decrease",
        target=0.06,
        expected_changed_fraction=(0.25, 0.90),
    ),
    ProposalDefinition(
        operation_id="grayscale",
        instruction="Convert the image to black and white while preserving structure and composition.",
        family="style",
        difficulty=3,
        scope="global",
        metric="saturation_level",
        direction="at_most",
        target=0.04,
        expected_changed_fraction=(0.70, 1.0),
    ),
]


def available_proposals(max_difficulty: int, families: list[str] | None = None) -> list[ProposalDefinition]:
    family_filter = set(families or [])
    proposals = [proposal for proposal in PROPOSAL_BANK if proposal.difficulty <= max_difficulty]
    if family_filter:
        proposals = [proposal for proposal in proposals if proposal.family in family_filter]
    return proposals
