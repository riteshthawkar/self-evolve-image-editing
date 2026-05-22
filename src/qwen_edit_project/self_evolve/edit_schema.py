from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from qwen_edit_project.self_evolve.types import ProposalDefinition


EDIT_TYPES = {
    "attribute_change",
    "color_change",
    "material_change",
    "object_replacement",
    "object_removal",
    "object_addition",
    "spatial_move",
    "style_transfer",
    "background_change",
    "global_adjustment",
    "local_enhancement",
}

LOCAL_EDIT_TYPES = {
    "attribute_change",
    "color_change",
    "material_change",
    "object_replacement",
    "object_removal",
    "object_addition",
    "spatial_move",
    "local_enhancement",
}

EXPECTED_CHANGE_FRACTIONS = {
    "attribute_change": (0.03, 0.45),
    "color_change": (0.03, 0.45),
    "material_change": (0.04, 0.50),
    "object_replacement": (0.05, 0.70),
    "object_removal": (0.04, 0.55),
    "object_addition": (0.04, 0.60),
    "spatial_move": (0.05, 0.70),
    "style_transfer": (0.15, 0.90),
    "background_change": (0.12, 0.85),
    "global_adjustment": (0.25, 1.0),
    "local_enhancement": (0.03, 0.45),
}

DEFAULT_PRESERVE = [
    "background",
    "scene layout",
    "camera viewpoint",
    "lighting consistency",
    "unrelated objects",
]


def _clean_text(value: Any, max_len: int = 160) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return re.sub(r"\s+", " ", text)[:max_len]


def _clean_list(value: Any, max_items: int = 8) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [part.strip() for part in re.split(r"[,;\n]", value)]
    elif isinstance(value, list):
        raw_items = value
    else:
        raw_items = [value]
    output = []
    seen = set()
    for item in raw_items:
        cleaned = _clean_text(item, max_len=80)
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            output.append(cleaned)
        if len(output) >= max_items:
            break
    return output


def normalize_edit_type(value: Any, fallback: str = "local_enhancement") -> str:
    text = _clean_text(value, max_len=64)
    if text is None:
        return fallback
    normalized = text.lower().replace(" ", "_").replace("-", "_")
    alias_map = {
        "replace": "object_replacement",
        "replacement": "object_replacement",
        "remove": "object_removal",
        "delete": "object_removal",
        "add": "object_addition",
        "insert": "object_addition",
        "move": "spatial_move",
        "relocate": "spatial_move",
        "color": "color_change",
        "colour_change": "color_change",
        "material": "material_change",
        "style": "style_transfer",
        "background": "background_change",
        "global": "global_adjustment",
        "local": "local_enhancement",
    }
    normalized = alias_map.get(normalized, normalized)
    return normalized if normalized in EDIT_TYPES else fallback


def infer_edit_type_from_instruction(instruction: str, family: str | None = None) -> str:
    lowered = instruction.lower()
    if any(term in lowered for term in ("replace ", "change the object", "turn the", "make the person into")):
        return "object_replacement"
    if any(term in lowered for term in ("remove ", "delete ", "erase ")):
        return "object_removal"
    if any(term in lowered for term in ("add ", "insert ", "place a new")):
        return "object_addition"
    if any(term in lowered for term in ("move ", "relocate ", "to the left", "to the right", "higher", "lower")):
        return "spatial_move"
    if any(term in lowered for term in ("color", "colour", "red", "blue", "green", "yellow", "black", "white")):
        return "color_change"
    if any(term in lowered for term in ("metal", "wood", "glass", "plastic", "denim", "leather")):
        return "material_change"
    if family in {"style", "tone"}:
        return "style_transfer"
    if family == "background":
        return "background_change"
    if family in {"exposure", "contrast", "color"}:
        return "global_adjustment"
    return "local_enhancement"


def normalize_structured_edit(payload: dict[str, Any] | None, instruction: str, family: str | None = None) -> dict[str, Any]:
    data = dict(payload or {})
    edit_type = normalize_edit_type(
        data.get("edit_type") or data.get("type"),
        fallback=infer_edit_type_from_instruction(instruction, family=family),
    )
    preserve = _clean_list(data.get("preserve") or data.get("preservation_constraints"))
    if not preserve:
        preserve = DEFAULT_PRESERVE[:]
    normalized = {
        "edit_type": edit_type,
        "instruction": _clean_text(data.get("instruction") or instruction, max_len=512) or instruction,
        "source_object": _clean_text(data.get("source_object") or data.get("object")),
        "target_object": _clean_text(data.get("target_object") or data.get("replacement_object")),
        "source_attribute": _clean_text(data.get("source_attribute")),
        "target_attribute": _clean_text(data.get("target_attribute") or data.get("attribute")),
        "source_material": _clean_text(data.get("source_material")),
        "target_material": _clean_text(data.get("target_material")),
        "source_style": _clean_text(data.get("source_style")),
        "target_style": _clean_text(data.get("target_style") or data.get("style")),
        "source_location": _clean_text(data.get("source_location")),
        "target_location": _clean_text(data.get("target_location") or data.get("relation")),
        "target_region": _clean_text(data.get("target_region") or data.get("region")) or "main visible target",
        "preserve": preserve,
        "difficulty": data.get("difficulty"),
    }
    return {key: value for key, value in normalized.items() if value not in (None, [], "")}


def proposal_definition_from_structured_edit(
    structured_edit: dict[str, Any],
    *,
    proposal_index: int,
    difficulty_level: int,
) -> ProposalDefinition:
    edit_type = normalize_edit_type(structured_edit.get("edit_type"))
    instruction = str(structured_edit.get("instruction", "")).strip()
    digest = hashlib.sha1(
        json.dumps(structured_edit, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:8]
    scope = "local" if edit_type in LOCAL_EDIT_TYPES else "global"
    expected_range = EXPECTED_CHANGE_FRACTIONS.get(edit_type, (0.04, 0.65))
    return ProposalDefinition(
        operation_id=f"learned_{edit_type}_{proposal_index:02d}_{digest}",
        instruction=instruction,
        family=edit_type,
        difficulty=max(1, int(structured_edit.get("difficulty") or difficulty_level)),
        scope=scope,
        metric="internal_prompt_gain",
        direction="increase",
        target=0.0,
        expected_changed_fraction=expected_range,
        verifier="internal_cepr_plus",
    )


def extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def structured_edit_prompt(difficulty_level: int, proposals_per_image: int) -> str:
    return (
        "Generate image-grounded edit instructions for self-training an image editing model. "
        "Return only JSON in the form {\"proposals\": [ ... ]}. The edits must be useful but feasible: avoid trivial brightness-only "
        "changes and avoid impossible multi-object scene rewrites. Prefer real image-editing tasks "
        "such as object replacement, object removal, object addition, object color/material/style "
        "changes, background changes, and spatial moves when visually plausible. "
        f"Target difficulty level: {difficulty_level}. Number of proposals: {proposals_per_image}. "
        "Each proposal must contain: edit_type, instruction, target_region, preserve, and when "
        "applicable source_object, target_object, source_attribute, target_attribute, "
        "source_location, target_location. Use concise concrete nouns."
    )
