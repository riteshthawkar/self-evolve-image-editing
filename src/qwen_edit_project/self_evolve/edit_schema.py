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

_SPATIAL_RELATION_PATTERN = re.compile(
    r"\s+("
    r"on|in|at|under|beneath|below|above|beside|near|next to|left of|right of|"
    r"in front of|behind|between|attached to|held by|worn by|around"
    r")\s+(.+)$",
    flags=re.IGNORECASE,
)


def _clean_text(value: Any, max_len: int = 160) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in {"none", "null", "n/a", "na", "not applicable", "unknown"}:
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


def _looks_generic_forbidden(text: str) -> bool:
    lowered = text.lower()
    generic_markers = (
        "no other",
        "not other",
        "any other",
        "other objects",
        "other background",
        "unrelated objects",
        "extra objects",
        "additional objects",
    )
    return any(marker in lowered for marker in generic_markers)


def _strip_article(text: str | None) -> str | None:
    cleaned = _clean_text(text)
    if cleaned is None:
        return None
    return re.sub(r"^(?:only\s+)?(?:a|an|the|any|new)\s+", "", cleaned, flags=re.IGNORECASE).strip() or None


def _trim_removal_object_tail(text: str | None) -> str | None:
    cleaned = _strip_article(text)
    if cleaned is None:
        return None
    cleanup_patterns = (
        r"\s+(?:completely|fully|entirely)\s+(?:and\s+)?(?:fill|inpaint|replace|remove|erase)\b.*$",
        r"\s+(?:and|then)\s+(?:fill|inpaint|replace|preserve|keep)\b.*$",
        r"\s+(?:while|but|without)\s+.*$",
        r"\s+(?:completely|fully|entirely)$",
        r"\s+(?:from|in)\s+(?:the\s+)?(?:image|scene|photo|picture)$",
    )
    for pattern in cleanup_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or None


def _split_spatial_relation(text: str | None) -> tuple[str | None, str | None]:
    cleaned = _trim_removal_object_tail(text)
    if cleaned is None:
        return None, None
    match = _SPATIAL_RELATION_PATTERN.search(cleaned)
    if not match:
        return cleaned, None
    object_text = cleaned[: match.start()].strip()
    relation = match.group(1).strip()
    anchor = match.group(2).strip()
    if not object_text or not anchor:
        return cleaned, None
    # Avoid turning descriptors such as "black and white" into fake regions.
    if len(object_text.split()) > 8:
        return cleaned, None
    return object_text, f"{relation} {anchor}"


def _normalize_removal_object_and_region(
    source_object: str | None,
    target_region: str | None,
) -> tuple[str | None, str | None]:
    object_text, inferred_region = _split_spatial_relation(source_object)
    region = _clean_text(target_region) or inferred_region
    if region:
        region = _trim_removal_object_tail(region) or region
    object_text = _trim_removal_object_tail(object_text)
    if object_text:
        object_text = re.sub(
            r"\b(?:cleanly|naturally|visible|unchanged|removed|filled|surrounding)\b",
            " ",
            object_text,
            flags=re.IGNORECASE,
        )
        object_text = re.sub(r"\s+", " ", object_text).strip()
    bad_fragments = ("fill the area", "area naturally", "surrounding", "unchanged")
    if object_text and any(fragment in object_text.lower() for fragment in bad_fragments):
        object_text = None
    return object_text, region


def _infer_object_slots_from_instruction(instruction: str, edit_type: str) -> dict[str, str]:
    lowered = instruction.strip()
    slots: dict[str, str] = {}
    if edit_type == "object_removal":
        match = re.search(
            r"\b(?:remove|delete|erase|get rid of|take out)\s+(.+?)(?:\s+from\s+(.+?))?(?:[.;,]|$)",
            lowered,
            flags=re.IGNORECASE,
        )
        if match:
            source_object, target_region = _normalize_removal_object_and_region(
                match.group(1),
                _clean_text(match.group(2)) if match.group(2) else None,
            )
            if source_object:
                slots["source_object"] = source_object
                slots["target"] = source_object
            if target_region:
                slots["target_region"] = target_region
        return slots

    if edit_type == "object_replacement":
        patterns = [
            r"\breplace\s+(.+?)\s+with\s+(.+?)(?:[.;,]|$)",
            r"\bchange\s+(.+?)\s+(?:into|to)\s+(.+?)(?:[.;,]|$)",
            r"\bturn\s+(.+?)\s+into\s+(.+?)(?:[.;,]|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, lowered, flags=re.IGNORECASE)
            if not match:
                continue
            source_object = _strip_article(match.group(1))
            target_object = _strip_article(match.group(2))
            if source_object:
                slots["source_object"] = source_object
                slots["target"] = source_object
            if target_object:
                slots["target_object"] = target_object
                slots["replacement"] = target_object
            break
        return slots

    if edit_type == "object_addition":
        match = re.search(
            r"\b(?:add|insert|place|put)\s+(.+?)(?:\s+(?:to|onto|on|in|into|near|beside)\s+(.+?))?(?:[.;,]|$)",
            lowered,
            flags=re.IGNORECASE,
        )
        if match:
            target_object = _strip_article(match.group(1))
            target_region = _clean_text(match.group(2)) if match.group(2) else None
            if target_object:
                slots["target_object"] = target_object
                slots["replacement"] = target_object
            if target_region:
                slots["target_region"] = target_region
    return slots


def _region_mentions_source_object(region: str | None, source_object: str | None) -> bool:
    if not region or not source_object:
        return False
    region_terms = set(re.findall(r"[a-z0-9]+", region.lower()))
    source_terms = {
        token
        for token in re.findall(r"[a-z0-9]+", source_object.lower())
        if token not in {"a", "an", "the", "any", "new", "old", "original"}
    }
    return bool(source_terms and source_terms.issubset(region_terms))


def _region_has_spatial_anchor(region: str | None) -> bool:
    if not region:
        return False
    lowered = f" {region.lower()} "
    anchor_markers = (
        " on ",
        " above ",
        " below ",
        " under ",
        " beneath ",
        " beside ",
        " near ",
        " next to ",
        " left of ",
        " right of ",
        " in front of ",
        " behind ",
        " between ",
        " attached to ",
        " held by ",
        " worn by ",
        " around ",
        " at the ",
        " in the ",
    )
    return any(marker in lowered for marker in anchor_markers)


def _region_phrase(region: str | None) -> str:
    cleaned = _clean_text(region) or "the target region"
    lowered = cleaned.lower()
    if lowered == "main visible target":
        return "in the target region"
    if lowered.startswith(
        (
            "on ",
            "above ",
            "below ",
            "under ",
            "beneath ",
            "beside ",
            "near ",
            "next to ",
            "left of ",
            "right of ",
            "in front of ",
            "behind ",
            "between ",
            "attached to ",
            "held by ",
            "worn by ",
            "around ",
            "at ",
            "in ",
        )
    ):
        return cleaned
    return f"at {cleaned}"


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
    family = (family or "").lower()
    if family in {"exposure", "contrast", "color", "tone"}:
        return "global_adjustment"
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
    if family == "style":
        return "style_transfer"
    if family == "background":
        return "background_change"
    return "local_enhancement"


def normalize_structured_edit(payload: dict[str, Any] | None, instruction: str, family: str | None = None) -> dict[str, Any]:
    data = dict(payload or {})
    edit_type = normalize_edit_type(
        data.get("edit_type") or data.get("type"),
        fallback=infer_edit_type_from_instruction(instruction, family=family),
    )
    inferred_slots = _infer_object_slots_from_instruction(instruction, edit_type)
    target_alias = _clean_text(data.get("target") or inferred_slots.get("target"))
    replacement_alias = _clean_text(data.get("replacement") or inferred_slots.get("replacement"))
    source_object = _clean_text(
        data.get("source_object")
        or data.get("object")
        or target_alias
        or inferred_slots.get("source_object")
    )
    target_object = _clean_text(
        data.get("target_object")
        or data.get("replacement_object")
        or replacement_alias
        or inferred_slots.get("target_object")
    )
    source_attribute = _clean_text(data.get("source_attribute"))
    target_attribute = _clean_text(data.get("target_attribute") or data.get("attribute"))
    source_material = _clean_text(data.get("source_material"))
    target_material = _clean_text(data.get("target_material"))
    source_style = _clean_text(data.get("source_style"))
    target_style = _clean_text(data.get("target_style") or data.get("style"))
    source_location = _clean_text(data.get("source_location"))
    target_location = _clean_text(data.get("target_location") or data.get("relation"))
    raw_target_region = _clean_text(data.get("target_region") or data.get("region"))
    inferred_target_region = _clean_text(inferred_slots.get("target_region"))
    target_region = raw_target_region or inferred_target_region or "main visible target"
    if edit_type == "object_removal":
        source_object, normalized_region = _normalize_removal_object_and_region(
            source_object or target_alias or inferred_slots.get("source_object"),
            raw_target_region or inferred_target_region,
        )
        if source_object:
            target_alias = source_object
            target_object = None
        if normalized_region:
            target_region = normalized_region
    if edit_type in {"object_replacement", "object_removal"} and _region_mentions_source_object(
        target_region,
        source_object,
    ) and not _region_has_spatial_anchor(target_region):
        target_region = inferred_target_region or (
            "the original location" if edit_type == "object_removal" else "the same location"
        )
    preserve = _clean_list(data.get("preserve") or data.get("preservation_constraints"))
    if not preserve:
        preserve = DEFAULT_PRESERVE[:]
    required_after = _clean_list(data.get("required_after") or data.get("must_have_after"))
    forbidden_after = _clean_list(data.get("forbidden_after") or data.get("must_not_have_after"))

    if edit_type == "object_removal" and source_object:
        canonical_required = f"the area {_region_phrase(target_region)} is cleanly filled after removing {source_object}"
        if not any("fill" in item.lower() and source_object.lower() in item.lower() for item in required_after):
            required_after.insert(0, canonical_required)
        forbidden_after = [
            item
            for item in forbidden_after
            if source_object.lower() in item.lower()
            and not _looks_generic_forbidden(item)
            and "any object other than" not in item.lower()
        ]

    if not required_after:
        if edit_type == "object_replacement" and target_object:
            required_after.append(f"{target_object} is visible {_region_phrase(target_region)}")
        elif edit_type == "object_addition" and target_object:
            required_after.append(f"{target_object} has been added {_region_phrase(target_region)}")
        elif edit_type == "object_removal" and source_object:
            required_after.append(f"the area {_region_phrase(target_region)} is cleanly filled after removing {source_object}")
        elif edit_type == "spatial_move" and (source_object or target_object) and target_location:
            moved_object = source_object or target_object
            required_after.append(f"{moved_object} is {target_location}")
        elif edit_type in {"attribute_change", "color_change", "material_change", "style_transfer"}:
            target_descriptor = (
                target_attribute
                or target_material
                or target_style
                or target_object
            )
            if target_descriptor:
                required_after.append(f"{target_region} has {target_descriptor}")
        elif edit_type == "background_change" and (target_attribute or target_object):
            required_after.append(f"background changed to {_clean_text(' '.join([part for part in [target_attribute, target_object] if part]))}")
        elif edit_type == "global_adjustment":
            lowered_instruction = instruction.lower()
            if "saturation" in lowered_instruction:
                if any(term in lowered_instruction for term in ("reduce", "decrease", "lower", "less")):
                    required_after.append("image has reduced color saturation")
                else:
                    required_after.append("image has stronger color saturation")
            elif "contrast" in lowered_instruction:
                if any(term in lowered_instruction for term in ("soften", "reduce", "decrease", "lower", "less")):
                    required_after.append("image has softer contrast")
                else:
                    required_after.append("image has stronger contrast")
            elif any(term in lowered_instruction for term in ("brighter", "brighten")):
                required_after.append("image is brighter")
            elif any(term in lowered_instruction for term in ("darker", "darken")):
                required_after.append("image is darker")
            elif "warmer" in lowered_instruction:
                required_after.append("image has a warmer tone")
            elif "cooler" in lowered_instruction:
                required_after.append("image has a cooler tone")
        elif edit_type == "style_transfer" and not (target_attribute or target_style):
            required_after.append(_clean_text(instruction, max_len=160))

    if not forbidden_after:
        if edit_type in {"object_replacement", "object_removal"} and source_object:
            forbidden_after.append(f"{source_object} remains visible {_region_phrase(target_region)}")
        elif edit_type == "spatial_move" and source_object and source_location:
            forbidden_after.append(f"{source_object} remains {source_location}")
        elif edit_type in {"attribute_change", "color_change", "material_change", "style_transfer"}:
            source_descriptor = source_attribute or source_material or source_style
            if source_descriptor and (source_object or target_region):
                forbidden_after.append(f"{source_object or target_region} still has {source_descriptor}")
    elif edit_type in {"object_replacement", "object_removal"} and source_object:
        has_specific_source_forbidden = any(
            source_object.lower() in item.lower() and not _looks_generic_forbidden(item)
            for item in forbidden_after
        )
        if not has_specific_source_forbidden:
            forbidden_after.append(f"{source_object} remains visible {_region_phrase(target_region)}")
    if edit_type == "object_replacement" and source_object:
        has_separate_visibility_forbidden = any(
            source_object.lower() in item.lower()
            and any(marker in item.lower() for marker in ("separate", "duplicate", "extra", "additional"))
            for item in forbidden_after
        )
        if not has_separate_visibility_forbidden:
            forbidden_after.append(f"a separate {source_object} remains visible {_region_phrase(target_region)}")

    normalized = {
        "edit_type": edit_type,
        "instruction": _clean_text(data.get("instruction") or instruction, max_len=512) or instruction,
        "target": target_alias or source_object,
        "replacement": replacement_alias or target_object,
        "source_object": source_object,
        "target_object": target_object,
        "source_attribute": source_attribute,
        "target_attribute": target_attribute,
        "source_material": source_material,
        "target_material": target_material,
        "source_style": source_style,
        "target_style": target_style,
        "source_location": source_location,
        "target_location": target_location,
        "target_region": target_region,
        "required_after": required_after,
        "forbidden_after": forbidden_after,
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
        "changes and avoid impossible multi-object scene rewrites. Prefer localized, checkable edits "
        "with an explicit visible target, such as object replacement, object removal, object addition, "
        "object color/material/style changes, and spatial moves when visually plausible. For removal "
        "or replacement, prefer a small, well-separated accessory or secondary object with clear "
        "boundaries; avoid removing or replacing the dominant person, animal, vehicle, or whole "
        "background. Phrase removal as complete removal plus natural fill from the surrounding "
        "region. Phrase replacement as the old object being replaced in the same location and scale "
        "by a concrete new object. Avoid broad "
        "whole-background replacement of sky, snow, ground, walls, or scenery unless the target region "
        "is cleanly separable and the unchanged subject can be preserved. "
        f"Target difficulty level: {difficulty_level}. Number of proposals: {proposals_per_image}. "
        "Each proposal must contain: edit_type, instruction, target_region, preserve, required_after, "
        "forbidden_after, and when "
        "applicable source_object, target_object, source_attribute, target_attribute, "
        "source_location, target_location. For replacement edits, target/source_object is the old object "
        "and replacement/target_object is the new object. For removal and replacement, target_region "
        "must be a spatial phrase anchored to stable visible context, such as 'above the table', "
        "'left of the person', or 'on the shelf'; avoid generic regions like 'the original location' "
        "when a spatial anchor is visible. Use concise concrete nouns."
    )
