#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


EDIT_TYPES = [
    "object_removal",
    "object_replacement",
    "object_addition",
    "attribute_change",
    "color_change",
    "material_change",
    "spatial_move",
    "background_change",
    "style_transfer",
    "local_enhancement",
]

DEFAULT_TARGET_FRACTIONS = {
    "object_removal": 0.16,
    "object_replacement": 0.14,
    "object_addition": 0.10,
    "attribute_change": 0.10,
    "color_change": 0.10,
    "material_change": 0.10,
    "spatial_move": 0.08,
    "background_change": 0.10,
    "style_transfer": 0.06,
    "local_enhancement": 0.06,
}

EDIT_TYPE_TO_FAMILY = {
    "object_removal": "object",
    "object_replacement": "object",
    "object_addition": "object",
    "attribute_change": "object",
    "color_change": "color",
    "material_change": "object",
    "spatial_move": "object",
    "background_change": "background",
    "style_transfer": "style",
    "local_enhancement": "local",
}


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def finite_float(value: Any, default: float = math.nan) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return sum(clean) / len(clean) if clean else 0.0


def gmean(values: list[float]) -> float:
    clean = [max(clamp(value), 1.0e-6) for value in values if math.isfinite(value)]
    if not clean:
        return 0.0
    return math.exp(sum(math.log(value) for value in clean) / len(clean))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def expand_paths(values: list[str] | None) -> list[Path]:
    paths: list[Path] = []
    for value in values or []:
        matches = [Path(item) for item in glob.glob(value)]
        paths.extend(matches or [Path(value)])
    return sorted(dict.fromkeys(paths))


def nested_value(container: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = container
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def first_value(record: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    containers = [
        record,
        record.get("metadata") or {},
        record.get("vlm") or {},
        record.get("stats_scores") or {},
        record.get("stats") or {},
    ]
    for key in keys:
        if "." in key:
            value = nested_value(record, key, None)
            if value is not None:
                return value
        for container in containers:
            if isinstance(container, dict) and key in container:
                value = container.get(key)
                if value is not None:
                    return value
    return default


def score_value(record: dict[str, Any], keys: list[str], default: float = math.nan) -> float:
    return finite_float(first_value(record, keys, default), default)


def source_key(record: dict[str, Any]) -> str:
    key = first_value(record, ["key", "record_key"], None)
    if key:
        return str(key).split("__reward_pool__", 1)[0]
    image = str(first_value(record, ["image", "image_path"], "unknown_image"))
    return Path(image).stem


def image_path(record: dict[str, Any]) -> str:
    return str(first_value(record, ["image", "image_path"], ""))


def caption(record: dict[str, Any]) -> str:
    return str(first_value(record, ["caption", "source_caption"], ""))


def families(record: dict[str, Any]) -> set[str]:
    raw = first_value(record, ["edit_families"], None)
    if raw is None:
        raw = first_value(record, ["vlm.edit_families"], None)
    if isinstance(raw, str):
        values = [item.strip() for item in raw.replace(";", ",").split(",")]
    elif isinstance(raw, list):
        values = [str(item).strip() for item in raw]
    else:
        values = []
    primary = str(first_value(record, ["primary_family"], "") or "").strip()
    if primary:
        values.append(primary)
    return {value for value in values if value and value != "unknown"}


def family_bonus(profile: dict[str, float], fams: set[str], family_name: str) -> float:
    if family_name in fams:
        return 1.0
    if family_name == "style" and "background" in fams:
        return 0.88
    if family_name == "local" and ("object" in fams or "color" in fams):
        return 0.88
    if family_name == "object" and profile["object_clarity"] >= 0.65:
        return 0.92
    if family_name == "background" and profile["background_hint"] >= 0.65:
        return 0.92
    return 0.72


def source_profile(record: dict[str, Any]) -> dict[str, float]:
    base_score = score_value(record, ["source_selection_score", "score"], 0.5)
    technical = score_value(record, ["technical_quality_score", "stats_scores.technical_quality_score"], base_score)
    quality = score_value(record, ["quality_score", "vlm.quality_score"], base_score)
    natural = score_value(record, ["natural_image_score", "vlm.natural_image_score"], base_score)
    editable = score_value(record, ["editable_content_score", "vlm.editable_content_score"], base_score)
    object_clarity = score_value(record, ["object_region_clarity", "vlm.object_region_clarity"], 0.45)
    preservation = score_value(record, ["preservation_potential", "vlm.preservation_potential"], base_score)
    clutter = score_value(record, ["clutter_penalty", "vlm.clutter_penalty"], 0.45)
    text = score_value(record, ["text_watermark_penalty", "vlm.text_watermark_penalty"], 0.20)
    saturation = score_value(record, ["saturation_score", "stats_scores.saturation_score"], 0.55)
    structure = score_value(record, ["structure_score", "stats_scores.structure_score"], 0.55)
    contrast = score_value(record, ["contrast_score", "stats_scores.contrast_score"], 0.55)

    fams = families(record)
    background_hint = 0.70 if "background" in fams else 0.48
    style_hint = 0.70 if "style" in fams else (0.58 if "background" in fams else 0.44)
    local_hint = 0.70 if "local" in fams else (0.62 if "object" in fams else 0.50)

    removable = score_value(record, ["removable_object_score", "vlm.removable_object_score"], math.nan)
    if not math.isfinite(removable):
        removable = (0.78 * object_clarity + 0.22 * editable) if "object" in fams else 0.55 * object_clarity
    separability = score_value(record, ["small_object_separability", "vlm.small_object_separability"], math.nan)
    if not math.isfinite(separability):
        separability = 0.60 * object_clarity + 0.25 * (1.0 - clutter) + 0.15 * structure
    fillability = score_value(record, ["removal_background_fill_score", "vlm.removal_background_fill_score"], math.nan)
    if not math.isfinite(fillability):
        fillability = 0.45 * background_hint + 0.35 * (1.0 - clutter) + 0.20 * natural

    source_quality = mean(
        [
            0.16 * clamp(technical),
            0.15 * clamp(quality),
            0.12 * clamp(natural),
            0.16 * clamp(editable),
            0.15 * clamp(preservation),
            0.10 * clamp(object_clarity),
            0.08 * clamp(1.0 - clutter),
            0.08 * clamp(1.0 - text),
        ]
    ) * 8.0

    return {
        "source_quality": clamp(source_quality),
        "technical_quality": clamp(technical),
        "vlm_quality": clamp(quality),
        "natural": clamp(natural),
        "editable": clamp(editable),
        "object_clarity": clamp(object_clarity),
        "removable": clamp(removable),
        "separability": clamp(separability),
        "fillability": clamp(fillability),
        "preservation": clamp(preservation),
        "clutter_inverse": clamp(1.0 - clutter),
        "text_inverse": clamp(1.0 - text),
        "saturation": clamp(saturation),
        "structure": clamp(structure),
        "contrast": clamp(contrast),
        "background_hint": clamp(background_hint),
        "style_hint": clamp(style_hint),
        "local_hint": clamp(local_hint),
    }


def edit_opportunity_scores(record: dict[str, Any], profile: dict[str, float]) -> dict[str, float]:
    fams = families(record)
    q = profile["source_quality"]
    editable = profile["editable"]
    preservation = profile["preservation"]
    no_clutter = profile["clutter_inverse"]
    no_text = profile["text_inverse"]
    obj = profile["object_clarity"]
    sep = profile["separability"]
    rem = profile["removable"]
    fill = profile["fillability"]
    natural = profile["natural"]
    structure = profile["structure"]
    saturation = profile["saturation"]
    contrast = profile["contrast"]
    background = profile["background_hint"]
    style = profile["style_hint"]
    local = profile["local_hint"]

    raw = {
        "object_removal": gmean([obj, rem, sep, fill, preservation, no_clutter, no_text, q])
        * family_bonus(profile, fams, "object"),
        "object_replacement": gmean([obj, sep, editable, preservation, no_clutter, no_text, q])
        * family_bonus(profile, fams, "object"),
        "object_addition": gmean([editable, fill, background, preservation, no_clutter, natural, q])
        * max(family_bonus(profile, fams, "object"), family_bonus(profile, fams, "background")),
        "attribute_change": gmean([obj, editable, preservation, structure, no_clutter, q])
        * family_bonus(profile, fams, "object"),
        "color_change": gmean([editable, preservation, saturation, contrast, no_text, q])
        * family_bonus(profile, fams, "color"),
        "material_change": gmean([obj, editable, preservation, structure, contrast, no_clutter, q])
        * family_bonus(profile, fams, "object"),
        "spatial_move": gmean([obj, sep, editable, preservation, no_clutter, q])
        * family_bonus(profile, fams, "object"),
        "background_change": gmean([background, editable, preservation, natural, no_clutter, q])
        * family_bonus(profile, fams, "background"),
        "style_transfer": gmean([style, editable, natural, q, max(0.45, preservation)])
        * family_bonus(profile, fams, "style"),
        "local_enhancement": gmean([local, editable, preservation, contrast, structure, no_text, q])
        * family_bonus(profile, fams, "local"),
    }
    return {key: clamp(value) for key, value in raw.items()}


def evaluator(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("evaluator") or row.get("solver") or {}
    return value if isinstance(value, dict) else {}


def component(row: dict[str, Any], *names: str, default: float = 0.0) -> float:
    ev = evaluator(row)
    containers = [ev.get("component_scores", {}), ev.get("signals", {}), ev]
    for name in names:
        for container in containers:
            if isinstance(container, dict) and name in container:
                value = finite_float(container.get(name), math.nan)
                if math.isfinite(value):
                    return value
    return default


def proposal_edit_type(row: dict[str, Any]) -> str:
    proposal = row.get("proposal") or {}
    structured = proposal.get("structured_edit") if isinstance(proposal, dict) else {}
    if isinstance(structured, dict) and structured.get("edit_type"):
        return str(structured["edit_type"])
    if isinstance(proposal, dict) and proposal.get("family"):
        family = str(proposal["family"])
        return family if family in EDIT_TYPES else f"{family}_change"
    return "unknown"


def proposal_record_key(row: dict[str, Any]) -> str:
    key = str(row.get("record_key") or row.get("key") or "")
    return key.split("__reward_pool__", 1)[0]


def accepted(row: dict[str, Any]) -> bool:
    if str(row.get("status")) == "accepted":
        return True
    if (evaluator(row).get("accepted") is True) or component(row, "accepted_by_ranker", default=0.0) >= 0.5:
        return True
    return False


def judge_supported(row: dict[str, Any]) -> bool:
    return component(row, "internal_vlm_judge_supported", default=0.0) >= 0.5


def noop_like(row: dict[str, Any]) -> bool:
    changed = component(row, "cepr_latent_changed_fraction", default=0.0)
    true_gain = component(row, "cepr_true_prompt_gain", default=0.0)
    required_gain = component(row, "rubric_required_after_gain", default=0.0)
    judge_semantic = component(row, "internal_vlm_judge_semantic", default=1.0)
    judge_score = component(row, "internal_vlm_judge_score", default=1.0)
    if judge_supported(row) and judge_score <= 0.05:
        return True
    return changed < 0.015 and true_gain <= 0.0 and required_gain <= 0.0 and judge_semantic < 0.50


def drift_like(row: dict[str, Any]) -> bool:
    changed = component(row, "cepr_latent_changed_fraction", default=0.0)
    outside = component(row, "cepr_latent_outside_preservation", default=1.0)
    preservation = min(
        component(row, "cepr_preservation", default=1.0),
        component(row, "rubric_preservation", default=1.0),
    )
    return changed > 0.75 or outside < 0.55 or preservation < 0.58


def candidate_quality(row: dict[str, Any]) -> float:
    semantic = max(
        component(row, "cepr_semantic_edit", default=0.0),
        component(row, "cepr_edit_specificity", default=0.0),
        component(row, "rubric_edit_success", default=0.0),
        component(row, "rubric_required_after", default=0.0),
    )
    preservation = min(
        component(row, "cepr_preservation", default=0.0),
        component(row, "rubric_preservation", default=0.0),
    )
    validity = min(
        component(row, "cepr_validity", default=0.0),
        component(row, "rubric_validity", default=0.0),
    )
    reward = max(
        component(row, "conservative_region_reward", default=0.0),
        component(row, "rubric_cepr_raw_reward", default=0.0),
        component(row, "cepr_raw_reward", default=0.0),
    )
    values = [semantic, preservation, validity, reward]
    if judge_supported(row):
        values.extend(
            [
                component(row, "internal_vlm_judge_score", default=0.0),
                component(row, "internal_vlm_judge_preservation", default=0.0),
                component(row, "internal_vlm_judge_artifact_free", default=0.0),
            ]
        )
    return gmean(values)


@dataclass
class FeedbackAggregate:
    candidates: int = 0
    accepted: int = 0
    quality_sum: float = 0.0
    noop_count: int = 0
    drift_count: int = 0
    artifact_count: int = 0

    def add(self, row: dict[str, Any]) -> None:
        self.candidates += 1
        if accepted(row):
            self.accepted += 1
        quality = candidate_quality(row)
        self.quality_sum += quality
        if noop_like(row):
            self.noop_count += 1
        if drift_like(row):
            self.drift_count += 1
        if judge_supported(row) and component(row, "internal_vlm_judge_artifact_free", default=1.0) < 0.45:
            self.artifact_count += 1

    @property
    def accepted_rate(self) -> float:
        return self.accepted / self.candidates if self.candidates else 0.0

    @property
    def mean_quality(self) -> float:
        return self.quality_sum / self.candidates if self.candidates else 0.0

    def multiplier(self) -> float:
        if self.candidates == 0:
            return 1.0
        rate = self.accepted_rate
        if 0.10 <= rate <= 0.70:
            productive = 1.0
        elif rate > 0.70:
            productive = 0.82
        else:
            productive = 0.62
        bad_rate = (self.noop_count + self.drift_count + self.artifact_count) / max(1, self.candidates)
        return clamp(0.62 + 0.55 * self.mean_quality + 0.23 * productive - 0.35 * bad_rate, 0.35, 1.35)

    def to_json(self) -> dict[str, Any]:
        return {
            "candidates": self.candidates,
            "accepted": self.accepted,
            "accepted_rate": self.accepted_rate,
            "mean_candidate_quality": self.mean_quality,
            "noop_count": self.noop_count,
            "drift_count": self.drift_count,
            "artifact_count": self.artifact_count,
            "utility_multiplier": self.multiplier(),
        }


def load_feedback(paths: list[Path]) -> dict[tuple[str, str], FeedbackAggregate]:
    feedback: dict[tuple[str, str], FeedbackAggregate] = defaultdict(FeedbackAggregate)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Feedback path not found: {path}")
        for row in read_jsonl(path):
            record_key = proposal_record_key(row)
            edit_type = proposal_edit_type(row)
            if not record_key or edit_type not in EDIT_TYPES:
                continue
            role = str(row.get("candidate_role") or "policy")
            if role != "policy" and not role.startswith("policy:"):
                continue
            feedback[(record_key, edit_type)].add(row)
    return dict(feedback)


def parse_fraction_map(raw: str | None) -> dict[str, float]:
    if not raw:
        return dict(DEFAULT_TARGET_FRACTIONS)
    output: dict[str, float] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split("=", 1)
        output[key.strip()] = float(value)
    missing = [edit_type for edit_type in EDIT_TYPES if edit_type not in output]
    for edit_type in missing:
        output[edit_type] = 0.0
    return output


def largest_remainder(labels: list[str], weights: dict[str, float], total: int) -> dict[str, int]:
    if total <= 0:
        return {label: 0 for label in labels}
    positive = {label: max(0.0, weights.get(label, 0.0)) for label in labels}
    total_weight = sum(positive.values())
    if total_weight <= 0:
        return {label: 0 for label in labels}
    raw = {label: total * positive[label] / total_weight for label in labels}
    quotas = {label: int(raw[label]) for label in labels}
    remaining = total - sum(quotas.values())
    order = sorted(labels, key=lambda label: raw[label] - quotas[label], reverse=True)
    for label in order[:remaining]:
        quotas[label] += 1
    return quotas


@dataclass(order=True)
class PoolCandidate:
    sort_score: float
    source_key: str = field(compare=False)
    edit_type: str = field(compare=False)
    record: dict[str, Any] = field(compare=False)
    profile: dict[str, float] = field(compare=False)
    opportunities: dict[str, float] = field(compare=False)
    feedback: FeedbackAggregate | None = field(compare=False)

    @property
    def utility(self) -> float:
        return self.sort_score


def merge_score_and_manifest(score_rows: list[dict[str, Any]], manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_by_key: dict[str, dict[str, Any]] = {}
    manifest_by_image: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        manifest_by_key[source_key(row)] = row
        if image_path(row):
            manifest_by_image[image_path(row)] = row

    merged: list[dict[str, Any]] = []
    for row in score_rows:
        key = source_key(row)
        base = dict(manifest_by_key.get(key) or manifest_by_image.get(image_path(row)) or {})
        metadata = dict(base.get("metadata") or {})
        score_metadata = {
            "source_selection_score": score_value(row, ["source_selection_score", "score"], 0.0),
            "source_primary_family": first_value(row, ["primary_family"], None),
            "source_edit_families": sorted(families(row)),
            "source_score_record_present": True,
        }
        metadata.update({key: value for key, value in score_metadata.items() if value is not None})
        merged_row = dict(row)
        if base:
            merged_row.setdefault("image", image_path(base))
            merged_row.setdefault("caption", caption(base))
            merged_row["metadata"] = metadata
        else:
            merged_row["metadata"] = metadata
        merged.append(merged_row)
    return merged


def deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[str, tuple[float, dict[str, Any]]] = {}
    for record in records:
        key = source_key(record)
        profile = source_profile(record)
        score = profile["source_quality"]
        previous = best.get(key)
        if previous is None or score > previous[0]:
            best[key] = (score, record)
    return [value[1] for value in best.values()]


def average_hash(record: dict[str, Any]) -> str | None:
    raw = first_value(record, ["average_hash", "stats.average_hash"], None)
    return str(raw) if raw else None


def hash_distance(left: str | None, right: str | None) -> int | None:
    if not left or not right or len(left) != len(right):
        return None
    try:
        left_int = int(left, 16)
        right_int = int(right, 16)
    except ValueError:
        return None
    return (left_int ^ right_int).bit_count()


def build_candidates(
    records: list[dict[str, Any]],
    feedback: dict[tuple[str, str], FeedbackAggregate],
    min_source_quality: float,
    min_utility: float,
) -> tuple[list[PoolCandidate], list[dict[str, Any]]]:
    candidates: list[PoolCandidate] = []
    rejected: list[dict[str, Any]] = []
    for record in records:
        key = source_key(record)
        profile = source_profile(record)
        if not image_path(record):
            rejected.append({"source_key": key, "reason": "missing_image_path"})
            continue
        if profile["source_quality"] < min_source_quality:
            rejected.append(
                {
                    "source_key": key,
                    "image": image_path(record),
                    "reason": "low_source_quality",
                    "source_quality": profile["source_quality"],
                }
            )
            continue
        opportunities = edit_opportunity_scores(record, profile)
        for edit_type in EDIT_TYPES:
            fb = feedback.get((key, edit_type))
            fb_multiplier = fb.multiplier() if fb else 1.0
            opportunity = opportunities[edit_type]
            utility = clamp((profile["source_quality"] ** 0.65) * (opportunity ** 1.15) * fb_multiplier, 0.0, 1.5)
            if utility < min_utility:
                rejected.append(
                    {
                        "source_key": key,
                        "image": image_path(record),
                        "edit_type": edit_type,
                        "reason": "low_data_utility",
                        "source_quality": profile["source_quality"],
                        "edit_opportunity": opportunity,
                        "data_utility": utility,
                        "feedback": fb.to_json() if fb else None,
                    }
                )
                continue
            candidates.append(
                PoolCandidate(
                    sort_score=utility,
                    source_key=key,
                    edit_type=edit_type,
                    record=record,
                    profile=profile,
                    opportunities=opportunities,
                    feedback=fb,
                )
            )
    return candidates, rejected


def candidate_manifest_record(candidate: PoolCandidate, index: int, seed: int) -> dict[str, Any]:
    source = candidate.record
    etype = candidate.edit_type
    fam = EDIT_TYPE_TO_FAMILY.get(etype, "unknown")
    key = f"{candidate.source_key}__reward_pool__{etype}__{index:06d}"
    metadata = {
        "reward_aware_training_pool": True,
        "original_key": candidate.source_key,
        "source_selection_score": score_value(source, ["source_selection_score", "score"], 0.0),
        "source_quality_score": candidate.profile["source_quality"],
        "data_utility_score": candidate.utility,
        "edit_opportunity_score": candidate.opportunities[etype],
        "edit_opportunity_scores": candidate.opportunities,
        "scheduled_edit_type": etype,
        "target_edit_types": [etype],
        "primary_family": fam,
        "source_edit_families": sorted(families(source)),
        "data_selection_seed": seed,
        "data_selection_components": candidate.profile,
        "feedback": candidate.feedback.to_json() if candidate.feedback else None,
    }
    return {
        "key": key,
        "image": image_path(source),
        "caption": caption(source),
        "metadata": metadata,
    }


def select_pool(
    candidates: list[PoolCandidate],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    target_fractions = parse_fraction_map(args.target_fractions)
    quotas = largest_remainder(EDIT_TYPES, target_fractions, args.max_records)
    by_type: dict[str, list[PoolCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_type[candidate.edit_type].append(candidate)
    for items in by_type.values():
        rng.shuffle(items)
        items.sort(key=lambda item: item.utility, reverse=True)

    selected_candidates: list[PoolCandidate] = []
    selected_source_counts: Counter[str] = Counter()
    selected_hashes: list[tuple[str, str | None]] = []
    selected_pairs: set[tuple[str, str]] = set()
    selection_rejects: list[dict[str, Any]] = []

    def can_select(candidate: PoolCandidate, enforce_hash: bool = True) -> tuple[bool, str | None]:
        pair = (candidate.source_key, candidate.edit_type)
        if pair in selected_pairs:
            return False, "duplicate_source_edit_type"
        if selected_source_counts[candidate.source_key] >= args.max_per_source:
            return False, "max_per_source"
        candidate_hash = average_hash(candidate.record)
        if enforce_hash and candidate.source_key not in selected_source_counts:
            for _selected_key, selected_hash in selected_hashes:
                distance = hash_distance(candidate_hash, selected_hash)
                if distance is not None and distance < args.min_hash_distance:
                    return False, "near_duplicate_hash"
        return True, None

    def add_candidate(candidate: PoolCandidate) -> None:
        selected_candidates.append(candidate)
        selected_source_counts[candidate.source_key] += 1
        selected_pairs.add((candidate.source_key, candidate.edit_type))
        if selected_source_counts[candidate.source_key] == 1:
            selected_hashes.append((candidate.source_key, average_hash(candidate.record)))

    for edit_type in EDIT_TYPES:
        quota = quotas.get(edit_type, 0)
        for candidate in by_type.get(edit_type, []):
            if sum(1 for item in selected_candidates if item.edit_type == edit_type) >= quota:
                break
            ok, reason = can_select(candidate)
            if ok:
                add_candidate(candidate)
            else:
                selection_rejects.append(
                    {
                        "source_key": candidate.source_key,
                        "edit_type": candidate.edit_type,
                        "reason": reason,
                        "data_utility": candidate.utility,
                    }
                )

    if len(selected_candidates) < args.max_records:
        fallback = list(candidates)
        rng.shuffle(fallback)
        fallback.sort(key=lambda item: item.utility, reverse=True)
        for candidate in fallback:
            if len(selected_candidates) >= args.max_records:
                break
            ok, reason = can_select(candidate, enforce_hash=not args.relax_hash_on_backfill)
            if ok:
                add_candidate(candidate)
            else:
                selection_rejects.append(
                    {
                        "source_key": candidate.source_key,
                        "edit_type": candidate.edit_type,
                        "reason": reason,
                        "data_utility": candidate.utility,
                    }
                )

    rng.shuffle(selected_candidates)
    manifest = [candidate_manifest_record(candidate, index, args.seed) for index, candidate in enumerate(selected_candidates)]
    profile_rows = [
        {
            "source_key": candidate.source_key,
            "image": image_path(candidate.record),
            "caption": caption(candidate.record),
            "edit_type": candidate.edit_type,
            "source_quality_score": candidate.profile["source_quality"],
            "edit_opportunity_score": candidate.opportunities[candidate.edit_type],
            "data_utility_score": candidate.utility,
            "source_edit_families": sorted(families(candidate.record)),
            "profile": candidate.profile,
            "edit_opportunity_scores": candidate.opportunities,
            "feedback": candidate.feedback.to_json() if candidate.feedback else None,
        }
        for candidate in sorted(candidates, key=lambda item: item.utility, reverse=True)
    ]
    summary = {
        "selected_records": len(manifest),
        "candidate_pairs": len(candidates),
        "target_quotas": quotas,
        "selected_edit_type_counts": dict(
            sorted(Counter((record["metadata"] or {}).get("scheduled_edit_type", "unknown") for record in manifest).items())
        ),
        "selected_family_counts": dict(
            sorted(Counter((record["metadata"] or {}).get("primary_family", "unknown") for record in manifest).items())
        ),
        "selected_source_count_histogram": dict(sorted(Counter(selected_source_counts.values()).items())),
        "selection_reject_counts": dict(sorted(Counter(row["reason"] for row in selection_rejects).items())),
        "mean_selected_data_utility": mean([record["metadata"]["data_utility_score"] for record in manifest]),
        "mean_selected_source_quality": mean([record["metadata"]["source_quality_score"] for record in manifest]),
        "seed": args.seed,
    }
    return manifest, profile_rows, {"selection_rejects": selection_rejects, "summary": summary}


def build_pool(args: argparse.Namespace) -> dict[str, Any]:
    score_paths = expand_paths(args.score_jsonl)
    manifest_paths = expand_paths(args.manifest_jsonl)
    feedback_paths = expand_paths(args.feedback_proposals)
    if not score_paths:
        raise ValueError("At least one --score-jsonl path is required.")
    for path in score_paths + manifest_paths + feedback_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    score_rows = [row for path in score_paths for row in read_jsonl(path)]
    manifest_rows = [row for path in manifest_paths for row in read_jsonl(path)]
    merged_records = merge_score_and_manifest(score_rows, manifest_rows)
    records = deduplicate_records(merged_records)
    feedback = load_feedback(feedback_paths)
    candidates, low_quality_rejects = build_candidates(
        records,
        feedback,
        min_source_quality=args.min_source_quality,
        min_utility=args.min_utility,
    )
    manifest, profile_rows, selection_payload = select_pool(candidates, args)

    rejected_rows = low_quality_rejects + selection_payload["selection_rejects"]
    summary = {
        "score_jsonl": [str(path) for path in score_paths],
        "manifest_jsonl": [str(path) for path in manifest_paths],
        "feedback_proposals": [str(path) for path in feedback_paths],
        "input_score_rows": len(score_rows),
        "input_manifest_rows": len(manifest_rows),
        "deduped_source_records": len(records),
        "feedback_record_edit_pairs": len(feedback),
        "min_source_quality": args.min_source_quality,
        "min_utility": args.min_utility,
        "max_records": args.max_records,
        "max_per_source": args.max_per_source,
        "min_hash_distance": args.min_hash_distance,
        **selection_payload["summary"],
        "reject_counts": dict(sorted(Counter(row.get("reason", "unknown") for row in rejected_rows).items())),
    }
    return {
        "manifest": manifest,
        "profile_rows": profile_rows,
        "rejected_rows": rejected_rows,
        "summary": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a reward-aware unlabeled training manifest. The script scores each source image "
            "per edit type, optionally uses prior self-evolution proposal feedback, and emits an "
            "auditable manifest for conservative image-editing self-evolution."
        )
    )
    parser.add_argument("--score-jsonl", action="append", required=True, help="Source-selection score JSONL. Repeatable.")
    parser.add_argument("--manifest-jsonl", action="append", default=[], help="Optional selected source manifest JSONL. Repeatable.")
    parser.add_argument(
        "--feedback-proposals",
        action="append",
        default=[],
        help="Optional prior proposal JSONL path or glob used to estimate edit-type utility. Repeatable.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output reward-aware training manifest JSONL.")
    parser.add_argument("--profile-output", type=Path, default=None, help="Output all candidate source/edit profiles JSONL.")
    parser.add_argument("--rejected-output", type=Path, default=None, help="Output rejected candidate/source audit JSONL.")
    parser.add_argument("--summary", type=Path, default=None, help="Output summary JSON.")
    parser.add_argument("--max-records", type=int, default=2048)
    parser.add_argument("--max-per-source", type=int, default=2)
    parser.add_argument("--min-source-quality", type=float, default=0.55)
    parser.add_argument("--min-utility", type=float, default=0.26)
    parser.add_argument("--min-hash-distance", type=int, default=6)
    parser.add_argument("--relax-hash-on-backfill", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target-fractions", default=None, help="Comma map like object_removal=0.16,color_change=0.10.")
    parser.add_argument("--seed", type=int, default=123)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = build_pool(args)
    write_jsonl(args.output, payload["manifest"])
    if args.profile_output:
        write_jsonl(args.profile_output, payload["profile_rows"])
    if args.rejected_output:
        write_jsonl(args.rejected_output, payload["rejected_rows"])
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    write_json(summary_path, payload["summary"])
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
