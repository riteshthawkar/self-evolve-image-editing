from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from qwen_edit_project.self_evolve.edit_schema import normalize_structured_edit
from qwen_edit_project.utils.paths import ensure_dir


def _evaluation(row: dict[str, Any]) -> dict[str, Any]:
    return row.get("evaluator") or row.get("solver") or {}


def _finite_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def _difficulty_band(acceptance_rate: float, center: float, sigma: float) -> float:
    return math.exp(-((acceptance_rate - center) ** 2) / (2.0 * max(sigma, 1e-6) ** 2))


def _proposal_target(proposal: dict[str, Any]) -> dict[str, Any]:
    instruction = str(proposal.get("instruction", "")).strip()
    structured_edit = proposal.get("structured_edit") if isinstance(proposal.get("structured_edit"), dict) else {}
    normalized = normalize_structured_edit(
        structured_edit,
        instruction=instruction,
        family=proposal.get("family"),
    )
    normalized["instruction"] = instruction
    return normalized


def _record_prompt(row: dict[str, Any]) -> str:
    caption = row.get("caption")
    proposal = row.get("proposal", {})
    parts = [
        "Generate one feasible, image-grounded edit instruction for self-training.",
        "Return only the structured edit JSON.",
    ]
    if caption:
        parts.append(f"Image caption: {caption}")
    if proposal.get("difficulty_level") is not None:
        parts.append(f"Target difficulty level: {proposal['difficulty_level']}")
    return "\n".join(parts)


def build_proposer_training_records(
    candidate_payloads: list[dict[str, Any]],
    *,
    reward_center: float = 0.50,
    reward_sigma: float = 0.25,
    min_reward: float = 0.35,
    min_quality: float = 0.30,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_payloads:
        group_id = str(row.get("group_id") or "")
        if group_id:
            groups[group_id].append(row)

    records: list[dict[str, Any]] = []
    all_rewards = []
    selected = 0
    for group_id, rows in groups.items():
        if not rows:
            continue
        first = rows[0]
        proposal = first.get("proposal", {})
        evaluations = [_evaluation(row) for row in rows]
        accepted_rows = [row for row in rows if row.get("status") == "accepted"]
        acceptance_rate = len(accepted_rows) / max(len(rows), 1)
        totals = [_finite_float(evaluation.get("total_score")) for evaluation in evaluations]
        raw_rewards = [
            _finite_float(evaluation.get("component_scores", {}).get("cepr_raw_reward"))
            for evaluation in evaluations
        ]
        edit_scores = [
            _finite_float(evaluation.get("component_scores", {}).get("cepr_semantic_edit"))
            or _finite_float(evaluation.get("component_scores", {}).get("cepr_edit_specificity"))
            for evaluation in evaluations
        ]
        preservation_scores = [
            _finite_float(evaluation.get("component_scores", {}).get("cepr_preservation"))
            for evaluation in evaluations
        ]
        validity_scores = [
            _finite_float(evaluation.get("component_scores", {}).get("cepr_validity"))
            for evaluation in evaluations
        ]
        best_total = max(totals) if totals else 0.0
        best_raw = max(raw_rewards) if raw_rewards else best_total
        quality = max(best_total, best_raw)
        band = _difficulty_band(acceptance_rate, reward_center, reward_sigma)
        edit = max(edit_scores) if edit_scores else 0.0
        preservation = max(preservation_scores) if preservation_scores else 0.0
        validity = max(validity_scores) if validity_scores else 0.0
        proposal_reward = 0.40 * band + 0.35 * quality + 0.15 * edit + 0.05 * preservation + 0.05 * validity
        if acceptance_rate <= 0.0:
            proposal_reward *= 0.35
        elif acceptance_rate >= 0.95:
            proposal_reward *= 0.75
        proposal_reward = max(0.0, min(1.0, proposal_reward))
        all_rewards.append(proposal_reward)
        use_for_sft = proposal_reward >= min_reward and quality >= min_quality and bool(proposal.get("instruction"))
        selected += int(use_for_sft)
        records.append(
            {
                "type": "proposer_sft",
                "group_id": group_id,
                "record_key": first.get("record_key"),
                "source_image": first.get("image_path"),
                "prompt": _record_prompt(first),
                "target": _proposal_target(proposal),
                "target_json": json.dumps(_proposal_target(proposal), ensure_ascii=True, sort_keys=True),
                "reward": proposal_reward,
                "use_for_sft": use_for_sft,
                "metrics": {
                    "acceptance_rate": acceptance_rate,
                    "best_total_score": best_total,
                    "best_raw_reward": best_raw,
                    "best_semantic_edit": edit,
                    "best_preservation": preservation,
                    "best_validity": validity,
                    "difficulty_band": band,
                    "candidate_count": len(rows),
                    "accepted_count": len(accepted_rows),
                },
            }
        )

    summary = {
        "groups": len(groups),
        "records": len(records),
        "selected_for_sft": selected,
        "avg_proposal_reward": sum(all_rewards) / max(len(all_rewards), 1),
        "max_proposal_reward": max(all_rewards) if all_rewards else 0.0,
        "min_reward": min_reward,
        "min_quality": min_quality,
        "reward_center": reward_center,
        "reward_sigma": reward_sigma,
    }
    return records, summary


def write_proposer_training_jsonl(records: list[dict[str, Any]], path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return path
