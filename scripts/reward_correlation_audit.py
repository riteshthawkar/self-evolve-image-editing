#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


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


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def gmean(values: list[float]) -> float:
    cleaned = [max(clamp(v), 1.0e-6) for v in values if math.isfinite(v)]
    if not cleaned:
        return 0.0
    return math.exp(sum(math.log(v) for v in cleaned) / len(cleaned))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def round_index_from_path(path: Path) -> int:
    for part in path.parts:
        if part.startswith("round_"):
            try:
                return int(part.split("_", 1)[1])
            except ValueError:
                return -1
    return -1


def evaluator(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("evaluator") or row.get("solver") or {}
    return value if isinstance(value, dict) else {}


def component(row: dict[str, Any], *names: str, default: float = 0.0) -> float:
    ev = evaluator(row)
    containers = [
        ev.get("component_scores", {}),
        ev.get("signals", {}),
        ev,
    ]
    for name in names:
        for container in containers:
            if isinstance(container, dict) and name in container:
                value = finite_float(container.get(name), math.nan)
                if math.isfinite(value):
                    return value
    return default


def proposal(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("proposal") or {}
    return value if isinstance(value, dict) else {}


def structured_edit(row: dict[str, Any]) -> dict[str, Any]:
    value = proposal(row).get("structured_edit") or {}
    return value if isinstance(value, dict) else {}


def edit_type(row: dict[str, Any]) -> str:
    return str(structured_edit(row).get("edit_type") or proposal(row).get("family") or "unknown")


def candidate_role(row: dict[str, Any]) -> str:
    return str(row.get("candidate_role") or "policy")


def is_policy(row: dict[str, Any]) -> bool:
    return candidate_role(row) == "policy"


def is_accepted(row: dict[str, Any]) -> bool:
    if str(row.get("status")) == "accepted":
        return True
    return component(row, "accepted_by_ranker", default=0.0) >= 0.5


def judge_reliable(row: dict[str, Any]) -> bool:
    return component(row, "internal_vlm_judge_reliable", default=0.0) >= 0.5


def judge_supported(row: dict[str, Any]) -> bool:
    return component(row, "internal_vlm_judge_supported", default=0.0) >= 0.5


def judge_label(row: dict[str, Any]) -> float | None:
    if not judge_reliable(row):
        return None
    return clamp(component(row, "internal_vlm_judge_score", default=0.0))


def expected_change_range(row: dict[str, Any]) -> tuple[float, float]:
    raw = proposal(row).get("expected_changed_fraction")
    if isinstance(raw, list) and len(raw) >= 2:
        lo = finite_float(raw[0], 0.03)
        hi = finite_float(raw[1], 0.65)
    else:
        ranges = {
            "attribute_change": (0.03, 0.45),
            "color_change": (0.03, 0.45),
            "material_change": (0.04, 0.50),
            "object_replacement": (0.05, 0.70),
            "object_removal": (0.04, 0.55),
            "object_addition": (0.04, 0.60),
            "spatial_move": (0.05, 0.70),
            "style_transfer": (0.15, 0.90),
            "background_change": (0.12, 0.85),
            "local_enhancement": (0.03, 0.45),
        }
        lo, hi = ranges.get(edit_type(row), (0.04, 0.65))
    return max(0.0, lo), min(1.0, max(lo, hi))


def noop_flag(row: dict[str, Any]) -> bool:
    lo, _hi = expected_change_range(row)
    changed = component(row, "cepr_latent_changed_fraction", default=0.0)
    true_gain = component(row, "cepr_true_prompt_gain", default=0.0)
    required_gain = component(row, "rubric_required_after_gain", default=0.0)
    judge_sem = component(row, "internal_vlm_judge_semantic", default=1.0)
    if judge_supported(row) and component(row, "internal_vlm_judge_score", default=1.0) <= 0.05:
        return True
    return changed < max(0.01, lo * 0.50) and true_gain <= 0.0 and required_gain <= 0.0 and judge_sem < 0.50


def drift_flag(row: dict[str, Any]) -> bool:
    _lo, hi = expected_change_range(row)
    changed = component(row, "cepr_latent_changed_fraction", default=0.0)
    outside = component(row, "cepr_latent_outside_preservation", default=1.0)
    preservation = min(
        component(row, "cepr_preservation", default=1.0),
        component(row, "rubric_preservation", default=1.0),
    )
    return changed > min(1.0, hi * 1.20) or outside < 0.55 or preservation < 0.58


def score_cepr_raw(row: dict[str, Any]) -> float:
    return clamp(component(row, "cepr_raw_reward", "cepr_reward", default=0.0))


def score_rubric(row: dict[str, Any]) -> float:
    return clamp(component(row, "rubric_reward", default=0.0))


def score_judge_strict(row: dict[str, Any]) -> float:
    if not judge_supported(row):
        return 0.0
    return clamp(component(row, "internal_vlm_judge_score", default=0.0))


def score_conservative(row: dict[str, Any]) -> float:
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
    values = [
        score_cepr_raw(row),
        semantic,
        preservation,
        validity,
        component(row, "cepr_taxonomy", default=1.0),
    ]
    if judge_reliable(row):
        values.extend(
            [
                component(row, "internal_vlm_judge_score", default=0.0),
                component(row, "internal_vlm_judge_preservation", default=0.0),
                component(row, "internal_vlm_judge_artifact_free", default=0.0),
            ]
        )
    return gmean(values)


def score_esc_minimal(row: dict[str, Any], group_stats: dict[str, float]) -> float:
    change = max(
        component(row, "rubric_edit_success", default=0.0),
        component(row, "rubric_required_after", default=0.0),
        component(row, "cepr_semantic_edit", default=0.0),
    )
    preservation = min(
        component(row, "cepr_preservation", default=0.0),
        component(row, "rubric_preservation", default=0.0),
    )
    validity = min(
        component(row, "cepr_validity", default=0.0),
        component(row, "rubric_validity", default=0.0),
    )
    if judge_supported(row):
        # Do not fail open when the internal judge explicitly says the edit failed.
        judge_score = component(row, "internal_vlm_judge_score", default=0.0)
        judge_sem = component(row, "internal_vlm_judge_semantic", default=0.0)
        judge_pres = component(row, "internal_vlm_judge_preservation", default=0.0)
        judge_art = component(row, "internal_vlm_judge_artifact_free", default=0.0)
        if judge_reliable(row) or judge_score <= 0.05:
            change = min(change, judge_sem)
            preservation = min(preservation, judge_pres)
            validity = min(validity, judge_art)
    base = gmean([change, preservation, validity])
    if noop_flag(row):
        base *= 0.20
    if drift_flag(row):
        base *= 0.45
    # Reward candidates whose change magnitude is close to the group's stable region.
    changed = component(row, "cepr_latent_changed_fraction", default=0.0)
    std = group_stats.get("changed_fraction_std", 0.0)
    med = group_stats.get("changed_fraction_median", changed)
    if std > 0.0:
        z = abs(changed - med) / max(std, 1.0e-6)
        base *= 0.50 + 0.50 * math.exp(-0.50 * z * z)
    return clamp(base)


SCORES = {
    "cepr_raw": score_cepr_raw,
    "rubric_reward": score_rubric,
    "current_conservative": score_conservative,
    "judge_strict": score_judge_strict,
    "esc_minimal": score_esc_minimal,
}


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def group_stats(rows: list[dict[str, Any]]) -> dict[str, float]:
    changed = [component(row, "cepr_latent_changed_fraction", default=math.nan) for row in rows]
    changed = [value for value in changed if math.isfinite(value)]
    if len(changed) <= 1:
        return {
            "changed_fraction_mean": changed[0] if changed else 0.0,
            "changed_fraction_median": changed[0] if changed else 0.0,
            "changed_fraction_std": 0.0,
        }
    avg = mean(changed)
    var = sum((value - avg) ** 2 for value in changed) / len(changed)
    return {
        "changed_fraction_mean": avg,
        "changed_fraction_median": median(changed),
        "changed_fraction_std": math.sqrt(var),
    }


def pairwise_accuracy(rows: list[dict[str, Any]], score_name: str, stats: dict[str, float]) -> tuple[int, int]:
    labelled = [(row, judge_label(row)) for row in rows]
    labelled = [(row, label) for row, label in labelled if label is not None]
    correct = 0
    total = 0
    score_fn = SCORES[score_name]
    for i in range(len(labelled)):
        for j in range(i + 1, len(labelled)):
            row_a, label_a = labelled[i]
            row_b, label_b = labelled[j]
            if abs(label_a - label_b) < 0.05:
                continue
            if score_name == "esc_minimal":
                score_a = score_fn(row_a, stats)
                score_b = score_fn(row_b, stats)
            else:
                score_a = score_fn(row_a)
                score_b = score_fn(row_b)
            if abs(score_a - score_b) < 1.0e-9:
                continue
            total += 1
            correct += int((score_a > score_b) == (label_a > label_b))
    return correct, total


def top1_by_score(rows: list[dict[str, Any]], score_name: str, stats: dict[str, float]) -> dict[str, Any] | None:
    if not rows:
        return None
    score_fn = SCORES[score_name]
    if score_name == "esc_minimal":
        return max(rows, key=lambda row: score_fn(row, stats))
    return max(rows, key=score_fn)


def row_brief(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "round": row.get("_round"),
        "group_id": row.get("group_id"),
        "candidate_index": row.get("candidate_index"),
        "role": candidate_role(row),
        "status": row.get("status"),
        "edit_type": edit_type(row),
        "instruction": proposal(row).get("instruction"),
        "cepr_raw": round(score_cepr_raw(row), 4),
        "rubric_reward": round(score_rubric(row), 4),
        "judge_score": round(component(row, "internal_vlm_judge_score", default=0.0), 4),
        "judge_confidence": round(component(row, "internal_vlm_judge_confidence", default=0.0), 4),
        "judge_reliable": judge_reliable(row),
        "true_prompt_gain": round(component(row, "cepr_true_prompt_gain", default=0.0), 4),
        "required_after_gain": round(component(row, "rubric_required_after_gain", default=0.0), 4),
        "changed_fraction": round(component(row, "cepr_latent_changed_fraction", default=0.0), 4),
        "outside_preservation": round(component(row, "cepr_latent_outside_preservation", default=0.0), 4),
        "reason": str(component_reason(row))[:220],
    }


def component_reason(row: dict[str, Any]) -> str:
    ev = evaluator(row)
    signals = ev.get("signals", {})
    if isinstance(signals, dict):
        return str(signals.get("internal_vlm_judge_reason") or signals.get("rubric_reject_reason") or "")
    return ""


def load_run_rows(run_dir: Path, rounds: list[int] | None) -> list[dict[str, Any]]:
    paths = sorted(run_dir.glob("round_*/proposals.jsonl"), key=round_index_from_path)
    rows: list[dict[str, Any]] = []
    allowed = set(rounds or [])
    for path in paths:
        round_idx = round_index_from_path(path)
        if allowed and round_idx not in allowed:
            continue
        for row in load_jsonl(path):
            row["_round"] = round_idx
            rows.append(row)
    return rows


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("group_id"))].append(row)
    return dict(grouped)


def summarize_signal_alignment(groups: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    signal_rows: list[dict[str, Any]] = []
    totals = {
        name: {
            "groups_with_label": 0,
            "top1_agree": 0,
            "top1_label_sum": 0.0,
            "top1_low_label": 0,
            "pair_correct": 0,
            "pair_total": 0,
        }
        for name in SCORES
    }
    for group_id, rows in groups.items():
        labels = [(row, judge_label(row)) for row in rows]
        labels = [(row, label) for row, label in labels if label is not None]
        if len(labels) < 2:
            continue
        stats = group_stats(rows)
        proxy_best = max(labels, key=lambda item: item[1])[0]
        proxy_best_id = int(proxy_best.get("candidate_index", -999))
        for name in SCORES:
            top = top1_by_score(rows, name, stats)
            if top is None:
                continue
            top_id = int(top.get("candidate_index", -999))
            top_label = judge_label(top)
            agree = top_id == proxy_best_id
            correct, total = pairwise_accuracy(rows, name, stats)
            totals[name]["groups_with_label"] += 1
            totals[name]["top1_agree"] += int(agree)
            totals[name]["top1_label_sum"] += top_label if top_label is not None else 0.0
            totals[name]["top1_low_label"] += int(top_label is not None and top_label < 0.35)
            totals[name]["pair_correct"] += correct
            totals[name]["pair_total"] += total
            signal_rows.append(
                {
                    "group_id": group_id,
                    "round": top.get("_round"),
                    "edit_type": edit_type(top),
                    "signal": name,
                    "top_candidate_index": top_id,
                    "proxy_best_candidate_index": proxy_best_id,
                    "top1_agrees_with_proxy": agree,
                    "top1_judge_label": top_label,
                    "pairwise_correct": correct,
                    "pairwise_total": total,
                }
            )
    summary = {}
    for name, item in totals.items():
        n = item["groups_with_label"]
        pair_total = item["pair_total"]
        summary[name] = {
            "groups_with_reliable_judge_labels": n,
            "top1_agreement": item["top1_agree"] / n if n else None,
            "mean_top1_judge_label": item["top1_label_sum"] / n if n else None,
            "top1_low_judge_label_rate": item["top1_low_label"] / n if n else None,
            "pairwise_accuracy": item["pair_correct"] / pair_total if pair_total else None,
            "pairwise_pairs": pair_total,
        }
    return signal_rows, summary


def summarize_accepted_failures(rows: list[dict[str, Any]], max_examples: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    accepted = [row for row in rows if is_accepted(row)]
    counts = Counter()
    by_type = Counter()
    bad_examples = []
    for row in accepted:
        by_type[edit_type(row)] += 1
        if judge_supported(row) and not judge_reliable(row):
            counts["vlm_supported_but_unreliable"] += 1
        if component(row, "internal_vlm_judge_score", default=1.0) < 0.35:
            counts["vlm_score_below_035"] += 1
        if component(row, "internal_vlm_judge_semantic", default=1.0) < 0.35:
            counts["vlm_semantic_below_035"] += 1
        if component(row, "cepr_true_prompt_gain", default=0.0) <= 0.0:
            counts["nonpositive_true_prompt_gain"] += 1
        if component(row, "rubric_required_after_gain", default=0.0) <= 0.0:
            counts["nonpositive_required_after_gain"] += 1
        if noop_flag(row):
            counts["noop_flag"] += 1
        if drift_flag(row):
            counts["drift_flag"] += 1
        if (
            score_cepr_raw(row) >= 0.60
            and component(row, "internal_vlm_judge_score", default=1.0) < 0.35
        ):
            counts["high_cepr_low_vlm"] += 1
            if len(bad_examples) < max_examples:
                bad_examples.append(row_brief(row))
    summary = {
        "accepted_count": len(accepted),
        "accepted_by_edit_type": dict(by_type),
        "failure_counts": dict(counts),
        "failure_rates": {
            key: value / len(accepted) if accepted else None
            for key, value in counts.items()
        },
    }
    return summary, bad_examples


def summarize_group_entropy(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    buckets = Counter()
    edit_type_stats: dict[str, Counter] = defaultdict(Counter)
    groups_seen = 0
    for rows in groups.values():
        policies = [row for row in rows if is_policy(row)]
        if not policies:
            continue
        groups_seen += 1
        success = 0
        for row in policies:
            if (
                judge_reliable(row)
                and component(row, "internal_vlm_judge_score", default=0.0) >= 0.55
                and component(row, "internal_vlm_judge_semantic", default=0.0) >= 0.55
                and component(row, "internal_vlm_judge_preservation", default=0.0) >= 0.55
                and component(row, "internal_vlm_judge_artifact_free", default=0.0) >= 0.55
            ):
                success += 1
        rate = success / len(policies)
        if rate == 0:
            bucket = "all_fail"
        elif rate == 1:
            bucket = "all_pass"
        elif 0.25 <= rate <= 0.75:
            bucket = "productive_band_025_075"
        else:
            bucket = "edge_band"
        buckets[bucket] += 1
        edit_type_stats[edit_type(policies[0])][bucket] += 1
    return {
        "policy_groups": groups_seen,
        "success_entropy_buckets": dict(buckets),
        "success_entropy_rates": {
            key: value / groups_seen if groups_seen else None
            for key, value in buckets.items()
        },
        "by_edit_type": {key: dict(value) for key, value in edit_type_stats.items()},
    }


def load_eval_summaries(imgedit_summary: Path | None, gedit_summary: Path | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if imgedit_summary and imgedit_summary.exists():
        data = load_json(imgedit_summary)
        metrics = data.get("metrics", {})
        out["imgedit"] = {
            "model_name": data.get("model_name"),
            "overall_average": metrics.get("overall_average"),
            "type_scores": metrics.get("type_scores", {}),
            "count": metrics.get("count"),
        }
    if gedit_summary and gedit_summary.exists():
        data = load_json(gedit_summary)
        metrics = data.get("metrics", {})
        out["gedit"] = {
            "model_name": data.get("model_name"),
            "average": metrics.get("average", {}),
            "groups": metrics.get("groups", {}),
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def make_report(summary: dict[str, Any]) -> str:
    signal_summary = summary["signal_alignment"]
    accepted = summary["accepted_failure_audit"]
    entropy = summary["group_entropy"]
    evals = summary.get("eval_summaries", {})

    lines = [
        "# Reward Correlation Audit",
        "",
        f"Run: `{summary['run_dir']}`",
        f"Rows: {summary['row_count']} candidates, {summary['group_count']} groups",
        "",
        "## Evaluation Context",
    ]
    if "imgedit" in evals:
        img = evals["imgedit"]
        lines.append(f"- ImgEdit `{img['model_name']}`: {img['overall_average']}")
    if "gedit" in evals:
        avg = evals["gedit"].get("average", {})
        lines.append(
            f"- GEdit `{evals['gedit']['model_name']}`: overall={avg.get('overall')}, "
            f"semantics={avg.get('semantics')}, quality={avg.get('quality')}"
        )
    if not evals:
        lines.append("- No evaluation summaries were supplied.")
    lines.extend(["", "## Signal Alignment With Reliable Internal VLM Labels", ""])
    lines.append("| signal | top1 agreement | pairwise accuracy | mean top1 VLM label | low-label top1 rate | groups |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, item in sorted(signal_summary.items()):
        def fmt(value: Any) -> str:
            return "n/a" if value is None else f"{float(value):.3f}"
        lines.append(
            f"| {name} | {fmt(item['top1_agreement'])} | {fmt(item['pairwise_accuracy'])} | "
            f"{fmt(item['mean_top1_judge_label'])} | {fmt(item['top1_low_judge_label_rate'])} | "
            f"{item['groups_with_reliable_judge_labels']} |"
        )
    lines.extend(["", "## Accepted Candidate Failure Audit", ""])
    lines.append(f"- Accepted candidates: {accepted['accepted_count']}")
    for key, value in sorted(accepted["failure_rates"].items()):
        lines.append(f"- {key}: {value:.3f} ({accepted['failure_counts'][key]})")
    lines.extend(["", "## Proposer/Productive Difficulty Audit", ""])
    lines.append(f"- Policy groups: {entropy['policy_groups']}")
    for key, value in sorted(entropy["success_entropy_rates"].items()):
        lines.append(f"- {key}: {value:.3f} ({entropy['success_entropy_buckets'][key]})")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If `cepr_raw` or `rubric_reward` has low top-1 agreement with reliable VLM labels, current per-candidate scoring is not a strong enough training signal.",
            "- Accepted high-CEPR/low-VLM examples indicate fail-open VLM behavior: the judge detects bad edits, but the candidate can still be selected because CEPR remains high.",
            "- A high `all_fail` proposer bucket means many instructions are too hard/ambiguous for current editor training. A high `all_pass` bucket means they are too easy. Productive self-evolution should focus on the middle band.",
            "",
            "## Recommended Next Reward",
            "",
            "Use a group-level Edit Self-Consistency reward before the next training run:",
            "",
            "`R_ESC = change_consistency + preservation_consistency - no_op/global_drift_penalty`",
            "",
            "Operational changes:",
            "",
            "1. Do not fail open on internal VLM judge outputs that explicitly score edit success near zero.",
            "2. Select training pairs only from groups with productive success entropy, not all-success or all-fail groups.",
            "3. Rank candidates with strict change verification plus preservation/no-op gates, then use CEPR/rubric as secondary tie-breakers.",
            "4. Use proposer reward based on medium success rate across K samples.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_rounds(value: str | None) -> list[int] | None:
    if not value:
        return None
    rounds: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            rounds.extend(range(int(start), int(end) + 1))
        else:
            rounds.append(int(part))
    return sorted(set(rounds))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit self-evolution reward signals against reliable proxy labels.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/self_evolve/qwen_edit_2509_conservative_pairwise_full_v1_20260604T183826"),
    )
    parser.add_argument("--rounds", type=str, default=None, help="Comma/range list, e.g. 1-18 or 4,12,18")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument(
        "--imgedit-summary",
        type=Path,
        default=Path("outputs/scores/imgedit/self_evolve_pairwise_r18_imgedit_summary.json"),
    )
    parser.add_argument(
        "--gedit-summary",
        type=Path,
        default=Path("outputs/scores/gedit/self_evolve_pairwise_r18_gedit_summary.json"),
    )
    args = parser.parse_args()

    rows = load_run_rows(args.run_dir, parse_rounds(args.rounds))
    groups = group_rows(rows)
    signal_rows, signal_summary = summarize_signal_alignment(groups)
    accepted_summary, bad_examples = summarize_accepted_failures(rows, args.max_examples)
    entropy_summary = summarize_group_entropy(groups)

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path("outputs/analysis") / f"reward_correlation_audit_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "run_dir": str(args.run_dir),
        "rounds": parse_rounds(args.rounds),
        "row_count": len(rows),
        "group_count": len(groups),
        "signal_alignment": signal_summary,
        "accepted_failure_audit": accepted_summary,
        "bad_accepted_examples": bad_examples,
        "group_entropy": entropy_summary,
        "eval_summaries": load_eval_summaries(args.imgedit_summary, args.gedit_summary),
    }

    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(output_dir / "signal_alignment_by_group.csv", signal_rows)
    with (output_dir / "bad_accepted_examples.jsonl").open("w", encoding="utf-8") as handle:
        for item in bad_examples:
            handle.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")
    (output_dir / "REPORT.md").write_text(make_report(summary), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
