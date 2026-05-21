from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from qwen_edit_project.utils.config import save_json
from qwen_edit_project.utils.paths import ensure_dir, resolve_path


CEPR_COMPONENTS = [
    "cepr_edit_specificity",
    "cepr_preservation",
    "cepr_validity",
    "cepr_raw_reward",
    "cepr_reward",
]

CEPR_SIGNALS = [
    "cepr_true_prompt_gain",
    "cepr_max_distractor_gain",
    "cepr_contrastive_margin",
    "cepr_contrastive_score",
    "cepr_absolute_edit_score",
    "cepr_semantic_preservation_score",
    "cepr_semantic_preservation_cosine",
    "cepr_latent_outside_preservation",
    "cepr_latent_region_score",
    "cepr_latent_validity_score",
    "cepr_latent_changed_fraction",
    "cepr_latent_inside_delta",
    "cepr_latent_outside_delta",
    "cepr_latent_total_delta",
    "cepr_latent_delta_std",
    "cepr_latent_drift_score",
]

DEFAULT_SWEEP = {
    "edit": [0.35, 0.40, 0.45, 0.48, 0.50, 0.53],
    "preservation": [0.10, 0.15, 0.20, 0.25, 0.30, 0.45, 0.60],
    "validity": [0.20, 0.35, 0.50, 0.65],
    "reward": [0.25, 0.30, 0.35, 0.40, 0.45, 0.52],
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def numeric_values(rows: list[dict[str, Any]], key: str, source: str) -> list[float]:
    values = []
    for row in rows:
        solver = row.get("solver", {})
        container = solver.get(source, {})
        value = container.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "mean": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "max": None,
            "std": None,
        }
    ordered = sorted(values)

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = q * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(values),
        "min": ordered[0],
        "p10": percentile(0.10),
        "p25": percentile(0.25),
        "mean": mean(values),
        "p50": percentile(0.50),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "max": ordered[-1],
        "std": pstdev(values) if len(values) > 1 else 0.0,
    }


def row_score(row: dict[str, Any], component: str, signal: str | None = None) -> float:
    solver = row.get("solver", {})
    if signal is not None:
        value = solver.get("signals", {}).get(signal, 0.0)
    else:
        value = solver.get("component_scores", {}).get(component, 0.0)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else 0.0


def gate_failures(row: dict[str, Any], thresholds: dict[str, float]) -> list[str]:
    failures = []
    if row_score(row, "cepr_edit_specificity") < thresholds["edit"]:
        failures.append("edit_specificity")
    if row_score(row, "cepr_preservation") < thresholds["preservation"]:
        failures.append("preservation")
    if row_score(row, "cepr_validity") < thresholds["validity"]:
        failures.append("validity")
    if row_score(row, "cepr_raw_reward") < thresholds["reward"]:
        failures.append("raw_reward")
    return failures


def grouped_by_proposal(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("group_id", ""))].append(row)
    return dict(groups)


def threshold_acceptance(rows: list[dict[str, Any]], thresholds: dict[str, float], top_m: int = 1) -> dict[str, Any]:
    groups = grouped_by_proposal(rows)
    accepted_rows = []
    feasible_rows = []
    for group_rows in groups.values():
        feasible = [row for row in group_rows if not gate_failures(row, thresholds)]
        feasible_rows.extend(feasible)
        feasible = sorted(feasible, key=lambda row: row_score(row, "cepr_raw_reward"), reverse=True)
        accepted_rows.extend(feasible[:top_m])

    accepted_group_count = len({row.get("group_id") for row in accepted_rows})
    return {
        "groups": len(groups),
        "candidates": len(rows),
        "feasible_candidates": len(feasible_rows),
        "accepted_candidates": len(accepted_rows),
        "accepted_groups": accepted_group_count,
        "candidate_acceptance_rate": len(accepted_rows) / max(len(rows), 1),
        "group_acceptance_rate": accepted_group_count / max(len(groups), 1),
    }


def component_correlations(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    output: dict[str, dict[str, float | None]] = {}
    for key in CEPR_COMPONENTS + CEPR_SIGNALS:
        source = "component_scores" if key in CEPR_COMPONENTS else "signals"
        pairs: list[tuple[float, float]] = []
        for row in rows:
            solver = row.get("solver", {})
            target_value = solver.get("component_scores", {}).get("cepr_raw_reward")
            value = solver.get(source, {}).get(key)
            if (
                isinstance(target_value, (int, float))
                and isinstance(value, (int, float))
                and math.isfinite(float(target_value))
                and math.isfinite(float(value))
            ):
                pairs.append((float(value), float(target_value)))
        if len(pairs) <= 1:
            output[key] = {"pearson_with_raw_reward": None, "pair_count": len(pairs)}
            continue
        values = [pair[0] for pair in pairs]
        target = [pair[1] for pair in pairs]
        x_mean = mean(values)
        y_mean = mean(target)
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(values, target))
        x_var = sum((x - x_mean) ** 2 for x in values)
        y_var = sum((y - y_mean) ** 2 for y in target)
        corr = numerator / math.sqrt(x_var * y_var) if x_var > 0 and y_var > 0 else None
        output[key] = {"pearson_with_raw_reward": corr, "pair_count": len(pairs)}
    return output


def summarize(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    runtime_errors = Counter()
    scoring_devices = Counter()
    recovered_gpu_oom = 0
    operations = Counter()
    families = Counter()
    gate_counter = Counter()
    first_failure_counter = Counter()

    for row in rows:
        proposal = row.get("proposal", {})
        operations[str(proposal.get("operation_id", "unknown"))] += 1
        families[str(proposal.get("family", "unknown"))] += 1
        signals = row.get("solver", {}).get("signals", {})
        error_type = signals.get("cepr_candidate_runtime_error_type")
        if error_type:
            runtime_errors[str(error_type)] += 1
        scoring_device = signals.get("cepr_scoring_device")
        if scoring_device:
            scoring_devices[str(scoring_device)] += 1
        recovered_gpu_oom += int(float(signals.get("cepr_gpu_oom_recovered", 0.0)))
        failures = gate_failures(row, thresholds)
        if failures:
            for failure in failures:
                gate_counter[failure] += 1
            first_failure_counter[failures[0]] += 1

    component_summary = {
        key: describe(numeric_values(rows, key, "component_scores"))
        for key in CEPR_COMPONENTS
    }
    signal_summary = {
        key: describe(numeric_values(rows, key, "signals"))
        for key in CEPR_SIGNALS
    }

    return {
        "counts": {
            "candidates": len(rows),
            "groups": len(grouped_by_proposal(rows)),
            "status": dict(status_counts),
            "runtime_errors": dict(runtime_errors),
            "scoring_devices": dict(scoring_devices),
            "gpu_oom_recovered": recovered_gpu_oom,
            "operations": dict(operations),
            "families": dict(families),
        },
        "thresholds": thresholds,
        "threshold_acceptance": threshold_acceptance(rows, thresholds),
        "gate_failures": {
            "all_failures": dict(gate_counter),
            "first_failure": dict(first_failure_counter),
        },
        "components": component_summary,
        "signals": signal_summary,
        "correlations": component_correlations(rows),
    }


def write_threshold_sweep(rows: list[dict[str, Any]], output_path: Path) -> list[dict[str, Any]]:
    sweep_rows = []
    for edit in DEFAULT_SWEEP["edit"]:
        for preservation in DEFAULT_SWEEP["preservation"]:
            for validity in DEFAULT_SWEEP["validity"]:
                for reward in DEFAULT_SWEEP["reward"]:
                    thresholds = {
                        "edit": edit,
                        "preservation": preservation,
                        "validity": validity,
                        "reward": reward,
                    }
                    summary = threshold_acceptance(rows, thresholds)
                    sweep_rows.append({**thresholds, **summary})

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)
    return sweep_rows


def write_component_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "group_id",
        "candidate_index",
        "status",
        "operation_id",
        "family",
        *CEPR_COMPONENTS,
        *CEPR_SIGNALS,
        "feasible",
        "accepted_by_ranker",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            proposal = row.get("proposal", {})
            solver = row.get("solver", {})
            component_scores = solver.get("component_scores", {})
            signals = solver.get("signals", {})
            writer.writerow(
                {
                    "group_id": row.get("group_id"),
                    "candidate_index": row.get("candidate_index"),
                    "status": row.get("status"),
                    "operation_id": proposal.get("operation_id"),
                    "family": proposal.get("family"),
                    **{key: component_scores.get(key) for key in CEPR_COMPONENTS},
                    **{key: signals.get(key) for key in CEPR_SIGNALS},
                    "feasible": signals.get("feasible"),
                    "accepted_by_ranker": signals.get("accepted_by_ranker"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze self-evolve reward components from proposals.jsonl.")
    parser.add_argument("--proposals", required=True, help="Path to round proposals.jsonl.")
    parser.add_argument("--output-dir", default=None, help="Directory for reward analysis outputs.")
    parser.add_argument("--edit-threshold", type=float, default=0.45)
    parser.add_argument("--preservation-threshold", type=float, default=0.20)
    parser.add_argument("--validity-threshold", type=float, default=0.50)
    parser.add_argument("--reward-threshold", type=float, default=0.30)
    args = parser.parse_args()

    proposals_path = resolve_path(args.proposals)
    if proposals_path is None or not proposals_path.exists():
        raise FileNotFoundError(f"proposals.jsonl not found: {args.proposals}")

    output_dir = resolve_path(args.output_dir) if args.output_dir else proposals_path.parent / "reward_analysis"
    if output_dir is None:
        raise ValueError("Could not resolve output directory")
    ensure_dir(output_dir)

    rows = load_jsonl(proposals_path)
    thresholds = {
        "edit": args.edit_threshold,
        "preservation": args.preservation_threshold,
        "validity": args.validity_threshold,
        "reward": args.reward_threshold,
    }
    summary = summarize(rows, thresholds)
    save_json(summary, output_dir / "summary.json")
    write_component_csv(rows, output_dir / "components.csv")
    sweep_rows = write_threshold_sweep(rows, output_dir / "threshold_sweep.csv")

    best = sorted(
        sweep_rows,
        key=lambda item: (item["group_acceptance_rate"], item["accepted_candidates"]),
        reverse=True,
    )[:10]
    save_json(best, output_dir / "top_thresholds.json")

    print(json.dumps(summary["counts"], indent=2))
    print(json.dumps(summary["threshold_acceptance"], indent=2))
    print(f"Wrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
