#!/usr/bin/env python
"""Offline gate-calibration analysis for the internal rubric-CEPR reward.

Pure post-hoc analysis of an existing reward-discrimination run
(``per_pair.jsonl``). No GPU, no model. The goal is to answer two questions
that the experiment machine cannot help with while it is unavailable:

  1. Which gate is the binding constraint that rejects genuine ("good") edits,
     per edit type? (recall bottleneck)
  2. How much headroom is there to relax each gate's threshold to recover good
     edits *without* letting any negative (noop/corrupt/wrong) become feasible?
     (the 0-false-accept guarantee must be preserved)

The reward is multi-gate: a candidate is feasible only if it passes ALL gates.
We exploit this: a negative that fails >= 2 stored gates stays rejected even if
we relax one of them. We therefore report, per negative candidate, how many
stored gates it fails ("gate redundancy"). The minimum redundancy across all
negatives is the safety margin for single-gate relaxation.

Usage:
  python scripts/calibrate_reward_gates.py \
    --per-pair outputs/analysis/reward_discrimination_full/per_pair.jsonl \
    --out outputs/analysis/reward_gate_calibration
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

NEG_CLASSES = ["noop", "corrupt", "wrong"]
ALL_CLASSES = ["good", *NEG_CLASSES]

# Stored gating components -> (config threshold, "higher is better" comparison).
# These mirror configs/self_evolve/qwen_edit_2509_balanced_cepr_v2.yaml.
GATE_THRESHOLDS = {
    "rubric_required_after": 0.45,        # rubric_required_threshold
    "rubric_forbidden_after_absent": 0.42,  # rubric_forbidden_threshold (soft-forbidden types)
    "rubric_preservation": 0.30,          # rubric_preservation_threshold
    "conservative_region_reward": 0.30,   # conservative_region_min_reward
    "object_detector_contract": 0.45,     # object_detector_score_threshold
    "internal_vlm_judge_score": 0.35,     # internal_vlm_judge.min_score_for_feasible
}
# Components that only act as a gate for a subset of edit types.
SOFT_FORBIDDEN_TYPES = {"object_removal", "object_replacement"}
DETECTOR_TYPES = {"object_removal", "object_replacement"}


def _passes(component: str, value: float, edit_type: str) -> bool:
    """Whether a stored component clears its gate. Returns True when the gate is
    not applicable to this edit type, or when the component was not computed
    (sentinel -1.0 for the judge means "not run", i.e. not a binding failure)."""
    thr = GATE_THRESHOLDS[component]
    if component == "rubric_forbidden_after_absent" and edit_type not in SOFT_FORBIDDEN_TYPES:
        return True
    if component == "object_detector_contract" and edit_type not in DETECTOR_TYPES:
        return True
    if component == "internal_vlm_judge_score" and value == -1.0:
        # Judge was not run (candidate already rejected by a cheaper gate).
        return True
    if value is None:
        return True
    return float(value) >= thr


def _gates_failed(cand: dict[str, Any], edit_type: str) -> list[str]:
    failed = []
    for component in GATE_THRESHOLDS:
        if component not in cand:
            continue
        if not _passes(component, cand.get(component), edit_type):
            failed.append(component)
    return failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--per-pair",
        default="outputs/analysis/reward_discrimination_full/per_pair.jsonl",
    )
    ap.add_argument("--out", default="outputs/analysis/reward_gate_calibration")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.per_pair).read_text().splitlines() if l.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- (A) good-edit binding reject reasons, overall + per type ---------------
    good_reason = Counter()
    good_reason_by_type: dict[str, Counter] = defaultdict(Counter)
    good_total_by_type: Counter = Counter()
    for r in rows:
        et = r["edit_type"]
        g = r["good"]
        good_total_by_type[et] += 1
        reason = "ACCEPTED" if g.get("feasible") else (g.get("reject_reason") or "unknown")
        good_reason[reason] += 1
        good_reason_by_type[et][reason] += 1

    # --- (B) negative gate redundancy (the safety margin) -----------------------
    neg_redundancy = Counter()              # n_gates_failed -> count
    neg_any_accepted = 0
    neg_fail_one_gate_examples = []
    for r in rows:
        et = r["edit_type"]
        for cls in NEG_CLASSES:
            cand = r[cls]
            if cand.get("feasible"):
                neg_any_accepted += 1
            failed = _gates_failed(cand, et)
            neg_redundancy[len(failed)] += 1
            if len(failed) <= 1:
                neg_fail_one_gate_examples.append(
                    {"key": r["key"], "class": cls, "edit_type": et, "failed_gates": failed}
                )

    # --- (C) per-component relaxation headroom ----------------------------------
    # For each stored gating component, for the edit types where it is active,
    # sweep the threshold downward and report how many GOOD edits would clear the
    # component at each value, vs the highest negative value (the point below
    # which negatives start clearing the component too).
    sweep: dict[str, Any] = {}
    for component, thr in GATE_THRESHOLDS.items():
        good_vals, neg_vals = [], []
        for r in rows:
            et = r["edit_type"]
            # Only consider edit types where this component is an active gate.
            if component == "rubric_forbidden_after_absent" and et not in SOFT_FORBIDDEN_TYPES:
                continue
            if component == "object_detector_contract" and et not in DETECTOR_TYPES:
                continue
            gv = r["good"].get(component)
            if gv is not None and not (component == "internal_vlm_judge_score" and gv == -1.0):
                good_vals.append(float(gv))
            for cls in NEG_CLASSES:
                nv = r[cls].get(component)
                if nv is not None and not (component == "internal_vlm_judge_score" and nv == -1.0):
                    neg_vals.append(float(nv))
        if not good_vals:
            continue
        good_vals.sort()
        neg_vals.sort()
        max_neg = max(neg_vals) if neg_vals else None
        # "safe floor": the lowest threshold we could drop to while still being
        # strictly above every negative value for this component.
        safe_floor = (max_neg + 1e-6) if max_neg is not None else 0.0
        good_pass_now = sum(1 for v in good_vals if v >= thr)
        good_pass_at_safe = sum(1 for v in good_vals if v >= safe_floor)
        sweep[component] = {
            "current_threshold": thr,
            "n_good": len(good_vals),
            "n_neg": len(neg_vals),
            "good_pass_current": good_pass_now,
            "good_pass_current_rate": round(good_pass_now / len(good_vals), 3),
            "max_negative_value": round(max_neg, 4) if max_neg is not None else None,
            "safe_floor_threshold": round(safe_floor, 4),
            "good_pass_at_safe_floor": good_pass_at_safe,
            "good_pass_at_safe_floor_rate": round(good_pass_at_safe / len(good_vals), 3),
            "recoverable_good": good_pass_at_safe - good_pass_now,
        }

    report = {
        "source": str(args.per_pair),
        "n_pairs": len(rows),
        "negatives_accepted": neg_any_accepted,
        "good_reject_reasons_overall": dict(good_reason.most_common()),
        "negative_gate_redundancy": {str(k): v for k, v in sorted(neg_redundancy.items())},
        "negatives_failing_le_1_gate": neg_fail_one_gate_examples,
        "component_relaxation_headroom": sweep,
    }
    (out_dir / "calibration.json").write_text(json.dumps(report, indent=2))

    # --- markdown summary -------------------------------------------------------
    lines = ["# Reward Gate Calibration (offline)", ""]
    lines.append(f"- Source: `{args.per_pair}`")
    lines.append(f"- Pairs: {len(rows)} (x4 candidate classes)")
    lines.append(f"- Negatives accepted (must be 0): **{neg_any_accepted}**")
    lines.append("")
    lines.append("## Good-edit reject reasons (recall bottleneck)")
    lines.append("")
    lines.append("| reason | count |")
    lines.append("|---|---|")
    for k, v in good_reason.most_common():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Negative gate redundancy (safety margin)")
    lines.append("")
    lines.append("How many stored gates each negative candidate fails. If the minimum is >= 2, "
                 "relaxing any single gate cannot make a negative feasible.")
    lines.append("")
    lines.append("| gates failed | # negative candidates |")
    lines.append("|---|---|")
    for k in sorted(neg_redundancy):
        lines.append(f"| {k} | {neg_redundancy[k]} |")
    min_red = min(neg_redundancy) if neg_redundancy else 0
    lines.append("")
    lines.append(f"**Minimum negative gate-failures = {min_red}** "
                 f"({'single-gate relaxation is SAFE' if min_red >= 2 else 'caution: some negatives ride on one gate'}).")
    lines.append("")
    lines.append("## Per-component relaxation headroom")
    lines.append("")
    lines.append("`safe_floor` = the lowest threshold that stays strictly above every negative "
                 "value for that component. `recoverable_good` = extra good edits that clear the "
                 "component if relaxed to the safe floor.")
    lines.append("")
    lines.append("| component | cur thr | good pass now | max neg | safe floor | good pass @ floor | recoverable |")
    lines.append("|---|---|---|---|---|---|---|")
    for comp, s in sweep.items():
        lines.append(
            f"| {comp} | {s['current_threshold']} | "
            f"{s['good_pass_current']} ({s['good_pass_current_rate']}) | "
            f"{s['max_negative_value']} | {s['safe_floor_threshold']} | "
            f"{s['good_pass_at_safe_floor']} ({s['good_pass_at_safe_floor_rate']}) | "
            f"{s['recoverable_good']} |"
        )
    (out_dir / "calibration.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote {out_dir/'calibration.md'} and calibration.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
