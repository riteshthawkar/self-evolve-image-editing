#!/usr/bin/env python3
"""Real-image reward-discrimination study for the internal rubric CEPR reward.

This is offline validation tooling. It loads ONLY the reward-relevant modules of
Qwen-Image-Edit-2509 (the Qwen2.5-VL understanding branch + the VAE) by skipping
the ~20B MMDiT transformer, so the whole study fits on a single 24 GB GPU. It
then runs the real `InternalRubricCEPREvaluator` candidate scorer on real edit
pairs and four known-label candidate classes:

    good    - the dataset's ground-truth edited image      (should be ACCEPTED)
    noop    - an exact copy of the source                  (should be REJECTED)
    corrupt - a globally blurred + noised source           (should be REJECTED)
    wrong   - a different pair's edited image               (should be REJECTED)

For each edit type it reports mean reward, accept rate, and the AUC of the raw
reward separating `good` from each negative class. This is the empirical
evidence that the reward is calibrated and not hackable by no-op edits - the
exact failure mode the training-signal audit flagged for object removal.

No editor sampling happens here; we only score pre-existing images, so the run
is deterministic and cheap.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter

# --------------------------------------------------------------------------- #
# Structured-edit construction from a dataset record.
# --------------------------------------------------------------------------- #

_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of evaluator signal values to JSON-serializable types."""
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _clean_object(text: str) -> str:
    text = text.strip().strip(".").strip()
    text = _ARTICLE_RE.sub("", text)
    return text.strip()


def _parse_replace(instruction: str) -> tuple[str | None, str | None]:
    m = re.search(r"replace\s+(.+?)\s+with\s+(.+)", instruction, re.IGNORECASE)
    if m:
        return _clean_object(m.group(1)), _clean_object(m.group(2))
    m = re.search(r"change\s+(.+?)\s+(?:in)?to\s+(.+)", instruction, re.IGNORECASE)
    if m:
        return _clean_object(m.group(1)), _clean_object(m.group(2))
    return None, None


_TRAILING_PREP_RE = re.compile(r"\s+(?:from|in|on|at|to|into|over|behind|near|with|of)\s+.*$", re.IGNORECASE)


def _parse_single(instruction: str, verb: str) -> str | None:
    m = re.search(rf"{verb}\s+(?:the|a|an)?\s*(.+)", instruction, re.IGNORECASE)
    if m:
        return _clean_object(_TRAILING_PREP_RE.sub("", m.group(1)))
    return None


def build_structured_edit(record: dict[str, Any]) -> dict[str, Any]:
    """Derive rubric fields (required/forbidden/preserve) from the instruction.

    These mirror what the proposer emits in the live loop. When a concrete
    object can be parsed we use it; otherwise we fall back to generic, edit-type
    appropriate phrasing so the rubric still has a meaningful target.
    """
    edit_type = record["edit_type"]
    instruction = record["instruction"]
    target_caption = (record.get("target_caption") or "").strip()
    spec: dict[str, Any] = {"edit_type": edit_type, "instruction": instruction}

    if edit_type == "object_removal":
        obj = _parse_single(instruction, "remove") or "the target object"
        spec.update(
            source_object=obj,
            required_after=[f"the {obj} is no longer present"],
            forbidden_after=[f"the {obj} is still visible"],
            preserve=["the background and all other objects"],
        )
    elif edit_type == "object_addition":
        obj = _parse_single(instruction, "add") or "the requested object"
        spec.update(
            target_object=obj,
            required_after=[f"a {obj} is visible in the scene"],
            forbidden_after=[],
            preserve=["the original scene and existing objects"],
        )
    elif edit_type == "object_replacement":
        src, tgt = _parse_replace(instruction)
        src = src or "the original object"
        tgt = tgt or (target_caption or "the new object")
        spec.update(
            source_object=src,
            target_object=tgt,
            required_after=[f"a {tgt} is visible"],
            forbidden_after=[f"the {src} is still visible"],
            preserve=["the background"],
        )
    elif edit_type == "color_change":
        spec.update(
            required_after=[instruction.rstrip(".") + " has been applied"],
            forbidden_after=[],
            preserve=["the shape and position of all objects", "the background"],
        )
    elif edit_type == "background_change":
        spec.update(
            required_after=["the background has been changed as instructed"],
            forbidden_after=[],
            preserve=["the main foreground subject"],
        )
    elif edit_type == "attribute_change":
        spec.update(
            required_after=[instruction.rstrip(".") + " has been applied"],
            forbidden_after=[],
            preserve=["the overall scene layout"],
        )
    elif edit_type == "action_change":
        spec.update(
            required_after=["the subject performs the requested action"],
            forbidden_after=[],
            preserve=["the subject identity and the background"],
        )
    elif edit_type == "style_transfer":
        spec.update(
            required_after=["the image is rendered in the requested style"],
            forbidden_after=[],
            preserve=["the overall composition and subject layout"],
        )
    else:
        spec.update(
            required_after=[instruction.rstrip(".") + " has been applied"],
            forbidden_after=[],
            preserve=["unrelated regions of the image"],
        )
    return spec


def build_proposal(record: dict[str, Any]):
    from qwen_edit_project.self_evolve.types import EditProposal, ProposalDefinition

    edit_type = record["edit_type"]
    instruction = record["instruction"]
    definition = ProposalDefinition(
        operation_id=f"probe_{record['key']}",
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
        record_key=record["key"],
        round_index=0,
        proposal_index=0,
        definition=definition,
        difficulty_level=2,
        instruction=instruction,
        structured_edit=build_structured_edit(record),
    )


# --------------------------------------------------------------------------- #
# Candidate construction.
# --------------------------------------------------------------------------- #


def make_corrupt(source: Image.Image, seed: int) -> Image.Image:
    """A global, content-destroying degradation that should fail preservation."""
    import numpy as np

    blurred = source.filter(ImageFilter.GaussianBlur(radius=max(source.size) / 18.0))
    rng = np.random.default_rng(seed)
    arr = np.asarray(blurred, dtype=np.float32)
    noise = rng.normal(0.0, 35.0, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype("uint8")
    return Image.fromarray(arr)


def load_image(path: str, max_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if max_side > 0:
        w, h = image.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
    return image


# --------------------------------------------------------------------------- #
# Metrics.
# --------------------------------------------------------------------------- #


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Probability a random positive outranks a random negative (Mann-Whitney)."""
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


# --------------------------------------------------------------------------- #
# Pipeline loading (reward modules only).
# --------------------------------------------------------------------------- #


def load_feature_pipe(model_id: str, device: str, dtype_name: str):
    import torch
    from diffusers import QwenImageEditPlusPipeline

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    # transformer=None skips downloading/loading the ~20B MMDiT entirely; we only
    # need the Qwen2.5-VL understanding branch (text_encoder) and the VAE.
    pipe = QwenImageEditPlusPipeline.from_pretrained(model_id, transformer=None, torch_dtype=dtype)
    if hasattr(pipe, "to"):
        pipe.to(device)
    if getattr(pipe, "torch_dtype", None) is None:
        try:
            pipe.torch_dtype = dtype
        except Exception:
            pass
    return pipe


CANDIDATE_CLASSES = ["good", "noop", "corrupt", "wrong"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", default="data/probe/anyedit_pairs")
    parser.add_argument("--model", default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--limit-per-type", type=int, default=0, help="0 = use all pairs.")
    parser.add_argument("--out", default="outputs/analysis/reward_discrimination")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--evaluator-config",
        default=None,
        help="Optional YAML whose `evaluator` block configures the rubric evaluator. "
        "When omitted, the lean embedding-only default (detector/conservative/judge disabled) is used.",
    )
    parser.add_argument(
        "--no-fast-judge",
        dest="fast_judge",
        action="store_false",
        help="Disable the skip-infeasible judge optimization (judge every candidate, much slower).",
    )
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        metavar="evaluator.key=value",
        help="Override an evaluator-config key (dotted, relative to the `evaluator` block). "
        "Repeatable. Used by the ablation matrix to knock out one gate per run.",
    )
    parser.set_defaults(fast_judge=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from qwen_edit_project.self_evolve.backends import InternalRubricCEPREvaluator
    from qwen_edit_project.utils.config import merge_override, parse_override

    evaluator_config: dict[str, Any] = {}
    if args.evaluator_config:
        import yaml

        loaded = yaml.safe_load(Path(args.evaluator_config).read_text())
        evaluator_config = dict(loaded.get("evaluator", {}))
        evaluator_config.pop("backend", None)
        if args.fast_judge and isinstance(evaluator_config.get("internal_vlm_judge"), dict):
            # Decision-faithful speedup: skip the expensive VLM judge on candidates already
            # rejected by cheaper gates (the judge can only remove feasibility, never grant it).
            judge_cfg = dict(evaluator_config["internal_vlm_judge"])
            judge_cfg["skip_infeasible"] = True
            evaluator_config["internal_vlm_judge"] = judge_cfg
        print(
            "Using production evaluator config: "
            f"detector={evaluator_config.get('object_detector_enabled')} "
            f"conservative={evaluator_config.get('conservative_region_reward_enabled')} "
            f"judge={evaluator_config.get('internal_vlm_judge', {}).get('enabled')} "
            f"fast_judge={args.fast_judge}",
            flush=True,
        )

    # Ablation overrides: dotted keys relative to the `evaluator` block, e.g.
    #   --set conservative_region_reward_enabled=false
    #   --set internal_vlm_judge.enabled=false
    # This lets the ablation matrix knock out one gate per run without cloning configs.
    for raw in args.set_overrides:
        key, value = parse_override(raw)
        evaluator_config = merge_override(evaluator_config, key, value)
    if args.set_overrides:
        print(f"Applied {len(args.set_overrides)} ablation override(s): {args.set_overrides}", flush=True)

    probe_dir = Path(args.probe_dir)
    manifest_path = probe_dir / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"ERROR: manifest not found at {manifest_path}", flush=True)
        return 2

    records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if args.limit_per_type > 0:
        by_type: dict[str, int] = defaultdict(int)
        kept = []
        for r in records:
            if by_type[r["edit_type"]] < args.limit_per_type:
                kept.append(r)
                by_type[r["edit_type"]] += 1
        records = kept
    print(f"Loaded {len(records)} probe pairs from {manifest_path}", flush=True)

    print(f"Loading reward modules from {args.model} (transformer skipped)...", flush=True)
    pipe = load_feature_pipe(args.model, args.device, args.dtype)
    print("Reward pipe ready.", flush=True)

    evaluator = InternalRubricCEPREvaluator(evaluator_config)
    # Route the evaluator's internal feature pipe to our reward-only pipeline so we
    # can run the real public scoring path (score_group -> per-candidate gates +
    # group VLM judge) without constructing a full QwenEditEditor.
    evaluator._get_internal_pipe = lambda _editor=None: pipe  # type: ignore[assignment]


    # Per (edit_type, candidate_class) accumulation.
    rewards: dict[tuple[str, str], list[float]] = defaultdict(list)
    accepts: dict[tuple[str, str], list[float]] = defaultdict(list)
    rubric_required: dict[tuple[str, str], list[float]] = defaultdict(list)
    rubric_forbidden: dict[tuple[str, str], list[float]] = defaultdict(list)
    rubric_preserve: dict[tuple[str, str], list[float]] = defaultdict(list)
    reject_reasons: dict[tuple[str, str], list[str]] = defaultdict(list)
    per_pair: list[dict[str, Any]] = []

    loaded = [(r, load_image(r["source_path"], args.max_side), load_image(r["edited_path"], args.max_side)) for r in records]

    for idx, (record, source, edited) in enumerate(loaded):
        wrong_src = loaded[(idx + 1) % len(loaded)][2].resize(source.size, Image.Resampling.LANCZOS)
        ordered = [
            ("good", edited),
            ("noop", source.copy()),
            ("corrupt", make_corrupt(source, args.seed + idx)),
            ("wrong", wrong_src),
        ]
        proposal = build_proposal(record)
        edit_type = record["edit_type"]
        pair_row: dict[str, Any] = {"key": record["key"], "edit_type": edit_type, "instruction": record["instruction"]}
        try:
            results = evaluator.score_group(proposal, source, [img for _, img in ordered], editor=None)
        except Exception as exc:
            print(f"  scoring error {record['key']}: {exc}", flush=True)
            continue
        for ci, (cls, _img) in enumerate(ordered):
            # Map back by candidate_index recorded in the signals.
            row = next(
                (r for r in results if int(r.signals.get("candidate_index", -1)) == ci),
                results[ci] if ci < len(results) else None,
            )
            if row is None:
                continue
            comp = row.component_scores
            feasible = float(row.signals.get("feasible", 1.0 if row.accepted else 0.0))
            raw_reward = float(comp.get("rubric_cepr_raw_reward", comp.get("cepr_raw_reward", row.total_score)))
            key = (edit_type, cls)
            rewards[key].append(raw_reward)
            accepts[key].append(feasible)
            rubric_required[key].append(float(comp.get("rubric_required_after", 0.0)))
            rubric_forbidden[key].append(float(comp.get("rubric_forbidden_after_absent", 0.0)))
            rubric_preserve[key].append(float(comp.get("rubric_preservation", 0.0)))
            reject_reasons[key].append(str(row.signals.get("rubric_reject_reason", "")))
            pair_row[cls] = {
                "raw_reward": raw_reward,
                "feasible": bool(feasible >= 0.5),
                "total_score": float(row.total_score),
                "reject_reason": str(row.signals.get("rubric_reject_reason", "")),
                "rubric_required_after": float(comp.get("rubric_required_after", 0.0)),
                "rubric_forbidden_after_absent": float(comp.get("rubric_forbidden_after_absent", 0.0)),
                "rubric_preservation": float(comp.get("rubric_preservation", 0.0)),
                "object_detector_contract": float(comp.get("object_detector_contract", -1.0)),
                "conservative_region_reward": float(comp.get("conservative_region_reward", -1.0)),
                "internal_vlm_judge_score": float(row.signals.get("internal_vlm_judge_score", -1.0)),
                # Full dumps so offline gate calibration can inspect every sub-gate
                # (e.g. conservative_region outside-preservation vs target-change)
                # without needing another GPU run.
                "component_scores": {k: _jsonable(v) for k, v in comp.items()},
                "signals": {k: _jsonable(v) for k, v in row.signals.items()},
            }
        per_pair.append(pair_row)
        if (idx + 1) % 5 == 0 or idx + 1 == len(loaded):
            print(f"  scored {idx + 1}/{len(loaded)} pairs", flush=True)

    edit_types = sorted({r["edit_type"] for r in records})

    def summarize(scope_types: list[str]) -> dict[str, Any]:
        block: dict[str, Any] = {}
        for cls in CANDIDATE_CLASSES:
            r = [v for t in scope_types for v in rewards[(t, cls)]]
            a = [v for t in scope_types for v in accepts[(t, cls)]]
            block[cls] = {
                "n": len(r),
                "mean_reward": mean(r),
                "accept_rate": mean(a),
                "mean_rubric_required_after": mean([v for t in scope_types for v in rubric_required[(t, cls)]]),
                "mean_rubric_forbidden_after_absent": mean([v for t in scope_types for v in rubric_forbidden[(t, cls)]]),
                "mean_rubric_preservation": mean([v for t in scope_types for v in rubric_preserve[(t, cls)]]),
            }
        good_r = [v for t in scope_types for v in rewards[(t, "good")]]
        block["auc_good_vs"] = {
            neg: auc(good_r, [v for t in scope_types for v in rewards[(t, neg)]])
            for neg in ["noop", "corrupt", "wrong"]
        }
        return block

    report: dict[str, Any] = {
        "model": args.model,
        "probe_dir": str(probe_dir),
        "num_pairs": len(records),
        "edit_types": edit_types,
        "candidate_classes": CANDIDATE_CLASSES,
        "overall": summarize(edit_types),
        "by_edit_type": {t: summarize([t]) for t in edit_types},
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "per_pair.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=True) for p in per_pair) + "\n", encoding="utf-8"
    )
    _write_markdown(out_dir / "report.md", report)
    print(f"\nWrote {out_dir/'report.json'} and {out_dir/'report.md'}", flush=True)
    print(json.dumps(report["overall"], indent=2), flush=True)
    return 0


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Internal Rubric CEPR — Reward Discrimination Study\n")
    lines.append(f"- Model: `{report['model']}` (reward modules only; transformer skipped)")
    lines.append(f"- Probe pairs: {report['num_pairs']}")
    lines.append(f"- Edit types: {', '.join(report['edit_types'])}\n")
    lines.append("Candidate classes: **good** (dataset edit, should ACCEPT), "
                 "**noop** (source copy, should REJECT), **corrupt** (global blur+noise, should REJECT), "
                 "**wrong** (different pair's edit, should REJECT).\n")

    def table(block: dict[str, Any]) -> list[str]:
        rows = ["| class | n | mean reward | accept rate | req_after | forbidden_absent | preservation |",
                "|---|---|---|---|---|---|---|"]
        for cls in report["candidate_classes"]:
            b = block[cls]
            rows.append(
                f"| {cls} | {b['n']} | {_fmt(b['mean_reward'])} | {_fmt(b['accept_rate'])} | "
                f"{_fmt(b['mean_rubric_required_after'])} | {_fmt(b['mean_rubric_forbidden_after_absent'])} | "
                f"{_fmt(b['mean_rubric_preservation'])} |"
            )
        auc_b = block["auc_good_vs"]
        rows.append("")
        rows.append(f"AUC good vs: noop={_fmt(auc_b['noop'])}, corrupt={_fmt(auc_b['corrupt'])}, wrong={_fmt(auc_b['wrong'])}")
        return rows

    lines.append("## Overall\n")
    lines.extend(table(report["overall"]))
    lines.append("\n## Per edit type\n")
    for t in report["edit_types"]:
        lines.append(f"### {t}\n")
        lines.extend(table(report["by_edit_type"][t]))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
