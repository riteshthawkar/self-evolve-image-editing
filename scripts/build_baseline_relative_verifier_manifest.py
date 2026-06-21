#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import random
import re
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image
from tqdm import tqdm


COMPONENTS = ["instruction", "preservation", "quality", "object_contract", "overedit_safety"]
DEFAULT_WEIGHTS = {
    "instruction": 0.35,
    "preservation": 0.25,
    "quality": 0.20,
    "object_contract": 0.15,
    "overedit_safety": 0.05,
}


def load_secret_env(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    if text.startswith("OPENAI_API_KEY="):
        text = text.split("=", 1)[1].strip()
    os.environ["OPENAI_API_KEY"] = text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"Expected JSON object rows in {path}")
                rows.append(row)
    return rows


def append_jsonl(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def atomic_write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    tmp.replace(path)


def image_to_jpeg_data_url(path: Path, *, max_side: int, quality: int) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def strip_json_markdown(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def parse_json_response(text: str) -> dict[str, Any]:
    cleaned = strip_json_markdown(text)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def normalize_scores(value: Any) -> dict[str, float]:
    scores: dict[str, float] = {}
    if isinstance(value, list):
        for name, raw in zip(COMPONENTS, value):
            score = finite_float(raw)
            if math.isfinite(score):
                scores[name] = max(1.0, min(5.0, score))
    elif isinstance(value, dict):
        aliases = {
            "prompt": "instruction",
            "prompt_compliance": "instruction",
            "instruction_following": "instruction",
            "visual_quality": "quality",
            "object": "object_contract",
            "object_grounding": "object_contract",
            "safety": "overedit_safety",
            "locality": "overedit_safety",
        }
        for raw_name, raw in value.items():
            name = aliases.get(str(raw_name), str(raw_name))
            if name not in COMPONENTS and name != "overall":
                continue
            score = finite_float(raw)
            if math.isfinite(score):
                scores[name] = max(1.0, min(5.0, score))
    return scores


def weighted_score(scores: dict[str, float]) -> float:
    if "overall" in scores:
        return scores["overall"]
    total = 0.0
    weight_sum = 0.0
    for name, weight in DEFAULT_WEIGHTS.items():
        if name in scores:
            total += scores[name] * weight
            weight_sum += weight
    return total / weight_sum if weight_sum else math.nan


def get_nested(row: dict[str, Any], dotted: str) -> Any:
    cur: Any = row
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def row_family(row: dict[str, Any]) -> str:
    for key in ("family", "edit_type", "structured_edit.edit_type"):
        value = get_nested(row, key)
        if value:
            return str(value)
    return "unknown"


def existing_path(raw: Any) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw))
    return path if path.exists() else None


def decision_id(source: Path, prompt: str, candidate: Path, reference: Path, *, verifier_id: str) -> str:
    payload = "\n".join([verifier_id, str(source), prompt, str(candidate), str(reference)])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def verifier_prompt(prompt: str, family: str) -> str:
    return (
        "You are a strict auto-rubric verifier for instruction-guided image editing. "
        "Compare the reference output and candidate output against the original image and instruction.\n"
        "Score each output on five 1-5 components:\n"
        "1. instruction: the requested edit is completed exactly.\n"
        "2. preservation: unchanged objects, identity, layout, lighting, camera, and background are preserved.\n"
        "3. quality: realism, texture, boundaries, resolution, and artifact-free rendering.\n"
        "4. object_contract: for add/remove/replace/extract, the target object state is correct; otherwise the target "
        "attribute or region is changed correctly and locally.\n"
        "5. overedit_safety: high only if there are no unnecessary global changes, hallucinations, or unrelated edits.\n"
        "The candidate should beat the reference only when it clearly improves instruction satisfaction without reducing "
        "preservation or quality. Prefer the reference on ties or uncertainty.\n\n"
        f"Edit family: {family}\n"
        f"Instruction: {prompt}\n\n"
        "Return strict JSON only:\n"
        '{"choice":"candidate|reference|tie","scores":{"candidate":[instruction,preservation,quality,object_contract,overedit_safety],'
        '"reference":[instruction,preservation,quality,object_contract,overedit_safety]},'
        '"failure_modes":{"candidate":["short_failure_if_any"],"reference":["short_failure_if_any"]},'
        '"confidence":0.0,"reason":"short reason"}'
    )


def call_verifier(
    *,
    source: Path,
    reference: Path,
    candidate: Path,
    prompt: str,
    family: str,
    model: str,
    max_side: int,
    jpeg_quality: int,
    timeout: float,
    max_retries: int,
    json_response_format: bool,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        timeout=timeout,
        max_retries=max_retries,
    )
    content: list[dict[str, Any]] = [
        {"type": "text", "text": verifier_prompt(prompt, family)},
        {"type": "text", "text": "Original image:"},
        {"type": "image_url", "image_url": {"url": image_to_jpeg_data_url(source, max_side=max_side, quality=jpeg_quality)}},
        {"type": "text", "text": "Reference output:"},
        {
            "type": "image_url",
            "image_url": {"url": image_to_jpeg_data_url(reference, max_side=max_side, quality=jpeg_quality)},
        },
        {"type": "text", "text": "Candidate output:"},
        {
            "type": "image_url",
            "image_url": {"url": image_to_jpeg_data_url(candidate, max_side=max_side, quality=jpeg_quality)},
        },
    ]
    request: dict[str, Any] = {
        "model": model,
        "stream": False,
        "temperature": 0,
        "messages": [{"role": "user", "content": content}],
    }
    if json_response_format:
        request["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**request)
    text = response.choices[0].message.content or ""
    parsed = parse_json_response(text)
    return {"raw_response": text, "parsed": parsed}


def exception_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def load_decision_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("decision_id") or "")
            if key:
                cache[key] = row
    return cache


def decision_is_complete(decision: dict[str, Any] | None) -> bool:
    if not isinstance(decision, dict) or decision.get("error"):
        return False
    parsed = decision.get("parsed")
    return isinstance(parsed, dict) and bool(parsed)


def candidate_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Counter[str]]:
    rng = random.Random(args.seed)
    include_families = {item.strip() for item in args.include_families.split(",") if item.strip()}
    rows: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for path in args.input_manifest:
        for row in load_jsonl(path):
            family = row_family(row)
            if include_families and family not in include_families:
                rejected["family_filtered"] += 1
                continue
            winner_score = finite_float(row.get("winner_score"), finite_float(row.get("score"), 1.0))
            if winner_score < args.min_internal_winner_score:
                rejected["internal_winner_score_low"] += 1
                continue
            internal_margin = finite_float(row.get("score_margin"), 1.0)
            if internal_margin < args.min_internal_margin:
                rejected["internal_margin_low"] += 1
                continue
            source = existing_path(row.get(args.source_image_key))
            candidate = existing_path(row.get(args.candidate_image_key))
            reference = existing_path(row.get(args.reference_image_key))
            reference_source = args.reference_image_key
            if reference is None and args.allow_rejected_as_reference:
                reference = existing_path(row.get(args.rejected_image_key))
                reference_source = args.rejected_image_key
            prompt = str(row.get(args.prompt_key) or "").strip()
            if source is None:
                rejected["missing_source_image"] += 1
                continue
            if candidate is None:
                rejected["missing_candidate_image"] += 1
                continue
            if reference is None:
                rejected["missing_reference_image"] += 1
                continue
            if not prompt:
                rejected["missing_prompt"] += 1
                continue
            record = dict(row)
            record["_family"] = family
            record["_source_path"] = str(source)
            record["_candidate_path"] = str(candidate)
            record["_reference_path"] = str(reference)
            record["_reference_source"] = reference_source
            verifier_id = f"{args.prompt_version}:{args.verifier_model}"
            record["_decision_id"] = decision_id(source, prompt, candidate, reference, verifier_id=verifier_id)
            rows.append(record)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["_family"])].append(row)
    selected: list[dict[str, Any]] = []
    for family, bucket in sorted(buckets.items()):
        bucket.sort(
            key=lambda item: (
                finite_float(item.get("winner_score"), finite_float(item.get("score"), 0.0)),
                finite_float(item.get("score_margin"), 0.0),
            ),
            reverse=True,
        )
        selected.extend(bucket[: args.per_family_limit])
    rng.shuffle(selected)
    if args.max_rows >= 0:
        selected = selected[: args.max_rows]
    return selected, rejected


def build_training_row(args: argparse.Namespace, source_row: dict[str, Any], decision: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    parsed = decision.get("parsed") if isinstance(decision, dict) else {}
    if not isinstance(parsed, dict):
        return None, "missing_parsed_response"
    scores = parsed.get("scores") if isinstance(parsed.get("scores"), dict) else {}
    candidate_scores = normalize_scores(scores.get("candidate"))
    reference_scores = normalize_scores(scores.get("reference"))
    candidate_overall = weighted_score(candidate_scores)
    reference_overall = weighted_score(reference_scores)
    if not math.isfinite(candidate_overall) or not math.isfinite(reference_overall):
        return None, "missing_verifier_scores"
    delta = candidate_overall - reference_overall
    choice = str(parsed.get("choice") or "").strip().lower()

    candidate_pass = (
        delta >= args.candidate_margin
        and candidate_overall >= args.min_candidate_overall
        and candidate_scores.get("instruction", 0.0) >= args.min_candidate_instruction
        and candidate_scores.get("preservation", 0.0) >= args.min_candidate_preservation
        and candidate_scores.get("quality", 0.0) >= args.min_candidate_quality
        and candidate_scores.get("object_contract", 0.0) >= args.min_candidate_object_contract
        and (choice == "candidate" or not args.require_verifier_choice)
    )
    baseline_pass = (
        args.include_reference_wins
        and -delta >= args.reference_margin
        and reference_overall >= args.min_reference_overall
        and (choice == "reference" or not args.require_verifier_choice)
    )

    if candidate_pass:
        chosen = source_row["_candidate_path"]
        rejected = source_row["_reference_path"]
        source = "baseline_relative_candidate_win"
        weight = min(args.max_sample_weight, args.candidate_weight * (1.0 + args.margin_weight_scale * min(delta, args.margin_clip)))
    elif baseline_pass:
        chosen = source_row["_reference_path"]
        rejected = source_row["_candidate_path"]
        source = "baseline_relative_reference_win"
        weight = min(
            args.max_sample_weight,
            args.reference_win_weight * (1.0 + args.margin_weight_scale * min(-delta, args.margin_clip)),
        )
    else:
        return None, "verifier_margin_or_threshold_failed"

    return (
        {
            "prompt": str(source_row.get(args.prompt_key)).strip(),
            "chosen_image": chosen,
            "rejected_image": rejected,
            "edit_image": source_row["_source_path"],
            "sample_weight": weight,
            "family": source_row["_family"],
            "prompt_variant": source_row.get("prompt_variant", "baseline_relative"),
            "preference_source": source,
            "record_key": source_row.get("record_key"),
            "group_id": source_row.get("group_id"),
            "operation_id": source_row.get("operation_id"),
            "structured_edit": source_row.get("structured_edit"),
            "reference_image": source_row["_reference_path"],
            "reference_source": source_row["_reference_source"],
            "candidate_image": source_row["_candidate_path"],
            "verifier_model": args.verifier_model,
            "verifier_choice": choice,
            "verifier_delta": delta,
            "candidate_overall": candidate_overall,
            "reference_overall": reference_overall,
            "candidate_scores": candidate_scores,
            "reference_scores": reference_scores,
            "verifier_failure_modes": parsed.get("failure_modes"),
            "verifier_reason": parsed.get("reason"),
            "decision_id": source_row["_decision_id"],
        },
        "accepted",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a baseline-relative preference manifest using a VLM edit verifier."
    )
    parser.add_argument("--input-manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--decision-cache", type=Path, default=None)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--source-image-key", default="edit_image")
    parser.add_argument("--candidate-image-key", default="chosen_image")
    parser.add_argument("--reference-image-key", default="baseline_image")
    parser.add_argument("--rejected-image-key", default="rejected_image")
    parser.add_argument("--allow-rejected-as-reference", action="store_true")
    parser.add_argument(
        "--include-families",
        default="object_removal,object_replacement,object_addition,background,color,color_change,background_change,global_adjustment,style_transfer",
    )
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--per-family-limit", type=int, default=64)
    parser.add_argument("--min-internal-winner-score", type=float, default=0.0)
    parser.add_argument("--min-internal-margin", type=float, default=0.0)
    parser.add_argument("--candidate-margin", type=float, default=0.35)
    parser.add_argument("--reference-margin", type=float, default=0.50)
    parser.add_argument("--min-candidate-overall", type=float, default=3.8)
    parser.add_argument("--min-candidate-instruction", type=float, default=4.0)
    parser.add_argument("--min-candidate-preservation", type=float, default=3.8)
    parser.add_argument("--min-candidate-quality", type=float, default=3.8)
    parser.add_argument("--min-candidate-object-contract", type=float, default=3.7)
    parser.add_argument("--min-reference-overall", type=float, default=3.8)
    parser.add_argument("--include-reference-wins", action="store_true")
    parser.add_argument(
        "--max-reference-win-ratio",
        type=float,
        default=0.5,
        help="Cap reference-win rows relative to candidate-win rows. Use a negative value to disable.",
    )
    parser.add_argument(
        "--min-output-rows",
        type=int,
        default=1,
        help="Fail after writing the summary if fewer than this many training rows survive verification.",
    )
    parser.set_defaults(require_verifier_choice=True)
    parser.add_argument("--require-verifier-choice", dest="require_verifier_choice", action="store_true")
    parser.add_argument("--no-require-verifier-choice", dest="require_verifier_choice", action="store_false")
    parser.add_argument("--candidate-weight", type=float, default=1.0)
    parser.add_argument("--reference-win-weight", type=float, default=0.35)
    parser.add_argument("--margin-weight-scale", type=float, default=0.35)
    parser.add_argument("--margin-clip", type=float, default=1.0)
    parser.add_argument("--max-sample-weight", type=float, default=1.5)
    parser.add_argument("--verifier-model", default="gpt-4o")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.set_defaults(json_response_format=True)
    parser.add_argument("--json-response-format", dest="json_response_format", action="store_true")
    parser.add_argument("--no-json-response-format", dest="json_response_format", action="store_false")
    parser.add_argument("--secret-env", type=Path, default=Path("secret.env"))
    parser.add_argument("--prompt-version", default="brv_arr_v1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    load_secret_env(args.secret_env)
    rows, input_rejected = candidate_records(args)
    cache_path = args.decision_cache or args.output.with_suffix(".verifier_decisions.jsonl")
    cache = {} if args.force else load_decision_cache(cache_path)
    pending = [row for row in rows if not decision_is_complete(cache.get(row["_decision_id"]))]
    print(f"Verifier rows={len(rows)} pending={len(pending)} cache={cache_path}", flush=True)

    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    call_verifier,
                    source=Path(row["_source_path"]),
                    reference=Path(row["_reference_path"]),
                    candidate=Path(row["_candidate_path"]),
                    prompt=str(row.get(args.prompt_key)).strip(),
                    family=str(row["_family"]),
                    model=args.verifier_model,
                    max_side=args.max_side,
                    jpeg_quality=args.jpeg_quality,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    json_response_format=args.json_response_format,
                ): row
                for row in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Verifying candidate/reference pairs"):
                row = futures[future]
                try:
                    decision = future.result()
                except Exception as exc:
                    decision = {"error": exception_payload(exc), "parsed": {}}
                decision.update(
                    {
                        "decision_id": row["_decision_id"],
                        "prompt": row.get(args.prompt_key),
                        "family": row["_family"],
                        "source_image": row["_source_path"],
                        "reference_image": row["_reference_path"],
                        "reference_source": row["_reference_source"],
                        "candidate_image": row["_candidate_path"],
                        "timestamp_unix": time.time(),
                    }
                )
                cache[row["_decision_id"]] = decision
                append_jsonl(decision, cache_path)

    output_rows: list[dict[str, Any]] = []
    accepted_reasons: Counter[str] = Counter()
    for row in rows:
        training_row, reason = build_training_row(args, row, cache.get(row["_decision_id"], {}))
        accepted_reasons[reason] += 1
        if training_row is not None:
            output_rows.append(training_row)

    reference_rows = [row for row in output_rows if row.get("preference_source") == "baseline_relative_reference_win"]
    candidate_rows = [row for row in output_rows if row.get("preference_source") == "baseline_relative_candidate_win"]
    capped_reference_rows = 0
    if args.max_reference_win_ratio >= 0:
        max_reference_rows = int(len(candidate_rows) * args.max_reference_win_ratio)
        if len(reference_rows) > max_reference_rows:
            reference_rows.sort(key=lambda item: -float(item.get("verifier_delta", 0.0)), reverse=True)
            capped_reference_rows = len(reference_rows) - max_reference_rows
            reference_rows = reference_rows[:max_reference_rows]
            output_rows = candidate_rows + reference_rows

    random.Random(args.seed).shuffle(output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "output": str(args.output),
        "decision_cache": str(cache_path),
        "input_rows_selected": len(rows),
        "output_rows": len(output_rows),
        "input_rejected": input_rejected,
        "training_row_reasons": accepted_reasons,
        "candidate_win_rows": len(candidate_rows),
        "reference_win_rows_after_cap": len(reference_rows),
        "reference_win_rows_capped": capped_reference_rows,
        "per_family": Counter(str(row.get("family", "")) for row in output_rows),
        "per_preference_source": Counter(str(row.get("preference_source", "")) for row in output_rows),
        "thresholds": {
            "candidate_margin": args.candidate_margin,
            "reference_margin": args.reference_margin,
            "min_candidate_overall": args.min_candidate_overall,
            "min_candidate_instruction": args.min_candidate_instruction,
            "min_candidate_preservation": args.min_candidate_preservation,
            "min_candidate_quality": args.min_candidate_quality,
            "min_candidate_object_contract": args.min_candidate_object_contract,
        },
        "verifier_model": args.verifier_model,
        "prompt_version": args.prompt_version,
        "seed": args.seed,
    }
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    atomic_write_json(summary, summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=True), flush=True)
    if len(output_rows) < args.min_output_rows:
        raise SystemExit(
            f"Only {len(output_rows)} verified training rows survived; "
            f"required at least {args.min_output_rows}. Summary: {summary_path}"
        )


if __name__ == "__main__":
    main()
