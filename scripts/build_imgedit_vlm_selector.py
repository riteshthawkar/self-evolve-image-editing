#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image
from tqdm import tqdm


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def load_secret_env(path: Path) -> None:
    if os.environ.get("OPENAI_API_KEY") or not path.exists():
        return
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    if text.startswith("OPENAI_API_KEY="):
        text = text.split("=", 1)[1].strip()
    os.environ["OPENAI_API_KEY"] = text


def atomic_write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
    tmp_path.replace(path)


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = str(row.get("key", ""))
            if key:
                rows[key] = row
    return rows


def append_jsonl(row: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_candidate(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Candidates must be LABEL=DIR")
    label, directory = raw.split("=", 1)
    label = label.strip()
    if not label or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise argparse.ArgumentTypeError(f"Invalid candidate label: {label!r}")
    return label, Path(directory)


def image_to_jpeg_data_url(path: Path, *, max_side: int, quality: int) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def parse_choice(text: str, labels: set[str]) -> tuple[str | None, dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    parsed: dict[str, Any] = {}
    try:
        parsed_obj = json.loads(cleaned)
        if isinstance(parsed_obj, dict):
            parsed = parsed_obj
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                parsed_obj = json.loads(match.group(0))
                if isinstance(parsed_obj, dict):
                    parsed = parsed_obj
            except json.JSONDecodeError:
                parsed = {}
    choice = parsed.get("choice")
    if isinstance(choice, str):
        choice = choice.strip()
        if choice in labels:
            return choice, parsed
    lowered = cleaned.lower()
    hits = [label for label in labels if re.search(rf"\b{re.escape(label.lower())}\b", lowered)]
    if len(hits) == 1:
        return hits[0], parsed
    return None, parsed


def exception_payload(exc: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }
    for attr in ("status_code", "request_id", "code", "param", "type"):
        value = getattr(exc, attr, None)
        if value is not None:
            payload[attr] = value
    return payload


def selector_prompt(key: str, item: dict[str, Any], labels: list[str], *, prompt_mode: str) -> str:
    label_text = ", ".join(labels)
    if prompt_mode == "arr":
        return (
            "You are an auto-rubric verifier for instruction-guided image editing. Choose the single best edited image, "
            "but be conservative: the baseline should win unless another candidate clearly improves the requested edit "
            "without damaging unchanged content.\n"
            "Score every candidate on five 1-5 components:\n"
            "1. instruction: the requested edit is completed exactly, with no missing or extra edit.\n"
            "2. preservation: unchanged objects, identity, layout, lighting, camera, and background remain faithful.\n"
            "3. quality: realism, boundaries, texture, resolution, and artifact-free rendering.\n"
            "4. object_contract: for add/remove/replace/extract, the specified object state is correct; for other edits, "
            "score whether the target attribute/region is correct and localized.\n"
            "5. overedit_safety: high only when the candidate avoids unnecessary global changes or hallucinations.\n"
            "Use instruction and preservation as the primary criteria. A candidate that performs a stronger edit but "
            "changes unrelated content should lose. A candidate that preserves well but does not perform the edit should "
            "also lose. Prefer the baseline on ties or uncertainty.\n\n"
            f"Example key: {key}\n"
            f"Edit type: {item.get('edit_type')}\n"
            f"Instruction: {item.get('prompt')}\n"
            f"Valid candidate labels: {label_text}\n\n"
            "Return strict JSON only with this schema:\n"
            '{"choice":"one_label_from_valid_candidates","scores":{"label":[instruction,preservation,quality,object_contract,overedit_safety]},'
            '"confidence":0.0,"failure_modes":{"label":["short_failure_if_any"]},"reason":"short reason"}'
        )
    if prompt_mode == "rubric":
        return (
            "You are a strict image-editing reward model. Choose the single best edited image.\n"
            "For every candidate, first estimate these three 1-5 scores:\n"
            "1. Prompt compliance: whether the requested edit is actually completed.\n"
            "2. Preservation and locality: unchanged regions, identity, structure, and layout stay faithful to the original.\n"
            "3. Visual quality and physical realism: no artifacts, distortions, hallucinated objects, or implausible boundaries.\n"
            "A candidate with a stronger edit but large unintended changes should lose to a faithful candidate. "
            "A candidate that preserves the image but does not do the edit should also lose. Use the average of the three "
            "scores as the main decision, with preservation as the tie breaker.\n\n"
            f"Example key: {key}\n"
            f"Edit type: {item.get('edit_type')}\n"
            f"Instruction: {item.get('prompt')}\n"
            f"Valid candidate labels: {label_text}\n\n"
            "Return strict JSON only with this schema:\n"
            '{"choice":"one_label_from_valid_candidates","scores":{"label":[prompt,preservation,quality]},"confidence":0.0,'
            '"reason":"short reason"}'
        )
    return (
        "You are selecting the best image-editing output for one benchmark example.\n"
        "Given the original image, the edit instruction, and candidate edited images, choose exactly one candidate.\n"
        "Use these criteria in order: instruction satisfaction, preservation of unchanged content, object identity/layout, "
        "visual realism, and absence of editing artifacts. Prefer the original baseline only when learned variants add "
        "unwanted changes or fail the edit. Do not reward exaggerated changes that violate the instruction.\n\n"
        f"Example key: {key}\n"
        f"Edit type: {item.get('edit_type')}\n"
        f"Instruction: {item.get('prompt')}\n"
        f"Valid candidate labels: {label_text}\n\n"
        "Return strict JSON only with this schema:\n"
        '{"choice":"one_label_from_valid_candidates","confidence":0.0,"reason":"short reason"}'
    )


def call_selector(
    *,
    key: str,
    item: dict[str, Any],
    source_path: Path,
    candidate_paths: dict[str, Path],
    selector_model: str,
    max_side: int,
    jpeg_quality: int,
    timeout: float,
    max_retries: int,
    prompt_mode: str,
) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    labels = list(candidate_paths)
    content: list[dict[str, Any]] = [
        {"type": "text", "text": selector_prompt(key, item, labels, prompt_mode=prompt_mode)},
        {"type": "text", "text": "Original image:"},
        {
            "type": "image_url",
            "image_url": {"url": image_to_jpeg_data_url(source_path, max_side=max_side, quality=jpeg_quality)},
        },
    ]
    for label, path in candidate_paths.items():
        content.extend(
            [
                {"type": "text", "text": f"Candidate {label}:"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_jpeg_data_url(path, max_side=max_side, quality=jpeg_quality),
                    },
                },
            ]
        )

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("OPENAI_BASE_URL") or None,
        timeout=timeout,
        max_retries=max_retries,
    )
    response = client.chat.completions.create(
        model=selector_model,
        stream=False,
        temperature=0,
        messages=[{"role": "user", "content": content}],
    )
    text = response.choices[0].message.content or ""
    choice, parsed = parse_choice(text, set(labels))
    if choice is None:
        choice = labels[0]
    return {
        "key": key,
        "edit_type": item.get("edit_type"),
        "prompt": item.get("prompt"),
        "choice": choice,
        "parsed": parsed,
        "raw_response": text,
        "selector_model": selector_model,
        "prompt_mode": prompt_mode,
        "candidate_paths": {label: str(path) for label, path in candidate_paths.items()},
    }


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def selected_keys(edit_specs: dict[str, Any], *, limit: int | None, offset: int, stride: int) -> list[str]:
    keys = list(edit_specs)
    if stride > 1:
        keys = keys[offset::stride]
    elif offset:
        keys = keys[offset:]
    if limit is not None:
        keys = keys[:limit]
    return [str(key) for key in keys]


def build_output_folder(
    *,
    edit_specs: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
    candidates: dict[str, Path],
    output_dir: Path,
    default_label: str,
    selected: set[str] | None,
    baseline_gate_margin: float | None,
    fallback_edit_types: set[str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {label: 0 for label in candidates}
    records = []
    for key, item in edit_specs.items():
        if selected is not None and key not in selected:
            continue
        edit_type = str(item.get("edit_type") or "")
        forced_fallback = edit_type in fallback_edit_types
        decision = decisions.get(key, {})
        if forced_fallback:
            label = default_label
        else:
            label = str(decision.get("choice") or default_label)
            if label not in candidates:
                label = default_label
            if baseline_gate_margin is not None and label != default_label:
                parsed = decision.get("parsed") if isinstance(decision, dict) else {}
                scores = parsed.get("scores") if isinstance(parsed, dict) else {}
                default_score = mean_score(scores.get(default_label)) if isinstance(scores, dict) else None
                candidate_score = mean_score(scores.get(label)) if isinstance(scores, dict) else None
                if default_score is None or candidate_score is None or candidate_score <= default_score + baseline_gate_margin:
                    label = default_label
        src = candidates[label] / f"{key}.png"
        if not src.exists():
            raise FileNotFoundError(src)
        dst = output_dir / f"{key}.png"
        link_or_copy(src, dst)
        counts[label] += 1
        records.append(
            {
                "key": key,
                "edit_type": edit_type,
                "choice": label,
                "forced_fallback": forced_fallback,
                "source_path": str(src),
                "target_path": str(dst),
            }
        )
    return {"counts": counts, "records": records}


def mean_score(value: object) -> float | None:
    if not isinstance(value, list) or not value:
        return None
    try:
        return sum(float(item) for item in value) / len(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ImgEdit method by VLM-selecting among candidate folders.")
    parser.add_argument("--edit-json", default="data/processed/benchmark/imgedit/basic_edit.json")
    parser.add_argument("--origin-img-root", default="data/processed/benchmark/imgedit/original_images")
    parser.add_argument("--image-root", default="outputs/benchmark_images/imgedit")
    parser.add_argument("--scores-root", default="outputs/scores/imgedit")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--default-label", default=None)
    parser.add_argument("--selector-model", default="gpt-4o")
    parser.add_argument("--prompt-mode", choices=["selector", "rubric", "arr"], default="selector")
    parser.add_argument("--decisions-path", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--selected-edit-json-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--secret-env", default="secret.env")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--fallback-edit-type",
        action="append",
        default=[],
        help="Force the default candidate for this ImgEdit edit_type. Can be passed more than once.",
    )
    parser.add_argument(
        "--baseline-gate-margin",
        type=float,
        default=None,
        help="Fallback to default label unless selected candidate's parsed mean score exceeds default by this margin.",
    )
    args = parser.parse_args()

    load_secret_env(Path(args.secret_env))
    edit_specs = load_json(Path(args.edit_json))
    origin_img_root = Path(args.origin_img_root)
    candidates = {label: directory for label, directory in args.candidate}
    if len(candidates) != len(args.candidate):
        raise ValueError("Candidate labels must be unique")
    default_label = args.default_label or next(iter(candidates))
    if default_label not in candidates:
        raise ValueError(f"default label {default_label!r} is not a candidate")
    for label, directory in candidates.items():
        if not directory.exists():
            raise FileNotFoundError(f"Candidate {label} directory not found: {directory}")

    keys = selected_keys(edit_specs, limit=args.limit, offset=max(0, args.offset), stride=max(1, args.stride))
    selected = set(keys) if args.limit is not None or args.offset or args.stride > 1 else None
    decisions_path = Path(args.decisions_path or Path(args.scores_root) / f"{args.model_name}_selector_decisions.jsonl")
    manifest_path = Path(args.manifest_path or Path(args.scores_root) / f"{args.model_name}_selector_manifest.json")
    selected_edit_json_path = (
        Path(args.selected_edit_json_path)
        if args.selected_edit_json_path
        else Path(args.scores_root) / f"{args.model_name}_selected_edit.json"
    )
    output_dir = Path(args.image_root) / args.model_name

    existing = {} if args.force else load_jsonl(decisions_path)
    pending = []
    for key in keys:
        item = edit_specs[key]
        if key in existing and existing[key].get("choice") in candidates:
            continue
        source_path = origin_img_root / item["id"]
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        candidate_paths = {label: directory / f"{key}.png" for label, directory in candidates.items()}
        missing = [str(path) for path in candidate_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing candidate image(s) for key {key}: {missing}")
        pending.append((key, item, source_path, candidate_paths))

    print(
        f"Selector model={args.selector_model}; candidates={list(candidates)}; "
        f"prompt_mode={args.prompt_mode}; selected_keys={len(keys)}; pending={len(pending)}; output={output_dir}",
        flush=True,
    )

    decisions = dict(existing)
    if pending and not args.build_only:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    call_selector,
                    key=key,
                    item=item,
                    source_path=source_path,
                    candidate_paths=candidate_paths,
                    selector_model=args.selector_model,
                    max_side=args.max_side,
                    jpeg_quality=args.jpeg_quality,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    prompt_mode=args.prompt_mode,
                ): key
                for key, item, source_path, candidate_paths in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Selecting ImgEdit candidates"):
                key = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "key": key,
                        "choice": default_label,
                        "error": exception_payload(exc),
                        "selector_model": args.selector_model,
                        "prompt_mode": args.prompt_mode,
                    }
                row["timestamp_unix"] = time.time()
                decisions[key] = row
                append_jsonl(row, decisions_path)

    build = build_output_folder(
        edit_specs=edit_specs,
        decisions=decisions,
        candidates=candidates,
        output_dir=output_dir,
        default_label=default_label,
        selected=selected,
        baseline_gate_margin=args.baseline_gate_margin,
        fallback_edit_types=set(args.fallback_edit_type),
    )
    if selected is not None:
        atomic_write_json({key: edit_specs[key] for key in keys}, selected_edit_json_path)
    manifest = {
        "model_name": args.model_name,
        "selector_model": args.selector_model,
        "prompt_mode": args.prompt_mode,
        "edit_json": args.edit_json,
        "origin_img_root": args.origin_img_root,
        "candidates": {label: str(directory) for label, directory in candidates.items()},
        "default_label": default_label,
        "baseline_gate_margin": args.baseline_gate_margin,
        "fallback_edit_types": sorted(set(args.fallback_edit_type)),
        "decisions_path": str(decisions_path),
        "selected_key_count": len(keys),
        "selected_edit_json_path": str(selected_edit_json_path) if selected is not None else args.edit_json,
        "limit": args.limit,
        "offset": args.offset,
        "stride": args.stride,
        **build,
    }
    atomic_write_json(manifest, manifest_path)
    print(f"Built {len(build['records'])} selected image(s) in {output_dir}", flush=True)
    print(f"Choice counts: {build['counts']}", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
