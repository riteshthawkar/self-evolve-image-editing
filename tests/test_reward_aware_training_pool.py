from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.build_reward_aware_training_pool import build_pool, edit_opportunity_scores, source_profile


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_old_vlm_scores_keep_object_removal_opportunity() -> None:
    row = {
        "key": "source__object_case",
        "image": "data/example.jpg",
        "caption": "a clear object on a plain background",
        "score": 0.79,
        "edit_families": ["object", "background"],
        "primary_family": "object",
        "vlm": {
            "quality_score": 0.80,
            "natural_image_score": 0.82,
            "editable_content_score": 0.78,
            "object_region_clarity": 0.76,
            "preservation_potential": 0.80,
            "clutter_penalty": 0.18,
            "text_watermark_penalty": 0.05,
        },
        "stats_scores": {
            "technical_quality_score": 0.78,
            "structure_score": 0.70,
            "saturation_score": 0.58,
            "contrast_score": 0.62,
        },
    }

    profile = source_profile(row)
    opportunities = edit_opportunity_scores(row, profile)

    assert profile["source_quality"] > 0.70
    assert opportunities["object_removal"] > 0.55
    assert opportunities["object_replacement"] > 0.55


def test_build_pool_uses_feedback_and_schedules_edit_types(tmp_path: Path) -> None:
    score_path = tmp_path / "scores.jsonl"
    manifest_path = tmp_path / "manifest.jsonl"
    feedback_path = tmp_path / "proposals.jsonl"
    output_path = tmp_path / "pool.jsonl"

    score_rows = [
        {
            "key": "source__object_case",
            "image": "data/object.jpg",
            "caption": "a mug on a clean table",
            "score": 0.82,
            "edit_families": ["object", "background"],
            "primary_family": "object",
            "vlm": {
                "quality_score": 0.86,
                "natural_image_score": 0.84,
                "editable_content_score": 0.82,
                "object_region_clarity": 0.80,
                "preservation_potential": 0.82,
                "clutter_penalty": 0.12,
                "text_watermark_penalty": 0.02,
            },
            "stats_scores": {
                "technical_quality_score": 0.82,
                "structure_score": 0.76,
                "saturation_score": 0.60,
                "contrast_score": 0.64,
            },
        },
        {
            "key": "source__color_case",
            "image": "data/color.jpg",
            "caption": "a red car on a street",
            "score": 0.80,
            "edit_families": ["color", "object"],
            "primary_family": "color",
            "vlm": {
                "quality_score": 0.82,
                "natural_image_score": 0.83,
                "editable_content_score": 0.80,
                "object_region_clarity": 0.65,
                "preservation_potential": 0.80,
                "clutter_penalty": 0.20,
                "text_watermark_penalty": 0.03,
            },
            "stats_scores": {
                "technical_quality_score": 0.80,
                "structure_score": 0.70,
                "saturation_score": 0.78,
                "contrast_score": 0.68,
            },
        },
    ]
    manifest_rows = [
        {"key": row["key"], "image": row["image"], "caption": row["caption"], "metadata": {}}
        for row in score_rows
    ]
    feedback_rows = [
        {
            "record_key": "source__object_case",
            "status": "accepted",
            "candidate_role": "policy",
            "proposal": {"structured_edit": {"edit_type": "object_removal"}},
            "evaluator": {
                "component_scores": {
                    "cepr_semantic_edit": 0.76,
                    "cepr_preservation": 0.82,
                    "cepr_validity": 1.0,
                    "cepr_raw_reward": 0.80,
                    "rubric_edit_success": 0.74,
                    "rubric_preservation": 0.84,
                    "rubric_validity": 1.0,
                },
                "signals": {
                    "cepr_latent_changed_fraction": 0.12,
                    "cepr_latent_outside_preservation": 0.78,
                },
            },
        }
    ]
    write_jsonl(score_path, score_rows)
    write_jsonl(manifest_path, manifest_rows)
    write_jsonl(feedback_path, feedback_rows)

    args = argparse.Namespace(
        score_jsonl=[str(score_path)],
        manifest_jsonl=[str(manifest_path)],
        feedback_proposals=[str(feedback_path)],
        output=output_path,
        profile_output=None,
        rejected_output=None,
        summary=None,
        max_records=4,
        max_per_source=2,
        min_source_quality=0.55,
        min_utility=0.20,
        min_hash_distance=0,
        relax_hash_on_backfill=True,
        target_fractions="object_removal=0.50,color_change=0.50",
        seed=7,
    )

    payload = build_pool(args)
    manifest = payload["manifest"]
    edit_types = [row["metadata"]["scheduled_edit_type"] for row in manifest]

    assert len(manifest) == 4
    assert "object_removal" in edit_types
    assert "color_change" in edit_types
    object_removal = next(
        row
        for row in manifest
        if row["metadata"]["scheduled_edit_type"] == "object_removal"
        and row["metadata"]["original_key"] == "source__object_case"
    )
    assert object_removal["metadata"]["feedback"]["accepted"] == 1
    assert object_removal["metadata"]["data_utility_score"] > 0.20
