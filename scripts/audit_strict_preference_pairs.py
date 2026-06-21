#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from reward_correlation_audit import (
    component,
    drift_flag,
    edit_type,
    group_rows,
    group_stats,
    judge_reliable,
    judge_supported,
    load_run_rows,
    noop_flag,
    proposal,
    score_cepr_raw,
    score_conservative,
    score_esc_minimal,
    score_judge_strict,
    score_rubric,
)


def parse_rounds(value: str | None) -> list[int] | None:
    if not value:
        return None
    rounds: set[int] = set()
    for chunk in value.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            start, end = chunk.split('-', 1)
            rounds.update(range(int(start), int(end) + 1))
        else:
            rounds.add(int(chunk))
    return sorted(rounds)


def finite_float(value: Any, default: float = math.nan) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def min_existing(*values: float, default: float = 0.0) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return min(clean) if clean else default


def semantic_score(row: dict[str, Any]) -> float:
    return max(
        component(row, 'cepr_semantic_edit', default=0.0),
        component(row, 'cepr_edit_specificity', default=0.0),
        component(row, 'rubric_edit_success', default=0.0),
        component(row, 'rubric_required_after', default=0.0),
    )


def preservation_score(row: dict[str, Any]) -> float:
    return min_existing(
        component(row, 'cepr_preservation', default=math.nan),
        component(row, 'rubric_preservation', default=math.nan),
        default=0.0,
    )


def validity_score(row: dict[str, Any]) -> float:
    return min_existing(
        component(row, 'cepr_validity', default=math.nan),
        component(row, 'rubric_validity', default=math.nan),
        default=0.0,
    )


def candidate_score(row: dict[str, Any], stats: dict[str, float], mode: str) -> float:
    mode = mode.strip().lower()
    if mode in {'judge', 'judge_strict', 'vlm'}:
        return score_judge_strict(row)
    if mode in {'cepr', 'cepr_raw'}:
        return score_cepr_raw(row)
    if mode in {'rubric', 'rubric_reward'}:
        return score_rubric(row)
    if mode in {'esc', 'esc_minimal'}:
        return score_esc_minimal(row, stats)
    return score_conservative(row)


def role_matches(row: dict[str, Any], roles: set[str]) -> bool:
    if not roles:
        return True
    role = str(row.get('candidate_role') or 'policy')
    return role in roles or any(role.startswith(f'{item}:') for item in roles)


def has_paths(row: dict[str, Any]) -> bool:
    return bool(row.get('image_path')) and bool(row.get('edited_image_path'))


def strict_success_reason(row: dict[str, Any], args: argparse.Namespace) -> str | None:
    if not has_paths(row):
        return 'missing_paths'
    if args.require_internal_vlm_judge and not judge_supported(row):
        return 'missing_internal_vlm_judge'
    if args.require_reliable_judge and not judge_reliable(row):
        return 'unreliable_internal_vlm_judge'
    if component(row, 'internal_vlm_judge_score', default=0.0) < args.min_judge_score:
        return 'judge_score_below_floor'
    if component(row, 'internal_vlm_judge_semantic', default=0.0) < args.min_judge_semantic:
        return 'judge_semantic_below_floor'
    if component(row, 'internal_vlm_judge_preservation', default=0.0) < args.min_judge_preservation:
        return 'judge_preservation_below_floor'
    if component(row, 'internal_vlm_judge_artifact_free', default=0.0) < args.min_judge_artifact_free:
        return 'judge_artifact_below_floor'
    if semantic_score(row) < args.min_semantic:
        return 'semantic_below_floor'
    if preservation_score(row) < args.min_preservation:
        return 'preservation_below_floor'
    if validity_score(row) < args.min_validity:
        return 'validity_below_floor'
    if args.reject_noop and noop_flag(row):
        return 'noop_flag'
    if args.reject_drift and drift_flag(row):
        return 'drift_flag'
    return None


def summarize_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    counts = Counter()
    by_edit_type: dict[str, Counter[str]] = {}
    for row in rows:
        reason = strict_success_reason(row, args)
        key = 'strict_success' if reason is None else reason
        counts[key] += 1
        by_edit_type.setdefault(edit_type(row), Counter())[key] += 1
    return {
        'rows': len(rows),
        'strict_success_rows': counts.get('strict_success', 0),
        'strict_success_rate': counts.get('strict_success', 0) / len(rows) if rows else None,
        'reasons': dict(counts),
        'by_edit_type': {key: dict(value) for key, value in sorted(by_edit_type.items())},
    }


def build_pairs(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    positive_roles = {item.strip() for item in args.positive_roles.split(',') if item.strip()}
    grouped = group_rows(rows)
    pairs: list[dict[str, Any]] = []
    skipped = Counter()
    productive = Counter()
    pair_by_type = Counter()
    success_by_type = Counter()
    positive_by_type = Counter()

    for group_id, group in sorted(grouped.items()):
        image_rows = [row for row in group if has_paths(row)]
        policy_rows = [row for row in image_rows if role_matches(row, {'policy'})]
        if len(policy_rows) < args.min_candidates:
            skipped['too_few_policy_candidates'] += 1
            continue

        successes = [row for row in policy_rows if strict_success_reason(row, args) is None]
        rate = len(successes) / len(policy_rows)
        etype = edit_type(policy_rows[0]) if policy_rows else 'unknown'
        if len(successes) == 0:
            productive['all_fail'] += 1
            skipped['productive_all_fail'] += 1
            continue
        if rate < args.min_success_rate:
            productive['below_min_success_rate'] += 1
            skipped['productive_below_min_success_rate'] += 1
            continue
        if rate > args.max_success_rate:
            productive['all_or_most_pass'] += 1
            skipped['productive_all_or_most_pass'] += 1
            continue
        productive['middle_band'] += 1
        success_by_type[etype] += len(successes)

        stats = group_stats(image_rows)
        positives = [row for row in successes if role_matches(row, positive_roles)]
        if not positives:
            skipped['no_positive_role_success'] += 1
            continue
        positives.sort(
            key=lambda row: candidate_score(row, stats, args.score_mode),
            reverse=True,
        )
        chosen = positives[0]
        chosen_score = candidate_score(chosen, stats, args.score_mode)
        chosen_judge = component(chosen, 'internal_vlm_judge_score', default=0.0)
        positive_by_type[etype] += 1

        rejected_rows = [row for row in image_rows if row is not chosen]
        rejected_rows.sort(
            key=lambda row: (
                strict_success_reason(row, args) is None,
                candidate_score(row, stats, args.score_mode),
            )
        )
        added = 0
        for rejected in rejected_rows:
            if added >= args.max_pairs_per_group:
                break
            rejected_success = strict_success_reason(rejected, args) is None
            if rejected_success and not args.allow_success_losers:
                skipped['success_loser_filtered'] += 1
                continue
            rejected_score = candidate_score(rejected, stats, args.score_mode)
            rejected_judge = component(rejected, 'internal_vlm_judge_score', default=0.0)
            score_margin = chosen_score - rejected_score
            judge_margin = chosen_judge - rejected_judge
            if score_margin < args.min_score_margin and judge_margin < args.min_judge_margin:
                skipped['margin_too_small'] += 1
                continue
            pairs.append(
                {
                    'prompt': str(proposal(chosen).get('instruction') or ''),
                    'chosen_image': chosen.get('edited_image_path'),
                    'rejected_image': rejected.get('edited_image_path'),
                    'edit_image': chosen.get('image_path'),
                    'sample_weight': round(args.sample_weight * (1.0 + min(max(score_margin, 0.0), 0.5)), 6),
                    'record_key': chosen.get('record_key'),
                    'group_id': group_id,
                    'family': etype,
                    'operation_id': proposal(chosen).get('operation_id'),
                    'structured_edit': proposal(chosen).get('structured_edit') or {},
                    'preference_source': 'strict_internal_vlm_pair_audit',
                    'chosen_candidate_role': chosen.get('candidate_role', 'policy'),
                    'rejected_candidate_role': rejected.get('candidate_role', 'policy'),
                    'chosen_candidate_index': chosen.get('candidate_index'),
                    'rejected_candidate_index': rejected.get('candidate_index'),
                    'chosen_score': round(chosen_score, 6),
                    'rejected_score': round(rejected_score, 6),
                    'score_margin': round(score_margin, 6),
                    'chosen_judge': round(chosen_judge, 6),
                    'rejected_judge': round(rejected_judge, 6),
                    'judge_margin': round(judge_margin, 6),
                    'strict_pair_builder': {
                        'score_mode': args.score_mode,
                        'success_rate': round(rate, 6),
                        'policy_candidates': len(policy_rows),
                        'strict_successes': len(successes),
                    },
                }
            )
            pair_by_type[etype] += 1
            added += 1
        if added == 0:
            skipped['no_pairs_added'] += 1

    if args.max_pairs_per_family > 0:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in pairs:
            buckets.setdefault(str(row.get('family') or 'unknown'), []).append(row)
        balanced: list[dict[str, Any]] = []
        for family, items in sorted(buckets.items()):
            items.sort(key=lambda row: (row.get('judge_margin', 0.0), row.get('score_margin', 0.0)), reverse=True)
            balanced.extend(items[: args.max_pairs_per_family])
        rng.shuffle(balanced)
        pairs = balanced

    summary = {
        'groups': len(grouped),
        'pairs': len(pairs),
        'per_edit_type': dict(Counter(str(row.get('family') or 'unknown') for row in pairs)),
        'productive_groups': dict(productive),
        'strict_success_candidates_by_edit_type': dict(success_by_type),
        'chosen_groups_by_edit_type': dict(positive_by_type),
        'raw_pair_attempts_by_edit_type': dict(pair_by_type),
        'skipped': dict(skipped),
    }
    return pairs, summary


def write_report(path: Path, summary: dict[str, Any], row_summary: dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        '# Strict Preference Pair Audit',
        '',
        f'Run: `{args.run_dir}`',
        f'Rounds: `{args.rounds or "all"}`',
        f'Score mode: `{args.score_mode}`',
        '',
        '## Candidate Signal',
        '',
        f'- Rows: {row_summary["rows"]}',
        f'- Strict-success rows: {row_summary["strict_success_rows"]}',
        f'- Strict-success rate: {row_summary["strict_success_rate"]:.4f}' if row_summary['strict_success_rate'] is not None else '- Strict-success rate: n/a',
        '',
        '## Pair Yield',
        '',
        f'- Groups: {summary["groups"]}',
        f'- Pairs after strict filtering/balancing: {summary["pairs"]}',
        f'- Productive groups: {summary["productive_groups"]}',
        f'- Pairs per edit type: {summary["per_edit_type"]}',
        '',
        '## Main Filters',
        '',
    ]
    for key, value in sorted(row_summary['reasons'].items(), key=lambda item: (-item[1], item[0]))[:20]:
        lines.append(f'- {key}: {value}')
    lines.extend(['', '## Pair Skips', ''])
    for key, value in sorted(summary['skipped'].items(), key=lambda item: (-item[1], item[0]))[:20]:
        lines.append(f'- {key}: {value}')
    path.write_text('\\n'.join(lines) + '\\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Audit strict internal-VLM preference pair yield from saved self-evolution proposals.')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--rounds', default=None, help='Comma/range syntax such as 1-18,20')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--score-mode', default='conservative', choices=['conservative', 'judge_strict', 'cepr_raw', 'rubric_reward', 'esc_minimal'])
    parser.add_argument('--min-candidates', type=int, default=2)
    parser.add_argument('--min-success-rate', type=float, default=0.20)
    parser.add_argument('--max-success-rate', type=float, default=0.80)
    parser.add_argument('--require-internal-vlm-judge', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--require-reliable-judge', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--min-judge-score', type=float, default=0.55)
    parser.add_argument('--min-judge-semantic', type=float, default=0.55)
    parser.add_argument('--min-judge-preservation', type=float, default=0.55)
    parser.add_argument('--min-judge-artifact-free', type=float, default=0.55)
    parser.add_argument('--min-semantic', type=float, default=0.35)
    parser.add_argument('--min-preservation', type=float, default=0.62)
    parser.add_argument('--min-validity', type=float, default=0.75)
    parser.add_argument('--reject-noop', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--reject-drift', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--positive-roles', default='policy')
    parser.add_argument('--max-pairs-per-group', type=int, default=5)
    parser.add_argument('--max-pairs-per-family', type=int, default=64)
    parser.add_argument('--min-score-margin', type=float, default=0.05)
    parser.add_argument('--min-judge-margin', type=float, default=0.05)
    parser.add_argument('--allow-success-losers', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--sample-weight', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    rounds = parse_rounds(args.rounds)
    rows = load_run_rows(args.run_dir, rounds)
    row_summary = summarize_rows(rows, args)
    pairs, pair_summary = build_pairs(rows, args)
    summary = {
        'args': vars(args) | {'run_dir': str(args.run_dir), 'output_dir': str(args.output_dir)},
        'candidate_summary': row_summary,
        'pair_summary': pair_summary,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / 'strict_preference_manifest.jsonl'
    with manifest_path.open('w', encoding='utf-8') as handle:
        for row in pairs:
            handle.write(json.dumps(row, ensure_ascii=True) + '\\n')
    with (args.output_dir / 'strict_pair_audit_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
    write_report(args.output_dir / 'REPORT.md', pair_summary, row_summary, args)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
