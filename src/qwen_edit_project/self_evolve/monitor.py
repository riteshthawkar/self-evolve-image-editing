from __future__ import annotations

import argparse
import csv
import html
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RoundMetrics:
    round_index: int
    status: str
    groups_total: int
    groups_evaluated: int
    accepted_groups: int
    candidates_total: int
    candidates_evaluated: int
    accepted_candidates: int
    generated_candidates: int
    group_acceptance_rate: float
    candidate_acceptance_rate: float
    avg_accepted_reward: float
    train_samples: int
    train_weight_sum: float
    train_weight_per_group: float
    health_score: float
    health_ema: float
    difficulty_level: int | None
    next_difficulty_level: int | None
    current_group_id: str | None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(lines):
                break
            raise
    return rows


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group_id = str(row.get("group_id") or "")
        if not group_id:
            continue
        grouped.setdefault(group_id, []).append(row)
    return grouped


def _round_index_from_dir(path: Path) -> int:
    try:
        return int(path.name.split("_")[-1])
    except ValueError:
        return 0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _extract_round_metrics(round_dir: Path) -> RoundMetrics:
    summary = _read_json(round_dir / "summary.json")
    progress = _read_json(round_dir / "progress.json")
    train_summary = _read_json(round_dir / "train_weight_summary.json")
    if not train_summary and isinstance(summary.get("train_weight_summary"), dict):
        train_summary = dict(summary["train_weight_summary"])

    rows = _read_jsonl(round_dir / "proposals.jsonl")
    grouped = _group_rows(rows)
    evaluated_statuses = {"accepted", "rejected"}
    evaluated_groups = {
        group_id: group_rows
        for group_id, group_rows in grouped.items()
        if any(str(row.get("status")) in evaluated_statuses for row in group_rows)
    }
    accepted_groups = sum(
        1 for group_rows in grouped.values() if any(str(row.get("status")) == "accepted" for row in group_rows)
    )
    accepted_rows = [row for row in rows if str(row.get("status")) == "accepted"]
    evaluated_rows = [row for row in rows if str(row.get("status")) in evaluated_statuses]
    generated_rows = [row for row in rows if str(row.get("status")) == "generated"]

    reward_values = []
    for row in accepted_rows:
        evaluator = row.get("evaluator") or row.get("solver") or {}
        if isinstance(evaluator, dict):
            reward_values.append(_safe_float(evaluator.get("total_score")))
    avg_reward = _mean(reward_values) or _safe_float(summary.get("avg_total_score"))

    groups_total = (
        _safe_int(summary.get("proposal_groups"))
        or _safe_int(progress.get("groups_total_estimate"))
        or len(grouped)
    )
    groups_evaluated = len(evaluated_groups) or _safe_int(progress.get("groups_completed"))
    candidates_total = _safe_int(summary.get("candidates")) or len(rows)
    candidates_evaluated = len(evaluated_rows)
    train_weight_sum = _safe_float(
        train_summary.get("weight_sum"),
        _safe_float(summary.get("round_training_weight_sum")),
    )
    train_samples = (
        _safe_int(train_summary.get("included"))
        or _safe_int(summary.get("round_training_samples"))
        or _safe_int(progress.get("train_manifest_samples"))
    )
    if train_weight_sum <= 0 and not summary and accepted_rows:
        # Live rounds do not write train_weight_summary until evaluation finishes.
        # Use accepted candidates as a conservative lower-bound signal.
        train_weight_sum = float(len(accepted_rows))
        train_samples = max(train_samples, len(accepted_rows))

    denominator_groups = max(groups_evaluated, 1)
    group_acceptance_rate = _safe_float(
        summary.get("group_acceptance_rate"),
        accepted_groups / denominator_groups if groups_evaluated else 0.0,
    )
    candidate_acceptance_rate = _safe_float(
        summary.get("candidate_acceptance_rate"),
        len(accepted_rows) / max(candidates_evaluated, 1) if candidates_evaluated else 0.0,
    )
    train_weight_per_group = train_weight_sum / denominator_groups if groups_evaluated else 0.0

    group_component = _clamp(group_acceptance_rate / 0.50)
    reward_component = _clamp((avg_reward - 0.30) / 0.35) if avg_reward > 0 else 0.0
    signal_component = _clamp(train_weight_per_group / 0.75)
    health_score = 100.0 * (
        0.50 * group_component
        + 0.30 * reward_component
        + 0.20 * signal_component
    )
    if groups_evaluated <= 0:
        health_score = 0.0
    if accepted_groups <= 0 and train_weight_sum <= 0:
        health_score = 0.0

    status = str(summary.get("status") or progress.get("status") or "unknown")
    return RoundMetrics(
        round_index=_safe_int(summary.get("round_index"), _round_index_from_dir(round_dir)),
        status=status,
        groups_total=groups_total,
        groups_evaluated=groups_evaluated,
        accepted_groups=accepted_groups,
        candidates_total=candidates_total,
        candidates_evaluated=candidates_evaluated,
        accepted_candidates=len(accepted_rows),
        generated_candidates=len(generated_rows),
        group_acceptance_rate=group_acceptance_rate,
        candidate_acceptance_rate=candidate_acceptance_rate,
        avg_accepted_reward=avg_reward,
        train_samples=train_samples,
        train_weight_sum=train_weight_sum,
        train_weight_per_group=train_weight_per_group,
        health_score=health_score,
        health_ema=health_score,
        difficulty_level=summary.get("difficulty_level") or progress.get("difficulty_level"),
        next_difficulty_level=summary.get("next_difficulty_level") or progress.get("next_difficulty_level"),
        current_group_id=progress.get("current_group_id"),
    )


def collect_metrics(root: Path) -> list[RoundMetrics]:
    rounds = [
        _extract_round_metrics(path)
        for path in sorted(root.glob("round_*"), key=_round_index_from_dir)
        if path.is_dir()
    ]
    ema: float | None = None
    for item in rounds:
        ema = item.health_score if ema is None else 0.40 * item.health_score + 0.60 * ema
        item.health_ema = ema
    return rounds


def _polyline(points: list[tuple[float, float]], color: str, width: int = 3, dash: str | None = None) -> str:
    if not points:
        return ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{text}" fill="none" stroke="{color}" stroke-width="{width}"{dash_attr}/>'


def _line(x1: float, y1: float, x2: float, y2: float, color: str = "#d8dee9", width: int = 1) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"/>'


def _circle(x: float, y: float, radius: float, color: str, stroke: str = "#ffffff") -> str:
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" '
        f'fill="{color}" stroke="{stroke}" stroke-width="2"/>'
    )


def render_svg(rounds: list[RoundMetrics]) -> str:
    width = 1180
    height = 620
    margin_left = 64
    margin_right = 34
    top = 46
    panel_h = 210
    gap = 92
    panel2_top = top + panel_h + gap
    plot_w = width - margin_left - margin_right
    max_rounds = max(len(rounds), 1)
    x_step = plot_w / max(max_rounds, 1)
    label_every = max(1, math.ceil(max_rounds / 14))

    def x_at(index: int) -> float:
        return margin_left + x_step * (index + 0.5)

    def y_at(value: float, panel_top: float, panel_height: float) -> float:
        return panel_top + panel_height - _clamp(value, 0, 100) / 100.0 * panel_height

    def point_series(values: list[float], panel_top: float) -> list[tuple[float, float]]:
        return [(x_at(index), y_at(value, panel_top, panel_h)) for index, value in enumerate(values)]

    def draw_panel(panel_top: float, title: str, subtitle: str) -> list[str]:
        panel: list[str] = [
            f'<text x="{margin_left}" y="{panel_top - 22}" font-family="Inter, Arial, sans-serif" '
            f'font-size="17" font-weight="700" fill="#172033">{title}</text>',
            f'<text x="{margin_left}" y="{panel_top - 4}" font-family="Inter, Arial, sans-serif" '
            f'font-size="12" fill="#667085">{subtitle}</text>',
        ]
        for tick in range(0, 101, 25):
            y = y_at(tick, panel_top, panel_h)
            panel.append(_line(margin_left, y, width - margin_right, y, "#e8edf4"))
            panel.append(
                f'<text x="{margin_left - 28}" y="{y + 4:.1f}" text-anchor="end" '
                f'font-family="Inter, Arial, sans-serif" font-size="11" fill="#758195">{tick}</text>'
            )
        panel.append(_line(margin_left, panel_top, margin_left, panel_top + panel_h, "#c8d1df", 1))
        panel.append(_line(margin_left, panel_top + panel_h, width - margin_right, panel_top + panel_h, "#c8d1df", 1))
        return panel

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]

    # Reference bands for the health panel.
    band_x = margin_left
    band_w = plot_w
    parts.append(
        f'<rect x="{band_x}" y="{y_at(100, top, panel_h):.1f}" width="{band_w:.1f}" '
        f'height="{y_at(70, top, panel_h) - y_at(100, top, panel_h):.1f}" fill="#ecfdf3" opacity="0.75"/>'
    )
    parts.append(
        f'<rect x="{band_x}" y="{y_at(70, top, panel_h):.1f}" width="{band_w:.1f}" '
        f'height="{y_at(45, top, panel_h) - y_at(70, top, panel_h):.1f}" fill="#fffbeb" opacity="0.75"/>'
    )
    parts.append(
        f'<rect x="{band_x}" y="{y_at(45, top, panel_h):.1f}" width="{band_w:.1f}" '
        f'height="{y_at(0, top, panel_h) - y_at(45, top, panel_h):.1f}" fill="#fff1f2" opacity="0.50"/>'
    )
    parts.extend(
        draw_panel(
            top,
            "Training Health Index",
            "Bars show per-round signal quality. The blue line is the smoothed trend.",
        )
    )
    parts.extend(
        draw_panel(
            panel2_top,
            "Signal Components",
            "Drivers normalized to 0-100: solved groups, accepted CEPR quality, and weighted training density.",
        )
    )
    for tick in (45, 70):
        y = y_at(tick, top, panel_h)
        parts.append(_line(margin_left, y, width - margin_right, y, "#cbd5e1", 1))
        parts.append(
            f'<text x="{width - margin_right - 4}" y="{y - 5:.1f}" text-anchor="end" '
            f'font-family="Inter, Arial, sans-serif" font-size="11" fill="#667085">{tick}</text>'
        )

    for index, item in enumerate(rounds):
        x = x_at(index)
        bar_w = max(7, min(24, x_step * 0.52))
        bar_h = (item.health_score / 100.0) * panel_h
        y = top + panel_h - bar_h
        color = "#16a34a" if item.health_score >= 70 else "#f59e0b" if item.health_score >= 45 else "#ef4444"
        parts.append(
            f'<rect x="{x - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" '
            f'rx="5" fill="{color}" opacity="0.82"/>'
        )
        if index % label_every == 0 or index == len(rounds) - 1:
            parts.append(
                f'<text x="{x:.1f}" y="{top + panel_h + 20}" text-anchor="middle" '
                f'font-family="Inter, Arial, sans-serif" font-size="11" fill="#667085">{item.round_index}</text>'
            )

    ema_points = point_series([item.health_ema for item in rounds], top)
    parts.append(_polyline(ema_points, "#2563eb", 4))
    for x, y in ema_points[-12:]:
        parts.append(_circle(x, y, 4, "#2563eb"))

    group_points = point_series([item.group_acceptance_rate * 100 for item in rounds], panel2_top)
    reward_points = point_series(
        [_clamp((item.avg_accepted_reward - 0.30) / 0.35) * 100 for item in rounds],
        panel2_top,
    )
    signal_points = point_series([_clamp(item.train_weight_per_group / 0.75) * 100 for item in rounds], panel2_top)
    parts.append(_polyline(group_points, "#7c3aed", 3))
    parts.append(_polyline(reward_points, "#059669", 3))
    parts.append(_polyline(signal_points, "#ea580c", 3))
    for series, color in ((group_points, "#7c3aed"), (reward_points, "#059669"), (signal_points, "#ea580c")):
        for x, y in series[-8:]:
            parts.append(_circle(x, y, 3.4, color))

    legend = [
        ("Health EMA", "#2563eb", 720),
        ("Group success", "#7c3aed", 835),
        ("Accepted CEPR quality", "#059669", 962),
        ("Training density", "#ea580c", 1110),
    ]
    legend_y = 18
    for label, color, x in legend:
        parts.append(f'<line x1="{x}" y1="{legend_y}" x2="{x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="4"/>')
        parts.append(
            f'<text x="{x}" y="{legend_y + 18}" font-family="Inter, Arial, sans-serif" '
            f'font-size="11" fill="#536079">{label}</text>'
        )

    if rounds:
        latest = rounds[-1]
        parts.append(
            f'<text x="{margin_left}" y="{height - 24}" font-family="Inter, Arial, sans-serif" '
            f'font-size="13" fill="#172033">'
            f'Latest: round {latest.round_index}, status={html.escape(latest.status)}, '
            f'groups={latest.accepted_groups}/{latest.groups_evaluated or latest.groups_total}, '
            f'health={latest.health_score:.1f}, EMA={latest.health_ema:.1f}'
            '</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def write_outputs(rounds: list[RoundMetrics], output_dir: Path, refresh_seconds: int = 60) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = [asdict(item) for item in rounds]
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        if metrics:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
            writer.writeheader()
            writer.writerows(metrics)
        else:
            handle.write("round_index,status,health_score,health_ema\n")
    svg = render_svg(rounds)
    (output_dir / "training_health.svg").write_text(svg, encoding="utf-8")

    latest = rounds[-1] if rounds else None
    verdict = "waiting for evaluated rounds"
    if latest is not None:
        if latest.health_ema >= 70:
            verdict = "good"
        elif latest.health_ema >= 45:
            verdict = "usable but sparse"
        else:
            verdict = "weak or too sparse"
    previous = rounds[-2] if len(rounds) >= 2 else None
    ema_delta = (latest.health_ema - previous.health_ema) if latest is not None and previous is not None else 0.0
    trend = "stable"
    if ema_delta >= 5:
        trend = "improving"
    elif ema_delta <= -5:
        trend = "declining"
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    latest_reward = latest.avg_accepted_reward if latest is not None else 0.0
    latest_signal = latest.train_weight_per_group if latest is not None else 0.0
    latest_health = latest.health_score if latest is not None else 0.0
    latest_ema = latest.health_ema if latest is not None else 0.0
    latest_groups = (
        f"{latest.accepted_groups}/{latest.groups_evaluated or latest.groups_total}" if latest is not None else "0/0"
    )
    current_group = html.escape(str(latest.current_group_id or "none")) if latest is not None else "none"
    verdict_class = "good" if latest_ema >= 70 else "warn" if latest_ema >= 45 else "bad"
    trend_class = "good" if trend == "improving" else "bad" if trend == "declining" else "neutral"
    table_rows = []
    for item in rounds[-20:]:
        row_class = "good-row" if item.health_score >= 70 else "warn-row" if item.health_score >= 45 else "bad-row"
        status_class = "running" if item.status in {"running", "training"} else "completed"
        table_rows.append(
            f'<tr class="{row_class}">'
            f"<td>{item.round_index}</td>"
            f'<td><span class="pill {status_class}">{html.escape(item.status)}</span></td>'
            f"<td>{item.accepted_groups}/{item.groups_evaluated or item.groups_total}</td>"
            f"<td>{item.group_acceptance_rate:.3f}</td>"
            f"<td>{item.candidate_acceptance_rate:.3f}</td>"
            f"<td>{item.avg_accepted_reward:.3f}</td>"
            f"<td>{item.train_samples}</td>"
            f"<td>{item.train_weight_sum:.2f}</td>"
            f"<td>{item.train_weight_per_group:.2f}</td>"
            f"<td>{item.health_score:.1f}</td>"
            f"<td>{item.health_ema:.1f}</td>"
            f"<td>{item.difficulty_level or ''}</td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="{refresh_seconds}">
  <title>Self-Evolve Training Monitor</title>
  <style>
    :root {{
      --bg: #f4f7fb;
      --card: #ffffff;
      --line: #e5ebf3;
      --text: #172033;
      --muted: #667085;
      --blue: #2563eb;
      --green: #059669;
      --amber: #d97706;
      --red: #dc2626;
      --purple: #7c3aed;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    .wrap {{ max-width: 1240px; margin: 0 auto; padding: 28px 28px 40px; }}
    .topbar {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 24px; margin-bottom: 20px; }}
    h1 {{ margin: 0; font-size: 32px; line-height: 1.15; }}
    .subtitle {{ margin-top: 8px; color: var(--muted); max-width: 820px; line-height: 1.45; }}
    .meta {{ display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; min-width: 360px; }}
    .chip {{ border: 1px solid var(--line); background: var(--card); border-radius: 999px; padding: 7px 11px; font-size: 13px; color: var(--muted); }}
    .chip strong {{ color: var(--text); }}
    .chip.good {{ border-color: #bbf7d0; background: #f0fdf4; color: #166534; }}
    .chip.warn {{ border-color: #fde68a; background: #fffbeb; color: #92400e; }}
    .chip.bad {{ border-color: #fecdd3; background: #fff1f2; color: #9f1239; }}
    .chart-head .chip {{ max-width: 520px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .cards {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 14px 13px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    }}
    .label {{ font-size: 12px; color: var(--muted); margin-bottom: 7px; }}
    .value {{ font-size: 25px; font-weight: 750; line-height: 1.1; }}
    .hint {{ font-size: 12px; color: var(--muted); margin-top: 7px; line-height: 1.35; }}
    .chart-card {{ background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 16px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04); }}
    .chart-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 18px; padding: 0 4px 10px; }}
    .chart-title {{ font-size: 19px; font-weight: 750; }}
    .chart-copy {{ color: var(--muted); font-size: 13px; margin-top: 4px; line-height: 1.4; }}
    img {{ width: 100%; display: block; background: white; }}
    .table-card {{ margin-top: 18px; background: var(--card); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04); }}
    .table-title {{ display: flex; justify-content: space-between; gap: 16px; padding: 14px 16px; border-bottom: 1px solid var(--line); }}
    .table-title strong {{ font-size: 16px; }}
    .table-title span {{ color: var(--muted); font-size: 13px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #edf1f7; text-align: right; font-size: 13px; white-space: nowrap; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ color: #536079; font-weight: 700; background: #fbfcfe; }}
    tr:last-child td {{ border-bottom: 0; }}
    .pill {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 3px 8px; font-size: 12px; font-weight: 650; }}
    .pill.completed {{ background: #eef2ff; color: #3730a3; }}
    .pill.running {{ background: #ecfeff; color: #155e75; }}
    .good-row td:nth-last-child(3) {{ color: var(--green); font-weight: 700; }}
    .warn-row td:nth-last-child(3) {{ color: var(--amber); font-weight: 700; }}
    .bad-row td:nth-last-child(3) {{ color: var(--red); font-weight: 700; }}
    code {{ color: #334155; background: #f1f5f9; padding: 2px 5px; border-radius: 5px; }}
    @media (max-width: 980px) {{
      .topbar {{ display: block; }}
      .meta {{ justify-content: flex-start; margin-top: 14px; min-width: 0; }}
      .cards {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .table-card {{ overflow-x: auto; }}
    }}
  </style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <div>
      <h1>Self-Evolve Training Monitor</h1>
      <div class="subtitle">Tracks whether the self-evolving loop is creating useful supervision. The main score should improve when more proposal groups are solved, accepted CEPR quality is high, and weighted SFT has enough training mass.</div>
    </div>
    <div class="meta">
      <div class="chip">Updated <strong>{updated}</strong></div>
      <div class="chip {verdict_class}">Verdict <strong>{html.escape(verdict)}</strong></div>
      <div class="chip {trend_class}">Trend <strong>{html.escape(trend)}</strong> ({ema_delta:+.1f})</div>
      <div class="chip">Rounds <strong>{len(rounds)}</strong></div>
    </div>
  </div>
  <div class="cards">
    <div class="card">
      <div class="label">Latest health</div>
      <div class="value">{latest_health:.1f}</div>
      <div class="hint">Per-round score from group success, CEPR reward, and training density.</div>
    </div>
    <div class="card">
      <div class="label">Smoothed health</div>
      <div class="value">{latest_ema:.1f}</div>
      <div class="hint">EMA is the easiest signal to watch while the run continues.</div>
    </div>
    <div class="card">
      <div class="label">Solved groups</div>
      <div class="value">{latest_groups}</div>
      <div class="hint">Target range is usually 2-5 solved groups out of 8.</div>
    </div>
    <div class="card">
      <div class="label">Avg accepted CEPR</div>
      <div class="value">{latest_reward:.3f}</div>
      <div class="hint">Accepted edits should remain clearly above the reward threshold.</div>
    </div>
    <div class="card">
      <div class="label">Signal per group</div>
      <div class="value">{latest_signal:.2f}</div>
      <div class="hint">Weighted training mass normalized by evaluated proposal groups.</div>
    </div>
  </div>
  <div class="chart-card">
    <div class="chart-head">
      <div>
        <div class="chart-title">Training Signal Dashboard</div>
        <div class="chart-copy">Health above 70 is strong, 45-70 is usable but sparse, and below 45 means the loop is not generating enough useful supervision.</div>
      </div>
      <div class="chip">Current group <strong>{current_group}</strong></div>
    </div>
    <img src="training_health.svg" alt="Self-evolve training health chart">
  </div>
  <div class="table-card">
    <div class="table-title">
      <strong>Round Details</strong>
      <span>Candidate rate is diagnostic only; group rate drives curriculum difficulty.</span>
    </div>
    <table>
      <thead>
        <tr>
          <th>Round</th><th>Status</th><th>Solved Groups</th><th>Group Rate</th><th>Candidate Rate</th>
          <th>Avg CEPR</th><th>Train Samples</th><th>Weight Sum</th><th>Weight/Group</th>
          <th>Health</th><th>EMA</th><th>Difficulty</th>
        </tr>
      </thead>
      <tbody>
        {''.join(table_rows)}
      </tbody>
    </table>
  </div>
</div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def update_dashboard(root: Path, output_dir: Path, refresh_seconds: int = 60) -> list[RoundMetrics]:
    rounds = collect_metrics(root)
    write_outputs(rounds, output_dir, refresh_seconds=refresh_seconds)
    return rounds


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a self-evolve training health dashboard.")
    parser.add_argument("--root", required=True, type=Path, help="Self-evolve run root directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Dashboard output directory.")
    parser.add_argument("--watch-seconds", type=int, default=0, help="Refresh interval. Use 0 for one update.")
    parser.add_argument("--refresh-seconds", type=int, default=60, help="HTML auto-refresh interval.")
    args = parser.parse_args()

    output_dir = args.output_dir or args.root / "monitor"
    while True:
        rounds = update_dashboard(args.root, output_dir, refresh_seconds=args.refresh_seconds)
        latest = rounds[-1] if rounds else None
        if latest is None:
            print(f"No rounds found. Dashboard written to {output_dir / 'index.html'}", flush=True)
        else:
            print(
                f"round={latest.round_index} status={latest.status} "
                f"health={latest.health_score:.1f} ema={latest.health_ema:.1f} "
                f"groups={latest.accepted_groups}/{latest.groups_evaluated or latest.groups_total} "
                f"dashboard={output_dir / 'index.html'}",
                flush=True,
            )
        if args.watch_seconds <= 0:
            break
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()
