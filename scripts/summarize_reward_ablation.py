#!/usr/bin/env python3
"""Aggregate reward-ablation arms into one reviewer-facing comparison table.

Each arm is a `run_reward_discrimination_study.py` output directory containing a
`report.json`. This tool reads the `overall` block of every arm and lays the
arms side by side so the contribution of each reward component is visible at a
glance:

    - good accept rate   -> recall (higher is better)
    - noop/corrupt/wrong accept rate -> FALSE-ACCEPT rate (MUST stay 0.000)
    - AUC good-vs-{noop,corrupt,wrong} -> separation of the raw reward

The central reviewer claim ("each component earns its place") is supported when
removing a gate either (a) reintroduces false-accepts or (b) costs recall, while
the surgical relaxation arm recovers recall WITHOUT raising noop acceptance.

No GPU and no model loading; this only reads JSON. Run with any python.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_overall(arm_dir: Path) -> dict[str, Any] | None:
    report_path = arm_dir / "report.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    return report.get("overall")


def _fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms-root",
        default="outputs/analysis/reward_ablation",
        help="Directory whose immediate subdirectories are arm outputs (each with report.json).",
    )
    parser.add_argument(
        "--order",
        nargs="*",
        default=None,
        help="Explicit arm subdirectory names in display order. Defaults to sorted discovery.",
    )
    parser.add_argument("--out", default="outputs/analysis/reward_ablation/ablation_summary")
    args = parser.parse_args()

    root = Path(args.arms_root)
    if args.order:
        arm_names = list(args.order)
    else:
        arm_names = sorted(p.name for p in root.iterdir() if p.is_dir())

    rows: list[dict[str, Any]] = []
    for name in arm_names:
        overall = _load_overall(root / name)
        if overall is None:
            print(f"WARNING: no report.json for arm '{name}', skipping.", flush=True)
            continue
        auc = overall.get("auc_good_vs", {})
        rows.append(
            {
                "arm": name,
                "good_accept": overall["good"]["accept_rate"],
                "noop_accept": overall["noop"]["accept_rate"],
                "corrupt_accept": overall["corrupt"]["accept_rate"],
                "wrong_accept": overall["wrong"]["accept_rate"],
                "false_accepts": (
                    overall["noop"]["accept_rate"]
                    + overall["corrupt"]["accept_rate"]
                    + overall["wrong"]["accept_rate"]
                ),
                "auc_noop": auc.get("noop"),
                "auc_corrupt": auc.get("corrupt"),
                "auc_wrong": auc.get("wrong"),
            }
        )

    if not rows:
        print("ERROR: no arms with report.json found.", flush=True)
        return 2

    header = (
        "| arm | good_accept (recall) | noop_accept | corrupt_accept | wrong_accept "
        "| any_false_accept | AUC noop | AUC corrupt | AUC wrong |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [
        "# Reward ablation summary\n",
        "Each row knocks out one component. `*_accept` on a negative class is a "
        "FALSE-ACCEPT and must stay **0.000**. `good_accept` is recall.\n",
        header,
        sep,
    ]
    for r in rows:
        any_fa = "YES" if r["false_accepts"] > 1e-9 else "no"
        lines.append(
            f"| {r['arm']} | {_fmt(r['good_accept'])} | {_fmt(r['noop_accept'])} "
            f"| {_fmt(r['corrupt_accept'])} | {_fmt(r['wrong_accept'])} | {any_fa} "
            f"| {_fmt(r['auc_noop'])} | {_fmt(r['auc_corrupt'])} | {_fmt(r['auc_wrong'])} |"
        )

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    (out_base.with_suffix(".md")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_base.with_suffix(".json")).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\nWrote {out_base.with_suffix('.md')} and {out_base.with_suffix('.json')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
