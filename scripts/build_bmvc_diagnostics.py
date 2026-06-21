#!/usr/bin/env python3
"""Build paper-ready diagnostic tables and figures for the BMVC draft.

The script is intentionally deterministic and reads only completed experiment
artifacts. It does not launch training or evaluation.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs" / "BMVCTemplate2026"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"

FINAL_RUN = (
    ROOT
    / "outputs/self_evolve/final_rubric_cepr_v1_512_k2_scripta30_steps128/"
    / "internal-cepr-rubric-trainable-proposer"
)

IMGEDIT_BASELINE = ROOT / "outputs/scores/imgedit/qwen_edit_2509_baseline_imgedit_summary.json"
IMGEDIT_FAILED = ROOT / "outputs/scores/imgedit/cepr_stable_continue_r24_summary.json"
IMGEDIT_BASELINE_AVG = ROOT / "outputs/scores/imgedit/qwen_edit_2509_baseline_imgedit_average_score.json"
IMGEDIT_FAILED_AVG = ROOT / "outputs/scores/imgedit/cepr_stable_continue_r24_average_score.json"
IMGEDIT_META = ROOT / "data/processed/benchmark/imgedit/basic_edit.json"
IMGEDIT_BASELINE_IMAGES = ROOT / "outputs/benchmark_images/imgedit/qwen_edit_2509_baseline_imgedit"
IMGEDIT_FAILED_IMAGES = ROOT / "outputs/benchmark_images/imgedit/cepr_stable_continue_r24"
GEDIT_BASELINE = ROOT / "outputs/scores/gedit/qwen_edit_2509_baseline_gedit_summary.json"

CANARY_32 = (
    ROOT
    / "outputs/quick_eval/imgedit_canary_o0_n32/"
    / "rubric_cepr_v1_64_r4_canary32_vs_baseline_comparison.json"
)
CANARY_64_O0 = (
    ROOT
    / "outputs/quick_eval/imgedit_canary_o0_n64/"
    / "rubric_cepr_v1_512_r4_canary64_vs_baseline_comparison.json"
)
CANARY_64_O64 = (
    ROOT
    / "outputs/quick_eval/imgedit_canary_o64_n64/"
    / "rubric_cepr_v1_512_r4_canary64_o64b_vs_baseline_comparison.json"
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def esc(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}"


def fmt(x: float | None, digits: int = 3) -> str:
    if x is None:
        return "--"
    return f"{x:.{digits}f}"


def status_of_manifest_row(row: dict[str, Any]) -> str:
    return (
        str(row.get("candidate_status") or row.get("status") or row.get("weight_reason") or "unknown")
        .strip()
        .lower()
    )


def family_of_row(row: dict[str, Any]) -> str:
    fam = row.get("family")
    if not fam and isinstance(row.get("structured_edit"), dict):
        fam = row["structured_edit"].get("edit_type")
    return str(fam or "unknown")


def round_summary(round_index: int) -> dict[str, Any]:
    return load_json(FINAL_RUN / f"round_{round_index:02d}" / "summary.json")


def manifest_status_counts(round_index: int) -> Counter:
    rows = read_jsonl(FINAL_RUN / f"round_{round_index:02d}" / "train_manifest.jsonl")
    return Counter(status_of_manifest_row(row) for row in rows)


def build_main_results() -> None:
    im_b = load_json(IMGEDIT_BASELINE)["metrics"]
    im_f = load_json(IMGEDIT_FAILED)["metrics"]
    ge_b = load_json(GEDIT_BASELINE)["metrics"]
    gedit_n = sum(group["count"] for group in ge_b["groups"].values())

    im_b_avg = im_b["overall_average"]
    im_f_avg = im_f["overall_average"]
    delta = im_f_avg - im_b_avg
    gedit_overall = ge_b["average"]["overall"]

    text = rf"""\begin{{table*}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{llccccc}}
\toprule
Model state & Role & ImgEdit $n$ & ImgEdit $\uparrow$ & $\Delta$ & GEdit $n$ & GEdit $\uparrow$ \\
\midrule
\qwenedit baseline & reference & 737 & {fmt(im_b_avg, 4)} & -- & {gedit_n} & {fmt(gedit_overall, 4)} \\
\method diagnostic run & failure diagnosis & 737 & {fmt(im_f_avg, 4)} & {fmt(delta, 4)} & -- & -- \\
\bottomrule
\end{{tabular}}
\caption{{Validated external measurements for completed runs. The completed \method diagnostic run is not reported as the proposed final method; it is used to explain the failure mode that motivated the balanced no-rejected and object-focused strict recovery protocol. Recovery rows should be added only after their full external evaluations complete.}}
\label{{tab:main_results}}
\end{{table*}}
"""
    (TABLES / "main_results.tex").write_text(text, encoding="utf-8")


def build_process_trace() -> None:
    rows = []
    for r in range(1, 5):
        s = round_summary(r)
        counts = manifest_status_counts(r)
        rows.append((r, s, counts))

    lines = []
    for r, s, counts in rows:
        rejected = counts.get("rejected", 0)
        replay = counts.get("reconstruction_replay", 0)
        lines.append(
            f"{r} & {s['proposal_groups']} & {s['candidates']} & {s['accepted']} & "
            f"{pct(s['candidate_acceptance_rate'])} & {s['train_manifest_samples']} & "
            f"{rejected} & {replay} \\\\"
        )

    text = "\\begin{table*}[t]\n\\centering\n\\small\n"
    text += "\\setlength{\\tabcolsep}{4pt}\n"
    text += "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
    text += (
        "Round & Groups & Candidates & Accepted & Cand. acc. (\\%) & "
        "Cumulative SFT rows & Rejected rows & Replay rows \\\\\n"
    )
    text += "\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\n"
    text += "\\end{tabular}\n"
    text += (
        "\\caption{Process trace for the completed diagnostic self-evolution run. "
        "Acceptance remains non-trivial across rounds, but the cumulative SFT manifest "
        "also contains many rejected candidates. This separation is important: internal "
        "candidate discovery was feasible, while the training target construction was not.}\n"
    )
    text += "\\label{tab:process_trace}\n\\end{table*}\n"
    (TABLES / "process_trace.tex").write_text(text, encoding="utf-8")


def build_imgedit_type_deltas() -> None:
    base = load_json(IMGEDIT_BASELINE)["metrics"]["type_scores"]
    failed = load_json(IMGEDIT_FAILED)["metrics"]["type_scores"]
    rows = []
    for typ in sorted(base):
        rows.append((typ, base[typ], failed[typ], failed[typ] - base[typ]))
    rows.sort(key=lambda x: x[3])

    lines = [
        f"{esc(t)} & {fmt(b, 2)} & {fmt(f, 2)} & {fmt(d, 2)} \\\\"
        for t, b, f, d in rows
    ]
    text = "\\begin{table}[t]\n\\centering\n\\small\n"
    text += "\\begin{tabular}{lrrr}\n\\toprule\n"
    text += "ImgEdit type & Baseline & Diagnostic run & $\\Delta$ \\\\\n\\midrule\n"
    text += "\n".join(lines)
    text += "\n\\bottomrule\n\\end{tabular}\n"
    text += (
        "\\caption{Full ImgEdit type-level deltas for the completed diagnostic run. "
        "The largest regression is on adjustment edits, followed by removal, composition, "
        "and action edits, which points to preservation-sensitive failure rather than a "
        "uniform quality drop.}\n"
    )
    text += "\\label{tab:imgedit_type_deltas}\n\\end{table}\n"
    (TABLES / "imgedit_type_deltas.tex").write_text(text, encoding="utf-8")


def build_selection_diagnosis() -> None:
    rows = read_jsonl(FINAL_RUN / "round_04" / "train_manifest.jsonl")
    status_counts = Counter(status_of_manifest_row(row) for row in rows)
    total = sum(status_counts.values())
    status_lines = []
    for key, label in [
        ("accepted", "Accepted edit"),
        ("rejected", "Rejected edit"),
        ("reconstruction_replay", "Source reconstruction replay"),
    ]:
        count = status_counts.get(key, 0)
        status_lines.append(f"{label} & {count} & {pct(count / total)} \\\\")

    family_counts: dict[str, Counter] = defaultdict(Counter)
    weight_sum: dict[str, float] = defaultdict(float)
    for row in rows:
        fam = family_of_row(row)
        status = status_of_manifest_row(row)
        if status in {"accepted", "rejected"}:
            family_counts[fam][status] += 1
            weight_sum[fam] += float(row.get("sample_weight", 0.0) or 0.0)

    ranked = sorted(
        family_counts.items(),
        key=lambda kv: kv[1].get("accepted", 0) + kv[1].get("rejected", 0),
        reverse=True,
    )[:6]
    family_lines = []
    for fam, c in ranked:
        accepted = c.get("accepted", 0)
        rejected = c.get("rejected", 0)
        denom = accepted + rejected
        rej_rate = rejected / denom if denom else 0.0
        family_lines.append(
            f"{esc(fam)} & {accepted} & {rejected} & {pct(rej_rate)} & {fmt(weight_sum[fam], 1)} \\\\"
        )

    text = "\\begin{table*}[t]\n\\centering\n\\small\n"
    text += "\\setlength{\\tabcolsep}{5pt}\n"
    text += "\\begin{tabular}{lrr@{\\hspace{1.5em}}lrrrr}\n\\toprule\n"
    text += (
        "\\multicolumn{3}{c}{Cumulative SFT rows} & "
        "\\multicolumn{5}{c}{Largest edit families in candidate SFT rows} \\\\\n"
    )
    text += "\\cmidrule(r){1-3}\\cmidrule(l){4-8}\n"
    text += "Row type & Count & Share (\\%) & Family & Accepted & Rejected & Rej. (\\%) & Weight \\\\\n"
    text += "\\midrule\n"
    max_len = max(len(status_lines), len(family_lines))
    for i in range(max_len):
        left = status_lines[i].rstrip("\\") if i < len(status_lines) else " &  & "
        right = family_lines[i] if i < len(family_lines) else " &  &  &  &  \\\\"
        text += f"{left} & {right}\n"
    text += "\\bottomrule\n\\end{tabular}\n"
    text += (
        "\\caption{Training-target diagnosis for the completed diagnostic run. "
        "The final cumulative SFT manifest contains more rejected edits than accepted edits. "
        "Removal and replacement examples are especially sparse among accepted targets, "
        "which explains why external regressions concentrate on preservation-sensitive edits.}\n"
    )
    text += "\\label{tab:selection_diagnosis}\n\\end{table*}\n"
    (TABLES / "selection_diagnosis.tex").write_text(text, encoding="utf-8")


def gate_reason(row: dict[str, Any]) -> str:
    evaluator = row.get("evaluator") or row.get("solver") or {}
    comps = evaluator.get("component_scores") or {}
    sig = evaluator.get("signals") or {}

    feasible = float(sig.get("feasible", 0.0) or 0.0)
    accepted_by_ranker = float(sig.get("accepted_by_ranker", 0.0) or 0.0)
    if feasible >= 0.5 and accepted_by_ranker < 0.5:
        return "Feasible but lower ranked"

    required = float(comps.get("rubric_required_after", sig.get("rubric_required_after_score", 1.0)) or 0.0)
    required_thr = float(sig.get("rubric_required_threshold", 0.45) or 0.45)
    if required < required_thr:
        return "Missing required after-state"

    forbidden = float(
        comps.get("rubric_forbidden_after_absent", sig.get("rubric_forbidden_after_absent_score", 1.0))
        or 0.0
    )
    forbidden_thr = float(sig.get("rubric_forbidden_threshold", 0.42) or 0.42)
    forbidden_supported = float(sig.get("rubric_forbidden_after_supported", 1.0) or 0.0)
    if forbidden_supported > 0.0 and forbidden < forbidden_thr:
        return "Forbidden old-state remains"

    preservation = float(comps.get("rubric_preservation", sig.get("rubric_preservation_score", 1.0)) or 0.0)
    preservation_thr = float(sig.get("rubric_preservation_threshold", 0.30) or 0.30)
    cepr_pres = float(comps.get("cepr_preservation", sig.get("cepr_preservation_score", 1.0)) or 0.0)
    cepr_pres_thr = float(sig.get("cepr_preservation_threshold", 0.20) or 0.20)
    if preservation < preservation_thr or cepr_pres < cepr_pres_thr:
        return "Preservation failure"

    validity = float(comps.get("cepr_validity", sig.get("cepr_validity_score", 1.0)) or 0.0)
    validity_thr = float(sig.get("cepr_validity_threshold", 0.50) or 0.50)
    if validity < validity_thr:
        return "Latent validity failure"

    edit = float(comps.get("cepr_edit_specificity", 1.0) or 0.0)
    edit_thr = float(sig.get("cepr_edit_threshold", 0.40) or 0.40)
    if edit < edit_thr:
        return "Weak edit specificity"

    reward = float(comps.get("cepr_reward", comps.get("rubric_cepr_reward", 1.0)) or 0.0)
    reward_thr = float(sig.get("cepr_reward_threshold", sig.get("rubric_reward_threshold", 0.30)) or 0.30)
    if reward < reward_thr:
        return "Below reward threshold"

    return "Other rejection"


def build_gate_failure_breakdown() -> None:
    rejected: list[dict[str, Any]] = []
    accepted = 0
    for r in range(1, 5):
        rows = read_jsonl(FINAL_RUN / f"round_{r:02d}" / "proposals.jsonl")
        for row in rows:
            status = str(row.get("status", "")).lower()
            if status == "accepted":
                accepted += 1
            elif status == "rejected":
                rejected.append(row)

    reason_counts = Counter(gate_reason(row) for row in rejected)
    total_rejected = sum(reason_counts.values())
    total_candidates = accepted + total_rejected
    lines = []
    for reason, count in reason_counts.most_common():
        lines.append(f"{esc(reason)} & {count} & {pct(count / total_candidates)} \\\\")

    text = "\\begin{table}[t]\n\\centering\n\\small\n"
    text += "\\begin{tabular}{lrr}\n\\toprule\n"
    text += "Primary candidate outcome & Count & Share of candidates (\\%) \\\\\n\\midrule\n"
    text += f"Accepted candidate & {accepted} & {pct(accepted / total_candidates)} \\\\\n"
    text += "\\midrule\n"
    text += "\n".join(lines)
    text += "\n\\bottomrule\n\\end{tabular}\n"
    text += (
        "\\caption{Primary rejection reasons across all candidates in the completed diagnostic run. "
        "A candidate can fail multiple checks; the table assigns a single primary reason, "
        "separating gate-passing candidates that lost within-group ranking from candidates "
        "that failed the edit contract.}\n"
    )
    text += "\\label{tab:gate_failure_breakdown}\n\\end{table}\n"
    (TABLES / "gate_failure_breakdown.tex").write_text(text, encoding="utf-8")


def build_canary_table() -> None:
    rows = []
    for name, path in [
        ("64-record pilot", CANARY_32),
        ("512-run, offset 0", CANARY_64_O0),
        ("512-run, offset 64", CANARY_64_O64),
    ]:
        d = load_json(path)
        rows.append(
            (
                name,
                d["count"],
                d["baseline_mean"],
                d["candidate_mean"],
                d["mean_delta"],
                d["wins"],
                d["ties"],
                d["losses"],
            )
        )

    lines = [
        f"{esc(name)} & {n} & {fmt(b, 4)} & {fmt(c, 4)} & {fmt(d, 4)} & {w}/{t}/{l} \\\\"
        for name, n, b, c, d, w, t, l in rows
    ]
    text = "\\begin{table}[t]\n\\centering\n\\small\n"
    text += "\\begin{tabular}{lrrrrl}\n\\toprule\n"
    text += "Subset & $n$ & Baseline & Candidate & $\\Delta$ & W/T/L \\\\\n\\midrule\n"
    text += "\n".join(lines)
    text += "\n\\bottomrule\n\\end{tabular}\n"
    text += (
        "\\caption{ImgEdit canary checks used for debugging. These subsets are not used as "
        "the main external evidence, but they reveal that the small pilot gain did not persist "
        "for the larger diagnostic checkpoint.}\n"
    )
    text += "\\label{tab:canary_diagnostics}\n\\end{table}\n"
    (TABLES / "canary_diagnostics.tex").write_text(text, encoding="utf-8")


def build_control_table() -> None:
    text = r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lp{0.28\linewidth}p{0.28\linewidth}p{0.20\linewidth}}
\toprule
Control & Reviewer question & Diagnostic motivation & Evaluation endpoint \\
\midrule
Balanced no-rejected SFT & Is the reward selector contradicted by the SFT targets? & The diagnostic run trained on more rejected than accepted candidate rows. & Full ImgEdit/GEdit plus manifest audit \\
Object-focused strict recovery & Are regressions driven by weak removal, replacement, and preservation? & Removal and replacement had very few accepted targets and high rejection rates. & Remove/replace subsets plus full ImgEdit \\
One-shot pseudo-training & Are rounds necessary, or is this just pseudo-labeling? & Round-based collection must beat a matched one-pass pseudo-data baseline. & Matched candidate and step budget \\
Scalar/additive reward & Do hard gates matter beyond reward shaping? & Additive rewards can compensate for a missing edit with preservation or quality. & Full ImgEdit plus gate analysis \\
No forbidden-after gate & Is old-state suppression necessary for remove/replace edits? & Forbidden old-state evidence is the largest candidate rejection source. & Remove/replace subset analysis \\
\bottomrule
\end{tabular}
\caption{Controls needed to answer the strongest reviewer objections. Each control follows from a specific failure observed in the completed diagnostic run rather than from an open-ended hyperparameter search.}
\label{tab:ablations}
\end{table*}
"""
    (TABLES / "ablations.tex").write_text(text, encoding="utf-8")


def build_qualitative_plan() -> None:
    notes_dir = ROOT / "docs/BMVC_Overleaf_Local_Archive/working_notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_json(IMGEDIT_BASELINE_AVG)
    diagnostic = load_json(IMGEDIT_FAILED_AVG)
    metadata = load_json(IMGEDIT_META)

    candidates = []
    for key in sorted(set(baseline) & set(diagnostic), key=lambda k: int(k)):
        b = float(baseline[key])
        d = float(diagnostic[key])
        meta = metadata.get(key, {})
        candidates.append(
            {
                "key": key,
                "baseline": b,
                "diagnostic": d,
                "delta": d - b,
                "edit_type": meta.get("edit_type", "unknown"),
                "prompt": meta.get("prompt", ""),
                "source_id": meta.get("id", ""),
                "baseline_image": IMGEDIT_BASELINE_IMAGES / f"{key}.png",
                "diagnostic_image": IMGEDIT_FAILED_IMAGES / f"{key}.png",
            }
        )

    losses = sorted(
        [row for row in candidates if row["baseline"] >= 4.5 and row["delta"] < -0.5],
        key=lambda row: row["delta"],
    )[:8]
    sensitive_types = {"adjust", "remove", "replace", "action", "compose"}
    sensitive_losses = sorted(
        [
            row
            for row in candidates
            if row["edit_type"] in sensitive_types and row["baseline"] >= 4.0 and row["delta"] < -0.5
        ],
        key=lambda row: row["delta"],
    )[:10]
    wins = sorted(
        [row for row in candidates if row["baseline"] <= 4.0 and row["delta"] > 0.5],
        key=lambda row: row["delta"],
        reverse=True,
    )[:6]

    def table(rows: list[dict[str, Any]]) -> str:
        lines = [
            "| Key | Type | Baseline | Diagnostic | Delta | Instruction | Baseline image | Diagnostic image |",
            "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
        for row in rows:
            lines.append(
                "| {key} | {edit_type} | {baseline:.2f} | {diagnostic:.2f} | {delta:.2f} | "
                "{prompt} | `{baseline_image}` | `{diagnostic_image}` |".format(**row)
            )
        return "\n".join(lines)

    text = """# Qualitative Grid Plan

This note selects candidate rows for the BMVC qualitative figure from completed ImgEdit evidence. Use these rows to build a source/instruction/baseline/diagnostic/recovered grid once recovered checkpoint outputs are available.

Selection logic:

- Failure rows require a strong baseline score, at least 4.5, and a diagnostic drop below the baseline by more than 0.5.
- Recovery-opportunity rows require a weaker baseline score, at most 4.0, and a diagnostic improvement above the baseline by more than 0.5.
- These are selection candidates, not final qualitative claims. Inspect the images visually before placing them in the paper.

## High-Ceiling Failures

""" + table(losses) + """

## Preservation-Sensitive Failures

""" + table(sensitive_losses) + """

## Recovery-Opportunity Wins

""" + table(wins) + """

## Recommended Final Grid

Use six rows if space permits:

1. Two preservation-sensitive failures from `adjust` or `action`.
2. Two object-sensitive failures from `remove` or `replace`.
3. One diagnostic win where the baseline is weak.
4. One recovered-checkpoint success once the new run finishes.

Each caption should answer three questions: did the required after-state appear, did forbidden old-state evidence remain, and did unrelated source content stay stable?
"""
    (notes_dir / "QUALITATIVE_GRID_PLAN.md").write_text(text, encoding="utf-8")


def build_diagnostic_questions() -> None:
    text = r"""\begin{table}[t]
\centering
\small
\begin{tabular}{lp{0.57\linewidth}}
\toprule
Question & Evidence used in this study \\
\midrule
Does self-training transfer? & Full ImgEdit and GEdit scores against the frozen \qwenedit baseline, not canary subsets. \\
What broke in the failed run? & Type-level external deltas plus the SFT manifest composition in Tables~\ref{tab:imgedit_type_deltas} and~\ref{tab:selection_diagnosis}. \\
Are accepted samples actually clean? & Gate-level rejection analysis and qualitative source/instruction/output grids. \\
Are rounds justified? & Matched one-shot pseudo-training with the same generated-candidate budget. \\
Do hard gates matter? & Scalar/additive CEPR and gate-removal controls, especially on remove and replace edits. \\
\bottomrule
\end{tabular}
\caption{Diagnostic questions that structure the empirical section. Each question maps to evidence that a reviewer can inspect independently.}
\label{tab:diagnostic_questions}
\end{table}
"""
    (TABLES / "diagnostic_questions.tex").write_text(text, encoding="utf-8")


def try_build_figures() -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on environment
        print(f"matplotlib unavailable; skipped figures: {exc}")
        return

    FIGURES.mkdir(parents=True, exist_ok=True)

    # Process trace.
    rounds = [1, 2, 3, 4]
    summaries = [round_summary(r) for r in rounds]
    cand_acc = [100.0 * s["candidate_acceptance_rate"] for s in summaries]
    train_rows = [s["train_manifest_samples"] for s in summaries]
    rejected_rows = [manifest_status_counts(r).get("rejected", 0) for r in rounds]

    fig, ax1 = plt.subplots(figsize=(5.4, 3.0))
    ax1.plot(rounds, cand_acc, marker="o", color="#0072B2", linewidth=2, label="Candidate acceptance")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Candidate acceptance (%)")
    ax1.set_xticks(rounds)
    ax1.set_ylim(0, max(cand_acc) + 10)
    ax2 = ax1.twinx()
    ax2.plot(rounds, train_rows, marker="s", color="#009E73", linewidth=2, label="SFT rows")
    ax2.plot(rounds, rejected_rows, marker="^", color="#D55E00", linewidth=2, label="Rejected rows in SFT")
    ax2.set_ylabel("Cumulative rows")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left", fontsize=8, frameon=False)
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "process_trace.pdf")
    plt.close(fig)

    # ImgEdit type deltas.
    base = load_json(IMGEDIT_BASELINE)["metrics"]["type_scores"]
    failed = load_json(IMGEDIT_FAILED)["metrics"]["type_scores"]
    rows = sorted(((t, failed[t] - base[t]) for t in base), key=lambda x: x[1])
    labels = [r[0] for r in rows]
    deltas = [r[1] for r in rows]
    colors = ["#D55E00" if d < 0 else "#009E73" for d in deltas]
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    ax.barh(labels, deltas, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ImgEdit score delta vs baseline")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "imgedit_type_deltas.pdf")
    plt.close(fig)

    # Manifest family composition.
    rows = read_jsonl(FINAL_RUN / "round_04" / "train_manifest.jsonl")
    fam_counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        status = status_of_manifest_row(row)
        if status in {"accepted", "rejected"}:
            fam_counts[family_of_row(row)][status] += 1
    ranked = sorted(
        fam_counts.items(),
        key=lambda kv: kv[1].get("accepted", 0) + kv[1].get("rejected", 0),
        reverse=True,
    )[:7]
    fams = [r[0] for r in ranked]
    accepted = [r[1].get("accepted", 0) for r in ranked]
    rejected = [r[1].get("rejected", 0) for r in ranked]
    fig, ax = plt.subplots(figsize=(5.7, 3.2))
    ax.barh(fams, accepted, color="#009E73", label="Accepted")
    ax.barh(fams, rejected, left=accepted, color="#D55E00", label="Rejected")
    ax.set_xlabel("Cumulative SFT rows")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "manifest_family_composition.pdf")
    plt.close(fig)

    # Composite paper figure.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))

    ax = axes[0]
    ax.plot(rounds, cand_acc, marker="o", color="#0072B2", linewidth=1.8)
    ax.set_title("(a) Process trace", fontsize=9)
    ax.set_xlabel("Round", fontsize=8)
    ax.set_ylabel("Cand. acc. (%)", fontsize=8)
    ax.set_xticks(rounds)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.25)
    axr = ax.twinx()
    axr.plot(rounds, rejected_rows, marker="^", color="#D55E00", linewidth=1.6)
    axr.set_ylabel("Rejected SFT rows", fontsize=8)
    axr.tick_params(labelsize=7)

    ax = axes[1]
    ax.barh(fams, accepted, color="#009E73", label="Accepted")
    ax.barh(fams, rejected, left=accepted, color="#D55E00", label="Rejected")
    ax.set_title("(b) Manifest families", fontsize=9)
    ax.set_xlabel("Rows", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    ax = axes[2]
    ax.barh(labels, deltas, color=colors)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_title("(c) ImgEdit deltas", fontsize=9)
    ax.set_xlabel("Score delta", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout(w_pad=1.0)
    fig.savefig(FIGURES / "diagnostic_summary.pdf")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    build_main_results()
    build_process_trace()
    build_imgedit_type_deltas()
    build_selection_diagnosis()
    build_gate_failure_breakdown()
    build_canary_table()
    build_control_table()
    build_diagnostic_questions()
    build_qualitative_plan()
    try_build_figures()
    print("Wrote BMVC diagnostic tables and figures.")


if __name__ == "__main__":
    main()
