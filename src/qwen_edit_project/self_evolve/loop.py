from __future__ import annotations

import json
import logging
import math
import gc
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from qwen_edit_project.self_evolve.backends import (
    DifficultyController,
    QwenEditEditor,
    build_editor,
    build_evaluator,
    build_proposer,
)
from qwen_edit_project.self_evolve.data import load_unlabeled_records
from qwen_edit_project.self_evolve.proposer_training import (
    build_proposer_training_records,
    write_proposer_training_jsonl,
)
from qwen_edit_project.self_evolve.types import (
    AcceptedSample,
    EditProposal,
    EvaluationResult,
    ProposalDefinition,
    UnlabeledImageRecord,
)
from qwen_edit_project.train.launch_train import build_train_command
from qwen_edit_project.utils.commands import run_and_tee, shell_join
from qwen_edit_project.utils.config import load_yaml_config, save_json
from qwen_edit_project.utils.paths import ensure_dir, relative_to_repo, resolve_path
from qwen_edit_project.utils.run_metadata import base_run_metadata, utc_timestamp


def write_jsonl(items: list[dict[str, Any]], path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")
    return path


def append_jsonl(item: dict[str, Any], path: Path) -> Path:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=True) + "\n")
        handle.flush()
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
        for line_number, line in enumerate(lines, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    if line_number == len(lines):
                        break
                    raise ValueError(f"Invalid JSONL row in {path} at line {line_number}") from exc
    return rows


def discover_latest_checkpoint(directory: Path) -> Path | None:
    candidates = sorted(directory.rglob("*.safetensors"), key=lambda path: path.stat().st_mtime_ns)
    if not candidates:
        return None
    return candidates[-1]


def discover_latest_step_checkpoint(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    checkpoints = []
    for path in directory.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        try:
            step = int(path.name.split("-")[-1])
        except ValueError:
            continue
        checkpoints.append((step, path))
    if not checkpoints:
        return None
    return sorted(checkpoints, key=lambda item: item[0])[-1][1]


def training_checkpoint_backend(training_cfg: dict[str, Any], train_config: dict[str, Any]) -> str:
    return str(
        train_config.get("output", {}).get("checkpoint_backend")
        or training_cfg.get("trained_checkpoint_backend", "diffsynth")
    )


def _append_flag(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    command.extend([flag, str(value)])


def _finite_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _resolve_num_rounds(value: Any, record_count: int, max_records_per_round: int) -> int:
    if isinstance(value, str) and value.lower() in {"auto", "all", "cover_dataset"}:
        if record_count <= 0:
            return 0
        return max(1, math.ceil(record_count / max(1, max_records_per_round)))
    return int(value)


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


EDITOR_TRAINING_OVERRIDE_KEYS = {
    "learning_rate",
    "num_epochs",
    "max_train_steps",
    "train_batch_size",
    "gradient_accumulation_steps",
    "checkpointing_steps",
    "checkpoints_total_limit",
    "resume_from_checkpoint",
    "mixed_precision",
    "use_8bit_adam",
    "offload",
}


class SelfEvolveRunner:
    def __init__(self, config: dict[str, Any], dry_run: bool = False, limit: int | None = None):
        self.config = config
        self.dry_run = dry_run or bool(config.get("runtime", {}).get("dry_run", False))
        dataset_limit = config.get("dataset", {}).get("limit")
        self.limit = limit if limit is not None else dataset_limit
        self.records = load_unlabeled_records(config["dataset"], limit=self.limit)
        self.output_root = ensure_dir(resolve_path(config["output"]["root_dir"]))
        self.logger = self._build_logger()
        self.proposer = build_proposer(config["proposer"])
        self.editor = build_editor(config["editor"])
        evaluator_config = config.get("evaluator", config.get("solver"))
        if evaluator_config is None:
            raise ValueError("self-evolve config requires an evaluator: section; solver: is still accepted as an alias")
        self.evaluator = build_evaluator(evaluator_config)
        self.solver = self.evaluator  # Backward-compatible alias for older code paths.
        curriculum = config["curriculum"]
        self.difficulty_controller = DifficultyController(
            initial_level=int(curriculum.get("initial_level", 1)),
            min_level=int(curriculum.get("min_level", 1)),
            max_level=int(curriculum.get("max_level", 3)),
            promote_at=float(curriculum.get("promote_at", 0.75)),
            demote_at=float(curriculum.get("demote_at", 0.45)),
        )
        self.record_by_key = {record.key: record for record in self.records}

    def _build_logger(self) -> logging.Logger:
        ensure_dir(self.output_root)
        logger = logging.getLogger(f"qwen_edit_project.self_evolve.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(self.output_root / "self_evolve.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.handlers = [stream_handler, file_handler]
        return logger

    def _release_resident_models(self, reason: str) -> None:
        released: list[str] = []
        for name in ("editor", "proposer", "evaluator"):
            component = getattr(self, name, None)
            release = getattr(component, "release_memory", None)
            if callable(release):
                release()
                released.append(name)
        if released:
            self.logger.info("Released resident model memory before %s: %s.", reason, ", ".join(released))

    def _release_component_model(self, component_name: str, reason: str) -> None:
        component = getattr(self, component_name, None)
        release = getattr(component, "release_memory", None)
        if callable(release):
            release()
            self.logger.info("Released %s model memory before %s.", component_name, reason)

    def _candidate_payload(
        self,
        record: UnlabeledImageRecord,
        proposal: EditProposal,
        evaluation_result: EvaluationResult | None,
        image_path: Path | None,
        status: str,
        candidate_index: int = 0,
        group_id: str | None = None,
        distractors: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "group_id": group_id,
            "candidate_index": candidate_index,
            "record_key": record.key,
            "image_path": relative_to_repo(record.image_path),
            "caption": record.caption,
            "proposal": {
                "round_index": proposal.round_index,
                "proposal_index": proposal.proposal_index,
                "operation_id": proposal.definition.operation_id,
                "family": proposal.definition.family,
                "difficulty": proposal.definition.difficulty,
                "difficulty_level": proposal.difficulty_level,
                "instruction": proposal.instruction,
                "scope": proposal.definition.scope,
                "metric": proposal.definition.metric,
                "direction": proposal.definition.direction,
                "target": proposal.definition.target,
                "expected_changed_fraction": list(proposal.definition.expected_changed_fraction),
                "verifier": proposal.definition.verifier,
                "inverse_operation_id": proposal.definition.inverse_operation_id,
                "structured_edit": proposal.structured_edit,
            },
            "distractors": distractors or [],
            "status": status,
            "edited_image_path": relative_to_repo(image_path) if image_path is not None else None,
        }
        if evaluation_result is not None:
            evaluator_payload = {
                "global_score": evaluation_result.global_score,
                "local_score": evaluation_result.local_score,
                "total_score": evaluation_result.total_score,
                "accepted": evaluation_result.accepted,
                "component_scores": evaluation_result.component_scores,
                "signals": evaluation_result.signals,
            }
            payload["evaluator"] = evaluator_payload
            payload["solver"] = evaluator_payload
        return payload

    @staticmethod
    def _candidate_key(payload: dict[str, Any]) -> tuple[str, int]:
        return str(payload.get("group_id", "")), int(payload.get("candidate_index", -1))

    @staticmethod
    def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            group_id = str(row.get("group_id") or "")
            if group_id:
                grouped[group_id].append(row)
        for values in grouped.values():
            values.sort(key=lambda row: int(row.get("candidate_index", 0)))
        return dict(grouped)

    def _round_records_for_round(
        self,
        round_index: int,
        max_records_per_round: int,
    ) -> tuple[list[UnlabeledImageRecord], dict[str, Any]]:
        if not self.records:
            return [], {
                "record_schedule": "empty",
                "record_start_index": 0,
                "record_count": 0,
                "record_indices": [],
                "record_wraparound": False,
            }
        count = min(max(1, max_records_per_round), len(self.records))
        schedule = str(self.config.get("curriculum", {}).get("record_schedule", "sequential_shards"))
        if schedule in {"fixed_first_slice", "first_slice", "fixed"}:
            indices = list(range(count))
        elif schedule in {"sequential_shards", "sequential", "sharded"}:
            start = ((round_index - 1) * count) % len(self.records)
            indices = [(start + offset) % len(self.records) for offset in range(count)]
        else:
            raise ValueError(
                "Unsupported curriculum.record_schedule. Use 'sequential_shards' or 'fixed_first_slice'."
            )
        return [self.records[index] for index in indices], {
            "record_schedule": schedule,
            "record_start_index": indices[0] if indices else 0,
            "record_count": len(indices),
            "record_indices": indices,
            "record_wraparound": bool(indices and indices[-1] < indices[0]),
        }

    @staticmethod
    def _is_completed_group(rows: list[dict[str, Any]], samples_per_proposal: int) -> bool:
        if len(rows) < samples_per_proposal:
            return False
        statuses = {str(row.get("status", "")) for row in rows[:samples_per_proposal]}
        return statuses.issubset({"accepted", "rejected"})

    @staticmethod
    def _evaluation_from_payload(payload: dict[str, Any]) -> EvaluationResult | None:
        data = payload.get("evaluator") or payload.get("solver")
        if not isinstance(data, dict):
            return None
        return EvaluationResult(
            global_score=float(data.get("global_score", 0.0)),
            local_score=float(data.get("local_score", 0.0)),
            total_score=float(data.get("total_score", 0.0)),
            accepted=bool(data.get("accepted", False)),
            signals=dict(data.get("signals", {})),
            component_scores=dict(data.get("component_scores", {})),
        )

    @staticmethod
    def _proposal_from_payload(payload: dict[str, Any]) -> EditProposal:
        proposal = payload.get("proposal", {})
        expected_range = proposal.get("expected_changed_fraction") or (0.0, 1.0)
        if len(expected_range) != 2:
            expected_range = (0.0, 1.0)
        definition = ProposalDefinition(
            operation_id=str(proposal.get("operation_id", "unknown")),
            instruction=str(proposal.get("instruction", "")),
            family=str(proposal.get("family", "unknown")),
            difficulty=int(proposal.get("difficulty", 1)),
            scope=str(proposal.get("scope", "global")),
            metric=str(proposal.get("metric", "internal_prompt_gain")),
            direction=str(proposal.get("direction", "increase")),
            target=float(proposal.get("target", 0.0)),
            expected_changed_fraction=(float(expected_range[0]), float(expected_range[1])),
            verifier=str(proposal.get("verifier", "internal")),
            inverse_operation_id=proposal.get("inverse_operation_id"),
        )
        return EditProposal(
            record_key=str(payload.get("record_key", proposal.get("record_key", ""))),
            round_index=int(proposal.get("round_index", 0)),
            proposal_index=int(proposal.get("proposal_index", 0)),
            definition=definition,
            difficulty_level=int(proposal.get("difficulty_level", definition.difficulty)),
            instruction=definition.instruction,
            structured_edit=dict(proposal.get("structured_edit", {})),
        )

    def _accepted_sample_from_payload(self, payload: dict[str, Any]) -> AcceptedSample | None:
        if payload.get("status") != "accepted":
            return None
        evaluation = self._evaluation_from_payload(payload)
        edited_path_raw = payload.get("edited_image_path")
        if evaluation is None or not edited_path_raw:
            return None
        edited_path = resolve_path(str(edited_path_raw))
        if edited_path is None or not edited_path.exists():
            return None
        record_key = str(payload.get("record_key", ""))
        record = self.record_by_key.get(record_key)
        if record is None:
            image_path = resolve_path(str(payload.get("image_path", "")))
            if image_path is None:
                return None
            record = UnlabeledImageRecord(
                key=record_key,
                image_path=image_path,
                caption=payload.get("caption"),
                metadata={},
            )
        return AcceptedSample(
            record=record,
            proposal=self._proposal_from_payload(payload),
            edited_image_path=edited_path,
            evaluation_result=evaluation,
            candidate_index=int(payload.get("candidate_index", 0)),
        )

    def _accepted_samples_from_payloads(self, rows: list[dict[str, Any]]) -> list[AcceptedSample]:
        samples = []
        seen = set()
        for row in rows:
            key = self._candidate_key(row)
            if key in seen:
                continue
            seen.add(key)
            sample = self._accepted_sample_from_payload(row)
            if sample is not None:
                samples.append(sample)
        return samples

    @staticmethod
    def _proposal_plan_payload(
        record_index: int,
        record: UnlabeledImageRecord,
        proposal: EditProposal,
        group_id: str,
    ) -> dict[str, Any]:
        return {
            "group_id": group_id,
            "record_index": record_index,
            "record_key": record.key,
            "image_path": relative_to_repo(record.image_path),
            "caption": record.caption,
            "proposal": {
                "round_index": proposal.round_index,
                "proposal_index": proposal.proposal_index,
                "operation_id": proposal.definition.operation_id,
                "family": proposal.definition.family,
                "difficulty": proposal.definition.difficulty,
                "difficulty_level": proposal.difficulty_level,
                "instruction": proposal.instruction,
                "scope": proposal.definition.scope,
                "metric": proposal.definition.metric,
                "direction": proposal.definition.direction,
                "target": proposal.definition.target,
                "expected_changed_fraction": list(proposal.definition.expected_changed_fraction),
                "verifier": proposal.definition.verifier,
                "inverse_operation_id": proposal.definition.inverse_operation_id,
                "structured_edit": proposal.structured_edit,
            },
        }

    def _build_evaluator_exports(
        self,
        candidate_payloads: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        evaluator_records: list[dict[str, Any]] = []
        preference_records: list[dict[str, Any]] = []
        grouped = self._group_rows(candidate_payloads)
        for group_id, rows in grouped.items():
            accepted_payloads = [row for row in rows if row.get("status") == "accepted"]
            rejected_payloads = [row for row in rows if row.get("status") == "rejected"]
            for row in rows:
                evaluation = row.get("evaluator") or row.get("solver")
                if not isinstance(evaluation, dict):
                    continue
                proposal = row.get("proposal", {})
                signals = evaluation.get("signals", {})
                evaluator_records.append(
                    {
                        "type": "candidate",
                        "group_id": group_id,
                        "candidate_index": row.get("candidate_index"),
                        "record_key": row.get("record_key"),
                        "source_image": row.get("image_path"),
                        "edited_image": row.get("edited_image_path"),
                        "instruction": proposal.get("instruction"),
                        "operation_id": proposal.get("operation_id"),
                        "family": proposal.get("family"),
                        "verifier": proposal.get("verifier"),
                        "structured_edit": proposal.get("structured_edit", {}),
                        "distractors": row.get("distractors", []),
                        "accepted": evaluation.get("accepted", row.get("status") == "accepted"),
                        "feasible": bool(signals.get("feasible", float(row.get("status") == "accepted"))),
                        "rank": int(signals.get("feasible_rank", 0.0)),
                        "scores": {
                            "total_score": evaluation.get("total_score"),
                            "global_score": evaluation.get("global_score"),
                            "local_score": evaluation.get("local_score"),
                            "component_scores": evaluation.get("component_scores", {}),
                        },
                    }
                )
            for winner in accepted_payloads:
                for loser in rejected_payloads:
                    winner_evaluation = winner.get("evaluator", winner.get("solver", {}))
                    loser_evaluation = loser.get("evaluator", loser.get("solver", {}))
                    proposal = winner.get("proposal", {})
                    preference_records.append(
                        {
                            "type": "preference",
                            "group_id": group_id,
                            "winner_candidate_index": winner.get("candidate_index"),
                            "loser_candidate_index": loser.get("candidate_index"),
                            "record_key": winner.get("record_key"),
                            "source_image": winner.get("image_path"),
                            "instruction": proposal.get("instruction"),
                            "operation_id": proposal.get("operation_id"),
                            "family": proposal.get("family"),
                            "verifier": proposal.get("verifier"),
                            "structured_edit": proposal.get("structured_edit", {}),
                            "distractors": winner.get("distractors", []),
                            "winner_image": winner.get("edited_image_path"),
                            "loser_image": loser.get("edited_image_path"),
                            "winner_score": winner_evaluation.get("total_score"),
                            "loser_score": loser_evaluation.get("total_score"),
                        }
                    )
        return evaluator_records, preference_records

    def _write_progress(
        self,
        path: Path,
        payload: dict[str, Any],
    ) -> None:
        save_json({**base_run_metadata(), **payload}, path)

    def _restore_completed_round_state(self, round_summary: dict[str, Any]) -> None:
        if "next_difficulty_level" in round_summary:
            self.difficulty_controller.level = int(round_summary["next_difficulty_level"])

        training_result = round_summary.get("training")
        if isinstance(training_result, dict):
            latest_checkpoint = training_result.get("latest_checkpoint")
            continue_with_checkpoint = bool(training_result.get("continue_with_trained_checkpoint", True))
            if latest_checkpoint and continue_with_checkpoint:
                self.config.setdefault("training", {})["current_checkpoint_path"] = latest_checkpoint
                if isinstance(self.editor, QwenEditEditor):
                    self.editor.set_model_checkpoint(
                        str(latest_checkpoint),
                        model_type=str(training_result.get("trained_model_type", "lora")),
                        backend=str(
                            training_result.get(
                                "trained_checkpoint_backend",
                                self.config.get("training", {}).get("trained_checkpoint_backend", "diffsynth"),
                            )
                        ),
                    )

        proposer_training_result = round_summary.get("proposer_training")
        if (
            isinstance(proposer_training_result, dict)
            and proposer_training_result.get("status") == "completed"
            and proposer_training_result.get("output_dir")
        ):
            checkpoint_path = str(proposer_training_result["output_dir"])
            self.config.setdefault("proposer", {})["checkpoint_path"] = checkpoint_path
            if hasattr(self.proposer, "set_checkpoint_path"):
                self.proposer.set_checkpoint_path(checkpoint_path)

    def _candidate_training_weight(self, payload: dict[str, Any]) -> tuple[float, str]:
        allowed_verifiers = self.config.get("output", {}).get("train_verifiers")
        allowed_verifier_set = set(allowed_verifiers) if allowed_verifiers else None
        proposal = payload.get("proposal", {})
        if allowed_verifier_set is not None and proposal.get("verifier") not in allowed_verifier_set:
            return 0.0, "verifier_filtered"
        if not payload.get("edited_image_path"):
            return 0.0, "missing_edited_image"

        training_cfg = self.config.get("training", {})
        weighted_cfg = dict(training_cfg.get("weighted_sft", {}))
        weighted_enabled = bool(weighted_cfg.get("enabled", False))
        status = str(payload.get("status", ""))
        if status == "accepted":
            return float(weighted_cfg.get("accepted_weight", 1.0)), "accepted"
        if not weighted_enabled:
            return 0.0, "not_accepted"
        if not bool(weighted_cfg.get("include_rejected", True)):
            return 0.0, "rejected_disabled"
        if status != "rejected":
            return 0.0, f"status_{status}"

        evaluator_cfg = self.config.get("evaluator", self.config.get("solver", {})) or {}
        evaluation = payload.get("evaluator") or payload.get("solver") or {}
        component_scores = evaluation.get("component_scores", {}) if isinstance(evaluation, dict) else {}
        signals = evaluation.get("signals", {}) if isinstance(evaluation, dict) else {}

        raw_reward = _finite_float(
            component_scores.get("cepr_raw_reward"),
            _finite_float(evaluation.get("total_score"), _finite_float(payload.get("total_score"))),
        )
        semantic_edit = max(
            _finite_float(component_scores.get("cepr_semantic_edit")),
            _finite_float(component_scores.get("cepr_edit_specificity")),
            _finite_float(component_scores.get("cepr_edit")),
        )
        preservation = _finite_float(component_scores.get("cepr_preservation"), _finite_float(signals.get("preservation")))
        validity = _finite_float(component_scores.get("cepr_validity"), _finite_float(signals.get("validity")))
        taxonomy = _finite_float(component_scores.get("cepr_taxonomy"), 1.0)

        reward_threshold = float(weighted_cfg.get("target_reward", evaluator_cfg.get("reward_threshold", 0.30)))
        min_raw_reward = float(weighted_cfg.get("min_raw_reward", reward_threshold * 0.5))
        min_semantic_edit = float(
            weighted_cfg.get("min_semantic_edit", float(evaluator_cfg.get("edit_threshold", 0.45)) * 0.5)
        )
        min_preservation = float(
            weighted_cfg.get("min_preservation", evaluator_cfg.get("preservation_threshold", 0.20))
        )
        min_validity = float(weighted_cfg.get("min_validity", evaluator_cfg.get("validity_threshold", 0.50)))
        min_taxonomy = float(weighted_cfg.get("min_taxonomy", 0.0))

        if validity < min_validity:
            return 0.0, "validity_below_threshold"
        if preservation < min_preservation:
            return 0.0, "preservation_below_threshold"
        if semantic_edit < min_semantic_edit:
            return 0.0, "semantic_edit_below_threshold"
        if taxonomy < min_taxonomy:
            return 0.0, "taxonomy_below_threshold"
        if raw_reward < min_raw_reward:
            return 0.0, "reward_below_threshold"

        min_weight = float(weighted_cfg.get("min_rejected_weight", 0.05))
        max_weight = float(weighted_cfg.get("max_rejected_weight", 0.50))
        score_power = float(weighted_cfg.get("score_power", 1.0))
        normalized = _clamp((raw_reward - min_raw_reward) / max(reward_threshold - min_raw_reward, 1e-6))
        normalized = normalized**max(score_power, 1e-6)
        return min_weight + (max_weight - min_weight) * normalized, "cepr_weighted_rejected"

    def _training_records_from_payloads(
        self,
        candidate_payloads: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        records: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []
        for payload in candidate_payloads:
            weight, reason = self._candidate_training_weight(payload)
            evaluation = payload.get("evaluator") or payload.get("solver") or {}
            proposal = payload.get("proposal", {})
            audit_row = {
                "group_id": payload.get("group_id"),
                "candidate_index": payload.get("candidate_index"),
                "record_key": payload.get("record_key"),
                "status": payload.get("status"),
                "instruction": proposal.get("instruction"),
                "sample_weight": weight,
                "weight_reason": reason,
                "edited_image": payload.get("edited_image_path"),
                "scores": {
                    "total_score": evaluation.get("total_score") if isinstance(evaluation, dict) else None,
                    "component_scores": evaluation.get("component_scores", {}) if isinstance(evaluation, dict) else {},
                },
            }
            audit.append(audit_row)
            if weight <= 0:
                continue
            edited_path = resolve_path(str(payload.get("edited_image_path", "")))
            source_path = resolve_path(str(payload.get("image_path", "")))
            if edited_path is None or source_path is None or not edited_path.exists() or not source_path.exists():
                audit_row["sample_weight"] = 0.0
                audit_row["weight_reason"] = "missing_training_image"
                continue
            records.append(
                {
                    "prompt": str(proposal.get("instruction", "")),
                    "image": relative_to_repo(edited_path),
                    "edit_image": relative_to_repo(source_path),
                    "sample_weight": round(float(weight), 6),
                    "record_key": payload.get("record_key"),
                    "group_id": payload.get("group_id"),
                    "candidate_index": payload.get("candidate_index"),
                    "candidate_status": payload.get("status"),
                    "weight_reason": reason,
                    "operation_id": proposal.get("operation_id"),
                    "family": proposal.get("family"),
                    "structured_edit": proposal.get("structured_edit", {}),
                    "scores": audit_row["scores"],
                }
            )
        included = len(records)
        weight_sum = sum(float(record.get("sample_weight", 1.0)) for record in records)
        accepted_included = sum(1 for record in records if record.get("candidate_status") == "accepted")
        rejected_included = sum(1 for record in records if record.get("candidate_status") == "rejected")
        summary = {
            "candidates": len(candidate_payloads),
            "included": included,
            "accepted_included": accepted_included,
            "rejected_included": rejected_included,
            "weight_sum": weight_sum,
            "avg_weight": weight_sum / max(included, 1),
        }
        return records, audit, summary

    def _write_manifest_records(
        self,
        training_records: list[dict[str, Any]],
        manifest_path: Path,
    ) -> tuple[Path, int, float]:
        training_cfg = self.config.get("training", {})
        replay_ratio = float(training_cfg.get("reconstruction_replay_ratio", 0.0))
        replay_weight = float(training_cfg.get("reconstruction_replay_weight", 0.50))
        replay_prompt = str(
            training_cfg.get(
                "reconstruction_replay_prompt",
                "Reconstruct the input image exactly. Preserve all content, layout, colors, and text.",
            )
        )
        manifest_records = list(training_records)
        replay_source_paths: list[Path] = []
        for record in training_records:
            source_path = resolve_path(str(record.get("edit_image", "")))
            if source_path is not None:
                replay_source_paths.append(source_path)
        if replay_ratio > 0 and manifest_records and replay_source_paths:
            replay_count = max(1, round(len(manifest_records) * replay_ratio))
            unique_sources = list(dict.fromkeys(replay_source_paths))
            for index in range(replay_count):
                source_path = unique_sources[index % len(unique_sources)]
                manifest_records.append(
                    {
                        "prompt": replay_prompt,
                        "image": relative_to_repo(source_path),
                        "edit_image": relative_to_repo(source_path),
                        "sample_weight": replay_weight,
                        "candidate_status": "reconstruction_replay",
                    }
                )
        save_json(manifest_records, manifest_path)
        write_jsonl(manifest_records, manifest_path.with_suffix(".jsonl"))
        weight_sum = sum(float(record.get("sample_weight", 1.0)) for record in manifest_records)
        return manifest_path, len(manifest_records), weight_sum

    def _run_training_round(self, round_index: int, round_dir: Path, manifest_path: Path) -> dict[str, Any] | None:
        training_cfg = self.config.get("training", {})
        if training_cfg.get("trigger", "emit_only") != "launch":
            return None
        if not manifest_path.exists():
            return None

        base_config_path = training_cfg.get("base_train_config")
        if not base_config_path:
            raise ValueError("training.base_train_config is required when trigger=launch")

        editor_state_before = self.editor.model_state() if isinstance(self.editor, QwenEditEditor) else None
        train_config = load_yaml_config(base_config_path)
        train_overrides = training_cfg.get("train_config_overrides") or training_cfg.get("editor_train_overrides")
        if isinstance(train_overrides, dict):
            train_config = _deep_update(train_config, train_overrides)
        train_training_config = train_config.setdefault("training", {})
        for key in EDITOR_TRAINING_OVERRIDE_KEYS:
            if key in training_cfg:
                train_training_config[key] = training_cfg[key]
        train_config["name"] = f"{train_config['name']}_self_evolve_r{round_index:02d}"
        train_config["dataset"]["dataset_base_path"] = "."
        train_config["dataset"]["dataset_metadata_path"] = relative_to_repo(manifest_path)
        train_config["output"]["output_path"] = relative_to_repo(round_dir / "training_output")
        train_config["output"]["command_file"] = relative_to_repo(round_dir / "training_command.txt")
        train_config["output"]["log_dir"] = relative_to_repo(round_dir / "training_logs")
        output_dir = resolve_path(train_config["output"]["output_path"])
        if output_dir is None:
            raise ValueError("Could not resolve training output path")
        if (
            bool(training_cfg.get("resume_from_latest", True))
            and not train_config.get("training", {}).get("resume_from_checkpoint")
            and discover_latest_step_checkpoint(output_dir) is not None
        ):
            train_config["training"]["resume_from_checkpoint"] = "latest"

        current_checkpoint = training_cfg.get("current_checkpoint_path")
        if current_checkpoint and train_config.get("mode") == "lora":
            lora_config = train_config.setdefault("lora", {})
            if "lora_checkpoint" in lora_config:
                lora_config["lora_checkpoint"] = current_checkpoint
            else:
                lora_config["checkpoint_path"] = current_checkpoint

        command, working_dir = build_train_command(train_config)
        command_path = resolve_path(train_config["output"]["command_file"])
        log_dir = ensure_dir(resolve_path(train_config["output"]["log_dir"]))
        if command_path is None:
            raise ValueError("Could not resolve training command file path")
        ensure_dir(command_path.parent)
        command_path.write_text(shell_join(command) + "\n", encoding="utf-8")
        log_path = log_dir / f"{train_config['name']}_{utc_timestamp()}.log"

        metadata = {
            **base_run_metadata(),
            "type": "self_evolve_training_round",
            "round_index": round_index,
            "config_path": train_config["_config_path"],
            "command": command,
            "working_dir": str(working_dir),
            "log_path": str(log_path),
            "dry_run": self.dry_run,
            "resume_from_checkpoint": train_config.get("training", {}).get("resume_from_checkpoint"),
        }
        save_json(metadata, round_dir / "training_metadata.json")

        if self.dry_run:
            return {
                "status": "planned",
                "command_path": str(command_path),
                "log_path": str(log_path),
                "editor_state_before_training": editor_state_before,
                "editor_state_after_training": editor_state_before,
                "continue_with_trained_checkpoint": bool(training_cfg.get("continue_with_trained_checkpoint", True)),
                "trained_checkpoint_backend": training_checkpoint_backend(training_cfg, train_config),
            }

        self._release_resident_models("editor training subprocess")
        return_code = run_and_tee(command, cwd=working_dir, log_path=log_path)
        if return_code != 0:
            raise SystemExit(return_code)

        latest_checkpoint = discover_latest_checkpoint(output_dir)
        editor_state_after = editor_state_before
        if latest_checkpoint is not None:
            training_cfg["current_checkpoint_path"] = str(latest_checkpoint)
            if isinstance(self.editor, QwenEditEditor):
                continue_with_checkpoint = bool(training_cfg.get("continue_with_trained_checkpoint", True))
                trained_checkpoint_backend = training_checkpoint_backend(training_cfg, train_config)
                trained_model_type = str(train_config.get("mode", "lora"))
                if continue_with_checkpoint:
                    self.editor.set_model_checkpoint(
                        str(latest_checkpoint),
                        model_type=trained_model_type,
                        backend=trained_checkpoint_backend,
                    )
                editor_state_after = self.editor.model_state()
        return {
            "status": "completed",
            "command_path": str(command_path),
            "log_path": str(log_path),
            "output_dir": str(output_dir),
            "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint is not None else None,
            "editor_state_before_training": editor_state_before,
            "editor_state_after_training": editor_state_after,
            "continue_with_trained_checkpoint": bool(training_cfg.get("continue_with_trained_checkpoint", True)),
            "trained_checkpoint_backend": training_checkpoint_backend(training_cfg, train_config),
        }

    def _run_proposer_training_round(
        self,
        round_index: int,
        round_dir: Path,
        candidate_payloads: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        proposer_cfg = self.config.get("proposer", {})
        training_cfg = proposer_cfg.get("training", {})
        trigger = str(training_cfg.get("trigger", "none"))
        if trigger in {"none", "disabled", "false"}:
            return None

        data_cfg = training_cfg.get("data", {})
        records, data_summary = build_proposer_training_records(
            candidate_payloads,
            reward_center=float(data_cfg.get("reward_center", 0.50)),
            reward_sigma=float(data_cfg.get("reward_sigma", 0.25)),
            min_reward=float(data_cfg.get("min_reward", 0.35)),
            min_quality=float(data_cfg.get("min_quality", 0.30)),
        )
        train_jsonl = write_proposer_training_jsonl(records, round_dir / "proposer_training.jsonl")
        save_json(data_summary, round_dir / "proposer_training_summary.json")
        if data_summary["selected_for_sft"] <= 0:
            return {
                "status": "skipped_no_positive_records",
                "train_jsonl": str(train_jsonl),
                "summary": data_summary,
            }

        output_dir = resolve_path(
            str(training_cfg.get("output_dir") or round_dir / "proposer_training_output")
        )
        if output_dir is None:
            raise ValueError("proposer.training.output_dir could not be resolved")
        command_path = resolve_path(
            str(training_cfg.get("command_file") or round_dir / "proposer_training_command.txt")
        )
        if command_path is None:
            raise ValueError("proposer.training.command_file could not be resolved")
        log_dir = ensure_dir(resolve_path(str(training_cfg.get("log_dir") or round_dir / "proposer_training_logs")))
        train_script = resolve_path(
            str(training_cfg.get("train_script", "src/qwen_edit_project/train/train_proposer_lora.py"))
        )
        if train_script is None:
            raise ValueError("proposer.training.train_script could not be resolved")

        command = [str(training_cfg.get("accelerate_executable", "accelerate")), "launch"]
        accelerate_config = training_cfg.get("accelerate_config_file")
        if accelerate_config:
            command.extend(["--config_file", str(resolve_path(str(accelerate_config)) or accelerate_config)])
        command.append(str(train_script))
        _append_flag(command, "--model_name_or_path", proposer_cfg.get("model_name_or_path"))
        _append_flag(command, "--model_subfolder", proposer_cfg.get("model_subfolder"))
        _append_flag(command, "--processor_subfolder", proposer_cfg.get("processor_subfolder"))
        _append_flag(command, "--model_class", proposer_cfg.get("model_class"))
        _append_flag(command, "--train_jsonl", train_jsonl)
        _append_flag(command, "--dataset_base_path", ".")
        _append_flag(command, "--output_dir", output_dir)
        _append_flag(command, "--checkpoint_path", proposer_cfg.get("checkpoint_path"))
        _append_flag(command, "--local_files_only", proposer_cfg.get("local_files_only", False))
        _append_flag(command, "--torch_dtype", proposer_cfg.get("torch_dtype", "auto"))
        _append_flag(command, "--mixed_precision", training_cfg.get("mixed_precision", "bf16"))
        _append_flag(command, "--learning_rate", training_cfg.get("learning_rate", 1e-5))
        _append_flag(command, "--num_train_epochs", training_cfg.get("num_train_epochs", 1))
        _append_flag(command, "--max_train_steps", training_cfg.get("max_train_steps"))
        _append_flag(command, "--train_batch_size", training_cfg.get("train_batch_size", 1))
        _append_flag(command, "--gradient_accumulation_steps", training_cfg.get("gradient_accumulation_steps", 4))
        _append_flag(command, "--max_grad_norm", training_cfg.get("max_grad_norm", 1.0))
        _append_flag(command, "--weight_decay", training_cfg.get("weight_decay", 0.01))
        _append_flag(command, "--seed", training_cfg.get("seed", self.config.get("runtime", {}).get("seed", 123)))
        _append_flag(command, "--min_reward", data_cfg.get("min_reward", 0.35))
        _append_flag(command, "--lora_rank", training_cfg.get("lora_rank", 16))
        _append_flag(command, "--lora_alpha", training_cfg.get("lora_alpha", 32))
        _append_flag(command, "--lora_dropout", training_cfg.get("lora_dropout", 0.05))
        _append_flag(command, "--lora_target_modules", training_cfg.get("lora_target_modules"))
        _append_flag(command, "--gradient_checkpointing", training_cfg.get("gradient_checkpointing", True))
        _append_flag(command, "--checkpointing_steps", training_cfg.get("checkpointing_steps", 0))
        _append_flag(command, "--checkpoints_total_limit", training_cfg.get("checkpoints_total_limit", 5))
        _append_flag(command, "--logging_steps", training_cfg.get("logging_steps", 10))
        if bool(training_cfg.get("resume_from_latest", True)) and discover_latest_step_checkpoint(output_dir) is not None:
            _append_flag(command, "--resume_from_checkpoint", "latest")

        ensure_dir(command_path.parent)
        command_path.write_text(shell_join(command) + "\n", encoding="utf-8")
        log_path = log_dir / f"proposer_r{round_index:02d}_{utc_timestamp()}.log"
        if self.dry_run or trigger == "emit_only":
            return {
                "status": "planned",
                "train_jsonl": str(train_jsonl),
                "command_path": str(command_path),
                "log_path": str(log_path),
                "output_dir": str(output_dir),
                "summary": data_summary,
            }
        if trigger != "launch":
            raise ValueError(f"Unsupported proposer.training.trigger: {trigger}")
        self._release_resident_models("proposer training subprocess")
        return_code = run_and_tee(command, cwd=resolve_path(".") or Path("."), log_path=log_path)
        if return_code != 0:
            raise SystemExit(return_code)
        proposer_cfg["checkpoint_path"] = str(output_dir)
        if hasattr(self.proposer, "set_checkpoint_path"):
            self.proposer.set_checkpoint_path(str(output_dir))
        return {
            "status": "completed",
            "train_jsonl": str(train_jsonl),
            "command_path": str(command_path),
            "log_path": str(log_path),
            "output_dir": str(output_dir),
            "summary": data_summary,
        }

    def run(self) -> dict[str, Any]:
        runtime = self.config.get("runtime", {})
        seed = int(runtime.get("seed", 123))
        curriculum = self.config["curriculum"]
        proposals_per_image = int(curriculum.get("proposals_per_image", 1))
        candidate_generation = self.config.get("candidate_generation", {})
        samples_per_proposal = int(candidate_generation.get("samples_per_proposal", 1))
        candidate_seed_stride = int(candidate_generation.get("seed_stride", 7919))
        max_records_per_round = int(curriculum.get("max_records_per_round", len(self.records)))
        num_rounds = _resolve_num_rounds(curriculum.get("num_rounds", 3), len(self.records), max_records_per_round)
        output_cfg = self.config["output"]
        resume_cfg = output_cfg.get("resume", {})
        resume_enabled = bool(resume_cfg.get("enabled", True))
        progress_log_every = max(1, int(resume_cfg.get("progress_log_every", output_cfg.get("progress_log_every", 10))))
        save_all_candidates = bool(output_cfg.get("save_all_candidates", False))
        use_cumulative_manifest = bool(output_cfg.get("use_cumulative_manifest", True))
        write_evaluator_training = bool(output_cfg.get("write_evaluator_training", True))

        overall_summary = {
            **base_run_metadata(),
            "type": "self_evolve_run",
            "config_path": self.config["_config_path"],
            "dry_run": self.dry_run,
            "resume_enabled": resume_enabled,
            "records_available": len(self.records),
            "output_root": str(self.output_root),
            "rounds": [],
        }
        save_json({**base_run_metadata(), "config": self.config}, self.output_root / "run_config_resolved.json")
        self.logger.info(
            "Self-evolve run started: rounds=%s records=%s max_records_per_round=%s samples_per_proposal=%s resume=%s",
            num_rounds,
            len(self.records),
            max_records_per_round,
            samples_per_proposal,
            resume_enabled,
        )
        cumulative_accepted: list[AcceptedSample] = []
        cumulative_training_records: list[dict[str, Any]] = []

        for round_index in range(1, num_rounds + 1):
            round_started_at = time.time()
            round_dir = ensure_dir(self.output_root / f"round_{round_index:02d}")
            candidates_dir = ensure_dir(round_dir / "candidates")
            accepted_dir = ensure_dir(round_dir / "accepted" / "images")
            candidate_image_dir = ensure_dir(candidates_dir / "images") if save_all_candidates else None
            summary_path = round_dir / "summary.json"
            progress_path = round_dir / "progress.json"
            proposals_path = round_dir / "proposals.jsonl"
            proposal_plan_path = round_dir / "proposal_plan.jsonl"
            evaluator_training_path = round_dir / "evaluator_training.jsonl"
            preference_path = round_dir / "evaluator_preferences.jsonl"

            if resume_enabled and summary_path.exists():
                round_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if str(round_summary.get("status", "completed")) == "completed":
                    candidate_rows = read_jsonl(proposals_path)
                    accepted_from_round = self._accepted_samples_from_payloads(candidate_rows)
                    cumulative_accepted.extend(accepted_from_round)
                    training_records_from_round, _, _ = self._training_records_from_payloads(candidate_rows)
                    cumulative_training_records.extend(training_records_from_round)
                    self._restore_completed_round_state(round_summary)
                    overall_summary["rounds"].append(round_summary)
                    self.logger.info(
                        "Round %02d already completed; restored %s accepted samples, %s training records, and skipped generation/training.",
                        round_index,
                        len(accepted_from_round),
                        len(training_records_from_round),
                    )
                    continue

            if not resume_enabled:
                for stale_path in [proposals_path, proposal_plan_path, progress_path]:
                    if stale_path.exists():
                        stale_path.unlink()

            difficulty_level = self.difficulty_controller.level
            editor_state_before_round = self.editor.model_state() if isinstance(self.editor, QwenEditEditor) else None
            proposer_state_before_round = (
                self.proposer.model_state() if hasattr(self.proposer, "model_state") else None
            )
            round_records, round_record_info = self._round_records_for_round(round_index, max_records_per_round)
            existing_candidate_rows = read_jsonl(proposals_path) if resume_enabled else []
            candidate_payload_by_key = {
                self._candidate_key(payload): payload
                for payload in existing_candidate_rows
                if payload.get("group_id") is not None
            }
            existing_groups = self._group_rows(list(candidate_payload_by_key.values()))
            skippable_groups = {
                group_id
                for group_id, rows in existing_groups.items()
                if self._is_completed_group(rows, samples_per_proposal)
            }
            if self.dry_run:
                skippable_groups.update(
                    group_id
                    for group_id, rows in existing_groups.items()
                    if len(rows) >= samples_per_proposal
                )

            proposal_plan_rows = read_jsonl(proposal_plan_path) if resume_enabled else []
            plan_rows_by_record: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for plan_row in proposal_plan_rows:
                plan_rows_by_record[str(plan_row.get("record_key", ""))].append(plan_row)
            for rows in plan_rows_by_record.values():
                rows.sort(key=lambda row: int(row.get("proposal", {}).get("proposal_index", 0)))

            self._write_progress(
                progress_path,
                {
                    "status": "running",
                    "round_index": round_index,
                    "difficulty_level": difficulty_level,
                    **round_record_info,
                    "records_seen": 0,
                    "records_total": len(round_records),
                    "groups_completed": len(skippable_groups),
                    "groups_total_estimate": len(round_records) * proposals_per_image,
                    "candidate_rows_loaded": len(existing_candidate_rows),
                    "candidate_rows_written": len(candidate_payload_by_key),
                    "accepted": len(self._accepted_samples_from_payloads(list(candidate_payload_by_key.values()))),
                    "elapsed_seconds": 0.0,
                    "proposals_path": str(proposals_path),
                    "proposal_plan_path": str(proposal_plan_path),
                },
            )
            if existing_candidate_rows or proposal_plan_rows:
                self.logger.info(
                    "Round %02d resume state loaded: proposal_plans=%s candidate_rows=%s skippable_groups=%s.",
                    round_index,
                    len(proposal_plan_rows),
                    len(existing_candidate_rows),
                    len(skippable_groups),
                )

            groups_seen = 0
            groups_completed_this_run = 0
            for record_index, record in enumerate(round_records):
                planned_rows = plan_rows_by_record.get(record.key, [])
                if planned_rows:
                    planned_groups = [
                        (str(plan_row["group_id"]), self._proposal_from_payload(plan_row))
                        for plan_row in planned_rows
                    ]
                else:
                    proposed = self.proposer.propose(
                        record=record,
                        round_index=round_index,
                        difficulty_level=difficulty_level,
                        proposals_per_image=proposals_per_image,
                        seed=seed + record_index,
                    )
                    planned_groups = []
                    for proposal in proposed:
                        group_id = (
                            f"{record.key}__r{round_index:02d}__p{proposal.proposal_index:02d}"
                            f"__{proposal.definition.operation_id}"
                        )
                        plan_payload = self._proposal_plan_payload(record_index, record, proposal, group_id)
                        append_jsonl(plan_payload, proposal_plan_path)
                        proposal_plan_rows.append(plan_payload)
                        planned_groups.append((group_id, proposal))

                self._release_component_model("proposer", "editor candidate generation")

                for group_id, proposal in planned_groups:
                    groups_seen += 1
                    if group_id in skippable_groups:
                        continue
                    distractors = (
                        self.evaluator.describe_distractors(proposal)
                        if hasattr(self.evaluator, "describe_distractors")
                        else []
                    )
                    if self.dry_run:
                        for candidate_index in range(samples_per_proposal):
                            payload = self._candidate_payload(
                                record,
                                proposal,
                                evaluation_result=None,
                                image_path=None,
                                status="planned",
                                candidate_index=candidate_index,
                                group_id=group_id,
                                distractors=distractors,
                            )
                            candidate_payload_by_key[self._candidate_key(payload)] = payload
                            append_jsonl(payload, proposals_path)
                        skippable_groups.add(group_id)
                        groups_completed_this_run += 1
                        continue

                    self.logger.info(
                        "Round %02d group %s: generating %s candidate(s) for record %s.",
                        round_index,
                        group_id,
                        samples_per_proposal,
                        record.key,
                    )
                    with Image.open(record.image_path) as original_image_handle:
                        original_image = original_image_handle.convert("RGB")
                    edited_images: list[Image.Image] = []
                    for candidate_index in range(samples_per_proposal):
                        candidate_seed = (
                            seed
                            + round_index * 1_000_003
                            + record_index * candidate_seed_stride
                            + proposal.proposal_index * 101
                            + candidate_index
                        )
                        if hasattr(self.editor, "edit_candidate"):
                            edited_image = self.editor.edit_candidate(record, proposal, candidate_index, candidate_seed)
                        else:
                            edited_image = self.editor.edit(record, proposal)
                        edited_images.append(edited_image)
                        if isinstance(self.editor, QwenEditEditor):
                            QwenEditEditor._empty_cuda_cache()

                    if hasattr(self.evaluator, "score_group"):
                        evaluation_results = self.evaluator.score_group(
                            proposal, original_image, edited_images, editor=self.editor
                        )
                    else:
                        evaluation_results = [
                            self.evaluator.score(proposal, original_image, edited_image, editor=self.editor)
                            for edited_image in edited_images
                        ]

                    for candidate_index, (edited_image, evaluation_result) in enumerate(
                        zip(edited_images, evaluation_results)
                    ):
                        output_name = (
                            f"{record.key}__r{round_index:02d}__p{proposal.proposal_index:02d}"
                            f"__c{candidate_index:02d}__{proposal.definition.operation_id}.png"
                        )
                        image_path = None
                        if evaluation_result.accepted:
                            image_path = accepted_dir / output_name
                            edited_image.save(image_path)
                        elif candidate_image_dir is not None:
                            image_path = candidate_image_dir / output_name
                            edited_image.save(image_path)

                        payload = self._candidate_payload(
                            record,
                            proposal,
                            evaluation_result=evaluation_result,
                            image_path=image_path,
                            status="accepted" if evaluation_result.accepted else "rejected",
                            candidate_index=candidate_index,
                            group_id=group_id,
                            distractors=distractors,
                        )
                        candidate_payload_by_key[self._candidate_key(payload)] = payload
                        append_jsonl(payload, proposals_path)

                    del edited_images
                    del evaluation_results
                    del original_image
                    gc.collect()
                    if isinstance(self.editor, QwenEditEditor):
                        QwenEditEditor._empty_cuda_cache()

                    skippable_groups.add(group_id)
                    groups_completed_this_run += 1
                    candidate_payloads_for_progress = list(candidate_payload_by_key.values())
                    accepted_for_progress = self._accepted_samples_from_payloads(candidate_payloads_for_progress)
                    self._write_progress(
                        progress_path,
                        {
                            "status": "running",
                            "round_index": round_index,
                            "difficulty_level": difficulty_level,
                            **round_record_info,
                            "records_seen": record_index + 1,
                            "records_total": len(round_records),
                            "groups_seen": groups_seen,
                            "groups_completed": len(skippable_groups),
                            "groups_completed_this_run": groups_completed_this_run,
                            "groups_total_estimate": len(round_records) * proposals_per_image,
                            "candidate_rows_written": len(candidate_payload_by_key),
                            "accepted": len(accepted_for_progress),
                            "current_record_key": record.key,
                            "current_group_id": group_id,
                            "elapsed_seconds": round(time.time() - round_started_at, 3),
                            "proposals_path": str(proposals_path),
                            "proposal_plan_path": str(proposal_plan_path),
                        },
                    )
                    if groups_completed_this_run % progress_log_every == 0:
                        self.logger.info(
                            "Round %02d progress: completed_groups=%s candidates=%s accepted=%s elapsed=%.1fs.",
                            round_index,
                            len(skippable_groups),
                            len(candidate_payload_by_key),
                            len(accepted_for_progress),
                            time.time() - round_started_at,
                        )

            candidate_payloads = sorted(
                candidate_payload_by_key.values(),
                key=lambda payload: (str(payload.get("group_id", "")), int(payload.get("candidate_index", 0))),
            )
            proposals_path = write_jsonl(candidate_payloads, proposals_path)
            accepted = self._accepted_samples_from_payloads(candidate_payloads)
            persisted_evaluator_training_path = None
            persisted_preference_path = None
            if write_evaluator_training:
                evaluator_training_records, preference_records = self._build_evaluator_exports(candidate_payloads)
                persisted_evaluator_training_path = write_jsonl(evaluator_training_records, evaluator_training_path)
                persisted_preference_path = write_jsonl(preference_records, preference_path)
            manifest_path = round_dir / "train_manifest.json"
            cumulative_accepted.extend(accepted)
            round_training_records, training_weight_audit, training_weight_summary = self._training_records_from_payloads(
                candidate_payloads
            )
            cumulative_training_records.extend(round_training_records)
            manifest_samples = cumulative_training_records if use_cumulative_manifest else round_training_records
            train_weight_audit_path = write_jsonl(training_weight_audit, round_dir / "train_weights.jsonl")
            save_json(training_weight_summary, round_dir / "train_weight_summary.json")
            _, train_manifest_sample_count, train_manifest_weight_sum = self._write_manifest_records(
                manifest_samples,
                manifest_path,
            )

            accepted_scores = [sample.evaluation_result.total_score for sample in accepted]
            global_scores = [sample.evaluation_result.global_score for sample in accepted]
            local_scores = [sample.evaluation_result.local_score for sample in accepted]
            component_score_totals: dict[str, list[float]] = {}
            for sample in accepted:
                for name, value in sample.evaluation_result.component_scores.items():
                    component_score_totals.setdefault(name, []).append(value)
            total_candidates = len(candidate_payloads)
            acceptance_rate = (len(accepted) / total_candidates) if total_candidates else 0.0
            next_level = self.difficulty_controller.update(acceptance_rate)
            self._write_progress(
                progress_path,
                {
                    "status": "training",
                    "round_index": round_index,
                    "difficulty_level": difficulty_level,
                    "next_difficulty_level": next_level,
                    **round_record_info,
                    "records_seen": len(round_records),
                    "records_total": len(round_records),
                    "groups_completed": len(skippable_groups),
                    "groups_total_estimate": len(round_records) * proposals_per_image,
                    "candidate_rows_written": total_candidates,
                    "accepted": len(accepted),
                    "acceptance_rate": acceptance_rate,
                    "train_manifest_samples": train_manifest_sample_count,
                    "train_manifest_weight_sum": train_manifest_weight_sum,
                    "elapsed_seconds": round(time.time() - round_started_at, 3),
                    "proposals_path": str(proposals_path),
                    "manifest_path": str(manifest_path),
                },
            )
            training_result = (
                self._run_training_round(round_index, round_dir, manifest_path)
                if train_manifest_sample_count > 0
                else None
            )
            proposer_training_result = self._run_proposer_training_round(round_index, round_dir, candidate_payloads)

            round_summary = {
                "status": "completed",
                "round_index": round_index,
                "difficulty_level": difficulty_level,
                "next_difficulty_level": next_level,
                **round_record_info,
                "records_seen": len(round_records),
                "proposal_groups": len(self._group_rows(candidate_payloads)),
                "candidates": total_candidates,
                "accepted": len(accepted),
                "cumulative_accepted": len(cumulative_accepted),
                "round_training_samples": len(round_training_records),
                "round_training_weight_sum": training_weight_summary["weight_sum"],
                "train_manifest_samples": train_manifest_sample_count,
                "train_manifest_weight_sum": train_manifest_weight_sum,
                "acceptance_rate": acceptance_rate,
                "avg_total_score": (sum(accepted_scores) / len(accepted_scores)) if accepted_scores else 0.0,
                "avg_global_score": (sum(global_scores) / len(global_scores)) if global_scores else 0.0,
                "avg_local_score": (sum(local_scores) / len(local_scores)) if local_scores else 0.0,
                "avg_component_scores": {
                    name: (sum(values) / len(values)) for name, values in sorted(component_score_totals.items())
                },
                "resume": {
                    "enabled": resume_enabled,
                    "candidate_rows_loaded": len(existing_candidate_rows),
                    "proposal_plan_rows_loaded": len(proposal_plan_rows),
                    "skippable_groups_loaded": len(skippable_groups) - groups_completed_this_run,
                    "groups_completed_this_run": groups_completed_this_run,
                    "progress_path": str(progress_path),
                    "proposal_plan_path": str(proposal_plan_path),
                },
                "elapsed_seconds": round(time.time() - round_started_at, 3),
                "proposals_path": str(proposals_path),
                "train_weights_path": str(train_weight_audit_path),
                "train_weight_summary": training_weight_summary,
                "evaluator_training_path": (
                    str(persisted_evaluator_training_path) if persisted_evaluator_training_path is not None else None
                ),
                "evaluator_preferences_path": (
                    str(persisted_preference_path) if persisted_preference_path is not None else None
                ),
                "manifest_path": str(manifest_path),
                "training": training_result,
                "proposer_training": proposer_training_result,
                "proposer_state_before_round": proposer_state_before_round,
                "proposer_state_after_round": (
                    self.proposer.model_state() if hasattr(self.proposer, "model_state") else None
                ),
                "editor_state_before_round": editor_state_before_round,
                "editor_state_after_round": self.editor.model_state() if isinstance(self.editor, QwenEditEditor) else None,
            }
            save_json(round_summary, summary_path)
            self._write_progress(
                progress_path,
                {
                    "status": "completed",
                    "round_index": round_index,
                    "difficulty_level": difficulty_level,
                    "next_difficulty_level": next_level,
                    **round_record_info,
                    "records_seen": len(round_records),
                    "records_total": len(round_records),
                    "groups_completed": len(skippable_groups),
                    "groups_total_estimate": len(round_records) * proposals_per_image,
                    "candidate_rows_written": total_candidates,
                    "accepted": len(accepted),
                    "acceptance_rate": acceptance_rate,
                    "train_manifest_samples": train_manifest_sample_count,
                    "train_manifest_weight_sum": train_manifest_weight_sum,
                    "elapsed_seconds": round(time.time() - round_started_at, 3),
                    "summary_path": str(summary_path),
                    "proposals_path": str(proposals_path),
                    "manifest_path": str(manifest_path),
                },
            )
            overall_summary["rounds"].append(round_summary)
            self.logger.info(
                "Round %02d completed: candidates=%s accepted=%s acceptance_rate=%.4f next_difficulty=%s elapsed=%.1fs.",
                round_index,
                total_candidates,
                len(accepted),
                acceptance_rate,
                next_level,
                time.time() - round_started_at,
            )

        save_json(overall_summary, self.output_root / "summary.json")
        self.logger.info("Self-evolve run finished. Summary written to %s", self.output_root / "summary.json")
        return overall_summary
