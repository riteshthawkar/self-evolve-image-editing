from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from qwen_edit_project.self_evolve.backends import (
    DifficultyController,
    QwenEditEditor,
    build_editor,
    build_proposer,
    build_solver,
)
from qwen_edit_project.self_evolve.data import load_unlabeled_records
from qwen_edit_project.self_evolve.types import AcceptedSample, EditProposal, SolverResult, UnlabeledImageRecord
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


def discover_latest_checkpoint(directory: Path) -> Path | None:
    candidates = sorted(directory.rglob("*.safetensors"), key=lambda path: path.stat().st_mtime_ns)
    if not candidates:
        return None
    return candidates[-1]


class SelfEvolveRunner:
    def __init__(self, config: dict[str, Any], dry_run: bool = False, limit: int | None = None):
        self.config = config
        self.dry_run = dry_run or bool(config.get("runtime", {}).get("dry_run", False))
        dataset_limit = config.get("dataset", {}).get("limit")
        self.limit = limit if limit is not None else dataset_limit
        self.records = load_unlabeled_records(config["dataset"], limit=self.limit)
        self.output_root = ensure_dir(resolve_path(config["output"]["root_dir"]))
        self.proposer = build_proposer(config["proposer"])
        self.editor = build_editor(config["editor"])
        self.solver = build_solver(config["solver"])
        curriculum = config["curriculum"]
        self.difficulty_controller = DifficultyController(
            initial_level=int(curriculum.get("initial_level", 1)),
            min_level=int(curriculum.get("min_level", 1)),
            max_level=int(curriculum.get("max_level", 3)),
            promote_at=float(curriculum.get("promote_at", 0.75)),
            demote_at=float(curriculum.get("demote_at", 0.45)),
        )

    def _candidate_payload(
        self,
        record: UnlabeledImageRecord,
        proposal: EditProposal,
        solver_result: SolverResult | None,
        image_path: Path | None,
        status: str,
        candidate_index: int = 0,
        group_id: str | None = None,
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
                "verifier": proposal.definition.verifier,
                "inverse_operation_id": proposal.definition.inverse_operation_id,
            },
            "status": status,
            "edited_image_path": relative_to_repo(image_path) if image_path is not None else None,
        }
        if solver_result is not None:
            payload["solver"] = {
                "global_score": solver_result.global_score,
                "local_score": solver_result.local_score,
                "total_score": solver_result.total_score,
                "accepted": solver_result.accepted,
                "component_scores": solver_result.component_scores,
                "signals": solver_result.signals,
            }
        return payload

    def _write_manifest(self, accepted: list[AcceptedSample], manifest_path: Path) -> tuple[Path, int]:
        allowed_verifiers = self.config.get("output", {}).get("train_verifiers")
        allowed_verifier_set = set(allowed_verifiers) if allowed_verifiers else None
        training_cfg = self.config.get("training", {})
        replay_ratio = float(training_cfg.get("reconstruction_replay_ratio", 0.0))
        replay_prompt = str(
            training_cfg.get(
                "reconstruction_replay_prompt",
                "Reconstruct the input image exactly. Preserve all content, layout, colors, and text.",
            )
        )
        manifest_records = []
        accepted_records = []
        replay_source_paths: list[Path] = []
        for sample in accepted:
            if allowed_verifier_set is not None and sample.proposal.definition.verifier not in allowed_verifier_set:
                continue
            replay_source_paths.append(sample.record.image_path)
            manifest_records.append(
                {
                    "prompt": sample.proposal.instruction,
                    "image": relative_to_repo(sample.edited_image_path),
                    "edit_image": relative_to_repo(sample.record.image_path),
                }
            )
            accepted_records.append(
                {
                    "record_key": sample.record.key,
                    "original_image": relative_to_repo(sample.record.image_path),
                    "edited_image": relative_to_repo(sample.edited_image_path),
                    "instruction": sample.proposal.instruction,
                    "operation_id": sample.proposal.definition.operation_id,
                    "difficulty": sample.proposal.definition.difficulty,
                    "candidate_index": sample.candidate_index,
                    "scores": {
                        "global_score": sample.solver_result.global_score,
                        "local_score": sample.solver_result.local_score,
                        "total_score": sample.solver_result.total_score,
                        "component_scores": sample.solver_result.component_scores,
                    },
                }
            )
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
                    }
                )
                accepted_records.append(
                    {
                        "type": "reconstruction_replay",
                        "original_image": relative_to_repo(source_path),
                        "edited_image": relative_to_repo(source_path),
                        "instruction": replay_prompt,
                    }
                )
        save_json(manifest_records, manifest_path)
        write_jsonl(accepted_records, manifest_path.with_suffix(".jsonl"))
        return manifest_path, len(manifest_records)

    def _run_training_round(self, round_index: int, round_dir: Path, manifest_path: Path) -> dict[str, Any] | None:
        training_cfg = self.config.get("training", {})
        if training_cfg.get("trigger", "emit_only") != "launch":
            return None
        if not manifest_path.exists():
            return None

        base_config_path = training_cfg.get("base_train_config")
        if not base_config_path:
            raise ValueError("training.base_train_config is required when trigger=launch")

        train_config = load_yaml_config(base_config_path)
        train_config["name"] = f"{train_config['name']}_self_evolve_r{round_index:02d}"
        train_config["dataset"]["dataset_base_path"] = "."
        train_config["dataset"]["dataset_metadata_path"] = relative_to_repo(manifest_path)
        train_config["output"]["output_path"] = relative_to_repo(round_dir / "training_output")
        train_config["output"]["command_file"] = relative_to_repo(round_dir / "training_command.txt")
        train_config["output"]["log_dir"] = relative_to_repo(round_dir / "training_logs")

        current_checkpoint = training_cfg.get("current_checkpoint_path")
        if current_checkpoint and train_config.get("mode") == "lora":
            train_config["lora"]["lora_checkpoint"] = current_checkpoint

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
        }
        save_json(metadata, round_dir / "training_metadata.json")

        if self.dry_run:
            return {
                "status": "planned",
                "command_path": str(command_path),
                "log_path": str(log_path),
            }

        return_code = run_and_tee(command, cwd=working_dir, log_path=log_path)
        if return_code != 0:
            raise SystemExit(return_code)

        output_dir = resolve_path(train_config["output"]["output_path"])
        if output_dir is None:
            raise ValueError("Could not resolve training output path")
        latest_checkpoint = discover_latest_checkpoint(output_dir)
        if latest_checkpoint is not None:
            training_cfg["current_checkpoint_path"] = str(latest_checkpoint)
            if isinstance(self.editor, QwenEditEditor):
                self.editor.config["model"]["model_type"] = train_config.get("mode", self.editor.config["model"].get("model_type", "base"))
                self.editor.set_checkpoint_path(str(latest_checkpoint))
        return {
            "status": "completed",
            "command_path": str(command_path),
            "log_path": str(log_path),
            "output_dir": str(output_dir),
            "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint is not None else None,
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
        num_rounds = int(curriculum.get("num_rounds", 3))
        save_all_candidates = bool(self.config["output"].get("save_all_candidates", False))
        use_cumulative_manifest = bool(self.config["output"].get("use_cumulative_manifest", True))
        write_evaluator_training = bool(self.config["output"].get("write_evaluator_training", True))

        overall_summary = {
            **base_run_metadata(),
            "type": "self_evolve_run",
            "config_path": self.config["_config_path"],
            "dry_run": self.dry_run,
            "records_available": len(self.records),
            "output_root": str(self.output_root),
            "rounds": [],
        }
        cumulative_accepted: list[AcceptedSample] = []

        for round_index in range(1, num_rounds + 1):
            round_dir = ensure_dir(self.output_root / f"round_{round_index:02d}")
            candidates_dir = ensure_dir(round_dir / "candidates")
            accepted_dir = ensure_dir(round_dir / "accepted" / "images")
            candidate_image_dir = ensure_dir(candidates_dir / "images") if save_all_candidates else None

            difficulty_level = self.difficulty_controller.level
            round_records = self.records[:max_records_per_round]
            accepted: list[AcceptedSample] = []
            candidate_payloads: list[dict[str, Any]] = []
            evaluator_training_records: list[dict[str, Any]] = []
            preference_records: list[dict[str, Any]] = []

            for record_index, record in enumerate(round_records):
                proposals = self.proposer.propose(
                    record=record,
                    round_index=round_index,
                    difficulty_level=difficulty_level,
                    proposals_per_image=proposals_per_image,
                    seed=seed + record_index,
                )
                for proposal in proposals:
                    group_id = (
                        f"{record.key}__r{round_index:02d}__p{proposal.proposal_index:02d}"
                        f"__{proposal.definition.operation_id}"
                    )
                    distractors = (
                        self.solver.describe_distractors(proposal)
                        if hasattr(self.solver, "describe_distractors")
                        else []
                    )
                    if self.dry_run:
                        for candidate_index in range(samples_per_proposal):
                            candidate_payloads.append(
                                self._candidate_payload(
                                    record,
                                    proposal,
                                    solver_result=None,
                                    image_path=None,
                                    status="planned",
                                    candidate_index=candidate_index,
                                    group_id=group_id,
                                )
                            )
                        continue

                    with Image.open(record.image_path) as original_image_handle:
                        original_image = original_image_handle.convert("RGB")
                    edited_images: list[Image.Image] = []
                    for candidate_index in range(samples_per_proposal):
                        candidate_seed = seed + record_index * candidate_seed_stride + proposal.proposal_index * 101 + candidate_index
                        if hasattr(self.editor, "edit_candidate"):
                            edited_image = self.editor.edit_candidate(record, proposal, candidate_index, candidate_seed)
                        else:
                            edited_image = self.editor.edit(record, proposal)
                        edited_images.append(edited_image)

                    if hasattr(self.solver, "score_group"):
                        solver_results = self.solver.score_group(proposal, original_image, edited_images, editor=self.editor)
                    else:
                        solver_results = [
                            self.solver.score(proposal, original_image, edited_image, editor=self.editor)
                            for edited_image in edited_images
                        ]

                    group_payloads: list[dict[str, Any]] = []
                    for candidate_index, (edited_image, solver_result) in enumerate(zip(edited_images, solver_results)):
                        output_name = (
                            f"{record.key}__r{round_index:02d}__p{proposal.proposal_index:02d}"
                            f"__c{candidate_index:02d}__{proposal.definition.operation_id}.png"
                        )
                        image_path = None
                        if solver_result.accepted:
                            image_path = accepted_dir / output_name
                            edited_image.save(image_path)
                            accepted.append(
                                AcceptedSample(
                                    record=record,
                                    proposal=proposal,
                                    edited_image_path=image_path,
                                    solver_result=solver_result,
                                    candidate_index=candidate_index,
                                )
                            )
                        elif candidate_image_dir is not None:
                            image_path = candidate_image_dir / output_name
                            edited_image.save(image_path)

                        payload = self._candidate_payload(
                            record,
                            proposal,
                            solver_result=solver_result,
                            image_path=image_path,
                            status="accepted" if solver_result.accepted else "rejected",
                            candidate_index=candidate_index,
                            group_id=group_id,
                        )
                        candidate_payloads.append(payload)
                        group_payloads.append(payload)
                        if write_evaluator_training:
                            evaluator_training_records.append(
                                {
                                    "type": "candidate",
                                    "group_id": group_id,
                                    "candidate_index": candidate_index,
                                    "record_key": record.key,
                                    "source_image": relative_to_repo(record.image_path),
                                    "edited_image": relative_to_repo(image_path) if image_path is not None else None,
                                    "instruction": proposal.instruction,
                                    "operation_id": proposal.definition.operation_id,
                                    "family": proposal.definition.family,
                                    "verifier": proposal.definition.verifier,
                                    "distractors": distractors,
                                    "accepted": solver_result.accepted,
                                    "feasible": bool(solver_result.signals.get("feasible", float(solver_result.accepted))),
                                    "rank": int(solver_result.signals.get("feasible_rank", 0.0)),
                                    "scores": {
                                        "total_score": solver_result.total_score,
                                        "global_score": solver_result.global_score,
                                        "local_score": solver_result.local_score,
                                        "component_scores": solver_result.component_scores,
                                    },
                                }
                            )

                    accepted_payloads = [payload for payload in group_payloads if payload["status"] == "accepted"]
                    rejected_payloads = [payload for payload in group_payloads if payload["status"] == "rejected"]
                    for winner in accepted_payloads:
                        for loser in rejected_payloads:
                            preference_records.append(
                                {
                                    "type": "preference",
                                    "group_id": group_id,
                                    "winner_candidate_index": winner["candidate_index"],
                                    "loser_candidate_index": loser["candidate_index"],
                                    "record_key": record.key,
                                    "source_image": relative_to_repo(record.image_path),
                                    "instruction": proposal.instruction,
                                    "operation_id": proposal.definition.operation_id,
                                    "family": proposal.definition.family,
                                    "verifier": proposal.definition.verifier,
                                    "distractors": distractors,
                                    "winner_image": winner.get("edited_image_path"),
                                    "loser_image": loser.get("edited_image_path"),
                                    "winner_score": winner.get("solver", {}).get("total_score"),
                                    "loser_score": loser.get("solver", {}).get("total_score"),
                                }
                            )

            proposals_path = write_jsonl(candidate_payloads, round_dir / "proposals.jsonl")
            evaluator_training_path = None
            preference_path = None
            if write_evaluator_training:
                evaluator_training_path = write_jsonl(evaluator_training_records, round_dir / "evaluator_training.jsonl")
                preference_path = write_jsonl(preference_records, round_dir / "evaluator_preferences.jsonl")
            manifest_path = round_dir / "train_manifest.json"
            cumulative_accepted.extend(accepted)
            manifest_samples = cumulative_accepted if use_cumulative_manifest else accepted
            _, train_manifest_sample_count = self._write_manifest(manifest_samples, manifest_path)

            accepted_scores = [sample.solver_result.total_score for sample in accepted]
            global_scores = [sample.solver_result.global_score for sample in accepted]
            local_scores = [sample.solver_result.local_score for sample in accepted]
            component_score_totals: dict[str, list[float]] = {}
            for sample in accepted:
                for name, value in sample.solver_result.component_scores.items():
                    component_score_totals.setdefault(name, []).append(value)
            total_candidates = len(candidate_payloads)
            acceptance_rate = (len(accepted) / total_candidates) if total_candidates else 0.0
            next_level = self.difficulty_controller.update(acceptance_rate)
            training_result = self._run_training_round(round_index, round_dir, manifest_path) if accepted else None

            round_summary = {
                "round_index": round_index,
                "difficulty_level": difficulty_level,
                "next_difficulty_level": next_level,
                "records_seen": len(round_records),
                "candidates": total_candidates,
                "accepted": len(accepted),
                "cumulative_accepted": len(cumulative_accepted),
                "train_manifest_samples": train_manifest_sample_count,
                "acceptance_rate": acceptance_rate,
                "avg_total_score": (sum(accepted_scores) / len(accepted_scores)) if accepted_scores else 0.0,
                "avg_global_score": (sum(global_scores) / len(global_scores)) if global_scores else 0.0,
                "avg_local_score": (sum(local_scores) / len(local_scores)) if local_scores else 0.0,
                "avg_component_scores": {
                    name: (sum(values) / len(values)) for name, values in sorted(component_score_totals.items())
                },
                "proposals_path": str(proposals_path),
                "evaluator_training_path": str(evaluator_training_path) if evaluator_training_path is not None else None,
                "evaluator_preferences_path": str(preference_path) if preference_path is not None else None,
                "manifest_path": str(manifest_path),
                "training": training_result,
            }
            save_json(round_summary, round_dir / "summary.json")
            overall_summary["rounds"].append(round_summary)

        save_json(overall_summary, self.output_root / "summary.json")
        return overall_summary
