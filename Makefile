PYTHON ?= python

.PHONY: bootstrap prepare-remote-data select-unlabeled-images train-lora train-full validate-lora validate-full \
		export-gedit score-gedit export-imgedit score-imgedit \
		export-geneval score-geneval export-dpgbench score-dpgbench \
		export-oneig-bench score-oneig-bench \
		self-evolve-demo self-evolve-qwen \
		run-edit-suite run-generation-suite run-self-evolve-matrix smoke-test

bootstrap:
	bash scripts/bootstrap.sh

prepare-remote-data:
	bash scripts/prepare_remote_data.sh $(ARGS)

select-unlabeled-images:
	bash scripts/select_unlabeled_images.sh $(ARGS)

train-lora:
	bash scripts/train_lora_2509.sh $(ARGS)

train-full:
	bash scripts/train_full_2509.sh $(ARGS)

validate-lora:
	@test -n "$(CHECKPOINT)" || (echo "Use: make validate-lora CHECKPOINT=/path/to/checkpoint" && exit 1)
	bash scripts/validate_lora_2509.sh "$(CHECKPOINT)" $(ARGS)

validate-full:
	@test -n "$(CHECKPOINT)" || (echo "Use: make validate-full CHECKPOINT=/path/to/checkpoint" && exit 1)
	bash scripts/validate_full_2509.sh "$(CHECKPOINT)" $(ARGS)

export-gedit:
	bash scripts/export_gedit.sh $(ARGS)

score-gedit:
	bash scripts/score_gedit.sh $(ARGS)

export-imgedit:
	bash scripts/export_imgedit.sh $(ARGS)

score-imgedit:
	bash scripts/score_imgedit.sh $(ARGS)

export-geneval:
	bash scripts/export_geneval.sh $(ARGS)

score-geneval:
	bash scripts/score_geneval.sh $(ARGS)

export-dpgbench:
	bash scripts/export_dpgbench.sh $(ARGS)

score-dpgbench:
	bash scripts/score_dpgbench.sh $(ARGS)

export-oneig-bench:
	bash scripts/export_oneig_bench.sh $(ARGS)

score-oneig-bench:
	bash scripts/score_oneig_bench.sh $(ARGS)

self-evolve-demo:
	bash scripts/self_evolve_pillow_demo.sh $(ARGS)

self-evolve-qwen:
	bash scripts/self_evolve_2509.sh $(ARGS)

run-edit-suite:
	bash scripts/run_edit_model_suite.sh $(ARGS)

run-generation-suite:
	bash scripts/run_generation_sanity_suite.sh $(ARGS)

run-self-evolve-matrix:
	bash scripts/run_self_evolve_matrix.sh $(ARGS)

smoke-test:
	bash scripts/run_pipeline_smoke_tests.sh $(ARGS)
