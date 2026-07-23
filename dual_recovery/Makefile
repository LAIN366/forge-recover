PYTHON ?= python3
PYTHONPATH := scripts

.PHONY: test compile smoke dual-smoke ablation dual-ablation

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m unittest discover -s tests -v

compile:
	$(PYTHON) -m compileall -q scripts tests

smoke:
	cd scripts && MUJOCO_GL=egl $(PYTHON) physical_grasp_demo.py \
		--headless --fault none --policy active_case --timeout 90

dual-smoke:
	cd scripts && MUJOCO_GL=egl $(PYTHON) dual_arm_transport_demo.py \
		--headless --timeout 90

ablation:
	cd scripts && MUJOCO_GL=egl $(PYTHON) run_batch_experiments.py \
		--policies no_recovery fixed_rule diagnosis_only active_case \
		--faults slip missed_grasp target_shift --repeats 3

dual-ablation:
	MUJOCO_GL=egl $(PYTHON) scripts/run_dual_arm_experiments.py \
		--faults synchronization_delay \
		--policies no_recovery full --seeds 7 17 27
