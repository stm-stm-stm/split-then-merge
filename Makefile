# Convenience targets. Override variables on the command line, e.g. `make data GPUS="0 0 0 0" TARGET=100000`.
PYTHON   ?= $(HOME)/venvs/stm/bin/python
GPUS     ?= 0 0 0 0
TARGET   ?= 100000
CONFIG   ?= training/configs/recipes/stm_main.conf
CKPT     ?= outputs/stm_main/0001/checkpoint-20000

.PHONY: setup ckpts index davis test-set data status manifest verify stats preview eval-segmenter train train-smoke validate eval test lint

setup:            ## create the venv and install everything
	bash setup/setup_env.sh

ckpts:            ## download all off-the-shelf models
	PYTHON=$(PYTHON) bash setup/download_checkpoints.sh

index:            ## rank OpenVid-1M zip parts by yield (needs data/raw/openvid/OpenVid-1M.csv)
	$(PYTHON) -c "from pathlib import Path; from stm.data_generation.sources.openvid import load_selection; from stm.data_generation.sources.openvid_index import index_parts, rank_parts; import json, re, requests; \
	sel=load_selection(Path('data/raw/openvid/OpenVid-1M.csv'), min_motion=2.0); \
	tree=requests.get('https://huggingface.co/api/datasets/nkp37/OpenVid-1M/tree/main', timeout=60).json(); \
	parts=sorted(int(m.group(1)) for x in tree if (m:=re.match(r'OpenVid_part(\d+)\.zip$$', x['path']))); \
	idx=index_parts(parts, Path('data/raw/openvid/part_index.json')); rows=rank_parts(idx, set(sel)); \
	json.dump(rows, open('data/raw/openvid/part_ranking_m2.0.json','w'), indent=1); print(rows[:5])"

davis:            ## DAVIS-2017 -> validation clips with GT masks
	$(PYTHON) data_processing/generate_data.py import-davis

test-set:         ## the paper's 93 test triplets from the released results
	$(PYTHON) data_processing/generate_data.py import-test-set

data:             ## unattended Decomposer run over OpenVid-1M (GPUS=...) until TARGET clips
	PYTHON=$(PYTHON) GPUS="$(GPUS)" TARGET_DONE=$(TARGET) nohup bash data_processing/run_openvid_pipeline.sh > data/openvid_driver.log 2>&1 &
	@echo "driver started; tail -f data/openvid_driver.log"

status:
	$(PYTHON) data_processing/generate_data.py status

manifest:
	$(PYTHON) data_processing/generate_data.py manifest

verify:
	$(PYTHON) data_processing/generate_data.py verify

stats:
	$(PYTHON) data_processing/generate_data.py stats

preview:          ## self-contained results page + contact sheet
	$(PYTHON) data_processing/make_preview.py --target $(TARGET)

eval-segmenter:   ## mask IoU of Segment-Any-Motion vs DAVIS GT
	$(PYTHON) data_processing/generate_data.py eval-segmenter --sources davis --gpus 0

train:            ## paper recipe (needs >= 4 GPUs for full fine-tuning)
	PYTHON=$(PYTHON) bash training/train.sh $(CONFIG)

train-smoke:      ## LoRA smoke test on one GPU (dataset -> loss -> checkpoint -> validation)
	PYTHON=$(PYTHON) bash training/train.sh training/configs/recipes/smoke_lora.conf

validate:         ## sample the test set from a checkpoint
	$(PYTHON) -m accelerate.commands.launch --config_file training/configs/accelerate/single_gpu.yaml inference/compose.py \
	  --checkpoint_path $(CKPT) --validation_dir data/datasets/stm_test --output_dir outputs/validation

eval:             ## M1-M5 on a results folder (RESULTS=..., OUT=...)
	$(PYTHON) evaluation/run_eval.py --results $(RESULTS) --out $(OUT)

test:             ## CPU checks
	$(PYTHON) -m pytest tests -q

lint:
	$(PYTHON) -m ruff check stm data_processing training inference evaluation tests && $(PYTHON) -m ruff format --check stm data_processing training inference evaluation tests
