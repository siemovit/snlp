# SNLP Steering Experiments

This repository reproduces the results from part 6 on the steering vectors from the paper Deng et al. 2025. There are code and experiment for:
- steering vectors and SAE-gated steering
- Adversarial Language Identification
- Cross-Lingual Continuation
- Flores-10 style data splits and top-feature extraction

## Overview

The code is organized around two experiment entry points in `part_6/`:

- `part_6/baseline_experiment.py`: one-layer toy steering baseline
- `part_6/lid_experiment.py`: Adversarial Language Identification
- `part_6/clc_experiment.py`: Cross-Lingual Continuation

Core and utils can be found into into:

- `utils.py`: dataset/model helpers and `compute_top_index_per_lan_for_layer`
- `part_6/steering_utils.py`: steering, gating, prompting, CE evaluation, and generation helpers
- `download.py`: downloads a supported base model preset

Data lives in `data/`, generated outputs are written under `results/`, and reusable local artifacts are stored in `cache/` (for building SV data bank for instance, accross experiments). 

## Repository Layout

```text
snlp/
├── data/
├── download.py
├── part_6/
│   ├── clc_experiment.py
│   ├── baseline_experiment.py
│   ├── lid_experiment.py
│   └── steering_utils.py
├── results/
│   ├── csv/
│   ├── json/
│   └── plots/
├── utils.py
├── pyproject.toml
└── README.md
```

## Quickstart

```bash
cd nlp/snlp
uv sync
uv run python download.py --model-name gemma-2-2b
uv run python -m part_6.lid_experiment
```

## Setup

### Requirements

- Python 3.11+
- `uv`
- enough disk space for the model, SAE checkpoints are not downloaded directly, only the needed are used on the fly.  
- GPU support

### Installation

```bash
cd nlp/snlp
uv sync
```

### Dependencies

Dependencies are declared in `pyproject.toml` and installed through `uv`. The main runtime packages are:

- `torch`
- `transformers`
- `sae-lens`
- `pandas`
- `numpy`
- `matplotlib`
- `huggingface-hub`

## Data

The repository includes:

- `data/multilingual_data.jsonl`
- `data/multilingual_data_test.jsonl`

The repository includes both a small and a larger dataset:

- `multilingual_data.jsonl`: lightweight default for debugging and local runs
- `multilingual_data_test.jsonl`: larger evaluation set for paper-style experiments

The larger file is large enough for the paper-style split:

- first `100` samples per language for steering/gate construction
- next `500` non-overlapping samples per language for evaluation

## Run

### Downloading the model

The downloader supports these presets:

- `qwen`: model `Qwen/Qwen3-0.6B`, SAE release `mwhanna-qwen3-0.6b-transcoders-lowl0`
- `gemma-2-2b`: model `google/gemma-2-2b`, SAE release `gemma-scope-2b-pt-res`

```bash
uv run python download.py --model-name qwen
uv run python download.py --model-name gemma-2-2b
```

The model is stored under `models/`.

The experiment scripts accept `--model-name` and automatically pick matching defaults for
`--model-path` and `--sae-release`. You can still override either of them manually.

Default base layer for SAE is 20. 

## Running The Experiments

### Baseline Steering Demo

```bash
uv run python -m part_6.baseline_experiment \
  --model-name qwen \
  --source-lang fr \
  --target-lang en \
  --base-layer 20 \
  --alpha 0.5
```

Output:

- `results/json/baseline_qwen3-0.6b_fr_to_es_alpha20_train4_eval2.json`

### Adversarial Language Identification

```bash
uv run python -m part_6.lid_experiment \
  --model-name gemma-2-2b \
  --source-lang fr \
  --target-lang en \
  --base-layer 20 \
  --alpha 0.5 \
  --train-n 20 \
  --eval-n 10 \
  --target-metric first-token \
  --device cuda \
  --dtype bfloat16 \
  --sae-device cuda
```

Outputs:

- `results/csv/lid_gemma-2-2b_fr_to_en_alpha0.5_train20_eval10_other10_first-token_<sha>.csv`
- `results/plots/lid_gemma-2-2b_fr_to_en_alpha0.5_train20_eval10_other10_first-token_<sha>.png`

This matches the current defaults closely enough that you can usually just run:

```bash
uv run python -m part_6.lid_experiment
```

For a larger paper-style run, pass the larger dataset explicitly and increase the sample counts:

```bash
uv run python -m part_6.lid_experiment \
  --dataset-path data/multilingual_data_test.jsonl \
  --train-n 100 \
  --eval-n 500 \
  --other-eval-n 500 \
  --device cuda
```

> [!WARNING]  
> Putting 100 for --train-n may make the GPU or memory blow up. On a Tesla V100, it was the case. 

`lid_experiment.py` caches per-layer `sv_bank` and `top_idx_layer` tensors under `cache/` by default. Disable that behavior with:

```bash
uv run python -m part_6.lid_experiment --no-cache
```

### Cross-Lingual Continuation

```bash
uv run python -m part_6.clc_experiment \
  --model-name qwen \
  --source-lang fr \
  --target-lang en \
  --base-layer 18 \
  --alpha 10.0
```

Outputs:

- `results/csv/clc_qwen3-0.6b_fr_to_en_alpha10_train20_eval5.csv`
- `results/plots/clc_qwen3-0.6b_fr_to_en_alpha10_train20_eval5.png`

## Notes

- Current `lid_experiment.py` defaults are:
  - `model-name=gemma-2-2b`
  - `dataset-path=data/multilingual_data_test.jsonl`
  - `source-lang=fr`
  - `target-lang=en`
  - `base-layer=20`
  - `alpha=0.5`
  - `train-n=20`
  - `eval-n=10`
  - `other-eval-n=eval-n`
  - `target-metric=first-token`
  - `device=cuda`
  - `dtype=bfloat16`
  - `sae-device=cuda`
- The default `lid_experiment.py` settings are still much smaller than the paper-style split; use `multilingual_data_test.jsonl` plus larger `--train-n/--eval-n/--other-eval-n` values for paper-style runs.
- `lid_experiment.py` defaults to CPU when `--device auto` is used on macOS, to avoid MPS disk-pressure crashes.
- `clc_experiment.py` uses `laurievb/OpenLID-v2` for language identification of continuations.
- The SAE loader expects a release string compatible with `SAE.from_pretrained(release, sae_id)`.
- The experiments are compute-heavy; CPU runs are possible but slow.
- Output filenames include the selected model tag and the current commit SHA so runs from different code states do not overwrite each other.
- Steering vectors are now used without L2 normalization in `lid_experiment.py`, `clc_experiment.py`, and `baseline_experiment.py`.
