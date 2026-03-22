# SNLP Steering Experiments

This repository extracts the steering-vector part of `SNLP_with_steering.ipynb` into a small, runnable project. It keeps only the utilities needed for the Part 6 experiments:

- steering vectors and SAE-gated steering
- Adversarial Language Identification
- Cross-Lingual Continuation
- Flores-10 style data splits and top-feature extraction

## Overview

The code is organized around two experiment entry points in `part_6/`:

- `part_6/baseline_experiment.py`: one-layer toy steering baseline
- `part_6/lid_experiment.py`: Adversarial Language Identification
- `part_6/clc_experiment.py`: Cross-Lingual Continuation

Shared notebook logic was moved into:

- `utils.py`: dataset/model helpers and `compute_top_index_per_lan_for_layer`
- `part_6/steering_utils.py`: steering, gating, prompting, CE evaluation, and generation helpers
- `download.py`: downloads a supported base model preset

Data lives in `data/`, generated CSV/PNG outputs are written to `results/`, and reusable local artifacts are stored in `cache/`.

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
├── utils.py
├── pyproject.toml
└── README.md
```

## Quickstart

```bash
cd nlp/snlp
uv sync
uv run python download.py --model-name qwen
uv run python -m part_6.lid_experiment
```

## Setup

### Requirements

- Python 3.11+
- `uv`
- enough disk space for the model and SAE checkpoints
- optional GPU support for practical runtimes

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

## Downloading Model

The downloader supports these presets:

- `qwen`: model `Qwen/Qwen3-0.6B`, SAE release `mwhanna-qwen3-0.6b-transcoders-lowl0`
- `gemma-2-2b`: model `google/gemma-2-2b`, SAE release `google/gemma-scope-2b-pt-res`

```bash
uv run python download.py --model-name qwen
uv run python download.py --model-name gemma-2-2b
```

The model is stored under `models/`.

For SAE loading, the code follows the notebook pattern directly:

```python
release = "mwhanna-qwen3-0.6b-transcoders-lowl0"
sae_id = "layer_18"
sae = SAE.from_pretrained(release, sae_id).to(device)
target_layer = 18
```

The experiment scripts accept `--model-name` and automatically pick matching defaults for
`--model-path` and `--sae-release`. You can still override either of them manually.

## Running The Experiments

### Baseline Steering Demo

```bash
uv run python -m part_6.baseline_experiment \
  --model-name qwen \
  --source-lang fr \
  --target-lang es \
  --layer 18 \
  --alpha 20.0
```

Output:

- `results/baseline_qwen3-0.6b_fr_to_es_alpha20_train4_eval2.json`

### Adversarial Language Identification

```bash
uv run python -m part_6.lid_experiment \
  --model-name qwen \
  --source-lang fr \
  --target-lang en \
  --base-layer 18 \
  --alpha 10.0 \
  --device cpu
```

Outputs:

- `results/lid_qwen3-0.6b_fr_to_en_alpha10_train20_eval5_other5.csv`
- `results/lid_qwen3-0.6b_fr_to_en_alpha10_train20_eval5_other5.png`

For Gemma:

```bash
uv run python -m part_6.lid_experiment \
  --model-name gemma-2-2b \
  --source-lang fr \
  --target-lang ja \
  --base-layer 20 \
  --alpha 10.0 \
  --device cpu
```

For a larger paper-style run, pass the larger dataset explicitly and increase the sample counts:

```bash
uv run python -m part_6.lid_experiment \
  --dataset-path data/multilingual_data_test.jsonl \
  --train-n 100 \
  --eval-n 500 \
  --other-eval-n 500 \
  --device cpu
```

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

- `results/clc_qwen3-0.6b_fr_to_en_alpha10_train20_eval5.csv`
- `results/clc_qwen3-0.6b_fr_to_en_alpha10_train20_eval5.png`

## Notes

- The default `lid_experiment.py` settings are intentionally small for local safety; use `multilingual_data_test.jsonl` plus larger `--train-n/--eval-n/--other-eval-n` values for paper-style runs.
- `lid_experiment.py` defaults to CPU when `--device auto` is used on macOS, to avoid MPS disk-pressure crashes.
- `clc_experiment.py` uses `laurievb/OpenLID-v2` for language identification of continuations.
- The SAE loader expects a release string compatible with `SAE.from_pretrained(release, sae_id)`.
- The experiments are compute-heavy; CPU runs are possible but slow.
- Output filenames include the selected model tag so Qwen and Gemma runs do not overwrite each other.
