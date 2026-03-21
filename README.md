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
- `download.py`: downloads the default base model

Data lives in `data/`, and generated CSV/PNG outputs are written to `results/`.

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
uv run python download.py
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

The default downloader fetches:

- model: `Qwen/Qwen3-0.6B`

```bash
uv run python download.py
```

The model is stored under `models/`.

For SAE loading, the code follows the notebook pattern directly:

```python
release = "mwhanna-qwen3-0.6b-transcoders-lowl0"
sae_id = "layer_18"
sae = SAE.from_pretrained(release, sae_id).to(device)
target_layer = 18
```

The experiment scripts therefore default to:

- `--sae-release mwhanna-qwen3-0.6b-transcoders-lowl0`

and rely on `sae-lens` to resolve the SAE checkpoints when needed.

## Running The Experiments

### Baseline Steering Demo

```bash
uv run python -m part_6.baseline_experiment \
  --source-lang fr \
  --target-lang es \
  --layer 18 \
  --alpha 20.0
```

Output:

- `results/baseline_fr_to_es.json`

### Adversarial Language Identification

```bash
uv run python -m part_6.lid_experiment \
  --source-lang fr \
  --target-lang en \
  --base-layer 18 \
  --alpha 10.0 \
  --device cpu
```

Outputs:

- `results/lid_fr_to_en.csv`
- `results/lid_fr_to_en.png`

For a larger paper-style run, pass the larger dataset explicitly and increase the sample counts:

```bash
uv run python -m part_6.lid_experiment \
  --dataset-path data/multilingual_data_test.jsonl \
  --train-n 100 \
  --eval-n 500 \
  --other-eval-n 500 \
  --device cpu
```

### Cross-Lingual Continuation

```bash
uv run python -m part_6.clc_experiment \
  --source-lang fr \
  --target-lang en \
  --base-layer 18 \
  --alpha 10.0
```

Outputs:

- `results/clc_fr_to_en.csv`
- `results/clc_fr_to_en.png`

## Notes

- The default `lid_experiment.py` settings are intentionally small for local safety; use `multilingual_data_test.jsonl` plus larger `--train-n/--eval-n/--other-eval-n` values for paper-style runs.
- `lid_experiment.py` defaults to CPU when `--device auto` is used on macOS, to avoid MPS disk-pressure crashes.
- `clc_experiment.py` uses `laurievb/OpenLID-v2` for language identification of continuations.
- The SAE loader expects a release string compatible with `SAE.from_pretrained(release, sae_id)`.
- The experiments are compute-heavy; CPU runs are possible but slow.
