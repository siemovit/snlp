from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from utils import MODEL_PRESETS, repo_root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a supported base model for the SNLP steering experiments."
    )
    parser.add_argument("--model-name", choices=sorted(MODEL_PRESETS), default="qwen")
    parser.add_argument("--model-repo", default=None, help="Optional override for the model repo id.")
    parser.add_argument("--models-dir", default=None, help="Optional override for the local models directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    preset = MODEL_PRESETS[args.model_name]
    if args.models_dir is None:
        models_dir = repo_root() / "models"
    else:
        models_dir = Path(args.models_dir)
        if not models_dir.is_absolute():
            models_dir = repo_root() / models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    model_repo = args.model_repo or preset["repo_id"]
    model_dir = models_dir / preset["model_dir"]
    print(f"Downloading {preset['label']} from {model_repo} -> {model_dir}")
    snapshot_download(repo_id=model_repo, local_dir=str(model_dir))
    print("Model download complete.")
    print(
        "SAE checkpoints are loaded separately by sae-lens via "
        "SAE.from_pretrained(release, sae_id)."
    )
    print(f"Default SAE release for this model: {preset['sae_release']}")


if __name__ == "__main__":
    main()
