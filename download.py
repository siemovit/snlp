from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL_REPO = "Qwen/Qwen3-0.6B"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download the base model for the SNLP steering experiments."
    )
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--models-dir", default="models")
    return parser.parse_args()


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_dir = models_dir / "qwen3-0.6b"
    print(f"Downloading model from {args.model_repo} -> {model_dir}")
    snapshot_download(repo_id=args.model_repo, local_dir=str(model_dir))
    print("Model download complete.")
    print(
        "SAE checkpoints are loaded separately by sae-lens via "
        "SAE.from_pretrained(release, sae_id)."
    )


if __name__ == "__main__":
    main()
