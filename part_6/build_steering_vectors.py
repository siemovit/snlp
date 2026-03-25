from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import tqdm

from part_6.steering_utils import compute_steering_vector
from utils import (
    MODEL_PRESETS,
    TARGET_LANGS,
    build_language_texts,
    ensure_dir,
    get_safe_default_device,
    load_model_and_tokenizer,
    load_multilingual_dataframe,
    model_tag,
    repo_root,
    resolve_model_artifacts,
)


def parse_args():
    root = repo_root()
    parser = argparse.ArgumentParser(description="Build and store steering vectors from multilingual_data.jsonl.")
    parser.add_argument("--model-name", choices=sorted(MODEL_PRESETS), default="gemma-2-2b")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--sae-release", default=None)
    parser.add_argument("--dataset-path", default=str(root / "data" / "multilingual_data.jsonl"))
    parser.add_argument("--target-lang", default="en")
    parser.add_argument(
        "--source-langs",
        nargs="*",
        default=None,
        help="Optional list of source languages. Defaults to all TARGET_LANGS except target-lang.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[20, 21, 22],
        help="List of layer indices for which steering vectors are computed and stored.",
    )
    parser.add_argument("--train-n", type=int, default=100)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument(
        "--output-dir",
        default=str(root / "steering_vectors"),
        help="Directory where precomputed steering-vector banks are stored.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_label, model_path, _sae_release = resolve_model_artifacts(repo_root(), args.model_name, args.model_path, args.sae_release)
    model_file_tag = model_tag(args.model_name, model_path)
    device = get_safe_default_device() if args.device == "auto" else args.device
    output_dir = ensure_dir(args.output_dir)

    model, tokenizer = load_model_and_tokenizer(model_path, device=device, dtype=args.dtype)
    data = load_multilingual_dataframe(args.dataset_path)
    lang_texts = build_language_texts(data, TARGET_LANGS)

    source_langs = args.source_langs or [code for code in TARGET_LANGS if code != args.target_lang]
    target_texts = lang_texts[args.target_lang][: args.train_n]
    if len(target_texts) < args.train_n:
        raise ValueError(f"Target language {args.target_lang} has only {len(target_texts)} texts, expected {args.train_n}.")

    layers = sorted(dict.fromkeys(args.layers))
    for source_lang in source_langs:
        source_texts = lang_texts[source_lang][: args.train_n]
        if len(source_texts) < args.train_n:
            raise ValueError(f"Source language {source_lang} has only {len(source_texts)} texts, expected {args.train_n}.")

        sv_bank = {}
        pbar = tqdm.tqdm(layers, desc=f"Steering bank {source_lang}->{args.target_lang}", unit="layer")
        for layer_idx in pbar:
            sv_bank[layer_idx] = compute_steering_vector(
                model,
                tokenizer,
                target_texts,
                source_texts,
                layer_idx,
                device=device,
                normalize=False,
            ).detach().cpu()

        payload = {
            "metadata": {
                "model_name": model_label,
                "model_path": str(Path(model_path).resolve()),
                "dataset_path": str(Path(args.dataset_path).resolve()),
                "source_lang": source_lang,
                "target_lang": args.target_lang,
                "train_n": args.train_n,
                "layers": layers,
            },
            "sv_bank": sv_bank,
        }
        output_path = output_dir / f"sv_{model_file_tag}_{source_lang}_to_{args.target_lang}_train{args.train_n}.pt"
        torch.save(payload, output_path)
        meta_path = output_dir / f"sv_{model_file_tag}_{source_lang}_to_{args.target_lang}_train{args.train_n}.json"
        meta_path.write_text(json.dumps(payload["metadata"], indent=2, ensure_ascii=False))
        print(f"Saved steering bank: {output_path}")


if __name__ == "__main__":
    main()
