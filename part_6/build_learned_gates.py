from __future__ import annotations

import argparse
from pathlib import Path

import torch

from part_6.steering_utils import (
    build_learned_gate_bank,
    build_sv_bank_and_gates,
    learned_gate_bank_filename,
    save_learned_gate_bank,
    window_layers,
)
from utils import (
    LANG_CODE_TO_NAME,
    MODEL_PRESETS,
    TARGET_LANGS,
    build_lang_split,
    build_language_texts,
    ensure_dir,
    flatten_language_texts,
    get_device,
    get_safe_default_device,
    load_model_and_tokenizer,
    load_multilingual_dataframe,
    model_tag,
    repo_root,
    resolve_model_artifacts,
    resolve_torch_dtype,
)


def parse_args():
    root = repo_root()
    parser = argparse.ArgumentParser(description="Precompute and save learned SAE gate banks.")
    parser.add_argument("--model-name", choices=sorted(MODEL_PRESETS), default="gemma-2-2b")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--sae-release", default=None)
    parser.add_argument("--dataset-path", default=str(root / "data" / "multilingual_data_test.jsonl"))
    parser.add_argument("--source-lang", default=None)
    parser.add_argument(
        "--source-langs",
        nargs="*",
        default=None,
        help="Optional list of source languages. Defaults to all TARGET_LANGS except target-lang.",
    )
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--base-layer", type=int, default=20)
    parser.add_argument("--train-n", type=int, default=20)
    parser.add_argument("--gate-train-n", type=int, default=20)
    parser.add_argument("--gate-topk", type=int, default=2)
    parser.add_argument("--learned-train-n", type=int, default=5)
    parser.add_argument("--learned-epochs", type=int, default=25)
    parser.add_argument("--learned-lr", type=float, default=0.1)
    parser.add_argument("--learned-collateral-weight", type=float, default=0.2)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--sae-device", choices=["cpu", "mps", "cuda", "same"], default="cuda")
    parser.add_argument("--cache-dir", default=str(root / "cache"))
    parser.add_argument("--v-scores-csv", default=str(root / "results" / "csv" / "v_scores_run_reprod_fig_1_top5.csv"))
    parser.add_argument("--output-dir", default=str(root / "learned_gates"))
    return parser.parse_args()


def main():
    args = parse_args()
    root = repo_root()
    _model_label, model_path, sae_release = resolve_model_artifacts(root, args.model_name, args.model_path, args.sae_release)
    model_file_tag = model_tag(args.model_name, model_path)
    device = get_safe_default_device() if args.device == "auto" else args.device
    available_device = get_device()
    if device == "cuda" and available_device != "cuda":
        raise ValueError("CUDA was requested but is not available on this machine.")
    sae_device = device if args.sae_device == "same" else args.sae_device
    model_dtype = resolve_torch_dtype(args.dtype, device)
    cache_dir = Path(args.cache_dir)
    output_dir = ensure_dir(args.output_dir)
    v_scores_csv = Path(args.v_scores_csv) if args.v_scores_csv else None
    if v_scores_csv is not None and not v_scores_csv.exists():
        v_scores_csv = None

    model, tokenizer = load_model_and_tokenizer(model_path, device=device, dtype=args.dtype)
    data = load_multilingual_dataframe(args.dataset_path)
    lang_texts = build_language_texts(data, TARGET_LANGS)
    by_lang = build_lang_split(lang_texts, train_n=args.train_n, eval_n=1, target_langs=TARGET_LANGS)
    multilingual_texts = flatten_language_texts(lang_texts, TARGET_LANGS)

    if args.source_langs is not None:
        source_langs = args.source_langs
    elif args.source_lang is not None:
        source_langs = [args.source_lang]
    else:
        source_langs = [code for code in TARGET_LANGS if code != args.target_lang]

    window = window_layers(args.base_layer, 3, model)
    for source_lang in source_langs:
        sv_bank, gate_bank = build_sv_bank_and_gates(
            window,
            model,
            tokenizer,
            by_lang[args.target_lang]["train"],
            by_lang[source_lang]["train"],
            TARGET_LANGS,
            multilingual_texts,
            source_lang,
            sae_release,
            device,
            args.train_n,
            gate_train_n=args.gate_train_n,
            sae_device=sae_device,
            sae_dtype=model_dtype if sae_device != "cpu" else torch.float32,
            cache_dir=cache_dir,
            cache_metadata={
                "model_path": str(Path(model_path).resolve()),
                "dataset_path": str(Path(args.dataset_path).resolve()),
                "source_lang": source_lang,
                "target_lang": args.target_lang,
                "base_layer": args.base_layer,
                "train_n": args.train_n,
                "gate_train_n": args.gate_train_n,
                "sae_release": sae_release,
                "target_lan": TARGET_LANGS,
            },
            gate_topk=args.gate_topk,
            v_scores_csv=v_scores_csv,
        )
        learned_gate_bank = build_learned_gate_bank(
            window,
            model,
            tokenizer,
            TARGET_LANGS,
            multilingual_texts,
            source_lang,
            sae_release,
            device,
            args.train_n,
            sv_bank,
            gate_bank,
            sae_device=sae_device,
            sae_dtype=model_dtype if sae_device != "cpu" else torch.float32,
            cache_dir=cache_dir,
            cache_metadata={
                "model_path": str(Path(model_path).resolve()),
                "dataset_path": str(Path(args.dataset_path).resolve()),
                "source_lang": source_lang,
                "target_lang": args.target_lang,
                "base_layer": args.base_layer,
                "train_n": args.train_n,
                "gate_train_n": args.gate_train_n,
                "sae_release": sae_release,
                "target_lan": TARGET_LANGS,
                "gate_topk": args.gate_topk,
                "learned_train_n": args.learned_train_n,
                "learned_epochs": args.learned_epochs,
                "learned_lr": args.learned_lr,
                "learned_collateral_weight": args.learned_collateral_weight,
            },
            target_word=LANG_CODE_TO_NAME[args.target_lang],
            learned_train_n=args.learned_train_n,
            learned_epochs=args.learned_epochs,
            learned_lr=args.learned_lr,
            learned_collateral_weight=args.learned_collateral_weight,
        )
        output_name = learned_gate_bank_filename(
            model_file_tag,
            source_lang,
            args.target_lang,
            base_layer=args.base_layer,
            gate_topk=args.gate_topk,
            train_n=args.train_n,
            gate_train_n=args.gate_train_n,
            learned_train_n=args.learned_train_n,
            learned_epochs=args.learned_epochs,
            learned_lr=args.learned_lr,
            learned_collateral_weight=args.learned_collateral_weight,
        )
        output_path = output_dir / output_name
        save_learned_gate_bank(
            output_path,
            learned_gate_bank,
            metadata={
                "model_path": str(Path(model_path).resolve()),
                "dataset_path": str(Path(args.dataset_path).resolve()),
                "source_lang": source_lang,
                "target_lang": args.target_lang,
                "base_layer": args.base_layer,
                "layers": window,
                "train_n": args.train_n,
                "gate_train_n": args.gate_train_n,
                "gate_topk": args.gate_topk,
                "learned_train_n": args.learned_train_n,
                "learned_epochs": args.learned_epochs,
                "learned_lr": args.learned_lr,
                "learned_collateral_weight": args.learned_collateral_weight,
            },
        )
        print(f"Saved learned gate bank: {output_path}")


if __name__ == "__main__":
    main()
