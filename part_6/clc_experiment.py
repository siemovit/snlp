from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm

from part_6.steering_utils import (
    build_cont_prompt,
    build_learned_gate_bank,
    build_patch_specs,
    build_sv_bank_and_gates,
    first_n_words,
    generate_continuation,
    lm_ce_loss_on_text,
    normalize_openlid_label,
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
    load_model_and_tokenizer,
    load_multilingual_dataframe,
    model_tag,
    repo_root,
    resolve_model_artifacts,
)

DEFAULT_ALPHA_BY_SOURCE_LANG = {
    "es": 0.5,
    "fr": 0.5,
    "ja": 1.0,
    "ko": 0.5,
    "pt": 0.5,
    "th": 0.5,
    "vi": 0.5,
    "zh": 0.5,
    "ar": 0.5,
}


def load_lid_predictor(model_id: str, device: str):
    """Load OpenLID-v2 through fastText when requested, otherwise fall back to a Transformers pipeline."""
    if model_id == "laurievb/OpenLID-v2":
        try:
            import fasttext
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "OpenLID-v2 is a fastText model, not a standard Transformers model. "
                "Install the required dependency in the project environment, e.g. "
                "`uv add fasttext`, then rerun `part_6.clc_experiment`."
            ) from exc

        try:
            from openlid_normer import clean_line  # type: ignore
        except ImportError:
            def clean_line(text: str) -> str:
                text = re.sub(r"\s+", " ", text).strip()
                return text

        model_path = hf_hub_download(repo_id=model_id, filename="model.bin")
        lid_model = fasttext.load_model(model_path)

        def predict_fn(text: str) -> str:
            cleaned = clean_line(text)
            labels, _scores = lid_model.predict(cleaned or text, k=1)
            return labels[0] if labels else ""

        return predict_fn

    from transformers import pipeline

    lid_pipe = pipeline(
        "text-classification",
        model=model_id,
        device=0 if "cuda" in device else -1,
    )

    def predict_fn(text: str) -> str:
        pred = lid_pipe(text[:1200], truncation=True, top_k=1)
        if isinstance(pred, list) and pred and isinstance(pred[0], dict):
            return pred[0].get("label", "")
        if isinstance(pred, list) and pred and isinstance(pred[0], list):
            return pred[0][0].get("label", "")
        return ""

    return predict_fn


def parse_args():
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run Cross-Lingual Continuation steering experiments.")
    parser.add_argument("--model-name", choices=sorted(MODEL_PRESETS), default="gemma-2-2b")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--sae-release", default=None)
    parser.add_argument("--dataset-path", default=str(root / "data" / "multilingual_data_test.jsonl"))
    parser.add_argument("--lid-model", default="laurievb/OpenLID-v2")
    parser.add_argument("--source-lang", default="fr")
    parser.add_argument(
        "--source-langs",
        nargs="*",
        default=None,
        help="Optional list of source languages. Defaults to all TARGET_LANGS except target-lang.",
    )
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--base-layer", type=int, default=20)
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Global steering scale. If omitted, uses a source-language-specific value from DEFAULT_ALPHA_BY_SOURCE_LANG.",
    )
    parser.add_argument("--gate-topk", type=int, default=2)
    parser.add_argument("--gate-threshold", type=float, default=0.0)
    parser.add_argument("--learned-gating", action="store_true", help="Evaluate a learned SAE gate in addition to SV-1L and SAE-3L.")
    parser.add_argument("--learned-train-n", type=int, default=5)
    parser.add_argument("--learned-epochs", type=int, default=25)
    parser.add_argument("--learned-lr", type=float, default=0.1)
    parser.add_argument("--learned-collateral-weight", type=float, default=0.2)
    parser.add_argument("--train-n", type=int, default=20)
    parser.add_argument("--eval-n", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--n-words-for-lid", type=int, default=20)
    parser.add_argument(
        "--cache-dir",
        default=str(root / "cache"),
        help="Directory used to cache per-layer steering vectors and top-index tensors.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable loading/saving the local cache for sv_bank and top_idx_layer.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_label, model_path, sae_release = resolve_model_artifacts(
        repo_root(), args.model_name, args.model_path, args.sae_release
    )
    model_file_tag = model_tag(args.model_name, model_path)
    cache_dir = None if args.no_cache else Path(args.cache_dir)
    device = get_device()
    results_dir = ensure_dir(repo_root() / "results")
    csv_dir = ensure_dir(results_dir / "csv")

    # Load the language identifier model (OpenLID-v2 by default) and the main language model, along with its tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, device=device)
    lid_predict = load_lid_predictor(args.lid_model, device)

    # Load the multilingual dataset
    data = load_multilingual_dataframe(args.dataset_path) # this is a dataframe with columns "language" and "text"
    
    # Groups the dataframe by language code
    lang_texts = build_language_texts(data, TARGET_LANGS)
    
    # Create a non-overlapping train/eval split for each language, with the specified number of examples for training and evaluation
    by_lang = build_lang_split(lang_texts, train_n=args.train_n, eval_n=args.eval_n, target_langs=TARGET_LANGS) # 
    
    # for SAE: Concatenate all per-language text lists into one single list, preserving language block order
    multilingual_texts = flatten_language_texts(lang_texts, TARGET_LANGS)
    
    # Determine which source languages to run on
    source_langs = args.source_langs or [code for code in TARGET_LANGS if code != args.target_lang]

    methods = [
        ("SV-1L", 1),
        ("SAE-3L", 3),
    ]
    if args.learned_gating:
        methods.append(("Learned-3L", 3))

    per_source_sizes = []
    for source_lang in source_langs:
        source_eval_texts = by_lang[source_lang]["eval"]
        non_source_texts = []
        for code in TARGET_LANGS:
            if code != source_lang:
                non_source_texts.extend(by_lang[code]["eval"])
        per_source_sizes.append((source_lang, len(source_eval_texts), len(non_source_texts)))

    rows = []
    total_steps = sum(len(methods) * (n_source + n_other) for _, n_source, n_other in per_source_sizes)
    pbar = tqdm.tqdm(total=total_steps, desc="CLC pipeline", unit="text")
    for source_lang in source_langs:
        alpha = args.alpha if args.alpha is not None else DEFAULT_ALPHA_BY_SOURCE_LANG.get(source_lang, 0.5)
        window = window_layers(args.base_layer, 3, model)
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
            cache_dir=cache_dir,
            cache_metadata={
                "model_path": str(Path(model_path).resolve()),
                "dataset_path": str(Path(args.dataset_path).resolve()),
                "source_lang": source_lang,
                "target_lang": args.target_lang,
                "base_layer": args.base_layer,
                "train_n": args.train_n,
                "sae_release": sae_release,
                "target_lan": TARGET_LANGS,
            },
            gate_topk=args.gate_topk,
        )
        learned_gate_bank = None
        if args.learned_gating:
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
                sae_device=device,
                sae_dtype=next(model.parameters()).dtype if device != "cpu" else torch.float32,
                cache_dir=cache_dir,
                cache_metadata={
                    "model_path": str(Path(model_path).resolve()),
                    "dataset_path": str(Path(args.dataset_path).resolve()),
                    "source_lang": source_lang,
                    "target_lang": args.target_lang,
                    "base_layer": args.base_layer,
                    "train_n": args.train_n,
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

        source_eval_texts = by_lang[source_lang]["eval"]
        non_source_texts = []
        for code in TARGET_LANGS:
            if code != source_lang:
                non_source_texts.extend(by_lang[code]["eval"])

        source_rows = []
        for method_name, k in methods:
            patch_specs = build_patch_specs(
                method_name,
                k,
                args.base_layer,
                model,
                sv_bank,
                gate_bank,
                learned_gate_bank=learned_gate_bank,
                alpha=alpha,
                sae_release=sae_release,
                sae_device=device,
                sae_dtype=next(model.parameters()).dtype if device != "cpu" else torch.float32,
                gate_threshold=args.gate_threshold,
            )

            ok = 0
            total = 0
            for text in source_eval_texts:
                prompt = build_cont_prompt(text, LANG_CODE_TO_NAME[args.target_lang])
                full = generate_continuation(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                    patch_specs=patch_specs,
                )
                continuation = full.split("Continuation:")[-1].strip()
                continuation_short = first_n_words(continuation, n_words=args.n_words_for_lid)
                label = lid_predict(continuation_short)
                ok += int(normalize_openlid_label(label) == args.target_lang)
                total += 1
                pbar.set_postfix_str(f"{source_lang} {method_name} source")
                pbar.update(1)

            ce_collateral = [
                lm_ce_loss_on_text(model, tokenizer, text, device, patch_specs)
                for text in non_source_texts
            ]
            pbar.set_postfix_str(f"{source_lang} {method_name} collateral")
            pbar.update(len(non_source_texts))

            row = {
                "source_lang": source_lang,
                "target_lang": args.target_lang,
                "method": method_name,
                "k": k,
                "alpha_used": alpha,
                "success_rate": ok / max(total, 1),
                "ce_non_source_flores10": float(pd.Series(ce_collateral).mean()),
            }
            rows.append(row)
            source_rows.append(row)
            del patch_specs
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    pbar.close()

    df = pd.DataFrame(rows)
    alpha_tag = f"alpha{args.alpha:g}" if args.alpha is not None else "alpha-by-lang"
    run_tag = f"{alpha_tag}_train{args.train_n}_eval{args.eval_n}"
    source_tag = "all" if args.source_langs else args.source_lang
    csv_path = csv_dir / f"clc_{model_file_tag}_{source_tag}_to_{args.target_lang}_{run_tag}.csv"
    df.to_csv(csv_path, index=False)

    print(df.sort_values(["source_lang", "method"]))
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
