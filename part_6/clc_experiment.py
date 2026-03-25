from __future__ import annotations

import argparse
import gc
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm

from part_6.steering_utils import (
    build_cont_prompt,
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
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--base-layer", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gate-topk", type=int, default=2)
    parser.add_argument("--gate-threshold", type=float, default=0.0)
    parser.add_argument("--train-n", type=int, default=20)
    parser.add_argument("--eval-n", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--n-words-for-lid", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    model_label, model_path, sae_release = resolve_model_artifacts(
        repo_root(), args.model_name, args.model_path, args.sae_release
    )
    model_file_tag = model_tag(args.model_name, model_path)
    device = get_device()
    results_dir = ensure_dir(repo_root() / "results")
    csv_dir = ensure_dir(results_dir / "csv")
    plots_dir = ensure_dir(results_dir / "plots")

    model, tokenizer = load_model_and_tokenizer(model_path, device=device)
    lid_predict = load_lid_predictor(args.lid_model, device)

    data = load_multilingual_dataframe(args.dataset_path)
    lang_texts = build_language_texts(data, TARGET_LANGS)
    by_lang = build_lang_split(lang_texts, train_n=args.train_n, eval_n=args.eval_n, target_langs=TARGET_LANGS)
    multilingual_texts = flatten_language_texts(lang_texts, TARGET_LANGS)

    window = window_layers(args.base_layer, 3, model)
    sv_bank, gate_bank = build_sv_bank_and_gates(
        window,
        model,
        tokenizer,
        by_lang[args.target_lang]["train"],
        by_lang[args.source_lang]["train"],
        TARGET_LANGS,
        multilingual_texts,
        args.source_lang,
        sae_release,
        device,
        args.train_n,
        gate_topk=args.gate_topk,
    )

    methods = [
        ("No SV", 0),
        ("SV-1L", 1),
        ("SV-2L", 2),
        ("SV-3L", 3),
        ("SAE-1L", 1),
        ("SAE-2L", 2),
        ("SAE-3L", 3),
    ]

    source_eval_texts = by_lang[args.source_lang]["eval"]
    non_source_texts = []
    for code in TARGET_LANGS:
        if code != args.source_lang:
            non_source_texts.extend(by_lang[code]["eval"])

    rows = []
    total_steps = len(methods) * (len(source_eval_texts) + len(non_source_texts))
    pbar = tqdm.tqdm(total=total_steps, desc="CLC pipeline", unit="text")
    for method_name, k in methods:
        patch_specs = build_patch_specs(
            method_name,
            k,
            args.base_layer,
            model,
            sv_bank,
            gate_bank,
            alpha=args.alpha,
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
            pbar.set_postfix_str(f"eval {method_name} source")
            pbar.update(1)

        ce_collateral = [
            lm_ce_loss_on_text(model, tokenizer, text, device, patch_specs)
            for text in non_source_texts
        ]
        pbar.set_postfix_str(f"eval {method_name} collateral")
        pbar.update(len(non_source_texts))

        rows.append(
            {
                "method": method_name,
                "k": k,
                "success_rate": ok / max(total, 1),
                "ce_non_source_flores10": float(pd.Series(ce_collateral).mean()),
            }
        )
        del patch_specs
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    pbar.close()

    df = pd.DataFrame(rows)
    run_tag = f"alpha{args.alpha:g}_train{args.train_n}_eval{args.eval_n}"
    csv_path = csv_dir / f"clc_{model_file_tag}_{args.source_lang}_to_{args.target_lang}_{run_tag}.csv"
    fig_path = plots_dir / f"clc_{model_file_tag}_{args.source_lang}_to_{args.target_lang}_{run_tag}.png"
    df.to_csv(csv_path, index=False)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    x = np.arange(len(df))
    ax1.bar(x - 0.18, df["success_rate"].values, width=0.35, label="Success rate")
    ax2.bar(x + 0.18, df["ce_non_source_flores10"].values, width=0.35, color="tab:orange", label="CE (non-source)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["method"].values, rotation=30, ha="right")
    ax1.set_ylabel("Success rate (higher is better)")
    ax2.set_ylabel("CE on non-source Flores-10 (lower is better)")
    ax1.set_title(f"Cross-Lingual Continuation ({model_label}): {args.source_lang} -> {args.target_lang}")
    ax1.grid(True, axis="y", alpha=0.3)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(df.sort_values(["method"]))
    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
