from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from part_6.steering_utils import (
    build_patch_specs,
    build_sv_bank_and_gates,
    lm_ce_loss_on_text,
    target_token_ce_from_prompt,
)
from utils import (
    LANG_CODE_TO_NAME,
    MODEL_PRESETS,
    TARGET_LANGS,
    ensure_dir,
    flatten_language_texts,
    get_device,
    get_safe_default_device,
    load_model_and_tokenizer,
    load_multilingual_dataframe,
    model_tag,
    repo_root,
    resolve_model_artifacts,
)


def split_lang_blocks(multilingual_texts, target_lan, train_n=50, eval_n=30):
    by_lang = {}
    for i, code in enumerate(target_lan):
        block = multilingual_texts[i * 100 : (i + 1) * 100]
        by_lang[code] = {
            "train": block[:train_n],
            "eval": block[train_n : train_n + eval_n],
        }
    return by_lang


def parse_args():
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Reproduce the notebook compact Adversarial LID cell as closely as possible."
    )
    parser.add_argument("--model-name", choices=sorted(MODEL_PRESETS), default="qwen")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--sae-release", default=None)
    parser.add_argument("--dataset-path", default=str(root / "data" / "multilingual_data.jsonl"))
    parser.add_argument("--source-lang", default="fr")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--base-layer", type=int, default=18)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--train-n", type=int, default=20)
    parser.add_argument("--eval-n", type=int, default=5)
    parser.add_argument("--gate-topk", type=int, default=2)
    parser.add_argument("--gate-threshold", type=float, default=0.0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Use 'mps' to match the notebook if you are on Apple Silicon.",
    )
    return parser.parse_args()


def build_lid_prompt(text):
    return (
        "Identify the language of the following text in one word.\n"
        f"Text: {text}\n"
        "Language:"
    )


def main():
    args = parse_args()
    model_label, model_path, sae_release = resolve_model_artifacts(
        repo_root(), args.model_name, args.model_path, args.sae_release
    )
    model_file_tag = model_tag(args.model_name, model_path)
    device = get_safe_default_device() if args.device == "auto" else args.device
    results_dir = ensure_dir(repo_root() / "results")
    csv_dir = ensure_dir(results_dir / "csv")
    plots_dir = ensure_dir(results_dir / "plots")

    model, tokenizer = load_model_and_tokenizer(model_path, device=device)
    data = load_multilingual_dataframe(args.dataset_path)
    multilingual_texts = flatten_language_texts(
        {code: data.loc[data["lan"] == code, "text"].tolist() for code in TARGET_LANGS},
        TARGET_LANGS,
    )

    by_lang = split_lang_blocks(multilingual_texts, TARGET_LANGS, train_n=args.train_n, eval_n=args.eval_n)
    pos_texts = by_lang[args.target_lang]["train"]
    neg_texts = by_lang[args.source_lang]["train"]
    window_layers = [args.base_layer, args.base_layer + 1, args.base_layer + 2]

    sv_bank = {}
    gate_bank = {}
    sv_bank, gate_bank = build_sv_bank_and_gates(
        window_layers,
        model,
        tokenizer,
        pos_texts,
        neg_texts,
        TARGET_LANGS,
        multilingual_texts,
        args.source_lang,
        sae_release,
        device=device,
        train_n=args.train_n,
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

    def method_to_specs(method_name, k):
        if method_name == "No SV":
            return []
        base = "SAE" if method_name.startswith("SAE") else "SV"
        return build_patch_specs(
            base,
            k,
            args.base_layer,
            model,
            sv_bank,
            gate_bank,
            alpha=args.alpha,
            sae_release=sae_release,
            sae_device=device,
            gate_threshold=args.gate_threshold,
        )

    rows = []
    target_word = LANG_CODE_TO_NAME[args.target_lang]
    eval_source = by_lang[args.source_lang]["eval"]
    other_non_target = []
    for code in TARGET_LANGS:
        if code != args.source_lang:
            other_non_target.extend(by_lang[code]["eval"])

    for method_name, k in methods:
        patch_specs = method_to_specs(method_name, k)
        ce_target = []
        for text in eval_source:
            ce_target.append(target_token_ce_from_prompt(model, tokenizer, build_lid_prompt(text), target_word, device, patch_specs))
        ce_other = []
        for text in other_non_target:
            ce_other.append(lm_ce_loss_on_text(model, tokenizer, text, device, patch_specs))
        rows.append(
            {
                "method": method_name,
                "k": k,
                "ce_target_token": float(np.nanmean(ce_target)),
                "ce_non_target_langs": float(np.nanmean(ce_other)),
            }
        )

    lid_df = pd.DataFrame(rows)
    print(lid_df.sort_values(["method"]))

    plot_order = ["SAE-1L", "SAE-2L", "SAE-3L", "SV-1L", "SV-2L", "SV-3L", "No SV"]
    plot_df = lid_df.set_index("method").loc[plot_order].reset_index()
    sae_df = plot_df[plot_df["method"].str.startswith("SAE")].sort_values("k")
    sv_df = plot_df[plot_df["method"].str.startswith("SV")].sort_values("k")
    no_sv_df = plot_df[plot_df["method"] == "No SV"]

    run_tag = f"alpha{args.alpha:g}_train{args.train_n}_eval{args.eval_n}"
    csv_path = csv_dir / f"notebook_lid_{model_file_tag}_{args.source_lang}_to_{args.target_lang}_{run_tag}.csv"
    fig_path = plots_dir / f"notebook_lid_{model_file_tag}_{args.source_lang}_to_{args.target_lang}_{run_tag}.png"
    lid_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8, 6))
    plt.plot(sae_df["ce_non_target_langs"], sae_df["ce_target_token"], color="green", marker="o", linewidth=1.6, label="SAE")
    plt.plot(sv_df["ce_non_target_langs"], sv_df["ce_target_token"], color="blue", marker="s", linewidth=1.6, label="SV")
    plt.scatter(no_sv_df["ce_non_target_langs"], no_sv_df["ce_target_token"], color="red", marker="D", s=90, label="No SV", zorder=3)
    for _, row in plot_df.iterrows():
        plt.text(row["ce_non_target_langs"] + 0.01, row["ce_target_token"] + 0.01, row["method"], fontsize=9)
    plt.xlabel(f"CE loss on Flores-10 without {LANG_CODE_TO_NAME[args.source_lang]}")
    plt.ylabel(f"CE loss for target token ({target_word})")
    plt.title(f"Adversarial LID ({model_label}, notebook): {args.source_lang} -> {args.target_lang}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
