from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import tqdm
from matplotlib.lines import Line2D

from part_6.steering_utils import (
    build_lid_prompt,
    build_patch_specs,
    build_sv_bank_and_gates,
    lm_ce_loss_on_text,
    target_token_ce_from_prompt,
    window_layers,
)
from utils import (
    LANG_CODE_TO_NAME,
    MODEL_PRESETS,
    TARGET_LANGS,
    build_lang_split,
    build_language_texts,
    ensure_min_free_memory,
    ensure_dir,
    flatten_language_texts,
    get_device,
    get_device_memory_report,
    get_safe_default_device,
    load_model_and_tokenizer,
    load_multilingual_dataframe,
    model_tag,
    print_device_memory_report,
    repo_root,
    resolve_model_artifacts,
    resolve_torch_dtype,
)


def parse_args():
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run Adversarial LID steering experiments.")
    parser.add_argument("--model-name", choices=sorted(MODEL_PRESETS), default="qwen")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--sae-release", default=None)
    parser.add_argument("--dataset-path", default=str(root / "data" / "multilingual_data.jsonl"))
    parser.add_argument("--source-lang", default="fr")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--base-layer", type=int, default=18)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--gate-topk", type=int, default=2, help="Number of source-language SAE features used for gating.")
    parser.add_argument("--gate-threshold", type=float, default=0.0, help="Activation threshold for SAE gating.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Execution device. 'auto' defaults to CPU on macOS to avoid MPS graph crashes.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="Model dtype. 'auto' defaults to bfloat16 on CUDA and float32 otherwise.",
    )
    parser.add_argument(
        "--sae-device",
        choices=["cpu", "mps", "cuda", "same"],
        default="cpu",
        help="Where to keep SAE gates. 'cpu' is safer on small GPUs; 'same' follows --device.",
    )
    parser.add_argument("--train-n", type=int, default=20, help="Number of source/target samples per language used to build the steering bank.")
    parser.add_argument("--eval-n", type=int, default=5, help="Number of source-language evaluation samples.")
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=2.0,
        help="Stop early if available CUDA memory drops below this threshold.",
    )
    parser.add_argument(
        "--other-eval-n",
        type=int,
        default=None,
        help="Optional cap for collateral evaluation samples per non-source language. Defaults to --eval-n.",
    )
    parser.add_argument(
        "--verbose-memory",
        action="store_true",
        help="Print detailed CUDA memory reports during bank construction and method evaluation.",
    )
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
    device = get_safe_default_device() if args.device == "auto" else args.device
    available_device = get_device()
    sae_device = device if args.sae_device == "same" else args.sae_device
    model_dtype = resolve_torch_dtype(args.dtype, device)
    results_dir = ensure_dir(repo_root() / "results")
    csv_dir = ensure_dir(results_dir / "csv")
    plots_dir = ensure_dir(results_dir / "plots")
    other_eval_n = args.other_eval_n if args.other_eval_n is not None else args.eval_n
    split_eval_n = max(args.eval_n, other_eval_n)
    cache_dir = None if args.no_cache else Path(args.cache_dir)

    if device == "cuda" and available_device != "cuda":
        raise ValueError("CUDA was requested but is not available on this machine.")
    if device == "mps" and available_device != "mps":
        raise ValueError("MPS was requested but is not available on this machine.")

    # Load the local model and assemble the paper-style per-language split.
    model, tokenizer = load_model_and_tokenizer(model_path, device=device, dtype=args.dtype)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        if args.verbose_memory:
            print_device_memory_report(device, "After model load")
        ensure_min_free_memory(device, args.min_free_gb, "SV/gate construction")
    data = load_multilingual_dataframe(args.dataset_path)
    lang_texts = build_language_texts(data, TARGET_LANGS)
    by_lang = build_lang_split(lang_texts, train_n=args.train_n, eval_n=split_eval_n, target_langs=TARGET_LANGS)
    multilingual_texts = flatten_language_texts(lang_texts, TARGET_LANGS)

    methods = [
        ("No SV", 0),
        ("SV-1L", 1),
        ("SV-2L", 2),
        ("SV-3L", 3),
        ("SAE-1L", 1),
        ("SAE-2L", 2),
        ("SAE-3L", 3),
    ]

    total_bank_steps = len(window_layers(args.base_layer, 3, model)) * (2 * args.train_n + len(TARGET_LANGS) * args.train_n)
    total_eval_steps = len(methods) * (args.eval_n + (len(TARGET_LANGS) - 1) * other_eval_n)
    overall_pbar = tqdm.tqdm(total=total_bank_steps + total_eval_steps, desc="LID pipeline", unit="text")

    def step_progress(phase: str):
        overall_pbar.set_postfix_str(phase)
        overall_pbar.update(1)

    # Build the 1L/2L/3L steering vectors and SAE gates starting from the chosen base layer.
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
        sae_device=sae_device,
        sae_dtype=model_dtype if sae_device != "cpu" else torch.float32,
        memory_report_fn=(lambda msg: print_device_memory_report(device, msg)) if device == "cuda" and args.verbose_memory else None,
        progress_callback=lambda: step_progress("building banks/gates"),
        cache_dir=cache_dir,
        cache_metadata={
            "model_path": str(Path(model_path).resolve()),
            "dataset_path": str(Path(args.dataset_path).resolve()),
            "source_lang": args.source_lang,
            "target_lang": args.target_lang,
            "base_layer": args.base_layer,
            "train_n": args.train_n,
            "sae_release": sae_release,
            "target_lan": TARGET_LANGS,
        },
        gate_topk=args.gate_topk,
    )

    # Adversarial LID: steer source-language texts toward the target-language label,
    # and also measure collateral CE on all languages except the original/source one.
    target_word = LANG_CODE_TO_NAME[args.target_lang]
    eval_source = by_lang[args.source_lang]["eval"][: args.eval_n]
    other_non_target = []
    for code in TARGET_LANGS:
        if code != args.source_lang:
            other_non_target.extend(by_lang[code]["eval"][:other_eval_n])

    print(
        f"Running Adversarial LID with dataset={args.dataset_path}, "
        f"train_n={args.train_n}, source_eval_n={len(eval_source)}, "
        f"other_eval_n={other_eval_n}, base_layer={args.base_layer}, alpha={args.alpha}, "
        f"model={model_label}, "
        f"device={device}, dtype={args.dtype}, sae_device={sae_device}, "
        f"gate_topk={args.gate_topk}, gate_threshold={args.gate_threshold}, cache_dir={cache_dir}"
    )
    print(
        f"Source language: {args.source_lang} -> target language: {args.target_lang} | "
        f"collateral samples={len(other_non_target)} | "
        f"total prompt evaluations={len(methods) * (len(eval_source) + len(other_non_target))}"
    )
    if device == "cuda":
        report = get_device_memory_report(device)
        if report is not None:
            # A crude workload estimate: if we are already close to full before evaluation,
            # stop instead of dying deep in the loop.
            if args.verbose_memory:
                print_device_memory_report(device, "Before evaluation loop")
            ensure_min_free_memory(device, args.min_free_gb, "method evaluation")

    rows = []
    # Progress bar total number of steps is computed as follows:
    # 1. Bank construction: 2 * train_n for steering vector and 10*train_n for compute_top_index_per_lan_for_layer and for each layer (so x3)
    # 2. Evaluation: eval_n + 9 * other_eval_n, for each method and there are 7 methods.
    # 3. Total = bank steps + eval steps, as computed above.
    
    for method_name, k in methods:
        if device == "cuda":
            if args.verbose_memory:
                print_device_memory_report(device, f"Before method {method_name}")
            ensure_min_free_memory(device, args.min_free_gb, f"method {method_name}")

        # Convert the method label into the corresponding stack of steering patches.
        patch_specs = build_patch_specs(
            method_name,
            k,
            args.base_layer,
            model,
            sv_bank,
            gate_bank,
            alpha=args.alpha,
            sae_release=sae_release,
            sae_device=sae_device,
            sae_dtype=model_dtype if sae_device != "cpu" else torch.float32,
            gate_threshold=args.gate_threshold,
        )
        ce_target = []
        for text in eval_source:
            ce_target.append(
                target_token_ce_from_prompt(model, tokenizer, build_lid_prompt(text), target_word, device, patch_specs)
            )
            step_progress(f"eval {method_name} source")
        ce_other = []
        for text in other_non_target:
            ce_other.append(
                lm_ce_loss_on_text(model, tokenizer, text, device, patch_specs)
            )
            step_progress(f"eval {method_name} collateral")
        rows.append(
            {
                "method": method_name,
                "k": k,
                "ce_target_token": float(pd.Series(ce_target).mean()),
                "ce_non_target_langs": float(pd.Series(ce_other).mean()),
            }
        )
        if device == "cuda":
            if args.verbose_memory:
                print_device_memory_report(device, f"After method {method_name}")
            torch.cuda.empty_cache()
        del patch_specs
        gc.collect()
    overall_pbar.close()

    # Save both the raw table and the paper-style scatter plot.
    df = pd.DataFrame(rows)
    run_tag = (
        f"alpha{args.alpha:g}_train{args.train_n}_eval{args.eval_n}_other{other_eval_n}"
    )
    csv_path = csv_dir / f"lid_{model_file_tag}_{args.source_lang}_to_{args.target_lang}_{run_tag}.csv"
    fig_path = plots_dir / f"lid_{model_file_tag}_{args.source_lang}_to_{args.target_lang}_{run_tag}.png"
    df.to_csv(csv_path, index=False)

    # Match the notebook/paper convention: SAE in green, SV in blue, No SV in red.
    plot_order = ["SAE-1L", "SAE-2L", "SAE-3L", "SV-1L", "SV-2L", "SV-3L", "No SV"]
    plot_df = df.set_index("method").loc[plot_order].reset_index()
    sae_df = plot_df[plot_df["method"].str.startswith("SAE")].sort_values("k")
    sv_df = plot_df[plot_df["method"].str.startswith("SV")].sort_values("k")
    no_sv_df = plot_df[plot_df["method"] == "No SV"]

    marker_map = {1: "o", 2: "^", 3: "s"}

    plt.figure(figsize=(8, 6))
    plt.plot(sae_df["ce_non_target_langs"], sae_df["ce_target_token"], color="green", linewidth=1.6)
    plt.plot(sv_df["ce_non_target_langs"], sv_df["ce_target_token"], color="blue", linewidth=1.6)
    for _, row in sae_df.iterrows():
        plt.scatter(
            row["ce_non_target_langs"],
            row["ce_target_token"],
            color="green",
            marker=marker_map.get(int(row["k"]), "o"),
            s=70,
            zorder=3,
        )
    for _, row in sv_df.iterrows():
        plt.scatter(
            row["ce_non_target_langs"],
            row["ce_target_token"],
            color="blue",
            marker=marker_map.get(int(row["k"]), "o"),
            s=70,
            zorder=3,
        )
    plt.scatter(no_sv_df["ce_non_target_langs"], no_sv_df["ce_target_token"], color="red", marker="D", s=90, label="No SV", zorder=3)
    for _, row in plot_df.iterrows():
        plt.text(row["ce_non_target_langs"] + 0.01, row["ce_target_token"] + 0.01, row["method"], fontsize=9)
    plt.xlabel(f"CE loss on Flores-10 without {LANG_CODE_TO_NAME[args.source_lang]}")
    plt.ylabel(f"CE loss for target token ({target_word})")
    plt.title(f"Adversarial LID ({model_label}): {args.source_lang} -> {args.target_lang}")
    plt.grid(True, alpha=0.3)
    legend_handles = [
        Line2D([0], [0], color="green", linewidth=1.6, label="SAE"),
        Line2D([0], [0], color="blue", linewidth=1.6, label="SV"),
        Line2D([0], [0], color="red", marker="D", linestyle="None", markersize=8, label="No SV"),
        Line2D([0], [0], color="black", marker="o", linestyle="None", markersize=7, label="1L"),
        Line2D([0], [0], color="black", marker="^", linestyle="None", markersize=7, label="2L"),
        Line2D([0], [0], color="black", marker="s", linestyle="None", markersize=7, label="3L"),
    ]
    plt.legend(handles=legend_handles, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(df.sort_values(["method"]))
    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
