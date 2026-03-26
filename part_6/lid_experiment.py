from __future__ import annotations

import argparse
import gc
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import tqdm
from matplotlib.lines import Line2D

from part_6.steering_utils import (
    build_lid_prompt,
    build_patch_specs,
    build_learned_gate_bank,
    build_sv_bank_and_gates,
    learned_gate_bank_filename,
    load_learned_gate_bank,
    lm_ce_loss_on_texts_batched,
    measure_sae_gate_activation_rate,
    target_label_ce_from_prompt,
    target_token_ce_from_prompt,
    window_layers,
)
from utils import (
    LANG_CODE_TO_NAME,
    MODEL_PRESETS,
    TARGET_LANGS,
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
    parser.add_argument("--model-name", choices=sorted(MODEL_PRESETS), default="gemma-2-2b")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--sae-release", default=None)
    parser.add_argument("--dataset-path", default=str(root / "data" / "multilingual_data_test.jsonl"))
    parser.add_argument("--source-lang", default="fr")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--base-layer", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5, help="IMPORTANT parameter for scaling steering vectors.")
    parser.add_argument(
        "--target-metric",
        choices=["first-token", "full-label"],
        default="first-token",
        help="Metric for the y-axis: first target-token CE or autoregressive CE over the full target label.",
    )
    parser.add_argument("--gate-topk", type=int, default=2, help="Number of source-language SAE features used for gating.")
    parser.add_argument("--gate-threshold", type=float, default=0.0, help="Activation threshold for SAE gating.")
    parser.add_argument(
        "--learned-gating",
        action="store_true",
        help="Enable the learned SAE gating baseline in addition to SV and heuristic SAE gating.",
    )
    parser.add_argument("--learned-train-n", type=int, default=5, help="Number of source and per-language collateral texts used to train the learned gate.")
    parser.add_argument("--learned-epochs", type=int, default=25, help="Number of optimization epochs for the learned gate.")
    parser.add_argument("--learned-lr", type=float, default=0.1, help="Learning rate for the learned gate.")
    parser.add_argument("--learned-collateral-weight", type=float, default=0.2, help="Weight of collateral LM CE in the learned gate objective.")
    parser.add_argument(
        "--learned-gate-dir",
        default=str(root / "learned_gates"),
        help="Directory containing explicitly saved learned gate banks. If a matching file exists, load it instead of retraining.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="cuda",
        help="Execution device. 'auto' defaults to CPU on macOS to avoid MPS graph crashes.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Model dtype. 'auto' defaults to bfloat16 on CUDA and float32 otherwise.",
    )
    parser.add_argument(
        "--sae-device",
        choices=["cpu", "mps", "cuda", "same"],
        default="cuda",
        help="Where to keep SAE gates. 'cpu' is safer on small GPUs; 'same' follows --device.",
    )
    parser.add_argument("--train-n", type=int, default=20, help="Number of source/target samples per language used to build the steering bank.")
    parser.add_argument(
        "--gate-train-n",
        type=int,
        default=20,
        help="Number of texts per language used only to estimate SAE top-features for gating.",
    )
    parser.add_argument("--eval-n", type=int, default=10, help="Number of source-language evaluation samples.")
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
        "--collateral-batch-size",
        type=int,
        default=1,
        help="Micro-batch size used only for collateral LM CE evaluation on the x-axis.",
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
        "--v-scores-csv",
        default=str(root / "results" / "csv" / "v_scores_run_reprod_fig_1_top5.csv"),
        help="Optional CSV exported by part_6.export_v_scores.py. If present, reuse its top-k features instead of recomputing v/nu scores.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable loading/saving the local cache for sv_bank and top_idx_layer.",
    )
    parser.add_argument(
        "--diagnose-gates",
        action="store_true",
        help="Print average SAE gate activation rates on a small source/collateral sample before evaluation.",
    )
    parser.add_argument(
        "--diagnose-n",
        type=int,
        default=5,
        help="Number of source and collateral texts used for SAE gate diagnostics.",
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
    v_scores_csv = Path(args.v_scores_csv) if args.v_scores_csv else None
    if v_scores_csv is not None and not v_scores_csv.exists():
        v_scores_csv = None
    try:
        commit_short_sha = (
            subprocess.check_output(
                ["git", "-C", str(repo_root()), "rev-parse", "--short", "HEAD"],
                text=True,
            ).strip()
        )
    except Exception:
        commit_short_sha = "nogit"

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
    by_lang = {}
    for code in TARGET_LANGS:
        texts = lang_texts[code]
        train_count = args.train_n if code in {args.source_lang, args.target_lang} else 0
        required = train_count + split_eval_n
        if len(texts) < required:
            raise ValueError(f"Language {code} has {len(texts)} texts, expected at least {required}.")
        by_lang[code] = {
            "train": texts[:train_count],
            "eval": texts[train_count : train_count + split_eval_n],
        }
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
    if args.learned_gating:
        methods.extend(
            [
                ("Learned-1L", 1),
                ("Learned-2L", 2),
                ("Learned-3L", 3),
            ]
        )

    total_bank_steps = len(window_layers(args.base_layer, 3, model)) * (
        2 * args.train_n + len(TARGET_LANGS) * args.gate_train_n
    )
    if args.learned_gating:
        total_bank_steps += len(window_layers(args.base_layer, 3, model)) * (len(TARGET_LANGS) * args.learned_train_n)
    total_eval_steps = len(methods) * (args.eval_n + (len(TARGET_LANGS) - 1) * other_eval_n)
    overall_pbar = tqdm.tqdm(total=total_bank_steps + total_eval_steps, desc="LID pipeline", unit="text")

    def step_progress(phase: str):
        overall_pbar.set_postfix_str(phase)
        overall_pbar.update(1)

    target_word = LANG_CODE_TO_NAME[args.target_lang]

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
        gate_train_n=args.gate_train_n,
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
            "gate_train_n": args.gate_train_n,
            "sae_release": sae_release,
            "target_lan": TARGET_LANGS,
        },
        gate_topk=args.gate_topk,
        v_scores_csv=v_scores_csv,
    )
    learned_gate_bank = None
    if args.learned_gating:
        learned_gate_path = Path(args.learned_gate_dir) / learned_gate_bank_filename(
            model_file_tag,
            args.source_lang,
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
        if learned_gate_path.exists():
            learned_gate_bank = load_learned_gate_bank(learned_gate_path)
            print(f"Loaded learned gate bank: {learned_gate_path}")
        else:
            learned_gate_bank = build_learned_gate_bank(
                window,
                model,
                tokenizer,
                TARGET_LANGS,
                multilingual_texts,
                args.source_lang,
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
                    "source_lang": args.source_lang,
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
                progress_callback=lambda: step_progress("building learned gates"),
                target_word=target_word,
                learned_train_n=args.learned_train_n,
                learned_epochs=args.learned_epochs,
                learned_lr=args.learned_lr,
                learned_collateral_weight=args.learned_collateral_weight,
            )

    # Adversarial LID: steer source-language texts toward the target-language label,
    # and also measure collateral CE on all languages except the original/source one.
    eval_source = by_lang[args.source_lang]["eval"][: args.eval_n]
    other_non_target = []

    for code in TARGET_LANGS:
        # Paper-style split: use non-overlapping evaluation texts after the first train_n examples.
        # Figure 8 caption measures the impact on non-original texts, so exclude only source A.
        if code == args.source_lang:
            continue
        other_non_target.extend(by_lang[code]["eval"][:other_eval_n])

    print(
        f"Running Adversarial LID with dataset={args.dataset_path}, "
        f"train_n={args.train_n}, source_eval_n={len(eval_source)}, "
        f"gate_train_n={args.gate_train_n}, other_eval_n={other_eval_n}, base_layer={args.base_layer}, alpha={args.alpha}, "
        f"model={model_label}, "
        f"target_metric={args.target_metric}, "
        f"device={device}, dtype={args.dtype}, sae_device={sae_device}, "
        f"gate_topk={args.gate_topk}, gate_threshold={args.gate_threshold}, "
        f"learned_gating={args.learned_gating}, cache_dir={cache_dir}, v_scores_csv={v_scores_csv}"
    )
    print(
        f"Source language: {args.source_lang} -> target language: {args.target_lang} | "
        f"collateral samples={len(other_non_target)} | "
        f"total prompt evaluations={len(methods) * (len(eval_source) + len(other_non_target))}"
    )
    if args.diagnose_gates:
        diag_source = eval_source[: args.diagnose_n]
        diag_other = other_non_target[: args.diagnose_n]
        print("SAE gate diagnostics:")
        for layer_idx in window:
            feature_indices = gate_bank.get(layer_idx)
            if feature_indices is None:
                continue
            source_rate = measure_sae_gate_activation_rate(
                model,
                tokenizer,
                diag_source,
                layer_idx,
                sae_release,
                feature_indices,
                device,
                sae_device=sae_device,
                sae_dtype=model_dtype if sae_device != "cpu" else torch.float32,
                threshold=args.gate_threshold,
            )
            other_rate = measure_sae_gate_activation_rate(
                model,
                tokenizer,
                diag_other,
                layer_idx,
                sae_release,
                feature_indices,
                device,
                sae_device=sae_device,
                sae_dtype=model_dtype if sae_device != "cpu" else torch.float32,
                threshold=args.gate_threshold,
            )
            print(
                f"  layer {layer_idx}: features={feature_indices} | "
                f"source_rate={source_rate:.4f} | other_rate={other_rate:.4f}"
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
    target_ce_fn = target_token_ce_from_prompt if args.target_metric == "first-token" else target_label_ce_from_prompt
    
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
            learned_gate_bank=learned_gate_bank,
            alpha=args.alpha,
            sae_release=sae_release,
            sae_device=sae_device,
            sae_dtype=model_dtype if sae_device != "cpu" else torch.float32,
            gate_threshold=args.gate_threshold,
        )
        ce_target = []
        for text in eval_source:
            ce_target.append(
                target_ce_fn(model, tokenizer, build_lid_prompt(text), target_word, device, patch_specs)
            )
            step_progress(f"eval {method_name} source")
        ce_other = lm_ce_loss_on_texts_batched(
            model,
            tokenizer,
            other_non_target,
            device,
            patch_specs,
            batch_size=args.collateral_batch_size,
        )
        for _ in ce_other:
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
        f"layer{args.base_layer}_alpha{args.alpha:g}_topk{args.gate_topk}_train{args.train_n}_eval{args.eval_n}_other{other_eval_n}_{args.target_metric}"
    )
    csv_path = csv_dir / f"lid_{model_file_tag}_{args.source_lang}_to_{args.target_lang}_{run_tag}_{commit_short_sha}.csv"
    fig_path = plots_dir / f"lid_{model_file_tag}_{args.source_lang}_to_{args.target_lang}_{run_tag}_{commit_short_sha}.png"
    df.to_csv(csv_path, index=False)

    # Match the notebook/paper convention: SAE in green, SV in blue, optional Learned in orange, No SV in red.
    plot_order = ["SAE-1L", "SAE-2L", "SAE-3L"]
    if args.learned_gating:
        plot_order.extend(["Learned-1L", "Learned-2L", "Learned-3L"])
    plot_order.extend(["SV-1L", "SV-2L", "SV-3L", "No SV"])
    plot_df = df.set_index("method").loc[plot_order].reset_index()
    sae_df = plot_df[plot_df["method"].str.startswith("SAE")].sort_values("k")
    learned_df = plot_df[plot_df["method"].str.startswith("Learned")].sort_values("k")
    sv_df = plot_df[plot_df["method"].str.startswith("SV")].sort_values("k")
    no_sv_df = plot_df[plot_df["method"] == "No SV"]

    marker_map = {1: "o", 2: "^", 3: "s"}

    plt.figure(figsize=(8, 6))
    plt.plot(sae_df["ce_non_target_langs"], sae_df["ce_target_token"], color="green", linewidth=1.6)
    if not learned_df.empty:
        plt.plot(learned_df["ce_non_target_langs"], learned_df["ce_target_token"], color="orange", linewidth=1.6)
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
    for _, row in learned_df.iterrows():
        plt.scatter(
            row["ce_non_target_langs"],
            row["ce_target_token"],
            color="orange",
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
    plt.xlabel(f"CE loss on Flores-10 without {LANG_CODE_TO_NAME[args.source_lang]}")
    plt.ylabel("CE Loss For Target Token")
    plt.title(f"Original Language: {LANG_CODE_TO_NAME[args.source_lang]}")
    plt.grid(True, alpha=0.3)
    legend_handles = [
        Line2D([0], [0], color="green", linewidth=1.6, label="SAE"),
        Line2D([0], [0], color="blue", linewidth=1.6, label="SV"),
        Line2D([0], [0], color="red", marker="D", linestyle="None", markersize=8, label="No SV"),
        Line2D([0], [0], color="black", marker="o", linestyle="None", markersize=7, label="1L"),
        Line2D([0], [0], color="black", marker="^", linestyle="None", markersize=7, label="2L"),
        Line2D([0], [0], color="black", marker="s", linestyle="None", markersize=7, label="3L"),
    ]
    if args.learned_gating:
        legend_handles.insert(1, Line2D([0], [0], color="orange", linewidth=1.6, label="Learned"))
    plt.legend(handles=legend_handles, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(df.sort_values(["method"]))
    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
