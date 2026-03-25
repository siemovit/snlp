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
    lm_ce_loss_on_texts_batched,
    load_gate_bank_from_v_scores,
    target_label_ce_from_prompt,
    target_token_ce_from_prompt,
    window_layers,
)
from utils import (
    LANG_CODE_TO_NAME,
    MODEL_PRESETS,
    TARGET_LANGS,
    build_lang_split,
    build_language_texts,
    ensure_dir,
    get_device,
    get_safe_default_device,
    load_model_and_tokenizer,
    load_multilingual_dataframe,
    model_tag,
    repo_root,
    resolve_model_artifacts,
)


def parse_args():
    root = repo_root()
    parser = argparse.ArgumentParser(description="Alternative LID experiment using precomputed steering vectors.")
    parser.add_argument("--model-name", choices=sorted(MODEL_PRESETS), default="gemma-2-2b")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--sae-release", default=None)
    parser.add_argument("--steering-dataset-path", default=str(root / "data" / "multilingual_data.jsonl"))
    parser.add_argument("--eval-dataset-path", default=str(root / "data" / "multilingual_data_test.jsonl"))
    parser.add_argument("--v-scores-csv", default=str(root / "data" / "v_scores_run_reprod_fig_1_top5.csv"))
    parser.add_argument("--steering-dir", default=str(root / "steering_vectors"))
    parser.add_argument("--source-lang", default="fr")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--base-layer", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--train-n", type=int, default=100, help="Number of texts per source/target language used to precompute the steering bank.")
    parser.add_argument("--eval-n", type=int, default=10)
    parser.add_argument("--other-eval-n", type=int, default=None)
    parser.add_argument("--gate-topk", type=int, default=2)
    parser.add_argument("--gate-threshold", type=float, default=0.0)
    parser.add_argument("--target-metric", choices=["first-token", "full-label"], default="first-token")
    parser.add_argument("--collateral-batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--sae-device", choices=["cpu", "mps", "cuda", "same"], default="cuda")
    return parser.parse_args()


def _load_sv_bank(path: Path, device: str, layers_to_use: list[int]) -> dict[int, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    stored_bank = payload["sv_bank"] if isinstance(payload, dict) and "sv_bank" in payload else payload
    bank = {}
    for layer_idx in layers_to_use:
        if layer_idx not in stored_bank:
            raise ValueError(f"Missing layer {layer_idx} in steering bank: {path}")
        bank[layer_idx] = stored_bank[layer_idx].to(device)
    return bank


def main():
    args = parse_args()
    model_label, model_path, sae_release = resolve_model_artifacts(repo_root(), args.model_name, args.model_path, args.sae_release)
    model_file_tag = model_tag(args.model_name, model_path)
    device = get_safe_default_device() if args.device == "auto" else args.device
    available_device = get_device()
    if device == "cuda" and available_device != "cuda":
        raise ValueError("CUDA was requested but is not available on this machine.")
    sae_device = device if args.sae_device == "same" else args.sae_device

    other_eval_n = args.other_eval_n if args.other_eval_n is not None else args.eval_n
    results_dir = ensure_dir(repo_root() / "results")
    csv_dir = ensure_dir(results_dir / "csv")
    plots_dir = ensure_dir(results_dir / "plots")

    model, tokenizer = load_model_and_tokenizer(model_path, device=device, dtype=args.dtype)
    eval_data = load_multilingual_dataframe(args.eval_dataset_path)
    eval_lang_texts = build_language_texts(eval_data, TARGET_LANGS)
    by_lang = build_lang_split(eval_lang_texts, train_n=0, eval_n=max(args.eval_n, other_eval_n), target_langs=TARGET_LANGS)

    window = window_layers(args.base_layer, 3, model)
    steering_path = Path(args.steering_dir) / f"sv_{model_file_tag}_{args.source_lang}_to_{args.target_lang}_train{args.train_n}.pt"
    if not steering_path.exists():
        raise FileNotFoundError(
            f"Missing steering bank: {steering_path}. Run `python -m part_6.build_steering_vectors` first."
        )
    sv_bank = _load_sv_bank(steering_path, device, window)
    gate_bank = load_gate_bank_from_v_scores(window, args.source_lang, args.gate_topk, args.v_scores_csv)

    methods = [
        ("No SV", 0),
        ("SV-1L", 1),
        ("SV-2L", 2),
        ("SV-3L", 3),
        ("SAE-1L", 1),
        ("SAE-2L", 2),
        ("SAE-3L", 3),
    ]

    eval_source = by_lang[args.source_lang]["eval"][: args.eval_n]
    other_non_target = []
    for code in TARGET_LANGS:
        if code == args.source_lang:
            continue
        other_non_target.extend(by_lang[code]["eval"][:other_eval_n])

    total_eval_steps = len(methods) * (len(eval_source) + len(other_non_target))
    pbar = tqdm.tqdm(total=total_eval_steps, desc="LID2 pipeline", unit="text")

    target_word = LANG_CODE_TO_NAME[args.target_lang]
    target_ce_fn = target_token_ce_from_prompt if args.target_metric == "first-token" else target_label_ce_from_prompt
    rows = []
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
            sae_device=sae_device,
            sae_dtype=next(model.parameters()).dtype if sae_device != "cpu" else torch.float32,
            gate_threshold=args.gate_threshold,
        )
        ce_target = []
        for text in eval_source:
            ce_target.append(target_ce_fn(model, tokenizer, build_lid_prompt(text), target_word, device, patch_specs))
            pbar.set_postfix_str(f"eval {method_name} source")
            pbar.update(1)

        ce_other = lm_ce_loss_on_texts_batched(
            model,
            tokenizer,
            other_non_target,
            device,
            patch_specs,
            batch_size=args.collateral_batch_size,
        )
        pbar.set_postfix_str(f"eval {method_name} collateral")
        pbar.update(len(ce_other))

        rows.append(
            {
                "method": method_name,
                "k": k,
                "ce_target_token": float(pd.Series(ce_target).mean()),
                "ce_non_target_langs": float(pd.Series(ce_other).mean()),
            }
        )
        del patch_specs
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    pbar.close()

    try:
        commit_short_sha = (
            subprocess.check_output(["git", "-C", str(repo_root()), "rev-parse", "--short", "HEAD"], text=True).strip()
        )
    except Exception:
        commit_short_sha = "nogit"

    df = pd.DataFrame(rows)
    run_tag = (
        f"layer{args.base_layer}_alpha{args.alpha:g}_topk{args.gate_topk}_train{args.train_n}_eval{args.eval_n}_other{other_eval_n}_{args.target_metric}"
    )
    csv_path = csv_dir / f"lid2_{model_file_tag}_{args.source_lang}_to_{args.target_lang}_{run_tag}_{commit_short_sha}.csv"
    fig_path = plots_dir / f"lid2_{model_file_tag}_{args.source_lang}_to_{args.target_lang}_{run_tag}_{commit_short_sha}.png"
    df.to_csv(csv_path, index=False)

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
        plt.scatter(row["ce_non_target_langs"], row["ce_target_token"], color="green", marker=marker_map[int(row["k"])], s=70, zorder=3)
    for _, row in sv_df.iterrows():
        plt.scatter(row["ce_non_target_langs"], row["ce_target_token"], color="blue", marker=marker_map[int(row["k"])], s=70, zorder=3)
    plt.scatter(no_sv_df["ce_non_target_langs"], no_sv_df["ce_target_token"], color="red", marker="D", s=90, zorder=3)
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
    plt.legend(handles=legend_handles, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"Loaded steering bank: {steering_path}")
    print(f"Using v-scores CSV: {args.v_scores_csv}")
    print(df.sort_values(["method"]))
    print(f"Saved CSV: {csv_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
