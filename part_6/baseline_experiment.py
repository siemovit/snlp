import argparse
import json

from part_6.steering_utils import (
    build_patch_specs,
    build_sv_bank_and_gates,
    build_lid_prompt,
    generate_continuation,
    target_token_ce_from_prompt,
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


def parse_args():
    root = repo_root()
    parser = argparse.ArgumentParser(description="Run the baseline one-layer steering demo.")
    parser.add_argument("--model-name", choices=sorted(MODEL_PRESETS), default="qwen")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dataset-path", default=str(root / "data" / "multilingual_data.jsonl"))
    parser.add_argument("--sae-release", default=None)
    parser.add_argument("--source-lang", default="fr")
    parser.add_argument("--target-lang", default="es")
    parser.add_argument("--layer", type=int, default=18)
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--gate-topk", type=int, default=2)
    parser.add_argument("--gate-threshold", type=float, default=0.0)
    parser.add_argument("--train-n", type=int, default=4)
    parser.add_argument("--eval-n", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    model_label, model_path, sae_release = resolve_model_artifacts(
        repo_root(), args.model_name, args.model_path, args.sae_release
    )
    model_file_tag = model_tag(args.model_name, model_path)
    device = get_device()
    results_dir = ensure_dir(repo_root() / "results")

    # Load the local model and the small multilingual dataset used for the toy demo.
    model, tokenizer = load_model_and_tokenizer(model_path, device=device)
    data = load_multilingual_dataframe(args.dataset_path)
    lang_texts = build_language_texts(data, TARGET_LANGS)
    by_lang = build_lang_split(lang_texts, train_n=args.train_n, eval_n=args.eval_n, target_langs=TARGET_LANGS)
    multilingual_texts = flatten_language_texts(lang_texts, TARGET_LANGS)

    # Build tiny source/target splits so the baseline stays fast to run (see default train_n and eval_n values). 
    pos_texts = by_lang[args.target_lang]["train"]
    neg_texts = by_lang[args.source_lang]["train"]
    seed_text = by_lang[args.source_lang]["eval"][0]
    window = [args.layer]

    # Define continuation prompt. 
    prompt = (
        f"Continue the following text in {LANG_CODE_TO_NAME[args.target_lang]}.\n"
        f"Text: {seed_text}\n"
        "Continuation:"
    )

    # Build a one-layer (only) steering bank and the corresponding SAE gate for that same layer.
    sv_bank, gate_bank = build_sv_bank_and_gates(
        window,
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

    # Compare the raw steering-vector patch against the SAE-gated version.
    # "specs" is for specifying which patches to apply during generation.
    sv_only_specs = build_patch_specs("SV", 1, args.layer, model, sv_bank, gate_bank, alpha=args.alpha)
    sae_gated_specs = build_patch_specs(
        "SAE",
        1,
        args.layer,
        model,
        sv_bank,
        gate_bank,
        alpha=args.alpha,
        sae_release=sae_release,
        sae_device=device,
        gate_threshold=args.gate_threshold,
    )

    # Generate three continuations: no steering, SV-only steering, and SAE-gated steering.
    # no steering
    baseline = generate_continuation(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        device=device,
        patch_specs=[],
    )
    # SV-only steering
    steered = generate_continuation(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        device=device,
        patch_specs=sv_only_specs,
    )
    # SAE-gated steering
    sae_gated = generate_continuation(
        model,
        tokenizer,
        prompt,
        max_new_tokens=args.max_new_tokens,
        device=device,
        patch_specs=sae_gated_specs,
    )

    # Measure how much each method changes the probability of the target language label.
    toy_lid_prompt = build_lid_prompt(seed_text)
    target_word = LANG_CODE_TO_NAME[args.target_lang]
    ce_base = target_token_ce_from_prompt(
        model,
        tokenizer,
        toy_lid_prompt,
        target_word,
        device,
        patch_specs=[],
    )
    ce_steered = target_token_ce_from_prompt(
        model,
        tokenizer,
        toy_lid_prompt,
        target_word,
        device,
        patch_specs=sv_only_specs,
    )
    ce_sae_gated = target_token_ce_from_prompt(
        model,
        tokenizer,
        toy_lid_prompt,
        target_word,
        device,
        patch_specs=sae_gated_specs,
    )

    # Save a compact JSON artifact for later comparison across runs.
    result = {
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "model_name": model_label,
        "model_path": model_path,
        "layer": args.layer,
        "alpha": args.alpha,
        "gate_topk": args.gate_topk,
        "gate_threshold": args.gate_threshold,
        "sae_release": sae_release,
        "train_n": args.train_n,
        "eval_n": args.eval_n,
        "prompt": prompt,
        "baseline_continuation": baseline,
        "sv_only_continuation": steered,
        "sae_gated_continuation": sae_gated,
        "ce_target_token_baseline": ce_base,
        "ce_target_token_sv_only": ce_steered,
        "ce_target_token_sae_gated": ce_sae_gated,
    }

    run_tag = f"alpha{args.alpha:g}_train{args.train_n}_eval{args.eval_n}"
    out_path = results_dir / f"baseline_{model_file_tag}_{args.source_lang}_to_{args.target_lang}_{run_tag}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"Toy direction ({model_label}): {args.source_lang} -> {args.target_lang} at layer {args.layer} (alpha={args.alpha})")
    print("\nPrompt:\n", prompt)
    print("\nBaseline continuation:\n", baseline)
    print("\nSV-only continuation:\n", steered)
    print("\nSAE-gated continuation:\n", sae_gated)
    print(
        f"\nTarget-token CE ({target_word}) | baseline: {ce_base:.4f} | "
        f"SV-only: {ce_steered:.4f} | SAE-gated: {ce_sae_gated:.4f}"
    )
    print(f"\nSaved results: {out_path}")


if __name__ == "__main__":
    main()
