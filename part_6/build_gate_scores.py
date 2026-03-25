from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import tqdm

from part_6.steering_utils import try_load_sae_for_layer
from utils import (
    LANG_CODE_TO_NAME,
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
    compute_top_index_per_lan_for_layer,
)


SNLP_TO_FLORES = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "pt": "por_Latn",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "zh": "cmn_Hans",
    "ar": "arb_Arab",
}


def parse_args():
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="Compute SAE top language features for selected layers and export them in a v-scores-compatible CSV."
    )
    parser.add_argument("--model-name", choices=sorted(MODEL_PRESETS), default="gemma-2-2b")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--sae-release", default=None)
    parser.add_argument("--dataset-path", default=str(root / "data" / "multilingual_data_test.jsonl"))
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[20, 21, 22],
        help="List of layers for which top language features are computed.",
    )
    parser.add_argument(
        "--n-texts-per-lang",
        type=int,
        default=20,
        help="Number of texts per language used to estimate top language features.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of top features to export per language and layer.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--sae-device", choices=["cpu", "mps", "cuda", "same"], default="cuda")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path. Defaults to data/v_scores_<model>_layers20-21-22_top5.csv.",
    )
    return parser.parse_args()


def _default_output_path(root: Path, model_file_tag: str, layers: list[int], top_k: int) -> Path:
    layers_tag = "-".join(str(layer) for layer in layers)
    return root / "data" / f"v_scores_{model_file_tag}_layers{layers_tag}_top{top_k}.csv"


def main():
    args = parse_args()
    root = repo_root()
    _model_label, model_path, sae_release = resolve_model_artifacts(root, args.model_name, args.model_path, args.sae_release)
    model_file_tag = model_tag(args.model_name, model_path)
    device = get_safe_default_device() if args.device == "auto" else args.device
    sae_device = device if args.sae_device == "same" else args.sae_device
    sae_dtype = torch.float32 if sae_device == "cpu" else None
    layers = sorted(dict.fromkeys(args.layers))

    data = load_multilingual_dataframe(args.dataset_path)
    lang_texts = build_language_texts(data, TARGET_LANGS)
    multilingual_texts = [text for code in TARGET_LANGS for text in lang_texts[code]]

    model, tokenizer = load_model_and_tokenizer(model_path, device=device, dtype=args.dtype)
    total_steps = len(layers) * len(TARGET_LANGS) * args.n_texts_per_lang
    pbar = tqdm.tqdm(total=total_steps, desc="Gate scores", unit="text")

    rows = []
    for layer_idx in layers:
        sae = try_load_sae_for_layer(sae_release, layer_idx, sae_device, dtype=sae_dtype)
        top_idx, top_val = compute_top_index_per_lan_for_layer(
            model,
            tokenizer,
            sae,
            layer_idx,
            TARGET_LANGS,
            multilingual_texts,
            device,
            n_texts_per_lan=args.n_texts_per_lang,
            progress_callback=pbar.update,
        )
        for lang_i, language_code in enumerate(TARGET_LANGS):
            language_flores = SNLP_TO_FLORES.get(language_code, language_code)
            for rank in range(min(args.top_k, top_idx.shape[1])):
                rows.append(
                    {
                        "layer": int(layer_idx),
                        "language_flores": language_flores,
                        "language_code": language_code,
                        "language_name": LANG_CODE_TO_NAME.get(language_code, language_code),
                        "rank": rank + 1,
                        "feature_index": int(top_idx[lang_i, rank]),
                        "nu_value": float(top_val[lang_i, rank]),
                    }
                )
        del top_idx
        del top_val
        del sae
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    pbar.close()

    output = Path(args.output) if args.output else _default_output_path(root, model_file_tag, layers, args.top_k)
    ensure_dir(output.parent)
    df = pd.DataFrame(rows).sort_values(["layer", "language_code", "rank"]).reset_index(drop=True)
    df.to_csv(output, index=False)
    print(df.head(20))
    print(f"Saved gate scores CSV: {output}")


if __name__ == "__main__":
    main()
