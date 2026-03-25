from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from utils import ensure_dir, repo_root


FLORES_PLUS_TO_LANGUAGE = {
    "arg_Latn": {"code": "an", "name": "Aragonese"},
    "azb_Arab": {"code": "azb", "name": "South Azerbaijani (Arabic script)"},
    "cmn_Hans": {"code": "zh", "name": "Chinese"},
    "eng_Latn": {"code": "en", "name": "English"},
    "fra_Latn": {"code": "fr", "name": "French"},
    "jpn_Jpan": {"code": "ja", "name": "Japanese"},
    "kas_Arab": {"code": "ks", "name": "Kashmiri (Arabic script)"},
    "kor_Hang": {"code": "ko", "name": "Korean"},
    "nus_Latn": {"code": "nus", "name": "Nuer"},
    "por_Latn": {"code": "pt", "name": "Portuguese"},
    "spa_Latn": {"code": "es", "name": "Spanish"},
    "tha_Thai": {"code": "th", "name": "Thai"},
    "vie_Latn": {"code": "vi", "name": "Vietnamese"},
    "wuu_Hans": {"code": "wuu", "name": "Wu Chinese"},
}


def parse_args():
    root = repo_root()
    parser = argparse.ArgumentParser(description="Export per-layer top-k v/nu scores from an MVA-SNLP run.")
    parser.add_argument(
        "--run-dir",
        default=str(root.parent / "MVA-SNLP" / "v_score_runs" / "run_reprod_fig_1"),
        help="Path to the MVA-SNLP v_score_runs/<run_name> directory.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="How many top features per language to export.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path. Defaults to results/csv/v_scores_<run_name>_top{k}.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    mva_root = run_dir.parent.parent
    sys.path.insert(0, str(mva_root))
    from score import load_v_score_run  # type: ignore

    meta, per_layer = load_v_score_run(run_dir)
    rows = []
    for layer, payload in per_layer.items():
        top_idx = payload["top_index_per_lan"]
        top_val = payload["top_values_per_lan"]
        for lang_i, language in enumerate(meta.languages):
            language_info = FLORES_PLUS_TO_LANGUAGE.get(language, {"code": language, "name": language})
            for rank in range(min(args.top_k, top_idx.shape[1])):
                rows.append(
                    {
                        "layer": int(layer),
                        "language_flores": language,
                        "language_code": language_info["code"],
                        "language_name": language_info["name"],
                        "rank": rank + 1,
                        "feature_index": int(top_idx[lang_i, rank]),
                        "nu_value": float(top_val[lang_i, rank]),
                    }
                )

    df = pd.DataFrame(rows).sort_values(["layer", "language_code", "rank"]).reset_index(drop=True)
    output = Path(args.output) if args.output else ensure_dir(repo_root() / "results" / "csv") / f"v_scores_{run_dir.name}_top{args.top_k}.csv"
    df.to_csv(output, index=False)
    print(df.head(20))
    print(f"Saved CSV: {output}")


if __name__ == "__main__":
    main()
