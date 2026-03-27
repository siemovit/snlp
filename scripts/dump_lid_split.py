from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import TARGET_LANGS, build_lang_split, build_language_texts, ensure_dir, load_multilingual_dataframe, repo_root


def parse_args():
    root = repo_root()
    parser = argparse.ArgumentParser(description="Dump the effective LID train/eval split used by part_6.lid_experiment.")
    parser.add_argument("--dataset-path", default=str(root / "data" / "multilingual_data_test.jsonl"))
    parser.add_argument("--source-lang", default="fr")
    parser.add_argument("--target-lang", default="en")
    parser.add_argument("--train-n", type=int, default=20)
    parser.add_argument("--eval-n", type=int, default=10)
    parser.add_argument("--other-eval-n", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to data/lid_split_<source>_to_<target>_train{n}_eval{n}_other{m}",
    )
    return parser.parse_args()


def _rows_from_texts(texts: list[str], language: str, split: str) -> list[dict]:
    return [{"lan": language, "split": split, "text": text} for text in texts]


def main():
    args = parse_args()
    other_eval_n = args.other_eval_n if args.other_eval_n is not None else args.eval_n
    split_eval_n = max(args.eval_n, other_eval_n)

    data = load_multilingual_dataframe(args.dataset_path)
    lang_texts = build_language_texts(data, TARGET_LANGS)
    by_lang = build_lang_split(lang_texts, train_n=args.train_n, eval_n=split_eval_n, target_langs=TARGET_LANGS)

    tag = f"lid_split_{args.source_lang}_to_{args.target_lang}_train{args.train_n}_eval{args.eval_n}_other{other_eval_n}"
    output_dir = Path(args.output_dir) if args.output_dir else repo_root() / "data" / tag
    output_dir = ensure_dir(output_dir)

    train_rows: list[dict] = []
    for code in TARGET_LANGS:
        train_rows.extend(_rows_from_texts(by_lang[code]["train"], code, "train"))

    source_eval_rows = _rows_from_texts(by_lang[args.source_lang]["eval"][: args.eval_n], args.source_lang, "source_eval")

    collateral_rows: list[dict] = []
    for code in TARGET_LANGS:
        if code == args.source_lang:
            continue
        collateral_rows.extend(_rows_from_texts(by_lang[code]["eval"][:other_eval_n], code, "collateral_eval"))

    train_path = output_dir / "train.jsonl"
    source_eval_path = output_dir / "source_eval.jsonl"
    collateral_path = output_dir / "collateral_eval.jsonl"
    summary_path = output_dir / "summary.json"

    pd.DataFrame(train_rows).to_json(train_path, orient="records", lines=True, force_ascii=False)
    pd.DataFrame(source_eval_rows).to_json(source_eval_path, orient="records", lines=True, force_ascii=False)
    pd.DataFrame(collateral_rows).to_json(collateral_path, orient="records", lines=True, force_ascii=False)

    summary = {
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "source_lang": args.source_lang,
        "target_lang": args.target_lang,
        "train_n": args.train_n,
        "eval_n": args.eval_n,
        "other_eval_n": other_eval_n,
        "target_langs": TARGET_LANGS,
        "output_dir": str(output_dir.resolve()),
        "counts": {
            "train_total": len(train_rows),
            "source_eval_total": len(source_eval_rows),
            "collateral_eval_total": len(collateral_rows),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Saved train split: {train_path}")
    print(f"Saved source eval split: {source_eval_path}")
    print(f"Saved collateral eval split: {collateral_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
