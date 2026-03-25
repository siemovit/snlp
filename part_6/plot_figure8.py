from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from utils import LANG_CODE_TO_NAME, ensure_dir, repo_root


MARKER_MAP = {1: "o", 2: "^", 3: "s"}


def parse_args():
    root = repo_root()
    parser = argparse.ArgumentParser(description="Plot a 2x2 Figure 8-style subplot from LID CSV files.")
    parser.add_argument(
        "--csv-dir",
        default=str(root / "results" / "csv" / "figure8"),
        help="Directory containing 4 LID CSV files.",
    )
    parser.add_argument(
        "--output",
        default=str(root / "results" / "plots" / "figure8_subplot.png"),
        help="Output PNG path.",
    )
    return parser.parse_args()


def _extract_langs_from_name(path: Path) -> tuple[str, str]:
    match = re.search(r"_([a-z]+)_to_([a-z]+)_alpha", path.name)
    if not match:
        raise ValueError(f"Could not parse source/target language from filename: {path.name}")
    return match.group(1), match.group(2)


def _plot_one_panel(ax, csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    source_lang, _target_lang = _extract_langs_from_name(csv_path)

    plot_order = ["SAE-1L", "SAE-2L", "SAE-3L", "SV-1L", "SV-2L", "SV-3L", "No SV"]
    plot_df = df.set_index("method").loc[plot_order].reset_index()
    sae_df = plot_df[plot_df["method"].str.startswith("SAE")].sort_values("k")
    sv_df = plot_df[plot_df["method"].str.startswith("SV")].sort_values("k")
    no_sv_df = plot_df[plot_df["method"] == "No SV"]

    ax.plot(sae_df["ce_non_target_langs"], sae_df["ce_target_token"], color="green", linewidth=1.6)
    ax.plot(sv_df["ce_non_target_langs"], sv_df["ce_target_token"], color="blue", linewidth=1.6)

    for _, row in sae_df.iterrows():
        ax.scatter(
            row["ce_non_target_langs"],
            row["ce_target_token"],
            color="green",
            marker=MARKER_MAP.get(int(row["k"]), "o"),
            s=70,
            zorder=3,
        )
    for _, row in sv_df.iterrows():
        ax.scatter(
            row["ce_non_target_langs"],
            row["ce_target_token"],
            color="blue",
            marker=MARKER_MAP.get(int(row["k"]), "o"),
            s=70,
            zorder=3,
        )
    ax.scatter(
        no_sv_df["ce_non_target_langs"],
        no_sv_df["ce_target_token"],
        color="red",
        marker="D",
        s=90,
        zorder=3,
    )

    source_name = LANG_CODE_TO_NAME.get(source_lang, source_lang)
    ax.set_title(f"Original Language: {source_name}", fontsize=18, fontweight="bold")
    ax.set_xlabel(f"CE loss on Flores-10 without {source_name}", fontsize=15, fontweight="bold")
    ax.set_ylabel("CE Loss For Target Token", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3)


def main():
    args = parse_args()
    csv_dir = Path(args.csv_dir).resolve()
    desired_order = ["fr", "ja", "es", "th"]
    csv_by_source = {}
    for csv_path in csv_dir.glob("*.csv"):
        source_lang, _ = _extract_langs_from_name(csv_path)
        csv_by_source[source_lang] = csv_path
    missing = [code for code in desired_order if code not in csv_by_source]
    if missing:
        raise ValueError(f"Missing CSV files for source languages: {missing}")
    csv_paths = [csv_by_source[code] for code in desired_order]
    if len(csv_paths) != 4:
        raise ValueError(f"Expected exactly 4 CSV files in {csv_dir}, found {len(csv_paths)}.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, csv_path in zip(axes.flat, csv_paths):
        _plot_one_panel(ax, csv_path)

    legend_handles = [
        Line2D([0], [0], color="green", linewidth=1.6, label="SAE"),
        Line2D([0], [0], color="blue", linewidth=1.6, label="SV"),
        Line2D([0], [0], color="red", marker="D", linestyle="None", markersize=8, label="No SV"),
        Line2D([0], [0], color="black", marker="o", linestyle="None", markersize=7, label="1L"),
        Line2D([0], [0], color="black", marker="^", linestyle="None", markersize=7, label="2L"),
        Line2D([0], [0], color="black", marker="s", linestyle="None", markersize=7, label="3L"),
    ]
    fig.legend(handles=legend_handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=(0, 0.07, 1, 1))

    output = Path(args.output)
    ensure_dir(output.parent)
    plt.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved figure: {output}")


if __name__ == "__main__":
    main()
