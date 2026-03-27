from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils import LANG_CODE_TO_NAME, ensure_dir, repo_root


MARKER_MAP = {1: "o", 2: "^", 3: "s"}


def parse_args():
    root = repo_root()
    parser = argparse.ArgumentParser(description="Plot a 2x2 layer-influence subplot from LID CSV files.")
    parser.add_argument(
        "--csv-dir",
        default=str(root / "scripts" / "layer_influence"),
        help="Directory containing the layer sweep CSV files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path. Defaults to results/plots/layer_influence_subplot.png",
    )
    return parser.parse_args()


def _extract_metadata(path: Path) -> tuple[str, str, int]:
    match = re.search(r"_([a-z]+)_to_([a-z]+)_layer(\d+)_alpha", path.name)
    if not match:
        raise ValueError(f"Could not parse source/target language and layer from filename: {path.name}")
    return match.group(1), match.group(2), int(match.group(3))


def _plot_one_panel(ax, csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    source_lang, _target_lang, layer = _extract_metadata(csv_path)

    plot_order = ["SAE-1L", "SAE-2L", "SAE-3L"]
    if any(df["method"].astype(str).str.startswith("Learned")):
        plot_order.extend(["Learned-1L", "Learned-2L", "Learned-3L"])
    plot_order.extend(["SV-1L", "SV-2L", "SV-3L", "No SV"])
    plot_order = [method for method in plot_order if method in set(df["method"].astype(str))]

    plot_df = df.set_index("method").loc[plot_order].reset_index()
    sae_df = plot_df[plot_df["method"].str.startswith("SAE")].sort_values("k")
    learned_df = plot_df[plot_df["method"].str.startswith("Learned")].sort_values("k")
    sv_df = plot_df[plot_df["method"].str.startswith("SV")].sort_values("k")
    no_sv_df = plot_df[plot_df["method"] == "No SV"]

    ax.plot(sae_df["ce_non_target_langs"], sae_df["ce_target_token"], color="green", linewidth=1.6)
    if not learned_df.empty:
        ax.plot(learned_df["ce_non_target_langs"], learned_df["ce_target_token"], color="orange", linewidth=1.6)
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
    for _, row in learned_df.iterrows():
        ax.scatter(
            row["ce_non_target_langs"],
            row["ce_target_token"],
            color="orange",
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
    ax.set_title(f"Original Language: {source_name} (Layer {layer})", fontsize=16, fontweight="bold")
    ax.set_xlabel(f"CE loss on Flores-10 without {source_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("CE Loss For Target Token", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)


def main():
    args = parse_args()
    csv_dir = Path(args.csv_dir).resolve()
    csv_paths = sorted(csv_dir.glob("*.csv"), key=lambda path: _extract_metadata(path)[2])
    if len(csv_paths) != 4:
        raise ValueError(f"Expected exactly 4 CSV files in {csv_dir}, found {len(csv_paths)}.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    has_learned = False
    for ax, csv_path in zip(axes.flat, csv_paths):
        df = pd.read_csv(csv_path)
        has_learned = has_learned or any(df["method"].astype(str).str.startswith("Learned"))
        _plot_one_panel(ax, csv_path)

    legend_handles = [
        Line2D([0], [0], color="green", linewidth=1.6, label="SAE"),
        Line2D([0], [0], color="blue", linewidth=1.6, label="SV"),
        Line2D([0], [0], color="red", marker="D", linestyle="None", markersize=8, label="No SV"),
        Line2D([0], [0], color="black", marker="o", linestyle="None", markersize=7, label="1L"),
        Line2D([0], [0], color="black", marker="^", linestyle="None", markersize=7, label="2L"),
        Line2D([0], [0], color="black", marker="s", linestyle="None", markersize=7, label="3L"),
    ]
    if has_learned:
        legend_handles.insert(1, Line2D([0], [0], color="orange", linewidth=1.6, label="Learned"))
    fig.legend(handles=legend_handles, ncol=len(legend_handles), loc="lower center", bbox_to_anchor=(0.5, 0.02))
    plt.tight_layout(rect=(0, 0.06, 1, 1))

    output = Path(args.output) if args.output else repo_root() / "results" / "plots" / "layer_influence_subplot.png"
    ensure_dir(output.parent)
    plt.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Saved figure: {output}")


if __name__ == "__main__":
    main()
