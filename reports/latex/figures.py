"""Figures for the PDF report: one colour scheme and one style for every chart (vector PDF)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# the colour scheme of the whole report (the LaTeX template defines the same colours)
NAVY = "#1B2A41"        # headings, the key model (LLM), dark cells
CONSPIRACY = "#8C3B76"  # conspiracy websites and articles
MAINSTREAM = "#2B7A6A"  # mainstream websites and articles
GOLD = "#C08A2B"        # Website 4 and other highlights
SLATE = "#5B6470"       # secondary text and the other models
GRID = "#E3E6EA"        # grid lines and light fills
INK = "#1F2328"         # main text

plt.rcParams.update({
    "font.family": "Inter", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": SLATE,
    "xtick.color": SLATE, "ytick.color": INK, "axes.labelcolor": SLATE, "pdf.fonttype": 42,
})


def style(ax, grid="x"):
    """Light grid behind the data, no box."""
    ax.grid(axis=grid, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(length=0)


def apa(value: float) -> str:
    """APA style for values that cannot exceed 1: three decimals, no leading zero (.721, 1.000)."""
    text = f"{value:.3f}"
    return text[1:] if text.startswith("0.") else text


def apa_axis(ax):
    """Tick labels of an F1 axis in the same style as the text (.90 instead of 0.90)."""
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".removeprefix("0")))


def save(fig, path: Path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def confusion_panel(ax, counts: dict, title: str):
    """2 x 2 confusion matrix in shades of navy, with counts and plain labels."""
    mat = np.array([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("navy", ["#F4F6F9", NAVY])
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=mat.max())
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{v}", ha="center", va="center", fontsize=11, fontweight="semibold",
                color="white" if v > mat.max() * 0.55 else INK)
    ax.set_xticks([0, 1], ["mainstream", "conspiracy"])
    ax.set_yticks([0, 1], ["mainstream", "conspiracy"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True website type")
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="semibold", color=INK)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)


def single_confusion(counts: dict, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    confusion_panel(ax, counts, title)
    save(fig, path)


def test_confusions(table: pd.DataFrame, names: dict, path: Path):
    """The four final models on the test set, side by side."""
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.6))
    for ax, model in zip(axes, names):
        row = table.loc[model]
        confusion_panel(ax, {k: int(row[k]) for k in ("tp", "fn", "fp", "tn")}, names[model])
        if ax is not axes[0]:
            ax.set_ylabel("")
            ax.set_yticks([0, 1], ["", ""])
    fig.subplots_adjust(wspace=0.12)
    save(fig, path)


def ablation(table: pd.DataFrame, path: Path):
    """Validation F1 of the six feature models as dots; the planned model in navy."""
    labels = {"TF-IDF alone": "TF-IDF alone", "+ n-grams (1-3)": "+ phrases (n-grams)",
              "+ certainty words": "+ certainty and doubt words", "+ style": "+ style",
              "+ all three": "+ all three (planned model)",
              "+ certainty words and style (no n-grams)": "+ word lists and style"}
    rows = table.set_index("Features").loc[["+ certainty words", "+ n-grams (1-3)", "TF-IDF alone",
                                            "+ certainty words and style (no n-grams)", "+ style", "+ all three"]]
    base = float(rows.loc["TF-IDF alone", "F1"])
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    style(ax, grid="both")
    for y, (feat, row) in zip(range(len(rows))[::-1], rows.iterrows()):
        f1 = float(row["F1"])
        colour = NAVY if feat == "+ all three" else (INK if feat == "TF-IDF alone" else SLATE)
        ax.scatter(f1, y, s=46, color=colour, zorder=3)
        left = f1 < base
        ax.text(f1 + (-0.0013 if left else 0.0013), y, apa(f1), va="center", ha="right" if left else "left",
                fontsize=8.5, color=INK, fontweight="semibold" if feat in ("+ all three", "TF-IDF alone") else "normal")
    ax.axvline(base, color=SLATE, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    ax.set_yticks(range(len(rows))[::-1], [labels[f] for f in rows.index])
    ax.set_xlim(0.88, 0.94)
    ax.set_xlabel("Validation F1 (conspiracy class; axis starts at .88)")
    apa_axis(ax)
    save(fig, path)


def f1_drop(drop: pd.DataFrame, path: Path):
    """Validation, test and test without Website 4 for each model."""
    without_col = drop.columns[3]
    rows = drop.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    style(ax)
    for i, row in rows.iterrows():
        valid, test, wo = float(row["Validation F1"]), float(row["Test F1"]), float(row[without_col])
        ax.plot([test, valid], [i, i], color=GRID, linewidth=5, solid_capstyle="round", zorder=1)
        ax.scatter(valid, i, marker="D", s=34, color=SLATE, zorder=2, label="Validation" if i == 0 else None)
        ax.scatter(wo, i, s=60, facecolors="white", edgecolors=GOLD, linewidths=1.8, zorder=3,
                   label="Test without Website 4" if i == 0 else None)
        ax.scatter(test, i, s=40, color=NAVY, zorder=4, label="Test" if i == 0 else None)
        ax.text(test, i - 0.32, apa(test), ha="center", fontsize=7.5, color=NAVY, fontweight="semibold")
        ax.text(wo, i + 0.2, apa(wo), ha="center", fontsize=7.5, color=GOLD)
    ax.set_yticks(rows.index, rows["Model"])
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("F1 for the conspiracy class (axis starts at .50)")
    apa_axis(ax)
    leg = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK)
    save(fig, path)


def website_skyline(sizes: pd.DataFrame, path: Path, large_site: tuple[str, bool, int]):
    """Title visual: every article as a dot, stacked by website, grouped into training, validation and test."""
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = 0
    centres = {}
    for split in ("train", "validation", "test"):
        start = x
        part = sizes[sizes["split"] == split].sort_values(["label", "articles"], ascending=[False, False])
        for _, row in part.iterrows():
            colour = CONSPIRACY if row["label"] else MAINSTREAM
            if (split, bool(row["label"]), int(row["articles"])) == large_site:
                colour = GOLD                                    # Website 4
            n = int(row["articles"])
            ax.scatter(np.full(n, x), np.arange(n), s=2.6 if colour == GOLD else 1.8, color=colour, linewidths=0)
            x += 1
        centres[split] = (start + x - 1) / 2
        x += 5                                                   # gap between the sets
    for split, cx in centres.items():
        ax.text(cx, -6, split if split != "train" else "training", ha="center", va="top", fontsize=8, color=SLATE)
    ax.set_xlim(-2, x - 3)
    ax.set_ylim(-12, sizes["articles"].max() + 2)
    ax.axis("off")
    fig.patch.set_alpha(0)                                       # no white box on the title page
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)


def build_all(repo: Path, out: Path, sizes: pd.DataFrame | None) -> None:
    """Write every figure the PDF uses into `out`."""
    out.mkdir(parents=True, exist_ok=True)
    tables = repo / "results" / "tables"
    metrics = json.loads((repo / "results" / "metrics.json").read_text(encoding="utf-8"))
    single_confusion(metrics["word_counts"]["confusion"], "Word counts (validation)", out / "cm_word_counts.pdf")
    single_confusion(metrics["tfidf"]["confusion"], "TF-IDF (validation)", out / "cm_tfidf.pdf")
    single_confusion(metrics["llm"]["validation"]["confusion"], "LLM (validation)", out / "cm_llm.pdf")
    names = {"final2_word_counts": "Word counts", "final2_tfidf": "TF-IDF",
             "final2_tfidf_features": "TF-IDF + features", "final_llm": "LLM"}
    test_confusions(pd.read_csv(tables / "comparison_test.csv").set_index("model"), names, out / "cm_test.pdf")
    ablation(pd.read_csv(tables / "ablation.csv"), out / "ablation.pdf")
    f1_drop(pd.read_csv(tables / "validation_vs_test.csv", dtype=str), out / "f1_drop.pdf")
    if sizes is not None:
        recall = pd.read_csv(tables / "test_website_recall.csv")
        big = int(recall["Test articles"].max())                   # the largest conspiracy test website
        website_skyline(sizes, out / "title_visual.pdf", ("test", True, big))
