"""Feature summary table and figure: what the numeric features look like per class (training rows only)."""
from __future__ import annotations

import pandas as pd

from . import NUMERIC_GROUPS
from ..results import CONSPIRACY_COLOUR, MAINSTREAM_COLOUR, savefig, write_table


def feature_matrix(texts) -> pd.DataFrame:
    """All epistemic and style columns for the given texts, unscaled."""
    parts = []
    for cls in NUMERIC_GROUPS.values():
        t = cls()
        parts.append(pd.DataFrame(t.transform(texts), columns=t.get_feature_names_out()))
    return pd.concat(parts, axis=1)


def feature_summary(train: pd.DataFrame) -> pd.DataFrame:
    """Per feature: mean per class and the number of training articles in each class."""
    feats = feature_matrix(train["txt"].tolist())
    y = train["label"].to_numpy(dtype=bool)
    rows = []
    for col in feats.columns:
        a, b = feats.loc[y, col].to_numpy(), feats.loc[~y, col].to_numpy()
        rows.append({"feature": col, "mean_conspiracy": a.mean(), "mean_mainstream": b.mean(),
                     "n_conspiracy": len(a), "n_mainstream": len(b)})
    return pd.DataFrame(rows)


def plot_style_by_class(train: pd.DataFrame):
    """Box plots of the six style features per class."""
    import matplotlib.pyplot as plt

    from .style import COLUMNS

    feats = feature_matrix(train["txt"].tolist())
    y = train["label"].to_numpy(dtype=bool)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, col in zip(axes.ravel(), COLUMNS):
        data = [feats.loc[~y, col], feats.loc[y, col]]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, tick_labels=["mainstream", "conspiracy"])
        for patch, colour in zip(bp["boxes"], (MAINSTREAM_COLOUR, CONSPIRACY_COLOUR)):
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)
        ax.set_title(col.replace("style_", "").replace("_", " "), fontsize=10)
    fig.suptitle("Style features by class (training documents only)")
    fig.tight_layout()
    return fig


EXAMPLE_SNIPPETS = {
    # invented sentences, not article text
    "invented_A": ("WAKE UP! The mainstream media will never tell you the hidden truth. Clearly it was an "
                   "inside job, and the cover-up goes on. Why do they hide it?"),
    "invented_B": ("Investigators reported that there is no evidence of a cover-up. The claim has been "
                   "debunked, although some details are possibly still disputed."),
}


def example_table() -> pd.DataFrame:
    """Feature values for the invented snippets."""
    feats = feature_matrix(list(EXAMPLE_SNIPPETS.values()))
    feats.insert(0, "n_words", [len(w.split()) for w in EXAMPLE_SNIPPETS.values()])
    feats.insert(0, "snippet", list(EXAMPLE_SNIPPETS))
    return feats


def features_stage() -> dict:
    """Write the feature summary table, the examples and the style figure."""
    from ..pipeline import analysis_frame
    from ..models import split_frames

    train = split_frames(analysis_frame(), ("train",))["train"]
    table = feature_summary(train)
    table.insert(1, "n_train_docs", len(train))
    write_table("feature_summary", table.round(4))
    write_table("feature_examples", example_table().round(3))
    savefig(plot_style_by_class(train), "style_features_by_class")
    return {"n_train_docs": int(len(train))}
