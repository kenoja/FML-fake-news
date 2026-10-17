"""Load the 9/11 part of the LOCO corpus and mark off-topic articles."""
from __future__ import annotations

import gzip
import json
import re

import pandas as pd

from .paths import data_dir, project_config


def load_corpus() -> pd.DataFrame:
    """Read all documents; `label` is True for conspiracy-oriented websites."""
    cfg = project_config()["data"]
    path = data_dir() / cfg["subset_file"]
    with gzip.open(path, "rt", encoding="utf-8") as f:
        df = pd.DataFrame(json.loads(line) for line in f)
    df["label"] = df[cfg["label_field"]] == cfg["positive_label"]     # the label belongs to the website
    return df


def on_topic_mask(df: pd.DataFrame) -> pd.Series:
    """True where the title or text mentions 9/11 (configured pattern)."""
    pattern = re.compile(project_config()["data"]["topic_filter_pattern"], re.IGNORECASE)
    text = df["title"].fillna("") + " " + df["txt"].fillna("")
    return text.map(lambda t: bool(pattern.search(t)))


def summary(df: pd.DataFrame) -> dict:
    """Document and website counts, also a check that the data repo is reachable."""
    on_topic = on_topic_mask(df)
    is_conspiracy = df["label"]
    return {
        "documents": int(len(df)),
        "conspiracy": int(is_conspiracy.sum()),
        "mainstream": int((~is_conspiracy).sum()),
        "websites_conspiracy": int(df.loc[is_conspiracy, "website"].nunique()),
        "websites_mainstream": int(df.loc[~is_conspiracy, "website"].nunique()),
        "on_topic_documents": int(on_topic.sum()),
        "on_topic_conspiracy": int((on_topic & is_conspiracy).sum()),
        "on_topic_mainstream": int((on_topic & ~is_conspiracy).sum()),
    }
