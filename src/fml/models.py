# Logistic regression on word counts, TF-IDF or TF-IDF plus extra features; settings are chosen on validation only.
from __future__ import annotations

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .evaluate import classification_metrics
from .features import scaled
from .paths import project_config
from .split import load_split


def split_frames(df: pd.DataFrame, which: tuple[str, ...] = ("train", "validation")) -> dict[str, pd.DataFrame]:
    """Rows of `df` for each requested split; the test split is refused here."""
    if "test" in which:
        raise PermissionError("the test split may only be read by fml.pipeline.evaluate_final")
    split = load_split()
    frames = {}
    for name in which:
        ids = set(split.loc[split["split"] == name, "doc_id"])
        frames[name] = df[df["doc_id"].isin(ids)].reset_index(drop=True)
    return frames


def word_counts_vectorizer() -> CountVectorizer:
    """Analysis 1: how often each word occurs (words in at least 2 articles)."""
    return CountVectorizer(lowercase=True, min_df=2)


def tfidf_vectorizer() -> TfidfVectorizer:
    """Analysis 2: the same words, weighted by TF-IDF."""
    return TfidfVectorizer(lowercase=True, min_df=2, sublinear_tf=True)


def fit_features(train: pd.DataFrame, feature_cfg: dict) -> list:
    """Fit every feature block on the training articles only; returns (column, block) pairs."""
    tfidf = TfidfVectorizer(ngram_range=tuple(feature_cfg.get("ngram_range", (1, 1))), min_df=feature_cfg["min_df"],
                            max_features=feature_cfg.get("max_features"), sublinear_tf=True, lowercase=True)
    blocks = [("txt", tfidf)]
    for group in feature_cfg.get("dense_groups", []):
        blocks.append(("txt", scaled(group)))
    numeric = list(feature_cfg.get("numeric_columns", []))
    if numeric:
        blocks.append((numeric, StandardScaler()))
    for column, block in blocks:
        block.fit(train[column])
    return blocks


def feature_matrix(blocks: list, df: pd.DataFrame) -> csr_matrix:
    """Put the matrices of all fitted blocks side by side."""
    parts = [csr_matrix(block.transform(df[column])) for column, block in blocks]
    return hstack(parts, format="csr")


def fit_logistic(X_train, y_train, C: float, class_weight) -> LogisticRegression:
    """Fit one logistic regression (L2 penalty is sklearn's default) on the training rows."""
    seed = project_config()["split"]["seed"]
    model = LogisticRegression(C=C, class_weight=class_weight, solver="liblinear", max_iter=2000, random_state=seed)
    return model.fit(X_train, y_train)


def tune_on_validation(X_train, y_train, X_valid, y_valid, c_grid=None) -> tuple[LogisticRegression, pd.DataFrame]:
    """Fit one model per C on train, score each on validation, return the best model and the grid."""
    lr = project_config()["lr"]
    rows = []
    for C in (lr["c_grid"] if c_grid is None else c_grid):
        model = fit_logistic(X_train, y_train, C, lr["class_weight"])
        scores = classification_metrics(y_valid, model.predict(X_valid))
        rows.append({"C": C, "precision": scores["precision"], "recall": scores["recall"], "f1": scores["f1"]})
    grid = pd.DataFrame(rows)
    best = int(grid["f1"].to_numpy().argmax())          # ties go to the smaller C
    grid["chosen"] = grid.index == best
    model = fit_logistic(X_train, y_train, float(grid.loc[best, "C"]), lr["class_weight"])
    return model, grid


def top_words(names, model: LogisticRegression, k: int = 25) -> pd.DataFrame:
    """The k words pushing most towards conspiracy and the k most towards mainstream."""
    table = pd.DataFrame({"word": names, "weight": model.coef_[0]})
    # sort by weight, then by word, so ties always come out in the same order
    conspiracy = table.sort_values(["weight", "word"], ascending=[False, True]).head(k)
    mainstream = table.sort_values(["weight", "word"], ascending=[True, True]).head(k)
    both = [conspiracy.assign(direction="conspiracy"), mainstream.assign(direction="mainstream")]
    return pd.concat(both)[["direction", "word", "weight"]].reset_index(drop=True)
