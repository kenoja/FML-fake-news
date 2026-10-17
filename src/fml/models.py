"""Logistic-regression pipelines, tuned on the validation split only.

Everything that is learned from text (vocabulary, IDF weights, scaler means, coefficients) sits inside
one sklearn Pipeline, so `fit()` on the training rows is the only place it is learned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import classification_metrics
from .paths import project_config
from .split import load_split

TEXT_COLUMN = "txt"
SELECTION_METRIC = "f1"          # conspiracy-class F1 on validation
DEFAULT_FEATURES = {"ngram_range": (1, 1), "min_df": 2, "max_features": None, "sublinear_tf": True,
                    "numeric_columns": [], "dense_groups": []}

_BLOCK_CACHE: dict[tuple, object] = {}      # fitted feature blocks, reused inside one run
_BLOCK_CACHE_MAX = 24


def remember(key: tuple, make):
    """Return the cached value for `key`, or make it once and keep it."""
    if key not in _BLOCK_CACHE:
        if len(_BLOCK_CACHE) >= _BLOCK_CACHE_MAX:
            _BLOCK_CACHE.clear()                # keep memory small
        _BLOCK_CACHE[key] = make()
    return _BLOCK_CACHE[key]


class SharedPart(BaseEstimator, TransformerMixin):
    """One feature block that is fitted once per set of training texts and then reused."""

    def __init__(self, inner):
        self.inner = inner

    @staticmethod
    def texts_key(X) -> int:
        """A short fingerprint of the texts a block is fitted on."""
        return hash(tuple(np.asarray(X).ravel().tolist()))

    def fit(self, X, y=None):
        """Fit the block on the training texts (or reuse the same fit)."""
        key = ("fit", repr(self.inner), self.texts_key(X))
        self.fitted_ = remember(key, lambda: clone(self.inner).fit(X, y))
        self.fit_key_ = key
        return self

    def transform(self, X):
        """Apply the fitted block to any texts."""
        key = ("transform", self.fit_key_, self.texts_key(X))
        return remember(key, lambda: self.fitted_.transform(X))

    def __getattr__(self, name: str):
        """Give access to what the block learned (`vocabulary_`, `idf_`)."""
        if name.startswith("__") or name in ("fitted_", "inner"):
            raise AttributeError(name)
        return getattr(self.fitted_, name)

    def get_feature_names_out(self, input_features=None):
        """Names of the columns this block makes."""
        return self.fitted_.get_feature_names_out(input_features)


def make_classifier(C: float, class_weight) -> LogisticRegression:
    """The logistic regression used everywhere (L2 penalty is sklearn's default)."""
    seed = project_config()["split"]["seed"]
    return LogisticRegression(C=C, class_weight=class_weight, solver="liblinear", max_iter=2000, random_state=seed)


def build_pipeline(feature_cfg: dict | None = None, C: float = 1.0, class_weight=None) -> Pipeline:
    """TF-IDF (plus optional scaled extra features) followed by logistic regression."""
    from .features import scaled

    cfg = {**DEFAULT_FEATURES, **(feature_cfg or {})}
    tfidf = TfidfVectorizer(ngram_range=tuple(cfg["ngram_range"]), min_df=cfg["min_df"],
                            sublinear_tf=cfg["sublinear_tf"], lowercase=True, max_features=cfg["max_features"])
    parts = [("tfidf", SharedPart(tfidf), TEXT_COLUMN)]
    for group in cfg["dense_groups"]:
        parts.append((group, SharedPart(scaled(group)), TEXT_COLUMN))     # scaler is fitted on train only
    if cfg["numeric_columns"]:
        parts.append(("scale", StandardScaler(), list(cfg["numeric_columns"])))
    features = ColumnTransformer(parts)
    return Pipeline([("features", features), ("clf", make_classifier(C, class_weight))])


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


def tune_on_validation(train: pd.DataFrame, valid: pd.DataFrame, feature_cfg: dict | None = None,
                       c_grid=None, class_weights=None) -> tuple[Pipeline, pd.DataFrame]:
    """Try every (C, class weight) on train, score on validation, return the best pipeline and the grid."""
    lr = project_config()["lr"]
    c_grid = lr["c_grid"] if c_grid is None else c_grid         # same small grid as analyses 1 and 2
    class_weights = [lr["class_weight"]] if class_weights is None else class_weights
    features = build_pipeline(feature_cfg).named_steps["features"]
    features.fit(train, train["label"])             # vocabulary and scalers: training rows only
    X_train = features.transform(train)
    X_valid = features.transform(valid)
    y_train = train["label"]

    def score(C, class_weight) -> dict:
        model = make_classifier(C, class_weight).fit(X_train, y_train)
        metrics = classification_metrics(valid["label"], model.predict(X_valid))
        return {"C": C, "class_weight": "none" if class_weight is None else class_weight, **metrics}

    rows = Parallel(n_jobs=-1)(delayed(score)(C, cw) for cw in class_weights for C in c_grid)
    grid = pd.DataFrame(rows)
    best_row = int(grid[SELECTION_METRIC].to_numpy().argmax())      # ties go to the first (smaller C)
    grid["selected"] = grid.index == best_row
    best = grid.iloc[best_row]
    class_weight = None if best["class_weight"] == "none" else best["class_weight"]
    classifier = make_classifier(float(best["C"]), class_weight).fit(X_train, y_train)
    return Pipeline([("features", features), ("clf", classifier)]), grid


def tune_variant(train, valid, features, c_grid=None):
    """Tune one feature set on validation; return the pipeline and its best (C, class weight, F1)."""
    pipe, grid = tune_on_validation(train, valid, features, c_grid=c_grid)
    best = grid.query("selected").iloc[0]
    row = {"C": float(best["C"]), "class_weight": best["class_weight"], "f1": float(best["f1"])}
    return pipe, row


def top_coefficients(pipe: Pipeline, k: int = 25) -> pd.DataFrame:
    """The k words pushing most towards conspiracy and the k most towards mainstream."""
    names = pipe.named_steps["features"].get_feature_names_out()
    table = pd.DataFrame({"feature": [str(n).split("__", 1)[-1] for n in names],
                          "coefficient": pipe.named_steps["clf"].coef_[0]})
    # sort by coefficient, then by word, so ties always come out in the same order
    conspiracy = table.sort_values(["coefficient", "feature"], ascending=[False, True]).head(k)
    mainstream = table.sort_values(["coefficient", "feature"], ascending=[True, True]).head(k)
    both = [conspiracy.assign(direction="conspiracy"), mainstream.assign(direction="mainstream")]
    return pd.concat(both)[["direction", "feature", "coefficient"]].reset_index(drop=True)


# ---------------------------------------------------------------- analyses 1 and 2 (shared helpers)

def fit_logistic(X_train, y_train, C: float, class_weight) -> LogisticRegression:
    """Fit one logistic regression on the training rows."""
    seed = project_config()["split"]["seed"]
    model = LogisticRegression(C=C, class_weight=class_weight, solver="liblinear", random_state=seed)
    return model.fit(X_train, y_train)


def choose_c(X_train, y_train, X_valid, y_valid) -> tuple[LogisticRegression, pd.DataFrame]:
    """Try each C on train, score on validation, return the best model and the grid."""
    lr = project_config()["lr"]
    rows = []
    for C in lr["c_grid"]:
        model = fit_logistic(X_train, y_train, C, lr["class_weight"])
        scores = classification_metrics(y_valid, model.predict(X_valid))
        rows.append({"C": C, "precision": scores["precision"], "recall": scores["recall"], "f1": scores["f1"]})
    grid = pd.DataFrame(rows)
    best = int(grid["f1"].to_numpy().argmax())          # ties go to the smaller C
    grid["chosen"] = grid.index == best
    model = fit_logistic(X_train, y_train, float(grid.loc[best, "C"]), lr["class_weight"])
    return model, grid


def word_counts_vectorizer() -> CountVectorizer:
    """Analysis 1: how often each word occurs (words in at least 2 articles)."""
    return CountVectorizer(lowercase=True, min_df=2)


def tfidf_vectorizer() -> TfidfVectorizer:
    """Analysis 2: the same words, weighted by TF-IDF."""
    return TfidfVectorizer(lowercase=True, min_df=2, sublinear_tf=True)


def top_words(vectorizer, model: LogisticRegression, k: int = 25) -> pd.DataFrame:
    """The k words pushing most towards conspiracy and the k most towards mainstream."""
    table = pd.DataFrame({"word": vectorizer.get_feature_names_out(), "weight": model.coef_[0]})
    conspiracy = table.sort_values(["weight", "word"], ascending=[False, True]).head(k)
    mainstream = table.sort_values(["weight", "word"], ascending=[True, True]).head(k)
    both = [conspiracy.assign(direction="conspiracy"), mainstream.assign(direction="mainstream")]
    return pd.concat(both)[["direction", "word", "weight"]].reset_index(drop=True)
