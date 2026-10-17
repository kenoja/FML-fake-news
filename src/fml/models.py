"""Logistic-regression pipelines and validation-only tuning.

Everything that learns from data (TF-IDF vocabulary and IDF weights, scaler means, coefficients) lives
inside one sklearn Pipeline, so `fit()` on the training rows is the only place it is learned. Fitting a
vectoriser on train+validation+test first would leak the other splits' vocabulary into training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluate import classification_metrics
from .paths import project_config
from .split import load_split

TEXT_COLUMN = "txt"
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
CLASS_WEIGHTS = (None, "balanced")
SELECTION_METRIC = "f1"          # conspiracy-class F1 on validation
DEFAULT_FEATURES = {"ngram_range": (1, 1), "min_df": 2, "max_features": None, "sublinear_tf": True,
                    "numeric_columns": [], "dense_groups": []}


_PART_CACHE: dict[tuple, object] = {}
_PART_CACHE_MAX = 24


class SharedPart(BaseEstimator, TransformerMixin):
    """Wraps one feature block so identical blocks are fitted and applied once per process.

    The ablation trains 8 models whose blocks overlap (the same 1-3-gram TF-IDF sits in four of them) and
    `evaluate_final` refits the winners on the same training rows. The cache key holds the block's settings
    and the exact texts it is fitted on, so a block can only be reused on the same training rows: nothing
    learned on one split is ever applied to a differently fitted block.
    """

    def __init__(self, inner):
        self.inner = inner

    @staticmethod
    def _texts_key(X) -> int:
        return hash(tuple(np.asarray(X).ravel().tolist()))

    def fit(self, X, y=None):
        key = ("fit", repr(self.inner), self._texts_key(X))
        if key not in _PART_CACHE:
            if len(_PART_CACHE) >= _PART_CACHE_MAX:
                _PART_CACHE.clear()
            _PART_CACHE[key] = clone(self.inner).fit(X, y)
        self.fitted_ = _PART_CACHE[key]
        self.fit_key_ = key
        return self

    def transform(self, X):
        key = ("transform", self.fit_key_, self._texts_key(X))
        if key not in _PART_CACHE:
            if len(_PART_CACHE) >= _PART_CACHE_MAX:
                _PART_CACHE.clear()
            _PART_CACHE[key] = self.fitted_.transform(X)
        return _PART_CACHE[key]

    def __getattr__(self, name: str):
        """Expose the fitted block's learned attributes (`vocabulary_`, `idf_`) for inspection and tests."""
        if name.startswith("__") or name in ("fitted_", "inner"):
            raise AttributeError(name)
        return getattr(self.fitted_, name)

    def get_feature_names_out(self, input_features=None):
        return self.fitted_.get_feature_names_out(input_features)


def build_pipeline(feature_cfg: dict | None = None, C: float = 1.0, class_weight=None,
                   random_state: int | None = None) -> Pipeline:
    """TF-IDF on the text column (+ optional standardised numeric columns) -> L2 logistic regression.

    sklearn's default penalty is L2 (ridge), which is what we want; it is not set explicitly because
    newer sklearn versions deprecate the `penalty` argument.

    Input X is a DataFrame with a `txt` column and any `numeric_columns` named in `feature_cfg`.
    `dense_groups` names feature groups from `fml.features.NUMERIC_GROUPS` ("epistemic", "style"); they are
    computed from `txt` and scaled inside the pipeline, so their scaler is fitted on the training rows only.
    """
    from .features import scaled

    cfg = {**DEFAULT_FEATURES, **(feature_cfg or {})}
    tfidf = TfidfVectorizer(ngram_range=tuple(cfg["ngram_range"]), min_df=cfg["min_df"],
                            sublinear_tf=cfg["sublinear_tf"], lowercase=True, max_features=cfg["max_features"])
    parts = [("tfidf", SharedPart(tfidf), TEXT_COLUMN)]
    parts += [(group, SharedPart(scaled(group)), TEXT_COLUMN) for group in cfg["dense_groups"]]
    if cfg["numeric_columns"]:
        parts.append(("scale", StandardScaler(), list(cfg["numeric_columns"])))
    seed = project_config()["split"]["seed"] if random_state is None else random_state
    clf = LogisticRegression(C=C, class_weight=class_weight, solver="liblinear", max_iter=2000, random_state=seed)
    return Pipeline([("features", ColumnTransformer(parts)), ("clf", clf)])


def split_frames(df: pd.DataFrame, which: tuple[str, ...] = ("train", "validation")) -> dict[str, pd.DataFrame]:
    """Rows of `df` per requested split. Asking for "test" here is an error: only evaluate_final may."""
    if "test" in which:
        raise PermissionError("the test split may only be read by fml.pipeline.evaluate_final")
    split = load_split()
    out = {}
    for name in which:
        ids = set(split.loc[split["split"] == name, "doc_id"])
        out[name] = df[df["doc_id"].isin(ids)].reset_index(drop=True)
    return out


def tune_on_validation(train: pd.DataFrame, valid: pd.DataFrame, feature_cfg: dict | None = None,
                       c_grid=C_GRID, class_weights=CLASS_WEIGHTS) -> tuple[Pipeline, pd.DataFrame]:
    """Try every (C, class_weight) on train, score on validation, return the best fitted pipeline.

    The returned grid lists every value tried (also the losers), so the choice can be audited. Ties go
    to the first row, i.e. the smaller C, i.e. the simpler model. Only train and validation rows are
    passed in: this function cannot see the test split.
    """
    # The feature block (vocabulary, IDF, scalers) depends only on the training rows, so it is fitted
    # once and reused for every (C, class_weight); only the cheap classifier is refitted.
    base = build_pipeline(feature_cfg)
    features = base.named_steps["features"].fit(train, train["label"])
    X_train, X_valid = features.transform(train), features.transform(valid)
    y_train = train["label"]
    seed = project_config()["split"]["seed"]

    def classifier(C, cw) -> LogisticRegression:
        return LogisticRegression(C=C, class_weight=cw, solver="liblinear", max_iter=2000, random_state=seed)

    def score(C, cw) -> dict:
        m = classification_metrics(valid["label"], classifier(C, cw).fit(X_train, y_train).predict(X_valid))
        return {"C": C, "class_weight": "none" if cw is None else cw, **m}

    # The fits are independent and the large-C ones on n-gram + dense features take seconds each, so they
    # run on all CPUs; the result order (and so the tie-break to the smaller C) is unchanged.
    rows = Parallel(n_jobs=-1)(delayed(score)(C, cw) for cw in class_weights for C in c_grid)
    grid = pd.DataFrame(rows)
    best = grid.iloc[int(grid[SELECTION_METRIC].to_numpy().argmax())]
    grid["selected"] = grid.index == best.name
    cw = None if best["class_weight"] == "none" else best["class_weight"]
    clf = classifier(float(best["C"]), cw).fit(X_train, y_train)
    final = Pipeline([("features", features), ("clf", clf)])
    return final, grid


def top_coefficients(pipe: Pipeline, k: int = 25) -> pd.DataFrame:
    """The k words pushing most towards conspiracy (positive) and mainstream (negative)."""
    names = pipe.named_steps["features"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    pick = [("conspiracy", i) for i in order[::-1][:k]] + [("mainstream", i) for i in order[:k]]
    return pd.DataFrame({"direction": [d for d, _ in pick],
                         "feature": [str(names[i]).split("__", 1)[-1] for _, i in pick],
                         "coefficient": [float(coef[i]) for _, i in pick]})
