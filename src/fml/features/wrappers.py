"""scikit-learn wrapper for the style numbers."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from ..paths import project_config
from .style import COLUMNS, style_matrix


class StyleFeatures(BaseEstimator, TransformerMixin):
    """Thin scikit-learn wrapper around style_matrix. Nothing is learned."""

    def fit(self, X, y=None):
        return self

    def transform(self, X) -> np.ndarray:
        return style_matrix(X, int(project_config()["features"]["mattr_window"]))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(COLUMNS)
