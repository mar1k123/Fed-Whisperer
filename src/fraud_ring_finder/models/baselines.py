"""
XGBoost Baselines for Fraud Ring Detection:
1. `TabularXGBoostBaseline`: Trained strictly on per-account/per-transaction tabular features.
2. `GraphEnhancedXGBoostBaseline`: Trained on tabular features + hand-engineered graph topology features
   (in/out degree, PageRank, shared-device count, flow pass-through, and cycle counts).
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore[assignment]

from sklearn.ensemble import GradientBoostingClassifier
from ..config import XGBoostConfig
from ..features.graph_features import TABULAR_FEATURE_COLUMNS, ALL_FEATURE_COLUMNS


class FraudXGBoostModel:
    """Wrapper supporting both Tabular-Only and Tabular+Graph feature sets with automatic class-weight scaling."""

    def __init__(
        self,
        config: XGBoostConfig,
        include_graph_features: bool = False,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.include_graph_features = include_graph_features
        self.seed = seed
        self.feature_columns: List[str] = (
            list(ALL_FEATURE_COLUMNS) if include_graph_features else list(TABULAR_FEATURE_COLUMNS)
        )
        self.model: Optional[object] = None

    def fit(self, features_df: pd.DataFrame, y: np.ndarray) -> "FraudXGBoostModel":
        X = features_df[self.feature_columns].to_numpy(dtype=float)
        pos_count = max(1, int(np.sum(y == 1)))
        neg_count = max(1, int(np.sum(y == 0)))
        scale_pos_weight = float(neg_count / pos_count)

        if XGBClassifier is not None:
            clf = XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                scale_pos_weight=scale_pos_weight,
                eval_metric=self.config.eval_metric,
                random_state=self.seed,
                n_jobs=1,
            )
            clf.fit(X, y)
            self.model = clf
        else:
            sample_weights = np.where(y == 1, scale_pos_weight, 1.0)
            fallback = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                random_state=self.seed,
            )
            fallback.fit(X, y, sample_weight=sample_weights)
            self.model = fallback

        return self

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model must be fit before calling predict_proba.")
        X = features_df[self.feature_columns].to_numpy(dtype=float)
        probs = self.model.predict_proba(X)  # type: ignore[attr-defined]
        return np.asarray(probs[:, 1], dtype=float)
