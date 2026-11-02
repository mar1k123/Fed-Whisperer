"""Inter-method agreement metrics (Pearson r, Spearman rho, Cohen's kappa, directional accuracy)."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def discretize_tone(scores: np.ndarray, threshold: float = 0.12) -> np.ndarray:
    """Map continuous tone scores in [-1, 1] into {-1: Dovish, 0: Neutral, +1: Hawkish}."""
    labels = np.zeros_like(scores, dtype=int)
    labels[scores > threshold] = 1
    labels[scores < -threshold] = -1
    return labels


def compute_method_agreement(
    df: pd.DataFrame,
    cols: tuple[str, str, str] = ("dict_score", "embed_score", "llm_score"),
    threshold: float = 0.12,
) -> Dict[str, Dict[str, float]]:
    """
    Compute pairwise agreement statistics across the three tone scoring methods:
    - Pearson linear correlation (r)
    - Spearman rank correlation (rho)
    - 3-class Cohen's kappa (Dovish / Neutral / Hawkish)
    - Directional agreement rate on first differences (delta tone sign match)
    """
    pairs = [
        (cols[0], cols[1]),
        (cols[0], cols[2]),
        (cols[1], cols[2]),
    ]
    results: Dict[str, Dict[str, float]] = {}

    for c1, c2 in pairs:
        sub = df[[c1, c2]].dropna()
        x = sub[c1].to_numpy(dtype=float)
        y = sub[c2].to_numpy(dtype=float)

        pearson_r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else 0.0
        spearman_rho = float(pd.Series(x).corr(pd.Series(y), method="spearman")) if len(x) > 1 else 0.0

        cat_x = discretize_tone(x, threshold=threshold)
        cat_y = discretize_tone(y, threshold=threshold)
        kappa = float(cohen_kappa_score(cat_x, cat_y)) if len(x) > 1 else 0.0

        dx = np.sign(np.diff(x))
        dy = np.sign(np.diff(y))
        dir_match = float(np.mean(dx == dy)) if len(dx) > 0 else 0.0

        key = f"{c1}_vs_{c2}"
        results[key] = {
            "pearson_r": round(pearson_r, 4),
            "spearman_rho": round(spearman_rho, 4),
            "cohen_kappa": round(kappa, 4),
            "delta_sign_agreement": round(dir_match, 4),
        }

    return results
