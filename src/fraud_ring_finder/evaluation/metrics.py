"""
Evaluation Suite for Fraud Ring Detection:
- Strict temporal split & zero-leakage verification (`verify_temporal_split_no_leakage`)
- Precision-Recall AUC (`compute_pr_auc`)
- Recall at Fixed Precision (`recall_at_fixed_precision`, e.g., Recall @ Precision >= 80%)
- Pattern-Level Ablation (`evaluate_pattern_ablation`) showing which planted fraud topologies
  (money_mule_chain, shared_device_ring, burst_activity, circular_transfer) are caught by which model
- Post-Training Distribution Shift Robustness (`evaluate_distribution_shift`)
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def verify_temporal_split_no_leakage(splits: Dict[str, object]) -> Dict[str, object]:
    """
    Verify that Train, Validation, and Test transaction windows are strictly ordered in time
    and have zero overlapping transaction IDs.
    """
    cutoffs = splits["cutoffs"]  # type: ignore[index]
    train_tx: pd.DataFrame = splits["train"]["tx"]  # type: ignore[index]
    val_tx: pd.DataFrame = splits["val"]["tx"]  # type: ignore[index]
    test_tx: pd.DataFrame = splits["test"]["tx"]  # type: ignore[index]

    train_max = float(train_tx["timestamp"].max())
    val_min = float(val_tx["timestamp"].min())
    val_max = float(val_tx["timestamp"].max())
    test_min = float(test_tx["timestamp"].min())

    train_ids = set(train_tx["tx_id"])
    val_ids = set(val_tx["tx_id"])
    test_ids = set(test_tx["tx_id"])

    overlap_train_val = len(train_ids.intersection(val_ids))
    overlap_val_test = len(val_ids.intersection(test_ids))
    overlap_train_test = len(train_ids.intersection(test_ids))

    strictly_ordered = bool((train_max < val_min) and (val_max < test_min))
    zero_tx_overlap = bool((overlap_train_val + overlap_val_test + overlap_train_test) == 0)

    return {
        "passed": strictly_ordered and zero_tx_overlap,
        "strictly_ordered_timestamps": strictly_ordered,
        "zero_transaction_id_overlap": zero_tx_overlap,
        "train_max_timestamp": train_max,
        "val_min_timestamp": val_min,
        "val_max_timestamp": val_max,
        "test_min_timestamp": test_min,
        "cutoffs": cutoffs,
    }


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the Precision-Recall Curve (PR-AUC / Average Precision)."""
    if np.sum(y_true == 1) == 0:
        return 0.0
    return float(average_precision_score(y_true, y_score))


def recall_at_fixed_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_precision: float = 0.80,
) -> Tuple[float, float]:
    """
    Return `(max_recall, threshold)` achieved at any operating threshold where `precision >= target_precision`.
    """
    if np.sum(y_true == 1) == 0:
        return 0.0, 1.0

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    # precisions and recalls have length len(thresholds) + 1
    valid_mask = precisions[:-1] >= target_precision
    if not np.any(valid_mask):
        best_idx = int(np.argmax(precisions[:-1]))
        return float(recalls[best_idx]), float(thresholds[best_idx])

    valid_recalls = recalls[:-1][valid_mask]
    valid_thresholds = thresholds[valid_mask]
    best_rel_idx = int(np.argmax(valid_recalls))
    return float(valid_recalls[best_rel_idx]), float(valid_thresholds[best_rel_idx])


def evaluate_model_predictions(
    y_true: np.ndarray,
    y_score: np.ndarray,
    patterns: List[str],
    target_precisions: List[float] = [0.70, 0.80, 0.90],
    primary_precision: float = 0.80,
) -> Dict[str, object]:
    """Compute overall PR-AUC, ROC-AUC, Recall@Precision targets, and per-pattern recall ablation."""
    pr_auc = compute_pr_auc(y_true, y_score)
    roc_auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else 0.0

    recall_at_p: Dict[str, float] = {}
    primary_threshold = 0.5
    for p_target in target_precisions:
        rec, thresh = recall_at_fixed_precision(y_true, y_score, target_precision=p_target)
        recall_at_p[f"recall@p={int(p_target * 100)}"] = round(rec, 4)
        if abs(p_target - primary_precision) < 1e-6:
            primary_threshold = thresh

    # Pattern-specific recall at the primary precision operating threshold
    preds_at_primary = (y_score >= primary_threshold).astype(int)
    pattern_names = ["money_mule_chain", "shared_device_ring", "burst_activity", "circular_transfer"]
    pattern_recall: Dict[str, float] = {}

    patterns_arr = np.asarray(patterns)
    for pat in pattern_names:
        mask = (patterns_arr == pat) & (y_true == 1)
        total_pat = int(np.sum(mask))
        if total_pat > 0:
            caught = int(np.sum(preds_at_primary[mask] == 1))
            pattern_recall[pat] = round(caught / total_pat, 4)
        else:
            pattern_recall[pat] = 0.0

    return {
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "recall_at_precision": recall_at_p,
        "operating_threshold_p80": round(primary_threshold, 4),
        "pattern_ablation_recall_at_p80": pattern_recall,
    }
