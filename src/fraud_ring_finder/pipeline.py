"""
End-to-End Reproducible Evaluation & Ablation Pipeline for `fraud-ring-finder`.

Runs:
1. Seeded synthetic network generation (standard test window + shifted-tactics test window)
2. Leak-free temporal split verification
3. Training of 4 models:
   - `XGBoost (Tabular Only)`
   - `XGBoost + Hand-Crafted Graph Features`
   - `GraphSAGE (Focal Loss)`
   - `GAT (Multi-Head + Focal Loss)`
4. Test set evaluation: PR-AUC, Recall@Precision=80%, Pattern-Level Ablation, and Distribution-Shift Robustness
5. Exports structured JSON results to `artifacts/latest_metrics.json`.

Usage:
    python -m src.fraud_ring_finder.pipeline --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from .config import load_config
from .data.generator import FraudNetworkGenerator
from .features.graph_features import build_temporal_splits
from .models.baselines import FraudXGBoostModel
from .models.gnn import train_gnn_Classifier, predict_gnn_proba
from .evaluation.metrics import (
    verify_temporal_split_no_leakage,
    evaluate_model_predictions,
)


def run_pipeline(config_path: str = "configs/config.yaml") -> Dict[str, object]:
    cfg = load_config(config_path)
    generator = FraudNetworkGenerator(cfg.generator, seed=cfg.seed)

    # 1. Generate Standard & Post-Training Shifted Networks
    net_standard = generator.generate(shifted_tactics=False)
    net_shifted = generator.generate(shifted_tactics=True)

    splits_std = build_temporal_splits(
        net_standard,
        train_ratio=cfg.split.train_ratio,
        val_ratio=cfg.split.val_ratio,
    )
    splits_shift = build_temporal_splits(
        net_shifted,
        train_ratio=cfg.split.train_ratio,
        val_ratio=cfg.split.val_ratio,
    )

    leakage_check = verify_temporal_split_no_leakage(splits_std)
    if not leakage_check["passed"]:
        raise RuntimeError(f"Temporal split leakage check failed: {leakage_check}")

    train_feat = splits_std["train"]["features"]  # type: ignore[index]
    train_tx = splits_std["train"]["tx"]  # type: ignore[index]
    y_train = splits_std["train"]["y"]  # type: ignore[index]

    test_feat = splits_std["test"]["features"]  # type: ignore[index]
    test_tx = splits_std["test"]["tx"]  # type: ignore[index]
    y_test = splits_std["test"]["y"]  # type: ignore[index]
    pat_test = splits_std["test"]["patterns"]  # type: ignore[index]

    shift_feat = splits_shift["test"]["features"]  # type: ignore[index]
    shift_tx = splits_shift["test"]["tx"]  # type: ignore[index]
    y_shift = splits_shift["test"]["y"]  # type: ignore[index]
    pat_shift = splits_shift["test"]["patterns"]  # type: ignore[index]

    # 2. Train Baselines
    xgb_tab = FraudXGBoostModel(cfg.xgboost, include_graph_features=False, seed=cfg.seed).fit(train_feat, y_train)
    xgb_graph = FraudXGBoostModel(cfg.xgboost, include_graph_features=True, seed=cfg.seed).fit(train_feat, y_train)

    # 3. Train GNNs (GraphSAGE & GAT)
    sage_model, sage_norm = train_gnn_Classifier("graphsage", cfg.graphsage, train_feat, train_tx, y_train, seed=cfg.seed)
    gat_model, gat_norm = train_gnn_Classifier("gat", cfg.gat, train_feat, train_tx, y_train, seed=cfg.seed)

    # 4. Evaluate on Standard Temporal Test Split
    models_std_scores = {
        "XGBoost (Tabular)": xgb_tab.predict_proba(test_feat),
        "XGBoost + Graph Features": xgb_graph.predict_proba(test_feat),
        "GraphSAGE (Focal Loss)": predict_gnn_proba(sage_model, sage_norm, test_feat, test_tx),
        "GAT (Multi-Head Focal)": predict_gnn_proba(gat_model, gat_norm, test_feat, test_tx),
    }

    models_shift_scores = {
        "XGBoost (Tabular)": xgb_tab.predict_proba(shift_feat),
        "XGBoost + Graph Features": xgb_graph.predict_proba(shift_feat),
        "GraphSAGE (Focal Loss)": predict_gnn_proba(sage_model, sage_norm, shift_feat, shift_tx),
        "GAT (Multi-Head Focal)": predict_gnn_proba(gat_model, gat_norm, shift_feat, shift_tx),
    }

    standard_eval = {}
    shifted_eval = {}
    for name, scores in models_std_scores.items():
        standard_eval[name] = evaluate_model_predictions(
            y_test,
            scores,
            pat_test,
            target_precisions=cfg.evaluation.fixed_precision_targets,
            primary_precision=cfg.evaluation.primary_precision_target,
        )
        shift_scores = models_shift_scores[name]
        shifted_eval[name] = evaluate_model_predictions(
            y_shift,
            shift_scores,
            pat_shift,
            target_precisions=cfg.evaluation.fixed_precision_targets,
            primary_precision=cfg.evaluation.primary_precision_target,
        )

    results = {
        "seed": cfg.seed,
        "generator_summary": {
            "num_accounts": len(net_standard.accounts_df),
            "num_devices": len(net_standard.devices_df),
            "num_merchants": len(net_standard.merchants_df),
            "num_transactions": len(net_standard.transactions_df),
            "observed_tx_fraud_rate": round(float(net_standard.transactions_df["is_fraud"].mean()), 4),
            "planted_rings_count": len(net_standard.rings_metadata),
        },
        "leakage_verification": leakage_check,
        "standard_test_evaluation": standard_eval,
        "distribution_shift_evaluation": shifted_eval,
    }

    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "latest_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fraud-ring-finder training & evaluation pipeline.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config file.")
    args = parser.parse_args()
    results = run_pipeline(args.config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
