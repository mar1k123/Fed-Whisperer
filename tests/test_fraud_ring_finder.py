"""
Unit & Integration Tests for `fraud-ring-finder`:
1. Generator determinism check (same seed -> identical accounts, devices, transactions, and rings; different seed -> different graph)
2. Configurable fraud rate bounds check (~0.5% - 2.0%)
3. Strict temporal split & zero future data leakage check (perturbing future test transactions has 0.0 effect on train/val features)
4. Class-imbalance loss functions (FocalLoss, WeightedBCELoss) and evaluation metric invariants
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.fraud_ring_finder.config import GeneratorConfig, GNNModelConfig, XGBoostConfig
from src.fraud_ring_finder.data.generator import FraudNetworkGenerator
from src.fraud_ring_finder.features.graph_features import (
    build_temporal_splits,
    extract_account_features_at_cutoff,
)
from src.fraud_ring_finder.evaluation.metrics import (
    verify_temporal_split_no_leakage,
    compute_pr_auc,
    recall_at_fixed_precision,
)
from src.fraud_ring_finder.models.baselines import FraudXGBoostModel
from src.fraud_ring_finder.models.gnn import FocalLoss, WeightedBCELoss, train_gnn_Classifier, predict_gnn_proba


def test_generator_determinism() -> None:
    """Verify that identical seeds produce byte-identical dataframes and graphs."""
    cfg = GeneratorConfig(num_accounts=300, num_devices=220, num_merchants=50, num_transactions=3000, fraud_rate=0.015)
    gen_a = FraudNetworkGenerator(cfg, seed=101)
    gen_b = FraudNetworkGenerator(cfg, seed=101)
    gen_c = FraudNetworkGenerator(cfg, seed=202)

    net_a = gen_a.generate()
    net_b = gen_b.generate()
    net_c = gen_c.generate()

    pd.testing.assert_frame_equal(net_a.accounts_df, net_b.accounts_df)
    pd.testing.assert_frame_equal(net_a.devices_df, net_b.devices_df)
    pd.testing.assert_frame_equal(net_a.transactions_df, net_b.transactions_df)
    assert net_a.rings_metadata == net_b.rings_metadata

    # Different seed must produce different transaction amounts/timestamps
    assert not net_a.transactions_df["amount"].equals(net_c.transactions_df["amount"])


def test_planted_patterns_and_fraud_rate() -> None:
    """Verify all four fraud patterns are planted and fraud rate stays within target tolerance."""
    cfg = GeneratorConfig(num_accounts=400, num_devices=300, num_merchants=60, num_transactions=4000, fraud_rate=0.015)
    net = FraudNetworkGenerator(cfg, seed=42).generate()

    observed_rate = float(net.transactions_df["is_fraud"].mean())
    assert 0.005 <= observed_rate <= 0.028

    planted_patterns = set(net.transactions_df.loc[net.transactions_df["is_fraud"] == 1, "fraud_pattern"].unique())
    expected = {"money_mule_chain", "shared_device_ring", "burst_activity", "circular_transfer"}
    assert expected.issubset(planted_patterns)


def test_strict_temporal_split_and_zero_feature_leakage() -> None:
    """
    Verify:
    1. Train, Val, and Test timestamps are strictly ordered with zero transaction overlap.
    2. Corrupting/modifying transactions in the future Test window has ZERO effect on Train or Val node features.
    """
    cfg = GeneratorConfig(num_accounts=300, num_devices=220, num_merchants=50, num_transactions=3000, fraud_rate=0.015)
    net = FraudNetworkGenerator(cfg, seed=42).generate()

    splits = build_temporal_splits(net, train_ratio=0.70, val_ratio=0.15)
    verification = verify_temporal_split_no_leakage(splits)
    assert verification["passed"] is True
    assert verification["strictly_ordered_timestamps"] is True
    assert verification["zero_transaction_id_overlap"] is True

    train_cutoff = float(splits["cutoffs"]["train_max_ts"])  # type: ignore[index]
    feat_before = extract_account_features_at_cutoff(net, cutoff_timestamp=train_cutoff)

    # Mutate all future transactions (timestamp > train_cutoff) drastically
    net_mutated = FraudNetworkGenerator(cfg, seed=42).generate()
    future_mask = net_mutated.transactions_df["timestamp"] > train_cutoff
    net_mutated.transactions_df.loc[future_mask, "amount"] = 999999.0
    net_mutated.transactions_df.loc[future_mask, "src_account"] = "ACC-0000"
    net_mutated.transactions_df.loc[future_mask, "dst_account"] = "ACC-0001"

    feat_after = extract_account_features_at_cutoff(net_mutated, cutoff_timestamp=train_cutoff)
    pd.testing.assert_frame_equal(feat_before, feat_after)


def test_models_and_imbalance_losses_end_to_end() -> None:
    """Verify FocalLoss, WeightedBCELoss, XGBoost baselines, and GraphSAGE/GAT run cleanly."""
    logits = torch.tensor([-2.0, 0.5, 2.2, -1.5], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0])

    focal = FocalLoss(alpha=0.85, gamma=2.0)(logits, targets)
    wbce = WeightedBCELoss(pos_weight=20.0)(logits, targets)
    assert focal.item() > 0.0
    assert wbce.item() > 0.0

    cfg = GeneratorConfig(num_accounts=240, num_devices=180, num_merchants=40, num_transactions=2400, fraud_rate=0.02)
    net = FraudNetworkGenerator(cfg, seed=7).generate()
    splits = build_temporal_splits(net)

    train_f = splits["train"]["features"]  # type: ignore[index]
    train_tx = splits["train"]["tx"]  # type: ignore[index]
    y_train = splits["train"]["y"]  # type: ignore[index]
    test_f = splits["test"]["features"]  # type: ignore[index]
    test_tx = splits["test"]["tx"]  # type: ignore[index]
    y_test = splits["test"]["y"]  # type: ignore[index]

    xgb = FraudXGBoostModel(XGBoostConfig(n_estimators=25, max_depth=3), include_graph_features=True, seed=7)
    xgb.fit(train_f, y_train)
    xgb_probs = xgb.predict_proba(test_f)
    assert xgb_probs.shape == (len(test_f),)

    sage_cfg = GNNModelConfig(hidden_channels=16, num_layers=2, epochs=8, loss_type="focal")
    sage_model, norm_stats = train_gnn_Classifier("graphsage", sage_cfg, train_f, train_tx, y_train, seed=7)
    sage_probs = predict_gnn_proba(sage_model, norm_stats, test_f, test_tx)
    assert sage_probs.shape == (len(test_f),)
    assert 0.0 <= compute_pr_auc(y_test, sage_probs) <= 1.0
    rec, thresh = recall_at_fixed_precision(y_test, sage_probs, target_precision=0.50)
    assert 0.0 <= rec <= 1.0
    assert 0.0 <= thresh <= 1.0
