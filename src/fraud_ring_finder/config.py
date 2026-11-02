"""Configuration loader and typed schema for fraud-ring-finder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import yaml


@dataclass
class DistributionShiftConfig:
    enabled: bool = True
    mule_chain_length_shifted: int = 6
    mule_hop_delay_minutes_shifted: int = 95
    device_ring_proxy_rotation_rate: float = 0.50
    burst_smurfing_threshold_usd: float = 490.0


@dataclass
class GeneratorConfig:
    num_accounts: int = 1200
    num_devices: int = 920
    num_merchants: int = 160
    num_transactions: int = 16000
    fraud_rate: float = 0.015
    time_span_days: int = 30
    pattern_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "money_mule_chain": 0.30,
            "shared_device_ring": 0.25,
            "burst_activity": 0.25,
            "circular_transfer": 0.20,
        }
    )
    distribution_shift: DistributionShiftConfig = field(default_factory=DistributionShiftConfig)


@dataclass
class SplitConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    strict_temporal_cutoff: bool = True


@dataclass
class FeaturesConfig:
    pagerank_alpha: float = 0.85
    velocity_windows_hours: List[int] = field(default_factory=lambda: [1, 6, 24])
    include_device_sharing: bool = True
    include_two_hop_motifs: bool = True


@dataclass
class XGBoostConfig:
    n_estimators: int = 220
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    eval_metric: str = "aucpr"


@dataclass
class GNNModelConfig:
    hidden_channels: int = 64
    heads: int = 4
    num_layers: int = 2
    dropout: float = 0.20
    lr: float = 0.003
    epochs: int = 80
    loss_type: str = "focal"
    focal_alpha: float = 0.85
    focal_gamma: float = 2.0


@dataclass
class EvaluationConfig:
    fixed_precision_targets: List[float] = field(default_factory=lambda: [0.70, 0.80, 0.90])
    primary_precision_target: float = 0.80
    explain_top_k_nodes: int = 15
    gnn_explainer_hops: int = 2
    gnn_explainer_epochs: int = 120


@dataclass
class ProjectConfig:
    name: str = "fraud-ring-finder"
    version: str = "1.0.0"
    seed: int = 42
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    graphsage: GNNModelConfig = field(default_factory=GNNModelConfig)
    gat: GNNModelConfig = field(default_factory=lambda: GNNModelConfig(hidden_channels=32, heads=4, lr=0.0025))
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_config(config_path: str | Path = "configs/config.yaml") -> ProjectConfig:
    """Load YAML configuration into typed dataclasses."""
    path = Path(config_path)
    if not path.exists():
        return ProjectConfig()

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    proj = raw.get("project", {})
    gen_raw = raw.get("generator", {})
    shift_raw = gen_raw.get("distribution_shift", {})
    split_raw = raw.get("split", {})
    feat_raw = raw.get("features", {})
    models_raw = raw.get("models", {})
    eval_raw = raw.get("evaluation", {})

    shift_cfg = DistributionShiftConfig(
        enabled=bool(shift_raw.get("enabled", True)),
        mule_chain_length_shifted=int(shift_raw.get("mule_chain_length_shifted", 6)),
        mule_hop_delay_minutes_shifted=int(shift_raw.get("mule_hop_delay_minutes_shifted", 95)),
        device_ring_proxy_rotation_rate=float(shift_raw.get("device_ring_proxy_rotation_rate", 0.50)),
        burst_smurfing_threshold_usd=float(shift_raw.get("burst_smurfing_threshold_usd", 490.0)),
    )

    gen_cfg = GeneratorConfig(
        num_accounts=int(gen_raw.get("num_accounts", 1200)),
        num_devices=int(gen_raw.get("num_devices", 920)),
        num_merchants=int(gen_raw.get("num_merchants", 160)),
        num_transactions=int(gen_raw.get("num_transactions", 16000)),
        fraud_rate=float(gen_raw.get("fraud_rate", 0.015)),
        time_span_days=int(gen_raw.get("time_span_days", 30)),
        pattern_weights=dict(
            gen_raw.get(
                "pattern_weights",
                {
                    "money_mule_chain": 0.30,
                    "shared_device_ring": 0.25,
                    "burst_activity": 0.25,
                    "circular_transfer": 0.20,
                },
            )
        ),
        distribution_shift=shift_cfg,
    )

    return ProjectConfig(
        name=str(proj.get("name", "fraud-ring-finder")),
        version=str(proj.get("version", "1.0.0")),
        seed=int(proj.get("seed", 42)),
        generator=gen_cfg,
        split=SplitConfig(**split_raw) if split_raw else SplitConfig(),
        features=FeaturesConfig(**feat_raw) if feat_raw else FeaturesConfig(),
        xgboost=XGBoostConfig(**models_raw.get("xgboost", {})),
        graphsage=GNNModelConfig(**models_raw.get("graphsage", {})),
        gat=GNNModelConfig(**models_raw.get("gat", {})),
        evaluation=EvaluationConfig(**eval_raw) if eval_raw else EvaluationConfig(),
    )
