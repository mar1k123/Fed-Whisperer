"""
PyTorch Geometric GNN Models for Fraud Ring Node & Edge Classification:
- `FraudGraphSAGE`: Inductive GraphSAGE (`SAGEConv`) with residual skip connections and node/edge heads.
- `FraudGAT`: Multi-head Graph Attention Network (`GATConv`) weighting suspicious counterparties & shared devices.
- Class Imbalance Losses:
  - `FocalLoss`: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
  - `WeightedBCELoss`: Pos-weight calibrated binary cross-entropy
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv, SAGEConv
except ImportError:
    Data = None  # type: ignore[assignment]
    GATConv = None  # type: ignore[assignment]
    SAGEConv = None  # type: ignore[assignment]

from ..config import GNNModelConfig
from ..features.graph_features import ALL_FEATURE_COLUMNS


class FocalLoss(nn.Module):
    """Binary Focal Loss for severe class imbalance (~0.5% - 2.0% fraud rate)."""

    def __init__(self, alpha: float = 0.85, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        modulating_factor = torch.pow(1.0 - p_t, self.gamma)
        loss = alpha_t * modulating_factor * bce_loss
        return loss.mean()


class WeightedBCELoss(nn.Module):
    """Class-weighted Binary Cross Entropy with automatic positive class weighting."""

    def __init__(self, pos_weight: float = 25.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=self.pos_weight.to(logits.device),
        )


class FraudGraphSAGE(nn.Module):
    """Inductive GraphSAGE model for node and transaction-edge fraud classification."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        curr_dim = in_channels
        for _ in range(num_layers):
            if SAGEConv is not None:
                self.convs.append(SAGEConv(curr_dim, hidden_channels, aggr="mean"))
            else:
                self.convs.append(nn.Linear(curr_dim, hidden_channels))
            self.norms.append(nn.LayerNorm(hidden_channels))
            curr_dim = hidden_channels

        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_channels + in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2 + 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv, norm in zip(self.convs, self.norms):
            if SAGEConv is not None and isinstance(conv, SAGEConv):
                h = conv(h, edge_index)
            else:
                h = conv(h)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encode(x, edge_index)
        combined = torch.cat([h, x], dim=-1)
        return self.node_classifier(combined).squeeze(-1)

    def predict_edges(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.encode(x, edge_index)
        src, dst = edge_index[0], edge_index[1]
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), 2), device=x.device)
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        return self.edge_classifier(edge_repr).squeeze(-1)


class FraudGAT(nn.Module):
    """Multi-head Graph Attention Network (GAT) learning attention weights over transaction and device edges."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        curr_dim = in_channels
        for layer_idx in range(num_layers):
            is_last = layer_idx == num_layers - 1
            out_heads = 1 if is_last else heads
            out_dim = hidden_channels * heads if is_last else hidden_channels
            if GATConv is not None:
                self.convs.append(GATConv(curr_dim, out_dim, heads=out_heads, concat=not is_last, dropout=dropout))
            else:
                self.convs.append(nn.Linear(curr_dim, hidden_channels * heads))
            self.norms.append(nn.LayerNorm(hidden_channels * heads))
            curr_dim = hidden_channels * heads

        final_dim = hidden_channels * heads
        self.node_classifier = nn.Sequential(
            nn.Linear(final_dim + in_channels, hidden_channels * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels * 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv, norm in zip(self.convs, self.norms):
            if GATConv is not None and isinstance(conv, GATConv):
                h = conv(h, edge_index)
            else:
                h = conv(h)
            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        combined = torch.cat([h, x], dim=-1)
        return self.node_classifier(combined).squeeze(-1)


def build_pyg_graph(
    features_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    labels: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, int]]:
    """
    Construct normalized node feature tensor `x` and bidirectional message-passing `edge_index`
    connecting both transaction counterparties and shared-device accounts within the window.
    """
    acc_ids = features_df["account_id"].tolist()
    acc_to_idx = {a: i for i, a in enumerate(acc_ids)}

    raw_x = features_df[ALL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    mean = raw_x.mean(axis=0, keepdims=True)
    std = raw_x.std(axis=0, keepdims=True) + 1e-6
    norm_x = (raw_x - mean) / std
    x_tensor = torch.tensor(norm_x, dtype=torch.float32)

    edges_src = []
    edges_dst = []
    device_groups: Dict[str, list] = {}

    for row in transactions_df.itertuples(index=False):
        u = acc_to_idx.get(str(row.src_account))
        v = acc_to_idx.get(str(row.dst_account))
        if u is not None and v is not None:
            edges_src.extend([u, v])
            edges_dst.extend([v, u])
        if row.device_id and u is not None:
            device_groups.setdefault(str(row.device_id), []).append(u)

    # Connect accounts sharing identical hardware devices inside the active time slice
    for members in device_groups.values():
        unique_members = sorted(set(members))
        if 1 < len(unique_members) <= 12:
            hub = unique_members[0]
            for m in unique_members[1:]:
                edges_src.extend([hub, m])
                edges_dst.extend([m, hub])

    if not edges_src:
        edges_src = list(range(len(acc_ids)))
        edges_dst = list(range(len(acc_ids)))

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    y_tensor = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
    return x_tensor, edge_index, y_tensor, acc_to_idx


def train_gnn_Classifier(
    model_type: str,
    config: GNNModelConfig,
    train_features: pd.DataFrame,
    train_tx: pd.DataFrame,
    y_train: np.ndarray,
    seed: int = 42,
) -> Tuple[nn.Module, Tuple[np.ndarray, np.ndarray]]:
    """Train GraphSAGE or GAT using Focal Loss or Weighted BCE Loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    raw_x = train_features[ALL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    feat_mean = raw_x.mean(axis=0, keepdims=True)
    feat_std = raw_x.std(axis=0, keepdims=True) + 1e-6

    x_train, edge_index, y_tensor, _ = build_pyg_graph(train_features, train_tx, y_train)
    in_channels = x_train.size(1)

    if model_type.lower() == "gat":
        model: nn.Module = FraudGAT(
            in_channels=in_channels,
            hidden_channels=config.hidden_channels,
            heads=config.heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
    else:
        model = FraudGraphSAGE(
            in_channels=in_channels,
            hidden_channels=config.hidden_channels,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

    pos_count = max(1, int(np.sum(y_train == 1)))
    neg_count = max(1, int(np.sum(y_train == 0)))
    if config.loss_type == "weighted_bce":
        criterion: nn.Module = WeightedBCELoss(pos_weight=float(neg_count / pos_count))
    else:
        criterion = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

    model.train()
    assert y_tensor is not None
    for _ in range(config.epochs):
        optimizer.zero_grad()
        logits = model(x_train, edge_index)
        loss = criterion(logits, y_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    return model, (feat_mean, feat_std)


@torch.no_grad()
def predict_gnn_proba(
    model: nn.Module,
    norm_stats: Tuple[np.ndarray, np.ndarray],
    features_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
) -> np.ndarray:
    """Compute node fraud probabilities on an unseen temporal graph snapshot."""
    model.eval()
    feat_mean, feat_std = norm_stats
    raw_x = features_df[ALL_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    norm_x = torch.tensor((raw_x - feat_mean) / feat_std, dtype=torch.float32)
    _, edge_index, _, _ = build_pyg_graph(features_df, transactions_df, None)
    logits = model(norm_x, edge_index)
    probs = torch.sigmoid(logits).cpu().numpy()
    return np.asarray(probs, dtype=float)
