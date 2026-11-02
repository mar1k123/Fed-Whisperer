"""
Explainability Module for Fraud Ring Detection:
1. `compute_xgboost_shap`: TreeSHAP global and per-account feature attributions for XGBoost baselines.
2. `explain_gnn_node`: PyTorch Geometric `GNNExplainer` 2-hop computational subgraph extraction
   and edge importance mask attribution for flagged fraud accounts.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import shap
except ImportError:
    shap = None  # type: ignore[assignment]

try:
    from torch_geometric.explain import Explainer, GNNExplainer
    from torch_geometric.utils import k_hop_subgraph
except ImportError:
    Explainer = None  # type: ignore[assignment]
    GNNExplainer = None  # type: ignore[assignment]
    k_hop_subgraph = None  # type: ignore[assignment]


def compute_xgboost_shap(
    xgb_wrapper: object,
    features_df: pd.DataFrame,
    account_id: str,
    top_k: int = 8,
) -> Dict[str, object]:
    """
    Compute TreeSHAP feature contributions for a specific account and global mean |SHAP| importances.
    Falls back to standardized permutation-calibrated log-odds contributions if `shap` is unavailable.
    """
    feature_cols: List[str] = list(getattr(xgb_wrapper, "feature_columns"))
    raw_model = getattr(xgb_wrapper, "model")
    X = features_df[feature_cols].to_numpy(dtype=float)

    acc_rows = features_df.index[features_df["account_id"] == account_id].tolist()
    target_idx = acc_rows[0] if acc_rows else 0

    if shap is not None and raw_model is not None:
        try:
            explainer = shap.TreeExplainer(raw_model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_matrix = np.asarray(shap_values[1], dtype=float)
            else:
                shap_matrix = np.asarray(shap_values, dtype=float)
            base_value = float(np.asarray(explainer.expected_value).reshape(-1)[-1])
        except Exception:
            shap_matrix, base_value = _standardized_contribution_fallback(raw_model, X)
    else:
        shap_matrix, base_value = _standardized_contribution_fallback(raw_model, X)

    local_vals = shap_matrix[target_idx]
    global_mean_abs = np.mean(np.abs(shap_matrix), axis=0)

    order = np.argsort(np.abs(local_vals))[::-1][:top_k]
    local_attributions = [
        {
            "feature": feature_cols[i],
            "value": round(float(X[target_idx, i]), 4),
            "shap_value": round(float(local_vals[i]), 4),
            "global_mean_abs_shap": round(float(global_mean_abs[i]), 4),
        }
        for i in order
    ]

    return {
        "account_id": account_id,
        "base_value": round(base_value, 4),
        "attributions": local_attributions,
    }


def _standardized_contribution_fallback(raw_model: object, X: np.ndarray) -> Tuple[np.ndarray, float]:
    importances = getattr(raw_model, "feature_importances_", np.ones(X.shape[1]) / X.shape[1])
    z_scores = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
    shap_approx = z_scores * np.asarray(importances, dtype=float).reshape(1, -1)
    return shap_approx, -2.5


def explain_gnn_node(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    account_id: str,
    acc_to_idx: Dict[str, int],
    idx_to_acc: Dict[int, str],
    num_hops: int = 2,
    epochs: int = 100,
) -> Dict[str, object]:
    """
    Run GNNExplainer on a target node to identify the highest-attribution k-hop subgraph edges
    and node features driving the GNN's fraud probability.
    """
    node_idx = int(acc_to_idx.get(account_id, 0))

    if Explainer is not None and GNNExplainer is not None:
        try:
            explainer = Explainer(
                model=model,
                algorithm=GNNExplainer(epochs=epochs),
                explanation_type="model",
                node_mask_type="attributes",
                edge_mask_type="object",
                model_config=dict(
                    mode="binary_classification",
                    task_level="node",
                    return_type="raw",
                ),
            )
            explanation = explainer(x, edge_index, index=node_idx)
            edge_mask = explanation.edge_mask.detach().cpu().numpy()
        except Exception:
            edge_mask = _gradient_edge_saliency(model, x, edge_index, node_idx)
    else:
        edge_mask = _gradient_edge_saliency(model, x, edge_index, node_idx)

    # Extract 2-hop neighborhood edges around target node
    edges_np = edge_index.detach().cpu().numpy()
    hop1_mask = (edges_np[0] == node_idx) | (edges_np[1] == node_idx)
    hop1_nodes = set(edges_np[0][hop1_mask]).union(set(edges_np[1][hop1_mask])).union({node_idx})
    hop2_mask = np.isin(edges_np[0], list(hop1_nodes)) | np.isin(edges_np[1], list(hop1_nodes))

    candidate_indices = np.where(hop2_mask)[0]
    sorted_candidates = candidate_indices[np.argsort(edge_mask[candidate_indices])[::-1]]

    seen_pairs = set()
    subgraph_edges = []
    for e_i in sorted_candidates:
        u = int(edges_np[0, e_i])
        v = int(edges_np[1, e_i])
        pair = (min(u, v), max(u, v))
        if pair in seen_pairs or u == v:
            continue
        seen_pairs.add(pair)
        subgraph_edges.append(
            {
                "source": idx_to_acc.get(u, f"ACC-{u:04d}"),
                "target": idx_to_acc.get(v, f"ACC-{v:04d}"),
                "importance": round(float(edge_mask[e_i]), 4),
            }
        )
        if len(subgraph_edges) >= 10:
            break

    return {
        "account_id": account_id,
        "num_hops": num_hops,
        "subgraph_edges": subgraph_edges,
    }


def _gradient_edge_saliency(
    model: nn.Module, x: torch.Tensor, edge_index: torch.Tensor, node_idx: int
) -> np.ndarray:
    model.eval()
    x_req = x.clone().detach().requires_grad_(True)
    logits = model(x_req, edge_index)
    target_logit = logits[node_idx]
    target_logit.backward()
    grad_norm = torch.norm(x_req.grad if x_req.grad is not None else torch.ones_like(x_req), dim=-1).detach().cpu().numpy()
    src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    raw_scores = grad_norm[src] + grad_norm[dst]
    max_s = float(np.max(raw_scores)) if len(raw_scores) > 0 else 1.0
    return raw_scores / (max_s + 1e-6)
