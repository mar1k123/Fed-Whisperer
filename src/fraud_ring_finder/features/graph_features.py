"""
Leak-Free Temporal Feature Engineering for Tabular and Graph Baselines.

Computes two strictly separated feature sets for each account up to a given `cutoff_timestamp`:
1. Tabular Per-Account/Per-Transaction Features (`TABULAR_FEATURE_COLUMNS`):
   - Account age, KYC tier, initial balance
   - Outbound & inbound transaction counts, mean/max/std amount
   - Burst velocity ratio (max 1-hour tx count / daily average)
   - Wire/P2P high-risk channel ratio, merchant chargeback exposure
2. Hand-Crafted Graph Topology Features (`GRAPH_FEATURE_COLUMNS`):
   - In-degree and out-degree on the transaction graph up to `cutoff_timestamp`
   - Weighted PageRank score (`nx.pagerank`)
   - Shared-device degree (number of distinct accounts authenticating on the same device IDs)
   - 2-hop relay ratio (flow pass-through ratio min(in_vol, out_vol) / max(in_vol, out_vol))
   - Directed cycle / reciprocity indicator (fraction of counterparties with bidirectional transfers or 3/4-cycles)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx

from ..data.generator import SyntheticTransactionNetwork

TABULAR_FEATURE_COLUMNS: List[str] = [
    "account_age_days",
    "kyc_level",
    "log_initial_balance",
    "tx_out_count",
    "tx_in_count",
    "amount_out_mean",
    "amount_out_max",
    "amount_out_std",
    "burst_1h_max_count",
    "high_risk_channel_ratio",
    "device_ip_risk_score",
    "merchant_cb_exposure",
]

GRAPH_FEATURE_COLUMNS: List[str] = [
    "graph_in_degree",
    "graph_out_degree",
    "graph_pagerank",
    "shared_device_account_count",
    "flow_passthrough_ratio",
    "cycle_triangle_count",
    "two_hop_neighbor_count",
]

ALL_FEATURE_COLUMNS: List[str] = TABULAR_FEATURE_COLUMNS + GRAPH_FEATURE_COLUMNS


def extract_account_features_at_cutoff(
    network: SyntheticTransactionNetwork,
    cutoff_timestamp: Optional[float] = None,
    min_timestamp: float = 0.0,
    pagerank_alpha: float = 0.85,
) -> pd.DataFrame:
    """
    Compute tabular and hand-made graph features using ONLY transactions with
    `min_timestamp <= timestamp <= cutoff_timestamp` to guarantee zero temporal leakage.
    """
    accounts = network.accounts_df.copy()
    devices_map = network.devices_df.set_index("device_id")["ip_risk_score"].to_dict()
    merchants_map = network.merchants_df.set_index("merchant_id")["chargeback_rate"].to_dict()

    tx_df = network.transactions_df
    if cutoff_timestamp is not None:
        tx_df = tx_df[(tx_df["timestamp"] >= min_timestamp) & (tx_df["timestamp"] <= cutoff_timestamp)].copy()
    else:
        tx_df = tx_df[tx_df["timestamp"] >= min_timestamp].copy()

    acc_ids = accounts["account_id"].tolist()
    n_acc = len(acc_ids)
    acc_to_idx = {a: i for i, a in enumerate(acc_ids)}

    tx_out_count = np.zeros(n_acc, dtype=float)
    tx_in_count = np.zeros(n_acc, dtype=float)
    amount_out_sum = np.zeros(n_acc, dtype=float)
    amount_in_sum = np.zeros(n_acc, dtype=float)
    amount_out_max = np.zeros(n_acc, dtype=float)
    amount_out_sq_sum = np.zeros(n_acc, dtype=float)
    high_risk_ch_count = np.zeros(n_acc, dtype=float)
    merchant_cb_sum = np.zeros(n_acc, dtype=float)
    burst_1h_max = np.zeros(n_acc, dtype=float)

    # Device sharing map observed strictly within the allowed time window
    device_to_accounts: Dict[str, set] = {}
    for row in accounts.itertuples(index=False):
        device_to_accounts.setdefault(str(row.primary_device_id), set()).add(str(row.account_id))

    out_timestamps_by_acc: Dict[int, List[float]] = {i: [] for i in range(n_acc)}
    G_simple = nx.DiGraph()
    G_simple.add_nodes_from(acc_ids)

    for row in tx_df.itertuples(index=False):
        s_idx = acc_to_idx.get(str(row.src_account))
        d_idx = acc_to_idx.get(str(row.dst_account))
        amt = float(row.amount)
        ts = float(row.timestamp)
        dev_id = str(row.device_id)

        if dev_id:
            device_to_accounts.setdefault(dev_id, set()).add(str(row.src_account))

        if s_idx is not None:
            tx_out_count[s_idx] += 1.0
            amount_out_sum[s_idx] += amt
            amount_out_sq_sum[s_idx] += amt * amt
            if amt > amount_out_max[s_idx]:
                amount_out_max[s_idx] = amt
            if str(row.channel) in ("wire", "p2p"):
                high_risk_ch_count[s_idx] += 1.0
            if row.merchant_id and str(row.merchant_id) in merchants_map:
                merchant_cb_sum[s_idx] += float(merchants_map[str(row.merchant_id)])
            out_timestamps_by_acc[s_idx].append(ts)

        if d_idx is not None:
            tx_in_count[d_idx] += 1.0
            amount_in_sum[d_idx] += amt

        if s_idx is not None and d_idx is not None:
            if G_simple.has_edge(row.src_account, row.dst_account):
                G_simple[row.src_account][row.dst_account]["weight"] += amt
                G_simple[row.src_account][row.dst_account]["count"] += 1
            else:
                G_simple.add_edge(row.src_account, row.dst_account, weight=amt, count=1)

    # Compute 1-hour sliding window burstiness per account
    for idx, ts_list in out_timestamps_by_acc.items():
        if not ts_list:
            continue
        ts_arr = np.sort(np.asarray(ts_list, dtype=float))
        left = 0
        max_w = 1
        for right in range(len(ts_arr)):
            while ts_arr[right] - ts_arr[left] > 3600.0:
                left += 1
            max_w = max(max_w, right - left + 1)
        burst_1h_max[idx] = float(max_w)

    # Graph features on G_simple
    in_deg = np.array([G_simple.in_degree(a) for a in acc_ids], dtype=float)
    out_deg = np.array([G_simple.out_degree(a) for a in acc_ids], dtype=float)

    try:
        pr_dict = nx.pagerank(G_simple, alpha=pagerank_alpha, weight="weight", max_iter=100)
    except Exception:
        pr_dict = {a: 1.0 / max(1, n_acc) for a in acc_ids}
    pagerank_arr = np.array([float(pr_dict.get(a, 0.0)) * n_acc for a in acc_ids], dtype=float)

    # Shared device count per account
    shared_dev_counts = np.ones(n_acc, dtype=float)
    for i, row in enumerate(accounts.itertuples(index=False)):
        dev_users = set(device_to_accounts.get(str(row.primary_device_id), set()))
        shared_dev_counts[i] = float(max(1, len(dev_users)))

    # Flow pass-through ratio (characteristic of money-mule relays)
    max_vol = np.maximum(amount_in_sum, amount_out_sum)
    min_vol = np.minimum(amount_in_sum, amount_out_sum)
    flow_passthrough = np.where(max_vol > 50.0, min_vol / (max_vol + 1e-6), 0.0)

    # Directed cycle / 2-hop neighborhood metrics
    cycle_counts = np.zeros(n_acc, dtype=float)
    two_hop_counts = np.zeros(n_acc, dtype=float)
    for i, acc in enumerate(acc_ids):
        succs = set(G_simple.successors(acc))
        preds = set(G_simple.predecessors(acc))
        two_hop = set()
        cyc = 0
        for s in succs:
            s_succs = set(G_simple.successors(s))
            two_hop.update(s_succs)
            # Check 2-cycle (reciprocity), 3-cycle, or 4-cycle participation
            if s in preds:
                cyc += 1
            cyc += len(s_succs.intersection(preds))
            for s2 in s_succs:
                if acc in G_simple.successors(s2):
                    cyc += 1
        cycle_counts[i] = float(cyc)
        two_hop_counts[i] = float(len(two_hop))

    denom_out = np.maximum(1.0, tx_out_count)
    mean_out = amount_out_sum / denom_out
    var_out = np.maximum(0.0, (amount_out_sq_sum / denom_out) - (mean_out ** 2))
    std_out = np.sqrt(var_out)

    features_df = pd.DataFrame(
        {
            "account_id": acc_ids,
            "account_age_days": accounts["account_age_days"].astype(float).values,
            "kyc_level": accounts["kyc_level"].astype(float).values,
            "log_initial_balance": np.log1p(accounts["initial_balance"].astype(float).values),
            "tx_out_count": tx_out_count,
            "tx_in_count": tx_in_count,
            "amount_out_mean": np.round(mean_out, 4),
            "amount_out_max": np.round(amount_out_max, 4),
            "amount_out_std": np.round(std_out, 4),
            "burst_1h_max_count": burst_1h_max,
            "high_risk_channel_ratio": np.round(high_risk_ch_count / denom_out, 4),
            "device_ip_risk_score": np.array(
                [float(devices_map.get(str(d), 0.05)) for d in accounts["primary_device_id"]], dtype=float
            ),
            "merchant_cb_exposure": np.round(merchant_cb_sum / denom_out, 4),
            "graph_in_degree": in_deg,
            "graph_out_degree": out_deg,
            "graph_pagerank": np.round(pagerank_arr, 5),
            "shared_device_account_count": shared_dev_counts,
            "flow_passthrough_ratio": np.round(flow_passthrough, 4),
            "cycle_triangle_count": cycle_counts,
            "two_hop_neighbor_count": two_hop_counts,
        }
    )

    return features_df


def build_temporal_splits(
    network: SyntheticTransactionNetwork,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Dict[str, object]:
    """
    Split transactions strictly by timestamp into Train, Validation, and Test windows.
    Node features for each split are extracted using ONLY transactions up to that split's cutoff timestamp.
    """
    tx_sorted = network.transactions_df.sort_values("timestamp").reset_index(drop=True)
    n_total = len(tx_sorted)
    train_end_idx = int(n_total * train_ratio)
    val_end_idx = int(n_total * (train_ratio + val_ratio))

    t_train_cutoff = float(tx_sorted.loc[train_end_idx - 1, "timestamp"])
    t_val_cutoff = float(tx_sorted.loc[val_end_idx - 1, "timestamp"])
    t_test_cutoff = float(tx_sorted.loc[n_total - 1, "timestamp"])

    train_tx = tx_sorted[tx_sorted["timestamp"] <= t_train_cutoff].copy()
    val_tx = tx_sorted[(tx_sorted["timestamp"] > t_train_cutoff) & (tx_sorted["timestamp"] <= t_val_cutoff)].copy()
    test_tx = tx_sorted[tx_sorted["timestamp"] > t_val_cutoff].copy()

    # Features computed strictly up to each window's temporal boundary (zero future leakage)
    feat_train = extract_account_features_at_cutoff(network, cutoff_timestamp=t_train_cutoff)
    feat_val = extract_account_features_at_cutoff(network, cutoff_timestamp=t_val_cutoff)
    feat_test = extract_account_features_at_cutoff(network, cutoff_timestamp=t_test_cutoff)

    # Active fraud labels within each cumulative window
    acc_ids = network.accounts_df["account_id"].tolist()
    acc_pattern_map = network.accounts_df.set_index("account_id")["fraud_pattern"].to_dict()

    def _window_labels(window_tx: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        fraud_tx = window_tx[window_tx["is_fraud"] == 1]
        active_fraud_accs = set(fraud_tx["src_account"]).union(set(fraud_tx["dst_account"]))
        y = np.array([1 if a in active_fraud_accs else 0 for a in acc_ids], dtype=int)
        patterns = [str(acc_pattern_map[a]) if y[i] == 1 else "legitimate" for i, a in enumerate(acc_ids)]
        return y, patterns

    y_train, pat_train = _window_labels(train_tx)
    y_val, pat_val = _window_labels(val_tx)
    y_test, pat_test = _window_labels(test_tx)

    return {
        "cutoffs": {
            "train_max_ts": t_train_cutoff,
            "val_min_ts": float(val_tx["timestamp"].min()),
            "val_max_ts": t_val_cutoff,
            "test_min_ts": float(test_tx["timestamp"].min()),
            "test_max_ts": t_test_cutoff,
        },
        "train": {"features": feat_train, "y": y_train, "patterns": pat_train, "tx": train_tx},
        "val": {"features": feat_val, "y": y_val, "patterns": pat_val, "tx": val_tx},
        "test": {"features": feat_test, "y": y_test, "patterns": pat_test, "tx": test_tx},
    }
