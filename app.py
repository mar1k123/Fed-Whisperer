"""
Streamlit + PyVis Interactive Workbench for `fraud-ring-finder`.

Features:
- Interactive PyVis transaction network view with flagged fraud rings highlighted by pattern topology
- Per-account risk card displaying GNN probability, XGBoost score, TreeSHAP feature attributions,
  and GNNExplainer 2-hop computational subgraph edges
- Live time-split benchmark evaluation, pattern ablation matrix, and post-training distribution shift test

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    from pyvis.network import Network
except ImportError:
    Network = None  # type: ignore[assignment]

from src.fraud_ring_finder.config import load_config
from src.fraud_ring_finder.data.generator import FraudNetworkGenerator, SyntheticTransactionNetwork
from src.fraud_ring_finder.features.graph_features import build_temporal_splits
from src.fraud_ring_finder.models.baselines import FraudXGBoostModel
from src.fraud_ring_finder.models.gnn import train_gnn_Classifier, predict_gnn_proba, build_pyg_graph
from src.fraud_ring_finder.evaluation.metrics import (
    verify_temporal_split_no_leakage,
    evaluate_model_predictions,
)
from src.fraud_ring_finder.explainability.explainer import (
    compute_xgboost_shap,
    explain_gnn_node,
)

PATTERN_COLORS = {
    "legitimate": "#64748b",
    "money_mule_chain": "#e11d48",
    "shared_device_ring": "#d97706",
    "burst_activity": "#7c3aed",
    "circular_transfer": "#0284c7",
}


st.set_page_config(
    page_title="Fraud Ring Finder — Graph ML Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Generating seeded transaction network & training GraphSAGE / GAT / XGBoost...")
def run_cached_experiment(seed: int, fraud_rate: float, shifted: bool):
    cfg = load_config("configs/config.yaml")
    cfg.seed = seed
    cfg.generator.fraud_rate = fraud_rate

    gen = FraudNetworkGenerator(cfg.generator, seed=seed)
    net = gen.generate(shifted_tactics=shifted)
    splits = build_temporal_splits(net, cfg.split.train_ratio, cfg.split.val_ratio)
    leakage = verify_temporal_split_no_leakage(splits)

    train_feat = splits["train"]["features"]
    train_tx = splits["train"]["tx"]
    y_train = splits["train"]["y"]

    test_feat = splits["test"]["features"]
    test_tx = splits["test"]["tx"]
    y_test = splits["test"]["y"]
    pat_test = splits["test"]["patterns"]

    xgb_tab = FraudXGBoostModel(cfg.xgboost, include_graph_features=False, seed=seed).fit(train_feat, y_train)
    xgb_graph = FraudXGBoostModel(cfg.xgboost, include_graph_features=True, seed=seed).fit(train_feat, y_train)
    sage_model, sage_norm = train_gnn_Classifier("graphsage", cfg.graphsage, train_feat, train_tx, y_train, seed=seed)
    gat_model, gat_norm = train_gnn_Classifier("gat", cfg.gat, train_feat, train_tx, y_train, seed=seed)

    scores = {
        "XGBoost (Tabular)": xgb_tab.predict_proba(test_feat),
        "XGBoost + Graph Features": xgb_graph.predict_proba(test_feat),
        "GraphSAGE (Focal Loss)": predict_gnn_proba(sage_model, sage_norm, test_feat, test_tx),
        "GAT (Multi-Head Focal)": predict_gnn_proba(gat_model, gat_norm, test_feat, test_tx),
    }

    metrics = {
        name: evaluate_model_predictions(y_test, sc, pat_test)
        for name, sc in scores.items()
    }

    return net, splits, leakage, xgb_graph, sage_model, sage_norm, gat_model, gat_norm, scores, metrics


def render_pyvis_ring_subgraph(net: SyntheticTransactionNetwork, selected_ring_id: str) -> str:
    """Build an interactive PyVis HTML network highlighting the selected fraud ring and its 1-hop context."""
    tx_df = net.transactions_df
    ring_tx = tx_df[tx_df["ring_id"] == selected_ring_id]
    ring_accounts = set(ring_tx["src_account"]).union(set(ring_tx["dst_account"]))

    context_tx = tx_df[
        tx_df["src_account"].isin(ring_accounts) | tx_df["dst_account"].isin(ring_accounts)
    ].head(65)

    if Network is None:
        return "<div style='padding:1rem;font-family:monospace;'>Install pyvis to render interactive HTML canvas.</div>"

    nt = Network(height="520px", width="100%", directed=True, bgcolor="#0f172a", font_color="#f8fafc")
    nt.barnes_hut(gravity=-2800, central_gravity=0.3, spring_length=130)

    acc_lookup = net.accounts_df.set_index("account_id").to_dict(orient="index")
    added_nodes = set()

    for row in context_tx.itertuples(index=False):
        for acc_id in (str(row.src_account), str(row.dst_account)):
            if acc_id not in added_nodes:
                meta = acc_lookup.get(acc_id, {})
                pat = str(meta.get("fraud_pattern", "legitimate"))
                is_fraud = int(meta.get("is_fraud", 0)) == 1
                nt.add_node(
                    acc_id,
                    label=acc_id,
                    title=f"{acc_id} | Pattern: {pat} | Ring: {meta.get('ring_id', 'None')}",
                    color=PATTERN_COLORS.get(pat, "#e11d48") if is_fraud else "#475569",
                    size=26 if acc_id in ring_accounts else 15,
                )
                added_nodes.add(acc_id)

        edge_is_fraud = int(row.is_fraud) == 1
        nt.add_edge(
            str(row.src_account),
            str(row.dst_account),
            title=f"{row.tx_id}: ${row.amount:,.2f} ({row.channel})",
            color="#f43f5e" if edge_is_fraud else "#334155",
            width=3.2 if edge_is_fraud else 1.2,
        )

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        nt.save_graph(tmp.name)
        return Path(tmp.name).read_text(encoding="utf-8")


def main() -> None:
    st.title("Fraud Ring Finder — Graph ML Workbench")
    st.caption(
        "Synthetic Transaction Network Generator · XGBoost vs. GraphSAGE & GAT · "
        "Temporal Split Verification · SHAP & GNNExplainer Subgraphs"
    )

    with st.sidebar:
        st.header("Generator & Split Controls")
        seed = int(st.number_input("Random Seed", min_value=1, max_value=99999, value=42, step=1))
        fraud_rate = float(
            st.slider("Planted Fraud Rate", min_value=0.005, max_value=0.020, value=0.015, step=0.001, format="%.3f")
        )
        shifted = bool(st.checkbox("Apply Post-Training Distribution Shift", value=False))

    net, splits, leakage, xgb_graph, sage_model, _, _, _, scores, metrics = run_cached_experiment(
        seed, fraud_rate, shifted
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accounts / Devices", f"{len(net.accounts_df):,} / {len(net.devices_df):,}")
    col2.metric("Transactions", f"{len(net.transactions_df):,}")
    col3.metric("Planted Fraud Rings", f"{len(net.rings_metadata)}")
    col4.metric("Temporal Leakage Check", "PASSED (0 overlap)" if leakage["passed"] else "FAILED")

    tab_graph, tab_eval, tab_explain = st.tabs(
        ["Interactive Ring Graph & Risk Card", "PR-AUC & Pattern Ablation", "SHAP & GNNExplainer"]
    )

    with tab_graph:
        left_col, right_col = st.columns([7, 5])
        ring_options = [str(r["ring_id"]) for r in net.rings_metadata]
        selected_ring = left_col.selectbox("Select Flagged Fraud Ring", options=ring_options, index=0)
        ring_info = next((r for r in net.rings_metadata if r["ring_id"] == selected_ring), net.rings_metadata[0])

        with left_col:
            st.subheader(f"PyVis Subgraph: {selected_ring} ({ring_info['pattern']})")
            html_content = render_pyvis_ring_subgraph(net, selected_ring)
            components.html(html_content, height=540, scrolling=False)

        with right_col:
            st.subheader("Per-Account Risk Card & Explanation")
            ring_accs = list(ring_info["accounts"])
            selected_acc = st.selectbox("Inspect Account in Ring", options=ring_accs, index=0)

            test_feat = splits["test"]["features"]
            test_tx = splits["test"]["tx"]
            acc_idx_list = test_feat.index[test_feat["account_id"] == selected_acc].tolist()
            acc_idx = acc_idx_list[0] if acc_idx_list else 0

            gat_prob = float(scores["GAT (Multi-Head Focal)"][acc_idx])
            sage_prob = float(scores["GraphSAGE (Focal Loss)"][acc_idx])
            xgb_prob = float(scores["XGBoost + Graph Features"][acc_idx])

            st.markdown(
                f"**Account:** `{selected_acc}` · **Pattern:** `{ring_info['pattern']}` · **Ring:** `{selected_ring}`"
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("GAT Risk Score", f"{gat_prob:.1%}")
            m2.metric("GraphSAGE Risk", f"{sage_prob:.1%}")
            m3.metric("XGB+Graph Risk", f"{xgb_prob:.1%}")

            shap_res = compute_xgboost_shap(xgb_graph, test_feat, selected_acc, top_k=6)
            st.markdown("#### Top SHAP Feature Contributions (XGBoost + Graph)")
            st.dataframe(pd.DataFrame(shap_res["attributions"]), use_container_width=True)

            x_t, edge_idx, _, acc_to_idx = build_pyg_graph(test_feat, test_tx, None)
            idx_to_acc = {v: k for k, v in acc_to_idx.items()}
            gnn_exp = explain_gnn_node(sage_model, x_t, edge_idx, selected_acc, acc_to_idx, idx_to_acc)
            st.markdown("#### GNNExplainer Top-Attributed 2-Hop Edges")
            st.dataframe(pd.DataFrame(gnn_exp["subgraph_edges"]), use_container_width=True)

    with tab_eval:
        st.subheader("Temporal Test Split Performance (No Future Leakage)")
        rows = []
        for model_name, m in metrics.items():
            rows.append(
                {
                    "Model": model_name,
                    "PR-AUC": m["pr_auc"],
                    "ROC-AUC": m["roc_auc"],
                    "Recall @ P>=70%": m["recall_at_precision"]["recall@p=70"],
                    "Recall @ P>=80%": m["recall_at_precision"]["recall@p=80"],
                    "Recall @ P>=90%": m["recall_at_precision"]["recall@p=90"],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.subheader("Ablation: Fraud Pattern Recall @ Precision >= 80%")
        ablation_rows = []
        for model_name, m in metrics.items():
            pat_rec = m["pattern_ablation_recall_at_p80"]
            ablation_rows.append({"Model": model_name, **pat_rec})
        st.dataframe(pd.DataFrame(ablation_rows), use_container_width=True)

    with tab_explain:
        st.subheader("Explainability Methodology: TreeSHAP vs. GNNExplainer")
        st.write(
            "TreeSHAP quantifies marginal tabular + hand-engineered feature contributions per account, "
            "while GNNExplainer optimizes a continuous edge mask over the 2-hop message-passing computational "
            "subgraph to reveal multi-hop mule chains, shared hardware devices, and circular transfer loops."
        )


if __name__ == "__main__":
    main()
