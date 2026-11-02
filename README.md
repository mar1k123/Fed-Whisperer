# Fraud Ring Finder (`fraud-ring-finder`)

> **Graph Machine Learning & Forensic Network Workbench for Detecting Organized Fraud Rings in Financial Transaction Networks**
>
> **Stack:** Python 3.10+ · PyTorch · PyTorch Geometric (`GraphSAGE`, `GAT`, `GNNExplainer`) · NetworkX · scikit-learn · XGBoost · SHAP · Streamlit · PyVis · Desktop/Web Forensic UI

---

## 1. Executive Summary & Architecture

Organized financial fraud rarely acts in isolation. While point-in-time anomaly detectors catch simple card-not-present velocity spikes, coordinated **fraud rings** distribute activity across dozens of synthetic or compromised accounts, shared hardware fingerprints, collusive merchants, and rapid multi-hop layering paths.

`fraud-ring-finder` is an end-to-end Graph Machine Learning codebase and interactive forensic desktop application that benchmarks **Tabular XGBoost**, **Graph-Feature-Enhanced XGBoost**, **Inductive GraphSAGE**, and **Multi-Head Graph Attention Networks (GAT)** under strict chronological splits with zero future leakage.

```text
┌───────────────────────────────────────────────────────────────────────────────────┐
│                    1. SEEDED SYNTHETIC TRANSACTION GENERATOR                      │
│  Accounts (KYC, Tenure, Balance) · Devices (ASN/VPN Risk) · Merchants (MCC/CB)    │
│  Planted Topologies (~0.5% - 2.0% Fraud Rate):                                    │
│    • Money-Mule Chains (A1 → A2 → A3 → A4 rapid layering with 2–4% fee decay)     │
│    • Shared-Device Rings (Multi-account star/clique sharing 1–2 device IDs)       │
│    • Burst Activity (Dormant account rapid high-velocity outbound drain)          │
│    • Circular Transfers (Directed wash/structuring cycles A1 → A2 → A3 → A4 → A1) │
└─────────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│             2. STRICT TEMPORAL SPLIT & LEAK-FREE FEATURE EXTRACTION               │
│  Train [0%, 70%]  │  Validation (70%, 85%]  │  Test (85%, 100%]                   │
│  In-degree, Out-degree, Weighted PageRank, Shared-Device Count, Flow Pass-Through │
│  and 2-Hop Cycle Motifs computed strictly using edges with t <= t_cutoff.         │
└───────────────────┬───────────────────────────────────────────┬───────────────────┘
                    ▼                                           ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│        3A. GRADIENT BOOSTING          │   │     3B. PYTORCH GEOMETRIC GNNs        │
│ • Baseline 1: XGBoost (Tabular Only)  │   │ • GraphSAGE (Inductive Mean/Skip)     │
│ • Baseline 2: XGBoost + Graph Motifs  │   │ • GAT (Multi-Head Edge Attention)     │
│ • Imbalance: scale_pos_weight         │   │ • Imbalance: Focal Loss & Weighted BCE│
└───────────────────┬───────────────────┘   └───────────────────┬───────────────────┘
                    ▼                                           ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│           4. EVALUATION, ABLATION, ROBUSTNESS & DUAL EXPLAINABILITY               │
│ • Metrics: PR-AUC, Recall @ Fixed Precision (P >= 70%, 80%, 90%)                  │
│ • Pattern Ablation: Which model catches which topological fraud ring?             │
│ • Distribution Shift: Post-training tactic shift (6-hop chains, device rotation)  │
│ • Explainability: TreeSHAP per-account waterfall + GNNExplainer 2-hop subgraphs   │
└───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Why This Project Uses Synthetic Data — And What That Implies

### Why Synthetic Data Is Necessary
1. **Regulatory & PII Constraints (GLBA / GDPR / PCI-DSS):** Real bank wire, ACH, P2P, and device-fingerprint graphs contain sensitive personally identifiable information and proprietary institutional topology that cannot be redistributed in public repositories.
2. **No Ground-Truth Ring Labels in Public Datasets:** Publicly available datasets (such as IEEE-CIS or anonymized credit-card tabular logs) either strip away counterparty graph identifiers (`src_account`, `dst_account`, `device_id`) or lack ground-truth annotations distinguishing *which structural fraud pattern* a flagged node belongs to.
3. **Controlled Ablation & Distribution Shift Testing:** A parameterized generator allows us to plant known topological motifs (`money_mule_chain`, `shared_device_ring`, `burst_activity`, `circular_transfer`) at exact prevalence rates ($0.5\%\text{--}2.0\%$) and systematically alter fraudster tactics after training to measure model degradation under distribution shift.

### What Synthetic Data Implies (Limitations to Keep in Mind)
- **Absence of Label Delay & Chargeback Censorship:** In production banking, fraud labels arrive 30–90 days late via customer disputes or SAR filings, and undetected fraud remains mislabeled as legitimate ("PU learning"). Our synthetic generator provides oracle ground-truth labels at the split boundary.
- **Cleaner Motif Signal-to-Noise Ratio (SNR):** Real-world P2P and corporate treasury graphs contain legitimate high-velocity payroll hubs, family shared devices, and marketplace escrow accounts that mimic mule chains or shared-device rings. Models trained on synthetic data will achieve higher precision at a given recall than on noisy production graphs unless those benign confounders are explicitly modeled.
- **Stationary Background Graph:** Aside from our explicit post-training distribution shift test, background legitimate behavior follows stationary log-normal and diurnal distributions without macroeconomic seasonality (e.g., Black Friday spikes).

---

## 3. How to Plug In Real Transaction Data

You can replace the synthetic generator with your own production or historical transaction tables without changing any downstream feature engineering, XGBoost, GraphSAGE, GAT, or explainability code.

### Step 1: Prepare Three DataFrames (or Parquet/CSV files) Matching the Canonical Schema

| Table | Required Columns | Description |
| :--- | :--- | :--- |
| **`accounts_df`** | `account_id` (str), `account_age_days` (int), `kyc_level` (int), `initial_balance` (float), `primary_device_id` (str), `is_fraud` (0/1), `fraud_pattern` (str), `ring_id` (str) | One row per unique account. Set `fraud_pattern="unknown_fraud"` and `ring_id=""` if ring sub-types are unknown. |
| **`devices_df`** | `device_id` (str), `os_family` (str), `is_vpn_or_proxy` (0/1), `ip_risk_score` (float in `[0, 1]`) | One row per hardware fingerprint or session device token. |
| **`merchants_df`** | `merchant_id` (str), `category` (str), `chargeback_rate` (float) | One row per merchant counterparty (can be empty if pure P2P network). |
| **`transactions_df`** | `tx_id` (str), `src_account` (str), `dst_account` (str), `merchant_id` (str), `device_id` (str), `amount` (float), `timestamp` (float, Unix epoch seconds), `channel` (str), `is_fraud` (0/1) | Directed timestamped transfers sorted chronologically. |

### Step 2: Construct `SyntheticTransactionNetwork` from Real Tables

```python
import pandas as pd
from src.fraud_ring_finder.data.generator import SyntheticTransactionNetwork, FraudNetworkGenerator
from src.fraud_ring_finder.features.graph_features import build_temporal_splits

accounts_df = pd.read_parquet("data/real_accounts.parquet")
devices_df = pd.read_parquet("data/real_devices.parquet")
merchants_df = pd.read_parquet("data/real_merchants.parquet")
transactions_df = pd.read_parquet("data/real_transactions.parquet")

graph = FraudNetworkGenerator._build_networkx_graph(
    accounts_df, devices_df, merchants_df, transactions_df
)

real_network = SyntheticTransactionNetwork(
    accounts_df=accounts_df,
    devices_df=devices_df,
    merchants_df=merchants_df,
    transactions_df=transactions_df,
    rings_metadata=[],
    graph=graph,
    seed=0,
)

# Run the exact same leak-free temporal split, feature builder, and GNN/XGBoost pipeline:
splits = build_temporal_splits(real_network, train_ratio=0.70, val_ratio=0.15)
```

---

## 4. Zero Invented Numbers Policy — Reproducing All Benchmark Results

**No benchmark numbers are hardcoded or invented in this README.** Every metric (PR-AUC, ROC-AUC, Recall at Fixed Precision $P \ge 70\%, 80\%, 90\%$, per-pattern recall ablation, and post-training distribution shift degradation) is computed dynamically from the seeded data and models.

### Generate Results via CLI Script

Run the reproducible evaluation pipeline to generate `artifacts/latest_metrics.json` and print the full benchmark report to stdout:

```bash
python -m src.fraud_ring_finder.pipeline --config configs/config.yaml
```

### Run Automated Determinism & Zero-Leakage Tests

```bash
pytest tests/test_fraud_ring_finder.py -v
```

The test suite verifies:
1. **Bitwise Generator Determinism:** `FraudNetworkGenerator(cfg, seed=101)` produces identical node, device, and transaction DataFrames across repeated invocations, and differs for `seed=202`.
2. **Zero Temporal Leakage:** Perturbing future test-window transactions (`timestamp > t_train_cutoff`) by setting amounts to `$999,999` and rewiring edges has **exact `0.0` difference** on training/validation tabular and graph features (`pd.testing.assert_frame_equal`).
3. **Imbalance Loss & Model Convergence:** `FocalLoss` ($\alpha=0.85, \gamma=2.0$) and `WeightedBCELoss` produce valid gradients and probability predictions in $[0, 1]$.

---

## 5. Launching the Interactive Application (PC Desktop / Streamlit / Web)

### Option A: Native PC Desktop Mode (Frameless App Window)
```bash
python desktop_app.py
# Or via Electron:
npm run desktop
```

### Option B: Interactive Web Workstation (Port 3000)
```bash
npm install
npm run dev
```

### Option C: Streamlit + PyVis Workbench
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 6. Repository Structure

```text
.
├── configs/
│   └── config.yaml                                  # Seed, fraud rate (0.5%-2%), motif weights, GNN & XGBoost params
├── src/
│   └── fraud_ring_finder/
│       ├── __init__.py                              # Package metadata
│       ├── config.py                                # Typed YAML configuration dataclasses
│       ├── data/
│       │   └── generator.py                         # Seeded generator + 4 fraud patterns + distribution shift
│       ├── features/
│       │   └── graph_features.py                    # Leak-free temporal tabular & graph topology features
│       ├── models/
│       │   ├── baselines.py                         # Tabular XGBoost & Graph-Enhanced XGBoost
│       │   └── gnn.py                               # PyG GraphSAGE, GAT, FocalLoss, WeightedBCELoss
│       ├── evaluation/
│       │   └── metrics.py                           # PR-AUC, Recall@Precision, Pattern Ablation, Leakage Verifier
│       ├── explainability/
│       │   └── explainer.py                         # TreeSHAP attributions & GNNExplainer 2-hop subgraphs
│       └── pipeline.py                              # End-to-end CLI runner exporting artifacts/latest_metrics.json
├── tests/
│   └── test_fraud_ring_finder.py                    # Determinism, fraud rate bounds, and zero-leakage tests
├── app.py                                           # Streamlit + PyVis interactive dashboard
├── desktop_app.py                                   # Native PC desktop window launcher (PyWebView / Chromium --app)
├── desktop/
│   └── main.cjs                                     # Electron desktop wrapper
├── requirements.txt                                 # Python 3.10+ dependencies
└── README.md                                        # Documentation, synthetic data analysis & real-data guide
```
