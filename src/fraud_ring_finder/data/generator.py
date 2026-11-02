"""
Synthetic Transaction Network Generator with Planted Fraud Ring Topologies.

Generates realistic multi-entity financial networks consisting of:
- Accounts (log-normal baseline balances, account age, KYC verification tier, jurisdiction risk)
- Devices (hardware fingerprints, IP ASN risk, OS family)
- Merchants (MCC category, baseline chargeback rate)
- Transactions (timestamped directed transfers with diurnal activity cycles and log-normal amounts)

Plants four configurable structural fraud patterns (~0.5% to 2.0% overall fraud prevalence):
1. `money_mule_chain`: Rapid multi-hop layering chain A_1 -> A_2 -> ... -> A_k with ~2-4% fee decay per hop.
2. `shared_device_ring`: Group of collusive accounts sharing 1-2 hardware devices and funneling funds to a mule/merchant.
3. `burst_activity`: Compromised dormant account executing rapid high-velocity outbound drains within < 2 hours.
4. `circular_transfer`: Directed cycle A_1 -> A_2 -> ... -> A_m -> A_1 (round-trip wash/structuring loop).

Also supports generating a post-training distribution-shift window (`shifted_tactics=True`) where
fraudsters elongate mule chains, rotate proxy devices, and split burst transfers below reporting thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
import pandas as pd
import networkx as nx

from ..config import GeneratorConfig

FraudPatternType = Literal[
    "legitimate",
    "money_mule_chain",
    "shared_device_ring",
    "burst_activity",
    "circular_transfer",
]


@dataclass
class SyntheticTransactionNetwork:
    accounts_df: pd.DataFrame
    devices_df: pd.DataFrame
    merchants_df: pd.DataFrame
    transactions_df: pd.DataFrame
    rings_metadata: List[Dict[str, object]]
    graph: nx.MultiDiGraph
    seed: int
    shifted_tactics: bool = False


class FraudNetworkGenerator:
    """Deterministic seeded generator for financial transaction graphs with planted fraud rings."""

    def __init__(self, config: GeneratorConfig, seed: int = 42) -> None:
        if not (0.002 <= config.fraud_rate <= 0.10):
            raise ValueError(f"fraud_rate must be between 0.002 and 0.10, got {config.fraud_rate}")
        self.config = config
        self.seed = seed

    def generate(self, shifted_tactics: bool = False, seed_override: Optional[int] = None) -> SyntheticTransactionNetwork:
        active_seed = self.seed if seed_override is None else seed_override
        rng = np.random.default_rng(active_seed)

        n_acc = self.config.num_accounts
        n_dev = self.config.num_devices
        n_mer = self.config.num_merchants
        n_tx = self.config.num_transactions
        span_seconds = self.config.time_span_days * 86400.0

        # 1. Generate Devices
        device_ids = [f"DEV-{i:04d}" for i in range(n_dev)]
        os_families = rng.choice(["iOS", "Android", "Windows", "macOS", "Linux"], size=n_dev, p=[0.34, 0.33, 0.18, 0.11, 0.04])
        vpn_flags = rng.binomial(1, 0.06, size=n_dev)
        ip_risk = np.clip(rng.beta(1.5, 12.0, size=n_dev) + vpn_flags * 0.35, 0.0, 1.0)
        devices_df = pd.DataFrame(
            {
                "device_id": device_ids,
                "os_family": os_families,
                "is_vpn_or_proxy": vpn_flags,
                "ip_risk_score": np.round(ip_risk, 4),
            }
        )

        # 2. Generate Merchants
        merchant_ids = [f"MER-{i:03d}" for i in range(n_mer)]
        categories = rng.choice(
            ["retail", "utilities", "groceries", "electronics", "crypto_exchange", "wire_remittance", "gaming"],
            size=n_mer,
            p=[0.28, 0.18, 0.22, 0.12, 0.06, 0.07, 0.07],
        )
        base_cb_rate = np.where(
            np.isin(categories, ["crypto_exchange", "wire_remittance", "gaming"]),
            rng.uniform(0.018, 0.045, size=n_mer),
            rng.uniform(0.001, 0.012, size=n_mer),
        )
        merchants_df = pd.DataFrame(
            {
                "merchant_id": merchant_ids,
                "category": categories,
                "chargeback_rate": np.round(base_cb_rate, 4),
            }
        )

        # 3. Generate Accounts
        account_ids = [f"ACC-{i:04d}" for i in range(n_acc)]
        account_age_days = rng.gamma(shape=2.5, scale=180.0, size=n_acc).astype(int) + 7
        kyc_level = rng.choice([1, 2, 3], size=n_acc, p=[0.18, 0.57, 0.25])
        init_balance = np.round(rng.lognormal(mean=7.8, sigma=1.1, size=n_acc), 2)
        primary_devices = rng.choice(device_ids, size=n_acc, replace=True)

        accounts_df = pd.DataFrame(
            {
                "account_id": account_ids,
                "account_age_days": account_age_days,
                "kyc_level": kyc_level,
                "initial_balance": init_balance,
                "primary_device_id": primary_devices,
                "is_fraud": np.zeros(n_acc, dtype=int),
                "fraud_pattern": ["legitimate"] * n_acc,
                "ring_id": [""] * n_acc,
            }
        )

        # 4. Generate Background Legitimate Transactions
        target_fraud_tx = max(24, int(round(n_tx * self.config.fraud_rate)))
        n_legit_tx = max(100, n_tx - target_fraud_tx)

        # Diurnal timestamp sampling over [0, span_seconds]
        raw_days = rng.integers(0, self.config.time_span_days, size=n_legit_tx)
        hour_of_day = rng.normal(loc=14.5, scale=4.2, size=n_legit_tx) % 24.0
        timestamps = np.sort(raw_days * 86400.0 + hour_of_day * 3600.0 + rng.uniform(0, 60, size=n_legit_tx))

        src_indices = rng.integers(0, n_acc, size=n_legit_tx)
        dst_indices = (src_indices + rng.integers(1, n_acc, size=n_legit_tx)) % n_acc
        is_merchant_payment = rng.binomial(1, 0.30, size=n_legit_tx)
        merchant_targets = rng.choice(merchant_ids, size=n_legit_tx)

        amounts = np.round(rng.lognormal(mean=4.15, sigma=0.95, size=n_legit_tx), 2)
        channels = rng.choice(["ach", "card", "p2p", "wire"], size=n_legit_tx, p=[0.35, 0.35, 0.22, 0.08])

        tx_records: List[Dict[str, object]] = []
        for idx in range(n_legit_tx):
            s_idx = int(src_indices[idx])
            d_idx = int(dst_indices[idx])
            tx_records.append(
                {
                    "tx_id": f"TX-{idx:06d}",
                    "src_account": account_ids[s_idx],
                    "dst_account": account_ids[d_idx],
                    "merchant_id": str(merchant_targets[idx]) if is_merchant_payment[idx] else "",
                    "device_id": str(primary_devices[s_idx]),
                    "amount": float(amounts[idx]),
                    "timestamp": float(round(timestamps[idx], 2)),
                    "channel": str(channels[idx]),
                    "is_fraud": 0,
                    "fraud_pattern": "legitimate",
                    "ring_id": "",
                }
            )

        # 5. Plant Fraud Rings across Train, Validation, and Test temporal windows
        # Ensure each temporal split contains instances of all 4 fraud patterns so evaluation is well-defined
        rings_metadata: List[Dict[str, object]] = []
        available_accounts = list(range(n_acc))
        rng.shuffle(available_accounts)
        acc_cursor = 0

        patterns: List[FraudPatternType] = [
            "money_mule_chain",
            "shared_device_ring",
            "burst_activity",
            "circular_transfer",
        ]
        # Temporal windows corresponding to train [0, 0.70], val [0.70, 0.85], test [0.85, 1.0]
        windows: List[Tuple[str, float, float]] = [
            ("train", 0.05 * span_seconds, 0.68 * span_seconds),
            ("val", 0.71 * span_seconds, 0.84 * span_seconds),
            ("test", 0.86 * span_seconds, 0.98 * span_seconds),
        ]

        fraud_tx_created = 0
        ring_counter = 1
        tx_id_counter = n_legit_tx

        while fraud_tx_created < target_fraud_tx and acc_cursor + 10 < n_acc:
            for split_name, w_start, w_end in windows:
                for pattern in patterns:
                    if fraud_tx_created >= target_fraud_tx or acc_cursor + 10 >= n_acc:
                        break

                    ring_id = f"RING-{ring_counter:03d}"
                    ring_counter += 1
                    base_ts = float(rng.uniform(w_start, w_end - 7200.0))

                    new_txs, used_Accs, shared_dev = self._plant_single_ring(
                        rng=rng,
                        pattern=pattern,
                        ring_id=ring_id,
                        base_ts=base_ts,
                        account_pool=available_accounts[acc_cursor : acc_cursor + 8],
                        account_ids=account_ids,
                        primary_devices=primary_devices,
                        device_ids=device_ids,
                        merchant_ids=merchant_ids,
                        tx_id_start=tx_id_counter,
                        shifted_tactics=shifted_tactics and (split_name == "test"),
                    )

                    acc_cursor += len(used_Accs)
                    tx_id_counter += len(new_txs)
                    fraud_tx_created += len(new_txs)
                    tx_records.extend(new_txs)

                    for a_idx in used_Accs:
                        accounts_df.loc[a_idx, "is_fraud"] = 1
                        accounts_df.loc[a_idx, "fraud_pattern"] = pattern
                        accounts_df.loc[a_idx, "ring_id"] = ring_id
                        if shared_dev is not None:
                            accounts_df.loc[a_idx, "primary_device_id"] = shared_dev

                    rings_metadata.append(
                        {
                            "ring_id": ring_id,
                            "pattern": pattern,
                            "split_window": split_name,
                            "accounts": [account_ids[a] for a in used_Accs],
                            "num_transactions": len(new_txs),
                            "total_volume": round(sum(float(t["amount"]) for t in new_txs), 2),
                            "start_timestamp": round(base_ts, 2),
                            "shifted_tactics": bool(shifted_tactics and split_name == "test"),
                        }
                    )

        transactions_df = pd.DataFrame(tx_records).sort_values("timestamp").reset_index(drop=True)
        graph = self._build_networkx_graph(accounts_df, devices_df, merchants_df, transactions_df)

        return SyntheticTransactionNetwork(
            accounts_df=accounts_df,
            devices_df=devices_df,
            merchants_df=merchants_df,
            transactions_df=transactions_df,
            rings_metadata=rings_metadata,
            graph=graph,
            seed=active_seed,
            shifted_tactics=shifted_tactics,
        )

    def _plant_single_ring(
        self,
        rng: np.random.Generator,
        pattern: FraudPatternType,
        ring_id: str,
        base_ts: float,
        account_pool: List[int],
        account_ids: List[str],
        primary_devices: np.ndarray,
        device_ids: List[str],
        merchant_ids: List[str],
        tx_id_start: int,
        shifted_tactics: bool,
    ) -> Tuple[List[Dict[str, object]], List[int], Optional[str]]:
        txs: List[Dict[str, object]] = []
        shared_dev: Optional[str] = None

        if pattern == "money_mule_chain":
            chain_len = self.config.distribution_shift.mule_chain_length_shifted if shifted_tactics else 4
            chain_nodes = account_pool[: min(len(account_pool), chain_len + 1)]
            amount = float(rng.uniform(1800.0, 4600.0))
            hop_delay = (
                self.config.distribution_shift.mule_hop_delay_minutes_shifted * 60.0
                if shifted_tactics
                else float(rng.uniform(180.0, 900.0))
            )
            curr_ts = base_ts
            for hop in range(len(chain_nodes) - 1):
                src_i = chain_nodes[hop]
                dst_i = chain_nodes[hop + 1]
                txs.append(
                    {
                        "tx_id": f"TX-{tx_id_start + len(txs):06d}",
                        "src_account": account_ids[src_i],
                        "dst_account": account_ids[dst_i],
                        "merchant_id": "",
                        "device_id": str(primary_devices[src_i]),
                        "amount": round(amount, 2),
                        "timestamp": round(curr_ts, 2),
                        "channel": "wire" if hop == len(chain_nodes) - 2 else "p2p",
                        "is_fraud": 1,
                        "fraud_pattern": pattern,
                        "ring_id": ring_id,
                    }
                )
                amount *= float(rng.uniform(0.95, 0.985))
                curr_ts += hop_delay
            return txs, chain_nodes, None

        if pattern == "shared_device_ring":
            ring_nodes = account_pool[:5]
            hub_node = ring_nodes[0]
            shared_dev = str(rng.choice(device_ids[: max(10, len(device_ids) // 5)]))
            collusive_merchant = str(rng.choice(merchant_ids[-15:]))
            curr_ts = base_ts
            for idx, member in enumerate(ring_nodes[1:]):
                # Under distribution shift, half the members rotate to a secondary proxy device
                use_dev = (
                    str(rng.choice(device_ids))
                    if (shifted_tactics and idx % 2 == 1)
                    else shared_dev
                )
                amt = float(rng.uniform(950.0, 2400.0))
                txs.append(
                    {
                        "tx_id": f"TX-{tx_id_start + len(txs):06d}",
                        "src_account": account_ids[member],
                        "dst_account": account_ids[hub_node],
                        "merchant_id": collusive_merchant,
                        "device_id": use_dev,
                        "amount": round(amt, 2),
                        "timestamp": round(curr_ts + idx * 240.0, 2),
                        "channel": "card",
                        "is_fraud": 1,
                        "fraud_pattern": pattern,
                        "ring_id": ring_id,
                    }
                )
            return txs, ring_nodes, shared_dev

        if pattern == "burst_activity":
            burst_nodes = account_pool[:4]
            compromised = burst_nodes[0]
            receivers = burst_nodes[1:]
            num_burst = 9 if shifted_tactics else 6
            for b_idx in range(num_burst):
                target = receivers[b_idx % len(receivers)]
                amt = (
                    float(rng.uniform(380.0, self.config.distribution_shift.burst_smurfing_threshold_usd))
                    if shifted_tactics
                    else float(rng.uniform(1100.0, 2800.0))
                )
                txs.append(
                    {
                        "tx_id": f"TX-{tx_id_start + len(txs):06d}",
                        "src_account": account_ids[compromised],
                        "dst_account": account_ids[target],
                        "merchant_id": "",
                        "device_id": str(primary_devices[compromised]),
                        "amount": round(amt, 2),
                        "timestamp": round(base_ts + b_idx * 75.0, 2),
                        "channel": "p2p",
                        "is_fraud": 1,
                        "fraud_pattern": pattern,
                        "ring_id": ring_id,
                    }
                )
            return txs, burst_nodes, None

        # Default: circular_transfer (A1 -> A2 -> A3 -> A4 -> A1)
        cycle_len = 5 if shifted_tactics else 4
        cycle_nodes = account_pool[:cycle_len]
        base_amt = float(rng.uniform(2200.0, 5200.0))
        for c_idx in range(len(cycle_nodes)):
            src_i = cycle_nodes[c_idx]
            dst_i = cycle_nodes[(c_idx + 1) % len(cycle_nodes)]
            txs.append(
                {
                    "tx_id": f"TX-{tx_id_start + len(txs):06d}",
                    "src_account": account_ids[src_i],
                    "dst_account": account_ids[dst_i],
                    "merchant_id": "",
                    "device_id": str(primary_devices[src_i]),
                    "amount": round(base_amt * float(rng.uniform(0.97, 1.01)), 2),
                    "timestamp": round(base_ts + c_idx * 420.0, 2),
                    "channel": "ach",
                    "is_fraud": 1,
                    "fraud_pattern": pattern,
                    "ring_id": ring_id,
                }
            )
        return txs, cycle_nodes, None

    @staticmethod
    def _build_networkx_graph(
        accounts_df: pd.DataFrame,
        devices_df: pd.DataFrame,
        merchants_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
    ) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph()
        for row in accounts_df.itertuples(index=False):
            G.add_node(
                row.account_id,
                node_type="account",
                is_fraud=int(row.is_fraud),
                fraud_pattern=str(row.fraud_pattern),
                ring_id=str(row.ring_id),
                kyc_level=int(row.kyc_level),
                account_age_days=int(row.account_age_days),
            )
        for row in devices_df.itertuples(index=False):
            G.add_node(
                row.device_id,
                node_type="device",
                ip_risk_score=float(row.ip_risk_score),
                os_family=str(row.os_family),
            )
        for row in merchants_df.itertuples(index=False):
            G.add_node(
                row.merchant_id,
                node_type="merchant",
                category=str(row.category),
                chargeback_rate=float(row.chargeback_rate),
            )
        for row in transactions_df.itertuples(index=False):
            G.add_edge(
                row.src_account,
                row.dst_account,
                key=row.tx_id,
                tx_id=row.tx_id,
                amount=float(row.amount),
                timestamp=float(row.timestamp),
                device_id=str(row.device_id),
                merchant_id=str(row.merchant_id),
                is_fraud=int(row.is_fraud),
                fraud_pattern=str(row.fraud_pattern),
                ring_id=str(row.ring_id),
            )
        return G
