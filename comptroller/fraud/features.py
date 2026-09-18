"""Behavioral-biometric and velocity feature engineering for card transactions.

Produces a leakage-free feature matrix (one row per card transaction) that the ML
ensemble and causal explainer consume. Ground-truth labels are attached in a
separate ``y`` column purely so the eval harness can score predictions — they are
never used as model inputs.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..domain import Dataset
from ..data.geo import haversine_km
from .graph import EntityGraph

# The exact, ordered set of columns the model trains on. Keep in sync with the
# feature construction below; anything not here is metadata, not a model input.
FEATURE_COLUMNS: list[str] = [
    "log_amount",
    "amount_z",
    "amount_round",
    "hour",
    "is_odd_hour",
    "is_weekend",
    "is_cnp",
    "is_recurring",
    "merchant_risk",
    "merchant_foreign",
    "geo_velocity_kmh",
    "impossible_travel",
    "device_novelty",
    "new_merchant_for_card",
    "velocity_1h",
    "velocity_24h",
    "device_card_fanout",
    "ip_card_fanout",
    "ip_cross_metro",
    "ring_component_size",
    "in_suspected_ring",
]


def _round_amount_flag(amount_cents: int) -> float:
    if amount_cents < 100_00:
        return 0.0
    return 1.0 if amount_cents % 50_00 == 0 else 0.0


def build_feature_frame(dataset: Dataset, graph: EntityGraph | None = None) -> pd.DataFrame:
    """Build the per-transaction feature matrix for a tenant.

    Returns a DataFrame indexed by transaction id with all :data:`FEATURE_COLUMNS`
    plus ``y`` (fraud label), ``employee_id``, ``card_id`` and ``merchant_id`` for
    downstream joins/reporting.
    """
    graph = graph or EntityGraph(dataset)
    gfeats = graph.txn_graph_features()
    merchant_index = dataset.merchant_index()

    # Per-card history for sequential features (sorted by time).
    by_card: dict[str, list] = {}
    for t in dataset.card_transactions:
        by_card.setdefault(t.card_id, []).append(t)
    for card_id in by_card:
        by_card[card_id].sort(key=lambda x: x.ts)

    # Per-card log-amount baseline for the amount z-score.
    card_logmean: dict[str, float] = {}
    card_logstd: dict[str, float] = {}
    for card_id, txns in by_card.items():
        logs = np.log([max(t.amount_cents, 1) for t in txns])
        card_logmean[card_id] = float(logs.mean())
        card_logstd[card_id] = float(logs.std() or 1.0)

    rows: list[dict] = []
    index: list[str] = []
    for card_id, txns in by_card.items():
        seen_devices: set[str] = set()
        seen_merchants: set[str] = set()
        prev = None  # previous txn on this card
        for t in txns:
            merchant = merchant_index[t.merchant_id]
            log_amount = math.log(max(t.amount_cents, 1))
            amount_z = (log_amount - card_logmean[card_id]) / card_logstd[card_id]

            # geo-velocity vs the previous transaction on this card.
            geo_kmh = 0.0
            if prev is not None and prev.geo and t.geo:
                hours = max((t.ts - prev.ts).total_seconds() / 3600.0, 1e-3)
                km = haversine_km(prev.geo, t.geo)
                geo_kmh = min(km / hours, 5000.0)

            # velocity windows on this card.
            v1h = sum(1 for o in txns if 0 <= (t.ts - o.ts).total_seconds() <= 3600 and o is not t)
            v24h = sum(1 for o in txns if 0 <= (t.ts - o.ts).total_seconds() <= 86400 and o is not t)

            g = gfeats.get(t.id, {})
            row = {
                "log_amount": log_amount,
                "amount_z": amount_z,
                "amount_round": _round_amount_flag(t.amount_cents),
                "hour": float(t.ts.hour),
                "is_odd_hour": 1.0 if (t.ts.hour <= 5 or t.ts.hour >= 23) else 0.0,
                "is_weekend": 1.0 if t.ts.weekday() >= 5 else 0.0,
                "is_cnp": 1.0 if t.channel.value == "card_not_present" else 0.0,
                "is_recurring": 1.0 if t.channel.value == "recurring" else 0.0,
                "merchant_risk": float(merchant.risk_score),
                "merchant_foreign": 1.0 if merchant.country != "US" else 0.0,
                "geo_velocity_kmh": geo_kmh,
                "impossible_travel": 1.0 if geo_kmh > 900 else 0.0,
                "device_novelty": 1.0 if (t.device_id and t.device_id not in seen_devices) else 0.0,
                "new_merchant_for_card": 1.0 if t.merchant_id not in seen_merchants else 0.0,
                "receipt_missing": 0.0 if t.has_receipt else 1.0,
                "velocity_1h": float(v1h),
                "velocity_24h": float(v24h),
                "device_card_fanout": g.get("device_card_fanout", 1.0),
                "ip_card_fanout": g.get("ip_card_fanout", 1.0),
                "ip_cross_metro": g.get("ip_cross_metro", 0.0),
                "ring_component_size": g.get("ring_component_size", 1.0),
                "in_suspected_ring": g.get("in_suspected_ring", 0.0),
                "y": 1 if t.ground_truth.is_fraud else 0,
                "employee_id": t.employee_id,
                "card_id": t.card_id,
                "merchant_id": t.merchant_id,
            }
            rows.append(row)
            index.append(t.id)

            if t.device_id:
                seen_devices.add(t.device_id)
            seen_merchants.add(t.merchant_id)
            prev = t

    frame = pd.DataFrame(rows, index=index)
    frame.index.name = "txn_id"
    return frame
