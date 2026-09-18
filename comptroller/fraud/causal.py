"""Counterfactual causal attribution for fraud scores.

Feature-importance tells you what the model relies on *in general*; it does not tell
a fraud analyst *why this transaction* scored high. We answer that with
counterfactual interventions in the spirit of Pearl's do-operator:

    For each feature f, set f to its "normal" (legitimate-population) value while
    holding everything else fixed, and measure how far the fraud probability drops.

A large drop means f is *causally driving* this transaction's risk. We pair each
driver with a plain-English explanation and a small hand-specified causal DAG over
the features, so the output reads like an analyst's note rather than a vector of
coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS
from .model import FraudModel

# Plain-English templates keyed by feature. {v} renders the transaction's value.
_EXPLAIN: dict[str, str] = {
    "amount_z": "amount is {v:.1f}σ above this card's normal spend",
    "log_amount": "unusually large ticket size",
    "amount_round": "suspiciously round amount (gift-card / cash-out pattern)",
    "is_odd_hour": "transacted at an odd hour (overnight)",
    "is_weekend": "occurred over the weekend",
    "is_cnp": "card-not-present (online / keyed) transaction",
    "merchant_risk": "merchant carries elevated chargeback / seller risk",
    "merchant_foreign": "foreign merchant, atypical for this cardholder",
    "geo_velocity_kmh": "implied travel speed of {v:.0f} km/h from the prior swipe",
    "impossible_travel": "impossible travel vs. the previous transaction",
    "device_novelty": "first time this card is seen on this device",
    "new_merchant_for_card": "brand-new merchant for this card",
    "receipt_missing": "no receipt attached",
    "velocity_1h": "{v:.0f} transactions on this card within an hour",
    "velocity_24h": "{v:.0f} transactions on this card within a day",
    "device_card_fanout": "device shared across {v:.0f} different cards",
    "ip_card_fanout": "IP shared across {v:.0f} different cards",
    "ip_cross_metro": "shared IP spans multiple home metros (ring signature)",
    "ring_component_size": "part of a {v:.0f}-card shared-infrastructure cluster",
    "in_suspected_ring": "card sits inside a suspected fraud ring",
}

# A compact structural causal sketch: feature -> downstream effect on fraud risk.
# Used for narrative / visualization, not for scoring.
CAUSAL_DAG: dict[str, list[str]] = {
    "device_novelty": ["account_takeover"],
    "device_card_fanout": ["ring_activity"],
    "ip_card_fanout": ["ring_activity"],
    "ip_cross_metro": ["ring_activity"],
    "impossible_travel": ["account_takeover"],
    "geo_velocity_kmh": ["account_takeover"],
    "merchant_risk": ["cash_out"],
    "merchant_foreign": ["cash_out"],
    "amount_round": ["cash_out"],
    "account_takeover": ["fraud"],
    "ring_activity": ["fraud"],
    "cash_out": ["fraud"],
}


@dataclass
class Driver:
    feature: str
    value: float
    delta: float  # drop in fraud probability if this feature were normalized
    explanation: str

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "value": round(self.value, 3),
            "risk_contribution": round(self.delta, 4),
            "explanation": self.explanation,
        }


class CausalExplainer:
    """Counterfactual driver attribution against a legitimate-population baseline."""

    def __init__(self, model: FraudModel, baseline: pd.Series):
        self.model = model
        # The "normal" value for each feature = median over low-risk transactions.
        self.baseline = baseline[FEATURE_COLUMNS].astype(float)

    @classmethod
    def from_frame(cls, model: FraudModel, frame: pd.DataFrame) -> "CausalExplainer":
        scores = model.predict_proba(frame)
        legit = frame.loc[scores < 0.2, FEATURE_COLUMNS]
        baseline = (legit if len(legit) > 30 else frame[FEATURE_COLUMNS]).median()
        return cls(model, baseline)

    def explain(self, row: pd.Series, top_k: int = 4) -> list[Driver]:
        x = row[FEATURE_COLUMNS].astype(float)
        base_p = float(self.model.predict_proba(x.to_frame().T)[0])
        drivers: list[Driver] = []
        for f in FEATURE_COLUMNS:
            if np.isclose(x[f], self.baseline[f]):
                continue
            cf = x.copy()
            cf[f] = self.baseline[f]
            cf_p = float(self.model.predict_proba(cf.to_frame().T)[0])
            delta = base_p - cf_p
            if delta <= 1e-4:
                continue
            tmpl = _EXPLAIN.get(f, f"{f} is abnormal")
            try:
                text = tmpl.format(v=float(x[f]))
            except (KeyError, ValueError):
                text = tmpl
            drivers.append(Driver(f, float(x[f]), float(delta), text))
        drivers.sort(key=lambda d: d.delta, reverse=True)
        return drivers[:top_k]
