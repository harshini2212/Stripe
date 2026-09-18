"""End-to-end fraud pipeline: graph + features + ensemble + causal explanation.

This is the object the fraud-investigator agent and the eval harness both lean on.
It trains on a held-out split for honest metrics, then refits on all data for
production scoring, and turns every score into an actionable, *explained* decision
(freeze the Stripe Issuing card, open a network dispute, monitor, or clear).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..domain import Dataset
from ..domain.enums import RiskBand
from .causal import CausalExplainer, Driver
from .features import build_feature_frame
from .graph import EntityGraph, RingFinding
from .model import FraudMetrics, FraudModel

# Risk band -> recommended Stripe operational action.
_ACTION = {
    RiskBand.CRITICAL: "freeze_card_and_open_dispute",
    RiskBand.HIGH: "open_dispute",
    RiskBand.MEDIUM: "monitor",
    RiskBand.LOW: "clear",
}


@dataclass
class FraudAssessment:
    txn_id: str
    employee_id: str
    card_id: str
    merchant_id: str
    merchant_name: str
    amount_cents: int
    risk_score: float
    risk_band: RiskBand
    predicted_fraud: bool
    recommended_action: str
    drivers: list[Driver] = field(default_factory=list)
    ring_id: str | None = None
    # Held-out truth, surfaced only for evaluation / reporting.
    actual_fraud: bool | None = None

    @property
    def amount_usd(self) -> float:
        return self.amount_cents / 100.0

    def headline(self) -> str:
        why = "; ".join(d.explanation for d in self.drivers[:2]) or "no dominant driver"
        return (f"{self.txn_id} ${self.amount_usd:,.0f} -> {self.risk_band.value.upper()} "
                f"({self.risk_score:.0%}) - {why}")

    def to_dict(self) -> dict:
        return {
            "txn_id": self.txn_id,
            "employee_id": self.employee_id,
            "card_id": self.card_id,
            "merchant": self.merchant_name,
            "amount_usd": round(self.amount_usd, 2),
            "risk_score": round(self.risk_score, 4),
            "risk_band": self.risk_band.value,
            "predicted_fraud": self.predicted_fraud,
            "recommended_action": self.recommended_action,
            "ring_id": self.ring_id,
            "drivers": [d.to_dict() for d in self.drivers],
            "actual_fraud": self.actual_fraud,
        }


class FraudPipeline:
    """Trains the ensemble and serves explained, actionable fraud assessments."""

    def __init__(self, dataset: Dataset, *, seed: int = 7, test_size: float = 0.3):
        self.dataset = dataset
        self.seed = seed
        self.graph = EntityGraph(dataset)
        self.frame = build_feature_frame(dataset, self.graph)
        self._txn_index = dataset.txn_index()
        self._merchant_index = dataset.merchant_index()

        X = self.frame
        y = self.frame["y"].to_numpy(dtype=int)

        # Honest holdout metrics (stratify when both classes are present).
        stratify = y if (0 < y.sum() < len(y)) else None
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=stratify)
        holdout = FraudModel(random_state=seed).fit(Xtr, ytr)
        self.holdout_metrics: FraudMetrics = holdout.evaluate(Xte, yte)

        # Production model: refit on all data for scoring every transaction.
        self.model = FraudModel(random_state=seed).fit(X, y)
        self.explainer = CausalExplainer.from_frame(self.model, X)

        # Score everything once.
        self._scores = pd.Series(self.model.predict_proba(X), index=X.index)
        self._ring_findings = self.graph.detect_rings()
        self._txn_to_ring: dict[str, str] = {}
        for r in self._ring_findings:
            for tid in r.txn_ids:
                self._txn_to_ring[tid] = r.ring_id

    # ---- queries -------------------------------------------------------------
    def assess(self, txn_id: str, *, with_drivers: bool = True) -> FraudAssessment:
        row = self.frame.loc[txn_id]
        score = float(self._scores.loc[txn_id])
        band = RiskBand.from_score(score)
        drivers = self.explainer.explain(row) if with_drivers and score >= 0.2 else []
        txn = self._txn_index[txn_id]
        merchant = self._merchant_index.get(txn.merchant_id)
        return FraudAssessment(
            txn_id=txn_id,
            employee_id=str(row["employee_id"]),
            card_id=str(row["card_id"]),
            merchant_id=str(row["merchant_id"]),
            merchant_name=merchant.name if merchant else str(row["merchant_id"]),
            amount_cents=txn.amount_cents,
            risk_score=score,
            risk_band=band,
            predicted_fraud=score >= self.model.threshold,
            recommended_action=_ACTION[band],
            drivers=drivers,
            ring_id=self._txn_to_ring.get(txn_id),
            actual_fraud=bool(txn.ground_truth.is_fraud),
        )

    def assess_all(self, *, min_score: float = 0.0, with_drivers: bool = False) -> list[FraudAssessment]:
        ids = self._scores[self._scores >= min_score].sort_values(ascending=False).index
        return [self.assess(tid, with_drivers=with_drivers) for tid in ids]

    def top_alerts(self, k: int = 10) -> list[FraudAssessment]:
        ids = self._scores.sort_values(ascending=False).head(k).index
        return [self.assess(tid, with_drivers=True) for tid in ids]

    def rings(self) -> list[RingFinding]:
        return self._ring_findings

    def scores(self) -> pd.Series:
        return self._scores

    def labels(self) -> np.ndarray:
        return self.frame["y"].to_numpy(dtype=int)
