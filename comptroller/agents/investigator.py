"""FraudInvestigator — an autonomous, multi-step fraud investigation workflow.

Given a single suspicious transaction, the investigator pulls the ML assessment,
expands to the entity graph to find co-conspirators, checks the card's recent
velocity, profiles the cardholder, and synthesizes a remediation plan (freeze the
Stripe Issuing card, open a network dispute, sweep the whole ring, rotate the device). The
recorded steps read like an analyst's case file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..domain import Dataset
from ..domain.enums import RiskBand
from .tools import Toolbox


@dataclass
class InvestigationReport:
    txn_id: str
    risk_score: float
    risk_band: RiskBand
    is_fraud: bool
    employee: str
    card_id: str
    merchant: str
    amount_usd: float
    ring_id: str | None
    ring_member_cards: list[str]
    ring_exposure_usd: float
    drivers: list[str]
    velocity: dict[str, Any]
    recommended_actions: list[str]
    narrative: str
    steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "txn_id": self.txn_id,
            "risk_score": round(self.risk_score, 4),
            "risk_band": self.risk_band.value,
            "is_fraud": self.is_fraud,
            "employee": self.employee,
            "merchant": self.merchant,
            "amount_usd": self.amount_usd,
            "ring_id": self.ring_id,
            "ring_member_cards": self.ring_member_cards,
            "ring_exposure_usd": self.ring_exposure_usd,
            "drivers": self.drivers,
            "velocity_72h": self.velocity,
            "recommended_actions": self.recommended_actions,
            "narrative": self.narrative,
            "steps": self.steps,
        }


class FraudInvestigator:
    def __init__(self, dataset: Dataset, pipeline):
        self.dataset = dataset
        self.pipeline = pipeline

    def investigate(self, txn_id: str) -> InvestigationReport:
        tools = Toolbox(self.dataset, self.pipeline)

        txn = tools.get_transaction(txn_id)
        assessment = tools.fraud_assessment(txn_id)
        merchant = tools.get_merchant(txn.merchant_id)
        employee = tools.get_employee(txn.employee_id)
        ring = tools.ring_for_card(txn.card_id)
        velocity = tools.card_velocity(txn.card_id, txn.ts)

        drivers = [d.explanation for d in assessment.drivers]
        ring_members = ring.card_ids if ring else []
        ring_exposure = (ring.total_exposure_cents / 100.0) if ring else 0.0

        actions = self._actions(assessment.risk_band, ring is not None, len(ring_members))
        narrative = self._narrative(txn, merchant, employee, assessment, ring, velocity)

        return InvestigationReport(
            txn_id=txn_id,
            risk_score=assessment.risk_score,
            risk_band=assessment.risk_band,
            is_fraud=assessment.predicted_fraud,
            employee=employee.name if employee else txn.employee_id,
            card_id=txn.card_id,
            merchant=merchant.name if merchant else txn.merchant_id,
            amount_usd=txn.amount,
            ring_id=ring.ring_id if ring else None,
            ring_member_cards=ring_members,
            ring_exposure_usd=ring_exposure,
            drivers=drivers,
            velocity=velocity,
            recommended_actions=actions,
            narrative=narrative,
            steps=tools.calls,
        )

    @staticmethod
    def _actions(band: RiskBand, in_ring: bool, n_members: int) -> list[str]:
        actions: list[str] = []
        if band in (RiskBand.CRITICAL, RiskBand.HIGH):
            actions.append("Freeze the Stripe Issuing card immediately")
            actions.append("Open a network dispute (reason 10.4 — fraud, card-not-present)")
            actions.append("Rotate the cardholder's device credentials")
            if in_ring:
                actions.append(f"Freeze all {n_members} cards in the ring and open a "
                               "coordinated investigation")
                actions.append("Escalate to the Stripe fraud-ops on-call")
        elif band == RiskBand.MEDIUM:
            actions.append("Apply step-up authentication on the card")
            actions.append("Request cardholder confirmation of recent activity")
            actions.append("Monitor for 24h and re-score")
        else:
            actions.append("Clear — no action required")
        return actions

    @staticmethod
    def _narrative(txn, merchant, employee, assessment, ring, velocity) -> str:
        who = employee.name if employee else txn.employee_id
        parts = [
            f"Transaction {txn.id} for ${txn.amount:,.0f} on card {txn.card_id} "
            f"({who}) at {merchant.name if merchant else txn.merchant_id} scored "
            f"{assessment.risk_score:.0%} ({assessment.risk_band.value.upper()})."
        ]
        if assessment.drivers:
            parts.append("Primary drivers: " + "; ".join(d.explanation for d in assessment.drivers[:3]) + ".")
        if ring:
            parts.append(
                f"The card is linked to {len(ring.card_ids)} cards via shared "
                f"{'devices' if ring.shared_devices else 'infrastructure'} "
                f"(ring {ring.ring_id}, ${ring.total_exposure_cents / 100:,.0f} total exposure) — "
                "this is a coordinated ring, not an isolated event.")
        parts.append(
            f"Card velocity over the last 72h: {velocity['count']} charges totalling "
            f"${velocity['sum_usd']:,.0f} across {velocity['distinct_merchants']} merchants "
            f"and {velocity['distinct_geos']} locations.")
        return " ".join(parts)
