"""Builders that turn domain objects into agent input payloads.

Shared by the live orchestrator and the eval golden-case builder so the two never
drift. Builders read only *observable* facts about a transaction (never the held-out
fraud / violation labels the agent is being asked to predict).
"""
from __future__ import annotations

from typing import Any

from ..domain import Dataset
from ..domain.enums import MCC_TABLE, PolicyViolationType


def categorization_inputs(txn, ds: Dataset, *, miscode: bool = False) -> dict[str, Any]:
    m = ds.merchant_index()[txn.merchant_id]
    emp = ds.employee_index().get(txn.employee_id)
    mcc = "5999" if miscode else txn.mcc
    label = MCC_TABLE.get(mcc, ("Misc Retail", None))[0]
    return {
        "merchant_name": m.name,
        "mcc": mcc,
        "mcc_label": label,
        "memo": txn.memo or "",
        "amount_usd": txn.amount,
        "department": emp.department if emp else "unknown",
    }


def policy_inputs(txn, ds: Dataset) -> dict[str, Any]:
    card = ds.card_index()[txn.card_id]
    is_dup = PolicyViolationType.DUPLICATE_SPEND in txn.ground_truth.policy_violations
    return {
        "amount_usd": txn.amount,
        "category": txn.ground_truth.true_category.value if txn.ground_truth.true_category
        else "other",
        "has_receipt": txn.has_receipt,
        "is_weekend": txn.ts.weekday() >= 5,
        "is_duplicate": is_dup,
        "card_per_txn_limit_usd": card.per_txn_limit_cents / 100.0,
        "receipt_required_over_usd": ds.policy.receipt_required_over_cents / 100.0,
        "blocked_categories": [c.value for c in ds.policy.blocked_categories],
        "approval_required_over_usd": ds.policy.approval_required_over_cents / 100.0,
    }


def triage_inputs(txn, assessment, ds: Dataset) -> dict[str, Any]:
    m = ds.merchant_index()[txn.merchant_id]
    return {
        "amount_usd": txn.amount,
        "merchant_name": m.name,
        "merchant_country": m.country,
        "channel": txn.channel.value,
        "risk_score": assessment.risk_score,
        "risk_band": assessment.risk_band.value,
        "in_ring": assessment.ring_id is not None,
        "drivers": [d.explanation for d in assessment.drivers],
        "threshold": 0.5,
    }


def _prior_history(txn, ds: Dataset) -> bool:
    for o in ds.card_transactions:
        if (o.card_id == txn.card_id and o.merchant_id == txn.merchant_id
                and o.ts < txn.ts and not o.ground_truth.is_fraud):
            return True
    return False


def dispute_inputs(dispute, ds: Dataset, *, fraud_score: float) -> dict[str, Any]:
    txn = ds.txn_index()[dispute.transaction_id]
    m = ds.merchant_index()[txn.merchant_id]
    trusted = txn.device_id == f"dev_{txn.employee_id}"
    return {
        "reason_code": dispute.reason_code.value,
        "amount_usd": dispute.amount,
        "cardholder_statement": dispute.cardholder_statement,
        "merchant_name": m.name,
        "merchant_country": m.country,
        "merchant_risk": m.risk_score,
        "channel": txn.channel.value,
        "trusted_device": trusted,
        "fraud_score": float(fraud_score),
        "prior_history": _prior_history(txn, ds),
    }
