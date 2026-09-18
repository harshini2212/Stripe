"""Dispute-adjudication agent — decide chargeback strategy on a Stripe Issuing card dispute.

True unauthorized fraud should win a chargeback for the cardholder; "friendly fraud"
(buyer's remorse, forgotten subscriptions) should be denied because the evidence shows
the cardholder transacted. The agent weighs the network reason code, the cardholder's
statement, and the fraud model's signal to recommend an action and the dollar impact.
"""
from __future__ import annotations

from typing import Any

from .base import BaseAgent

_RECS = ["pursue_chargeback", "deny_dispute", "request_more_evidence"]

_FRAUD_REASONS = {"fraud_cnp", "fraud_cp", "not_recognized"}
_FRIENDLY_REASONS = {"canceled_recurring", "not_received", "defective", "credit_not_processed"}


class DisputeAgent(BaseAgent):
    task_name = "dispute"
    output_schema = {
        "type": "object",
        "properties": {
            "recommendation": {"type": "string", "enum": _RECS},
            "cardholder_should_win": {"type": "boolean"},
            "financial_impact_usd": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["recommendation", "cardholder_should_win"],
        "additionalProperties": False,
    }

    def build_messages(self, inputs: dict[str, Any]) -> tuple[str, str]:
        user = (
            "Adjudicate this Stripe Issuing card dispute. Decide whether the cardholder should win "
            "(true unauthorized fraud) or the dispute should be denied (friendly fraud / "
            "the cardholder did transact), and recommend an action.\n\n"
            f"Network reason code: {inputs['reason_code']}\n"
            f"Disputed amount: ${inputs['amount_usd']:.2f}\n"
            f"Cardholder statement: \"{inputs['cardholder_statement']}\"\n"
            f"Merchant: {inputs['merchant_name']} ({inputs['merchant_country']}), "
            f"risk={inputs['merchant_risk']:.2f}\n"
            f"Channel: {inputs['channel']}\n"
            f"Transaction on the cardholder's trusted device: {inputs['trusted_device']}\n"
            f"Fraud model risk score for the underlying charge: {inputs['fraud_score']:.2f}\n"
            f"Cardholder has prior legitimate history with this merchant: "
            f"{inputs['prior_history']}\n\n"
            "Actions: pursue_chargeback (cardholder wins), deny_dispute (cardholder "
            "loses), request_more_evidence (genuinely ambiguous). financial_impact_usd "
            "is the dollar exposure Stripe carries if the cardholder wins."
        )
        return self.persona, user

    def solve(self, inputs: dict[str, Any]) -> dict[str, Any]:
        reason = str(inputs["reason_code"])
        score = float(inputs["fraud_score"])
        trusted = bool(inputs["trusted_device"])
        prior = bool(inputs["prior_history"])
        amount = float(inputs["amount_usd"])

        # Strong fraud signal + fraud-type reason -> cardholder wins.
        if reason in _FRAUD_REASONS and score >= 0.5 and not (trusted and prior):
            rec, win = "pursue_chargeback", True
        # Trusted device + prior history + low risk -> friendly fraud, deny.
        elif trusted and prior and score < 0.35:
            rec, win = "deny_dispute", False
        elif reason in _FRIENDLY_REASONS and score < 0.4:
            rec, win = "deny_dispute", False
        elif score >= 0.5:
            rec, win = "pursue_chargeback", True
        else:
            rec, win = "request_more_evidence", score >= 0.5
        return {
            "recommendation": rec,
            "cardholder_should_win": win,
            "financial_impact_usd": round(amount if win else 0.0, 2),
            "rationale": f"reason={reason}, fraud_score={score:.2f}, trusted_device={trusted}, "
                         f"prior_history={prior}.",
        }

    def perturb(self, inputs: dict[str, Any], base: dict[str, Any], rng) -> dict[str, Any]:
        roll = rng.random()
        if roll < 0.5:  # hedge instead of committing
            return {**base, "recommendation": "request_more_evidence"}
        win = not base.get("cardholder_should_win", False)
        rec = "pursue_chargeback" if win else "deny_dispute"
        return {**base, "cardholder_should_win": win, "recommendation": rec,
                "financial_impact_usd": round(float(inputs["amount_usd"]) if win else 0.0, 2)}

    def coerce(self, data: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
        rec = self._clean_enum(data.get("recommendation", ""), _RECS, "request_more_evidence")
        win = bool(data.get("cardholder_should_win", rec == "pursue_chargeback"))
        impact = data.get("financial_impact_usd", inputs["amount_usd"] if win else 0.0)
        try:
            impact = float(impact)
        except (TypeError, ValueError):
            impact = float(inputs["amount_usd"]) if win else 0.0
        return {"recommendation": rec, "cardholder_should_win": win,
                "financial_impact_usd": round(impact, 2), "rationale": data.get("rationale", "")}
