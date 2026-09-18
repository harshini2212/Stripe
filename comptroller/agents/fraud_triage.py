"""Fraud-triage agent — the analyst copilot.

Given the ML risk score and the causal drivers, the agent makes the operational call:
is this fraud, and what should Stripe do (freeze the card and open a dispute, open a
dispute, monitor, or clear)? It models the human-in-the-loop that consumes model
output and acts.
"""
from __future__ import annotations

from typing import Any

from ..domain.enums import RiskBand
from .base import BaseAgent

_ACTIONS = ["freeze_card_and_open_dispute", "open_dispute", "monitor", "clear"]
_BAND_ACTION = {
    RiskBand.CRITICAL: "freeze_card_and_open_dispute",
    RiskBand.HIGH: "open_dispute",
    RiskBand.MEDIUM: "monitor",
    RiskBand.LOW: "clear",
}


class FraudTriageAgent(BaseAgent):
    task_name = "fraud_triage"
    output_schema = {
        "type": "object",
        "properties": {
            "is_fraud": {"type": "boolean"},
            "recommended_action": {"type": "string", "enum": _ACTIONS},
            "rationale": {"type": "string"},
        },
        "required": ["is_fraud", "recommended_action"],
        "additionalProperties": False,
    }

    def build_messages(self, inputs: dict[str, Any]) -> tuple[str, str]:
        drivers = "; ".join(inputs.get("drivers", [])) or "none"
        user = (
            "Triage this Stripe Issuing card transaction for fraud using the model's risk score and "
            "causal drivers, then choose an action.\n\n"
            f"Amount: ${inputs['amount_usd']:.2f}\n"
            f"Merchant: {inputs['merchant_name']} ({inputs['merchant_country']})\n"
            f"Channel: {inputs['channel']}\n"
            f"Model risk score: {inputs['risk_score']:.2f} (band: {inputs['risk_band']})\n"
            f"In a suspected fraud ring: {inputs.get('in_ring', False)}\n"
            f"Top causal drivers: {drivers}\n\n"
            "Decide is_fraud and one action: freeze_card_and_open_dispute (critical), "
            "open_dispute (high), monitor (medium), clear (low)."
        )
        return self.persona, user

    def solve(self, inputs: dict[str, Any]) -> dict[str, Any]:
        score = float(inputs["risk_score"])
        band = RiskBand.from_score(score)
        return {
            "is_fraud": score >= float(inputs.get("threshold", 0.5)),
            "recommended_action": _BAND_ACTION[band],
            "rationale": f"risk={score:.2f} ({band.value}); "
                         + ("ring-linked; " if inputs.get("in_ring") else "")
                         + "acted on model band.",
        }

    def perturb(self, inputs: dict[str, Any], base: dict[str, Any], rng) -> dict[str, Any]:
        score = float(inputs["risk_score"])
        # Models are most error-prone near the decision boundary.
        if 0.3 <= score <= 0.7 or rng.random() < 0.3:
            flipped = not base.get("is_fraud", False)
            action = "open_dispute" if flipped else "monitor"
            return {**base, "is_fraud": flipped, "recommended_action": action}
        # Otherwise just soften the action by one notch.
        idx = _ACTIONS.index(base.get("recommended_action", "monitor"))
        return {**base, "recommended_action": _ACTIONS[min(idx + 1, len(_ACTIONS) - 1)]}

    def coerce(self, data: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
        action = self._clean_enum(data.get("recommended_action", ""), _ACTIONS, "monitor")
        return {
            "is_fraud": bool(data.get("is_fraud", False)),
            "recommended_action": action,
            "rationale": data.get("rationale", ""),
        }
