"""Tieout agent — reconcile an expense report to the card statement.

This is financial correctness in its purest form: sum the line items, compare to the
submitted total, and report whether it ties out and by how much. Transposed digits,
dropped lines, and double-counts must be caught to the cent. The eval scores the
computed total with a numeric tolerance, so arithmetic mistakes are penalized.
"""
from __future__ import annotations

from typing import Any

from .base import BaseAgent

# Anything within half a cent is considered tied out.
TOLERANCE_USD = 0.005


class TieoutAgent(BaseAgent):
    task_name = "tieout"
    output_schema = {
        "type": "object",
        "properties": {
            "computed_total_usd": {"type": "number"},
            "ties_out": {"type": "boolean"},
            "discrepancy_usd": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["computed_total_usd", "ties_out", "discrepancy_usd"],
        "additionalProperties": False,
    }

    def build_messages(self, inputs: dict[str, Any]) -> tuple[str, str]:
        lines = "\n".join(
            f"  {i + 1:>2}. {li['merchant']:<28} ${li['amount_usd']:>10,.2f}"
            for i, li in enumerate(inputs["line_items"])
        )
        user = (
            "Reconcile this Stripe expense report. Sum the line items, compare to the "
            "submitted total, and report the discrepancy.\n\n"
            f"Employee: {inputs.get('employee', 'unknown')} | Period: {inputs.get('period', '')}\n"
            f"Line items ({len(inputs['line_items'])}):\n{lines}\n\n"
            f"Submitted total: ${inputs['submitted_total_usd']:,.2f}\n\n"
            "computed_total_usd is the true sum of the line items. ties_out is true only "
            "if it matches the submitted total to the cent. discrepancy_usd = "
            "submitted_total - computed_total (signed)."
        )
        return self.persona, user

    def solve(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # Work in integer cents to avoid floating-point drift.
        total_cents = sum(int(round(li["amount_usd"] * 100)) for li in inputs["line_items"])
        submitted_cents = int(round(inputs["submitted_total_usd"] * 100))
        disc_cents = submitted_cents - total_cents
        return {
            "computed_total_usd": round(total_cents / 100.0, 2),
            "ties_out": disc_cents == 0,
            "discrepancy_usd": round(disc_cents / 100.0, 2),
            "rationale": f"Summed {len(inputs['line_items'])} line items to "
                         f"${total_cents / 100:,.2f}.",
        }

    def perturb(self, inputs: dict[str, Any], base: dict[str, Any], rng) -> dict[str, Any]:
        items = inputs["line_items"]
        computed = float(base["computed_total_usd"])
        # Emulate an arithmetic slip: drop or double-count a line, or fat-finger a cent.
        roll = rng.random()
        if items and roll < 0.6:
            li = items[int(rng.integers(0, len(items)))]
            computed = round(computed + (li["amount_usd"] if roll < 0.3 else -li["amount_usd"]), 2)
        else:
            computed = round(computed + float(rng.choice([-100, -10, 10, 100, 0.9])), 2)
        disc = round(float(inputs["submitted_total_usd"]) - computed, 2)
        return {**base, "computed_total_usd": computed, "discrepancy_usd": disc,
                "ties_out": abs(disc) < TOLERANCE_USD}

    def coerce(self, data: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
        def _num(v, default=0.0):
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        computed = _num(data.get("computed_total_usd"))
        disc = data.get("discrepancy_usd")
        disc = _num(disc, round(float(inputs["submitted_total_usd"]) - computed, 2))
        return {
            "computed_total_usd": round(computed, 2),
            "ties_out": bool(data.get("ties_out", abs(disc) < TOLERANCE_USD)),
            "discrepancy_usd": round(disc, 2),
            "rationale": data.get("rationale", ""),
        }
