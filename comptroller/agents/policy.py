"""Policy-audit agent — replicate Stripe's spend-policy rulebook from natural language.

The deterministic engine *is* the canonical rule engine, so this task measures how
faithfully a model reproduces a financial controller's judgments: which violations
fire, and whether the charge needs pre-approval.
"""
from __future__ import annotations

from typing import Any

from ..domain.enums import ExpenseCategory, PolicyViolationType
from ..domain.models import CategoryBudget, SpendPolicy
from ..domain.policy import evaluate_policy, requires_approval
from .base import BaseAgent

_VIOLATIONS = [v.value for v in PolicyViolationType]


class PolicyAuditAgent(BaseAgent):
    task_name = "policy_audit"
    output_schema = {
        "type": "object",
        "properties": {
            "violations": {"type": "array", "items": {"type": "string", "enum": _VIOLATIONS}},
            "requires_approval": {"type": "boolean"},
            "rationale": {"type": "string"},
        },
        "required": ["violations", "requires_approval"],
        "additionalProperties": False,
    }

    def build_messages(self, inputs: dict[str, Any]) -> tuple[str, str]:
        blocked = ", ".join(inputs.get("blocked_categories", [])) or "none"
        user = (
            "Audit this Stripe Issuing card transaction against company spend policy and list "
            "every violation that applies.\n\n"
            f"Amount: ${inputs['amount_usd']:.2f}\n"
            f"Category: {inputs['category']}\n"
            f"Has receipt: {inputs['has_receipt']}\n"
            f"Weekend: {inputs['is_weekend']}\n"
            f"Identical earlier charge on this card today: {inputs.get('is_duplicate', False)}\n\n"
            "POLICY:\n"
            f"- Per-transaction limit on this card: ${inputs['card_per_txn_limit_usd']:,.0f} "
            "(over the limit => over_transaction_limit)\n"
            f"- Receipt required for charges over ${inputs['receipt_required_over_usd']:,.0f} "
            "(missing => missing_receipt)\n"
            f"- Blocked categories: {blocked} (charge in one => blocked_category)\n"
            "- Weekend spend in meals_entertainment is treated as personal "
            "(=> weekend_personal_spend)\n"
            "- An identical earlier charge today => duplicate_spend\n"
            f"- Pre-approval required for charges over ${inputs['approval_required_over_usd']:,.0f}\n\n"
            "Return the exact violation codes that apply and whether pre-approval is required."
        )
        return self.persona, user

    def _policy(self, inputs: dict[str, Any]) -> SpendPolicy:
        blocked = [ExpenseCategory(c) for c in inputs.get("blocked_categories", [])
                   if c in {e.value for e in ExpenseCategory}]
        return SpendPolicy(
            company_id="_",
            per_txn_limit_cents=int(inputs["card_per_txn_limit_usd"] * 100),
            receipt_required_over_cents=int(inputs["receipt_required_over_usd"] * 100),
            blocked_categories=blocked,
            category_budgets=[CategoryBudget(category=ExpenseCategory.MEALS, monthly_limit_cents=0)],
            block_weekend_personal=True,
            approval_required_over_cents=int(inputs["approval_required_over_usd"] * 100),
        )

    def solve(self, inputs: dict[str, Any]) -> dict[str, Any]:
        policy = self._policy(inputs)
        cents = int(round(inputs["amount_usd"] * 100))
        violations = evaluate_policy(
            amount_cents=cents,
            category=ExpenseCategory(inputs["category"]),
            has_receipt=bool(inputs["has_receipt"]),
            is_weekend=bool(inputs["is_weekend"]),
            card_per_txn_limit_cents=int(inputs["card_per_txn_limit_usd"] * 100),
            policy=policy,
            is_duplicate=bool(inputs.get("is_duplicate", False)),
        )
        return {
            "violations": [v.value for v in violations],
            "requires_approval": requires_approval(cents, policy),
            "rationale": "Applied per-txn limit, receipt, blocked-category, weekend and "
                         "duplicate rules.",
        }

    def perturb(self, inputs: dict[str, Any], base: dict[str, Any], rng) -> dict[str, Any]:
        viols = list(base.get("violations", []))
        approval = base.get("requires_approval", False)
        roll = rng.random()
        # Weaker models tend to miss the subtle ones (weekend / duplicate) or
        # occasionally hallucinate an extra violation.
        subtle = [v for v in viols if v in ("weekend_personal_spend", "duplicate_spend",
                                            "missing_receipt")]
        if viols and roll < 0.55:
            drop = subtle[0] if subtle else viols[int(rng.integers(0, len(viols)))]
            viols = [v for v in viols if v != drop]
        elif roll < 0.75:
            spurious = [v for v in _VIOLATIONS if v not in viols]
            if spurious:
                viols = sorted(viols + [spurious[int(rng.integers(0, len(spurious)))]])
        else:
            approval = not approval
        return {**base, "violations": viols, "requires_approval": approval}

    def coerce(self, data: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
        raw = data.get("violations", []) or []
        viols = sorted({self._clean_enum(v, _VIOLATIONS, "") for v in raw} - {""})
        return {
            "violations": viols,
            "requires_approval": bool(data.get("requires_approval", False)),
            "rationale": data.get("rationale", ""),
        }
