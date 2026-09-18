"""Expense categorization agent — assign the correct Stripe GL/expense category.

The interesting cases are *miscoded* transactions: the network MCC is the generic
"5999 — Misc Retail", but the merchant name makes the true category obvious
(``Datadog`` is software, ``United Airlines`` is travel). A model that blindly trusts
the MCC gets these wrong; the analytical engine recovers them from the merchant name.
"""
from __future__ import annotations

from typing import Any

from ..domain.enums import MCC_TABLE, ExpenseCategory
from .base import BaseAgent

_CATEGORIES = [c.value for c in ExpenseCategory]

# Merchant-name keywords -> category, used when the MCC is generic/missing.
_NAME_KEYWORDS: list[tuple[str, ExpenseCategory]] = [
    ("datadog", ExpenseCategory.SOFTWARE), ("notion", ExpenseCategory.SOFTWARE),
    ("vercel", ExpenseCategory.SOFTWARE), ("snowflake", ExpenseCategory.SOFTWARE),
    ("github", ExpenseCategory.SOFTWARE), ("figma", ExpenseCategory.SOFTWARE),
    ("linear", ExpenseCategory.SOFTWARE), ("pagerduty", ExpenseCategory.SOFTWARE),
    ("apple", ExpenseCategory.HARDWARE), ("dell", ExpenseCategory.HARDWARE),
    ("cdw", ExpenseCategory.HARDWARE), ("best buy", ExpenseCategory.HARDWARE),
    ("framework", ExpenseCategory.HARDWARE),
    ("airlines", ExpenseCategory.TRAVEL), ("delta", ExpenseCategory.TRAVEL),
    ("marriott", ExpenseCategory.TRAVEL), ("hilton", ExpenseCategory.TRAVEL),
    ("uber", ExpenseCategory.TRAVEL), ("lyft", ExpenseCategory.TRAVEL),
    ("airbnb", ExpenseCategory.TRAVEL),
    ("blue bottle", ExpenseCategory.MEALS), ("sweetgreen", ExpenseCategory.MEALS),
    ("doordash", ExpenseCategory.MEALS), ("chipotle", ExpenseCategory.MEALS),
    ("tartine", ExpenseCategory.MEALS), ("philz", ExpenseCategory.MEALS),
    ("google ads", ExpenseCategory.ADVERTISING), ("meta ads", ExpenseCategory.ADVERTISING),
    ("linkedin", ExpenseCategory.ADVERTISING), ("reddit ads", ExpenseCategory.ADVERTISING),
    ("tiktok", ExpenseCategory.ADVERTISING),
    ("deloitte", ExpenseCategory.PROFESSIONAL_SERVICES), ("pwc", ExpenseCategory.PROFESSIONAL_SERVICES),
    ("cooley", ExpenseCategory.PROFESSIONAL_SERVICES), ("gunderson", ExpenseCategory.PROFESSIONAL_SERVICES),
    ("carta", ExpenseCategory.PROFESSIONAL_SERVICES),
    ("staples", ExpenseCategory.OFFICE), ("amazon business", ExpenseCategory.OFFICE),
    ("wb mason", ExpenseCategory.OFFICE), ("costco", ExpenseCategory.OFFICE),
    ("pg&e", ExpenseCategory.UTILITIES), ("comcast", ExpenseCategory.UTILITIES),
    ("at&t", ExpenseCategory.UTILITIES),
    ("wework", ExpenseCategory.RENT), ("industrious", ExpenseCategory.RENT),
    ("fedex", ExpenseCategory.SHIPPING), ("ups", ExpenseCategory.SHIPPING),
    ("usps", ExpenseCategory.SHIPPING), ("shippo", ExpenseCategory.SHIPPING),
    ("shell", ExpenseCategory.FUEL), ("chevron", ExpenseCategory.FUEL),
]


def recoverable_from_name(name: str) -> bool:
    """Whether a generic-MCC transaction can be recovered from the merchant name."""
    n = str(name).lower()
    return any(kw in n for kw, _ in _NAME_KEYWORDS)


class CategorizationAgent(BaseAgent):
    task_name = "categorization"
    output_schema = {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": _CATEGORIES},
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": ["category", "confidence"],
        "additionalProperties": False,
    }

    def build_messages(self, inputs: dict[str, Any]) -> tuple[str, str]:
        user = (
            "Classify this Stripe Issuing card transaction into exactly one expense category.\n"
            f"Merchant: {inputs['merchant_name']}\n"
            f"MCC: {inputs['mcc']} ({inputs.get('mcc_label', 'unknown')})\n"
            f"Memo: {inputs.get('memo', '')}\n"
            f"Amount: ${inputs['amount_usd']:.2f}\n"
            f"Cardholder department: {inputs.get('department', 'unknown')}\n\n"
            "If the MCC is generic (e.g. 5999) but the merchant name clearly implies a "
            "category, trust the merchant name. Allowed categories: "
            + ", ".join(_CATEGORIES) + "."
        )
        return self.persona, user

    def solve(self, inputs: dict[str, Any]) -> dict[str, Any]:
        mcc = str(inputs.get("mcc", "")).strip()
        name = str(inputs.get("merchant_name", "")).lower()
        # 1) A specific (non-generic) MCC is authoritative.
        if mcc in MCC_TABLE and mcc != "5999":
            cat = MCC_TABLE[mcc][1]
            return {"category": cat.value, "confidence": 0.96,
                    "rationale": f"MCC {mcc} maps directly to {cat.value}."}
        # 2) Generic/unknown MCC -> recover from the merchant name.
        for kw, cat in _NAME_KEYWORDS:
            if kw in name:
                return {"category": cat.value, "confidence": 0.78,
                        "rationale": f"Generic MCC; merchant name '{name}' implies {cat.value}."}
        # 3) Nothing matched -> OTHER.
        return {"category": ExpenseCategory.OTHER.value, "confidence": 0.4,
                "rationale": "No specific MCC or name signal; defaulting to other."}

    def perturb(self, inputs: dict[str, Any], base: dict[str, Any], rng) -> dict[str, Any]:
        # Emulate a model that naively trusts the (generic) MCC, or slips to a sibling.
        mcc = str(inputs.get("mcc", "")).strip()
        naive = MCC_TABLE.get(mcc, ("", ExpenseCategory.OTHER))[1].value
        if naive != base.get("category") and rng.random() < 0.7:
            return {**base, "category": naive, "confidence": 0.55}
        alt = [c for c in _CATEGORIES if c != base.get("category")]
        return {**base, "category": alt[int(rng.integers(0, len(alt)))], "confidence": 0.5}

    def coerce(self, data: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
        cat = self._clean_enum(data.get("category", ""), _CATEGORIES, ExpenseCategory.OTHER.value)
        conf = data.get("confidence", 0.5)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.5
        return {"category": cat, "confidence": conf, "rationale": data.get("rationale", "")}
