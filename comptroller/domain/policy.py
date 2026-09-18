"""The canonical Stripe spend-policy rule engine.

This single function is the source of truth for policy violations. The synthetic
generator uses it to label transactions, and the policy-audit agent's deterministic
engine uses the very same rules — so the evaluation question becomes a sharp one:
*can a language model replicate the controller's rulebook from a natural-language
policy and a transaction?*
"""
from __future__ import annotations

from .enums import ExpenseCategory, PolicyViolationType
from .models import SpendPolicy


def evaluate_policy(
    *,
    amount_cents: int,
    category: ExpenseCategory,
    has_receipt: bool,
    is_weekend: bool,
    card_per_txn_limit_cents: int,
    policy: SpendPolicy,
    is_duplicate: bool = False,
) -> list[PolicyViolationType]:
    """Return the sorted list of policy violations for a single transaction."""
    violations: list[PolicyViolationType] = []
    if amount_cents > card_per_txn_limit_cents:
        violations.append(PolicyViolationType.OVER_TXN_LIMIT)
    if category in policy.blocked_categories:
        violations.append(PolicyViolationType.BLOCKED_CATEGORY)
    if not has_receipt and amount_cents > policy.receipt_required_over_cents:
        violations.append(PolicyViolationType.MISSING_RECEIPT)
    if policy.block_weekend_personal and is_weekend and category == ExpenseCategory.MEALS:
        violations.append(PolicyViolationType.WEEKEND_PERSONAL)
    if is_duplicate:
        violations.append(PolicyViolationType.DUPLICATE_SPEND)
    # Stable, deterministic ordering for set comparisons in the eval harness.
    return sorted(violations, key=lambda v: v.value)


def requires_approval(amount_cents: int, policy: SpendPolicy) -> bool:
    """Whether the transaction needs pre-approval under the policy."""
    return amount_cents > policy.approval_required_over_cents
