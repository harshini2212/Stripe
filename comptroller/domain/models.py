"""Pydantic domain models for the Stripe spend graph.

These are the canonical records the platform reasons over: companies, employees,
Stripe Issuing cards, Stripe Treasury accounts, merchants, card and cash transactions, disputes,
and the spend policy. Transactions carry an optional ``GroundTruth`` block used by
the evaluation harness — it is never visible to agents at inference time.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from .enums import (
    CardStatus,
    CardType,
    CashTxnType,
    Channel,
    DisputeReasonCode,
    DisputeStatus,
    EmployeeRole,
    ExpenseCategory,
    PolicyViolationType,
    TxnStatus,
)


class Company(BaseModel):
    id: str
    name: str
    industry: str
    founded_year: int
    headcount: int
    monthly_card_limit_cents: int
    cash_balance_cents: int
    cash_apy: float = 0.0


class Employee(BaseModel):
    id: str
    company_id: str
    name: str
    email: str
    role: EmployeeRole
    department: str
    manager_id: Optional[str] = None
    home_geo: str  # "City, ST"


class StripeCard(BaseModel):
    id: str
    company_id: str
    employee_id: str
    type: CardType
    last4: str
    status: CardStatus = CardStatus.ACTIVE
    per_txn_limit_cents: int
    monthly_limit_cents: int
    # Stripe virtual cards are frequently vendor-locked.
    locked_merchant_id: Optional[str] = None


class StripeTreasuryAccount(BaseModel):
    id: str
    company_id: str
    balance_cents: int
    apy: float
    account_type: str = "operating"  # operating | reserve | yield


class Merchant(BaseModel):
    id: str
    name: str
    mcc: str
    country: str = "US"
    # Intrinsic merchant risk in [0, 1] (e.g. seller fraud, chargeback history).
    risk_score: float = 0.0


class GroundTruth(BaseModel):
    """Labels for evaluation only — never passed to an agent at inference."""

    is_fraud: bool = False
    fraud_ring_id: Optional[str] = None
    true_category: Optional[ExpenseCategory] = None
    policy_violations: list[PolicyViolationType] = Field(default_factory=list)
    # For disputes: the correct adjudication outcome.
    dispute_should_win: Optional[bool] = None


class CardTransaction(BaseModel):
    id: str
    company_id: str
    card_id: str
    employee_id: str
    merchant_id: str
    amount_cents: int
    currency: str = "USD"
    ts: datetime
    mcc: str
    status: TxnStatus = TxnStatus.SETTLED
    channel: Channel = Channel.CARD_NOT_PRESENT
    # Device / network fingerprints powering behavioral biometrics + graph links.
    device_id: Optional[str] = None
    ip: Optional[str] = None
    geo: Optional[str] = None
    memo: Optional[str] = None
    has_receipt: bool = True
    ground_truth: GroundTruth = Field(default_factory=GroundTruth)

    @property
    def amount(self) -> float:
        return self.amount_cents / 100.0


class CashTransaction(BaseModel):
    id: str
    company_id: str
    account_id: str
    type: CashTxnType
    amount_cents: int  # signed: positive = inflow, negative = outflow
    ts: datetime
    counterparty: Optional[str] = None
    memo: Optional[str] = None

    @property
    def amount(self) -> float:
        return self.amount_cents / 100.0


class Dispute(BaseModel):
    id: str
    company_id: str
    transaction_id: str
    reason_code: DisputeReasonCode
    amount_cents: int
    status: DisputeStatus = DisputeStatus.OPEN
    opened_ts: datetime
    cardholder_statement: str = ""
    ground_truth: GroundTruth = Field(default_factory=GroundTruth)

    @property
    def amount(self) -> float:
        return self.amount_cents / 100.0


class CategoryBudget(BaseModel):
    category: ExpenseCategory
    monthly_limit_cents: int


class SpendPolicy(BaseModel):
    """A company's spend policy — the rulebook agents must enforce."""

    company_id: str
    per_txn_limit_cents: int = 500_000  # $5,000
    receipt_required_over_cents: int = 7_500  # $75
    blocked_categories: list[ExpenseCategory] = Field(default_factory=list)
    category_budgets: list[CategoryBudget] = Field(default_factory=list)
    block_weekend_personal: bool = True
    approval_required_over_cents: int = 1_000_000  # $10,000

    def budget_for(self, category: ExpenseCategory) -> Optional[int]:
        for b in self.category_budgets:
            if b.category == category:
                return b.monthly_limit_cents
        return None


class Dataset(BaseModel):
    """A fully-materialized synthetic Stripe tenant."""

    company: Company
    policy: SpendPolicy
    employees: list[Employee]
    cards: list[StripeCard]
    cash_accounts: list[StripeTreasuryAccount]
    merchants: list[Merchant]
    card_transactions: list[CardTransaction]
    cash_transactions: list[CashTransaction]
    disputes: list[Dispute]

    # ---- convenience indexes -------------------------------------------------
    def merchant_index(self) -> dict[str, Merchant]:
        return {m.id: m for m in self.merchants}

    def employee_index(self) -> dict[str, Employee]:
        return {e.id: e for e in self.employees}

    def card_index(self) -> dict[str, StripeCard]:
        return {c.id: c for c in self.cards}

    def txn_index(self) -> dict[str, CardTransaction]:
        return {t.id: t for t in self.card_transactions}

    def summary(self) -> dict:
        n_fraud = sum(t.ground_truth.is_fraud for t in self.card_transactions)
        n_violations = sum(
            bool(t.ground_truth.policy_violations) for t in self.card_transactions
        )
        spend = sum(t.amount_cents for t in self.card_transactions) / 100.0
        return {
            "company": self.company.name,
            "employees": len(self.employees),
            "cards": len(self.cards),
            "merchants": len(self.merchants),
            "card_transactions": len(self.card_transactions),
            "cash_transactions": len(self.cash_transactions),
            "disputes": len(self.disputes),
            "total_card_spend_usd": round(spend, 2),
            "fraud_transactions": n_fraud,
            "policy_violations": n_violations,
        }
