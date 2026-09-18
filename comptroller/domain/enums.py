"""Enumerations modeling Stripe spend-management primitives."""
from __future__ import annotations

from enum import Enum


class CardType(str, Enum):
    PHYSICAL = "physical"
    VIRTUAL = "virtual"  # Stripe virtual cards (per-vendor, burner)


class CardStatus(str, Enum):
    ACTIVE = "active"
    FROZEN = "frozen"
    TERMINATED = "terminated"


class TxnStatus(str, Enum):
    PENDING = "pending"
    SETTLED = "settled"
    DECLINED = "declined"
    REVERSED = "reversed"


class Channel(str, Enum):
    CARD_PRESENT = "card_present"  # chip / tap, in person
    CARD_NOT_PRESENT = "card_not_present"  # online / keyed
    RECURRING = "recurring"  # subscription / SaaS rails


class EmployeeRole(str, Enum):
    ADMIN = "admin"  # Stripe admin (finance / ops)
    MANAGER = "manager"
    EMPLOYEE = "employee"
    CONTRACTOR = "contractor"


class ExpenseCategory(str, Enum):
    """Stripe expense categories used for accounting + policy."""

    TRAVEL = "travel"
    MEALS = "meals_entertainment"
    SOFTWARE = "software_saas"
    OFFICE = "office_supplies"
    ADVERTISING = "advertising"
    PROFESSIONAL_SERVICES = "professional_services"
    HARDWARE = "hardware_equipment"
    UTILITIES = "utilities"
    RENT = "rent_facilities"
    SHIPPING = "shipping_logistics"
    FUEL = "fuel"
    OTHER = "other"


class CashTxnType(str, Enum):
    """Stripe Treasury money movement."""

    ACH_CREDIT = "ach_credit"
    ACH_DEBIT = "ach_debit"
    WIRE_OUT = "wire_out"
    WIRE_IN = "wire_in"
    CARD_SETTLEMENT = "card_settlement"  # nightly card spend sweep
    YIELD_ACCRUAL = "yield_accrual"  # Stripe Treasury yield
    VENDOR_PAYMENT = "vendor_payment"  # bill pay


class DisputeReasonCode(str, Enum):
    """Card-network style dispute reason codes (Visa/MC aligned)."""

    FRAUD_CARD_NOT_PRESENT = "fraud_cnp"  # 10.4
    FRAUD_CARD_PRESENT = "fraud_cp"  # 10.1
    PRODUCT_NOT_RECEIVED = "not_received"  # 13.1
    PRODUCT_DEFECTIVE = "defective"  # 13.3
    DUPLICATE_PROCESSING = "duplicate"  # 12.6.1
    CREDIT_NOT_PROCESSED = "credit_not_processed"  # 13.2
    SUBSCRIPTION_CANCELED = "canceled_recurring"  # 13.7
    INCORRECT_AMOUNT = "incorrect_amount"  # 12.5
    NOT_RECOGNIZED = "not_recognized"  # cardholder doesn't recognize


class DisputeStatus(str, Enum):
    OPEN = "open"
    EVIDENCE_REVIEW = "evidence_review"
    PROVISIONAL_CREDIT = "provisional_credit"
    REPRESENTMENT = "representment"
    WON = "won"
    LOST = "lost"
    WITHDRAWN = "withdrawn"


class PolicyViolationType(str, Enum):
    OVER_TXN_LIMIT = "over_transaction_limit"
    OVER_CATEGORY_LIMIT = "over_category_budget"
    BLOCKED_CATEGORY = "blocked_category"
    MISSING_RECEIPT = "missing_receipt"
    OUT_OF_POLICY_MERCHANT = "out_of_policy_merchant"
    WEEKEND_PERSONAL = "weekend_personal_spend"
    DUPLICATE_SPEND = "duplicate_spend"


class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @classmethod
    def from_score(cls, score: float) -> "RiskBand":
        if score >= 0.85:
            return cls.CRITICAL
        if score >= 0.6:
            return cls.HIGH
        if score >= 0.3:
            return cls.MEDIUM
        return cls.LOW


# Representative MCC -> (label, default ExpenseCategory).
MCC_TABLE: dict[str, tuple[str, ExpenseCategory]] = {
    "3000": ("Airlines", ExpenseCategory.TRAVEL),
    "3501": ("Hotels & Lodging", ExpenseCategory.TRAVEL),
    "4111": ("Transit / Rideshare", ExpenseCategory.TRAVEL),
    "4121": ("Taxis & Rideshare", ExpenseCategory.TRAVEL),
    "5812": ("Restaurants", ExpenseCategory.MEALS),
    "5813": ("Bars & Nightlife", ExpenseCategory.MEALS),
    "5814": ("Fast Food", ExpenseCategory.MEALS),
    "5734": ("Software Stores", ExpenseCategory.SOFTWARE),
    "7372": ("SaaS / Cloud", ExpenseCategory.SOFTWARE),
    "5045": ("Computers & Peripherals", ExpenseCategory.HARDWARE),
    "5111": ("Office Supplies", ExpenseCategory.OFFICE),
    "7311": ("Advertising Services", ExpenseCategory.ADVERTISING),
    "7392": ("Consulting / Professional", ExpenseCategory.PROFESSIONAL_SERVICES),
    "8931": ("Accounting / Audit", ExpenseCategory.PROFESSIONAL_SERVICES),
    "4900": ("Utilities", ExpenseCategory.UTILITIES),
    "6513": ("Real Estate / Rent", ExpenseCategory.RENT),
    "4215": ("Courier & Shipping", ExpenseCategory.SHIPPING),
    "5541": ("Fuel / Service Stations", ExpenseCategory.FUEL),
    "5999": ("Misc Retail", ExpenseCategory.OTHER),
}
