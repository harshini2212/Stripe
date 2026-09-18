"""Deterministic synthetic Stripe tenant generator.

Produces a fully-materialized :class:`~comptroller.domain.models.Dataset` — a single
Stripe customer with employees, Stripe Issuing cards, Stripe Treasury accounts, merchants, ~thousands
of card transactions, cash money-movement, and disputes.

Realistic *latent* structure is planted so the downstream ML / agent layers have
genuine signal to recover:

* **Fraud rings** — clusters of compromised cards sharing a device + IP, transacting
  in bursts at high-risk card-not-present merchants in a far metro, producing
  "impossible travel" relative to the cardholder's normal home-metro activity.
* **Lone-wolf fraud** — single compromised cards (new device, geo jump) to exercise
  non-graph signals.
* **Policy violations** — over-limit, blocked-category, missing-receipt, weekend
  personal, and duplicate spend, each tagged in held-out ground truth.

Everything is seeded, so the same ``seed`` always yields byte-identical data — which
is what makes the multi-model eval leaderboard reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from ..domain import (
    StripeCard,
    StripeTreasuryAccount,
    CardTransaction,
    CardType,
    CashTransaction,
    CashTxnType,
    CategoryBudget,
    Channel,
    Company,
    Dataset,
    Dispute,
    DisputeReasonCode,
    DisputeStatus,
    Employee,
    EmployeeRole,
    ExpenseCategory,
    GroundTruth,
    Merchant,
    PolicyViolationType,
    SpendPolicy,
    TxnStatus,
)
from ..domain.enums import MCC_TABLE
from ..domain.policy import evaluate_policy
from .geo import FAR_METROS, HOME_METROS

# Anchor the data to a fixed window so generation never depends on wall-clock time.
END_DATE = datetime(2026, 1, 1)

DEPARTMENTS = ("Engineering", "Sales", "Marketing", "Finance", "Operations", "Executive")

# Department -> categories that department legitimately spends in (weighted).
DEPT_CATEGORIES: dict[str, list[ExpenseCategory]] = {
    "Engineering": [ExpenseCategory.SOFTWARE, ExpenseCategory.HARDWARE, ExpenseCategory.MEALS],
    "Sales": [ExpenseCategory.TRAVEL, ExpenseCategory.MEALS, ExpenseCategory.SOFTWARE],
    "Marketing": [ExpenseCategory.ADVERTISING, ExpenseCategory.SOFTWARE, ExpenseCategory.MEALS],
    "Finance": [ExpenseCategory.PROFESSIONAL_SERVICES, ExpenseCategory.SOFTWARE, ExpenseCategory.OFFICE],
    "Operations": [ExpenseCategory.OFFICE, ExpenseCategory.SHIPPING, ExpenseCategory.UTILITIES, ExpenseCategory.RENT],
    "Executive": [ExpenseCategory.TRAVEL, ExpenseCategory.MEALS, ExpenseCategory.PROFESSIONAL_SERVICES],
}

# Typical per-transaction amount range (USD) by category. Fraud overrides these.
CATEGORY_AMOUNT_USD: dict[ExpenseCategory, tuple[float, float]] = {
    ExpenseCategory.TRAVEL: (180, 1400),
    ExpenseCategory.MEALS: (14, 240),
    ExpenseCategory.SOFTWARE: (29, 1200),
    ExpenseCategory.OFFICE: (12, 400),
    ExpenseCategory.ADVERTISING: (300, 9000),
    ExpenseCategory.PROFESSIONAL_SERVICES: (500, 12000),
    ExpenseCategory.HARDWARE: (120, 3200),
    ExpenseCategory.UTILITIES: (80, 900),
    ExpenseCategory.RENT: (3000, 22000),
    ExpenseCategory.SHIPPING: (8, 320),
    ExpenseCategory.FUEL: (25, 140),
    ExpenseCategory.OTHER: (10, 500),
}

# MCCs grouped by the category they roll up to (inverse of MCC_TABLE).
_CATEGORY_MCCS: dict[ExpenseCategory, list[str]] = {}
for _mcc, (_label, _cat) in MCC_TABLE.items():
    _CATEGORY_MCCS.setdefault(_cat, []).append(_mcc)

# Merchant name fragments per category, so generated merchants read realistically.
MERCHANT_NAMES: dict[ExpenseCategory, list[str]] = {
    ExpenseCategory.SOFTWARE: ["Datadog", "Notion Labs", "Vercel", "Snowflake", "GitHub", "Figma", "Linear", "PagerDuty"],
    ExpenseCategory.HARDWARE: ["Apple Store", "Dell Technologies", "CDW", "Best Buy Biz", "Framework"],
    ExpenseCategory.TRAVEL: ["United Airlines", "Delta", "Marriott", "Hilton", "Uber", "Lyft", "Airbnb Biz"],
    ExpenseCategory.MEALS: ["Blue Bottle", "Sweetgreen", "DoorDash", "Chipotle", "Tartine", "Philz Coffee"],
    ExpenseCategory.ADVERTISING: ["Google Ads", "Meta Ads", "LinkedIn Mktg", "Reddit Ads", "TikTok Biz"],
    ExpenseCategory.PROFESSIONAL_SERVICES: ["Gunderson Dettmer", "Deloitte", "PwC", "Carta", "Cooley LLP"],
    ExpenseCategory.OFFICE: ["Staples", "Amazon Business", "WB Mason", "Costco Biz"],
    ExpenseCategory.UTILITIES: ["PG&E", "Comcast Biz", "AT&T Business"],
    ExpenseCategory.RENT: ["WeWork", "Industrious", "Hudson Pacific"],
    ExpenseCategory.SHIPPING: ["FedEx", "UPS", "Shippo", "USPS Biz"],
    ExpenseCategory.FUEL: ["Shell", "Chevron", "76 Station"],
    ExpenseCategory.OTHER: ["Square Vendor", "Stripe Vendor", "Misc Retail"],
}

# Merchants used to host fraud bursts — high-risk, card-not-present, foreign.
FRAUD_MERCHANT_NAMES = ["GiftCardVault", "CryptoTopUp Pro", "QuickPrepaid", "LuxeResell", "EverGadget Outlet"]

# Domestic "mule" merchants used for *stealth* fraud — US-based, business-hours,
# moderate amounts. These defeat the naive foreign/odd-hour heuristics, so the model
# must lean on device/ring/velocity structure to catch them.
DOMESTIC_MULE_NAMES = ["Peer2Peer Cash US", "MarketplaceX US", "QuickResale US", "PrepaidHub US"]

# A subset of *legitimate* merchants are foreign (overseas SaaS, international travel
# vendors), so "foreign" is a noisy fraud signal rather than a perfect separator.
FOREIGN_COUNTRIES = ("GB", "IE", "DE", "CA", "SG")

# Employees cluster into a handful of office metros; with several employees per metro
# the shared office IP is clearly *infrastructure* (high fan-out), not a ring link.
OFFICE_METROS = HOME_METROS[:6]


@dataclass
class GenSpec:
    """Knobs controlling tenant size and how much latent fraud / abuse to plant."""

    seed: int = 7
    company_name: str = "Northwind Robotics"
    industry: str = "Hardware / Robotics"
    n_employees: int = 42
    n_merchants: int = 64
    days: int = 90
    txns_per_employee_week: float = 11.0
    fraud_ring_count: int = 3
    lone_wolf_count: int = 12
    dispute_count: int = 22
    # Probability a given legit transaction carries each kind of policy violation.
    p_missing_receipt: float = 0.05
    p_over_limit: float = 0.015
    p_blocked_category: float = 0.01
    p_duplicate: float = 0.015
    cash_balance_usd: float = 4_200_000.0
    cash_apy: float = 0.043


class _Builder:
    """Stateful, seeded builder. One instance produces one tenant."""

    def __init__(self, spec: GenSpec):
        self.spec = spec
        self.rng = np.random.default_rng(spec.seed)
        self.company: Company
        self.policy: SpendPolicy
        self.employees: list[Employee] = []
        self.cards: list[StripeCard] = []
        self.cash_accounts: list[StripeTreasuryAccount] = []
        self.merchants: list[Merchant] = []
        self.fraud_merchants: list[Merchant] = []
        self.mule_merchants: list[Merchant] = []
        self.card_txns: list[CardTransaction] = []
        self.cash_txns: list[CashTransaction] = []
        self.disputes: list[Dispute] = []
        self._txn_seq = 0

    # ---- small helpers -------------------------------------------------------
    def _next_txn_id(self) -> str:
        self._txn_seq += 1
        return f"txn_{self._txn_seq:06d}"

    def _ts(self, day: int, *, business_hours: bool = True, weekend_ok: bool = False) -> datetime:
        base = END_DATE - timedelta(days=self.spec.days - day)
        if business_hours:
            hour = int(self.rng.integers(8, 19))
            minute = int(self.rng.integers(0, 60))
        else:  # odd hours used by fraud
            hour = int(self.rng.choice([0, 1, 2, 3, 4, 23]))
            minute = int(self.rng.integers(0, 60))
        ts = base.replace(hour=hour, minute=minute, second=int(self.rng.integers(0, 60)))
        if not weekend_ok and ts.weekday() >= 5:
            ts -= timedelta(days=2)
        return ts

    def _amount_cents(self, category: ExpenseCategory) -> int:
        lo, hi = CATEGORY_AMOUNT_USD[category]
        # Log-uniform so most spend is small with a realistic heavy tail.
        val = float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))
        return int(round(val * 100))

    # ---- entity construction -------------------------------------------------
    def build_company(self) -> None:
        s = self.spec
        self.company = Company(
            id="cmp_northwind",
            name=s.company_name,
            industry=s.industry,
            founded_year=2019,
            headcount=s.n_employees,
            monthly_card_limit_cents=250_000_00,
            cash_balance_cents=int(s.cash_balance_usd * 100),
            cash_apy=s.cash_apy,
        )
        self.policy = SpendPolicy(
            company_id=self.company.id,
            per_txn_limit_cents=5_000_00,
            receipt_required_over_cents=75_00,
            blocked_categories=[ExpenseCategory.FUEL],  # company has no fleet; fuel is out of policy
            category_budgets=[
                CategoryBudget(category=ExpenseCategory.MEALS, monthly_limit_cents=40_000_00),
                CategoryBudget(category=ExpenseCategory.TRAVEL, monthly_limit_cents=120_000_00),
                CategoryBudget(category=ExpenseCategory.ADVERTISING, monthly_limit_cents=200_000_00),
                CategoryBudget(category=ExpenseCategory.SOFTWARE, monthly_limit_cents=150_000_00),
            ],
            block_weekend_personal=True,
            approval_required_over_cents=10_000_00,
        )

    def build_cash_accounts(self) -> None:
        s = self.spec
        bal = int(s.cash_balance_usd * 100)
        self.cash_accounts = [
            StripeTreasuryAccount(id="cash_operating", company_id=self.company.id,
                            balance_cents=int(bal * 0.45), apy=0.0, account_type="operating"),
            StripeTreasuryAccount(id="cash_yield", company_id=self.company.id,
                            balance_cents=int(bal * 0.45), apy=s.cash_apy, account_type="yield"),
            StripeTreasuryAccount(id="cash_reserve", company_id=self.company.id,
                            balance_cents=int(bal * 0.10), apy=s.cash_apy * 0.6, account_type="reserve"),
        ]

    def build_employees_and_cards(self) -> None:
        s = self.spec
        first = ["James", "Oliver", "William", "Henry", "George", "Jack", "Harry", "Thomas",
                 "Edward", "Charlotte", "Emma", "Olivia", "Grace", "Alice", "Emily", "Sophie",
                 "Amelia", "Eleanor", "Benjamin", "Daniel"]
        last = ["Smith", "Johnson", "Williams", "Brown", "Taylor", "Davies", "Wilson", "Evans",
                "Walker", "Wright", "Thompson", "White", "Hughes", "Green", "Hall", "Harris",
                "Clarke", "Baker", "Turner", "Bennett"]
        # Department leads (managers) created first so reports can point at them.
        dept_lead: dict[str, str] = {}
        for i in range(s.n_employees):
            dept = DEPARTMENTS[i % len(DEPARTMENTS)] if i < len(DEPARTMENTS) else str(
                self.rng.choice(DEPARTMENTS))
            eid = f"emp_{i:04d}"
            is_lead = dept not in dept_lead
            role = (EmployeeRole.ADMIN if dept == "Finance" and is_lead
                    else EmployeeRole.MANAGER if is_lead
                    else EmployeeRole.EMPLOYEE)
            name = f"{first[i % len(first)]} {last[(i * 7) % len(last)]}"
            home = OFFICE_METROS[i % len(OFFICE_METROS)]
            emp = Employee(
                id=eid, company_id=self.company.id, name=name,
                email=f"{name.split()[0].lower()}.{name.split()[1].lower()}@northwind.co",
                role=role, department=dept,
                manager_id=dept_lead.get(dept), home_geo=home,
            )
            if is_lead:
                dept_lead[dept] = eid
            self.employees.append(emp)

            # Physical card for everyone; per-txn limit scales with seniority.
            per_txn = 10_000_00 if role in (EmployeeRole.ADMIN, EmployeeRole.MANAGER) else 5_000_00
            self.cards.append(StripeCard(
                id=f"card_{i:04d}p", company_id=self.company.id, employee_id=eid,
                type=CardType.PHYSICAL, last4=f"{int(self.rng.integers(1000, 9999))}",
                per_txn_limit_cents=per_txn, monthly_limit_cents=per_txn * 8,
            ))

    def build_merchants(self) -> None:
        s = self.spec
        cats = list(CATEGORY_AMOUNT_USD.keys())
        for i in range(s.n_merchants):
            cat = cats[i % len(cats)] if i < len(cats) else ExpenseCategory(str(self.rng.choice(
                [c.value for c in cats])))
            mccs = _CATEGORY_MCCS.get(cat) or ["5999"]
            mcc = str(self.rng.choice(mccs))
            names = MERCHANT_NAMES.get(cat, ["Vendor"])
            name = str(self.rng.choice(names))
            # Most merchants are clean; a meaningful minority carry genuinely
            # elevated chargeback / seller risk yet are used entirely in-policy — so
            # merchant risk overlaps both classes and can't separate fraud on its own.
            risk = float(np.clip(self.rng.beta(1.4, 12.0), 0, 1))
            if self.rng.random() < 0.16:
                risk = float(self.rng.uniform(0.28, 0.58))
            # ~12% of legitimate merchants are foreign (overseas SaaS / travel vendors),
            # so "foreign" overlaps both classes instead of perfectly flagging fraud.
            foreign = self.rng.random() < 0.12
            country = str(self.rng.choice(FOREIGN_COUNTRIES)) if foreign else "US"
            self.merchants.append(Merchant(
                id=f"mrc_{i:04d}", name=name, mcc=mcc, country=country,
                risk_score=round(risk + (0.05 if foreign else 0.0), 3)))
        # Foreign high-risk fraud-host merchants (gift cards, crypto top-ups, resale).
        for j, fname in enumerate(FRAUD_MERCHANT_NAMES):
            self.fraud_merchants.append(Merchant(
                id=f"mrc_fraud_{j:02d}", name=fname, mcc="5999",
                country=str(self.rng.choice(["NG", "RO", "SG", "GB", "BR"])),
                risk_score=round(float(self.rng.uniform(0.42, 0.95)), 3)))
        # Domestic mule merchants for stealth fraud (US, risk overlapping legit).
        for j, mname in enumerate(DOMESTIC_MULE_NAMES):
            self.mule_merchants.append(Merchant(
                id=f"mrc_mule_{j:02d}", name=mname, mcc="5999", country="US",
                risk_score=round(float(self.rng.uniform(0.20, 0.65)), 3)))
        self.merchants.extend(self.fraud_merchants)
        self.merchants.extend(self.mule_merchants)

    # ---- transaction streams -------------------------------------------------
    def _merchants_for_dept(self, dept: str) -> list[Merchant]:
        cats = set(DEPT_CATEGORIES.get(dept, []))
        pool = [m for m in self.merchants if m.id.startswith("mrc_0")
                and MCC_TABLE.get(m.mcc, ("", ExpenseCategory.OTHER))[1] in cats]
        return pool or [m for m in self.merchants if m.id.startswith("mrc_0")]

    def build_legit_transactions(self) -> None:
        s = self.spec
        n_weeks = max(1, s.days // 7)
        # Shared per-metro office egress IP: legit employees in the same metro
        # often share it (VPN / office network), so IP fan-out is a *noisy* fraud
        # signal rather than a perfect one.
        metro_office_ip = {m: f"172.16.{i}.1" for i, m in enumerate(HOME_METROS)}
        for emp, card in zip(self.employees, self.cards):
            device = f"dev_{emp.id}"  # stable, trusted device
            personal_ip = f"10.0.{int(emp.id[-2:]) if emp.id[-2:].isdigit() else 0}.{int(self.rng.integers(2, 254))}"
            office_ip = metro_office_ip.get(emp.home_geo, personal_ip)
            travels = emp.department in ("Sales", "Executive")
            pool = self._merchants_for_dept(emp.department)
            n_txn = int(self.rng.poisson(s.txns_per_employee_week * n_weeks))
            for _ in range(n_txn):
                day = int(self.rng.integers(0, s.days))
                merchant = pool[int(self.rng.integers(0, len(pool)))]
                category = MCC_TABLE.get(merchant.mcc, ("", ExpenseCategory.OTHER))[1]
                amount = self._amount_cents(category)
                weekend = bool(self.rng.random() < 0.12)
                # ~6% of legit spend is off-hours (on-call eng, overnight SaaS renewals)
                # so "odd hour" is not a clean fraud tell either.
                off_hours = self.rng.random() < 0.06
                ts = self._ts(day, business_hours=not off_hours, weekend_ok=weekend)
                # Most spend is home-metro on a shared/office or personal IP; Sales &
                # Executive legitimately travel, producing geo jumps on a TRUSTED device.
                geo = emp.home_geo
                ip = office_ip if self.rng.random() < 0.35 else personal_ip
                if travels and self.rng.random() < 0.15:
                    geo = HOME_METROS[(HOME_METROS.index(emp.home_geo) + 1 +
                                       int(self.rng.integers(0, len(HOME_METROS) - 1))) % len(HOME_METROS)]
                    ip = personal_ip
                channel = Channel.RECURRING if category == ExpenseCategory.SOFTWARE and self.rng.random() < 0.5 \
                    else (Channel.CARD_PRESENT if self.rng.random() < 0.45 else Channel.CARD_NOT_PRESENT)

                # Create violating *conditions* (the rule engine decides the labels).
                if self.rng.random() < s.p_over_limit:
                    amount = card.per_txn_limit_cents + self._amount_cents(category)
                has_receipt = not (self.rng.random() < s.p_missing_receipt
                                   and amount > self.policy.receipt_required_over_cents)
                if self.rng.random() < s.p_blocked_category:
                    fuel = [m for m in self.merchants if MCC_TABLE.get(m.mcc, ("", ExpenseCategory.OTHER))[1]
                            == ExpenseCategory.FUEL]
                    if fuel:
                        merchant = fuel[0]
                        category = ExpenseCategory.FUEL

                is_weekend = ts.weekday() >= 5
                violations = evaluate_policy(
                    amount_cents=amount, category=category, has_receipt=has_receipt,
                    is_weekend=is_weekend, card_per_txn_limit_cents=card.per_txn_limit_cents,
                    policy=self.policy)
                txn = CardTransaction(
                    id=self._next_txn_id(), company_id=self.company.id, card_id=card.id,
                    employee_id=emp.id, merchant_id=merchant.id, amount_cents=amount,
                    ts=ts, mcc=merchant.mcc, status=TxnStatus.SETTLED, channel=channel,
                    device_id=device, ip=ip, geo=geo,
                    memo=f"{merchant.name} - {category.value}", has_receipt=has_receipt,
                    ground_truth=GroundTruth(true_category=category, policy_violations=violations),
                )
                self.card_txns.append(txn)

                # Duplicate spend: emit a near-identical sibling minutes later.
                if self.rng.random() < s.p_duplicate:
                    dup = txn.model_copy(deep=True)
                    dup.id = self._next_txn_id()
                    dup.ts = ts + timedelta(minutes=int(self.rng.integers(1, 9)))
                    dup_is_weekend = dup.ts.weekday() >= 5
                    dup.ground_truth = GroundTruth(
                        true_category=category,
                        policy_violations=evaluate_policy(
                            amount_cents=amount, category=category, has_receipt=has_receipt,
                            is_weekend=dup_is_weekend, card_per_txn_limit_cents=card.per_txn_limit_cents,
                            policy=self.policy, is_duplicate=True),
                    )
                    self.card_txns.append(dup)

    def build_fraud(self) -> None:
        """Plant fraud rings (shared device/IP across cards) and lone-wolf fraud."""
        s = self.spec
        eligible = list(zip(self.employees, self.cards))

        # --- rings ---
        for ring_idx in range(s.fraud_ring_count):
            ring_id = f"ring_{ring_idx:02d}"
            device = f"dev_fraud_{ring_idx:02d}"
            ip = f"185.{int(self.rng.integers(10, 250))}.{int(self.rng.integers(1, 250))}.{int(self.rng.integers(2, 254))}"
            far_geo = FAR_METROS[ring_idx % len(FAR_METROS)]
            # ~40% of rings are "stealth": domestic mule merchants, business hours,
            # no geo jump — caught only by the shared-device graph structure.
            stealth = self.rng.random() < 0.4
            n_cards = int(self.rng.integers(2, 5))
            members = [eligible[i] for i in self.rng.choice(len(eligible), size=n_cards, replace=False)]
            base_day = int(self.rng.integers(5, s.days - 2))
            for emp, card in members:
                n_burst = int(self.rng.integers(3, 8))
                for _ in range(n_burst):
                    self.card_txns.append(self._fraud_txn(
                        emp, card, device, ip, ring_id, base_day, stealth, far_geo))

        # --- lone wolves: single compromised card, new device ---
        for _ in range(s.lone_wolf_count):
            emp, card = eligible[int(self.rng.integers(0, len(eligible)))]
            device = f"dev_lw_{int(self.rng.integers(0, 99999)):05d}"
            ip = f"45.{int(self.rng.integers(10, 250))}.{int(self.rng.integers(1, 250))}.{int(self.rng.integers(2, 254))}"
            far_geo = FAR_METROS[int(self.rng.integers(0, len(FAR_METROS)))]
            day = int(self.rng.integers(2, s.days))
            # ~30% of lone wolves are account-takeover: the fraudster rides the
            # cardholder's *own* trusted device & home geo (stolen session). These are
            # the hardest cases — almost indistinguishable from legitimate spend.
            ato = self.rng.random() < 0.45
            stealth = self.rng.random() < 0.35
            for _ in range(int(self.rng.integers(1, 4))):
                self.card_txns.append(self._fraud_txn(
                    emp, card, device, ip, None, day, stealth, far_geo, ato=ato))

        self.card_txns.sort(key=lambda t: t.ts)

    def _fraud_txn(self, emp, card, device, ip, ring_id, day, stealth, far_geo,
                   ato: bool = False) -> CardTransaction:
        if ato:
            device = f"dev_{emp.id}"  # trusted device — no novelty, no graph link
            ip = f"10.0.{int(emp.id[-2:]) if emp.id[-2:].isdigit() else 0}.{int(self.rng.integers(2, 254))}"
            # Half the time the fraudster even uses an ordinary in-policy merchant at a
            # normal ticket size — these are nearly impossible to distinguish and form
            # the irreducible tail that caps recall (as real-world ATO does).
            if self.rng.random() < 0.5:
                pool = self._merchants_for_dept(emp.department)
                merchant = pool[int(self.rng.integers(0, len(pool)))]
                category = MCC_TABLE.get(merchant.mcc, ("", ExpenseCategory.OTHER))[1]
                amount = self._amount_cents(category)
            else:
                merchant = self.mule_merchants[int(self.rng.integers(0, len(self.mule_merchants)))]
                amount = int(self.rng.integers(150, 900)) * 100
            ts = self._ts(day + int(self.rng.integers(0, 2)), business_hours=True, weekend_ok=True)
            return CardTransaction(
                id=self._next_txn_id(), company_id=self.company.id, card_id=card.id,
                employee_id=emp.id, merchant_id=merchant.id, amount_cents=amount,
                ts=ts, mcc=merchant.mcc, status=TxnStatus.SETTLED,
                channel=Channel.CARD_NOT_PRESENT, device_id=device, ip=ip, geo=emp.home_geo,
                memo=f"{merchant.name}", has_receipt=False,
                ground_truth=GroundTruth(is_fraud=True, fraud_ring_id=ring_id,
                                         true_category=ExpenseCategory.OTHER),
            )
        if stealth:
            merchant = self.mule_merchants[int(self.rng.integers(0, len(self.mule_merchants)))]
            amount = int(self.rng.integers(180, 1400)) * 100  # moderate, non-round
            ts = self._ts(day + int(self.rng.integers(0, 2)), business_hours=True, weekend_ok=True)
            geo = emp.home_geo  # no geo anomaly — only the device graph gives it away
        else:
            merchant = self.fraud_merchants[int(self.rng.integers(0, len(self.fraud_merchants)))]
            amount = int(self.rng.choice([250, 500, 750, 1000, 1500, 2000])) * 100  # round, high
            ts = self._ts(day + int(self.rng.integers(0, 2)), business_hours=False, weekend_ok=True)
            geo = far_geo
        return CardTransaction(
            id=self._next_txn_id(), company_id=self.company.id, card_id=card.id,
            employee_id=emp.id, merchant_id=merchant.id, amount_cents=amount,
            ts=ts, mcc=merchant.mcc, status=TxnStatus.SETTLED,
            channel=Channel.CARD_NOT_PRESENT, device_id=device, ip=ip, geo=geo,
            memo=f"{merchant.name}", has_receipt=False,
            ground_truth=GroundTruth(is_fraud=True, fraud_ring_id=ring_id,
                                     true_category=ExpenseCategory.OTHER),
        )

    def build_cash_transactions(self) -> None:
        """Stripe Treasury money movement, generated to look like a real operating account.

        Revenue arrives from several customers on their own jittered cadences (not one
        identical weekly spike), payroll and vendor bills land irregularly, and a couple
        of lumpy one-off events (an annual prepay in, an estimated-tax payment out)
        punctuate the series. That week-to-week irregularity is what keeps the
        reconstructed balance — and its forecast — from collapsing into a synthetic
        sawtooth.
        """
        s = self.spec
        op, yld = self.cash_accounts[0], self.cash_accounts[1]
        rng = self.rng
        day0 = (END_DATE - timedelta(days=s.days)).date()

        def at(day_idx, hour=11, minute=0):
            day_idx = max(0, min(s.days, int(day_idx)))
            return (END_DATE - timedelta(days=s.days - day_idx)).replace(hour=hour, minute=minute)

        # Daily nightly card-settlement sweep = sum of that day's settled card spend.
        by_day: dict[int, int] = {}
        for t in self.card_txns:
            if t.status == TxnStatus.SETTLED and not t.ground_truth.is_fraud:
                d = (t.ts.date() - day0).days
                by_day[d] = by_day.get(d, 0) + t.amount_cents
        for d, total in sorted(by_day.items()):
            self.cash_txns.append(CashTransaction(
                id=f"cash_sweep_{d:03d}", company_id=self.company.id, account_id=op.id,
                type=CashTxnType.CARD_SETTLEMENT, amount_cents=-total, ts=at(d, 23, 30),
                counterparty="Stripe Issuing Settlement", memo="Nightly card spend sweep"))

        # Semi-monthly payroll (~every 15 days, jittered) — the largest recurring outflow.
        payroll_out = 0
        d = 4 + int(rng.integers(0, 3))
        while d < s.days:
            amt = max(120_000, int(rng.normal(215_000, 16_000))) * 100
            payroll_out += amt
            self.cash_txns.append(CashTransaction(
                id=f"cash_payroll_{d:03d}", company_id=self.company.id, account_id=op.id,
                type=CashTxnType.VENDOR_PAYMENT, amount_cents=-amt, ts=at(d, 6),
                counterparty="Gusto Payroll", memo="Semi-monthly payroll"))
            d += 15 + int(rng.integers(-1, 2))

        # Vendor / bill pay on an irregular cadence with varied amounts.
        vendor_out, k, d = 0, 0, int(rng.integers(2, 6))
        while d < s.days:
            amt = int(rng.integers(25_000, 145_000)) * 100
            vendor_out += amt
            self.cash_txns.append(CashTransaction(
                id=f"cash_vendor_{k:02d}", company_id=self.company.id, account_id=op.id,
                type=CashTxnType.VENDOR_PAYMENT, amount_cents=-amt, ts=at(d, 11),
                counterparty=f"Vendor Bill #{500 + k}", memo="Bill pay"))
            d += int(rng.integers(5, 12)); k += 1

        # Customer revenue: several accounts, each paying on its own cadence and phase, so
        # ACH credits are spread across the week and vary run-to-run (no single big spike).
        out_total = sum(by_day.values()) + payroll_out + vendor_out
        weekly_target = out_total * 0.82 / max(1.0, s.days / 7.0)  # revenue < spend → modest burn
        cadences = [7, 7, 14, 15, 30]
        shares = rng.uniform(0.6, 1.4, len(cadences)); shares = shares / shares.sum()
        for c, (cad, share) in enumerate(zip(cadences, shares)):
            d = int(rng.integers(0, cad))
            while d < s.days:
                base = weekly_target * share * (cad / 7.0)         # bigger cheque, less often
                growth = 1.0 + 0.010 * (d / 7.0)                   # slight upward trend
                amt = max(400_000, int(base * growth * float(rng.uniform(0.78, 1.22))))
                self.cash_txns.append(CashTransaction(
                    id=f"cash_ach_{c}_{d:03d}", company_id=self.company.id, account_id=op.id,
                    type=CashTxnType.ACH_CREDIT, amount_cents=amt,
                    ts=at(d + int(rng.integers(0, 2)), 9),
                    counterparty=f"Customer {chr(65 + c)} Invoice", memo="Revenue ACH"))
                d += cad + int(rng.integers(-1, 2))

        # A couple of lumpy one-offs that make the curve look lived-in.
        self.cash_txns.append(CashTransaction(
            id="cash_prepay", company_id=self.company.id, account_id=op.id,
            type=CashTxnType.ACH_CREDIT, amount_cents=int(rng.integers(320_000, 520_000)) * 100,
            ts=at(int(s.days * 0.30), 14),
            counterparty="Enterprise Customer — Annual Prepay", memo="Annual contract prepayment"))
        self.cash_txns.append(CashTransaction(
            id="cash_tax", company_id=self.company.id, account_id=op.id,
            type=CashTxnType.VENDOR_PAYMENT, amount_cents=-int(rng.integers(170_000, 250_000)) * 100,
            ts=at(int(s.days * 0.62), 10),
            counterparty="IRS EFTPS", memo="Estimated quarterly tax"))

        # Monthly Stripe Treasury yield accrual on the yield account.
        for mo in range(max(1, s.days // 30)):
            accrual = int(yld.balance_cents * (yld.apy / 12))
            self.cash_txns.append(CashTransaction(
                id=f"cash_yield_{mo:02d}", company_id=self.company.id, account_id=yld.id,
                type=CashTxnType.YIELD_ACCRUAL, amount_cents=accrual, ts=at(mo * 30, 0, 5),
                counterparty="Stripe Treasury", memo=f"Yield accrual @ {yld.apy:.2%} APY"))
        self.cash_txns.sort(key=lambda t: t.ts)

    def build_disputes(self) -> None:
        s = self.spec
        fraud_txns = [t for t in self.card_txns if t.ground_truth.is_fraud]
        legit_txns = [t for t in self.card_txns
                      if not t.ground_truth.is_fraud and not t.ground_truth.policy_violations
                      and t.amount_cents > 50_00]
        n_fraud_disp = min(len(fraud_txns), int(s.dispute_count * 0.6))
        chosen_fraud = list(self.rng.choice(fraud_txns, size=n_fraud_disp, replace=False)) if fraud_txns else []
        n_friendly = s.dispute_count - n_fraud_disp
        chosen_legit = list(self.rng.choice(legit_txns, size=min(n_friendly, len(legit_txns)),
                                            replace=False)) if legit_txns else []

        di = 0
        for t in chosen_fraud:
            # True unauthorized fraud -> cardholder/issuer should win the chargeback.
            self.disputes.append(Dispute(
                id=f"dsp_{di:04d}", company_id=self.company.id, transaction_id=t.id,
                reason_code=DisputeReasonCode.FRAUD_CARD_NOT_PRESENT, amount_cents=t.amount_cents,
                status=DisputeStatus.OPEN, opened_ts=t.ts + timedelta(days=int(self.rng.integers(1, 6))),
                cardholder_statement=(
                    "I do not recognize this charge. I never shopped with this merchant and my "
                    "card was in my possession. Requesting a full chargeback."),
                ground_truth=GroundTruth(is_fraud=True, dispute_should_win=True),
            ))
            di += 1
        for t in chosen_legit:
            # Friendly fraud / buyer's remorse: evidence shows the cardholder transacted,
            # so on representment the *merchant* wins and the dispute should be denied.
            reason = DisputeReasonCode(str(self.rng.choice([
                DisputeReasonCode.NOT_RECOGNIZED.value,
                DisputeReasonCode.SUBSCRIPTION_CANCELED.value,
                DisputeReasonCode.PRODUCT_NOT_RECEIVED.value,
            ])))
            stmt = {
                DisputeReasonCode.NOT_RECOGNIZED: (
                    "I don't recognize this on my statement."),
                DisputeReasonCode.SUBSCRIPTION_CANCELED: (
                    "I thought I cancelled this subscription before the renewal."),
                DisputeReasonCode.PRODUCT_NOT_RECEIVED: (
                    "I haven't received the item yet."),
            }[reason]
            self.disputes.append(Dispute(
                id=f"dsp_{di:04d}", company_id=self.company.id, transaction_id=t.id,
                reason_code=reason, amount_cents=t.amount_cents, status=DisputeStatus.OPEN,
                opened_ts=t.ts + timedelta(days=int(self.rng.integers(2, 20))),
                cardholder_statement=stmt,
                ground_truth=GroundTruth(is_fraud=False, dispute_should_win=False),
            ))
            di += 1

    def build_subscriptions(self) -> None:
        """Recurring SaaS subscriptions: stable monthly charges on a fixed cadence.

        These create genuine recurring patterns the spend-intelligence layer detects,
        and shared vendors across employees become 'redundant license' consolidation
        opportunities.
        """
        soft = [m for m in self.merchants if m.id.startswith("mrc_0")
                and MCC_TABLE.get(m.mcc, ("", ExpenseCategory.OTHER))[1] == ExpenseCategory.SOFTWARE]
        if not soft:
            return
        n_months = max(1, self.spec.days // 30 + 1)
        for emp, card in zip(self.employees, self.cards):
            if self.rng.random() > 0.6:
                continue
            k = int(self.rng.integers(1, 4))
            chosen = [soft[int(i)] for i in self.rng.choice(len(soft), size=min(k, len(soft)),
                                                            replace=False)]
            for m in chosen:
                monthly = int(self.rng.choice([20, 49, 99, 150, 300, 500])) * 100
                anchor = int(self.rng.integers(1, 27))
                for mo in range(n_months):
                    day = mo * 30 + anchor
                    if day >= self.spec.days:
                        break
                    ts = (END_DATE - timedelta(days=self.spec.days - day)).replace(hour=3, minute=0)
                    self.card_txns.append(CardTransaction(
                        id=self._next_txn_id(), company_id=self.company.id, card_id=card.id,
                        employee_id=emp.id, merchant_id=m.id, amount_cents=monthly, ts=ts,
                        mcc=m.mcc, status=TxnStatus.SETTLED, channel=Channel.RECURRING,
                        device_id=f"dev_{emp.id}", ip="10.0.0.1", geo=emp.home_geo,
                        memo=f"{m.name} subscription", has_receipt=True,
                        ground_truth=GroundTruth(true_category=ExpenseCategory.SOFTWARE)))

    def build(self) -> Dataset:
        self.build_company()
        self.build_cash_accounts()
        self.build_employees_and_cards()
        self.build_merchants()
        self.build_legit_transactions()
        self.build_subscriptions()
        self.build_fraud()
        self.build_cash_transactions()
        self.build_disputes()
        return Dataset(
            company=self.company, policy=self.policy, employees=self.employees,
            cards=self.cards, cash_accounts=self.cash_accounts, merchants=self.merchants,
            card_transactions=self.card_txns, cash_transactions=self.cash_txns,
            disputes=self.disputes,
        )


def generate_tenant(spec: GenSpec | None = None, *, seed: int | None = None) -> Dataset:
    """Generate one deterministic synthetic Stripe tenant.

    >>> ds = generate_tenant(seed=7)
    >>> ds.summary()["fraud_transactions"] > 0
    True
    """
    if spec is None:
        spec = GenSpec(seed=seed if seed is not None else 7)
    elif seed is not None:
        spec = GenSpec(**{**spec.__dict__, "seed": seed})
    return _Builder(spec).build()
