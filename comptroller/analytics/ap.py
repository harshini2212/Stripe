"""Accounts-Payable / Bill Pay intelligence.

The Stripe domain models card and cash activity; bills live in the AP workflow, so we
synthesize a deterministic invoice ledger from the tenant's vendors and analyze it:
duplicate / double-billed invoices, vendor-concentration (supply-chain) risk via an
HHI, and payment-timing optimization — hold cash in the Stripe Treasury yield account until the due
date, while capturing early-pay (2/10 net 30) discounts where they beat the yield.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from ..domain import Dataset

_MMF_APY = 0.041
# (vendor, category, base monthly $, terms) — one large vendor drives concentration.
_VENDORS = [
    ("AWS (cloud infra)", "software_saas", 82000, "net30"),
    ("Snowflake", "software_saas", 21000, "net30"),
    ("Gunderson Dettmer (legal)", "professional_services", 18000, "2/10net30"),
    ("Deloitte (audit)", "professional_services", 26000, "net30"),
    ("WeWork (facilities)", "rent_facilities", 31000, "net30"),
    ("Carta", "software_saas", 6400, "net30"),
    ("Rippling (payroll)", "professional_services", 14500, "net15"),
    ("Datadog", "software_saas", 9300, "2/10net30"),
    ("Ramp Contractors LLC", "professional_services", 12200, "net30"),
    ("Comcast Business", "utilities", 2100, "net30"),
    ("Iron Mountain", "office_supplies", 1750, "net30"),
    ("FedEx Freight", "shipping_logistics", 4300, "net30"),
]


@dataclass
class _Invoice:
    id: str
    vendor: str
    category: str
    amount_usd: float
    issue: date
    due: date
    terms: str
    status: str  # paid | scheduled | open


class APIntelligence:
    def __init__(self, dataset: Dataset, *, seed: int = 7):
        self.ds = dataset
        self.rng = np.random.default_rng(seed + 202)
        self.today = max((t.ts.date() for t in dataset.cash_transactions),
                         default=date(2026, 1, 1))
        self.invoices = self._build()

    def _terms_days(self, terms: str) -> int:
        return {"net15": 15, "net30": 30, "2/10net30": 30}.get(terms, 30)

    def _build(self) -> list[_Invoice]:
        inv: list[_Invoice] = []
        i = 0
        for vendor, cat, base, terms in _VENDORS:
            for m in range(3):  # last 3 monthly cycles
                issue = self.today - timedelta(days=m * 30 + int(self.rng.integers(0, 6)))
                due = issue + timedelta(days=self._terms_days(terms))
                amount = round(base * float(self.rng.uniform(0.92, 1.08)), 2)
                status = "paid" if due < self.today else (
                    "scheduled" if (due - self.today).days <= 7 else "open")
                inv.append(_Invoice(f"inv_{i:04d}", vendor, cat, amount, issue, due, terms, status))
                i += 1
        # Plant a few duplicate / double-billed invoices (same vendor + amount, days apart).
        for _ in range(4):
            src = inv[int(self.rng.integers(0, len(inv)))]
            dup = _Invoice(f"inv_{i:04d}", src.vendor, src.category, src.amount_usd,
                           src.issue + timedelta(days=int(self.rng.integers(1, 5))),
                           src.due + timedelta(days=int(self.rng.integers(1, 5))),
                           src.terms, "open")
            inv.append(dup)
            i += 1
        return inv

    def summary(self) -> dict:
        open_bills = [v for v in self.invoices if v.status in ("open", "scheduled")]
        due_7 = [v for v in open_bills if 0 <= (v.due - self.today).days <= 7]
        return {
            "total_invoices": len(self.invoices),
            "open_invoices": len(open_bills),
            "open_amount_usd": round(sum(v.amount_usd for v in open_bills), 2),
            "due_next_7d_usd": round(sum(v.amount_usd for v in due_7), 2),
            "as_of": self.today.isoformat(),
        }

    def duplicate_invoices(self) -> dict:
        seen: dict[tuple, _Invoice] = {}
        dups = []
        for v in sorted(self.invoices, key=lambda x: x.issue):
            key = (v.vendor, round(v.amount_usd, 2))
            if key in seen and abs((v.issue - seen[key].issue).days) <= 7:
                dups.append({"invoice": v.id, "duplicate_of": seen[key].id, "vendor": v.vendor,
                             "amount_usd": v.amount_usd, "issued": v.issue.isoformat()})
            else:
                seen[key] = v
        return {"count": len(dups),
                "exposure_usd": round(sum(d["amount_usd"] for d in dups), 2),
                "items": dups}

    def vendor_concentration(self) -> dict:
        by_vendor: dict[str, float] = {}
        for v in self.invoices:
            by_vendor[v.vendor] = by_vendor.get(v.vendor, 0.0) + v.amount_usd
        total = sum(by_vendor.values()) or 1.0
        shares = {k: val / total for k, val in by_vendor.items()}
        hhi = sum(s * s for s in shares.values())
        top = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
        flags = [{"vendor": k, "share": round(s, 4)} for k, s in top if s >= 0.20]
        level = "high" if hhi >= 0.18 else "moderate" if hhi >= 0.12 else "low"
        return {
            "hhi": round(hhi, 4),
            "concentration": level,
            "top_vendors": [{"vendor": k, "amount_usd": round(by_vendor[k], 2),
                             "share": round(s, 4)} for k, s in top[:6]],
            "single_source_flags": flags,
        }

    def payment_timing(self) -> dict:
        open_bills = [v for v in self.invoices if v.status in ("open", "scheduled")]
        float_yield = 0.0
        discount_capture = 0.0
        recs = []
        for v in open_bills:
            days_to_due = max((v.due - self.today).days, 0)
            held = v.amount_usd * _MMF_APY * (days_to_due / 365.0)
            float_yield += held
            if v.terms == "2/10net30" and (v.issue + timedelta(days=10)) >= self.today:
                discount = v.amount_usd * 0.02
                discount_capture += discount
                recs.append({"invoice": v.id, "vendor": v.vendor, "action": "pay_early_capture_discount",
                             "amount_usd": v.amount_usd, "discount_usd": round(discount, 2)})
            else:
                recs.append({"invoice": v.id, "vendor": v.vendor, "action": "pay_on_due_date",
                             "amount_usd": v.amount_usd, "due": v.due.isoformat()})
        return {
            "open_bills": len(open_bills),
            "float_yield_usd": round(float_yield, 2),
            "early_pay_discount_usd": round(discount_capture, 2),
            "recommendation": (
                f"Hold ${sum(v.amount_usd for v in open_bills):,.0f} in Stripe Treasury until due "
                f"(~${float_yield:,.0f} extra yield), and pre-pay 2/10-net-30 vendors to capture "
                f"~${discount_capture:,.0f} in discounts."),
            "items": recs[:12],
        }
