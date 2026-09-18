"""Deterministic AP ledger — purchase orders + invoices with planted anomalies.

Built so the AP agent has real work: most invoices match a PO, but one exceeds its PO
amount, one is a duplicate (double-billed), and one arrives with CHANGED vendor bank
details (vendor-impersonation fraud) vs. the vendor's payment history.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np

_VENDORS = [
    ("AWS (cloud infra)", 82000, "ACH-8841"),
    ("Snowflake", 21000, "ACH-2207"),
    ("Gunderson Dettmer", 18000, "ACH-5532"),
    ("Deloitte (audit)", 26000, "ACH-9014"),
    ("WeWork (facilities)", 31000, "ACH-3380"),
    ("Rippling (payroll)", 14500, "ACH-7765"),
    ("Datadog", 9300, "ACH-1190"),
    ("Iron Mountain", 1750, "ACH-4421"),
]


@dataclass
class PO:
    id: str
    vendor: str
    amount: float
    date: str


@dataclass
class Invoice:
    id: str
    vendor: str
    amount: float
    issue: str
    due: str
    po_id: str | None
    bank_account: str
    status: str = "open"   # open | paid | held
    anomaly: str = "clean"  # clean | over_po | duplicate | bank_changed | unmatched


@dataclass
class APLedger:
    pos: list[PO]
    invoices: list[Invoice]
    vendor_bank: dict[str, str]  # canonical bank per vendor (the source of truth)
    invoices_by_id: dict[str, Invoice] = field(default_factory=dict)
    pos_by_id: dict[str, PO] = field(default_factory=dict)

    def index(self) -> "APLedger":
        self.invoices_by_id = {i.id: i for i in self.invoices}
        self.pos_by_id = {p.id: p for p in self.pos}
        return self


def build_ap_ledger(*, seed: int = 7, today: date | None = None) -> APLedger:
    rng = np.random.default_rng(seed + 404)
    today = today or date(2026, 1, 1)
    vendor_bank = {v: bank for v, _, bank in _VENDORS}
    pos: list[PO] = []
    invoices: list[Invoice] = []
    pi = 0
    for vi, (vendor, base, bank) in enumerate(_VENDORS):
        amount = round(base * float(rng.uniform(0.95, 1.05)), 2)
        po = PO(f"PO-{1000 + vi}", vendor, round(amount * 1.02, 2),
                (today - timedelta(days=int(rng.integers(25, 55)))).isoformat())
        pos.append(po)
        issue = today - timedelta(days=int(rng.integers(3, 20)))
        invoices.append(Invoice(f"INV-{2000 + pi}", vendor, amount, issue.isoformat(),
                                (issue + timedelta(days=30)).isoformat(), po.id, bank))
        pi += 1

    # Plant anomalies (tied to real vendors).
    # 1) an invoice OVER its PO amount.
    over_po = pos[0]
    issue = today - timedelta(days=6)
    invoices.append(Invoice(f"INV-{2000 + pi}", over_po.vendor, round(over_po.amount * 1.18, 2),
                            issue.isoformat(), (issue + timedelta(days=30)).isoformat(),
                            over_po.id, vendor_bank[over_po.vendor], anomaly="over_po")); pi += 1
    # 2) a duplicate of an existing invoice (double-billed).
    src = invoices[1]
    invoices.append(Invoice(f"INV-{2000 + pi}", src.vendor, src.amount,
                            (date.fromisoformat(src.issue) + timedelta(days=3)).isoformat(),
                            src.due, src.po_id, src.bank_account, anomaly="duplicate")); pi += 1
    # 3) an invoice with CHANGED bank details (impersonation fraud).
    bad = pos[3]
    issue = today - timedelta(days=4)
    invoices.append(Invoice(f"INV-{2000 + pi}", bad.vendor, round(bad.amount * 0.99, 2),
                            issue.isoformat(), (issue + timedelta(days=30)).isoformat(),
                            bad.id, "ACH-0000-NEW", anomaly="bank_changed")); pi += 1
    # 4) an unmatched invoice (no PO).
    issue = today - timedelta(days=8)
    invoices.append(Invoice(f"INV-{2000 + pi}", "Brightspark Media", 12400.0,
                            issue.isoformat(), (issue + timedelta(days=30)).isoformat(),
                            None, "ACH-7781", anomaly="unmatched")); pi += 1

    return APLedger(pos, invoices, vendor_bank).index()
