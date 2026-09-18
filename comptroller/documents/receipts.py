"""Render genuine receipt images from Stripe Issuing card transactions.

Each receipt is a real PNG (drawn with Pillow) that Claude vision actually reads. Some
carry planted anomalies so the autopilot's flagging is real, not theater:
  * ``amount_mismatch``  — printed total differs from what the card was charged
  * ``personal_items``   — an out-of-policy line item (alcohol) buried in the total
  * ``missing_tax``      — no tax line (common non-compliant receipt)
  * ``clean``            — everything ties out
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..domain import Dataset
from ..domain.enums import ExpenseCategory

# Plausible line items per category: (description, rough unit price).
_ITEMS: dict[ExpenseCategory, list[tuple[str, float]]] = {
    ExpenseCategory.MEALS: [("Cappuccino", 5.5), ("Avocado Toast", 14), ("Cobb Salad", 17),
                            ("Burger", 19), ("Sparkling Water", 4.5), ("Espresso", 4),
                            ("Margherita Pizza", 21), ("Side Fries", 8)],
    ExpenseCategory.TRAVEL: [("Airfare SFO-JFK", 410), ("Hotel night", 289), ("Rideshare", 38),
                             ("Baggage fee", 35), ("Airport parking", 48)],
    ExpenseCategory.SOFTWARE: [("Pro plan - monthly", 99), ("Seat license", 49),
                               ("API usage overage", 120)],
    ExpenseCategory.HARDWARE: [("USB-C dock", 189), ("Mechanical keyboard", 145),
                               ("27in monitor", 329), ("Webcam", 89)],
    ExpenseCategory.OFFICE: [("Printer paper", 32), ("Notebooks (12)", 24), ("Whiteboard markers", 18),
                             ("Standing mat", 59)],
    ExpenseCategory.ADVERTISING: [("Campaign spend", 1200), ("Creative assets", 450)],
    ExpenseCategory.PROFESSIONAL_SERVICES: [("Advisory hours", 1500), ("Filing fee", 425)],
}
_PERSONAL = [("Glass of Cabernet", 16), ("Bottle of wine", 62), ("Cocktail", 18)]


@dataclass
class GeneratedReceipt:
    receipt_id: str
    txn_id: str
    merchant: str
    date: str
    category: str
    line_items: list[dict] = field(default_factory=list)
    subtotal: float = 0.0
    tax: float = 0.0
    tip: float = 0.0
    printed_total: float = 0.0   # what the receipt shows
    charged_amount: float = 0.0  # what the card was actually charged (the txn)
    anomaly: str = "clean"
    png: bytes = b""

    def truth(self) -> dict:
        """Ground truth for evaluating the autopilot (never shown to the model)."""
        return {"txn_id": self.txn_id, "printed_total": self.printed_total,
                "charged_amount": self.charged_amount, "anomaly": self.anomaly,
                "has_personal_items": self.anomaly == "personal_items"}


def _font(size: int):
    for path in (r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\cour.ttf",
                 r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _items_for(category: ExpenseCategory, subtotal: float, rng) -> list[dict]:
    pool = _ITEMS.get(category) or _ITEMS[ExpenseCategory.OFFICE]
    k = int(rng.integers(1, min(4, len(pool)) + 1))
    chosen = [pool[int(i)] for i in rng.choice(len(pool), size=k, replace=False)]
    weights = np.array([p for _, p in chosen], dtype=float)
    weights = weights / weights.sum()
    items = []
    allocated = 0.0
    for i, (desc, _) in enumerate(chosen):
        amt = round(subtotal * float(weights[i]), 2) if i < k - 1 else round(subtotal - allocated, 2)
        allocated += amt
        items.append({"description": desc, "amount": amt})
    return items


def _render(r: GeneratedReceipt) -> bytes:
    W = 460
    head = _font(26)
    body = _font(20)
    small = _font(16)
    lines = 9 + len(r.line_items) + (1 if r.tip else 0)
    H = 120 + lines * 28
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    y = 22

    def center(text, font, yy, fill="black"):
        w = d.textlength(text, font=font)
        d.text(((W - w) / 2, yy), text, font=font, fill=fill)

    def row(left, right, yy, font=body, fill="black"):
        d.text((24, yy), left, font=font, fill=fill)
        rw = d.textlength(right, font=font)
        d.text((W - 24 - rw, yy), right, font=font, fill=fill)

    center(r.merchant.upper(), head, y); y += 38
    center(r.date, small, y, "#555"); y += 26
    d.line((24, y, W - 24, y), fill="#999"); y += 14
    for it in r.line_items:
        row(it["description"][:28], f"${it['amount']:.2f}", y); y += 28
    d.line((24, y, W - 24, y), fill="#ccc"); y += 12
    row("Subtotal", f"${r.subtotal:.2f}", y, small, "#333"); y += 26
    if r.tax or r.anomaly != "missing_tax":
        row("Tax", f"${r.tax:.2f}", y, small, "#333"); y += 26
    if r.tip:
        row("Tip", f"${r.tip:.2f}", y, small, "#333"); y += 26
    d.line((24, y, W - 24, y), fill="#999"); y += 12
    row("TOTAL", f"${r.printed_total:.2f}", y, body); y += 34
    center("Thank you!", small, y, "#777")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_receipt(txn, dataset: Dataset, *, anomaly: str = "clean",
                     rng=None) -> GeneratedReceipt:
    rng = rng or np.random.default_rng(abs(hash(txn.id)) % (2**32))
    merchant = dataset.merchant_index()[txn.merchant_id].name
    category = txn.ground_truth.true_category or ExpenseCategory.OTHER
    charged = txn.amount

    is_meal = category == ExpenseCategory.MEALS
    tax_rate = 0.0 if anomaly == "missing_tax" else 0.085
    tip_rate = 0.18 if is_meal else 0.0
    # Work backwards from the charged amount so a clean receipt ties out exactly.
    subtotal = round(charged / (1 + tax_rate + tip_rate), 2)
    items = _items_for(category, subtotal, rng)
    if anomaly == "personal_items":  # one line is alcohol/personal; total still ties out
        idx = int(rng.integers(0, len(items)))
        items[idx] = {"description": _PERSONAL[int(rng.integers(0, len(_PERSONAL)))][0],
                      "amount": items[idx]["amount"]}
    tax = round(subtotal * tax_rate, 2)
    tip = round(subtotal * tip_rate, 2)
    printed_total = round(subtotal + tax + tip, 2)
    if anomaly == "amount_mismatch":  # tip added after the receipt printed, etc.
        printed_total = round(charged * float(rng.choice([0.82, 0.88, 1.0])), 2)
        if printed_total == charged:
            printed_total = round(charged - 7.50, 2)

    r = GeneratedReceipt(
        receipt_id=f"rcpt_{txn.id.split('_')[-1]}", txn_id=txn.id, merchant=merchant,
        date=txn.ts.strftime("%b %d, %Y  %I:%M %p"), category=category.value,
        line_items=items, subtotal=subtotal, tax=tax, tip=tip,
        printed_total=printed_total, charged_amount=charged, anomaly=anomaly)
    r.png = _render(r)
    return r


def build_sample_receipts(dataset: Dataset, *, seed: int = 7) -> list[GeneratedReceipt]:
    """A demo set: clean receipts plus one of each anomaly, tied to real transactions."""
    rng = np.random.default_rng(seed + 909)
    pool = [t for t in dataset.card_transactions
            if not t.ground_truth.is_fraud and t.ground_truth.true_category
            in (ExpenseCategory.MEALS, ExpenseCategory.TRAVEL, ExpenseCategory.SOFTWARE,
                ExpenseCategory.HARDWARE, ExpenseCategory.OFFICE)
            and t.amount > 25]
    plan = ["clean", "clean", "amount_mismatch", "personal_items", "missing_tax", "clean"]
    out = []
    picks = rng.choice(len(pool), size=min(len(plan), len(pool)), replace=False)
    for anomaly, i in zip(plan, picks):
        out.append(generate_receipt(pool[int(i)], dataset, anomaly=anomaly, rng=rng))
    return out
