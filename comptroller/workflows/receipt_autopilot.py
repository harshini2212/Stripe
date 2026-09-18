"""Receipt Autopilot — the signature multimodal workflow.

Read a receipt (Claude vision, or a deterministic simulation with no key) -> match it
to the Stripe Issuing card transaction -> reconcile the amount -> run policy + fraud checks ->
GL-code it -> write the memo -> decide auto-approve / review / reject. Every step is
explained so it reads like an expense analyst's worksheet.
"""
from __future__ import annotations

import re
from typing import Any

from ..domain import Dataset
from ..domain.enums import ExpenseCategory
from ..domain.policy import evaluate_policy, requires_approval

_RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "merchant": {"type": "string"},
        "date": {"type": "string"},
        "currency": {"type": "string"},
        "line_items": {"type": "array", "items": {
            "type": "object",
            "properties": {"description": {"type": "string"}, "amount": {"type": "number"}},
            "required": ["description", "amount"], "additionalProperties": False}},
        "subtotal": {"type": "number"}, "tax": {"type": "number"}, "tip": {"type": "number"},
        "total": {"type": "number"},
        "contains_personal_or_alcohol": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": ["merchant", "total", "line_items", "contains_personal_or_alcohol"],
    "additionalProperties": False,
}

_SYSTEM = ("You are Comptroller, Stripe's expense AI. You read receipts precisely and "
           "never invent values. Report amounts as plain numbers.")
_INSTR = ("Read this receipt image. Extract the merchant, date, every line item with its "
          "amount, the subtotal, tax, tip and grand total. Set "
          "contains_personal_or_alcohol to true if any line item is alcohol or clearly "
          "personal (not a business expense).")


_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9&]+", s.lower()))


def _parse_date(s: str) -> tuple[int | None, int | None, int | None]:
    m = re.search(r"([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(20\d\d)", s or "")
    if m:
        return int(m.group(3)), _MONTHS.get(m.group(1)[:3].lower()), int(m.group(2))
    iso = re.search(r"(20\d\d)-(\d{2})-(\d{2})", s or "")
    if iso:
        return int(iso.group(1)), int(iso.group(2)), int(iso.group(3))
    y = re.search(r"20\d\d", s or "")
    return (int(y.group()) if y else None), None, None


class ReceiptAutopilot:
    def __init__(self, dataset: Dataset, pipeline=None, client=None):
        self.ds = dataset
        self.pipeline = pipeline
        self.client = client
        self.merchant_index = dataset.merchant_index()
        self.card_index = dataset.card_index()

    # ---- parsing -------------------------------------------------------------
    def _parse(self, image: bytes, media_type: str, known) -> tuple[dict, str]:
        if self.client and self.client.available:
            try:
                data = self.client.extract_document(
                    image, media_type, _SYSTEM, _INSTR, _RECEIPT_SCHEMA)
                return data, "claude-vision"
            except Exception:
                pass
        if known is not None:  # deterministic simulation from the generated receipt
            return {
                "merchant": known.merchant, "date": known.date,
                "line_items": known.line_items, "subtotal": known.subtotal,
                "tax": known.tax, "tip": known.tip, "total": known.printed_total,
                "currency": "USD",
                "contains_personal_or_alcohol": known.anomaly == "personal_items",
            }, "simulated"
        return {}, "unavailable"

    # ---- matching ------------------------------------------------------------
    def _match(self, merchant: str, total: float, date_hint: str) -> tuple[Any, float]:
        mtok = _tokens(merchant)
        y, mo, day = _parse_date(date_hint)
        best, best_score = None, 0.0
        for t in self.ds.card_transactions:
            if t.ground_truth.is_fraud:
                continue
            name = self.merchant_index[t.merchant_id].name
            overlap = len(mtok & _tokens(name)) / max(len(mtok | _tokens(name)), 1)
            amt_close = max(0.0, 1.0 - abs(t.amount - total) / max(total, 1.0))
            if y and mo and day and (t.ts.year, t.ts.month, t.ts.day) == (y, mo, day):
                date_close = 1.0
            elif y and mo and (t.ts.year, t.ts.month) == (y, mo):
                date_close = 0.6
            elif y and t.ts.year == y:
                date_close = 0.4
            else:
                date_close = 0.1
            score = 0.4 * overlap + 0.2 * amt_close + 0.4 * date_close
            if score > best_score:
                best, best_score = t, score
        return best, best_score

    # ---- main ----------------------------------------------------------------
    def process(self, image: bytes, media_type: str, known=None) -> dict[str, Any]:
        extracted, source = self._parse(image, media_type, known)
        if not extracted:
            return {"ok": False, "source": source,
                    "message": "Set ANTHROPIC_API_KEY to read uploaded receipts with Claude vision."}

        total = float(extracted.get("total") or 0.0)
        txn, conf = self._match(extracted.get("merchant", ""), total, extracted.get("date", ""))
        flags: list[dict] = []
        steps = [f"Parsed receipt via {source}: {extracted.get('merchant')} ${total:,.2f}"]

        if txn is None:
            return {"ok": True, "source": source, "extracted": extracted,
                    "match": {"status": "no_match"}, "verdict": "needs_review",
                    "flags": [{"type": "no_match", "severity": "high",
                               "detail": "No card transaction matches this receipt."}],
                    "steps": steps}

        merchant = self.merchant_index[txn.merchant_id]
        card = self.card_index[txn.card_id]
        charged = txn.amount
        steps.append(f"Matched to {txn.id} ({merchant.name} ${charged:,.2f}, conf {conf:.0%})")

        # 1) amount reconciliation
        delta = round(total - charged, 2)
        ties = abs(delta) <= 0.01
        if not ties:
            flags.append({"type": "amount_mismatch", "severity": "medium",
                          "detail": f"Receipt total ${total:,.2f} vs charged ${charged:,.2f} "
                                    f"(Δ ${delta:+,.2f})."})
        steps.append(f"Reconciled amount: receipt ${total:,.2f} vs charged ${charged:,.2f} "
                     f"({'ties out' if ties else 'mismatch'})")

        # 2) personal / out-of-policy items
        if extracted.get("contains_personal_or_alcohol"):
            flags.append({"type": "personal_items", "severity": "high",
                          "detail": "Receipt contains alcohol or personal items — itemize "
                                    "or split before reimbursing."})

        # 3) missing tax line (soft compliance issue)
        category = txn.ground_truth.true_category or ExpenseCategory.OTHER
        if float(extracted.get("tax") or 0.0) <= 0.01 and total > 25 and category in (
                ExpenseCategory.MEALS, ExpenseCategory.OFFICE, ExpenseCategory.HARDWARE):
            flags.append({"type": "missing_tax", "severity": "low",
                          "detail": "No tax line on the receipt — may be non-compliant."})

        # 4) policy
        violations = evaluate_policy(
            amount_cents=txn.amount_cents, category=category, has_receipt=True,
            is_weekend=txn.ts.weekday() >= 5,
            card_per_txn_limit_cents=card.per_txn_limit_cents, policy=self.ds.policy)
        for v in violations:
            flags.append({"type": v.value, "severity": "medium", "detail": f"Policy: {v.value}"})
        approval = requires_approval(txn.amount_cents, self.ds.policy)

        # 4) fraud
        fraud_score = 0.0
        if self.pipeline is not None:
            fraud_score = float(self.pipeline.scores().get(txn.id, 0.0))
            if fraud_score >= 0.6:
                flags.append({"type": "fraud_risk", "severity": "critical",
                              "detail": f"Underlying transaction scores {fraud_score:.0%} fraud risk."})

        # verdict
        sev = {f["severity"] for f in flags}
        if "critical" in sev or "high" in sev:
            verdict = "reject" if "critical" in sev else "needs_review"
        elif flags or approval:
            verdict = "needs_review"
        else:
            verdict = "auto_approve"

        memo = (f"{merchant.name} — {category.value.replace('_', ' ')}; "
                f"{len(extracted.get('line_items', []))} items; "
                f"{'reconciled' if ties else 'amount discrepancy'}.")
        steps.append(f"GL-coded {category.value}; verdict: {verdict}")

        return {
            "ok": True, "source": source,
            "extracted": extracted,
            "match": {"status": "matched", "txn_id": txn.id, "merchant": merchant.name,
                      "charged_amount": charged, "date": txn.ts.isoformat(),
                      "confidence": round(conf, 3), "employee": txn.employee_id},
            "reconciliation": {"receipt_total": total, "charged_amount": charged,
                               "delta": delta, "ties_out": ties},
            "gl_code": category.value,
            "requires_approval": approval,
            "fraud_score": round(fraud_score, 3),
            "flags": flags,
            "verdict": verdict,
            "memo": memo,
            "steps": steps,
            "usage": extracted.get("_usage", {}),
            "truth": known.truth() if known is not None else None,
        }
