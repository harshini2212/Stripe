"""A small tool surface the multi-step agents call, with a recorded call log.

Keeping the queries behind an explicit ``Toolbox`` mirrors how a real Stripe agent would
be wired: each method is an auditable "tool call", and the log becomes the agent's
investigation trail (and, with a live model, maps directly onto Claude tool-use).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from ..domain import Dataset


@dataclass
class Toolbox:
    dataset: Dataset
    pipeline: Any = None  # FraudPipeline (optional)
    calls: list[str] = field(default_factory=list)

    def _log(self, name: str, **kwargs: Any) -> None:
        args = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        self.calls.append(f"{name}({args})")

    # ---- lookups -------------------------------------------------------------
    def get_transaction(self, txn_id: str):
        self._log("get_transaction", txn_id=txn_id)
        return self.dataset.txn_index().get(txn_id)

    def get_merchant(self, merchant_id: str):
        self._log("get_merchant", merchant_id=merchant_id)
        return self.dataset.merchant_index().get(merchant_id)

    def get_employee(self, employee_id: str):
        self._log("get_employee", employee_id=employee_id)
        return self.dataset.employee_index().get(employee_id)

    # ---- behavioral / velocity ----------------------------------------------
    def card_velocity(self, card_id: str, ts, hours: int = 72) -> dict[str, Any]:
        self._log("card_velocity", card_id=card_id, hours=hours)
        lo = ts - timedelta(hours=hours)
        window = [t for t in self.dataset.card_transactions
                  if t.card_id == card_id and lo <= t.ts <= ts]
        return {
            "count": len(window),
            "sum_usd": round(sum(t.amount_cents for t in window) / 100.0, 2),
            "distinct_merchants": len({t.merchant_id for t in window}),
            "distinct_geos": len({t.geo for t in window if t.geo}),
        }

    # ---- fraud / graph -------------------------------------------------------
    def fraud_assessment(self, txn_id: str):
        self._log("fraud_assessment", txn_id=txn_id)
        if self.pipeline is None:
            return None
        return self.pipeline.assess(txn_id, with_drivers=True)

    def ring_for_card(self, card_id: str):
        self._log("ring_for_card", card_id=card_id)
        if self.pipeline is None:
            return None
        for ring in self.pipeline.rings():
            if card_id in ring.card_ids:
                return ring
        return None
