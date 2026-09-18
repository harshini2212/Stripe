"""Eval task definitions: golden-case builders + per-task scoring.

Each task pulls held-out ground truth from the synthetic tenant and turns it into
``(inputs, expected)`` cases the agents are graded on. The five tasks span the kinds
of correctness a Stripe finance org actually cares about:

* ``categorization``  — exact-match accuracy on GL coding
* ``policy_audit``    — set-F1 over policy violations + approval routing
* ``dispute``         — chargeback adjudication (true vs friendly fraud)
* ``fraud_triage``    — fraud decision from ML signal
* ``tieout``          — numeric reconciliation to the cent
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..agents import inputs as build
from ..agents.categorization import CategorizationAgent, recoverable_from_name
from ..agents.dispute import DisputeAgent
from ..agents.fraud_triage import FraudTriageAgent
from ..agents.policy import PolicyAuditAgent
from ..agents.tieout import TieoutAgent
from ..domain import Dataset
from .scorers import numeric_within, set_prf


@dataclass
class EvalCase:
    task: str
    case_id: str
    inputs: dict[str, Any]
    expected: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


class EvalTask:
    name: str = "task"

    def __init__(self):
        self.agent = None

    def build_cases(self, ds: Dataset, pipeline, rng: np.random.Generator,
                    limit: int) -> list[EvalCase]:  # pragma: no cover
        raise NotImplementedError

    def score(self, pred: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
class CategorizationTask(EvalTask):
    name = "categorization"

    def __init__(self):
        self.agent = CategorizationAgent()

    def build_cases(self, ds, pipeline, rng, limit) -> list[EvalCase]:
        pool = [t for t in ds.card_transactions
                if not t.ground_truth.is_fraud and t.ground_truth.true_category]
        idx = rng.choice(len(pool), size=min(limit, len(pool)), replace=False)
        merchant_index = ds.merchant_index()
        cases = []
        for i in idx:
            t = pool[int(i)]
            name = merchant_index[t.merchant_id].name
            miscode = recoverable_from_name(name) and bool(rng.random() < 0.5)
            cases.append(EvalCase(
                self.name, t.id,
                build.categorization_inputs(t, ds, miscode=miscode),
                {"category": t.ground_truth.true_category.value},
                {"miscoded": miscode},
            ))
        return cases

    def score(self, pred, expected) -> dict:
        return {"correct": pred.get("category") == expected["category"]}


# --------------------------------------------------------------------------- #
class PolicyAuditTask(EvalTask):
    name = "policy_audit"

    def __init__(self):
        self.agent = PolicyAuditAgent()

    def build_cases(self, ds, pipeline, rng, limit) -> list[EvalCase]:
        nonfraud = [t for t in ds.card_transactions if not t.ground_truth.is_fraud]
        violators = [t for t in nonfraud if t.ground_truth.policy_violations]
        clean = [t for t in nonfraud if not t.ground_truth.policy_violations]
        k_v = min(len(violators), max(1, limit // 2))
        k_c = min(len(clean), limit - k_v)
        chosen = ([violators[int(i)] for i in rng.choice(len(violators), k_v, replace=False)]
                  + [clean[int(i)] for i in rng.choice(len(clean), k_c, replace=False)])
        cases = []
        for t in chosen:
            expected = {
                "violations": sorted(v.value for v in t.ground_truth.policy_violations),
                "requires_approval": t.amount_cents > ds.policy.approval_required_over_cents,
            }
            cases.append(EvalCase(self.name, t.id, build.policy_inputs(t, ds), expected))
        return cases

    def score(self, pred, expected) -> dict:
        prf = set_prf(pred.get("violations", []), expected["violations"])
        approval_ok = bool(pred.get("requires_approval")) == expected["requires_approval"]
        return {
            "correct": prf["exact"] and approval_ok,
            "viol_tp": prf["tp"], "viol_fp": prf["fp"], "viol_fn": prf["fn"],
            "approval_ok": approval_ok,
        }


# --------------------------------------------------------------------------- #
class DisputeTask(EvalTask):
    name = "dispute"

    def __init__(self):
        self.agent = DisputeAgent()

    def build_cases(self, ds, pipeline, rng, limit) -> list[EvalCase]:
        scores = pipeline.scores()
        cases = []
        for d in ds.disputes[:limit]:
            txn = ds.txn_index().get(d.transaction_id)
            if txn is None:
                continue
            fs = float(scores.get(txn.id, 0.0))
            cases.append(EvalCase(
                self.name, d.id,
                build.dispute_inputs(d, ds, fraud_score=fs),
                {"cardholder_should_win": bool(d.ground_truth.dispute_should_win)},
            ))
        return cases

    def score(self, pred, expected) -> dict:
        return {"correct": bool(pred.get("cardholder_should_win"))
                == expected["cardholder_should_win"]}


# --------------------------------------------------------------------------- #
class FraudTriageTask(EvalTask):
    name = "fraud_triage"

    def __init__(self):
        self.agent = FraudTriageAgent()

    def build_cases(self, ds, pipeline, rng, limit) -> list[EvalCase]:
        fraud = [t for t in ds.card_transactions if t.ground_truth.is_fraud]
        legit = [t for t in ds.card_transactions if not t.ground_truth.is_fraud]
        k_f = min(len(fraud), max(1, limit // 3))
        k_l = min(len(legit), limit - k_f)
        chosen = ([fraud[int(i)] for i in rng.choice(len(fraud), k_f, replace=False)]
                  + [legit[int(i)] for i in rng.choice(len(legit), k_l, replace=False)])
        cases = []
        for t in chosen:
            a = pipeline.assess(t.id, with_drivers=True)
            cases.append(EvalCase(
                self.name, t.id,
                build.triage_inputs(t, a, ds),
                {"is_fraud": bool(t.ground_truth.is_fraud)},
            ))
        return cases

    def score(self, pred, expected) -> dict:
        return {"correct": bool(pred.get("is_fraud")) == expected["is_fraud"]}


# --------------------------------------------------------------------------- #
class TieoutTask(EvalTask):
    name = "tieout"

    def __init__(self):
        self.agent = TieoutAgent()

    def build_cases(self, ds, pipeline, rng, limit) -> list[EvalCase]:
        merchant_index = ds.merchant_index()
        by_emp: dict[str, list] = {}
        for t in ds.card_transactions:
            if not t.ground_truth.is_fraud:
                by_emp.setdefault(t.employee_id, []).append(t)
        emps = [e for e in ds.employees if len(by_emp.get(e.id, [])) >= 4]
        cases = []
        for n in range(limit):
            emp = emps[int(rng.integers(0, len(emps)))]
            txns = by_emp[emp.id]
            k = int(rng.integers(4, min(12, len(txns)) + 1))
            picks = [txns[int(i)] for i in rng.choice(len(txns), k, replace=False)]
            line_items = [{"merchant": merchant_index[t.merchant_id].name,
                           "amount_usd": t.amount} for t in picks]
            true_cents = sum(t.amount_cents for t in picks)
            submitted_cents = self._submitted(true_cents, picks, rng)
            cases.append(EvalCase(
                self.name, f"report_{n:03d}",
                {"employee": emp.name, "period": "trailing-30d",
                 "line_items": line_items, "submitted_total_usd": submitted_cents / 100.0},
                {"computed_total_usd": round(true_cents / 100.0, 2),
                 "ties_out": submitted_cents == true_cents,
                 "discrepancy_usd": round((submitted_cents - true_cents) / 100.0, 2)},
                {"error_kind": "none" if submitted_cents == true_cents else "injected"},
            ))
        return cases

    @staticmethod
    def _submitted(true_cents: int, picks, rng) -> int:
        if rng.random() < 0.5:
            return true_cents  # ties out
        kind = int(rng.integers(0, 4))
        amt = picks[int(rng.integers(0, len(picks)))].amount_cents
        if kind == 0:       # a line omitted from the stated total
            return true_cents - amt
        if kind == 1:       # a line double-counted
            return true_cents + amt
        if kind == 2:       # transposed digits in the stated total
            return true_cents + int(rng.choice([900, 9000, -900, 90]))
        return true_cents + int(rng.choice([1, 7, -3, 50]))  # fat-finger

    def score(self, pred, expected) -> dict:
        total_ok = numeric_within(pred.get("computed_total_usd"), expected["computed_total_usd"])
        ties_ok = bool(pred.get("ties_out")) == expected["ties_out"]
        abs_err = abs(float(pred.get("computed_total_usd", 0.0)) - expected["computed_total_usd"])
        return {"correct": total_ok and ties_ok, "total_ok": total_ok,
                "ties_ok": ties_ok, "abs_error_usd": abs_err}


def build_tasks(names: list[str] | None = None) -> list[EvalTask]:
    """Instantiate the eval task suite (optionally a subset by name)."""
    registry = {
        "categorization": CategorizationTask,
        "policy_audit": PolicyAuditTask,
        "dispute": DisputeTask,
        "fraud_triage": FraudTriageTask,
        "tieout": TieoutTask,
    }
    chosen = names or list(registry)
    return [registry[n]() for n in chosen if n in registry]
