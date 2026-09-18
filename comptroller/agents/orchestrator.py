"""ComptrollerOrchestrator — the autonomous agent that runs the whole desk.

For any Stripe Issuing card transaction it sequences the specialist agents — categorize, audit
policy, score & triage fraud — and, when the risk is high, escalates to a full
FraudInvestigator workflow and resolves any attached dispute. Every step runs through
the *same* pluggable backend, so the identical autonomous workflow executes on the
deterministic engine, a simulated model, or live Claude. It returns one consolidated
decision with a financial-impact tally and a full reasoning trace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..domain import Dataset
from ..domain.enums import RiskBand
from ..llm.base import Backend, Usage
from . import inputs as build
from .categorization import CategorizationAgent
from .dispute import DisputeAgent
from .fraud_triage import FraudTriageAgent
from .investigator import FraudInvestigator
from .policy import PolicyAuditAgent


@dataclass
class OrchestratorDecision:
    txn_id: str
    backend: str
    category: str
    policy_violations: list[str]
    requires_approval: bool
    fraud_score: float
    fraud_band: str
    is_fraud: bool
    recommended_actions: list[str]
    financial_impact_usd: float
    dispute: dict[str, Any] | None
    latency_ms: float
    usage: dict[str, Any]
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "txn_id": self.txn_id,
            "backend": self.backend,
            "category": self.category,
            "policy_violations": self.policy_violations,
            "requires_approval": self.requires_approval,
            "fraud_score": round(self.fraud_score, 4),
            "fraud_band": self.fraud_band,
            "is_fraud": self.is_fraud,
            "recommended_actions": self.recommended_actions,
            "financial_impact_usd": round(self.financial_impact_usd, 2),
            "dispute": self.dispute,
            "latency_ms": round(self.latency_ms, 1),
            "usage": self.usage,
            "trace": self.trace,
        }


class ComptrollerOrchestrator:
    def __init__(self, dataset: Dataset, pipeline, backend: Backend):
        self.dataset = dataset
        self.pipeline = pipeline
        self.backend = backend
        self.cat = CategorizationAgent()
        self.policy = PolicyAuditAgent()
        self.triage = FraudTriageAgent()
        self.dispute_agent = DisputeAgent()
        self.investigator = FraudInvestigator(dataset, pipeline)
        self._disputes = {d.transaction_id: d for d in dataset.disputes}

    def handle_transaction(self, txn_id: str) -> OrchestratorDecision:
        txn = self.dataset.txn_index()[txn_id]
        trace: list[str] = [f"Comptroller agent engaged on {txn_id} via backend "
                            f"'{self.backend.name}'."]
        total_usage = Usage()
        total_latency = 0.0

        # 1) Categorize.
        r_cat = self.backend.run(self.cat, build.categorization_inputs(txn, self.dataset))
        total_usage += r_cat.usage
        total_latency += r_cat.latency_ms
        category = r_cat.data.get("category", "other")
        trace.append(f"[categorization] -> {category} (conf {r_cat.data.get('confidence', 0):.2f})")

        # 2) Policy audit.
        r_pol = self.backend.run(self.policy, build.policy_inputs(txn, self.dataset))
        total_usage += r_pol.usage
        total_latency += r_pol.latency_ms
        violations = r_pol.data.get("violations", [])
        approval = bool(r_pol.data.get("requires_approval", False))
        trace.append(f"[policy_audit] -> violations={violations or 'none'}, "
                     f"requires_approval={approval}")

        # 3) Fraud score + triage.
        assessment = self.pipeline.assess(txn_id, with_drivers=True)
        r_tri = self.backend.run(self.triage, build.triage_inputs(txn, assessment, self.dataset))
        total_usage += r_tri.usage
        total_latency += r_tri.latency_ms
        is_fraud = bool(r_tri.data.get("is_fraud", False))
        action = r_tri.data.get("recommended_action", "monitor")
        trace.append(f"[fraud_triage] risk={assessment.risk_score:.2f} "
                     f"({assessment.risk_band.value}) -> is_fraud={is_fraud}, action={action}")

        # 4) Escalate to the full investigation workflow when warranted.
        actions: list[str] = []
        if assessment.risk_band in (RiskBand.HIGH, RiskBand.CRITICAL) or is_fraud:
            report = self.investigator.investigate(txn_id)
            actions = list(report.recommended_actions)
            trace.append(f"[escalation] FraudInvestigator ran {len(report.steps)} tool calls; "
                         + (f"ring {report.ring_id} "
                            f"({len(report.ring_member_cards)} cards) implicated."
                            if report.ring_id else "no ring linkage."))
        else:
            actions = [action.replace("_", " ")]

        # Policy-driven actions.
        if violations:
            actions.append(f"Flag for controller review: {', '.join(violations)}")
        if approval:
            actions.append("Route for pre-approval (over approval threshold)")

        # 5) Resolve an attached dispute, if any.
        dispute_out: dict[str, Any] | None = None
        impact = txn.amount if is_fraud else 0.0
        if txn_id in self._disputes:
            d = self._disputes[txn_id]
            r_dsp = self.backend.run(
                self.dispute_agent,
                build.dispute_inputs(d, self.dataset, fraud_score=assessment.risk_score))
            total_usage += r_dsp.usage
            total_latency += r_dsp.latency_ms
            dispute_out = {"dispute_id": d.id, **r_dsp.data}
            impact = max(impact, float(r_dsp.data.get("financial_impact_usd", 0.0)))
            trace.append(f"[dispute] {d.id} ({d.reason_code.value}) -> "
                         f"{r_dsp.data.get('recommendation')}, "
                         f"cardholder_should_win={r_dsp.data.get('cardholder_should_win')}")

        # de-dup actions, preserving order
        seen: set[str] = set()
        actions = [a for a in actions if not (a in seen or seen.add(a))]

        return OrchestratorDecision(
            txn_id=txn_id,
            backend=self.backend.name,
            category=category,
            policy_violations=violations,
            requires_approval=approval,
            fraud_score=assessment.risk_score,
            fraud_band=assessment.risk_band.value,
            is_fraud=is_fraud,
            recommended_actions=actions,
            financial_impact_usd=impact,
            dispute=dispute_out,
            latency_ms=total_latency,
            usage={"input_tokens": total_usage.input_tokens,
                   "output_tokens": total_usage.output_tokens,
                   "cost_usd": round(total_usage.cost_usd, 6)},
            trace=trace,
        )
