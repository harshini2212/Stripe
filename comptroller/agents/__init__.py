"""Agentic AI workflows over Stripe spend — categorization, policy, dispute, fraud, tieout.

Each single-shot agent is *backend-agnostic* and satisfies
:class:`~comptroller.llm.base.AgentProtocol`, so it runs identically on the analytical
engine, a simulated model, or live Claude — which is what makes the eval harness a
real multi-model leaderboard. The multi-step :class:`ComptrollerOrchestrator` and
:class:`FraudInvestigator` chain these into autonomous workflows.
"""
from .base import BaseAgent
from .categorization import CategorizationAgent
from .policy import PolicyAuditAgent
from .dispute import DisputeAgent
from .fraud_triage import FraudTriageAgent
from .tieout import TieoutAgent
from .investigator import FraudInvestigator, InvestigationReport
from .orchestrator import ComptrollerOrchestrator, OrchestratorDecision

EVAL_AGENTS = {
    "categorization": CategorizationAgent,
    "policy_audit": PolicyAuditAgent,
    "dispute": DisputeAgent,
    "fraud_triage": FraudTriageAgent,
    "tieout": TieoutAgent,
}

__all__ = [
    "BaseAgent",
    "CategorizationAgent",
    "PolicyAuditAgent",
    "DisputeAgent",
    "FraudTriageAgent",
    "TieoutAgent",
    "FraudInvestigator",
    "InvestigationReport",
    "ComptrollerOrchestrator",
    "OrchestratorDecision",
    "EVAL_AGENTS",
]
