"""Base class shared by the evaluable single-shot agents."""
from __future__ import annotations

from typing import Any


class BaseAgent:
    """Common scaffolding for agents that satisfy ``AgentProtocol``.

    Subclasses must set ``task_name`` / ``output_schema`` and implement
    ``build_messages`` and ``solve``. ``perturb`` and ``coerce`` have sensible
    defaults (identity) that subclasses override to model error / parse LLM output.
    """

    task_name: str = "agent"
    output_schema: dict[str, Any] = {}

    # The natural-language framing every agent shares.
    persona = ("You are Comptroller, Stripe's financial-correctness AI. You reason "
               "carefully about Stripe Issuing card and Stripe Treasury activity and you NEVER guess "
               "when the rules give a definite answer. Respond only via the required "
               "structured output.")

    def build_messages(self, inputs: dict[str, Any]) -> tuple[str, str]:  # pragma: no cover
        raise NotImplementedError

    def solve(self, inputs: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def perturb(self, inputs: dict[str, Any], base: dict[str, Any], rng) -> dict[str, Any]:
        return dict(base)

    def coerce(self, data: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
        return data

    # ---- small shared helpers ------------------------------------------------
    @staticmethod
    def _usd(cents: int) -> float:
        return round(cents / 100.0, 2)

    @staticmethod
    def _clean_enum(value: Any, allowed: list[str], default: str) -> str:
        v = str(value).strip().lower().replace(" ", "_")
        return v if v in allowed else default
