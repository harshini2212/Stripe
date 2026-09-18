"""Backend-facing types and the Agent protocol they operate on."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cost_usd + other.cost_usd,
        )


@dataclass
class AgentResult:
    """The output of running one agent on one input under one backend."""

    backend: str
    data: dict[str, Any]
    latency_ms: float = 0.0
    usage: Usage = field(default_factory=Usage)
    ok: bool = True
    error: str | None = None
    trace: list[str] = field(default_factory=list)


@runtime_checkable
class AgentProtocol(Protocol):
    """The contract every evaluable agent satisfies.

    An agent owns the task logic; a backend only decides *who* solves it (a
    deterministic engine, a simulated model, or live Claude).
    """

    task_name: str
    output_schema: dict[str, Any]

    def build_messages(self, inputs: dict[str, Any]) -> tuple[str, str]:
        """Return ``(system_prompt, user_prompt)`` for an LLM backend."""
        ...

    def solve(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Deterministic analytical answer (the offline engine)."""
        ...

    def perturb(self, inputs: dict[str, Any], base: dict[str, Any],
                rng: "np.random.Generator") -> dict[str, Any]:
        """Return a degraded variant of ``base`` to emulate model error."""
        ...

    def coerce(self, data: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
        """Normalize/validate a raw LLM response into the canonical output shape."""
        ...


class Backend:
    """Base class for all backends."""

    name: str = "backend"
    kind: str = "deterministic"  # deterministic | simulated | llm

    def run(self, agent: AgentProtocol, inputs: dict[str, Any]) -> AgentResult:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"
