"""Model backends: a deterministic analytical engine, simulated models, and live Claude.

Every agent is backend-agnostic. The same agent runs on:

* ``AnalyticalBackend`` — a strong deterministic financial engine (no network), used
  as the offline baseline and the thing live models are graded against.
* ``SimulatedBackend`` — emulates model-quality variance offline so the eval
  leaderboard has multiple rows even without an API key (clearly labelled "sim").
* ``ClaudeBackend`` — a live Claude model (Opus 4.8 / Sonnet 4.6 / Haiku 4.5) via the
  Anthropic SDK with structured outputs. Lights up when ``ANTHROPIC_API_KEY`` is set.
"""
from .base import AgentProtocol, AgentResult, Backend, Usage
from .backends import AnalyticalBackend, ClaudeBackend, SimulatedBackend, build_backends

__all__ = [
    "AgentProtocol",
    "AgentResult",
    "Backend",
    "Usage",
    "AnalyticalBackend",
    "ClaudeBackend",
    "SimulatedBackend",
    "build_backends",
]
