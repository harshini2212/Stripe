"""Concrete backends: deterministic analytical engine, simulated models, live Claude."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np

from ..config import CLAUDE_BACKENDS, OFFLINE_BACKEND, Config, load_config
from .base import AgentProtocol, AgentResult, Backend, Usage

# Per-1M-token pricing (USD) for the cost column on the leaderboard.
_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def _seed_from(*parts: str) -> int:
    h = hashlib.blake2b("|".join(parts).encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")


class AnalyticalBackend(Backend):
    """Deterministic financial engine — the offline baseline and grading reference."""

    name = OFFLINE_BACKEND
    kind = "deterministic"

    def run(self, agent: AgentProtocol, inputs: dict[str, Any]) -> AgentResult:
        t0 = time.perf_counter()
        try:
            data = agent.solve(inputs)
            return AgentResult(self.name, data, (time.perf_counter() - t0) * 1000, Usage())
        except Exception as exc:  # pragma: no cover - defensive
            return AgentResult(self.name, {}, (time.perf_counter() - t0) * 1000, Usage(),
                               ok=False, error=str(exc))


class SimulatedBackend(Backend):
    """Emulates a model's quality offline by perturbing the analytical answer.

    ``skill`` in [0, 1] is the probability the model reproduces the engine's answer;
    otherwise it returns a plausible-but-degraded variant. Higher skill -> fewer
    perturbations -> higher accuracy, producing a believable leaderboard ordering
    without any network access. Clearly labelled so it's never mistaken for a live run.
    """

    kind = "simulated"

    def __init__(self, label: str, skill: float, tier_latency_ms: float = 900.0):
        self.name = label
        self.skill = float(skill)
        self.tier_latency_ms = tier_latency_ms

    def run(self, agent: AgentProtocol, inputs: dict[str, Any]) -> AgentResult:
        t0 = time.perf_counter()
        base = agent.solve(inputs)
        _, user = agent.build_messages(inputs)
        rng = np.random.default_rng(_seed_from(self.name, agent.task_name, user))
        data = base
        if rng.random() > self.skill:
            try:
                data = agent.perturb(inputs, base, rng)
            except Exception:
                data = base
        # Notional usage so the leaderboard cost column is populated offline.
        approx_in = max(1, len(user) // 4)
        approx_out = 60
        usage = Usage(approx_in, approx_out, 0.0)
        return AgentResult(self.name, data, self.tier_latency_ms + (time.perf_counter() - t0) * 1000,
                           usage, trace=["simulated"])


class ClaudeBackend(Backend):
    """A live Claude model via the Anthropic SDK with structured outputs."""

    kind = "llm"

    def __init__(self, model: str, effort: str = "high", max_tokens: int = 4096):
        self.name = model
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self._client = None

    def _client_or_init(self):
        if self._client is None:
            import anthropic  # imported lazily so offline installs don't need a key

            self._client = anthropic.Anthropic()
        return self._client

    def _params(self) -> dict[str, Any]:
        # Opus 4.8 / Sonnet 4.6 support adaptive thinking + effort; Haiku 4.5 does not.
        if self.model in ("claude-opus-4-8", "claude-sonnet-4-6"):
            return {"thinking": {"type": "adaptive"},
                    "output_config": {"effort": self.effort}}
        return {}

    def run(self, agent: AgentProtocol, inputs: dict[str, Any]) -> AgentResult:
        t0 = time.perf_counter()
        system, user = agent.build_messages(inputs)
        try:
            client = self._client_or_init()
            params = self._params()
            output_config = params.pop("output_config", {})
            output_config["format"] = {
                "type": "json_schema",
                "schema": agent.output_schema,
            }
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config=output_config,
                **params,
            )
            text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
            raw = json.loads(text) if text else {}
            data = agent.coerce(raw, inputs)
            usage = self._usage(resp)
            return AgentResult(self.name, data, (time.perf_counter() - t0) * 1000, usage)
        except Exception as exc:
            return AgentResult(self.name, {}, (time.perf_counter() - t0) * 1000, Usage(),
                               ok=False, error=f"{type(exc).__name__}: {exc}")

    def _usage(self, resp: Any) -> Usage:
        u = getattr(resp, "usage", None)
        if u is None:
            return Usage()
        ti = int(getattr(u, "input_tokens", 0) or 0)
        to = int(getattr(u, "output_tokens", 0) or 0)
        pin, pout = _PRICES.get(self.model, (0.0, 0.0))
        return Usage(ti, to, ti / 1e6 * pin + to / 1e6 * pout)


# Simulated-model quality profiles used when no API key is present.
_SIM_PROFILES = {
    "claude-opus-4-8": (0.96, 1700.0),
    "claude-sonnet-4-6": (0.92, 950.0),
    "claude-haiku-4-5": (0.85, 420.0),
}


def build_backends(config: Config | None = None, *, include_baseline: bool = True) -> list[Backend]:
    """Construct the backend roster for the eval leaderboard.

    With a key: live Claude models + the analytical baseline.
    Without a key: the analytical baseline + simulated stand-ins, so the harness is
    fully demonstrable offline.
    """
    config = config or load_config()
    backends: list[Backend] = []
    if include_baseline:
        backends.append(AnalyticalBackend())
    if config.has_live_models:
        for model in config.leaderboard_backends:
            backends.append(ClaudeBackend(model, effort=config.effort))
    else:
        for model in config.leaderboard_backends:
            skill, latency = _SIM_PROFILES.get(model, (0.9, 800.0))
            backends.append(SimulatedBackend(f"{model} (sim)", skill, latency))
    return backends
