"""A thin, purpose-built Claude client for the workflows.

Three capabilities, all through the Anthropic SDK:
  * ``extract_document`` — vision over a receipt/invoice image or PDF -> structured JSON
  * ``run_agent``        — an agentic tool-use loop (the model calls your Python tools)
  * ``complete_json``    — plain structured output

Everything degrades gracefully: with no key, ``available`` is False and callers run
their own deterministic simulation instead.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import load_config


@dataclass
class Tool:
    """A tool the agent can call: name, JSON-schema input, and a Python executor."""

    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[[dict[str, Any]], Any]

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "input_schema": self.input_schema}


@dataclass
class AgentRun:
    final_text: str
    steps: list[dict[str, Any]] = field(default_factory=list)  # tool calls + results
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    simulated: bool = False


_PRICES = {"claude-opus-4-8": (5.0, 25.0), "claude-sonnet-4-6": (3.0, 15.0),
           "claude-haiku-4-5": (1.0, 5.0)}


class ClaudeClient:
    def __init__(self, model: str = "claude-opus-4-8", effort: str = "high"):
        cfg = load_config()
        self.available = cfg.has_live_models
        self.model = model
        self.effort = effort
        self._client = None
        if self.available:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except Exception:
                self.available = False

    # ---- params per model ----------------------------------------------------
    def _thinking_params(self) -> dict[str, Any]:
        if self.model in ("claude-opus-4-8", "claude-sonnet-4-6"):
            return {"thinking": {"type": "adaptive"},
                    "output_config": {"effort": self.effort}}
        return {}

    def _cost(self, usage) -> float:
        pin, pout = _PRICES.get(self.model, (0.0, 0.0))
        ti = int(getattr(usage, "input_tokens", 0) or 0)
        to = int(getattr(usage, "output_tokens", 0) or 0)
        return ti / 1e6 * pin + to / 1e6 * pout

    @staticmethod
    def _text(resp) -> str:
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")

    # ---- vision: document -> structured JSON --------------------------------
    def extract_document(self, data: bytes, media_type: str, system: str,
                         instructions: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Read a receipt/invoice image or PDF and return validated structured fields."""
        b64 = base64.standard_b64encode(data).decode("utf-8")
        if media_type == "application/pdf":
            doc = {"type": "document", "source": {"type": "base64",
                   "media_type": media_type, "data": b64}}
        else:
            doc = {"type": "image", "source": {"type": "base64",
                   "media_type": media_type, "data": b64}}
        params = self._thinking_params()
        output_config = params.pop("output_config", {})
        output_config["format"] = {"type": "json_schema", "schema": schema}
        resp = self._client.messages.create(
            model=self.model, max_tokens=4096, system=system,
            messages=[{"role": "user", "content": [doc, {"type": "text", "text": instructions}]}],
            output_config=output_config, **params)
        text = self._text(resp)
        data_out = json.loads(text) if text else {}
        data_out["_usage"] = {"cost_usd": round(self._cost(resp.usage), 6),
                              "model": self.model}
        return data_out

    # ---- agentic tool-use loop ----------------------------------------------
    def run_agent(self, system: str, user: str, tools: list[Tool],
                  max_steps: int = 8) -> AgentRun:
        registry = {t.name: t for t in tools}
        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        steps: list[dict[str, Any]] = []
        cost = 0.0
        ti = to = 0
        params = self._thinking_params()
        params.pop("output_config", None)  # no structured output during tool loop
        eff = {"output_config": {"effort": self.effort}} if self.model in (
            "claude-opus-4-8", "claude-sonnet-4-6") else {}

        for _ in range(max_steps):
            resp = self._client.messages.create(
                model=self.model, max_tokens=8000, system=system, messages=messages,
                tools=[t.spec() for t in tools], **params, **eff)
            cost += self._cost(resp.usage)
            ti += int(getattr(resp.usage, "input_tokens", 0) or 0)
            to += int(getattr(resp.usage, "output_tokens", 0) or 0)
            if resp.stop_reason != "tool_use":
                return AgentRun(self._text(resp), steps, ti, to, round(cost, 6), self.model)
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                tool = registry.get(block.name)
                try:
                    out = tool.run(dict(block.input)) if tool else {"error": "unknown tool"}
                except Exception as exc:  # surface tool errors to the model
                    out = {"error": str(exc)}
                steps.append({"tool": block.name, "input": dict(block.input), "output": out})
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(out, default=str)[:6000]})
            messages.append({"role": "user", "content": results})
        return AgentRun("(stopped: max steps reached)", steps, ti, to, round(cost, 6), self.model)

    # ---- plain structured output --------------------------------------------
    def complete_json(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        params = self._thinking_params()
        output_config = params.pop("output_config", {})
        output_config["format"] = {"type": "json_schema", "schema": schema}
        resp = self._client.messages.create(
            model=self.model, max_tokens=4096, system=system,
            messages=[{"role": "user", "content": user}],
            output_config=output_config, **params)
        text = self._text(resp)
        return json.loads(text) if text else {}
