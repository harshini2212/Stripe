"""The interactive, multi-turn, persona-scoped agent runner — the spine.

One reusable runner powers every tab. It is LIVE-ONLY (requires a Claude key). The agent
gathers the user's scope and preferences across multiple rounds (via the
``request_user_selection`` tool, which pauses the loop and surfaces options to the UI),
confirms before any write, calls only the persona's permitted tools, and returns a
citable trace with token cost for every run.
"""
from __future__ import annotations

import json
import uuid
from typing import Any

from ..ai import ClaudeClient
from .agent_tools import REQUEST_SELECTION, ToolContext, allowed_tools
from .personas import TAB_TITLES, Persona

_TAB_GUIDE = {
    "forensics": ("Investigate spend. Use query_transactions, fraud_scan, "
                  "find_duplicate_spend, vendor_price_changes and subscription_audit to "
                  "answer the user's question with evidence and dollar amounts."),
    "policy_studio": ("Compile the user's plain-English policy into a structured rule set "
                      "(per_txn_limit_usd, blocked_categories, receipt_required_over_usd, "
                      "approval_required_over_usd, no_weekend_meals). Confirm the rules and "
                      "the replay window via selections, then call replay_policy and report "
                      "the dollar impact grouped by rule, vs the current policy."),
    "investor_room": ("Answer board-level diligence over company aggregates and the credit "
                      "model. EVERY dollar figure you state in a conclusion MUST be verified "
                      "with tieout_check; if you cannot verify it, say so. You have no "
                      "employee-level tools — decline employee drill-downs."),
    "my_spend": ("Help this one employee with their own card only. Explain flags by calling "
                 "transaction_detail and tying them to the policy rule, and draft fixes."),
    "ap": ("Process accounts-payable. Use list_invoices, three_way_match, "
           "detect_duplicate_invoices and vendor_bank_change_check to decide pay/hold per "
           "invoice. You MUST catch duplicates and changed-bank invoices and refuse to pay "
           "them. Confirm the batch with the user before calling any pay_invoice."),
    "runway": ("Causal cash-runway lab. After eliciting the scenario via selections, call "
               "cash_runway_baseline (Rung 1) AND runway_intervention (Rung 2 — the do() "
               "what-if) and contrast them. Use confounded_driver_check to show how the naive "
               "correlational effect of recruiting on revenue differs from the interventional "
               "effect — explain the demand confounder. Report the runway change and the causal "
               "chain (e.g. hire 20 -> payroll up -> burn up -> runway down)."),
    "close": ("Run the month-end close: call reconcile_close, summarize the tieout, the policy "
              "and fraud exceptions, and the proposed accruals; flag what needs human sign-off."),
    "card_issuance": ("Issue a Stripe Issuing card. Compile the request into a spec with propose_card, show "
                      "it, and CONFIRM with the user via request_user_selection before calling "
                      "issue_card (a write)."),
    "treasury": ("Optimize idle cash. After the user picks a buffer, call treasury_ladder and "
                 "explain the laddered allocation and the incremental yield."),
}


def _system_prompt(persona: Persona, tab: str) -> str:
    return (
        f"You are Comptroller, an AI finance analyst operating in {persona.label} mode "
        f"({persona.tier}), in the \"{TAB_TITLES.get(tab, tab)}\" workspace.\n\n"
        f"WORKSPACE: {_TAB_GUIDE.get(tab, 'Assist the user.')}\n\n"
        "HOW YOU WORK:\n"
        "1. Be interactive. BEFORE heavy analysis, call request_user_selection to gather "
        "the user's scope and preferences across at least TWO rounds (e.g. what to focus "
        "on, then the specific window or filters). Make the user choose — don't assume.\n"
        "2. Cite every number. Get figures from tools; for any dollar amount in a "
        "conclusion, verify it with tieout_check when that tool is available. Never invent "
        "numbers.\n"
        "3. Confirm before writing. Before any write tool (e.g. flag_for_review), confirm "
        "the action with request_user_selection.\n"
        "4. Stay in your lane. You only hold the tools for your role; if asked for something "
        "outside them, explain that you can't and why.\n"
        "5. Finish with a concise, structured answer: the key findings, the dollars, and any "
        "actions taken.")


class Session:
    def __init__(self, persona: Persona, tab: str, dataset, pipeline,
                 employee_id: str | None, model: str):
        self.id = "sess_" + uuid.uuid4().hex[:12]
        self.persona = persona
        self.tab = tab
        self.cc = ClaudeClient(model=model)
        self.ctx = ToolContext(dataset, pipeline, employee_id=employee_id)
        self.tools = allowed_tools(persona, self.ctx)
        self.tool_map = {t.name: t for t in self.tools}
        self.system = _system_prompt(persona, tab)
        self.messages: list[dict[str, Any]] = []
        self.trace: list[dict[str, Any]] = []
        self.cost = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.status = "running"
        self.answer = ""
        self._pending_results: list[dict] = []
        self._pending_selection: dict | None = None
        self._selections_answered = 0  # force >=2 elicitation rounds before any work

    # ---- the loop ------------------------------------------------------------
    def _params(self) -> dict:
        if self.cc.model in ("claude-opus-4-8", "claude-sonnet-4-6"):
            return {"thinking": {"type": "adaptive"}, "output_config": {"effort": self.cc.effort}}
        return {}

    def _track(self, usage) -> None:
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        self.cost += self.cc._cost(usage)

    def advance(self) -> dict:
        for _ in range(24):  # safety bound on tool rounds
            # Force the first two turns to be user-selection prompts so the agent always
            # elicits scope/preferences interactively before doing any heavy work. (Forced
            # tool_choice is incompatible with thinking, so we drop it on those turns.)
            force = self._selections_answered < 2
            kwargs: dict = {"model": self.cc.model, "system": self.system,
                            "messages": self.messages, "tools": [t.spec() for t in self.tools]}
            if force:
                kwargs.update(max_tokens=1500,
                              tool_choice={"type": "tool", "name": REQUEST_SELECTION})
            else:
                kwargs.update(max_tokens=8000, **self._params())
            resp = self.cc._client.messages.create(**kwargs)
            self._track(resp.usage)
            if resp.stop_reason != "tool_use":
                self.status = "complete"
                self.answer = self.cc._text(resp)
                return self.state()
            self.messages.append({"role": "assistant", "content": resp.content})
            results, pending = [], None
            for block in resp.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                if block.name == REQUEST_SELECTION:
                    pending = {"tool_use_id": block.id,
                               "prompt": block.input.get("prompt", "Choose:"),
                               "options": list(block.input.get("options", [])),
                               "allow_multiple": bool(block.input.get("allow_multiple", False))}
                else:
                    tool = self.tool_map.get(block.name)
                    try:
                        out = tool.run(dict(block.input)) if tool else {"error": "no such tool"}
                    except Exception as exc:
                        out = {"error": str(exc)}
                    self.trace.append({"tool": block.name, "input": dict(block.input), "output": out})
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": json.dumps(out, default=str)[:8000]})
            if pending:
                self._pending_results = results
                self._pending_selection = pending
                self.status = "need_selection"
                return self.state()
            self.messages.append({"role": "user", "content": results})
        self.status = "complete"
        self.answer = "(stopped: too many tool rounds)"
        return self.state()

    def step(self, selection: Any) -> dict:
        if self._pending_selection is None:
            return self.state()
        text = selection if isinstance(selection, str) else json.dumps(selection)
        self.trace.append({"tool": REQUEST_SELECTION, "input": {"prompt": self._pending_selection["prompt"]},
                           "output": {"user_selected": text}})
        results = self._pending_results + [{
            "type": "tool_result", "tool_use_id": self._pending_selection["tool_use_id"],
            "content": f"User selected: {text}"}]
        self.messages.append({"role": "user", "content": results})
        self._pending_results, self._pending_selection = [], None
        self._selections_answered += 1
        self.status = "running"
        return self.advance()

    # ---- state for the UI ----------------------------------------------------
    def state(self) -> dict:
        base = {"session_id": self.id, "status": self.status, "persona": self.persona.value,
                "tab": self.tab, "trace": self.trace,
                "cost_usd": round(self.cost, 6),
                "usage": {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens}}
        if self.status == "need_selection" and self._pending_selection:
            base.update(prompt=self._pending_selection["prompt"],
                        options=self._pending_selection["options"],
                        allow_multiple=self._pending_selection["allow_multiple"])
        elif self.status == "complete":
            base.update(answer=self.answer,
                        findings=[s["output"] for s in self.trace if s["tool"] != REQUEST_SELECTION])
        return base


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def start(self, persona: Persona, tab: str, dataset, pipeline, *,
              query: str | None, employee_id: str | None, model: str) -> dict:
        s = Session(persona, tab, dataset, pipeline, employee_id, model)
        if not s.cc.available:
            return {"session_id": s.id, "status": "error",
                    "message": "Live agent requires ANTHROPIC_API_KEY. Set it in .env and restart."}
        self._sessions[s.id] = s
        opener = query or f"Begin the {TAB_TITLES.get(tab, tab)} workflow."
        s.messages.append({"role": "user", "content": opener})
        try:
            return s.advance()
        except Exception as exc:
            s.status = "error"
            return {"session_id": s.id, "status": "error", "message": str(exc)}

    def step(self, session_id: str, selection: Any) -> dict:
        s = self._sessions.get(session_id)
        if s is None:
            return {"status": "error", "message": "unknown session"}
        try:
            return s.step(selection)
        except Exception as exc:
            s.status = "error"
            return {"session_id": session_id, "status": "error", "message": str(exc)}


SESSIONS = SessionManager()
