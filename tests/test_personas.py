"""Persona RBAC (permission-by-toolset) and the replay-policy tool — all offline."""
from comptroller.agents.agent_tools import ToolContext, allowed_tools, replay_policy, tool_names
from comptroller.agents.personas import PERSONA_TABS, Persona


def test_investor_cannot_drill_into_employees():
    t = tool_names(Persona.INVESTOR)
    assert "query_transactions" not in t
    assert "transaction_detail" not in t
    assert "get_my_status" not in t


def test_employee_cannot_see_aggregates_or_others():
    t = tool_names(Persona.EMPLOYEE)
    assert "company_aggregates" not in t
    assert "fraud_scan" not in t
    assert "query_transactions" in t  # but scoped to self via ToolContext


def test_writes_are_finance_only():
    assert "flag_for_review" in tool_names(Persona.FINANCE)
    assert "flag_for_review" not in tool_names(Persona.EXECUTIVE)
    assert "flag_for_review" not in tool_names(Persona.INVESTOR)


def test_request_user_selection_available_to_all():
    for p in Persona:
        assert "request_user_selection" in tool_names(p)


def test_allowed_tools_builds_callable_tools(dataset, pipeline):
    ctx = ToolContext(dataset, pipeline)
    tools = allowed_tools(Persona.FINANCE, ctx)
    names = {t.name for t in tools}
    assert "fraud_scan" in names and "replay_policy" in names
    assert all(callable(t.run) for t in tools)


def test_employee_tools_are_scoped_to_self(dataset, pipeline):
    emp = dataset.employees[0].id
    ctx = ToolContext(dataset, pipeline, employee_id=emp)
    tool = {t.name: t for t in allowed_tools(Persona.EMPLOYEE, ctx)}["query_transactions"]
    out = tool.run({"limit": 50})
    assert out["transactions"]
    assert all(r["employee_id"] == emp for r in out["transactions"])  # only own rows


def test_replay_policy_catches_planted_anomalies(dataset, pipeline):
    ctx = ToolContext(dataset, pipeline)
    out = replay_policy(ctx, {"rules": {
        "blocked_categories": ["fuel"], "per_txn_limit_usd": 5000, "no_weekend_meals": True}})
    rules = out["by_rule"]
    assert "blocked_category:fuel" in rules and rules["blocked_category:fuel"]["count"] > 0
    assert out["total_dollar_impact_usd"] > 0


def test_persona_tabs_are_defined():
    for p in Persona:
        assert PERSONA_TABS[p]  # every persona has at least one workspace


def test_runway_tools_are_executive_investor_only():
    for t in ("runway_intervention", "confounded_driver_check"):
        assert t in tool_names(Persona.EXECUTIVE)
        assert t in tool_names(Persona.INVESTOR)
        assert t not in tool_names(Persona.EMPLOYEE)


def test_ap_agent_refuses_to_pay_duplicate(dataset, pipeline):
    from comptroller.agents.agent_tools import detect_duplicate_invoices, pay_invoice
    ctx = ToolContext(dataset, pipeline)
    dup = detect_duplicate_invoices(ctx, {})["duplicates"][0]["invoice"]
    res = pay_invoice(ctx, {"invoice_id": dup})
    assert res.get("refused") is True  # cannot double-pay


def test_ap_catches_bank_change_and_over_po(dataset, pipeline):
    from comptroller.agents.agent_tools import three_way_match, vendor_bank_change_check
    ctx = ToolContext(dataset, pipeline)
    assert vendor_bank_change_check(ctx, {})["count"] >= 1
    over = next(i.id for i in ctx.ap.invoices if i.anomaly == "over_po")
    assert three_way_match(ctx, {"invoice_id": over})["over_po"] is True
