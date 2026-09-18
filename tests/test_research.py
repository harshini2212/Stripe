"""SEC EDGAR research profile (mocked, no network) + transaction drill-downs."""
from fastapi.testclient import TestClient

from comptroller.api import app
from comptroller.research.edgar import build_company_profile

client = TestClient(app)


def _fact(end, val, *, start=None, fy=2024, filed="2025-02-01"):
    row = {"end": end, "val": val, "fy": fy, "fp": "FY", "form": "10-K", "filed": filed}
    if start:
        row["start"] = start
    return row


_FACTS = {
    "entityName": "TESTCO INC",
    "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [
            _fact("2023-12-31", 1000, start="2023-01-01", fy=2023, filed="2024-02-01"),
            _fact("2024-12-31", 1200, start="2024-01-01", fy=2024)]}},
        "NetIncomeLoss": {"units": {"USD": [
            _fact("2023-12-31", 100, start="2023-01-01", fy=2023, filed="2024-02-01"),
            _fact("2024-12-31", 150, start="2024-01-01", fy=2024)]}},
        "Assets": {"units": {"USD": [_fact("2024-12-31", 5000)]}},
        "Liabilities": {"units": {"USD": [_fact("2024-12-31", 3000)]}},
        "StockholdersEquity": {"units": {"USD": [_fact("2024-12-31", 2000)]}},
    }},
}


class _FakeEdgar:
    def resolve(self, ticker):
        return {"ticker": "TESTCO", "name": "TESTCO INC", "cik": "0000000001"}

    def company_facts(self, cik):
        return _FACTS


def test_profile_series_ratios_and_tieouts():
    p = build_company_profile("TESTCO", _FakeEdgar())
    assert p["name"] == "TESTCO INC" and p["latest_fy"] == 2024
    rev = [(r["fy"], r["value"]) for r in p["series"]["revenue"]]
    assert rev == [(2023, 1000.0), (2024, 1200.0)]
    assert round(p["ratios"]["revenue_yoy"], 3) == 0.2           # 1000 -> 1200
    assert round(p["ratios"]["net_margin"], 4) == 0.125          # 150 / 1200
    assert round(p["ratios"]["debt_to_equity"], 2) == 1.5        # 3000 / 2000
    # Assets (5000) == Liabilities (3000) + Equity (2000) — ties out exactly.
    ae = next(t for t in p["tieouts"] if t["check"].startswith("Assets"))
    assert ae["passed"] and ae["delta_usd"] == 0.0


def test_flow_ignores_quarterly_windows():
    facts = {"entityName": "Q", "facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        _fact("2024-03-31", 300, start="2024-01-01", fy=2024),   # Q1 stub — must be dropped
        _fact("2024-12-31", 1200, start="2024-01-01", fy=2024)]}}}}}

    class F:
        def resolve(self, t):
            return {"ticker": "Q", "name": "Q", "cik": "0"}

        def company_facts(self, c):
            return facts

    p = build_company_profile("Q", F())
    assert [(r["fy"], r["value"]) for r in p["series"]["revenue"]] == [(2024, 1200.0)]


def test_research_search_and_bad_ticker_offline_paths(monkeypatch):
    # search hitting a broken EDGAR should 502, not crash
    import comptroller.research.edgar as e

    class Boom:
        def search(self, q):
            raise OSError("no net")

    monkeypatch.setattr(e, "_client", lambda: Boom())
    assert client.get("/api/research/search", params={"q": "x"}).status_code == 502


def test_card_and_vendor_and_category_drilldowns():
    cards = client.get("/api/cards", params={"seed": 7}).json()["cards"]
    cid = cards[0]["card_id"]
    cd = client.get(f"/api/cards/{cid}", params={"seed": 7}).json()
    assert cd["txns"] > 0 and cd["monthly_trend"] and len(cd["recent"]) <= 15
    assert cd["by_category"] and cd["card"]["card_id"] == cid

    vendor = client.get("/api/vendors", params={"seed": 7}).json()["top_vendors"][0]["merchant"]
    vd = client.get("/api/vendors/detail", params={"seed": 7, "name": vendor}).json()
    assert vd["vendor"] == vendor and vd["unique_users"] >= 1 and vd["avg_txn_usd"] > 0

    bud = client.get("/api/budgets", params={"seed": 7}).json()["categories"][0]["category"]
    catd = client.get(f"/api/categories/{bud}", params={"seed": 7}).json()
    assert catd["category"] == bud and catd["txns"] > 0

    assert client.get("/api/cards/nope", params={"seed": 7}).status_code == 404
    assert client.get("/api/categories/not_a_cat", params={"seed": 7}).status_code == 404


def test_spend_block_and_filing_links():
    facts = {"entityName": "SPENDCO", "facts": {"us-gaap": {
        "Revenues": {"units": {"USD": [_fact("2024-12-31", 1000, start="2024-01-01")]}},
        "CostOfRevenue": {"units": {"USD": [_fact("2024-12-31", 600, start="2024-01-01")]}},
        "SellingGeneralAndAdministrativeExpense": {
            "units": {"USD": [_fact("2024-12-31", 200, start="2024-01-01")]}},
        "ResearchAndDevelopmentExpense": {
            "units": {"USD": [_fact("2024-12-31", 100, start="2024-01-01")]}},
        "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [_fact("2024-12-31", 500)]}},
    }}}

    class F:
        def resolve(self, t):
            return {"ticker": "SPENDCO", "name": "SPENDCO", "cik": "0000000042"}

        def company_facts(self, c):
            return facts

    p = build_company_profile("SPENDCO", F())
    sp = p["spend"]
    labels = [c["label"] for c in sp["cost_structure"]]
    assert {"Cost of revenue", "SG&A", "R&D"} <= set(labels)
    # SG&A is 20% of revenue; addressable = 45% of SG&A(200) = 90.
    assert round(sp["addressable_spend_usd"], 2) == 90.0 and "SG&A" in sp["addressable_basis"]
    # 3 addressable levers + 1 idle-cash yield lever.
    assert len(sp["levers"]) == 4
    yld = next(l for l in sp["levers"] if "yield" in l["lever"].lower())
    assert round(yld["savings_usd"], 2) == 22.5                  # 500 * 4.5%
    # 90 * (2.5% + 1.5% + 1.0%) + 22.5 = 4.5 + 22.5 = 27.0
    assert round(sp["total_savings_usd"], 2) == 27.0
    # browse URL always resolves from the CIK; the 10-K doc needs submissions() (absent here).
    assert "0000000042" in p["filing"]["browse_url"]
    assert p["filing"]["latest_10k_url"] is None


_FAKE_PROFILE = {
    "name": "TESTCO INC", "ticker": "TESTCO", "latest_fy": 2024, "cik": "0000000001",
    "series": {"revenue": [{"fy": 2024, "value": 1200.0}], "cash": [{"fy": 2024, "value": 500.0}]},
    "ratios": {"net_margin": 0.125},
    "spend": {"cost_structure": [{"label": "SG&A", "usd": 200.0, "pct_of_revenue": 0.1667}],
              "addressable_spend_usd": 90.0, "addressable_basis": "~45% of SG&A",
              "levers": [{"lever": "Eliminate duplicate & unused SaaS", "rate": 0.025,
                          "savings_usd": 2.25, "note": "n"}],
              "total_savings_usd": 2.25, "savings_pct_of_revenue": 0.0019},
}


def test_research_improve_endpoint_shape(monkeypatch):
    import comptroller.ai as ai
    import comptroller.research.edgar as e
    monkeypatch.setattr(e, "cached_profile", lambda t: _FAKE_PROFILE)

    class FakeCC:
        def __init__(self, *a, **k):
            self.available = True

        def complete_json(self, system, user, schema):
            return {"headline": "H", "improvements": [
                {"area": "SaaS", "problem": "p", "action": "a", "savings_pct": 0.02,
                 "savings_usd": 1.0, "effort": "low", "timeline": "30 days"}],
                "total_savings_usd": 1.0, "summary": "s"}

    monkeypatch.setattr(ai, "ClaudeClient", FakeCC)
    r = client.post("/api/research/improve", json={"ticker": "TESTCO"})
    assert r.status_code == 200
    j = r.json()
    assert j["ticker"] == "TESTCO" and j["total_savings_usd"] == 1.0
    assert j["improvements"][0]["effort"] == "low"


def test_research_agent_runs_grounded_tools(monkeypatch):
    import comptroller.ai as ai
    import comptroller.research.edgar as e
    from comptroller.ai.claude_client import AgentRun
    monkeypatch.setattr(e, "cached_profile", lambda t: _FAKE_PROFILE)

    class FakeCC:
        def __init__(self, *a, **k):
            self.available = True

        def run_agent(self, system, user, tools, max_steps=8):
            # exercise the real grounded tools once so a tool bug fails the test
            steps = []
            for t in tools:
                inp = {"annual_savings_usd": [1, 2]} if t.name == "project_total_savings" else {}
                steps.append({"tool": t.name, "input": inp, "output": t.run(inp)})
            return AgentRun("done", steps, 10, 20, 0.01, "claude-haiku-4-5")

    monkeypatch.setattr(ai, "ClaudeClient", FakeCC)
    r = client.post("/api/research/agent", json={"ticker": "TESTCO", "issue": "cut saas"})
    assert r.status_code == 200
    j = r.json()
    assert j["ticker"] == "TESTCO" and j["issue"] == "cut saas" and j["final"] == "done"
    tools_called = {s["tool"] for s in j["steps"]}
    assert {"pull_spend_breakdown", "project_total_savings"} <= tools_called
    sweep = next(s for s in j["steps"] if s["tool"] == "deploy_treasury_sweep")
    assert round(sweep["output"]["annual_yield_usd"], 2) == 22.5   # 500 * 4.5%, grounded
