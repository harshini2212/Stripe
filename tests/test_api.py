"""The FastAPI service surface (in-process, no network)."""
from fastapi.testclient import TestClient

from comptroller.api import app

client = TestClient(app)


def test_root_serves_app():
    r = client.get("/")
    assert r.status_code == 200
    assert "Stripify" in r.text and 'id="nav"' in r.text  # the app shell, not the picker


def test_welcome_serves_picker():
    r = client.get("/welcome")
    assert r.status_code == 200
    assert "Who's signing in" in r.text


def test_dashboard_and_new_surfaces():
    body = client.get("/api/dashboard", params={"seed": 7}).json()
    assert body["kpis"]["cash_usd"] > 0 and body["cashflow"]["history"]
    assert isinstance(body["insights"], list)
    b = client.get("/api/budgets", params={"seed": 7}).json()
    assert b["categories"] and "utilization" in b["categories"][0]
    v = client.get("/api/vendors", params={"seed": 7}).json()
    assert "subscriptions" in v
    i = client.get("/api/insights", params={"seed": 7}).json()
    assert i["count"] == len(i["insights"])


def test_api_info():
    r = client.get("/api/info")
    assert r.status_code == 200
    assert r.json()["service"] == "comptroller"


def test_tenant_summary():
    r = client.get("/tenant/summary", params={"seed": 7})
    assert r.status_code == 200
    assert r.json()["fraud_transactions"] > 0


def test_fraud_alerts():
    r = client.get("/fraud/alerts", params={"seed": 7, "top": 5})
    body = r.json()
    assert len(body["alerts"]) == 5
    assert "roc_auc" in body["metrics"]


def test_assess_unknown_txn_404():
    assert client.get("/fraud/assess/nope", params={"seed": 7}).status_code == 404


def test_orchestrate_top_alert():
    r = client.post("/agent/orchestrate", json={"backend": "offline", "seed": 7})
    assert r.status_code == 200
    body = r.json()
    assert body["is_fraud"] is True
    assert body["recommended_actions"]
