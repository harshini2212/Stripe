"""Financial-operations analytics: treasury, underwriting, spend, AP, registry."""
from comptroller.analytics import (
    APIntelligence,
    SpendIntelligence,
    TreasuryForecaster,
    Underwriter,
    model_registry,
)


def test_treasury_forecast(dataset):
    fc = TreasuryForecaster(dataset, horizon_days=21).forecast()
    d = fc.to_dict()
    assert d["history"] and d["forecast"]
    assert len(d["forecast"]) == 21
    assert 0 <= d["backtest_mape"] < 0.5      # backtest is reasonable
    assert d["status"] in ("healthy", "watch", "critical")
    assert d["yield_opportunity"]["incremental_annual_yield_usd"] > 0


def test_underwriter_model_and_assessment(dataset):
    ca = Underwriter(dataset, seed=7).assess()
    assert ca.model_auc > 0.8                 # the PD model discriminates well
    assert 0.0 <= ca.pd <= 1.0
    assert ca.recommended_limit_usd > 0
    assert ca.action in ("increase_limit", "hold", "reduce_within_24h")
    assert ca.drivers


def test_spend_subscriptions_and_compliance(dataset):
    si = SpendIntelligence(dataset)
    subs = si.recurring_subscriptions()
    assert subs["count"] >= 1                  # recurring detection finds subscriptions
    assert subs["redundant_savings_usd"] >= 0
    comp = si.compliance()
    assert 0.0 < comp["compliance_rate"] <= 1.0
    assert comp["flagged_transactions"] >= 0
    assert "duplicate_spend" in comp["violations_by_type"] or comp["flagged_transactions"] >= 0


def test_spend_summary_breakdowns(dataset):
    s = SpendIntelligence(dataset).summary()
    assert s["total_spend_usd"] > 0
    assert s["by_category"] and s["by_department"]
    assert s["top_merchants"]


def test_ap_intelligence(dataset):
    a = APIntelligence(dataset, seed=7)
    assert a.summary()["total_invoices"] > 0
    dups = a.duplicate_invoices()
    assert dups["count"] >= 1                  # we plant double-billed invoices
    conc = a.vendor_concentration()
    assert conc["concentration"] in ("low", "moderate", "high")
    assert 0 < conc["hhi"] <= 1
    assert "recommendation" in a.payment_timing()


def test_model_registry(dataset):
    cards = model_registry(dataset)
    assert len(cards) >= 5
    for c in cards:
        assert c["name"] and "primary_metric" in c and "description" in c
