"""Causal runway engine (Rung-2) + the AP ledger — offline."""
from comptroller.analytics.causal_runway import CausalRunway, Scenario
from comptroller.documents.invoices import build_ap_ledger


def test_baseline_runway_is_sane():
    b = CausalRunway().baseline(36)
    assert b.runway_months is not None and 6 <= b.runway_months <= 30


def test_interventions_move_runway_the_right_way():
    cr = CausalRunway()
    base = cr.baseline(40).runway_months
    assert cr.intervene(Scenario(hires_per_month=20), 40, mc=0).runway_months < base   # hiring burns
    assert cr.intervene(Scenario(freeze_hiring=True), 40, mc=0).runway_months > base    # freeze extends
    assert cr.intervene(Scenario(churn_monthly=0.0), 40, mc=0).runway_months > base     # less churn extends
    assert cr.intervene(Scenario(price_change_pct=0.10), 40, mc=0).runway_months >= base


def test_confounded_driver_proof():
    """Recruiting: large naive correlational effect, ~zero interventional effect."""
    c = CausalRunway().confounded_effect("recruiting", 0.5)
    naive = c["naive_correlational_revenue_effect_usd"]
    inter = c["interventional_revenue_effect_usd"]
    assert naive > 1000                                   # correlation says big effect
    assert abs(inter) < abs(naive) * 0.2                  # intervention says ~nothing
    # Marketing has a genuine causal path, so its interventional effect is non-trivial.
    m = CausalRunway().confounded_effect("marketing", 0.5)
    assert abs(m["interventional_revenue_effect_usd"]) > abs(inter)


def test_monte_carlo_band():
    fc = CausalRunway().intervene(Scenario(hires_per_month=10), 12, mc=100)
    assert len(fc.p10_cash) == 12 and len(fc.p90_cash) == 12


def test_ap_ledger_has_all_planted_anomalies():
    led = build_ap_ledger()
    anomalies = {i.anomaly for i in led.invoices}
    assert {"over_po", "duplicate", "bank_changed", "unmatched"} <= anomalies
    assert led.pos and led.vendor_bank
