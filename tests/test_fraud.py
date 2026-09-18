"""Fraud ensemble + scoring behavior."""
import numpy as np


def test_holdout_metrics_are_strong(pipeline):
    m = pipeline.holdout_metrics
    assert m.roc_auc > 0.78          # credible, not perfect
    assert m.roc_auc < 1.0           # not a trivial separator
    assert m.precision >= 0.7
    assert m.recall >= 0.5


def test_fraud_scores_separate_from_legit(pipeline):
    scores = pipeline.scores().to_numpy()
    y = pipeline.labels()
    assert scores[y == 1].mean() > scores[y == 0].mean() + 0.3


def test_top_alerts_are_mostly_fraud(pipeline):
    alerts = pipeline.top_alerts(15)
    hit = sum(1 for a in alerts if a.actual_fraud)
    assert hit >= 12


def test_feature_importances_are_spread(pipeline):
    imp = pipeline.model.feature_importances
    assert imp
    assert abs(sum(imp.values()) - 1.0) < 1e-6
    # no single feature should be a near-perfect oracle
    assert max(imp.values()) < 0.9


def test_assessment_has_action_and_drivers(pipeline):
    a = pipeline.top_alerts(1)[0]
    assert a.recommended_action in (
        "freeze_card_and_open_dispute", "open_dispute", "monitor", "clear")
    assert a.drivers  # high-risk alert should have explanations
    assert all(d.delta > 0 for d in a.drivers)
