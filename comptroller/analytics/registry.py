"""Model registry — a card per model with its live metric, so the whole ML portfolio
is inspectable in one place (the architecture story behind the product)."""
from __future__ import annotations

from typing import Any

from ..domain import Dataset


def model_registry(dataset: Dataset, *, pipeline=None, underwriter=None,
                   forecaster=None) -> list[dict[str, Any]]:
    """Return model cards for every model in the platform, with live metrics."""
    from ..fraud import FraudPipeline
    from .forecasting import TreasuryForecaster
    from .underwriting import Underwriter

    pipeline = pipeline or FraudPipeline(dataset)
    underwriter = underwriter or Underwriter(dataset)
    forecaster = forecaster or TreasuryForecaster(dataset)
    fm = pipeline.holdout_metrics
    fc = forecaster.forecast()

    return [
        {
            "name": "Fraud Ensemble",
            "family": "Isolation Forest + Gradient Boosting",
            "task": "Card fraud detection",
            "primary_metric": {"name": "ROC-AUC", "value": round(fm.roc_auc, 3)},
            "secondary": {"PR-AUC": round(fm.pr_auc, 3), "precision": round(fm.precision, 3),
                          "recall": round(fm.recall, 3)},
            "n_features": len(pipeline.model.columns),
            "supervised": True, "status": "production",
            "description": "Blends unsupervised anomaly detection with supervised "
                           "gradient-boosted trees over behavioral, velocity and graph features.",
        },
        {
            "name": "Fraud-Ring Graph",
            "family": "Graph analytics (networkx)",
            "task": "Fraud-ring / collusion detection",
            "primary_metric": {"name": "rings detected", "value": len(pipeline.rings())},
            "secondary": {"method": "shared-device + cross-metro IP components"},
            "n_features": 0, "supervised": False, "status": "production",
            "description": "Links Stripe Issuing cards by shared devices and cross-metro IPs; flags "
                           "connected components as candidate rings, excluding office VPNs.",
        },
        {
            "name": "Credit-Risk PD Model",
            "family": "Gradient Boosting",
            "task": "Underwriting probability-of-loss",
            "primary_metric": {"name": "ROC-AUC", "value": round(underwriter.model_auc, 3)},
            "secondary": {"portfolio": underwriter.portfolio,
                          "default_rate": round(underwriter.default_rate, 3)},
            "n_features": 8, "supervised": True, "status": "production",
            "description": "Trained on a synthetic lending portfolio; scores PD from cash "
                           "coverage, revenue/spend, volatility, runway and headcount.",
        },
        {
            "name": "Treasury Forecaster",
            "family": "Ridge regression (calendar + lag features)",
            "task": "Cash-flow / runway forecasting",
            "primary_metric": {"name": "backtest MAPE", "value": round(fc.backtest_mape, 4)},
            "secondary": {"runway_months": fc.runway_months, "horizon_days": forecaster.horizon},
            "n_features": 10, "supervised": True, "status": "production",
            "description": "Forecasts daily Stripe Treasury balance with weekly-seasonality features "
                           "and a backtested confidence band; drives runway + yield sweeps.",
        },
        {
            "name": "Causal Explainer",
            "family": "Counterfactual attribution (do-operator)",
            "task": "Fraud-score explanation",
            "primary_metric": {"name": "drivers / alert", "value": 4},
            "secondary": {"method": "baseline interventions"},
            "n_features": len(pipeline.model.columns), "supervised": False, "status": "production",
            "description": "Explains each fraud score by setting features to their legit "
                           "baseline and measuring the risk drop — analyst-grade reasons.",
        },
        {
            "name": "Policy Rule Engine",
            "family": "Deterministic rules",
            "task": "Spend-policy compliance",
            "primary_metric": {"name": "reference", "value": "1.000"},
            "secondary": {"graded_against": "Claude Opus/Sonnet/Haiku in Benchmark"},
            "n_features": 0, "supervised": False, "status": "reference",
            "description": "Canonical controller rulebook; the ground-truth reference LLM "
                           "agents are benchmarked against on financial correctness.",
        },
        {
            "name": "Recurring-Spend Detector",
            "family": "Inter-arrival cadence analysis",
            "task": "Subscription / SaaS detection",
            "primary_metric": {"name": "method", "value": "cadence + amount stability"},
            "secondary": {},
            "n_features": 0, "supervised": False, "status": "production",
            "description": "Detects monthly/weekly/quarterly subscriptions and redundant "
                           "licenses from charge cadence and amount consistency.",
        },
    ]
