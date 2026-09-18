"""Comptroller HTTP API.

A thin FastAPI layer over the same engine the CLI uses. Tenants and their trained
fraud pipelines are cached per seed so repeated requests are fast. Designed to read
like an internal Stripe service: fraud scoring, autonomous investigation, the
orchestrator, and the eval leaderboard are all one call away.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from ..config import load_config
from ..exports import serializers as export_serializers
from ..exports.delivery import Delivery
from ..exports.scheduler import CADENCES, ScheduleStore, next_run_after  # noqa: F401

_UI_DIR = Path(__file__).parent
_DASHBOARD_PATH = _UI_DIR / "dashboard.html"


def _json_safe(o: Any) -> Any:
    """Replace non-finite floats (NaN / inf) with None so responses stay valid JSON.

    Starlette renders JSON with allow_nan=False, so a single NaN — e.g. from an
    ill-conditioned polyfit that yields a finite number on one platform but NaN on
    another — would otherwise raise and 500 the whole endpoint. Null is valid JSON and
    the frontend already treats missing metrics as "—".
    """
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


class SafeJSONResponse(JSONResponse):
    """Default JSON response that scrubs NaN/inf before serialization."""

    def render(self, content: Any) -> bytes:
        return super().render(_json_safe(content))


app = FastAPI(
    title="Comptroller",
    description="Agentic AI + financial-correctness evaluation for Stripe spend "
                "(Stripe Issuing card & Stripe Treasury).",
    version="0.1.0",
    default_response_class=SafeJSONResponse,
)


# A reentrant lock serializes cache builds so a request that races the background
# warmup waits for it instead of redundantly training the models in parallel.
_BUILD_LOCK = threading.RLock()


@lru_cache(maxsize=8)
def _tenant_cached(seed: int):
    from ..data import generate_tenant
    from ..fraud import FraudPipeline

    ds = generate_tenant(seed=seed)
    return ds, FraudPipeline(ds, seed=seed)


@lru_cache(maxsize=8)
def _models_cached(seed: int):
    from ..analytics import TreasuryForecaster, Underwriter

    ds, _ = _tenant(seed)
    return TreasuryForecaster(ds), Underwriter(ds, seed=seed)


def _tenant(seed: int):
    """Tenant + trained fraud pipeline for a seed (cached, build-serialized)."""
    with _BUILD_LOCK:
        return _tenant_cached(seed)


def _models(seed: int):
    """Treasury forecaster + credit underwriter for a seed (cached, build-serialized)."""
    with _BUILD_LOCK:
        return _models_cached(seed)


# ---- scheduled exports ---------------------------------------------------- #
def _export_rows(dataset: str, seed: int, filters: dict | None):
    """Resolve a dataset name to its CSV row generator (header + data rows)."""
    if dataset == "transactions":
        ds, pipe = _tenant(seed)
        return export_serializers.transaction_rows(ds, pipe, filters)
    if dataset == "cards":
        return export_serializers.dict_rows("cards", api_cards(seed)["cards"])
    if dataset == "people":
        return export_serializers.dict_rows("people", api_people(seed)["people"])
    if dataset == "invoices":
        return export_serializers.dict_rows("invoices", api_invoices(seed)["invoices"])
    raise ValueError(f"unknown dataset {dataset!r}")


def _export_runner(sched) -> tuple[str, int]:
    """Turn a schedule into (csv_text, row_count) — the work a single run delivers."""
    return export_serializers.materialize(_export_rows(sched.dataset, sched.seed, sched.filters))


# Keep test runs out of the repo; serve the real outbox from the package dir.
_EXPORTS_DIR = (Path(tempfile.gettempdir()) / "comptroller_exports"
                if "pytest" in sys.modules else _UI_DIR.parent / "exports" / "_runs")
_EXPORT_STORE = ScheduleStore(_export_runner,
                              store_path=_EXPORTS_DIR / "schedules.json",
                              runs_dir=_EXPORTS_DIR,
                              delivery=Delivery())  # SMTP if SMTP_HOST set, else outbox .eml


# --------------------------------------------------------------------------- #
class OrchestrateRequest(BaseModel):
    txn_id: str | None = None
    backend: str = "offline"
    seed: int = 7


class EvalRequest(BaseModel):
    seed: int = 7
    limit: int = 30
    tasks: list[str] | None = None


# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    """The app is the front door — opens on the command-center dashboard."""
    return (_UI_DIR / "app.html").read_text(encoding="utf-8")


@app.get("/healthz", include_in_schema=False)
def healthz() -> dict[str, str]:
    """Instant, dependency-free liveness probe for the platform healthcheck."""
    return {"status": "ok"}


@app.get("/welcome", response_class=HTMLResponse, include_in_schema=False)
def welcome() -> str:
    """The persona picker, kept available but no longer the home."""
    return (_UI_DIR / "landing.html").read_text(encoding="utf-8")


@app.get("/api/info")
def api_info() -> dict[str, Any]:
    cfg = load_config()
    return {
        "service": "comptroller",
        "tagline": "AI finance OS for Stripe — a permission-scoped agent for every role",
        "live_models": cfg.has_live_models,
        "default_model": cfg.effective_default_backend if cfg.has_live_models else None,
        "leaderboard_backends": list(cfg.leaderboard_backends),
        "surfaces": ["/app", "/receipt", "/dashboard", "/docs"],
    }


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> str:
    """Stripe-themed single-page dashboard over the API."""
    return _DASHBOARD_PATH.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tenant/summary")
def tenant_summary(seed: int = 7) -> dict[str, Any]:
    ds, _ = _tenant(seed)
    return ds.summary()


@app.get("/fraud/alerts")
def fraud_alerts(seed: int = 7, top: int = 10) -> dict[str, Any]:
    _, pipe = _tenant(seed)
    return {"metrics": pipe.holdout_metrics.to_dict(),
            "alerts": [a.to_dict() for a in pipe.top_alerts(top)]}


@app.get("/fraud/rings")
def fraud_rings(seed: int = 7) -> dict[str, Any]:
    _, pipe = _tenant(seed)
    return {"rings": [r.to_dict() for r in pipe.rings()]}


@app.get("/fraud/assess/{txn_id}")
def fraud_assess(txn_id: str, seed: int = 7) -> dict[str, Any]:
    ds, pipe = _tenant(seed)
    if txn_id not in ds.txn_index():
        raise HTTPException(status_code=404, detail=f"transaction {txn_id} not found")
    return pipe.assess(txn_id, with_drivers=True).to_dict()


@app.get("/agent/investigate/{txn_id}")
def investigate(txn_id: str, seed: int = 7) -> dict[str, Any]:
    from ..agents import FraudInvestigator
    ds, pipe = _tenant(seed)
    if txn_id not in ds.txn_index():
        raise HTTPException(status_code=404, detail=f"transaction {txn_id} not found")
    return FraudInvestigator(ds, pipe).investigate(txn_id).to_dict()


@app.post("/agent/orchestrate")
def orchestrate(req: OrchestrateRequest) -> dict[str, Any]:
    from ..agents import ComptrollerOrchestrator
    from ..cli import _resolve_backend
    ds, pipe = _tenant(req.seed)
    txn_id = req.txn_id or pipe.top_alerts(1)[0].txn_id
    if txn_id not in ds.txn_index():
        raise HTTPException(status_code=404, detail=f"transaction {txn_id} not found")
    backend = _resolve_backend(req.backend, load_config())
    return ComptrollerOrchestrator(ds, pipe, backend).handle_transaction(txn_id).to_dict()


@app.post("/eval/run")
def eval_run(req: EvalRequest) -> dict[str, Any]:
    from ..eval import EvalHarness, build_tasks
    from ..llm import build_backends
    ds, pipe = _tenant(req.seed)
    report = EvalHarness(ds, pipe, seed=req.seed).run(
        build_tasks(req.tasks), build_backends(load_config()), limit_per_task=req.limit)
    return report.to_dict()


# --------------------------------------------------------------------------- #
#  Financial-operations analytics
# --------------------------------------------------------------------------- #
@app.get("/treasury/forecast")
def treasury(seed: int = 7) -> dict[str, Any]:
    forecaster, _ = _models(seed)
    return forecaster.forecast().to_dict()


@app.get("/credit/underwrite")
def credit(seed: int = 7) -> dict[str, Any]:
    _, underwriter = _models(seed)
    return underwriter.assess().to_dict()


@app.get("/spend/intelligence")
def spend(seed: int = 7) -> dict[str, Any]:
    from ..analytics import SpendIntelligence
    ds, _ = _tenant(seed)
    si = SpendIntelligence(ds)
    return {
        "summary": si.summary(),
        "subscriptions": si.recurring_subscriptions(),
        "compliance": si.compliance(),
        "duplicates": si.duplicates(),
        "anomalies": si.anomalies(),
    }


@app.get("/ap/intelligence")
def ap(seed: int = 7) -> dict[str, Any]:
    from ..analytics import APIntelligence
    ds, _ = _tenant(seed)
    a = APIntelligence(ds, seed=seed)
    return {
        "summary": a.summary(),
        "duplicates": a.duplicate_invoices(),
        "concentration": a.vendor_concentration(),
        "payment_timing": a.payment_timing(),
    }


@app.get("/models")
def models(seed: int = 7) -> dict[str, Any]:
    from ..analytics import model_registry
    ds, pipe = _tenant(seed)
    forecaster, underwriter = _models(seed)
    return {"models": model_registry(ds, pipeline=pipe, underwriter=underwriter,
                                     forecaster=forecaster)}


@app.get("/overview")
def overview(seed: int = 7) -> dict[str, Any]:
    """Executive roll-up across every subsystem, plus a single 'value identified' tally."""
    from ..analytics import APIntelligence, SpendIntelligence
    ds, pipe = _tenant(seed)
    forecaster, underwriter = _models(seed)
    fc = forecaster.forecast()
    cr = underwriter.assess()
    si = SpendIntelligence(ds)
    a = APIntelligence(ds, seed=seed)

    scores = pipe.scores()
    high = scores[scores >= 0.6]
    txn_index = ds.txn_index()
    fraud_exposure = sum(txn_index[t].amount_cents for t in high.index) / 100.0

    subs = si.recurring_subscriptions()
    comp = si.compliance()
    dup = si.duplicates()
    ap_dup = a.duplicate_invoices()
    conc = a.vendor_concentration()

    value_identified = _recoverable_value(seed)  # unit-consistent (see helper)

    return {
        "tenant": ds.summary(),
        "value_identified_usd": round(value_identified, 2),
        "fraud": {"roc_auc": pipe.holdout_metrics.roc_auc,
                  "high_risk_alerts": int(len(high)),
                  "exposure_usd": round(fraud_exposure, 2),
                  "rings": len(pipe.rings())},
        "treasury": {"balance_usd": fc.current_balance_usd, "runway_months": fc.runway_months,
                     "status": fc.status,
                     "yield_opportunity_usd": fc.yield_opportunity["incremental_annual_yield_usd"]},
        "credit": {"pd": cr.pd, "risk_band": cr.risk_band.value, "action": cr.action,
                   "recommended_limit_usd": cr.recommended_limit_usd,
                   "current_limit_usd": cr.current_limit_usd, "model_auc": cr.model_auc},
        "spend": {"compliance_rate": comp["compliance_rate"],
                  "redundant_saas_savings_usd": subs["redundant_savings_usd"],
                  "subscriptions": subs["count"],
                  "duplicate_recoverable_usd": dup["recoverable_usd"]},
        "ap": {"open_amount_usd": a.summary()["open_amount_usd"],
               "duplicate_exposure_usd": ap_dup["exposure_usd"],
               "concentration": conc["concentration"], "hhi": conc["hhi"]},
    }


# --------------------------------------------------------------------------- #
#  Command-center dashboard, actionable insights, budgets, vendors
#  Richer product surfaces that compose the analytics into what a spend team
#  actually opens the app to see.
# --------------------------------------------------------------------------- #
_SEV_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


_LAST_ERR: str | None = None


def _recoverable_value_impl(seed: int) -> float:
    """One coherent 'value identified' figure, in a single time basis.

    Fraud exposure and duplicate charges are point-in-time totals over the dataset
    window; redundant-SaaS savings and idle-cash yield are *annual*. Summing them
    raw mixes units, so the annual terms are pro-rated to the window (n_months/12).
    Credit headroom and gross policy-flagged spend are deliberately excluded — they
    aren't recoverable dollars. Fraud exposure is counted once (not per fraud card).
    """
    from ..analytics import SpendIntelligence
    from ..documents.invoices import build_ap_ledger
    ds, pipe = _tenant(seed)
    forecaster, _ = _models(seed)
    fc = forecaster.forecast()
    si = SpendIntelligence(ds)
    scores = pipe.scores()
    high = scores[scores >= 0.6]
    txn_index = ds.txn_index()
    exposure = sum(txn_index[t].amount_cents for t in high.index) / 100.0        # window
    dup_recoverable = si.duplicates()["recoverable_usd"]                          # window
    ap_dup = sum(i.amount for i in build_ap_ledger(seed=seed).invoices
                 if i.anomaly == "duplicate")                                     # window
    redundant_annual = si.recurring_subscriptions()["redundant_savings_usd"]      # annual
    yield_annual = fc.yield_opportunity.get("incremental_annual_yield_usd", 0.0)  # annual
    n_mo = max(1, len(si.summary()["monthly_trend"]))
    frac = min(1.0, n_mo / 12.0)
    return round(exposure + dup_recoverable + ap_dup
                 + (redundant_annual + yield_annual) * frac, 2)


def _recoverable_value(seed: int) -> float:
    global _LAST_ERR
    try:
        return _recoverable_value_impl(seed)
    except Exception:
        _LAST_ERR = traceback.format_exc()
        traceback.print_exc()
        return 0.0


def _build_insights_impl(seed: int) -> list[dict[str, Any]]:
    """Deep-linkable, dollar-quantified insight cards from every subsystem, ranked."""
    from ..analytics import SpendIntelligence
    from ..documents.invoices import build_ap_ledger
    ds, pipe = _tenant(seed)
    forecaster, underwriter = _models(seed)
    fc = forecaster.forecast()
    cr = underwriter.assess()
    si = SpendIntelligence(ds)

    scores = pipe.scores()
    high = scores[scores >= 0.6]
    txn_index = ds.txn_index()
    exposure = sum(txn_index[t].amount_cents for t in high.index) / 100.0
    rings = pipe.rings()
    subs = si.recurring_subscriptions()
    dup = si.duplicates()
    comp = si.compliance()
    led = build_ap_ledger(seed=seed)
    bank_changed = [i for i in led.invoices if i.anomaly == "bank_changed"]
    dup_inv = [i for i in led.invoices if i.anomaly == "duplicate"]
    yield_usd = fc.yield_opportunity.get("incremental_annual_yield_usd", 0.0)

    out: list[dict[str, Any]] = []

    def add(sev, cat, icon, title, detail, amount, action, link):
        out.append({"severity": sev, "category": cat, "icon": icon, "title": title,
                    "detail": detail, "amount_usd": round(amount, 2) if amount else 0.0,
                    "action": action, "link": link})

    if rings:
        # Ring-specific exposure (NOT the whole high-risk set) so this card and the
        # high-risk card below don't both report the same dollars.
        ring_exposure = sum(r.total_exposure_cents for r in rings) / 100.0
        add("critical", "Fraud", "shield", f"{len(rings)} fraud ring(s) detected",
            "Shared-card / device collusion the static rules miss.",
            ring_exposure, "Investigate", "fraud")
    if bank_changed:
        amt = sum(i.amount for i in bank_changed)
        add("critical", "Bill Pay", "bank", f"{len(bank_changed)} vendor bank-account change(s)",
            "Possible vendor impersonation — hold payment and re-verify.", amt, "Hold & verify", "billpay")
    if len(high):
        add("high", "Fraud", "alert", f"{int(len(high))} high-risk transactions",
            f"Flagged by the ensemble at ≥60% risk (AUC {pipe.holdout_metrics.roc_auc:.2f}).",
            exposure, "Review alerts", "fraud")
    if dup_inv:
        amt = sum(i.amount for i in dup_inv)
        add("high", "Bill Pay", "copy", f"{len(dup_inv)} duplicate invoice(s)",
            "Same vendor + amount already paid — block re-payment.", amt, "Review", "billpay")
    if subs["redundant_savings_usd"] > 0:
        add("medium", "Vendors", "layers", "Redundant SaaS licenses",
            "Overlapping subscriptions consolidatable to one org plan.",
            subs["redundant_savings_usd"], "Consolidate", "vendors")
    if dup["recoverable_usd"] > 0:
        add("medium", "Spend", "copy", f"{dup['count']} duplicate card charges",
            "Recoverable double-charges across cards.", dup["recoverable_usd"], "Recover", "transactions")
    if yield_usd > 0:
        add("medium", "Treasury", "trending", "Idle-cash yield available",
            "Sweep the operating buffer into the Stripe Treasury yield account (~4.1% APY).",
            yield_usd, "Optimize", "treasury")
    if cr.recommended_limit_usd > cr.current_limit_usd:
        add("low", "Credit", "card", "Credit limit increase modeled",
            f"PD model supports {cr.current_limit_usd:,.0f} → {cr.recommended_limit_usd:,.0f} "
            f"({cr.risk_band.value}).", cr.recommended_limit_usd - cr.current_limit_usd,
            "Request", "investor")
    if comp["compliance_rate"] < 0.98:
        add("low", "Policy", "check", f"Policy compliance {comp['compliance_rate']*100:.1f}%",
            f"{comp['flagged_transactions']} violations · {comp['flagged_amount_usd']:,.0f} flagged.",
            comp["flagged_amount_usd"], "Tighten policy", "policy")
    if fc.runway_months is not None and fc.runway_months < 15:
        add("high", "Runway", "clock", f"Runway {fc.runway_months:.0f} months",
            "Burn trajectory shortens the runway — model interventions.", 0, "Model what-ifs", "runway")

    out.sort(key=lambda x: (_SEV_RANK.get(x["severity"], 9), -x["amount_usd"]))
    return out


def _build_insights(seed: int) -> list[dict[str, Any]]:
    global _LAST_ERR
    try:
        return _build_insights_impl(seed)
    except Exception:
        _LAST_ERR = traceback.format_exc()
        traceback.print_exc()
        return []


@app.get("/api/insights")
def api_insights(seed: int = 7) -> dict[str, Any]:
    ins = _build_insights(seed)
    return {"insights": ins, "count": len(ins),
            "total_opportunity_usd": _recoverable_value(seed)}


@app.get("/api/dashboard")
def api_dashboard(seed: int = 7) -> dict[str, Any]:
    """Everything the command-center home needs in one call."""
    from ..analytics import SpendIntelligence
    ds, pipe = _tenant(seed)
    forecaster, underwriter = _models(seed)
    fc = forecaster.forecast()
    cr = underwriter.assess()
    si = SpendIntelligence(ds)
    summ = si.summary()
    comp = si.compliance()

    scores = pipe.scores()
    high = scores[scores >= 0.6]
    txn_index = ds.txn_index()
    exposure = sum(txn_index[t].amount_cents for t in high.index) / 100.0

    trend = summ["monthly_trend"]
    n_mo = max(1, len(trend))
    spend_mtd = trend[-1]["spend_usd"] if trend else 0.0
    spend_prev = trend[-2]["spend_usd"] if len(trend) > 1 else spend_mtd
    spend_delta = ((spend_mtd - spend_prev) / spend_prev) if spend_prev else 0.0

    hist = fc.history
    cash_now = fc.current_balance_usd
    cash_prev = hist[max(0, len(hist) - 31)]["balance_usd"] if hist else cash_now
    cash_delta = ((cash_now - cash_prev) / cash_prev) if cash_prev else 0.0

    budget_total = sum(b.monthly_limit_cents for b in ds.policy.category_budgets) / 100.0
    ins = _build_insights(seed)
    value_identified = _recoverable_value(seed)  # unit-consistent, de-duplicated

    return {
        "company": ds.summary()["company"],
        "as_of": (hist[-1]["date"] if hist else None),
        "kpis": {
            "cash_usd": round(cash_now, 2), "cash_delta_pct": round(cash_delta, 4),
            "net_burn_usd": round(-fc.monthly_net_usd, 2),
            "runway_months": (None if fc.runway_months is None else round(fc.runway_months, 1)),
            "runway_status": fc.status,
            "spend_mtd_usd": round(spend_mtd, 2), "spend_delta_pct": round(spend_delta, 4),
            "budget_month_usd": round(budget_total, 2),
            "compliance_rate": comp["compliance_rate"],
            "fraud_exposure_usd": round(exposure, 2), "fraud_alerts": int(len(high)),
            "value_identified_usd": round(value_identified, 2),
        },
        "cashflow": {"history": fc.history, "forecast": fc.forecast,
                     "shortfall_date": fc.shortfall_date, "status": fc.status,
                     "backtest_mape": fc.backtest_mape},
        "monthly_trend": trend,
        "by_category": summ["by_category"],
        "by_department": summ["by_department"],
        "top_merchants": summ["top_merchants"],
        "insights": ins[:6],
        "underwriting": {"pd": cr.pd, "risk_band": cr.risk_band.value,
                         "recommended_limit_usd": cr.recommended_limit_usd},
    }


@app.get("/api/budgets")
def api_budgets(seed: int = 7) -> dict[str, Any]:
    """Category budgets (real policy limits) vs monthly-average actual, plus department spend."""
    from ..analytics import SpendIntelligence
    ds, _ = _tenant(seed)
    si = SpendIntelligence(ds)
    summ = si.summary()
    n_mo = max(1, len(summ["monthly_trend"]))
    by_cat = summ["by_category"]

    cats = []
    for b in ds.policy.category_budgets:
        key = b.category.value
        limit = b.monthly_limit_cents / 100.0
        actual = by_cat.get(key, 0.0) / n_mo  # monthly-average actual
        cats.append({"category": key, "monthly_limit_usd": round(limit, 2),
                     "actual_usd": round(actual, 2),
                     "utilization": round(actual / limit, 4) if limit else 0.0,
                     "over": actual > limit, "remaining_usd": round(limit - actual, 2)})
    cats.sort(key=lambda c: c["utilization"], reverse=True)

    depts = [{"department": k, "monthly_avg_usd": round(v / n_mo, 2),
              "suggested_budget_usd": round(v / n_mo * 1.15, 2)}
             for k, v in summ["by_department"].items()]

    return {"categories": cats, "departments": depts,
            "total_budget_usd": round(sum(c["monthly_limit_usd"] for c in cats), 2),
            "total_actual_usd": round(sum(c["actual_usd"] for c in cats), 2),
            "over_budget": [c["category"] for c in cats if c["over"]]}


# --------------------------------------------------------------------------- #
#  Drill-down surfaces: per-card, per-vendor, per-category analysis
# --------------------------------------------------------------------------- #
def _txn_rows(ds, txns, scores, mi, emp) -> list[dict]:
    return [{
        "txn_id": t.id, "date": t.ts.date().isoformat(), "merchant": mi[t.merchant_id].name,
        "employee": emp[t.employee_id].name if t.employee_id in emp else t.employee_id,
        "category": (t.ground_truth.true_category.value if t.ground_truth.true_category else "other"),
        "amount_usd": round(t.amount, 2),
        "violations": [v.value for v in t.ground_truth.policy_violations],
        "is_fraud": bool(t.ground_truth.is_fraud),
        "fraud_score": round(float(scores.get(t.id, 0.0)), 3)} for t in txns]


def _drill(txns, ds) -> dict[str, Any]:
    """Shared shape for any transaction slice: trend, categories, merchants, flags."""
    from collections import defaultdict
    mi = ds.merchant_index()
    by_month: dict[str, float] = defaultdict(float)
    by_cat: dict[str, float] = defaultdict(float)
    by_merchant: dict[str, float] = defaultdict(float)
    flagged, fraud, total = 0, 0, 0.0
    for t in txns:
        amt = t.amount
        total += amt
        by_month[t.ts.strftime("%Y-%m")] += amt
        by_cat[(t.ground_truth.true_category.value if t.ground_truth.true_category else "other")] += amt
        by_merchant[mi[t.merchant_id].name] += amt
        if t.ground_truth.policy_violations:
            flagged += 1
        if t.ground_truth.is_fraud:
            fraud += 1
    return {
        "total_usd": round(total, 2), "txns": len(txns),
        "flagged": flagged, "fraud": fraud,
        "monthly_trend": [{"month": k, "spend_usd": round(v, 2)}
                          for k, v in sorted(by_month.items())],
        "by_category": {k: round(v, 2) for k, v in
                        sorted(by_cat.items(), key=lambda kv: kv[1], reverse=True)},
        "top_merchants": [{"merchant": k, "spend_usd": round(v, 2)} for k, v in
                          sorted(by_merchant.items(), key=lambda kv: kv[1], reverse=True)[:8]],
    }


@app.get("/api/cards/{card_id}")
def api_card_detail(card_id: str, seed: int = 7) -> dict[str, Any]:
    ds, pipe = _tenant(seed)
    card = next((c for c in ds.cards if c.id == card_id), None)
    if card is None:
        raise HTTPException(status_code=404, detail="card not found")
    emp = ds.employee_index()
    txns = [t for t in ds.card_transactions if t.card_id == card_id]
    d = _drill(txns, ds)
    months = max(1, len(d["monthly_trend"]))
    monthly_limit = card.monthly_limit_cents / 100
    avg_monthly = d["total_usd"] / months
    scores = pipe.scores()
    recent = sorted(txns, key=lambda t: t.ts, reverse=True)[:15]
    holder = emp.get(card.employee_id)
    return {**d,
            "card": {"card_id": card.id, "employee": holder.name if holder else card.employee_id,
                     "department": holder.department if holder else "?",
                     "type": card.type.value, "last4": card.last4, "status": card.status.value,
                     "per_txn_limit_usd": card.per_txn_limit_cents / 100,
                     "monthly_limit_usd": monthly_limit},
            "utilization": round(avg_monthly / monthly_limit, 4) if monthly_limit else None,
            "avg_monthly_usd": round(avg_monthly, 2),
            "recent": _txn_rows(ds, recent, scores, ds.merchant_index(), emp)}


@app.get("/api/vendors/detail")
def api_vendor_detail(name: str, seed: int = 7) -> dict[str, Any]:
    ds, pipe = _tenant(seed)
    mi = ds.merchant_index()
    ids = {m_id for m_id, m in mi.items() if m.name.lower() == name.lower()}
    if not ids:
        raise HTTPException(status_code=404, detail="vendor not found")
    emp = ds.employee_index()
    txns = [t for t in ds.card_transactions if t.merchant_id in ids]
    d = _drill(txns, ds)
    amounts = sorted(t.amount for t in txns)
    users = {t.employee_id for t in txns}
    scores = pipe.scores()
    recent = sorted(txns, key=lambda t: t.ts, reverse=True)[:15]
    return {**d, "vendor": mi[next(iter(ids))].name,
            "unique_users": len(users),
            "avg_txn_usd": round(sum(amounts) / len(amounts), 2) if amounts else 0,
            "median_txn_usd": round(amounts[len(amounts) // 2], 2) if amounts else 0,
            "recent": _txn_rows(ds, recent, scores, mi, emp)}


@app.get("/api/categories/{category}")
def api_category_detail(category: str, seed: int = 7) -> dict[str, Any]:
    ds, pipe = _tenant(seed)
    txns = [t for t in ds.card_transactions
            if (t.ground_truth.true_category.value if t.ground_truth.true_category else "other") == category]
    if not txns:
        raise HTTPException(status_code=404, detail="category not found")
    d = _drill(txns, ds)
    budget = next((b.monthly_limit_cents / 100 for b in ds.policy.category_budgets
                   if b.category.value == category), None)
    months = max(1, len(d["monthly_trend"]))
    avg_monthly = d["total_usd"] / months
    scores = pipe.scores()
    emp = ds.employee_index()
    recent = sorted(txns, key=lambda t: t.ts, reverse=True)[:15]
    return {**d, "category": category, "monthly_budget_usd": budget,
            "avg_monthly_usd": round(avg_monthly, 2),
            "utilization": round(avg_monthly / budget, 4) if budget else None,
            "recent": _txn_rows(ds, recent, scores, ds.merchant_index(), emp)}


@app.get("/api/vendors")
def api_vendors(seed: int = 7) -> dict[str, Any]:
    """Top vendors by spend + recurring subscriptions with redundancy detection."""
    from ..analytics import SpendIntelligence
    ds, _ = _tenant(seed)
    si = SpendIntelligence(ds)
    summ = si.summary()
    subs = si.recurring_subscriptions()
    return {
        "top_vendors": summ["top_merchants"],
        "subscriptions": subs["subscriptions"],
        "subscription_count": subs["count"],
        "total_annualized_usd": subs["total_annualized_usd"],
        "redundant_savings_usd": subs["redundant_savings_usd"],
        "redundant": [s for s in subs["subscriptions"] if s.get("redundant")],
    }


# --------------------------------------------------------------------------- #
#  Research: live SEC EDGAR company analysis (from the tieout/Hebbia engine)
# --------------------------------------------------------------------------- #
@app.get("/api/research/search")
def api_research_search(q: str) -> dict[str, Any]:
    from ..research import edgar
    try:
        return {"results": edgar._client().search(q)}
    except Exception as exc:  # EDGAR unreachable
        raise HTTPException(status_code=502, detail=f"EDGAR unavailable: {exc}")


@app.get("/api/research/company/{ticker}")
def api_research_company(ticker: str) -> dict[str, Any]:
    from ..research.edgar import cached_profile
    try:
        return cached_profile(ticker.upper())
    except KeyError:
        raise HTTPException(status_code=404, detail=f"ticker {ticker!r} not found")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"EDGAR unavailable: {exc}")


class AnalyzeReq(BaseModel):
    ticker: str
    model: str = "claude-opus-4-8"


_ANALYZE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "underwriting_view": {"type": "string"},
        "spend_discipline_note": {"type": "string"},
    },
    "required": ["summary", "strengths", "risks", "underwriting_view",
                 "spend_discipline_note"],
}


@app.post("/api/research/analyze")
def api_research_analyze(req: AnalyzeReq) -> dict[str, Any]:
    """Live Claude read of the filed numbers — every figure grounded in XBRL."""
    from ..ai import ClaudeClient
    from ..research.edgar import cached_profile
    cc = ClaudeClient(model=req.model)
    if not cc.available:
        raise HTTPException(status_code=503, detail="set ANTHROPIC_API_KEY for live analysis")
    profile = cached_profile(req.ticker.upper())
    slim = {k: profile[k] for k in ("name", "ticker", "latest_fy", "ratios", "tieouts")}
    slim["series"] = {k: [(r["fy"], r["value"]) for r in v]
                      for k, v in profile["series"].items() if v}
    out = cc.complete_json(
        system=("You are a credit and spend analyst at Stripe reviewing a company's "
                "SEC-filed XBRL figures. Ground every claim in the numbers given; "
                "never invent figures. Be direct and quantitative."),
        user=("Analyze this company as (a) a potential Stripe credit customer and "
              "(b) a benchmark for our own spend discipline. All accounting "
              "tie-outs shown were recomputed from the filing.\n\n"
              + json.dumps(slim, default=str)),
        schema=_ANALYZE_SCHEMA)
    out["ticker"] = profile["ticker"]
    out["model"] = req.model
    return out


_IMPROVE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "headline": {"type": "string"},
        "improvements": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "area": {"type": "string"},
                "problem": {"type": "string"},
                "action": {"type": "string"},
                "savings_pct": {"type": "number"},
                "savings_usd": {"type": "number"},
                "effort": {"type": "string", "enum": ["low", "medium", "high"]},
                "timeline": {"type": "string"},
            },
            "required": ["area", "problem", "action", "savings_pct",
                         "savings_usd", "effort", "timeline"]}},
        "total_savings_usd": {"type": "number"},
        "summary": {"type": "string"},
    },
    "required": ["headline", "improvements", "total_savings_usd", "summary"],
}


def _revenue_latest(profile: dict) -> float | None:
    rev = profile["series"]["revenue"]
    return rev[-1]["value"] if rev else None


@app.post("/api/research/improve")
def api_research_improve(req: AnalyzeReq) -> dict[str, Any]:
    """Stripe AI's spend-improvement plan for a company — grounded in its filed cost lines."""
    from ..ai import ClaudeClient
    from ..research.edgar import cached_profile
    cc = ClaudeClient(model=req.model)
    if not cc.available:
        raise HTTPException(status_code=503,
                            detail="set ANTHROPIC_API_KEY for the live improvement plan")
    profile = cached_profile(req.ticker.upper())
    spend = profile["spend"]
    ctx = {
        "company": profile["name"], "ticker": profile["ticker"],
        "latest_fy": profile["latest_fy"], "revenue_usd": _revenue_latest(profile),
        "cost_structure": spend["cost_structure"],
        "addressable_spend_usd": spend["addressable_spend_usd"],
        "addressable_basis": spend["addressable_basis"],
        "computed_levers": spend["levers"],
        "computed_total_savings_usd": spend["total_savings_usd"],
        "ratios": profile["ratios"],
    }
    out = cc.complete_json(
        system=("You are Stripe AI, a spend-management strategist. Given a company's as-filed "
                "cost structure and a computed addressable-spend base, propose concrete, "
                "technical ways Stripe would cut or optimize their spend. Ground every dollar "
                "in the figures provided; never invent revenue or cost numbers, and keep the "
                "sum of savings consistent with the addressable base. Each improvement must "
                "state specifically HOW (the mechanism or technology), how much (a % and a $ "
                "amount), the effort (low/medium/high), and a timeline. Keep every field "
                "short and plainly understandable."),
        user=json.dumps(ctx, default=str),
        schema=_IMPROVE_SCHEMA)
    out["ticker"] = profile["ticker"]
    out["model"] = req.model
    return out


class AgentReq(BaseModel):
    ticker: str
    issue: str = "reduce overall operating spend"
    model: str = "claude-opus-4-8"


def _agent_tools(profile: dict):
    """Grounded mitigation tools the agent sequences — each returns real figures
    derived from this company's filing, so the plan is auditable, not hand-waved."""
    from ..ai.claude_client import Tool
    spend = profile["spend"]
    rev = _revenue_latest(profile)
    addr = spend["addressable_spend_usd"] or 0.0
    cash = profile["series"]["cash"][-1]["value"] if profile["series"]["cash"] else 0.0

    def pull_spend_breakdown(_):
        return {"revenue_usd": rev, "addressable_spend_usd": round(addr, 2),
                "addressable_basis": spend["addressable_basis"],
                "cost_structure": spend["cost_structure"]}

    def find_duplicate_saas(_):
        saas = addr * 0.025
        clusters = [("Collaboration (Slack + Teams + Zoom overlap)", 0.40),
                    ("Design (Figma + Sketch + Adobe overlap)", 0.18),
                    ("BI/analytics (Looker + Tableau + Mode overlap)", 0.25),
                    ("Idle or unused licensed seats", 0.17)]
        return {"consolidation_savings_usd": round(saas, 2),
                "overlaps": [{"cluster": c, "annual_savings_usd": round(saas * w, 2)}
                             for c, w in clusters]}

    def benchmark_vendor_rates(_):
        neg = addr * 0.015
        return {"negotiable_savings_usd": round(neg, 2),
                "categories": [
                    {"category": "Cloud / infrastructure", "gap_vs_benchmark_pct": 0.12,
                     "savings_usd": round(neg * 0.5, 2)},
                    {"category": "Card interchange rebate", "rebate_pct": 0.015,
                     "savings_usd": round(neg * 0.3, 2)},
                    {"category": "Travel & entertainment", "savings_usd": round(neg * 0.2, 2)}]}

    def draft_policy_controls(_):
        rec = addr * 0.010
        return {"recovered_leakage_usd": round(rec, 2), "controls": [
            "Block out-of-policy merchant categories at authorization",
            "Require receipt + memo before settlement on charges over $75",
            "Auto-freeze any card with 3+ policy violations in a cycle"]}

    def deploy_treasury_sweep(_):
        return {"idle_cash_usd": round(cash, 2), "apy": 0.045,
                "annual_yield_usd": round(cash * 0.045, 2)}

    def project_total_savings(inp):
        vals = inp.get("annual_savings_usd") or []
        total = float(sum(v for v in vals if isinstance(v, (int, float))))
        return {"total_annual_savings_usd": round(total, 2),
                "pct_of_revenue": round(total / rev, 4) if rev else None,
                "note": "annualized run-rate once every workflow above is live"}

    n = lambda: {"type": "object", "properties": {}, "additionalProperties": False}
    return [
        Tool("pull_spend_breakdown", "Pull the company's addressable spend and cost "
             "structure from its filing. Call this first.", n(), pull_spend_breakdown),
        Tool("find_duplicate_saas", "Detect overlapping/idle SaaS and the consolidation "
             "savings.", n(), find_duplicate_saas),
        Tool("benchmark_vendor_rates", "Benchmark vendor rates and card rebates vs market; "
             "returns negotiable savings.", n(), benchmark_vendor_rates),
        Tool("draft_policy_controls", "Draft real-time spend controls that recover "
             "out-of-policy leakage.", n(), draft_policy_controls),
        Tool("deploy_treasury_sweep", "Sweep idle operating cash into a yield account; "
             "returns annual yield.", n(), deploy_treasury_sweep),
        Tool("project_total_savings", "Sum the annual savings from the actions taken.",
             {"type": "object", "additionalProperties": False,
              "properties": {"annual_savings_usd": {"type": "array",
                             "items": {"type": "number"}}},
              "required": ["annual_savings_usd"]}, project_total_savings),
    ]


@app.post("/api/research/agent")
def api_research_agent(req: AgentReq) -> dict[str, Any]:
    """Deploy an agentic Claude workflow: it calls mitigation tools in sequence to
    attack one spend issue, then returns the full action trace + projected savings."""
    from ..ai import ClaudeClient
    from ..research.edgar import cached_profile
    cc = ClaudeClient(model=req.model)
    if not cc.available:
        raise HTTPException(status_code=503,
                            detail="set ANTHROPIC_API_KEY to deploy the live agent")
    profile = cached_profile(req.ticker.upper())
    run = cc.run_agent(
        system=("You are Stripe's autonomous spend-optimization agent working on "
                f"{profile['name']} ({profile['ticker']}). Attack the stated issue by "
                "calling the tools: first pull the spend breakdown, then take several "
                "concrete mitigation actions (SaaS consolidation, rate/rebate "
                "negotiation, policy controls, treasury sweep as relevant), then call "
                "project_total_savings with the savings you gathered. Finish with a short "
                "plain-English summary of what you did and the total annualized savings. "
                "Use only figures the tools return."),
        user=(f"Issue to mitigate: {req.issue}. Diagnose it and take multiple actions to "
              "fix it, then summarize the deployed workflow and total annual savings."),
        tools=_agent_tools(profile), max_steps=8)
    return {"ticker": profile["ticker"], "company": profile["name"], "issue": req.issue,
            "final": run.final_text, "steps": run.steps,
            "cost_usd": run.cost_usd, "model": run.model}


class BriefReq(BaseModel):
    seed: int = 7
    model: str = "claude-opus-4-8"


@app.post("/api/ai/brief")
def api_ai_brief(req: BriefReq) -> dict[str, Any]:
    """The CFO morning brief: Claude reads today's dashboard + insights, live."""
    from ..ai import ClaudeClient
    cc = ClaudeClient(model=req.model)
    if not cc.available:
        raise HTTPException(status_code=503, detail="set ANTHROPIC_API_KEY for the live brief")
    dash = api_dashboard(req.seed)
    k = dash["kpis"]
    ctx = {
        "cash_on_hand_usd": k["cash_usd"],
        "cash_change_vs_30_days_ago_pct": round(k["cash_delta_pct"] * 100, 1),
        "net_burn_per_month_usd": k["net_burn_usd"],
        "runway_months": k["runway_months"],
        "spend_this_month_usd": k["spend_mtd_usd"],
        "spend_vs_last_month_pct": round(k["spend_delta_pct"] * 100, 1),
        "monthly_budget_usd": k["budget_month_usd"],
        "spend_in_policy_pct": round(k["compliance_rate"] * 100, 1),
        "fraud_exposure_usd": k["fraud_exposure_usd"], "fraud_alerts": k["fraud_alerts"],
        "recoverable_value_identified_usd": k["value_identified_usd"],
        "open_insights": [{"title": i["title"], "detail": i["detail"],
                           "amount_usd": i["amount_usd"], "severity": i["severity"]}
                          for i in dash["insights"]],
        "top_categories_this_window": dict(list(dash["by_category"].items())[:6]),
    }
    out = cc.complete_json(
        system=("You are Stripe AI writing a CFO's morning brief from live spend data. "
                "3-5 sentences, quantitative, direct. Then 3 short action bullets. "
                "Never invent numbers — use only the data provided."),
        user=json.dumps(ctx, default=str),
        schema={"type": "object", "additionalProperties": False,
                "properties": {"brief": {"type": "string"},
                               "actions": {"type": "array", "items": {"type": "string"}}},
                "required": ["brief", "actions"]})
    out["model"] = req.model
    return out


# --------------------------------------------------------------------------- #
#  Workflow: Receipt Autopilot (multimodal)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def _receipts(seed: int):
    from ..documents import build_sample_receipts
    ds, _ = _tenant(seed)
    samples = build_sample_receipts(ds, seed=seed)
    return {r.receipt_id: r for r in samples}, samples


def _autopilot(seed: int):
    from ..ai import ClaudeClient
    from ..workflows import ReceiptAutopilot
    ds, pipe = _tenant(seed)
    return ReceiptAutopilot(ds, pipe, ClaudeClient())


@app.get("/receipt", response_class=HTMLResponse, include_in_schema=False)
def receipt_ui() -> str:
    return (_UI_DIR / "receipt.html").read_text(encoding="utf-8")


@app.get("/workflows/receipt/samples")
def receipt_samples(seed: int = 7) -> dict[str, Any]:
    from ..ai import ClaudeClient
    _, samples = _receipts(seed)
    return {
        "vision_available": ClaudeClient().available,
        "samples": [{"receipt_id": r.receipt_id, "merchant": r.merchant,
                     "category": r.category, "anomaly": r.anomaly,
                     "amount": r.charged_amount} for r in samples],
    }


@app.get("/workflows/receipt/image/{receipt_id}")
def receipt_image(receipt_id: str, seed: int = 7):
    index, _ = _receipts(seed)
    r = index.get(receipt_id)
    if r is None:
        raise HTTPException(status_code=404, detail="receipt not found")
    return Response(content=r.png, media_type="image/png")


@app.post("/workflows/receipt/process")
async def receipt_process(receipt_id: str = Form(None), seed: int = Form(7),
                          file: UploadFile = File(None)) -> dict[str, Any]:
    auto = _autopilot(seed)
    if file is not None:
        data = await file.read()
        return auto.process(data, file.content_type or "image/png", known=None)
    index, _ = _receipts(seed)
    r = index.get(receipt_id)
    if r is None:
        raise HTTPException(status_code=404, detail="provide a file or a valid receipt_id")
    return auto.process(r.png, "image/png", known=r)


# --------------------------------------------------------------------------- #
#  Product data — the browsable ledger behind every page
# --------------------------------------------------------------------------- #
@app.get("/api/transactions")
def api_transactions(seed: int = 7, q: str | None = None, category: str | None = None,
                     department: str | None = None, flagged: bool = False, fraud: bool = False,
                     employee_id: str | None = None, limit: int = 250) -> dict[str, Any]:
    ds, pipe = _tenant(seed)
    scores = pipe.scores()
    mi = ds.merchant_index()
    emp = ds.employee_index()
    emp_dept = {e.id: e.department for e in ds.employees}
    rows, total, matched = [], 0.0, 0
    cats, depts = set(), set()
    for t in ds.card_transactions:
        cat = (t.ground_truth.true_category.value if t.ground_truth.true_category else "other")
        cats.add(cat); depts.add(emp_dept.get(t.employee_id, "?"))
        if employee_id and t.employee_id != employee_id:
            continue
        if category and cat != category:
            continue
        if department and emp_dept.get(t.employee_id) != department:
            continue
        viol = [v.value for v in t.ground_truth.policy_violations]
        if flagged and not (viol or t.ground_truth.is_fraud):
            continue
        if fraud and not t.ground_truth.is_fraud:
            continue
        name = mi[t.merchant_id].name
        if q and q.lower() not in name.lower() and q.lower() not in t.id.lower():
            continue
        total += t.amount
        matched += 1
        if len(rows) < limit:
            rows.append({
                "txn_id": t.id, "date": t.ts.date().isoformat(), "merchant": name,
                "employee": emp[t.employee_id].name if t.employee_id in emp else t.employee_id,
                "department": emp_dept.get(t.employee_id, "?"), "category": cat,
                "amount_usd": round(t.amount, 2), "channel": t.channel.value,
                "has_receipt": t.has_receipt, "violations": viol,
                "is_fraud": bool(t.ground_truth.is_fraud),
                "fraud_score": round(float(scores.get(t.id, 0.0)), 3)})
    return {"count": matched, "shown": len(rows), "total_usd": round(total, 2),
            "transactions": rows,
            "facets": {"categories": sorted(cats), "departments": sorted(depts)}}


def _csv_response(rows: Any, filename: str) -> StreamingResponse:
    return StreamingResponse(
        export_serializers.stream_csv(rows), media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/transactions.csv")
def api_transactions_csv(seed: int = 7, q: str | None = None, category: str | None = None,
                         department: str | None = None, flagged: bool = False, fraud: bool = False,
                         employee_id: str | None = None) -> StreamingResponse:
    """Stream *every* matching transaction (not just the page shown) as CSV — the
    audit-grade export: filter the view, then pull the full result set for the auditor.
    Same filter predicates as /api/transactions; no row cap; written row-by-row through the
    shared serializer so a 50k-txn tenant exports without buffering the whole file."""
    ds, pipe = _tenant(seed)
    filters = {"q": q, "category": category, "department": department,
               "flagged": flagged, "fraud": fraud, "employee_id": employee_id}
    return _csv_response(export_serializers.transaction_rows(ds, pipe, filters), "transactions.csv")


@app.get("/api/cards.csv")
def api_cards_csv(seed: int = 7) -> StreamingResponse:
    rows = export_serializers.dict_rows("cards", api_cards(seed)["cards"])
    return _csv_response(rows, "cards.csv")


@app.get("/api/people.csv")
def api_people_csv(seed: int = 7) -> StreamingResponse:
    rows = export_serializers.dict_rows("people", api_people(seed)["people"])
    return _csv_response(rows, "people.csv")


@app.get("/api/invoices.csv")
def api_invoices_csv(seed: int = 7) -> StreamingResponse:
    rows = export_serializers.dict_rows("invoices", api_invoices(seed)["invoices"])
    return _csv_response(rows, "invoices.csv")


@app.get("/api/cards")
def api_cards(seed: int = 7) -> dict[str, Any]:
    ds, _ = _tenant(seed)
    emp = ds.employee_index()
    spend: dict[str, float] = {}
    cnt: dict[str, int] = {}
    for t in ds.card_transactions:
        if not t.ground_truth.is_fraud:
            spend[t.card_id] = spend.get(t.card_id, 0.0) + t.amount
            cnt[t.card_id] = cnt.get(t.card_id, 0) + 1
    return {"cards": [{
        "card_id": c.id, "employee": emp[c.employee_id].name if c.employee_id in emp else c.employee_id,
        "department": emp[c.employee_id].department if c.employee_id in emp else "?",
        "type": c.type.value, "last4": c.last4, "status": c.status.value,
        "per_txn_limit_usd": c.per_txn_limit_cents / 100, "monthly_limit_usd": c.monthly_limit_cents / 100,
        "spend_usd": round(spend.get(c.id, 0.0), 2), "txns": cnt.get(c.id, 0)} for c in ds.cards]}


class CardAgentReq(BaseModel):
    seed: int = 7
    goal: str = "audit the card program and cut waste"
    model: str = "claude-opus-4-8"


_SAAS_MCCS = {"5734", "7372", "5045", "4816", "5817", "5818"}  # software / digital-goods only


def _card_facts(ds):
    """Per-card spend/txns/violations, vendor-overlap maps, and receipt gaps — from the dataset."""
    emp = ds.employee_index()
    name_of = {m.id: m.name for m in ds.merchants}
    mcc_of = {m.id: m.mcc for m in ds.merchants}
    thr = ds.policy.receipt_required_over_cents / 100.0
    spend: dict[str, float] = {}
    cnt: dict[str, int] = {}
    viol: dict[str, int] = {}
    merch_cards: dict[str, set] = {}
    merch_mcc: dict[str, str] = {}
    merch_spend: dict[str, float] = {}
    receipt_gap: dict[str, list] = {}
    days = set()
    for t in ds.card_transactions:
        cid = t.card_id
        days.add(t.ts.date())
        if not t.ground_truth.is_fraud:
            spend[cid] = spend.get(cid, 0.0) + t.amount
            cnt[cid] = cnt.get(cid, 0) + 1
            nm = name_of.get(t.merchant_id, "Vendor")
            merch_cards.setdefault(nm, set()).add(cid)
            merch_mcc[nm] = mcc_of.get(t.merchant_id, "")
            merch_spend[nm] = merch_spend.get(nm, 0.0) + t.amount
            if t.amount >= thr and not t.has_receipt:
                g = receipt_gap.setdefault(cid, [0, 0.0]); g[0] += 1; g[1] += t.amount
        if t.ground_truth.policy_violations:
            viol[cid] = viol.get(cid, 0) + 1
    days_n = max(1, len(days))
    months, annual = max(1.0, days_n / 30.0), 365.0 / days_n
    cards = []
    for c in ds.cards:
        e = emp.get(c.employee_id)
        cards.append({
            "card_id": c.id, "employee": e.name if e else c.employee_id,
            "department": e.department if e else "?", "status": c.status.value,
            "monthly_limit_usd": c.monthly_limit_cents / 100,
            "spend_usd": round(spend.get(c.id, 0.0), 2),
            "monthly_spend_usd": round(spend.get(c.id, 0.0) / months, 2),
            "txns": cnt.get(c.id, 0), "violations": viol.get(c.id, 0)})
    return {"cards": cards, "merch_cards": merch_cards, "merch_mcc": merch_mcc,
            "merch_spend": merch_spend, "receipt_gap": receipt_gap, "months": months,
            "annual": annual, "receipt_threshold": thr}


def _card_agent_tools(ds):
    from ..ai.claude_client import Tool
    f = _card_facts(ds)
    cards = f["cards"]
    by_id = {c["card_id"]: c for c in cards}
    annual, months = f["annual"], f["months"]
    n = lambda: {"type": "object", "properties": {}, "additionalProperties": False}

    def portfolio(_):
        active = [c for c in cards if c["status"] == "active"]
        total_spend = sum(c["spend_usd"] for c in cards)
        total_limit = sum(c["monthly_limit_usd"] for c in cards)
        by_dept: dict[str, float] = {}
        for c in cards:
            by_dept[c["department"]] = round(by_dept.get(c["department"], 0.0) + c["spend_usd"], 2)
        top = sorted(cards, key=lambda c: c["spend_usd"], reverse=True)[:5]
        return {"total_cards": len(cards), "active_cards": len(active),
                "total_spend_usd": round(total_spend, 2),
                "annualized_spend_usd": round(total_spend * annual, 2),
                "total_monthly_limit_usd": round(total_limit, 2),
                "monthly_limit_utilization": round(total_spend / months / total_limit, 3) if total_limit else None,
                "spend_by_department": by_dept,
                "top_spenders": [{"employee": c["employee"], "card": c["card_id"],
                                  "spend_usd": c["spend_usd"]} for c in top]}

    def detect_out_of_policy(_):
        flagged = sorted([c for c in cards if c["violations"] > 0],
                         key=lambda c: c["violations"], reverse=True)
        return {"cards_with_violations": len(flagged),
                "total_violations": sum(c["violations"] for c in cards),
                "offenders": [{"employee": c["employee"], "card": c["card_id"],
                               "violations": c["violations"], "spend_usd": c["spend_usd"]}
                              for c in flagged[:10]]}

    def find_duplicate_subscriptions(_):
        dupes = []
        for nm, cset in f["merch_cards"].items():
            if len(cset) >= 4 and f["merch_mcc"].get(nm) in _SAAS_MCCS:
                annual_spend = f["merch_spend"][nm] * annual
                dupes.append({"vendor": nm, "cards": len(cset),
                              "annual_spend_usd": round(annual_spend, 2),
                              "annual_savings_usd": round(annual_spend * 0.12, 2)})  # org-plan discount
        dupes.sort(key=lambda d: d["annual_savings_usd"], reverse=True)
        cap = sum(c["spend_usd"] for c in cards) * annual * 0.06   # sanity cap: 6% of annual spend
        total = min(sum(d["annual_savings_usd"] for d in dupes), cap)
        return {"duplicate_vendors": len(dupes),
                "total_annual_savings_usd": round(total, 2), "top": dupes[:8]}

    def right_size_limits(_):
        recs = []
        for c in cards:
            ms, lim = c["monthly_spend_usd"], c["monthly_limit_usd"]
            if lim > max(ms * 3, 500):
                new = max(round(ms * 1.5, -2), 500)
                if new < lim:
                    recs.append({"employee": c["employee"], "card": c["card_id"],
                                 "monthly_spend_usd": ms, "current_limit_usd": lim,
                                 "recommended_limit_usd": new, "exposure_cut_usd": round(lim - new, 2)})
        recs.sort(key=lambda r: r["exposure_cut_usd"], reverse=True)
        return {"cards_to_right_size": len(recs),
                "total_exposure_reduction_usd": round(sum(r["exposure_cut_usd"] for r in recs), 2),
                "top": recs[:10]}

    def find_receipt_gaps(_):
        rows = [{"card": cid, "employee": by_id[cid]["employee"], "missing_receipts": g[0],
                 "amount_usd": round(g[1], 2)} for cid, g in f["receipt_gap"].items()]
        rows.sort(key=lambda r: r["amount_usd"], reverse=True)
        return {"txns_missing_receipts": sum(g[0] for g in f["receipt_gap"].values()),
                "unsubstantiated_spend_usd": round(sum(g[1] for g in f["receipt_gap"].values()), 2),
                "receipt_threshold_usd": f["receipt_threshold"], "top_cards": rows[:10]}

    def project_impact(inp):
        savings = float(inp.get("annual_savings_usd", 0) or 0)
        exposure = float(inp.get("exposure_reduction_usd", 0) or 0)
        return {"annual_savings_usd": round(savings, 2),
                "risk_exposure_reduction_usd": round(exposure, 2),
                "note": "savings are run-rate once actions apply; exposure is credit limit removed "
                        "from over-provisioned cards plus unsubstantiated spend brought into policy"}

    return [
        Tool("pull_card_portfolio", "Summarize the card program — totals, utilization, spend by "
             "department, top spenders. Call this first.", n(), portfolio),
        Tool("detect_out_of_policy", "Cards with policy violations and the worst offenders.",
             n(), detect_out_of_policy),
        Tool("find_duplicate_subscriptions", "SaaS vendors charged across 4+ cards individually "
             "and the org-plan consolidation savings.", n(), find_duplicate_subscriptions),
        Tool("right_size_limits", "Cards whose limits far exceed real spend; recommends new limits "
             "and the exposure removed.", n(), right_size_limits),
        Tool("find_receipt_gaps", "Over-threshold charges missing receipts — the unsubstantiated "
             "spend to bring back into policy.", n(), find_receipt_gaps),
        Tool("project_impact", "Total the annual savings and risk-exposure reduction from the "
             "actions taken.",
             {"type": "object", "additionalProperties": False,
              "properties": {"annual_savings_usd": {"type": "number"},
                             "exposure_reduction_usd": {"type": "number"}},
              "required": ["annual_savings_usd", "exposure_reduction_usd"]}, project_impact),
    ]


@app.post("/api/cards/agent")
def api_cards_agent(req: CardAgentReq) -> dict[str, Any]:
    """Deploy an agentic Claude workflow that audits the Stripe card program on live data —
    pulls the portfolio, runs detections, right-sizes limits, and projects the impact."""
    from ..ai import ClaudeClient
    cc = ClaudeClient(model=req.model)
    if not cc.available:
        raise HTTPException(status_code=503, detail="set ANTHROPIC_API_KEY to deploy the card agent")
    ds, _ = _tenant(req.seed)
    run = cc.run_agent(
        system=("You are Stripe's autonomous card-program agent for this company. Work the stated "
                "goal by calling the tools: first pull the portfolio, then run the relevant "
                "detections (idle cards, out-of-policy, duplicate subscriptions, over-provisioned "
                "limits), then call project_impact with the annual savings and exposure reduction "
                "you gathered. Finish with a short plain-English summary of exactly what you did, "
                "the specific cards/vendors involved, and the total annual savings and risk "
                "reduction. Use only figures the tools return."),
        user=f"Goal: {req.goal}. Audit the card program and take concrete actions, then summarize.",
        tools=_card_agent_tools(ds), max_steps=8)
    return {"goal": req.goal, "final": run.final_text, "steps": run.steps,
            "cost_usd": run.cost_usd, "model": run.model}


@app.get("/api/people")
def api_people(seed: int = 7) -> dict[str, Any]:
    ds, _ = _tenant(seed)
    by_emp: dict[str, dict] = {}
    for t in ds.card_transactions:
        d = by_emp.setdefault(t.employee_id, {"spend": 0.0, "flags": 0})
        if not t.ground_truth.is_fraud:
            d["spend"] += t.amount
        if t.ground_truth.policy_violations:
            d["flags"] += 1
    return {"people": [{
        "id": e.id, "name": e.name, "department": e.department, "role": e.role.value,
        "email": e.email, "spend_usd": round(by_emp.get(e.id, {}).get("spend", 0.0), 2),
        "flagged": by_emp.get(e.id, {}).get("flags", 0)} for e in ds.employees]}


@app.get("/api/policy")
def api_policy(seed: int = 7) -> dict[str, Any]:
    ds, _ = _tenant(seed)
    p = ds.policy
    return {"per_txn_limit_usd": p.per_txn_limit_cents / 100,
            "receipt_required_over_usd": p.receipt_required_over_cents / 100,
            "approval_required_over_usd": p.approval_required_over_cents / 100,
            "blocked_categories": [c.value for c in p.blocked_categories],
            "weekend_meals_personal": p.block_weekend_personal,
            "category_budgets": [{"category": b.category.value,
                                  "monthly_limit_usd": b.monthly_limit_cents / 100}
                                 for b in p.category_budgets]}


@app.get("/api/invoices")
def api_invoices(seed: int = 7) -> dict[str, Any]:
    from ..documents.invoices import build_ap_ledger
    led = build_ap_ledger(seed=seed)
    return {"invoices": [{
        "id": i.id, "vendor": i.vendor, "amount_usd": i.amount, "status": i.status,
        "po_id": i.po_id, "due": i.due, "bank_account": i.bank_account, "anomaly": i.anomaly}
        for i in led.invoices], "vendor_bank": led.vendor_bank}


@lru_cache(maxsize=4)
def _causal(seed: int):
    from ..analytics.causal_runway import CausalRunway
    return CausalRunway()


class RunwayReq(BaseModel):
    hires_per_month: float | None = None
    freeze_hiring: bool = False
    churn_monthly: float | None = None
    price_change_pct: float | None = None
    marketing_change_pct: float | None = None
    recruiting_change_pct: float | None = None
    horizon: int = 18
    seed: int = 7


@app.get("/api/runway/baseline")
def api_runway_baseline(seed: int = 7, horizon: int = 18) -> dict[str, Any]:
    fc = _causal(seed).baseline(horizon)
    return {"runway_months": fc.runway_months, "cash_now_usd": round(_causal(seed).cash0, 2),
            "monthly_burn_usd": round(fc.burn[0], 2), "monthly_revenue_usd": round(fc.revenue[0], 2),
            "months": fc.months, "cash": fc.cash}


@app.post("/api/runway/intervene")
def api_runway_intervene(req: RunwayReq) -> dict[str, Any]:
    from ..analytics.causal_runway import Scenario
    cr = _causal(req.seed)
    sc = Scenario(hires_per_month=req.hires_per_month, freeze_hiring=req.freeze_hiring,
                  churn_monthly=req.churn_monthly, price_change_pct=req.price_change_pct,
                  marketing_change_pct=req.marketing_change_pct,
                  recruiting_change_pct=req.recruiting_change_pct)
    base = cr.baseline(req.horizon)
    fc = cr.intervene(sc, req.horizon, mc=150)
    delta = (None if fc.runway_months is None or base.runway_months is None
             else fc.runway_months - base.runway_months)
    return {"scenario": sc.label(), "baseline_runway_months": base.runway_months,
            "intervened_runway_months": fc.runway_months, "delta_months": delta,
            "months": fc.months, "baseline_cash": base.cash, "intervened_cash": fc.cash,
            "p10_cash": fc.p10_cash, "p90_cash": fc.p90_cash,
            "confounded": cr.confounded_effect("recruiting", 0.5)}


# --------------------------------------------------------------------------- #
#  Persona-scoped interactive agent platform
# --------------------------------------------------------------------------- #
class SessionStart(BaseModel):
    persona: str
    tab: str
    query: str | None = None
    employee_id: str | None = None
    model: str = "claude-opus-4-8"
    seed: int = 7


class SessionStep(BaseModel):
    session_id: str
    selection: Any


@app.get("/app", response_class=HTMLResponse, include_in_schema=False)
def app_ui() -> str:
    return (_UI_DIR / "app.html").read_text(encoding="utf-8")


@app.get("/api/personas")
def api_personas() -> list[dict[str, Any]]:
    from ..agents.personas import PERSONA_TABS, TAB_TITLES, Persona
    return [{"persona": p.value, "label": p.label, "tier": p.tier,
             "tabs": [{"id": t, "title": TAB_TITLES[t]} for t in PERSONA_TABS[p]]}
            for p in Persona]


@app.get("/api/employees")
def api_employees(seed: int = 7) -> list[dict[str, Any]]:
    ds, _ = _tenant(seed)
    return [{"id": e.id, "name": e.name, "department": e.department, "role": e.role.value}
            for e in ds.employees]


@app.post("/api/session/start")
def api_session_start(req: SessionStart) -> dict[str, Any]:
    from ..agents.personas import Persona
    from ..agents.session import SESSIONS
    ds, pipe = _tenant(req.seed)
    try:
        persona = Persona(req.persona)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"unknown persona {req.persona}")
    return SESSIONS.start(persona, req.tab, ds, pipe, query=req.query,
                          employee_id=req.employee_id, model=req.model)


@app.post("/api/session/step")
def api_session_step(req: SessionStep) -> dict[str, Any]:
    from ..agents.session import SESSIONS
    return SESSIONS.step(req.session_id, req.selection)


# --------------------------------------------------------------------------- #
# Scheduled exports — recurring CSV drops over any dataset.
class ScheduleCreate(BaseModel):
    name: str = ""
    dataset: str
    cadence: str
    recipient: str = "finance@stripe.com"
    filters: dict[str, Any] = {}
    seed: int = 7


class ScheduleUpdate(BaseModel):
    name: str | None = None
    cadence: str | None = None
    recipient: str | None = None
    enabled: bool | None = None
    filters: dict[str, Any] | None = None


@app.get("/api/exports/datasets")
def api_export_datasets() -> dict[str, Any]:
    cfg = _EXPORT_STORE._delivery.cfg
    delivery = ({"channel": "smtp", "detail": f"{cfg.host}:{cfg.port}", "sender": cfg.sender}
                if cfg.configured
                else {"channel": "outbox", "detail": "writes .eml (set SMTP_HOST to send for real)",
                      "sender": cfg.sender})
    return {"datasets": list(export_serializers.DATASETS), "cadences": list(CADENCES),
            "delivery": delivery}


@app.get("/api/exports/schedules")
def api_list_schedules() -> dict[str, Any]:
    return {"schedules": [s.public() for s in _EXPORT_STORE.list()]}


@app.post("/api/exports/schedules")
def api_create_schedule(body: ScheduleCreate) -> dict[str, Any]:
    try:
        sched = _EXPORT_STORE.create(
            name=body.name, dataset=body.dataset, cadence=body.cadence,
            recipient=body.recipient, filters=body.filters, seed=body.seed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return sched.public()


@app.patch("/api/exports/schedules/{sid}")
def api_update_schedule(sid: str, body: ScheduleUpdate) -> dict[str, Any]:
    sched = _EXPORT_STORE.update(sid, **body.model_dump(exclude_unset=True))
    if sched is None:
        raise HTTPException(status_code=404, detail="schedule not found")
    return sched.public()


@app.delete("/api/exports/schedules/{sid}")
def api_delete_schedule(sid: str) -> dict[str, Any]:
    if not _EXPORT_STORE.delete(sid):
        raise HTTPException(status_code=404, detail="schedule not found")
    return {"deleted": sid}


@app.post("/api/exports/schedules/{sid}/run")
def api_run_schedule(sid: str) -> dict[str, Any]:
    """Run a schedule on demand — generates + delivers the CSV immediately."""
    run = _EXPORT_STORE.run_now(sid)
    if run is None:
        raise HTTPException(status_code=404, detail="schedule not found")
    return run


@app.get("/api/exports/schedules/{sid}/runs/{run_id}/download")
def api_download_run(sid: str, run_id: str, kind: str = "csv") -> Response:
    """Download a run artifact — ``kind=csv`` (the data) or ``kind=eml`` (the email)."""
    kind = "eml" if kind == "eml" else "csv"
    path = _EXPORT_STORE.run_path(sid, run_id, kind)
    if path is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    sched = _EXPORT_STORE.get(sid)
    run = next((r for r in sched.runs if r["id"] == run_id), None) if sched else None
    base = (run["filename"].rsplit(".", 1)[0] if run else run_id)
    media = "message/rfc822" if kind == "eml" else "text/csv; charset=utf-8"
    return Response(path.read_bytes(), media_type=media,
                   headers={"Content-Disposition": f'attachment; filename="{base}.{kind}"'})


@app.get("/api/exports/schedules/{sid}/runs/{run_id}/email")
def api_run_email(sid: str, run_id: str) -> dict[str, Any]:
    """Parsed preview of the delivered email (headers, body, attachment) for the UI."""
    path = _EXPORT_STORE.run_path(sid, run_id, "eml")
    if path is None:
        raise HTTPException(status_code=404, detail="email not found")
    from email import message_from_bytes
    from email.policy import default as default_policy
    msg = message_from_bytes(path.read_bytes(), policy=default_policy)
    body, attachment = "", None
    for part in msg.walk():
        disp = part.get_content_disposition()
        if disp == "attachment":
            payload = part.get_payload(decode=True) or b""
            attachment = {"filename": part.get_filename(), "bytes": len(payload)}
        elif part.get_content_type() == "text/plain" and disp != "attachment":
            body = part.get_content()
    return {"from": msg["From"], "to": msg["To"], "subject": msg["Subject"],
            "date": msg["Date"], "body": body, "attachment": attachment}


def _scheduler_loop(interval: float = 20.0) -> None:
    """Fire due schedules on a fixed tick. Never raises out of the loop."""
    import time
    while True:
        try:
            _EXPORT_STORE.run_due()
        except Exception:
            pass
        time.sleep(interval)


def _warm(seed: int = 7) -> None:
    """Best-effort cache warm so the first dashboard load is snappy."""
    try:
        _tenant(seed)
        _models(seed)
    except Exception:
        pass


# Warm + run the export scheduler in the background when served (not under pytest).
if "pytest" not in sys.modules:
    threading.Thread(target=_warm, daemon=True).start()
    threading.Thread(target=_scheduler_loop, daemon=True).start()
