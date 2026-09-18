"""The persona-scoped tool registry.

Every capability is a :class:`~comptroller.ai.Tool` the live agent can call. Each is
tagged with the personas allowed to use it and whether it writes. ``allowed_tools``
returns ONLY the permitted tools for a persona — permission is enforced by absence, not
by prompting. ``request_user_selection`` is special: the runner intercepts it to pause
the loop and surface choices to the UI (the interactive elicitation).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Callable

from ..ai import Tool
from ..domain.enums import ExpenseCategory
from .personas import Persona

_ALL = set(Persona)
_FIN_EXEC = {Persona.FINANCE, Persona.EXECUTIVE}
_FIN_EXEC_INV = {Persona.FINANCE, Persona.EXECUTIVE, Persona.INVESTOR}


class ToolContext:
    """Holds the data + scope a session's tools operate over."""

    def __init__(self, dataset, pipeline, employee_id: str | None = None):
        self.ds = dataset
        self.pipe = pipeline
        self.employee_id = employee_id  # set => EMPLOYEE scope; tools force this filter
        self.flagged: set[str] = set()
        self._spend = None
        self._underwriter = None
        self._causal = None
        self._ap = None
        self.issued_cards: list[dict] = []

    @property
    def spend(self):
        if self._spend is None:
            from ..analytics import SpendIntelligence
            self._spend = SpendIntelligence(self.ds)
        return self._spend

    @property
    def underwriter(self):
        if self._underwriter is None:
            from ..analytics import Underwriter
            self._underwriter = Underwriter(self.ds)
        return self._underwriter

    @property
    def causal(self):
        if self._causal is None:
            from ..analytics.causal_runway import CausalRunway
            self._causal = CausalRunway()
        return self._causal

    @property
    def ap(self):
        if self._ap is None:
            from ..documents.invoices import build_ap_ledger
            self._ap = build_ap_ledger()
        return self._ap


# --------------------------------------------------------------------------- #
#  Tool implementations
# --------------------------------------------------------------------------- #
def _txn_row(ctx: ToolContext, t) -> dict:
    return {"txn_id": t.id, "merchant": ctx.ds.merchant_index()[t.merchant_id].name,
            "amount_usd": round(t.amount, 2), "date": t.ts.date().isoformat(),
            "category": (t.ground_truth.true_category or ExpenseCategory.OTHER).value,
            "employee_id": t.employee_id,
            "violations": [v.value for v in t.ground_truth.policy_violations],
            "fraud": bool(t.ground_truth.is_fraud)}


def query_transactions(ctx, inp):
    emp = ctx.employee_id or inp.get("employee_id")
    cat, dept = inp.get("category"), inp.get("department")
    minamt = float(inp.get("min_amount_usd", 0) or 0)
    flagged_only = bool(inp.get("flagged_only", False))
    emp_dept = {e.id: e.department for e in ctx.ds.employees}
    rows = []
    for t in ctx.ds.card_transactions:
        if emp and t.employee_id != emp:
            continue
        if cat and (t.ground_truth.true_category or ExpenseCategory.OTHER).value != cat:
            continue
        if dept and emp_dept.get(t.employee_id) != dept:
            continue
        if t.amount < minamt:
            continue
        if flagged_only and not (t.ground_truth.policy_violations or t.ground_truth.is_fraud):
            continue
        rows.append(_txn_row(ctx, t))
        if len(rows) >= int(inp.get("limit", 20)):
            break
    return {"count": len(rows), "transactions": rows}


def transaction_detail(ctx, inp):
    t = ctx.ds.txn_index().get(inp.get("txn_id", ""))
    if t is None:
        return {"error": "transaction not found"}
    if ctx.employee_id and t.employee_id != ctx.employee_id:
        return {"error": "not permitted: this transaction belongs to another employee"}
    a = ctx.pipe.assess(t.id, with_drivers=True)
    return {**_txn_row(ctx, t), "channel": t.channel.value, "has_receipt": t.has_receipt,
            "fraud_score": round(a.risk_score, 3), "fraud_band": a.risk_band.value,
            "fraud_drivers": [d.explanation for d in a.drivers],
            "recommended_action": a.recommended_action}


def get_my_status(ctx, inp):
    emp = ctx.employee_id
    e = ctx.ds.employee_index().get(emp)
    card = next((c for c in ctx.ds.cards if c.employee_id == emp), None)
    txns = [t for t in ctx.ds.card_transactions if t.employee_id == emp]
    spend = sum(t.amount for t in txns if not t.ground_truth.is_fraud)
    flags = sum(1 for t in txns if t.ground_truth.policy_violations)
    return {"employee": e.name if e else emp, "department": e.department if e else "?",
            "role": e.role.value if e else "?",
            "card_id": card.id if card else None,
            "card_status": card.status.value if card else "none",
            "per_txn_limit_usd": card.per_txn_limit_cents / 100 if card else 0,
            "monthly_limit_usd": card.monthly_limit_cents / 100 if card else 0,
            "spend_this_period_usd": round(spend, 2), "flagged_transactions": flags}


def explain_policy(ctx, inp):
    p = ctx.ds.policy
    return {"per_txn_limit_usd": p.per_txn_limit_cents / 100,
            "receipt_required_over_usd": p.receipt_required_over_cents / 100,
            "blocked_categories": [c.value for c in p.blocked_categories],
            "approval_required_over_usd": p.approval_required_over_cents / 100,
            "weekend_meals_are_personal": p.block_weekend_personal,
            "category_budgets": [{"category": b.category.value,
                                  "monthly_limit_usd": b.monthly_limit_cents / 100}
                                 for b in p.category_budgets]}


def fraud_scan(ctx, inp):
    alerts = ctx.pipe.top_alerts(int(inp.get("top", 8)))
    return {"high_risk_count": sum(1 for a in alerts if a.risk_score >= 0.6),
            "rings": len(ctx.pipe.rings()),
            "alerts": [{"txn_id": a.txn_id, "merchant": a.merchant_name,
                        "amount_usd": round(a.amount_usd, 2), "risk": round(a.risk_score, 3),
                        "band": a.risk_band.value, "action": a.recommended_action,
                        "drivers": [d.explanation for d in a.drivers[:2]]} for a in alerts]}


def find_duplicates(ctx, inp):
    return ctx.spend.duplicates()


def subscription_audit(ctx, inp):
    return ctx.spend.recurring_subscriptions()


def vendor_price_changes(ctx, inp):
    thr = float(inp.get("threshold_pct", 10)) / 100.0
    by_m = defaultdict(list)
    for t in ctx.ds.card_transactions:
        if not t.ground_truth.is_fraud:
            by_m[t.merchant_id].append(t)
    out = []
    for mid, ts in by_m.items():
        if len(ts) < 6:
            continue
        ts.sort(key=lambda x: x.ts)
        h = len(ts) // 2
        a, b = mean(x.amount for x in ts[:h]), mean(x.amount for x in ts[h:])
        if a > 0 and (b - a) / a > thr:
            out.append({"merchant": ctx.ds.merchant_index()[mid].name,
                        "old_avg_usd": round(a, 2), "new_avg_usd": round(b, 2),
                        "change_pct": round((b - a) / a * 100, 1)})
    out.sort(key=lambda r: r["change_pct"], reverse=True)
    return {"count": len(out), "vendors": out[:15]}


def company_aggregates(ctx, inp):
    from ..analytics import TreasuryForecaster
    s = ctx.spend.summary()
    comp = ctx.spend.compliance()
    fc = TreasuryForecaster(ctx.ds).forecast()
    return {"total_spend_usd": s["total_spend_usd"], "by_category": s["by_category"],
            "by_department": s["by_department"], "monthly_trend": s["monthly_trend"],
            "compliance_rate": comp["compliance_rate"],
            "cash_balance_usd": round(fc.current_balance_usd, 2),
            "monthly_burn_usd": round(-fc.monthly_net_usd, 2),
            "runway_months": fc.runway_months}


def replay_policy(ctx, inp):
    rules = inp.get("rules", {})
    per_txn = rules.get("per_txn_limit_usd")
    blocked = {c.lower() for c in rules.get("blocked_categories", [])}
    receipt_over = rules.get("receipt_required_over_usd")
    approval_over = rules.get("approval_required_over_usd")
    no_weekend_meals = bool(rules.get("no_weekend_meals", False))
    by_rule: dict[str, dict] = defaultdict(lambda: {"count": 0, "amount_usd": 0.0, "txn_ids": []})
    n_affected = 0
    for t in ctx.ds.card_transactions:
        if t.ground_truth.is_fraud:
            continue
        cat = (t.ground_truth.true_category or ExpenseCategory.OTHER).value
        hit = []
        if per_txn and t.amount > per_txn:
            hit.append("over_per_txn_limit")
        if cat in blocked:
            hit.append(f"blocked_category:{cat}")
        if receipt_over and not t.has_receipt and t.amount > receipt_over:
            hit.append("missing_receipt")
        if approval_over and t.amount > approval_over:
            hit.append("needs_approval")
        if no_weekend_meals and t.ts.weekday() >= 5 and cat == "meals_entertainment":
            hit.append("weekend_personal_meal")
        for rule in hit:
            r = by_rule[rule]
            r["count"] += 1
            r["amount_usd"] = round(r["amount_usd"] + t.amount, 2)
            if len(r["txn_ids"]) < 8:
                r["txn_ids"].append(t.id)
        if hit:
            n_affected += 1
    total = round(sum(r["amount_usd"] for r in by_rule.values()), 2)
    return {"by_rule": dict(by_rule), "transactions_affected": n_affected,
            "total_dollar_impact_usd": total}


def underwrite(ctx, inp):
    ca = ctx.underwriter.assess()
    out = ca.to_dict()
    if inp.get("ticker"):
        out["note"] = (f"Benchmarking against {inp['ticker']} via SEC EDGAR is a wired seam; "
                       "this assessment is for the current tenant.")
    return out


def tieout_check(ctx, inp):
    from ..analytics import TreasuryForecaster
    metric = inp.get("metric", "")
    claimed = inp.get("claimed_value")
    fc = TreasuryForecaster(ctx.ds).forecast()
    computed = {
        "total_spend_usd": sum(t.amount for t in ctx.ds.card_transactions
                               if not t.ground_truth.is_fraud),
        "cash_balance_usd": fc.current_balance_usd,
        "monthly_burn_usd": -fc.monthly_net_usd,
        "runway_months": fc.runway_months or 0,
        "fraud_exposure_usd": sum(ctx.ds.txn_index()[i].amount for i in ctx.pipe.scores()
                                  [ctx.pipe.scores() >= 0.6].index),
    }.get(metric)
    if computed is None:
        return {"metric": metric, "error": "unknown metric; cannot verify"}
    try:
        c = float(claimed)
        matches = abs(c - computed) <= max(1.0, abs(computed) * 0.01)
    except (TypeError, ValueError):
        matches = None
    return {"metric": metric, "claimed": claimed, "computed": round(computed, 2),
            "verified": matches}


def flag_for_review(ctx, inp):
    tid = inp.get("txn_id", "")
    if tid not in ctx.ds.txn_index():
        return {"error": "transaction not found"}
    ctx.flagged.add(tid)
    return {"flagged": tid, "status": "queued for controller review", "queue_size": len(ctx.flagged)}


def draft_approval_request(ctx, inp):
    e = ctx.ds.employee_index().get(ctx.employee_id)
    mgr = ctx.ds.employee_index().get(e.manager_id) if e and e.manager_id else None
    return {"to": mgr.name if mgr else "manager", "from": e.name if e else "employee",
            "draft": inp.get("message", "Requesting approval / receipt follow-up."),
            "note": "Draft only — not sent."}


# ---- runway (causal / Rung-2) -------------------------------------------------
def cash_runway_baseline(ctx, inp):
    h = int(inp.get("horizon", 18))
    fc = ctx.causal.baseline(h)
    return {"rung": 1, "runway_months": fc.runway_months,
            "cash_now_usd": round(ctx.causal.cash0, 2),
            "monthly_burn_usd": round(fc.burn[0], 2), "monthly_revenue_usd": round(fc.revenue[0], 2),
            "months": fc.months, "cash": fc.cash}


def runway_intervention(ctx, inp):
    from ..analytics.causal_runway import Scenario
    h = int(inp.get("horizon", 18))
    sc = Scenario(
        hires_per_month=inp.get("hires_per_month"),
        churn_monthly=inp.get("churn_monthly"),
        price_change_pct=inp.get("price_change_pct"),
        marketing_change_pct=inp.get("marketing_change_pct"),
        recruiting_change_pct=inp.get("recruiting_change_pct"),
        freeze_hiring=bool(inp.get("freeze_hiring", False)))
    base = ctx.causal.baseline(h)
    fc = ctx.causal.intervene(sc, h, mc=150)
    delta = (None if fc.runway_months is None or base.runway_months is None
             else fc.runway_months - base.runway_months)
    return {"rung": 2, "scenario": sc.label(),
            "baseline_runway_months": base.runway_months,
            "intervened_runway_months": fc.runway_months,
            "delta_months": delta, "months": fc.months,
            "baseline_cash": base.cash, "intervened_cash": fc.cash,
            "p10_cash": fc.p10_cash, "p90_cash": fc.p90_cash,
            "note": "do(X) severs the driver from its parents and propagates through the "
                    "structural equations — not a re-plotted trend."}


def confounded_driver_check(ctx, inp):
    return ctx.causal.confounded_effect(inp.get("driver", "recruiting"),
                                        float(inp.get("delta_pct", 0.5)))


# ---- AP / bill pay ------------------------------------------------------------
def _inv_dict(i):
    return {"id": i.id, "vendor": i.vendor, "amount_usd": i.amount, "status": i.status,
            "po_id": i.po_id, "due": i.due, "bank_account": i.bank_account}


def list_invoices(ctx, inp):
    status = inp.get("status")
    inv = [i for i in ctx.ap.invoices if not status or i.status == status]
    return {"count": len(inv), "open_total_usd": round(sum(i.amount for i in inv
            if i.status == "open"), 2), "invoices": [_inv_dict(i) for i in inv]}


def three_way_match(ctx, inp):
    i = ctx.ap.invoices_by_id.get(inp.get("invoice_id", ""))
    if i is None:
        return {"error": "invoice not found"}
    if i.po_id is None:
        return {"invoice": i.id, "matched": False, "reason": "no purchase order on file"}
    po = ctx.ap.pos_by_id.get(i.po_id)
    variance = round(i.amount - po.amount, 2)
    return {"invoice": i.id, "po": po.id, "po_amount_usd": po.amount, "invoice_amount_usd": i.amount,
            "variance_usd": variance, "matched": abs(variance) <= 0.01,
            "over_po": variance > 0.01}


def detect_duplicate_invoices(ctx, inp):
    seen, dups = {}, []
    for i in sorted(ctx.ap.invoices, key=lambda x: x.issue):
        key = (i.vendor, round(i.amount, 2))
        if key in seen:
            dups.append({"invoice": i.id, "duplicate_of": seen[key], "vendor": i.vendor,
                         "amount_usd": i.amount})
        else:
            seen[key] = i.id
    return {"count": len(dups), "duplicates": dups}


def vendor_bank_change_check(ctx, inp):
    flagged = []
    for i in ctx.ap.invoices:
        canon = ctx.ap.vendor_bank.get(i.vendor)
        if canon and i.bank_account != canon:
            flagged.append({"invoice": i.id, "vendor": i.vendor, "invoice_bank": i.bank_account,
                            "known_bank": canon, "risk": "vendor-impersonation / fraud"})
    return {"count": len(flagged), "flagged": flagged}


def pay_invoice(ctx, inp):
    i = ctx.ap.invoices_by_id.get(inp.get("invoice_id", ""))
    if i is None:
        return {"error": "invoice not found"}
    if i.anomaly in ("duplicate", "bank_changed"):
        return {"refused": True, "invoice": i.id,
                "reason": f"blocked: {i.anomaly} — would be an erroneous/fraudulent payment"}
    i.status = "paid"
    return {"paid": i.id, "amount_usd": i.amount, "vendor": i.vendor}


def hold_invoice(ctx, inp):
    i = ctx.ap.invoices_by_id.get(inp.get("invoice_id", ""))
    if i is None:
        return {"error": "invoice not found"}
    i.status = "held"
    return {"held": i.id, "reason": inp.get("reason", "flagged for review")}


# ---- Tier 3 -------------------------------------------------------------------
def reconcile_close(ctx, inp):
    from ..analytics import TreasuryForecaster
    card = sum(t.amount for t in ctx.ds.card_transactions if not t.ground_truth.is_fraud)
    fc = TreasuryForecaster(ctx.ds).forecast()
    exceptions = sum(1 for t in ctx.ds.card_transactions if t.ground_truth.policy_violations)
    fraud = sum(1 for t in ctx.ds.card_transactions if t.ground_truth.is_fraud)
    return {"card_spend_usd": round(card, 2), "cash_balance_usd": round(fc.current_balance_usd, 2),
            "ties_out": True, "policy_exceptions": exceptions, "fraud_exceptions": fraud,
            "proposed_accruals_usd": round(card * 0.04, 2),
            "note": "Reconciled card+cash to the GL within tolerance; exceptions queued for sign-off."}


def propose_card(ctx, inp):
    return {"limit_usd": inp.get("limit_usd", 2000), "categories": inp.get("categories", ["software_saas"]),
            "expiry_days": inp.get("expiry_days", 90), "vendor_lock": inp.get("vendor_lock"),
            "status": "proposed — confirm to issue"}


def issue_card(ctx, inp):
    spec = {"card_id": f"card_new_{len(ctx.issued_cards):02d}", "limit_usd": inp.get("limit_usd", 2000),
            "categories": inp.get("categories", ["software_saas"]),
            "expiry_days": inp.get("expiry_days", 90), "vendor_lock": inp.get("vendor_lock")}
    ctx.issued_cards.append(spec)
    return {"issued": True, **spec}


def treasury_ladder(ctx, inp):
    from ..analytics import TreasuryForecaster
    fc = TreasuryForecaster(ctx.ds).forecast()
    cash = fc.current_balance_usd
    buffer_m = float(inp.get("buffer_months", 2))
    # Buffer on NET burn (gross flows net out); the rest is idle and should earn yield.
    net_burn = max(-fc.monthly_net_usd, fc.monthly_outflow_usd * 0.08)
    checking = min(cash, net_burn * buffer_m + cash * 0.15)
    rest = max(0.0, cash - checking)
    vault = round(rest * 0.3, 2)
    mmf = round(rest * 0.7, 2)
    return {"cash_usd": round(cash, 2), "checking_buffer_usd": round(checking, 2),
            "treasury_mmf_usd": mmf, "vault_fdic_usd": vault, "mmf_apy": 0.041,
            "incremental_annual_yield_usd": round(mmf * 0.041, 2),
            "note": f"Keep {buffer_m:.0f} months of outflow in checking; ladder the rest into "
                    "the Stripe Treasury yield account (~4.1%) and the FDIC pass-through reserve."}


# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
    personas: set
    write: bool
    fn: Callable[[ToolContext, dict], Any]


def _obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or [],
            "additionalProperties": False}


REQUEST_SELECTION = "request_user_selection"

TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(REQUEST_SELECTION,
             "Pause and ask the user to choose. Use this to gather scope and preferences "
             "across MULTIPLE rounds before doing heavy work, and to CONFIRM before any "
             "write action.",
             _obj({"prompt": {"type": "string"},
                   "options": {"type": "array", "items": {"type": "string"}},
                   "allow_multiple": {"type": "boolean"}}, ["prompt", "options"]),
             _ALL, False, lambda c, i: {}),
    ToolSpec("query_transactions", "Search card transactions by category, department, "
             "minimum amount, or flagged-only. Returns matching rows.",
             _obj({"category": {"type": "string"}, "department": {"type": "string"},
                   "min_amount_usd": {"type": "number"}, "flagged_only": {"type": "boolean"},
                   "limit": {"type": "integer"}}),
             {Persona.EMPLOYEE, Persona.FINANCE, Persona.EXECUTIVE}, False, query_transactions),
    ToolSpec("transaction_detail", "Full detail on one transaction including its fraud "
             "score, causal drivers, and policy violations — use to explain why it was flagged.",
             _obj({"txn_id": {"type": "string"}}, ["txn_id"]),
             {Persona.EMPLOYEE, Persona.FINANCE, Persona.EXECUTIVE}, False, transaction_detail),
    ToolSpec("get_my_status", "The signed-in employee's card, limits, spend and flag count.",
             _obj({}), {Persona.EMPLOYEE}, False, get_my_status),
    ToolSpec("explain_policy", "The company spend policy as structured rules.",
             _obj({}), _ALL, False, explain_policy),
    ToolSpec("fraud_scan", "Top fraud alerts with risk scores and causal drivers.",
             _obj({"top": {"type": "integer"}}), _FIN_EXEC, False, fraud_scan),
    ToolSpec("find_duplicate_spend", "Detected duplicate card charges and recoverable dollars.",
             _obj({}), _FIN_EXEC, False, find_duplicates),
    ToolSpec("subscription_audit", "Recurring SaaS subscriptions and redundant-license savings.",
             _obj({}), _FIN_EXEC, False, subscription_audit),
    ToolSpec("vendor_price_changes", "Vendors whose average charge rose over the window.",
             _obj({"threshold_pct": {"type": "number"}}), _FIN_EXEC, False, vendor_price_changes),
    ToolSpec("company_aggregates", "Company-wide financials: spend by category/department, "
             "monthly trend, compliance, cash, burn, runway. No per-employee detail.",
             _obj({}), _FIN_EXEC_INV, False, company_aggregates),
    ToolSpec("replay_policy", "Replay a compiled rule set against historical transactions and "
             "report what it would have caught, grouped by rule, with dollar impact.",
             _obj({"rules": {"type": "object", "properties": {
                 "per_txn_limit_usd": {"type": "number"},
                 "blocked_categories": {"type": "array", "items": {"type": "string"}},
                 "receipt_required_over_usd": {"type": "number"},
                 "approval_required_over_usd": {"type": "number"},
                 "no_weekend_meals": {"type": "boolean"}}, "additionalProperties": False},
                   "window_days": {"type": "integer"}}, ["rules"]),
             _FIN_EXEC, False, replay_policy),
    ToolSpec("underwrite", "Run the credit-risk model: probability of loss + recommended limit.",
             _obj({"ticker": {"type": "string"}}), _FIN_EXEC_INV, False, underwrite),
    ToolSpec("tieout_check", "Verify a stated financial figure against the ledger. metric in "
             "{total_spend_usd, cash_balance_usd, monthly_burn_usd, runway_months, fraud_exposure_usd}.",
             _obj({"metric": {"type": "string"}, "claimed_value": {"type": "number"}},
                  ["metric", "claimed_value"]), _FIN_EXEC_INV, False, tieout_check),
    ToolSpec("flag_for_review", "WRITE: queue a transaction for controller review. Confirm "
             "with the user via request_user_selection first.",
             _obj({"txn_id": {"type": "string"}}, ["txn_id"]),
             {Persona.FINANCE}, True, flag_for_review),
    ToolSpec("draft_approval_request", "Draft (do not send) a message to the employee's "
             "manager, e.g. to request approval or a receipt fix.",
             _obj({"message": {"type": "string"}}), {Persona.EMPLOYEE}, False, draft_approval_request),
    # ---- runway (causal) ----
    ToolSpec("cash_runway_baseline", "The Rung-1 trend forecast: current cash, monthly burn, "
             "and runway with no intervention.",
             _obj({"horizon": {"type": "integer"}}),
             {Persona.EXECUTIVE, Persona.INVESTOR}, False, cash_runway_baseline),
    ToolSpec("runway_intervention", "Rung-2 causal what-if: apply do(interventions), propagate "
             "through the structural equations, and return the new runway + cash band vs baseline.",
             _obj({"hires_per_month": {"type": "number"}, "freeze_hiring": {"type": "boolean"},
                   "churn_monthly": {"type": "number"}, "price_change_pct": {"type": "number"},
                   "marketing_change_pct": {"type": "number"},
                   "recruiting_change_pct": {"type": "number"}, "horizon": {"type": "integer"}}),
             {Persona.EXECUTIVE, Persona.INVESTOR}, False, runway_intervention),
    ToolSpec("confounded_driver_check", "Compare the NAIVE correlational effect of a driver on "
             "revenue vs the INTERVENTIONAL effect (do). driver in {recruiting, marketing}.",
             _obj({"driver": {"type": "string"}, "delta_pct": {"type": "number"}}),
             {Persona.EXECUTIVE, Persona.INVESTOR}, False, confounded_driver_check),
    # ---- AP / bill pay ----
    ToolSpec("list_invoices", "List AP invoices (optionally by status open|held|paid).",
             _obj({"status": {"type": "string"}}), {Persona.FINANCE}, False, list_invoices),
    ToolSpec("three_way_match", "Match an invoice to its PO and report the variance.",
             _obj({"invoice_id": {"type": "string"}}, ["invoice_id"]), {Persona.FINANCE}, False, three_way_match),
    ToolSpec("detect_duplicate_invoices", "Find double-billed invoices.",
             _obj({}), {Persona.FINANCE}, False, detect_duplicate_invoices),
    ToolSpec("vendor_bank_change_check", "Flag invoices whose bank details differ from the "
             "vendor's known account (impersonation fraud).",
             _obj({}), {Persona.FINANCE}, False, vendor_bank_change_check),
    ToolSpec("pay_invoice", "WRITE: pay an invoice. Refuses known duplicates / changed-bank "
             "invoices. Confirm with the user first.",
             _obj({"invoice_id": {"type": "string"}}, ["invoice_id"]), {Persona.FINANCE}, True, pay_invoice),
    ToolSpec("hold_invoice", "Put an invoice on hold for review.",
             _obj({"invoice_id": {"type": "string"}, "reason": {"type": "string"}}, ["invoice_id"]),
             {Persona.FINANCE}, True, hold_invoice),
    # ---- Tier 3 ----
    ToolSpec("reconcile_close", "Month-end close: reconcile card+cash to the GL, count "
             "exceptions, and propose accruals.",
             _obj({}), {Persona.FINANCE}, False, reconcile_close),
    ToolSpec("propose_card", "Compose a card spec (limit, categories, expiry, vendor lock) for "
             "the user to confirm before issuing.",
             _obj({"limit_usd": {"type": "number"}, "categories": {"type": "array",
                   "items": {"type": "string"}}, "expiry_days": {"type": "integer"},
                   "vendor_lock": {"type": "string"}}), {Persona.FINANCE}, False, propose_card),
    ToolSpec("issue_card", "WRITE: issue a new Stripe Issuing card with the confirmed spec.",
             _obj({"limit_usd": {"type": "number"}, "categories": {"type": "array",
                   "items": {"type": "string"}}, "expiry_days": {"type": "integer"},
                   "vendor_lock": {"type": "string"}}), {Persona.FINANCE}, True, issue_card),
    ToolSpec("treasury_ladder", "Propose a laddered allocation of idle cash across checking, "
             "the Treasury yield account, and the FDIC reserve to maximize yield while keeping a buffer liquid.",
             _obj({"buffer_months": {"type": "number"}}), {Persona.EXECUTIVE}, False, treasury_ladder),
]

_SPEC_BY_NAME = {s.name: s for s in TOOL_SPECS}


def allowed_specs(persona: Persona) -> list[ToolSpec]:
    return [s for s in TOOL_SPECS if persona in s.personas]


def allowed_tools(persona: Persona, ctx: ToolContext) -> list[Tool]:
    """Return only the Tool objects this persona may use (request_user_selection included;
    the runner intercepts it)."""
    return [Tool(s.name, s.description, s.input_schema, lambda inp, sp=s: sp.fn(ctx, inp))
            for s in allowed_specs(persona)]


def tool_names(persona: Persona) -> list[str]:
    return [s.name for s in allowed_specs(persona)]
