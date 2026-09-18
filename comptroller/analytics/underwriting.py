"""Dynamic credit underwriting — Stripe Capital's revenue-based model, made probabilistic.

Stripe Capital underwrites offers from a company's real-time processing volume, cash position and revenue patterns
(no personal guarantee), scaling limits up as cash grows and cutting them within 24h
for at-risk accounts. We make that quantitative: a gradient-boosted probability-of-loss
model trained on a synthetic portfolio, plus a cash-coverage limit recommendation and a
dynamic action (raise / hold / reduce-within-24h).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from ..domain import Dataset
from ..domain.enums import RiskBand

_FEATURES = ["cash_log", "runway", "rev_to_spend", "spend_log",
             "rev_volatility", "spend_growth", "coverage", "headcount"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class CreditAssessment:
    company: str
    cash_balance_usd: float
    current_limit_usd: float
    recommended_limit_usd: float
    utilization: float
    coverage_months: float
    runway_months: float
    pd: float                  # modeled probability of loss
    risk_score: float
    risk_band: RiskBand
    action: str                # increase_limit | hold | reduce_within_24h
    drivers: list[dict]
    model_auc: float
    rationale: str

    def to_dict(self) -> dict:
        return {
            "company": self.company,
            "cash_balance_usd": round(self.cash_balance_usd, 2),
            "current_limit_usd": round(self.current_limit_usd, 2),
            "recommended_limit_usd": round(self.recommended_limit_usd, 2),
            "utilization": round(self.utilization, 4),
            "coverage_months": round(self.coverage_months, 2),
            "runway_months": round(self.runway_months, 1),
            "pd": round(self.pd, 4),
            "risk_score": round(self.risk_score, 4),
            "risk_band": self.risk_band.value,
            "action": self.action,
            "drivers": self.drivers,
            "model_auc": round(self.model_auc, 4),
            "rationale": self.rationale,
        }


class Underwriter:
    def __init__(self, dataset: Dataset, *, seed: int = 7, portfolio: int = 700):
        self.dataset = dataset
        self.seed = seed
        self.portfolio = portfolio
        self._train_portfolio_model()

    # ---- synthetic portfolio + model ----------------------------------------
    def _train_portfolio_model(self) -> None:
        rng = np.random.default_rng(self.seed + 101)
        n = self.portfolio
        cash = np.exp(rng.uniform(np.log(2e5), np.log(2e7), n))
        monthly_spend = np.exp(rng.uniform(np.log(3e4), np.log(3e6), n))
        rev_to_spend = rng.uniform(0.35, 1.5, n)
        rev_vol = rng.uniform(0.04, 0.85, n)
        spend_growth = rng.normal(0.03, 0.14, n)
        net = monthly_spend * (rev_to_spend - 1.0)            # negative => burning
        runway = np.clip(np.where(net < 0, cash / np.clip(-net, 1, None), 60.0), 0, 60)
        coverage = np.clip(cash / np.clip(monthly_spend, 1, None), 0, 36)
        headcount = np.clip((monthly_spend / 4000 + rng.normal(0, 8, n)).astype(int), 3, 4000)
        cash_log, spend_log = np.log10(cash), np.log10(monthly_spend)

        # Loss is driven by short runway, spend outpacing revenue, and volatility —
        # continuous signals so the model discriminates sharply (high AUC).
        z = (-0.6
             - 0.085 * runway
             - 1.3 * rev_to_spend
             + 2.4 * rev_vol
             - 0.5 * (cash_log - 6)
             + 1.2 * (runway < 3)
             - 0.05 * coverage)
        p = _sigmoid(z)
        y = (rng.random(n) < p).astype(int)

        X = np.column_stack([cash_log, runway, rev_to_spend, spend_log,
                             rev_vol, spend_growth, coverage, headcount])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=self.seed,
                                              stratify=y if 0 < y.sum() < n else None)
        self.model = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.07, random_state=self.seed)
        self.model.fit(Xtr, ytr)
        self.model_auc = float(roc_auc_score(yte, self.model.predict_proba(Xte)[:, 1]))
        self.default_rate = float(y.mean())

    # ---- focal-company features ---------------------------------------------
    def _company_features(self) -> dict:
        ds = self.dataset
        cash = sum(a.balance_cents for a in ds.cash_accounts) / 100.0
        months = max((ds.card_transactions[-1].ts - ds.card_transactions[0].ts).days / 30.0, 1e-6) \
            if ds.card_transactions else 1.0
        card_spend = sum(t.amount_cents for t in ds.card_transactions
                         if not t.ground_truth.is_fraud) / 100.0
        monthly_spend = card_spend / months
        inflow = sum(t.amount_cents for t in ds.cash_transactions if t.amount_cents > 0) / 100.0
        monthly_rev = inflow / months
        # revenue volatility: CV of weekly ACH credits
        rev = [t.amount_cents / 100.0 for t in ds.cash_transactions
               if t.type.value == "ach_credit"]
        rev_vol = float(np.std(rev) / np.mean(rev)) if rev and np.mean(rev) else 0.3
        rev_to_spend = monthly_rev / monthly_spend if monthly_spend else 1.0
        net = monthly_rev - monthly_spend
        runway = min(cash / -net, 60.0) if net < 0 else 60.0
        coverage = cash / monthly_spend if monthly_spend else 12.0
        headcount = len(ds.employees)
        return {
            "cash": cash, "monthly_spend": monthly_spend, "coverage": coverage,
            "rev_to_spend": rev_to_spend, "rev_vol": rev_vol, "spend_growth": 0.05,
            "runway": runway, "headcount": headcount,
        }

    def assess(self) -> CreditAssessment:
        f = self._company_features()
        x = np.array([[np.log10(max(f["cash"], 1)), f["runway"], f["rev_to_spend"],
                       np.log10(max(f["monthly_spend"], 1)), f["rev_vol"], f["spend_growth"],
                       min(f["coverage"], 36), f["headcount"]]])
        pd = float(self.model.predict_proba(x)[0, 1])
        band = RiskBand.from_score(pd)

        # Cash-coverage limit: strong credits can borrow close to their balance; risky
        # ones are capped well below it. (Stripe Capital offer ~ share of trailing processing volume + cash.)
        mult = float(np.clip(1.15 - 1.5 * pd, 0.25, 1.15))
        recommended = f["cash"] * mult
        current = self.dataset.company.monthly_card_limit_cents / 100.0
        spent = sum(t.amount_cents for t in self.dataset.card_transactions
                    if not t.ground_truth.is_fraud) / 100.0
        months = max((self.dataset.card_transactions[-1].ts
                      - self.dataset.card_transactions[0].ts).days / 30.0, 1e-6)
        utilization = (spent / months) / current if current else 0.0

        if recommended > current * 1.15 and pd < 0.4:
            action = "increase_limit"
        elif pd >= 0.6 or f["runway"] < 3:
            action = "reduce_within_24h"
        else:
            action = "hold"

        return CreditAssessment(
            company=self.dataset.company.name,
            cash_balance_usd=f["cash"], current_limit_usd=current,
            recommended_limit_usd=recommended, utilization=utilization,
            coverage_months=f["coverage"], runway_months=f["runway"],
            pd=pd, risk_score=pd, risk_band=band, action=action,
            drivers=self._drivers(f),
            model_auc=self.model_auc,
            rationale=(f"PD {pd:.1%} from a portfolio GBM (AUC {self.model_auc:.2f}); "
                       f"{f['coverage']:.1f}mo cash coverage, runway "
                       f"{f['runway']:.0f}mo. Cash-coverage limit ≈ {mult:.2f}× balance."))

    @staticmethod
    def _drivers(f: dict) -> list[dict]:
        out = []
        if f["coverage"] >= 6:
            out.append({"factor": "cash coverage", "value": f"{f['coverage']:.1f} months",
                        "effect": "lowers risk"})
        else:
            out.append({"factor": "thin cash coverage", "value": f"{f['coverage']:.1f} months",
                        "effect": "raises risk"})
        out.append({"factor": "revenue / spend", "value": f"{f['rev_to_spend']:.2f}",
                    "effect": "lowers risk" if f["rev_to_spend"] >= 0.9 else "raises risk"})
        out.append({"factor": "revenue volatility", "value": f"{f['rev_vol']:.0%}",
                    "effect": "raises risk" if f["rev_vol"] > 0.4 else "lowers risk"})
        out.append({"factor": "runway", "value": f"{f['runway']:.0f} months",
                    "effect": "lowers risk" if f["runway"] >= 9 else "raises risk"})
        return out
