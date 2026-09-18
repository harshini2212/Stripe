"""Causal cash-runway engine — genuinely Rung-2 (interventional), not a spreadsheet.

A spreadsheet (or a regression on history) is Rung-1: it treats correlation as the
effect of acting. This engine holds an explicit causal DAG over the drivers, so an
intervention ``do(headcount := +20)`` SEVERS the node from its parents and propagates
through the structural equations — ``do(X) != observing X``.

The discriminating proof lives here: **recruiting spend is confounded with revenue** by
an unobserved ``market_demand`` factor (both rise in strong quarters). A naive
correlational read says "raise recruiting -> revenue follows"; the interventional read
says forcing recruiting up does NOT move revenue, because the confounder didn't move.
The two estimates come out different — that gap is the product.

The structural-equation propagation and Monte-Carlo banding are exposed as tools the
live agent calls; the agent picks interventions from the user's selections and narrates.
The DAG is hand-specified for v1 with a seam to swap in a discovered one (DirectLiNGAM).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# The hand-specified causal DAG (parent -> child). Exposed so the agent can cite edges.
DAG: dict[str, list[str]] = {
    "market_demand": ["recruiting_spend", "new_customers", "revenue"],  # the confounder
    "hires": ["headcount"],
    "headcount": ["payroll", "tooling"],
    "marketing_spend": ["new_customers"],   # the genuine causal lever on revenue
    "new_customers": ["customers"],
    "churn": ["customers"],
    "customers": ["revenue"],
    "price": ["revenue"],
    "payroll": ["opex"], "tooling": ["opex"], "marketing_spend2": ["opex"],
    "recruiting_spend": ["opex"],           # cost only — NO edge to revenue
    "revenue": ["burn"], "opex": ["burn"], "burn": ["cash"],
}


@dataclass
class Scenario:
    """A do() intervention set. Each field overrides a structural input going forward."""

    hires_per_month: float | None = None      # do(hires := value)
    churn_monthly: float | None = None        # do(churn := value)
    price_change_pct: float | None = None     # do(price := *(1+x))
    marketing_change_pct: float | None = None
    recruiting_change_pct: float | None = None
    freeze_hiring: bool = False

    def label(self) -> str:
        bits = []
        if self.freeze_hiring:
            bits.append("freeze hiring")
        if self.hires_per_month is not None:
            bits.append(f"hire {self.hires_per_month:+.0f}/mo")
        if self.churn_monthly is not None:
            bits.append(f"churn={self.churn_monthly:.1%}")
        if self.price_change_pct:
            bits.append(f"price {self.price_change_pct:+.0%}")
        if self.marketing_change_pct:
            bits.append(f"marketing {self.marketing_change_pct:+.0%}")
        if self.recruiting_change_pct:
            bits.append(f"recruiting {self.recruiting_change_pct:+.0%}")
        return ", ".join(bits) or "baseline"


@dataclass
class Forecast:
    months: list[str]
    cash: list[float]
    burn: list[float]
    revenue: list[float]
    runway_months: float | None
    p10_cash: list[float] = field(default_factory=list)
    p90_cash: list[float] = field(default_factory=list)


class CausalRunway:
    def __init__(self, *, seed: int = 7, history_months: int = 24):
        self.rng = np.random.default_rng(seed + 303)
        self.H = history_months
        self._build_history()

    # ---- exogenous history (with the planted confounder) --------------------
    def _build_history(self) -> None:
        H = self.H
        rng = self.rng
        # market_demand: AR(1) latent factor — the confounder of recruiting & revenue.
        demand = np.zeros(H)
        for t in range(1, H):
            demand[t] = 0.75 * demand[t - 1] + rng.normal(0, 0.5)
        self.demand = demand
        # coefficients (structural; exposed). Calibrated to a Series-B SaaS: ~80 heads,
        # ~$1.2M/mo revenue, ~$1.6M/mo opex, ~$400k net burn.
        self.salary = 165_000.0
        self.tool_per_head = 1_100.0
        self.arpu = 1_400.0           # monthly revenue per customer
        self.cac_inv = 0.00015        # customers acquired per $ of marketing (CAC ~$6.7k)
        self.rec_base = 80_000.0
        self.mkt_base = 200_000.0
        self.fixed = 150_000.0
        self.churn0 = 0.030

        headcount = np.zeros(H)
        headcount[0] = 45
        customers = np.zeros(H)
        customers[0] = 850
        for t in range(1, H):
            headcount[t] = headcount[t - 1] + rng.integers(1, 3)
            mkt = self.mkt_base * (1 + rng.normal(0, 0.05))
            newc = self.cac_inv * mkt * (1 + 0.4 * demand[t])     # marketing(causal)+demand
            customers[t] = customers[t - 1] * (1 - self.churn0) + newc
        self.headcount0 = float(headcount[-1])
        self.customers0 = float(customers[-1])
        # recruiting is confounded with revenue via demand (no causal edge to revenue)
        self.recruiting_hist = self.rec_base * (1 + 0.7 * demand)
        self.revenue_hist = self.arpu * customers * (1 + 0.3 * demand)
        # starting cash sized to ~12-16 months of recent burn
        recent_burn = (self.headcount0 * self.salary / 12 + self.headcount0 * self.tool_per_head
                       + self.mkt_base + self.rec_base + self.fixed - self.revenue_hist[-1])
        self.cash0 = float(max(recent_burn, 1) * 14)

    # ---- forward simulation via the structural equations --------------------
    def _simulate(self, sc: Scenario, horizon: int, demand_path: np.ndarray) -> Forecast:
        head = self.headcount0
        cust = self.customers0
        cash = self.cash0
        churn = sc.churn_monthly if sc.churn_monthly is not None else self.churn0
        price_mult = 1 + (sc.price_change_pct or 0)
        mkt_mult = 1 + (sc.marketing_change_pct or 0)
        rec_mult = 1 + (sc.recruiting_change_pct or 0)
        hires = 0.0 if sc.freeze_hiring else (sc.hires_per_month
                                              if sc.hires_per_month is not None else 4.0)
        months, cashs, burns, revs = [], [], [], []
        runway = None
        for t in range(horizon):
            d = demand_path[t]
            head = head + hires
            payroll = head * self.salary / 12
            tooling = head * self.tool_per_head
            marketing = self.mkt_base * mkt_mult
            recruiting = self.rec_base * rec_mult * (1 + 0.7 * d)
            newc = self.cac_inv * marketing * (1 + 0.4 * d)
            cust = cust * (1 - churn) + newc
            revenue = self.arpu * cust * price_mult * (1 + 0.3 * d)
            opex = payroll + tooling + marketing + recruiting + self.fixed
            burn = opex - revenue
            cash = cash - burn
            months.append(f"M{t + 1}")
            cashs.append(round(cash, 2)); burns.append(round(burn, 2)); revs.append(round(revenue, 2))
            if runway is None and cash <= 0:
                runway = t + 1
        return Forecast(months, cashs, burns, revs,
                        float(runway) if runway else None)

    def _demand_path(self, horizon: int) -> np.ndarray:
        # continue the AR(1) demand forward (mean-reverting to 0)
        d = np.zeros(horizon)
        prev = self.demand[-1]
        for t in range(horizon):
            prev = 0.75 * prev
            d[t] = prev
        return d

    # ---- public: Rung 1 / Rung 2 / Monte Carlo ------------------------------
    def baseline(self, horizon: int = 18) -> Forecast:
        return self._simulate(Scenario(), horizon, self._demand_path(horizon))

    def intervene(self, sc: Scenario, horizon: int = 18, mc: int = 200) -> Forecast:
        base_d = self._demand_path(horizon)
        fc = self._simulate(sc, horizon, base_d)
        # Monte-Carlo band over demand shocks (skipped when mc<=0).
        if mc > 0:
            paths = [self._simulate(sc, horizon,
                     base_d + self.rng.normal(0, 0.25, horizon).cumsum() * 0.3).cash
                     for _ in range(mc)]
            arr = np.array(paths)
            fc.p10_cash = np.percentile(arr, 10, axis=0).round(2).tolist()
            fc.p90_cash = np.percentile(arr, 90, axis=0).round(2).tolist()
        return fc

    def confounded_effect(self, driver: str = "recruiting", delta_pct: float = 0.5) -> dict:
        """Naive correlational effect vs interventional effect of a driver on revenue."""
        # Naive: regress historical revenue on the (confounded) driver -> slope * delta.
        x = self.recruiting_hist if driver == "recruiting" else (
            self.mkt_base * (1 + np.zeros(self.H)))
        y = self.revenue_hist
        slope = float(np.polyfit(x, y, 1)[0])
        naive = slope * (x.mean() * delta_pct)   # predicted monthly revenue change
        # Interventional: do(driver *= 1+delta) and read the revenue change (recruiting
        # has no causal edge to revenue -> ~0; marketing does -> positive).
        base = self.baseline(1).revenue[0]
        sc = (Scenario(recruiting_change_pct=delta_pct) if driver == "recruiting"
              else Scenario(marketing_change_pct=delta_pct))
        inter = self.intervene(sc, 1, mc=0).revenue[0] - base
        return {
            "driver": driver, "delta_pct": delta_pct,
            "naive_correlational_revenue_effect_usd": round(naive, 2),
            "interventional_revenue_effect_usd": round(inter, 2),
            "explanation": (
                "Recruiting spend and revenue both rise with unobserved market demand, so "
                "the naive correlation shows a large 'effect'. Intervening on recruiting "
                "(do) holds demand fixed — revenue barely moves. The gap is the confounder."
                if driver == "recruiting" else
                "Marketing has a real causal path to revenue (new customers), so the "
                "interventional effect is genuine, not just correlation.")}
