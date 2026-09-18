"""Treasury & cash-flow forecasting for the Stripe Treasury financial account.

Reconstructs a daily cash-balance series from Stripe Treasury money-movement, then forecasts
forward with a Ridge model over calendar + lag features (weekly revenue seasonality is
the dominant signal). Reports a backtested accuracy (MAPE), a runway/burn estimate, a
liquidity-shortfall date, and an idle-cash yield-optimization recommendation
(overnight sweep into the Stripe Treasury yield-bearing financial account, ~4% APY).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ..domain import Dataset

_MMF_APY = 0.041  # Stripe Treasury yield-bearing financial account (partner-bank) APY ballpark


@dataclass
class CashForecast:
    history: list[dict]              # [{date, balance_usd}]
    forecast: list[dict]            # [{date, p50_usd, p10_usd, p90_usd}]
    current_balance_usd: float
    monthly_inflow_usd: float
    monthly_outflow_usd: float
    monthly_net_usd: float
    runway_months: float | None     # None when cash-flow positive
    shortfall_date: str | None
    status: str                     # healthy | watch | critical
    backtest_mape: float
    yield_opportunity: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "current_balance_usd": round(self.current_balance_usd, 2),
            "monthly_inflow_usd": round(self.monthly_inflow_usd, 2),
            "monthly_outflow_usd": round(self.monthly_outflow_usd, 2),
            "monthly_net_usd": round(self.monthly_net_usd, 2),
            "runway_months": (None if self.runway_months is None
                              else round(self.runway_months, 1)),
            "shortfall_date": self.shortfall_date,
            "status": self.status,
            "backtest_mape": round(self.backtest_mape, 4),
            "history": self.history,
            "forecast": self.forecast,
            "yield_opportunity": self.yield_opportunity,
        }


class TreasuryForecaster:
    def __init__(self, dataset: Dataset, horizon_days: int = 30):
        self.dataset = dataset
        self.horizon = horizon_days
        self.opening = sum(a.balance_cents for a in dataset.cash_accounts) / 100.0
        self._build_series()

    def _build_series(self) -> None:
        days = max((t.ts.date() for t in self.dataset.cash_transactions),
                   default=date(2026, 1, 1))
        start = min((t.ts.date() for t in self.dataset.cash_transactions), default=days)
        n = (days - start).days + 1
        flows = np.zeros(n)
        for t in self.dataset.cash_transactions:
            i = (t.ts.date() - start).days
            if 0 <= i < n:
                flows[i] += t.amount_cents / 100.0
        self.start = start
        self.flows = flows
        # Anchor the series so *today's* point equals the real current balance and the
        # history is that balance minus the net flows since — rather than drifting away
        # from the opening figure.
        cum = np.cumsum(flows)
        self.balance = self.opening + cum - (cum[-1] if len(cum) else 0.0)

    # ---- feature engineering -------------------------------------------------
    @staticmethod
    def _features(flows: np.ndarray, idx: int, lag_window: np.ndarray) -> list[float]:
        dow = [0.0] * 7
        dow[idx % 7] = 1.0
        trend = idx
        lag7 = lag_window[idx - 7] if idx >= 7 else lag_window[:idx].mean() if idx else 0.0
        roll7 = lag_window[max(0, idx - 7):idx].mean() if idx else 0.0
        return [*dow, trend, lag7, roll7]

    def _fit(self, flows: np.ndarray) -> tuple[Ridge, StandardScaler, float]:
        X, y = [], []
        for i in range(7, len(flows)):
            X.append(self._features(flows, i, flows))
            y.append(flows[i])
        scaler = StandardScaler().fit(X)
        # Heavier regularization keeps the forward median a gently-undulating central
        # projection instead of a mechanical day-of-week sawtooth.
        model = Ridge(alpha=18.0).fit(scaler.transform(X), y)
        resid = np.array(y) - model.predict(scaler.transform(X))
        return model, scaler, float(resid.std())

    def _roll_forward(self, flows: np.ndarray, model, scaler, h: int) -> np.ndarray:
        series = list(flows)
        preds = []
        for _ in range(h):
            i = len(series)
            feat = self._features(np.array(series), i, np.array(series))
            p = float(model.predict(scaler.transform([feat]))[0])
            preds.append(p)
            series.append(p)
        return np.array(preds)

    # ---- public --------------------------------------------------------------
    def forecast(self) -> CashForecast:
        model, scaler, resid_std = self._fit(self.flows)
        fut = self._roll_forward(self.flows, model, scaler, self.horizon)

        cur = float(self.balance[-1])
        fut_balance = cur + np.cumsum(fut)
        # Confidence band widens with the square root of the horizon.
        band = resid_std * np.sqrt(np.arange(1, self.horizon + 1))

        last_date = self.start + timedelta(days=len(self.flows) - 1)
        history = [{"date": (self.start + timedelta(days=i)).isoformat(),
                    "balance_usd": round(float(self.balance[i]), 2)}
                   for i in range(len(self.balance))]
        forecast = []
        shortfall_date = None
        for k in range(self.horizon):
            d = last_date + timedelta(days=k + 1)
            p50 = float(fut_balance[k])
            p10 = p50 - 1.28 * band[k]
            forecast.append({"date": d.isoformat(), "p50_usd": round(p50, 2),
                             "p10_usd": round(p10, 2), "p90_usd": round(p50 + 1.28 * band[k], 2)})
            if shortfall_date is None and p10 <= 0:
                shortfall_date = d.isoformat()

        inflow = float(self.flows[self.flows > 0].sum())
        outflow = float(-self.flows[self.flows < 0].sum())
        months = max(len(self.flows) / 30.0, 1e-6)
        m_in, m_out = inflow / months, outflow / months
        m_net = m_in - m_out
        runway = None if m_net >= 0 else cur / (-m_net)
        status = ("healthy" if (runway is None or runway >= 9)
                  else "watch" if runway >= 4 else "critical")

        return CashForecast(
            history=history, forecast=forecast, current_balance_usd=cur,
            monthly_inflow_usd=m_in, monthly_outflow_usd=m_out, monthly_net_usd=m_net,
            runway_months=runway, shortfall_date=shortfall_date, status=status,
            backtest_mape=self._backtest(), yield_opportunity=self._yield(cur, m_out))

    def _backtest(self, holdout: int = 14) -> float:
        if len(self.flows) <= holdout + 10:
            return float("nan")
        train = self.flows[:-holdout]
        model, scaler, _ = self._fit(train)
        pred_flows = self._roll_forward(train, model, scaler, holdout)
        base = self.opening + np.cumsum(train)[-1]
        pred_bal = base + np.cumsum(pred_flows)
        actual_bal = self.balance[-holdout:]
        denom = np.where(np.abs(actual_bal) < 1, 1, np.abs(actual_bal))
        return float(np.mean(np.abs(pred_bal - actual_bal) / denom))

    def _yield(self, current_balance: float, monthly_outflow: float) -> dict:
        # Stripe Treasury sweeps idle operating cash into a yield-bearing financial account *overnight* and
        # back for liquidity — so checking cash earns ~4% without being locked up. The
        # opportunity is the operating (0%-APY) balance currently not earning yield.
        operating = current_balance * 0.45
        return {
            "operating_cash_usd": round(operating, 2),
            "deployable_to_yield_usd": round(operating, 2),
            "mmf_apy": _MMF_APY,
            "incremental_annual_yield_usd": round(operating * _MMF_APY, 2),
            "recommendation": (
                f"Auto-sweep your ${operating:,.0f} operating balance into Stripe Treasury "
                f"(partner-bank yield account, ~{_MMF_APY:.1%} APY). Overnight sweep keeps it liquid "
                f"while earning ~${operating * _MMF_APY:,.0f}/yr that 0%-APY checking forgoes."),
        }
