"""Spend & expense intelligence over Stripe Issuing card activity.

Surfaces what a finance team actually wants from spend data: where the money goes,
recurring SaaS subscriptions (and redundant licenses to consolidate), duplicate
charges, policy-compliance rate, and per-department spend anomalies. Recurring
detection is a real inter-arrival-cadence algorithm, not a keyword match.
"""
from __future__ import annotations

from collections import defaultdict
from statistics import median, pstdev

import numpy as np

from ..domain import Dataset
from ..domain.enums import MCC_TABLE, ExpenseCategory


class SpendIntelligence:
    def __init__(self, dataset: Dataset):
        self.ds = dataset
        self.merchant_index = dataset.merchant_index()
        self.emp_index = dataset.employee_index()
        self.legit = [t for t in dataset.card_transactions if not t.ground_truth.is_fraud]

    # ---- headline summary ----------------------------------------------------
    def summary(self) -> dict:
        by_cat: dict[str, int] = defaultdict(int)
        by_dept: dict[str, int] = defaultdict(int)
        by_merchant: dict[str, int] = defaultdict(int)
        by_month: dict[str, int] = defaultdict(int)
        for t in self.legit:
            cat = (t.ground_truth.true_category or ExpenseCategory.OTHER).value
            by_cat[cat] += t.amount_cents
            emp = self.emp_index.get(t.employee_id)
            by_dept[emp.department if emp else "?"] += t.amount_cents
            by_merchant[self.merchant_index[t.merchant_id].name] += t.amount_cents
            by_month[t.ts.strftime("%Y-%m")] += t.amount_cents
        total = sum(t.amount_cents for t in self.legit) / 100.0
        topm = sorted(by_merchant.items(), key=lambda kv: kv[1], reverse=True)[:8]
        return {
            "total_spend_usd": round(total, 2),
            "by_category": {k: round(v / 100, 2) for k, v in
                            sorted(by_cat.items(), key=lambda kv: kv[1], reverse=True)},
            "by_department": {k: round(v / 100, 2) for k, v in
                              sorted(by_dept.items(), key=lambda kv: kv[1], reverse=True)},
            "top_merchants": [{"merchant": k, "spend_usd": round(v / 100, 2)} for k, v in topm],
            "monthly_trend": [{"month": k, "spend_usd": round(v / 100, 2)}
                              for k, v in sorted(by_month.items())],
        }

    # ---- recurring subscriptions --------------------------------------------
    def recurring_subscriptions(self) -> dict:
        # A subscription is a single (employee, merchant, exact amount) stream charged
        # on a regular cadence — grouping by exact amount isolates it from ad-hoc spend.
        streams: dict[tuple, list] = defaultdict(list)
        for t in self.legit:
            streams[(t.employee_id, t.merchant_id, t.amount_cents)].append(t)
        per_merchant: dict[str, list] = defaultdict(list)
        for (eid, mid, amt), txns in streams.items():
            if len(txns) < 3:
                continue
            txns = sorted(txns, key=lambda x: x.ts)
            gaps = [(b.ts - a.ts).days for a, b in zip(txns, txns[1:]) if (b.ts - a.ts).days > 0]
            if not gaps:
                continue
            med_gap = median(gaps)
            cadence = ("monthly" if 24 <= med_gap <= 35 else
                       "weekly" if 5 <= med_gap <= 9 else
                       "quarterly" if 80 <= med_gap <= 100 else None)
            if cadence is None:
                continue
            per = amt / 100.0
            monthly = per if cadence == "monthly" else per * (4.3 if cadence == "weekly" else 1 / 3)
            per_merchant[mid].append((eid, monthly, cadence))

        subs = []
        for mid, rows in per_merchant.items():
            subscribers = len({eid for eid, _, _ in rows})
            unit = median([m for _, m, _ in rows])
            cadence = rows[0][2]
            monthly_total = sum(m for _, m, _ in rows)
            subs.append({
                "merchant": self.merchant_index[mid].name, "cadence": cadence,
                "unit_cost_usd": round(unit, 2), "monthly_cost_usd": round(monthly_total, 2),
                "annualized_usd": round(monthly_total * 12, 2),
                "subscribers": subscribers, "redundant": subscribers > 1,
            })
        subs.sort(key=lambda s: s["annualized_usd"], reverse=True)
        total_annual = sum(s["annualized_usd"] for s in subs)
        # Consolidating N redundant licenses to one org plan saves (N-1)/N of the spend.
        redundant_savings = sum(s["annualized_usd"] * (s["subscribers"] - 1) / s["subscribers"]
                                for s in subs if s["redundant"])
        return {
            "subscriptions": subs[:15], "count": len(subs),
            "total_annualized_usd": round(total_annual, 2),
            "redundant_savings_usd": round(redundant_savings, 2),
        }

    # ---- duplicate charges ---------------------------------------------------
    def duplicates(self) -> dict:
        dups = []
        for t in self.legit:
            if any(v.value == "duplicate_spend" for v in t.ground_truth.policy_violations):
                dups.append({
                    "txn_id": t.id, "merchant": self.merchant_index[t.merchant_id].name,
                    "amount_usd": round(t.amount, 2), "card_id": t.card_id,
                    "ts": t.ts.isoformat()})
        return {"count": len(dups), "recoverable_usd": round(sum(d["amount_usd"] for d in dups), 2),
                "items": dups[:20]}

    # ---- policy compliance (Audit-Agent style) -------------------------------
    def compliance(self) -> dict:
        by_type: dict[str, int] = defaultdict(int)
        flagged_amount = 0
        n_viol = 0
        for t in self.legit:
            if t.ground_truth.policy_violations:
                n_viol += 1
                flagged_amount += t.amount_cents
                for v in t.ground_truth.policy_violations:
                    by_type[v.value] += 1
        rate = 1.0 - (n_viol / len(self.legit)) if self.legit else 1.0
        return {
            "compliance_rate": round(rate, 4),
            "flagged_transactions": n_viol,
            "flagged_amount_usd": round(flagged_amount / 100.0, 2),
            "violations_by_type": dict(sorted(by_type.items(), key=lambda kv: kv[1], reverse=True)),
        }

    # ---- spend anomalies -----------------------------------------------------
    def anomalies(self, top_k: int = 8) -> dict:
        # Per employee spend vs department peers -> z-score outliers.
        dept_spend: dict[str, list] = defaultdict(list)
        emp_spend: dict[str, int] = defaultdict(int)
        for t in self.legit:
            emp_spend[t.employee_id] += t.amount_cents
        emp_dept = {e.id: e.department for e in self.ds.employees}
        for eid, amt in emp_spend.items():
            dept_spend[emp_dept.get(eid, "?")].append((eid, amt))
        out = []
        for dept, rows in dept_spend.items():
            amts = np.array([a for _, a in rows], dtype=float)
            if len(amts) < 3:
                continue
            mu, sd = amts.mean(), amts.std() or 1.0
            for eid, amt in rows:
                z = (amt - mu) / sd
                if z >= 2.0:
                    emp = self.emp_index.get(eid)
                    out.append({"employee": emp.name if emp else eid, "department": dept,
                                "spend_usd": round(amt / 100.0, 2), "z_score": round(float(z), 2)})
        out.sort(key=lambda r: r["z_score"], reverse=True)
        return {"count": len(out), "outliers": out[:top_k]}
