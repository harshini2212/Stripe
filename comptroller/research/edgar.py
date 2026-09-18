"""SEC EDGAR company research — search, XBRL company facts, ratios, tie-outs.

The search/CIK-resolution logic is ported from the tieout project's EdgarClient.
Instead of loading full inline-XBRL through arelle, this uses SEC's lightweight
``companyfacts`` JSON API (every numeric XBRL fact a company has ever filed) and
projects it into an analyst-ready profile:

  * annual series (last ~6 fiscal years) for revenue, net income, operating
    income, assets, liabilities, equity, cash, operating cash flow
  * derived ratios (margins, YoY growth, D/E, ROE) — computed, not quoted
  * accounting tie-outs (Assets = Liabilities + Equity, margin consistency),
    the same financial-correctness discipline the spend engine applies

SEC fair-access requires a descriptive User-Agent; all requests carry one.
Everything is cached in-process so a demo hits EDGAR once per company.
"""
from __future__ import annotations

import json
import threading
import urllib.request
from functools import lru_cache
from typing import Any

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"

DEFAULT_USER_AGENT = "comptroller-research hv2201@nyu.edu"

# Brand/product -> ticker, for searches that won't match the legal entity name.
_ALIASES = {
    "google": "GOOGL", "alphabet": "GOOGL", "youtube": "GOOGL",
    "facebook": "META", "instagram": "META", "whatsapp": "META",
    "aws": "AMZN", "windows": "MSFT", "xbox": "MSFT", "chatgpt": "MSFT",
    "iphone": "AAPL", "stripe": "PYPL"  # Stripe is private; nearest listed payments comp,
}

# Concept -> ordered us-gaap tag candidates (filers differ in which they use).
_CONCEPTS: dict[str, list[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    # ---- spend-management lens: the operating cost lines Stripe-style controls touch ----
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "sga": ["SellingGeneralAndAdministrativeExpense", "GeneralAndAdministrativeExpense"],
    "rd": ["ResearchAndDevelopmentExpense"],
    "opex": ["OperatingExpenses", "CostsAndExpenses"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
}
_FLOW = {"revenue", "net_income", "operating_income", "gross_profit", "operating_cash_flow",
         "cost_of_revenue", "sga", "rd", "opex", "capex"}


class EdgarClient:
    """Thin, cached EDGAR HTTP client (ticker index + companyfacts)."""

    def __init__(self, user_agent: str = DEFAULT_USER_AGENT, timeout: int = 30) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._companies: list[dict] | None = None
        self._lock = threading.Lock()

    def _get(self, url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.read()

    def _load_companies(self) -> list[dict]:
        with self._lock:
            if self._companies is None:
                raw = json.loads(self._get(_TICKERS_URL))
                # dict ordered by descending market cap; keep that as `rank`.
                self._companies = [
                    {"ticker": r["ticker"].upper(), "name": r["title"],
                     "cik": f"{int(r['cik_str']):010d}", "rank": i}
                    for i, r in enumerate(raw.values())
                ]
        return self._companies

    def search(self, query: str, limit: int = 8) -> list[dict]:
        """Typeahead over EDGAR companies, ranked by match quality then size."""
        q = query.strip().lower()
        if not q:
            return []
        alias = _ALIASES.get(q)
        scored = []
        for c in self._load_companies():
            t, name = c["ticker"].lower(), c["name"].lower()
            if t == q:
                s = 1000
            elif t.startswith(q):
                s = 720
            elif name.startswith(q):
                s = 600
            elif (" " + q) in (" " + name):
                s = 380
            elif q in name:
                s = 220
            elif q in t:
                s = 150
            else:
                s = 0
            if alias and c["ticker"] == alias:
                s = max(s, 900)
            if s:
                scored.append((s, -c["rank"], c))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [{"ticker": c["ticker"], "name": c["name"], "cik": c["cik"]}
                for _, _, c in scored[:limit]]

    def resolve(self, ticker: str) -> dict:
        for c in self._load_companies():
            if c["ticker"] == ticker.upper():
                return c
        raise KeyError(f"ticker {ticker!r} not found in EDGAR index")

    def company_facts(self, cik10: str) -> dict:
        return json.loads(self._get(_FACTS_URL.format(cik10=cik10)))

    def submissions(self, cik10: str) -> dict:
        return json.loads(self._get(_SUBMISSIONS_URL.format(cik10=cik10)))


def _filing_links(cik10: str, client: EdgarClient) -> dict[str, Any]:
    """The EDGAR browse page plus, best-effort, the latest 10-K's primary document.

    Never raises — a submissions hiccup must not sink the whole profile; the browse
    URL always resolves from the CIK alone.
    """
    cik_int = int(cik10)
    browse = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
              f"&CIK={cik10}&type=10-K&dateb=&owner=include&count=40")
    links: dict[str, Any] = {"browse_url": browse, "latest_10k_url": None,
                             "latest_10k_date": None, "form": None}
    try:
        recent = client.submissions(cik10).get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        for i, form in enumerate(forms):
            if form in ("10-K", "10-K/A", "20-F"):
                acc = recent["accessionNumber"][i].replace("-", "")
                doc = recent["primaryDocument"][i]
                links["latest_10k_url"] = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc}/{doc}")
                links["latest_10k_date"] = recent.get("filingDate", [None] * (i + 1))[i]
                links["form"] = form
                break
    except Exception:
        pass
    return links


_CLIENT: EdgarClient | None = None


def _client() -> EdgarClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = EdgarClient()
    return _CLIENT


def _annual_series(gaap: dict, tags: list[str], flow: bool) -> list[dict]:
    """Merge candidate tags into one USD value per fiscal year.

    Keyed by the period-END year (a 10-K's comparative years share the filing's
    ``fy`` field, so that field can't identify the data year). Flow concepts
    (revenue…) take ~12-month 10-K durations; stock concepts (assets…) take the
    fiscal-year-end instant. Filers switch tags over time (e.g. pre/post ASC 606
    revenue), so later candidate tags fill years the primary tag lacks; within a
    tag, the latest-filed value wins (amendments supersede).
    """
    rows: dict[int, dict] = {}
    for rank, tag in enumerate(tags):
        node = gaap.get(tag)
        if not node:
            continue
        for u in node.get("units", {}).get("USD", []):
            end, val = u.get("end"), u.get("val")
            if val is None or not end or not u.get("form", "").startswith("10-K"):
                continue
            if flow:
                if u.get("fp") != "FY" or not u.get("start"):
                    continue
                # A full-year duration is ~12 months; skip stub/quarterly windows.
                try:
                    months = (int(end[:4]) - int(u["start"][:4])) * 12 + \
                             (int(end[5:7]) - int(u["start"][5:7]))
                except Exception:
                    continue
                if not 10 <= months <= 14:
                    continue
            key = int(end[:4])
            prev = rows.get(key)
            # Earlier tags in the candidate list outrank later ones for the same
            # year; within a tag, prefer the most recently filed figure.
            if prev is None or rank < prev["rank"] or \
                    (rank == prev["rank"] and u.get("filed", "") >= prev["filed"]):
                rows[key] = {"fy": key, "end": end, "val": float(val),
                             "filed": u.get("filed", ""), "rank": rank}
    series = [dict(fy=r["fy"], end=r["end"], value=r["val"])
              for r in sorted(rows.values(), key=lambda r: r["fy"])]
    return series[-6:]


def _latest(series: list[dict]) -> float | None:
    return series[-1]["value"] if series else None


def _yoy(series: list[dict]) -> float | None:
    if len(series) < 2 or not series[-2]["value"]:
        return None
    return (series[-1]["value"] - series[-2]["value"]) / abs(series[-2]["value"])


def _spend_analysis(series: dict, cash: float | None) -> dict[str, Any]:
    """Frame the filed operating-cost lines as a Stripe spend-management opportunity.

    Every figure is derived from as-filed aggregates. SEC filings don't break spend
    into cards / SaaS / T&E, so the addressable base and lever rates are explicitly
    labeled estimates — the point is to size the prize the way a spend platform would,
    off real reported cost lines rather than invented ones.
    """
    rev = _latest(series["revenue"])
    cor, sga = _latest(series["cost_of_revenue"]), _latest(series["sga"])
    rd, opex = _latest(series["rd"]), _latest(series["opex"])

    def pctrev(x):
        return round(x / rev, 4) if (rev and x is not None) else None

    cost_structure = [{"label": lbl, "usd": v, "pct_of_revenue": pctrev(v)}
                      for lbl, v in (("Cost of revenue", cor), ("SG&A", sga),
                                     ("R&D", rd), ("Operating expenses", opex))
                      if v is not None]

    # Addressable = non-payroll operating spend a card/AP/SaaS platform actually routes.
    # SG&A is the best public proxy; ~45% of it is typically vendor/software/T&E/card
    # spend rather than salaries. Fall back to a slice of opex when SG&A isn't broken out.
    if sga is not None:
        addressable = sga * 0.45
        basis = "~45% of SG&A (non-payroll vendor, software, travel & card spend)"
    elif opex is not None:
        addressable = opex * 0.30
        basis = "~30% of operating expenses (SG&A not separately filed)"
    else:
        addressable, basis = None, "insufficient expense detail filed"

    levers, total = [], 0.0
    if addressable:
        for lever, rate, note in (
            ("Eliminate duplicate & unused SaaS", 0.025,
             "overlapping tools and idle seats consolidated to one org plan"),
            ("Negotiate vendor rates & capture card rebates", 0.015,
             "rate cards renegotiated; spend moved onto rebate-earning cards"),
            ("Recover out-of-policy leakage", 0.010,
             "real-time controls block off-policy spend before it settles")):
            usd = addressable * rate
            total += usd
            levers.append({"lever": lever, "rate": rate,
                           "savings_usd": round(usd, 2), "note": note})
    if cash:
        usd = cash * 0.045
        total += usd
        levers.append({"lever": "Earn yield on idle operating cash", "rate": 0.045,
                       "savings_usd": round(usd, 2),
                       "note": "operating balance swept into a ~4.5% yield account"})

    return {
        "cost_structure": cost_structure,
        "addressable_spend_usd": round(addressable, 2) if addressable else None,
        "addressable_basis": basis,
        "levers": levers,
        "total_savings_usd": round(total, 2) if levers else None,
        "savings_pct_of_revenue": round(total / rev, 4) if (rev and levers) else None,
    }


def build_company_profile(ticker: str, client: EdgarClient | None = None) -> dict[str, Any]:
    """Everything the Research page needs for one company, computed from raw XBRL."""
    cl = client or _client()
    meta = cl.resolve(ticker)
    facts = cl.company_facts(meta["cik"])
    gaap = facts.get("facts", {}).get("us-gaap", {})

    series = {name: _annual_series(gaap, tags, name in _FLOW)
              for name, tags in _CONCEPTS.items()}

    rev, ni = series["revenue"], series["net_income"]
    assets, liab, eq = series["assets"], series["liabilities"], series["equity"]
    latest_fy = rev[-1]["fy"] if rev else (assets[-1]["fy"] if assets else None)

    r_latest = _latest(rev)
    ni_latest = _latest(ni)
    oi_latest = _latest(series["operating_income"])
    a_latest, l_latest, e_latest = _latest(assets), _latest(liab), _latest(eq)

    ratios = {
        "revenue_yoy": _yoy(rev),
        "net_margin": (ni_latest / r_latest) if (r_latest and ni_latest is not None) else None,
        "operating_margin": (oi_latest / r_latest) if (r_latest and oi_latest is not None) else None,
        "debt_to_equity": (l_latest / e_latest) if (l_latest is not None and e_latest) else None,
        "roe": (ni_latest / e_latest) if (ni_latest is not None and e_latest) else None,
        "cash_ratio": (_latest(series["cash"]) / l_latest)
                      if (_latest(series["cash"]) is not None and l_latest) else None,
    }
    ratios = {k: (round(v, 4) if v is not None else None) for k, v in ratios.items()}

    # ---- accounting tie-outs: recompute identities from the filed numbers ----
    tieouts = []
    if a_latest is not None and l_latest is not None and e_latest is not None:
        delta = a_latest - (l_latest + e_latest)
        tol = max(abs(a_latest) * 0.005, 2e6)  # rounding + noncontrolling noise
        tieouts.append({
            "check": "Assets = Liabilities + Equity",
            "lhs_usd": a_latest, "rhs_usd": l_latest + e_latest,
            "delta_usd": round(delta, 2), "passed": abs(delta) <= tol})
    if r_latest and ni_latest is not None and series["gross_profit"]:
        gp = _latest(series["gross_profit"])
        tieouts.append({
            "check": "Gross profit ≤ Revenue",
            "lhs_usd": gp, "rhs_usd": r_latest,
            "delta_usd": round(r_latest - gp, 2), "passed": gp <= r_latest * 1.001})
    if ni_latest is not None and oi_latest is not None and r_latest:
        tieouts.append({
            "check": "Margins internally consistent (|NI| ≤ |Revenue|)",
            "lhs_usd": ni_latest, "rhs_usd": r_latest,
            "delta_usd": round(abs(r_latest) - abs(ni_latest), 2),
            "passed": abs(ni_latest) <= abs(r_latest)})

    return {
        "ticker": meta["ticker"], "name": facts.get("entityName") or meta["name"],
        "cik": meta["cik"], "latest_fy": latest_fy,
        "series": series, "ratios": ratios, "tieouts": tieouts,
        "spend": _spend_analysis(series, _latest(series["cash"])),
        "filing": _filing_links(meta["cik"], cl),
        "source": "SEC EDGAR companyfacts (XBRL as filed)",
    }


@lru_cache(maxsize=32)
def cached_profile(ticker: str) -> dict[str, Any]:
    return build_company_profile(ticker)
