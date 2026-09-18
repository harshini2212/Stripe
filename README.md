# Comptroller

**An AI-native financial-operations platform + evaluation layer — built for Stripe.**

Comptroller is a backend that does the AI-finance work Stripe is racing to own, and adds
the one thing that makes shipping it safe: a way to **measure** whether the AI is
correct. It spans fraud, spend, treasury, accounts-payable, and credit — each backed by
a real ML model — and grades every agent decision against ground truth on a
**multi-model leaderboard**.

It runs **fully offline** on deterministic ML; set `ANTHROPIC_API_KEY` and the eval
leaderboard lights up with live **Claude Opus 4.8 / Sonnet 4.6 / Haiku 4.5**.

```bash
python -m venv .venv && .venv/Scripts/pip install -e .   # (Linux/Mac: .venv/bin/pip)
comptroller serve          # → http://127.0.0.1:8000/dashboard
comptroller demo           # or: full narrated CLI showcase
```

---

## Why it's built for Stripe

Stripe is the programmable financial-infrastructure layer: **Stripe Issuing** (physical
and virtual corporate cards with real-time authorization controls), **Stripe Treasury**
(banking-as-a-service financial accounts that hold and earn on operating cash),
**Stripe Capital** (revenue-based financing underwritten from live processing volume, no
personal guarantee, no credit check), **Stripe Radar** (ML fraud scoring on every
charge), and the **Agent Toolkit** that lets LLM agents call Stripe APIs and move real
money. The gaps Comptroller fills are exactly the ones that appear once agents can spend:
cash-flow forecasting, dynamic underwriting ML, AP duplicate detection, SaaS-spend
optimization, vendor-concentration risk, and **agent-output evaluation with a promotion
gate before models ship**.

Comptroller builds those, and ties them together with the correctness layer:

> **It extends the Agent Toolkit with measurable correctness — every agent decision is
> graded against ground truth before it goes live.**

---

## The platform — 9 surfaces

| Tab | What it does | ML / method |
|---|---|---|
| **Overview** | Exec roll-up + a single "value identified" tally | aggregates everything |
| **Fraud & Risk** | Alerts, fraud rings, causal explanations | Isolation Forest + GBM + graph |
| **Spend & Expense** | Categories, subscriptions, duplicates, compliance, anomalies | cadence detection + rules |
| **Treasury · Cash** | Cash-flow forecast, runway, idle-cash yield sweep | Ridge forecaster (backtested) |
| **Bill Pay · AP** | Duplicate invoices, vendor concentration, payment timing | dedup + HHI + float math |
| **Credit** | Dynamic limit + probability-of-loss | Gradient-boosted PD model |
| **Agents** | Orchestrator + autonomous fraud investigation | agent toolkit + tools |
| **Benchmark** | Financial-correctness leaderboard (the promotion gate) | eval harness + bootstrap CI |
| **Models** | A card per model with its live metric | the architecture story |

---

## The ML portfolio (live, honest metrics)

| Model | Type | Task | Metric |
|---|---|---|---|
| Fraud Ensemble | Isolation Forest + Gradient Boosting | card fraud | **ROC-AUC ≈ 0.95** (0.89 ± 0.08 5-fold CV) |
| Credit-Risk PD | Gradient Boosting | underwriting loss | **ROC-AUC ≈ 0.94** |
| Treasury Forecaster | Ridge (calendar + lag) | cash-flow / runway | **backtest MAPE ≈ 10%** |
| Fraud-Ring Graph | networkx components | collusion rings | shared-device clusters |
| Causal Explainer | counterfactual do-operator | fraud explanation | per-alert drivers |
| Recurring detector | inter-arrival cadence | SaaS subscriptions | redundant-license savings |

The data is engineered to be *hard* — legit international travel on trusted devices,
shared office IPs, account-takeover fraud, merchant-risk overlap — so nothing is a
trivial oracle and the models have to learn interactions. Metrics are credible, not
synthetic-perfect.

---

## Headline capabilities

**Fraud & graph intelligence** — a blended anomaly + supervised ensemble over
behavioral-biometric, velocity and **graph** features. Links Stripe Issuing cards by shared
devices and cross-metro IPs to surface fraud rings (excluding office VPNs). Every alert
gets **counterfactual causal drivers** ("if geo-velocity were normal, risk drops 0.42")
and an action: freeze the card, open a dispute, monitor, clear.

**Treasury & cash-flow forecasting** — reconstructs the daily Stripe Treasury balance and
forecasts it forward with a backtested confidence band; computes runway, a liquidity-
shortfall date, and an **idle-cash yield sweep** into the Stripe Treasury yield account (~4% APY).

**Dynamic credit underwriting** — Stripe Capital's revenue-based model made probabilistic: a
gradient-boosted probability-of-loss model (AUC ≈ 0.94) on a synthetic lending
portfolio, a cash-coverage limit recommendation, and a dynamic action (raise / hold /
reduce-within-24h).

**Spend & AP intelligence** — recurring-subscription detection with redundant-license
consolidation savings, duplicate-charge recovery, policy-compliance scoring, plus AP
duplicate-invoice detection, vendor-concentration (HHI) risk, and payment-timing that
holds cash in the Stripe Treasury yield account until due while capturing 2/10-net-30 discounts.

**Agentic workflows** — narrow single-task agents (categorize, audit policy, triage
fraud, adjudicate disputes, reconcile expense reports) plus a **ComptrollerOrchestrator**
that sequences them and escalates to a full **FraudInvestigator** workflow, all running
on any backend (deterministic, simulated, or live Claude).

**Financial-correctness benchmark** — the promotion gate. Five tasks (GL coding, policy
set-F1, dispute adjudication, fraud triage, tieout-to-the-cent) graded against held-out
ground truth with bootstrap confidence intervals, cost and latency. The deterministic
engine is the reference; models are ranked on how faithfully they reproduce a
controller's judgment.

```
Financial-correctness leaderboard (offline run)
 backend                 kind          overall   cost     ms
 offline-heuristic       deterministic   1.000   $0.000     0   ← reference
 claude-opus-4-8 (sim)   simulated       0.974   ...
 claude-sonnet-4-6 (sim) simulated       0.928   ...
 claude-haiku-4-5 (sim)  simulated       0.862   ...
```

---

## Run it

```bash
comptroller serve                 # multi-tab dashboard + API (http://127.0.0.1:8000)
comptroller demo                  # narrated end-to-end CLI showcase
comptroller fraud                 # fraud metrics, alerts, rings
comptroller orchestrate --top     # full agent decision on the top alert
comptroller eval --task-detail    # the benchmark leaderboard
```

API (all surfaces are one call away):

```
GET  /overview                    # exec roll-up + value identified
GET  /fraud/alerts  /fraud/rings  /fraud/assess/{txn}
GET  /treasury/forecast  /credit/underwrite
GET  /spend/intelligence  /ap/intelligence  /models
POST /agent/orchestrate  /eval/run
GET  /agent/investigate/{txn}     ·   docs at /docs
```

---

## Going live with Claude

Set `ANTHROPIC_API_KEY` and the simulated leaderboard rows become real graded models —
called through the Anthropic SDK with **structured outputs**, **adaptive thinking**, and
**effort** (Opus 4.8 / Sonnet 4.6 think adaptively; Haiku 4.5 runs lean). Cost and
latency are tracked per call, so the board answers *which model is most correct, at what
price* — the data you need to migrate traffic on Stripe's LLM gateway.

---

## Repo layout

```
comptroller/
  domain/      Stripe primitives (Card, Cash, disputes, policy) + the rule engine
  data/        deterministic synthetic-tenant generator (fraud, subscriptions, GT)
  fraud/       entity graph · features · ML ensemble · causal explainer · pipeline
  analytics/   forecasting · underwriting · spend · ap · model registry
  agents/      5 evaluable agents + orchestrator + investigator + tools
  llm/         backend abstraction: analytical · simulated · live Claude
  eval/        scorers · tasks · harness · leaderboard
  api/         FastAPI service + the multi-tab dashboard
  reporting.py · cli.py
scripts/       demo · train_fraud (CV) · run_eval
tests/         offline test suite
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the design deep-dive. Everything is
deterministic and reproducible from a seed; the ML trains, the agents run, the eval
grades — nothing here is a mock.

Built by Harsh Vardhan as a Stripe-specific demonstration of production-quality AI-finance
+ evaluation engineering.
