# Architecture

Comptroller is a layered backend. Each layer has one job and a clean seam to the next,
which is what lets the same agents run on a deterministic engine or live Claude, and
lets the eval harness grade any of them against the same ground truth.

```
 domain ──▶ data ──▶ fraud ──▶ agents ──▶ eval
   │          │         │         │         │
 Stripe      seeded    graph +    backend-   leaderboard
 model     tenant    ML +      agnostic    vs ground
 + rules   + GT      causal    workflows   truth
                                  │
                                 llm (analytical | simulated | Claude)
```

## 1 · Domain (`comptroller/domain`)

Pydantic models for the Stripe spend graph — `Company`, `Employee`, `StripeCard`
(physical/virtual, vendor-lockable), `StripeTreasuryAccount` (operating/yield/reserve),
`Merchant`, `CardTransaction`, `CashTransaction`, `Dispute` (real Visa/MC reason
codes), and `SpendPolicy`. Every transaction carries a `GroundTruth` block
(`is_fraud`, `fraud_ring_id`, `true_category`, `policy_violations`,
`dispute_should_win`) that is **never shown to an agent** — it exists only for the eval
harness.

The **canonical rule engine** (`domain/policy.py::evaluate_policy`) is the single source
of truth for policy violations. The generator labels data with it *and* the policy
agent's deterministic engine uses it — so the eval question is sharp and honest: *can a
model reproduce the controller's rulebook?*

## 2 · Data (`comptroller/data`)

A seeded generator (`numpy.default_rng`) materializes one Stripe tenant: ~40 employees,
~70 merchants, ~5,700 card transactions over 90 days, cash money-movement (nightly card
sweeps, ACH revenue, bill pay, **Stripe Treasury yield accrual**), and disputes.

The realism is the point — it's engineered so **no single feature separates fraud**:

- **Legit business travel** produces geo jumps on *trusted* devices (Sales/Exec).
- **Shared office IPs** make IP fan-out a noisy signal, not an oracle.
- **Fraud has a gradient**: blatant foreign gift-card bursts → stealth domestic mule
  rings → **account-takeover that rides the cardholder's own device** (the irreducible
  tail that honestly caps recall).
- Merchant risk **overlaps** both classes.

Same seed → byte-identical tenant → reproducible leaderboard.

## 3 · Fraud (`comptroller/fraud`)

- **`graph.py`** — heterogeneous card/device/IP/merchant graph (networkx). Cards are
  linked by shared devices (strong) and *cross-metro* shared IPs (ring victims are
  random across the org; office IPs are single-metro and excluded — a
  **count-independent** heuristic). Connected components ≥ 2 cards are candidate rings,
  scored by shared-device presence, merchant risk, foreign and odd-hour share. Ring
  exposure is scoped to the *suspicious* shared-infrastructure transactions.
- **`features.py`** — leakage-free per-transaction features: amount z-score vs the
  card's own history, geo-velocity (haversine ÷ elapsed → implied km/h, → impossible
  travel), device novelty, velocity windows, plus the graph features. Labels live in a
  separate `y` column the model never reads.
- **`model.py`** — `StandardScaler` → **Isolation Forest** (unsupervised anomaly) blended
  with a **gradient-boosted classifier** (supervised, inverse-frequency class weights).
  F1-optimal threshold picked on training; ROC/PR-AUC reported on holdout.
- **`causal.py`** — **counterfactual attribution** in the spirit of Pearl's do-operator:
  for each feature, set it to its legitimate-population baseline and measure the drop in
  fraud probability. Large drop ⇒ that feature is causally driving the risk. Paired with
  plain-English templates and a small structural DAG, so the output reads like an
  analyst's note, not a coefficient vector. (No heavy SHAP dependency.)
- **`pipeline.py`** — trains a holdout model for honest metrics, refits on all data for
  scoring, and turns every score into an **explained, actionable** `FraudAssessment`.

## 4 · LLM backends (`comptroller/llm`)

The seam that makes everything model-pluggable. An agent implements `AgentProtocol`
(`build_messages`, `solve`, `perturb`, `coerce`); a `Backend` decides *who solves it*:

- **`AnalyticalBackend`** — the deterministic engine (`agent.solve`). Offline baseline
  and grading reference.
- **`SimulatedBackend(skill)`** — emulates model-quality variance offline by perturbing
  the engine's answer at rate `1 − skill` (seeded by the rendered prompt, so it's
  reproducible). Gives a believable multi-row leaderboard with no key. Clearly labelled.
- **`ClaudeBackend(model)`** — live Anthropic SDK with **structured outputs**
  (`output_config.format`, JSON-schema-validated), **adaptive thinking**, and **effort**
  (Opus 4.8 / Sonnet 4.6; Haiku 4.5 runs lean). Tracks tokens + cost.

`build_backends()` returns live Claude when a key is present, simulated stand-ins
otherwise — so the demo is identical offline and online except for which rows are real.

## 5 · Agents (`comptroller/agents`)

Five single-shot evaluable agents (categorization, policy audit, dispute, fraud triage,
tieout) plus two multi-step workflows. Shared `inputs.py` builders turn a transaction
into agent payloads (reading only observable facts) so the orchestrator and the eval
harness never drift. The **orchestrator** sequences the specialists and escalates to the
**investigator** on high risk; both emit full traces.

## 6 · Eval (`comptroller/eval`)

- **`tasks.py`** — golden-case builders pull held-out ground truth into `(inputs,
  expected)` pairs, and define per-task scoring (exact / set-F1 / numeric tolerance).
- **`harness.py`** — builds cases **once per task** (identical inputs for every backend),
  runs each agent through each backend, scores against ground truth, and aggregates into
  per-(task, backend) scores with **bootstrap 95% confidence intervals**, plus a
  cross-task leaderboard with cost and latency.

## Design decisions worth calling out

- **Offline-first.** The entire system — ML training, agents, eval — runs with no
  network. A key only swaps simulated rows for live Claude. CI-friendly, demo-friendly,
  and it proves the architecture rather than a single API call.
- **Determinism everywhere.** Seeded generation, seeded model training, seeded bootstrap.
  The same seed reproduces the same leaderboard, which is the precondition for trusting
  an eval.
- **Ground truth ≠ the engine being graded… except on purpose.** For policy and tieout
  the deterministic engine *is* the rule engine (it should be perfect; the question is
  whether LLMs match it). For fraud and categorization the engine is a strong-but-
  fallible method, so models can genuinely win or lose.
- **Explainability is first-class**, not an afterthought — causal drivers on fraud,
  reasoning traces on agents, auditable tool calls on the investigator.
