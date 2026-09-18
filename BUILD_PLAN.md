# Comptroller — Build Bible

**A persona-scoped, fully-agentic spend-intelligence platform for Stripe.**
Work from this across Claude Code sessions. Paste the prompts in order, one feature at a
time. Prompt 0 first (context), then 1 → 2 → 3 (the v1), then 4 (the differentiator),
then the rest.

---

## 0. Global directives — these apply to EVERY prompt below

Tell Claude Code to honor all of these on every feature. They override anything in the
older code that contradicts them.

1. **Live-only. Require the API key.** Every feature runs **real Claude** (vision +
   agentic tool-use). There is **no offline-simulation fallback** for any new feature.
   If `ANTHROPIC_API_KEY` is missing, the UI shows a clear "add your key" state and the
   Run button is disabled — it does not silently fake a result.
   - *Nuance, state it explicitly so Claude Code doesn't delete the wrong thing:* the
     **deterministic functions** (`evaluate_policy`, the tieout sum, the fraud model,
     the forecaster, the underwriter) are **not** an "offline mode" — they are **tools
     the agent calls** and the **ground-truth verifier** for tieout. Keep them. The
     *reasoning/explanation/orchestration* layer is always live Claude; the *math* stays
     deterministic and is exposed to the agent as tools.

2. **Use the Claude API as much as possible.** Prefer an **agentic tool-use loop**
   (the model decides which tools to call) over hard-coded control flow for anything
   involving judgment, explanation, compilation of natural language, or
   multi-step reasoning. Maximize tool calls and reasoning depth (effort `high`).

3. **No static pages.** Every tab is a **configure → run → inspect** interaction.
   Nothing is a precomputed dashboard. Content appears only as the *result of a user
   action*, with a live loading state and a streamed/returned agent trace.

4. **Multi-step interactive selection — prompt the user to choose more than once.**
   Before any run, the user moves through a short **configurator**:
   - Step 1 — **Persona** (Employee / Finance / Executive / Investor).
   - Step 2 — **Scope & controls** (date range, department/employee/vendor multiselect,
     model dropdown, effort, thresholds, sliders — feature-specific).
   - Step 3 — **Confirm** and Run.
   The agent may also **pause mid-run to ask a clarifying question** when the query is
   ambiguous (return a `needs_input` state with 2–4 options; the UI renders them as
   buttons and re-runs with the choice). Selecting is part of the product, not a chore.

5. **Always return a citable trace.** Every agent response includes: the **tool calls**
   it made (name + inputs + outputs), every **figure with its source**, a tieout/verify
   block where money is stated, and **cost + latency + tokens**. The UI renders the
   trace.

6. **Verify after every prompt** with the in-process FastAPI `TestClient`, with the key
   present (these are live tests now). Confirm permission scoping where relevant.

---

## 1. The organizing idea

Stop thinking "tabs." Think **one ledger, many role-scoped agent lenses.** The same
underlying data (cards, transactions, invoices, SEC pulls) is viewed through a different
**agent** — different system prompt, different *allowed* tools, different default
queries — depending on **who you are**.

- An **employee** asks *"why was my Lyft flagged?"*
- a **controller** asks *"where is policy leaking money?"*
- an **investor** asks *"what's the burn, and is this underwritable?"*

Same backend, three different products. The real engineering signal (not just UI) is
**permission-scoped toolsets**: in Investor mode the agent **literally does not have the
tool** to open an individual employee's receipt. That's RBAC enforced at the agent
layer, and it's the thing to demo: *"Stripe flags; we run a permissioned reasoning agent
and explain every decision with a citable trace."*

---

## 2. The persona model (the spine)

| Persona | Sees | Agent CAN | Agent CANNOT |
|---|---|---|---|
| **Employee** | only their own card/txns | explain policy, check own status, draft a fix | read others, aggregate, any write |
| **Finance / Admin** | everything | full toolset incl. writes (flag, issue card, compile policy, pay invoice, close) | — |
| **Executive (CFO)** | company-wide | all read tools + forecasting | most writes |
| **Investor / Board** | aggregates + financial health | underwriting, runway, diligence Q&A | employee-level PII, any write |

(Manager = scoped Finance; Auditor = read-only Finance — add later.)

---

## 3. The tab menu, ranked by leverage

**Tier 1 — build first (highest demo payoff, most on-brand):**
1. **Persona system + shared agent-runner** — the spine. Everything plugs into it.
2. **Policy Studio** — plain-English policy → agent compiles to rules → **replays** over
   90 days → shows what it would have caught + the **$ impact**. Best single demo.
3. **Investor / Board Room** — read-only diligence agent; every figure is
   **tieout-verified** (no hallucinated financials — the eval-harness ethos as a feature).

**Tier 2 — strong adds:**
4. **Runway & Burn Forecasting (causal / Rung-2)** — interventional sliders (headcount,
   churn, price) → *"hire 20 → runway drops to N."* The differentiator.
5. **AP / Bill Pay agent** — invoice → PO/contract match → 3-way reconciliation →
   approve/hold with reasoning. Biggest surface Stripe actually monetizes.
6. **Employee Self-Service agent** — *"what's my limit, why was this flagged, how do I
   fix it"* — completes the persona triangle so the role-switch demo lands.

**Tier 3 — rounding out:**
7. **Month-End Close agent** — recurring forensics → draft journal adjustments → close memo.
8. **Card Issuance agent** — *"make a card for the contractor, $2K/mo, software only."*
9. **Treasury console** — idle cash → laddered allocation suggestion.

**Recommended order:** ship **1 → 2 → 3** as a coherent v1 (the role switch + the two
flagship tabs), then **4** as the thing that separates you from everyone.

---

## 4. The prompts

> Reference files in this repo: `comptroller/ai/claude_client.py` (the `ClaudeClient`
> with `run_agent` tool-use loop, `extract_document` vision, `complete_json`; the `Tool`
> dataclass), `comptroller/agents/` (orchestrator, investigator, `tools.py` Toolbox),
> `comptroller/workflows/receipt_autopilot.py`, `comptroller/analytics/`
> (forecasting, underwriting, spend, ap), `comptroller/eval/` (the benchmark),
> `comptroller/data/synthetic.py` + `comptroller/domain/policy.py` (the tenant + policy
> engine), `comptroller/api/app.py` (+ `dashboard.html`, `receipt.html`),
> `comptroller/config.py` (`.env` / `has_live_models`).

### Prompt 0 — context primer

```text
Read these files before doing anything and summarize back how the agent loop, the tool
dispatch, and the FastAPI endpoints currently fit together:
  comptroller/ai/claude_client.py, comptroller/agents/tools.py,
  comptroller/agents/orchestrator.py, comptroller/agents/investigator.py,
  comptroller/workflows/receipt_autopilot.py, comptroller/analytics/underwriting.py,
  comptroller/api/app.py, comptroller/data/synthetic.py, comptroller/domain/policy.py.

Direction I'm moving toward, keep it in mind: the app should support multiple user
PERSONAS (Employee, Finance/Admin, Executive, Investor). The SAME ledger is viewed
through different agent "lenses" — each persona gets a different system prompt, a
different ALLOWED SET of tools, and different default queries. Permission is enforced at
the agent layer: an Investor-mode agent must not even HAVE the tool to read an
individual employee's receipts.

Also internalize my Global Directives: everything must run on the live Anthropic API
(no offline simulation for new features — but keep deterministic functions like
evaluate_policy and the tieout/fraud/forecast models as TOOLS the agent calls and as the
ground-truth verifier); no static pages (configure → run → inspect); the UI must prompt
the user to select persona + scope + controls in stages before running, and the agent
may pause mid-run to ask a clarifying question; every answer returns a citable trace with
cost. Don't change anything yet — confirm you understand the code and this direction.
```

### Prompt 1 — the persona spine + shared agent-runner

```text
Apply the Global Directives. Build the persona spine.

1. Add a Persona enum: EMPLOYEE, FINANCE, EXECUTIVE, INVESTOR (e.g.
   comptroller/agents/personas.py).

2. Build a single TOOL REGISTRY of comptroller.ai.Tool objects (one module, e.g.
   comptroller/agents/agent_tools.py) wrapping the existing deterministic capabilities,
   each tagged with: allowed personas, and read|write. At minimum:
     - query_transactions(filters)            [read]  FINANCE, EXECUTIVE; EMPLOYEE only-self
     - get_my_status()                         [read]  EMPLOYEE
     - explain_policy(question)                [read]  all
     - run_fraud_scan(scope)                   [read]  FINANCE, EXECUTIVE
     - find_duplicate_spend()                  [read]  FINANCE, EXECUTIVE
     - vendor_price_changes()                  [read]  FINANCE, EXECUTIVE
     - subscription_audit()                    [read]  FINANCE, EXECUTIVE
     - compile_policy(text) / replay_policy()  [read]  FINANCE, EXECUTIVE  (Policy Studio)
     - forecast_cash(interventions)            [read]  EXECUTIVE, INVESTOR
     - underwrite(ticker?)                     [read]  EXECUTIVE, INVESTOR
     - company_aggregates()                    [read]  EXECUTIVE, INVESTOR   (NO per-employee)
     - flag_for_review(txn_id)                 [write] FINANCE
     - issue_card(spec)                        [write] FINANCE
     - pay_or_hold_invoice(id, action)         [write] FINANCE
   EMPLOYEE tools must be physically scoped to the caller's own card/txns (pass an
   employee_id into the runner and have those tools filter by it). INVESTOR gets ONLY
   read aggregates + underwriting + forecast — NO tool that returns employee-level rows.

3. Add allowed_tools(persona, employee_id=None) -> list[Tool]. The agent must PHYSICALLY
   receive only the permitted tools — not be asked nicely to avoid them.

4. Extract ONE reusable agent-runner (e.g. comptroller/agents/runner.py): run(persona,
   query, controls, model, employee_id=None) -> {answer, findings, trace, cost, usage,
   needs_input?}. It builds the persona system prompt + allowed_tools, calls
   ClaudeClient.run_agent, and returns the full trace. Support a mid-run clarifying
   question: if the model calls a special ask_user tool, return needs_input with 2-4
   options instead of an answer. Make agents/investigator + the existing Spend-Forensics
   idea become callers of this runner. This runner is LIVE-ONLY (requires the key).

5. Endpoints in app.py:
     POST /agent/run            {persona, query, controls, model, employee_id?}
     POST /agent/continue       {run_id|state, choice}   (answer a clarifying question)
     GET  /personas/{persona}/capabilities   -> tabs + tool names available
   The frontend uses /capabilities to show/hide tabs per persona.

6. Dashboard: add a PERSONA SWITCHER at the top. Switching persona re-queries
   /capabilities and changes which tabs are visible and what the agent may do. Show the
   active persona and its permission tier prominently.

Verify with the FastAPI TestClient (key present): an INVESTOR forensics call cannot
access employee-level tools (the tool is absent from its toolset), and a FINANCE call
can. Confirm EMPLOYEE tools only ever return the caller's own rows.
```

### Prompt 2 — Policy Studio

```text
Apply the Global Directives. Add a "Policy Studio" tab, available to FINANCE and
EXECUTIVE personas, built on the shared agent-runner.

UI (configure → run): a plain-English policy textbox, plus a staged control panel —
Step 1 confirm persona; Step 2 pick the replay window (date-range, default last 90 days),
a department multiselect (which teams to apply to), a model dropdown, effort; Step 3
"Compile & Simulate". The user selects at each step before running.

Behavior, via the runner (live Claude):
1. compile_policy tool: the agent reads the natural-language policy and compiles it into
   the structured SpendPolicy shape in comptroller/domain (category limits, blocked
   categories/MCCs, approval thresholds, receipt rules, weekend rules).
2. replay_policy tool: REPLAY the compiled rules against the historical transactions in
   the chosen window using the deterministic evaluate_policy engine. Report which
   transactions it would have blocked/flagged, GROUPED BY RULE, with total $ impact per
   rule, and a plain-English summary of what changed vs the current policy.
3. Return: compiled rules, affected transaction IDs, $ impact per rule, before/after
   compliance rate, and the agent trace.

The win is concrete ROI: "this policy would have caught $X across N transactions." The
compile step is live Claude (NL -> rules); the replay is deterministic (the verifier).
Verify against the planted anomalies in comptroller/data/synthetic.py — e.g. a policy
"no fuel, alcohol only with a meal, anything over the card limit needs approval" should
recover the blocked-category and over-limit anomalies. Render the per-rule $ impact as
bars and the affected transactions as a table with the receipt link.
```

### Prompt 3 — Investor / Board Room

```text
Apply the Global Directives. Add an "Investor Room" tab, available ONLY to the INVESTOR
persona (read-only), built on the shared agent-runner with the INVESTOR toolset.

UI (configure → run): a diligence question box, plus staged controls — Step 1 (persona
is locked to Investor); Step 2 optional public-company ticker to benchmark against (uses
comptroller/analytics/underwriting.py SEC pull), reporting-period selector, model
dropdown; Step 3 Run. Default the tab to a one-click "Generate diligence summary".

Behavior (live Claude tool-use):
1. The agent answers diligence questions ("monthly burn?", "runway?", "top 3 spend
   risks?", "is this company underwritable?") over the real ledger aggregates, plus the
   live SEC pull if a ticker is given.
2. CRITICAL — no hallucinated financials: EVERY dollar figure the agent reports must come
   from a tool result or pass a tieout check. If it can't verify a number, it must say so
   explicitly. Surface the tieout/verify block in the response (figure -> source ->
   verified Y/N). This is the eval-harness ethos turned into a product guarantee.
3. The INVESTOR toolset must NOT include any employee-level drill-down. Confirm that when
   asked "show me Jane's expenses" the agent declines because it lacks the tool (don't
   rely on prompt-only refusal).
4. The "Generate diligence summary" path returns: burn, runway, spend efficiency, top
   risks, and an underwriting grade — each with its source and tieout status — plus the
   trace and cost.
```

### Prompt 4 — Runway & Burn Forecasting (causal / Rung-2)  ← the differentiator

```text
Apply the Global Directives. Add a "Runway Lab" tab (EXECUTIVE + INVESTOR), built on the
shared agent-runner. This must be a CAUSAL / interventional simulator (Pearl's Rung 2 —
do-operator), NOT a spreadsheet projection. Make the distinction explicit in the product.

1. Structural causal model (comptroller/analytics/causal_runway.py). Define an explicit
   DAG over the cash drivers, fit from the tenant's history:
     headcount --> payroll --> burn
     price * customers --> revenue ;  churn --> customers(t) = customers(t-1)*(1-churn) + new
     marketing_spend --> new_customers (with a fitted CAC) --> revenue
     burn = payroll + vendor_spend + card_spend - revenue
     cash(t) = cash(t-1) - burn(t) ;  runway = first t where cash<=0
   Expose every edge/coefficient so the agent can cite them.

2. Two rungs, shown side by side:
   - Rung 1 (observational): the existing trend forecast — "given current trajectory,
     runway = N months." (reuse comptroller/analytics/forecasting.py)
   - Rung 2 (interventional): apply do(X) — SET a driver, hold others fixed per the DAG,
     PROPAGATE through the structural equations, recompute the cash trajectory + runway.
   The point: Rung 1 says what happens if nothing changes; Rung 2 answers
   "what happens IF WE hire 20 / churn rises 5% / we raise price 10%" — a different
   computation, not a re-plotted line.

3. UI (configure → run): interventional sliders/inputs — net hires, hiring start month,
   monthly churn, price change %, marketing spend — plus model dropdown. The user sets
   the intervention(s), then Run. Show: the baseline runway, the post-intervention
   runway, the delta, and the shifted cash curve. Let the user stack interventions and
   compare scenarios.

4. The agent (live Claude) does the narration + sensitivity: it calls a
   forecast_intervention tool with the chosen do(X), then explains the causal CHAIN
   ("hiring 20 raises payroll by ~$Y/mo -> burn +$Y -> runway -K months"), names which
   driver runway is most sensitive to, and proposes the cheapest intervention to hit a
   target runway. Every number cites a structural equation or tool result.

Frame it in the UI as "causal what-if (Rung 2)" vs "trend forecast (Rung 1)" so it reads
as interventional reasoning, which is the moat. Verify: do(hire +20) must REDUCE runway
vs baseline; do(churn -> 0) must INCREASE it; and the agent's narrated deltas must match
the structural recomputation (assert it in a TestClient test).
```

### Prompt 5 — AP / Bill Pay agent

```text
Apply the Global Directives. Add an "AP / Bill Pay" tab (FINANCE), built on the shared
agent-runner + Claude vision.

UI (configure → run): upload an invoice (PDF/image) OR pick a generated sample; staged
controls — match strictness, auto-approve threshold, model dropdown; then Run.

Behavior (live Claude vision + tool-use):
1. Vision-extract the invoice (reuse the extract_document pattern from
   receipt_autopilot): vendor, remit-to, BANK DETAILS, line items, PO number, totals,
   terms.
2. 3-way match: the agent calls tools to fetch the matching PO and the goods-receipt
   (synthesize a PO ledger in comptroller/documents/invoices.py), and reconciles
   quantity/price/total. Detect: duplicate invoices, price/quantity variance, math
   errors, and VENDOR-IMPERSONATION FRAUD (bank details changed vs. this vendor's
   history — keep a vendor bank-detail history to diff against).
3. Decision: pay / short-pay / hold, with the reasoning, the variances, and the dollar
   exposure. Return the trace.

Plant a double-billed invoice and a changed-bank-details invoice in the sample set and
verify the agent catches both.
```

### Prompt 6 — Employee Self-Service agent

```text
Apply the Global Directives. Add an "My Spend" tab (EMPLOYEE persona), built on the
shared agent-runner with the EMPLOYEE (own-data-only) toolset.

UI (configure → run): a persona-scoped chat. Step 1 the demo picks WHICH employee you're
signed in as (a dropdown of employees — this drives the employee_id scoping); Step 2 ask
a question or pick a suggested one ("what's my limit?", "why was this charge flagged?",
"how do I fix a missing receipt?"). Live Claude answers in plain language.

Hard requirement: the agent can ONLY see this employee's own transactions/cards (enforced
by allowed_tools(EMPLOYEE, employee_id) from Prompt 1). Verify it cannot answer
"what did the sales team spend" — the aggregate tool is not in its set. When a charge was
flagged, it explains why (cites the policy rule or fraud driver) and drafts the fix
(e.g., "upload the receipt / add a memo"). This completes the persona triangle so the
role-switch demo lands.
```

### Prompt 7 — Month-End Close agent (Tier 3)

```text
Apply the Global Directives. Add a "Close" tab (FINANCE), built on the shared
agent-runner. The agent runs a recurring close: reconcile card+cash to the GL, run the
forensics tools (duplicates, uncategorized, policy exceptions), estimate accruals (e.g.
cloud usage not yet invoiced), and DRAFT journal adjustments with a confidence score,
surfacing only the exceptions for human sign-off. Output a close memo. Each proposed
journal entry cites the transactions it rolls up; the close is not "done" until the
deterministic tieout passes (cash + card == GL within tolerance). Stream the steps.
```

### Prompt 8 — Card Issuance agent (Tier 3, write tool)

```text
Apply the Global Directives. Add a "Issue Card" action (FINANCE, write). The agent takes
a natural-language request ("a card for the new contractor, $2K/mo, software only, expires
in 90 days"), compiles it into a card spec (limit, allowed categories/MCCs, expiry,
vendor lock), shows the spec for CONFIRMATION (multi-step: it must ask the user to
approve before "issuing"), then calls the issue_card write tool to add a StripeCard to the
tenant. Confirm the new card is scoped exactly as requested and verify the limit/category
lock is enforced on a test charge.
```

### Prompt 9 — Treasury console (Tier 3)

```text
Apply the Global Directives. Add a "Treasury" tab (EXECUTIVE). The agent reads the cash
position + forecast, and proposes a LADDERED allocation of idle cash across the Stripe
Business Account tiers (checking buffer / Treasury yield / FDIC reserve) to maximize yield
while keeping N months of buffer liquid, explaining the trade-off and the incremental
annual yield. The user sets the buffer months and risk tolerance (multi-step), then Run.
Every yield/allocation number cites the rate and the balance it's computed from.
```

---

## 5. The causal moat — making Runway Lab read as Rung 2

This is the part that separates you from every other "AI finance dashboard." Pearl's
ladder:

- **Rung 1 — observational** ("seeing"): *given the trend, runway is N months.* This is
  what every forecasting tool does — fit a line, extend it.
- **Rung 2 — interventional** ("doing", the do-operator): *if WE hire 20, runway becomes
  M.* You **set** a variable, **sever its normal causes**, and **propagate** the change
  through the causal structure. It is a different computation, not a re-plotted forecast.
- **Rung 3 — counterfactual** ("imagining"): *had we not raised prices last quarter,
  where would cash be now?* (Optional stretch — same SCM, run backward on a realized
  trajectory.)

To make Runway Lab unmistakably Rung 2 (do this, don't just project):
1. **Write the structural equations down** (the DAG in Prompt 4) and expose the
   coefficients. A spreadsheet hides them; an SCM names them.
2. **Implement `do(X)`**: when the user sets `headcount += 20`, you don't re-fit on
   history — you **clamp** headcount, recompute payroll → burn → cash via the structural
   equations, leave the unaffected exogenous drivers as they were, and read off the new
   runway. Show baseline vs intervened side by side.
3. **Show the causal chain in the answer**, not just the number: "hire 20 → payroll
   +$Y/mo → burn +$Y → runway −K months." The chain *is* the explanation.
4. **Sensitivity / cheapest-lever**: have the agent sweep each driver and report which
   one runway is most elastic to, and the smallest intervention that hits a target
   runway. That's interventional optimization — pure Rung-2 framing.
5. **Label it in the UI**: a "Rung 1: trend" panel next to a "Rung 2: causal what-if"
   panel, so the distinction is the headline.

A Stripe interviewer reading this sees: *not a chart — a causal engine that answers
"what should we do" with a traced, defensible chain.*

---

## 6. Build order & portfolio story

1. **Prompt 1** (persona spine) — nothing works well until this exists.
2. **Prompt 2** (Policy Studio) — the instant-ROI demo.
3. **Prompt 3** (Investor Room) — the "no hallucinated financials" guarantee.
   → *That trio is a shippable, coherent v1: switch role, watch the product change.*
4. **Prompt 4** (Runway Lab, causal) — the differentiator; build it carefully.
5. **Prompts 5–6** — complete the surface (AP is what Stripe monetizes; Employee closes
   the persona triangle).
6. **Prompts 7–9** — round out the menu.

**The one-line pitch:** *"Stripe flags transactions; Comptroller runs a permission-scoped
reasoning agent that explains every decision with a citable, tieout-verified trace — and
answers 'what should we do' causally."*
