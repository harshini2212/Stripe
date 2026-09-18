"""Rich console reporting for Comptroller — leaderboards, alerts, investigations.

Centralizes all human-facing output so the CLI and demo render consistently. Uses a
Stripe-flavored accent and degrades gracefully on legacy Windows terminals.
"""
from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ACCENT = "medium_purple"   # Stripe purple
GOOD = "green3"
WARN = "yellow3"
BAD = "red3"
_console = Console(highlight=False)

_BAND_STYLE = {"critical": "bold red3", "high": "red3", "medium": "yellow3", "low": "green3"}


def console() -> Console:
    return _console


def rule(title: str) -> None:
    _console.rule(f"[bold {ACCENT}]{title}")


def banner() -> None:
    _console.print(Panel.fit(
        Text.assemble(
            ("COMPTROLLER\n", f"bold {ACCENT}"),
            ("Agentic AI + financial-correctness evaluation for Stripe spend", "white"),
        ),
        border_style=ACCENT, padding=(0, 2)))


# --------------------------------------------------------------------------- #
def dataset_summary(summary: dict[str, Any]) -> None:
    t = Table(title="Synthetic Stripe tenant", title_style=f"bold {ACCENT}", show_header=False,
              box=None)
    t.add_column(style="white")
    t.add_column(justify="right", style="bold")
    order = ["company", "employees", "cards", "merchants", "card_transactions",
             "cash_transactions", "disputes", "total_card_spend_usd",
             "fraud_transactions", "policy_violations"]
    for k in order:
        if k in summary:
            v = summary[k]
            v = f"${v:,.0f}" if k == "total_card_spend_usd" else f"{v:,}" if isinstance(v, int) else v
            t.add_row(k.replace("_", " ").title(), str(v))
    _console.print(t)


def fraud_metrics(metrics: dict[str, Any], importances: dict[str, float] | None = None) -> None:
    body = Text()
    body.append(f"ROC-AUC   {metrics['roc_auc']:.3f}\n", style="bold")
    body.append(f"PR-AUC    {metrics['pr_auc']:.3f}\n")
    body.append(f"Precision {metrics['precision']:.3f}   Recall {metrics['recall']:.3f}   "
                f"F1 {metrics['f1']:.3f}\n")
    body.append(f"Holdout   {metrics['n']:,} txns ({metrics['n_fraud']} fraud) @ "
                f"threshold {metrics['threshold']:.2f}", style="dim")
    _console.print(Panel(body, title="Fraud model — held-out performance",
                         border_style=ACCENT, title_align="left"))
    if importances:
        t = Table(title="Top causal features", box=None, title_style="dim")
        t.add_column("feature"); t.add_column("importance", justify="right")
        for f, v in list(importances.items())[:8]:
            if v > 0.001:
                t.add_row(f, f"{v:.3f}")
        _console.print(t)


def _band(text: str) -> Text:
    return Text(text.upper(), style=_BAND_STYLE.get(text, "white"))


_ACTION_SHORT = {
    "freeze_card_and_open_dispute": "freeze+dispute",
    "open_dispute": "dispute",
    "monitor": "monitor",
    "clear": "clear",
}


def alerts_table(assessments: list, *, show_truth: bool = False) -> None:
    t = Table(title="Top fraud alerts", title_style=f"bold {ACCENT}", header_style="bold")
    t.add_column("txn"); t.add_column("employee"); t.add_column("merchant", max_width=18)
    t.add_column("amount", justify="right"); t.add_column("risk", justify="right")
    t.add_column("band"); t.add_column("action"); t.add_column("ring")
    if show_truth:
        t.add_column("actual")
    for a in assessments:
        row = [a.txn_id, a.employee_id, getattr(a, "merchant_name", a.merchant_id),
               f"${a.amount_usd:,.0f}", f"{a.risk_score:.0%}", _band(a.risk_band.value),
               _ACTION_SHORT.get(a.recommended_action, a.recommended_action),
               a.ring_id or "—"]
        if show_truth:
            row.append(Text("FRAUD", style="red3") if a.actual_fraud
                       else Text("legit", style="green3"))
        t.add_row(*row)
    _console.print(t)


def rings_table(rings: list) -> None:
    if not rings:
        _console.print("[dim]No fraud rings detected.[/dim]")
        return
    t = Table(title="Detected fraud rings", title_style=f"bold {ACCENT}", header_style="bold")
    t.add_column("ring"); t.add_column("cards", justify="right")
    t.add_column("txns", justify="right"); t.add_column("exposure", justify="right")
    t.add_column("shared dev", justify="right"); t.add_column("suspicion"); t.add_column("band")
    for r in rings:
        d = r.to_dict()
        t.add_row(d["ring_id"], str(len(d["cards"])), str(d["n_txns"]),
                  f"${d['total_exposure_usd']:,.0f}", str(r.shared_devices),
                  f"{d['suspicion']:.2f}", _band(d["risk_band"]))
    _console.print(t)


def investigation(report: dict[str, Any]) -> None:
    body = Text()
    body.append(report["narrative"] + "\n\n")
    body.append("Recommended actions:\n", style=f"bold {ACCENT}")
    for a in report["recommended_actions"]:
        body.append(f"  • {a}\n")
    body.append("\nAgent tool calls: ", style="dim")
    body.append(" -> ".join(report["steps"]), style="dim")
    _console.print(Panel(body, title=f"Fraud investigation — {report['txn_id']}",
                         border_style=ACCENT, title_align="left"))


def orchestrator_decision(decision: dict[str, Any]) -> None:
    body = Text()
    for line in decision["trace"]:
        body.append("  " + line + "\n")
    body.append("\nConsolidated actions:\n", style=f"bold {ACCENT}")
    for a in decision["recommended_actions"]:
        body.append(f"  • {a}\n")
    if decision.get("dispute"):
        d = decision["dispute"]
        body.append(f"\nDispute {d['dispute_id']}: {d.get('recommendation')} "
                    f"(cardholder_should_win={d.get('cardholder_should_win')})\n", style="white")
    u = decision["usage"]
    body.append(f"\nfinancial impact ${decision['financial_impact_usd']:,.0f}  |  "
                f"backend {decision['backend']}  |  {decision['latency_ms']:.0f} ms  |  "
                f"{u['input_tokens']}+{u['output_tokens']} tok  |  ${u['cost_usd']:.4f}",
                style="dim")
    _console.print(Panel(body, title=f"Comptroller decision — {decision['txn_id']}",
                         border_style=ACCENT, title_align="left"))


def leaderboard(report) -> None:
    rows = report.leaderboard()
    tasks = report.tasks
    t = Table(title="Financial-correctness leaderboard", title_style=f"bold {ACCENT}",
              header_style="bold", caption="accuracy vs held-out ground truth; "
              "deterministic engine is the reference")
    t.add_column("backend"); t.add_column("kind", style="dim")
    t.add_column("overall", justify="right")
    for task in tasks:
        t.add_column(task.replace("_", "\n"), justify="right")
    t.add_column("cost", justify="right"); t.add_column("ms", justify="right")
    for i, r in enumerate(rows):
        style = f"bold {ACCENT}" if i == 0 else ""
        cells = [Text(r["backend"], style=style), r["kind"],
                 Text(f"{r['overall_accuracy']:.3f}", style=style or "bold")]
        for task in tasks:
            v = r["by_task"].get(task)
            cells.append("—" if v is None else f"{v:.2f}")
        cells.append(f"${r['total_cost_usd']:.3f}")
        cells.append(f"{r['avg_latency_ms']:.0f}")
        t.add_row(*cells)
    _console.print(t)


def task_detail(report, task: str) -> None:
    t = Table(title=f"Task: {task}", title_style=f"bold {ACCENT}", header_style="bold")
    t.add_column("backend"); t.add_column("accuracy", justify="right")
    t.add_column("95% CI"); t.add_column("aux"); t.add_column("err", justify="right")
    for s in sorted([s for s in report.scores if s.task == task],
                    key=lambda x: x.accuracy, reverse=True):
        aux = ", ".join(f"{k}={v:.3f}" for k, v in s.aux.items()) or "—"
        t.add_row(s.backend, f"{s.accuracy:.3f}",
                  f"[{s.ci_low:.2f}, {s.ci_high:.2f}]", aux, f"{s.error_rate:.0%}")
    _console.print(t)


def info(msg: str) -> None:
    _console.print(f"[dim]{msg}[/dim]")
