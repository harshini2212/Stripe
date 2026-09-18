"""End-to-end narrated showcase tying every subsystem together.

Run via ``comptroller demo`` or ``python -m comptroller.demo``. Generates a Stripe
tenant, trains the fraud ensemble, runs the autonomous investigation + orchestrator
workflows, and prints the multi-model financial-correctness leaderboard.
"""
from __future__ import annotations

from . import reporting as R
from .config import load_config


def run_demo(seed: int = 7, eval_limit: int = 30) -> None:
    from .agents import ComptrollerOrchestrator, FraudInvestigator
    from .data import generate_tenant
    from .eval import EvalHarness, build_tasks
    from .fraud import FraudPipeline
    from .llm import AnalyticalBackend, build_backends

    cfg = load_config()
    R.banner()
    R.info(f"seed={seed} | live models: {cfg.has_live_models} "
           f"({'Claude Opus/Sonnet/Haiku' if cfg.has_live_models else 'offline + simulated'})")

    # 1) Tenant -----------------------------------------------------------------
    R.rule("1 / Synthetic Stripe tenant (Stripe Issuing card + Stripe Treasury)")
    ds = generate_tenant(seed=seed)
    R.dataset_summary(ds.summary())

    # 2) Fraud intelligence -----------------------------------------------------
    R.rule("2 / Fraud intelligence — graph + behavioral ML + causal explanations")
    pipe = FraudPipeline(ds, seed=seed)
    R.fraud_metrics(pipe.holdout_metrics.to_dict(), pipe.model.feature_importances)
    alerts = pipe.top_alerts(8)
    R.alerts_table(alerts, show_truth=True)
    R.rings_table(pipe.rings())

    # 3) Autonomous investigation ----------------------------------------------
    R.rule("3 / Autonomous fraud investigation (multi-step agent + tools)")
    target = alerts[0].txn_id
    investigation = FraudInvestigator(ds, pipe).investigate(target)
    R.investigation(investigation.to_dict())

    # 4) Orchestrator -----------------------------------------------------------
    R.rule("4 / Comptroller orchestrator — one agent, many tasks, one decision")
    decision = ComptrollerOrchestrator(ds, pipe, AnalyticalBackend()).handle_transaction(target)
    R.orchestrator_decision(decision.to_dict())
    # also resolve a real dispute end-to-end
    if ds.disputes:
        dtxn = ds.disputes[0].transaction_id
        ddec = ComptrollerOrchestrator(ds, pipe, AnalyticalBackend()).handle_transaction(dtxn)
        if ddec.dispute:
            R.info(f"Dispute {ddec.dispute['dispute_id']}: "
                   f"{ddec.dispute.get('recommendation')} "
                   f"(cardholder_should_win={ddec.dispute.get('cardholder_should_win')}, "
                   f"exposure ${ddec.financial_impact_usd:,.0f})")

    # 5) Eval leaderboard -------------------------------------------------------
    R.rule("5 / Financial-correctness leaderboard (the eval harness)")
    report = EvalHarness(ds, pipe, seed=seed).run(
        build_tasks(), build_backends(cfg), limit_per_task=eval_limit)
    R.leaderboard(report)

    R.rule("Done")
    R.info("Set ANTHROPIC_API_KEY to replace the simulated rows with live "
           "Claude Opus 4.8 / Sonnet 4.6 / Haiku 4.5 and grade real model output.")


if __name__ == "__main__":  # pragma: no cover
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    run_demo()
