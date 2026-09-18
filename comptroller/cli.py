"""Comptroller command-line interface.

Subcommands:
  generate     build a synthetic Stripe tenant and print its summary
  fraud        train the fraud ensemble; show metrics, top alerts and rings
  investigate  run the autonomous fraud-investigation workflow on a transaction
  orchestrate  run the full Comptroller agent on a transaction (any backend)
  eval         run the multi-model financial-correctness leaderboard
  demo         end-to-end narrated showcase
  serve        launch the FastAPI service
"""
from __future__ import annotations

import argparse
import json
import sys

from . import reporting as R
from .config import load_config


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


def _build(seed: int, **spec_kwargs):
    from .data import GenSpec, generate_tenant
    from .fraud import FraudPipeline

    spec = GenSpec(seed=seed, **{k: v for k, v in spec_kwargs.items() if v is not None})
    ds = generate_tenant(spec)
    pipe = FraudPipeline(ds, seed=seed)
    return ds, pipe


def _resolve_backend(name: str, cfg):
    from .llm import AnalyticalBackend, build_backends
    if not name or name in ("offline", "offline-heuristic", "analytical"):
        return AnalyticalBackend()
    for b in build_backends(cfg):
        if name.lower() in b.name.lower():
            return b
    R.info(f"backend '{name}' not found; falling back to offline engine")
    return AnalyticalBackend()


def _pick_txn(pipe, txn: str | None, use_top: bool) -> str:
    if txn:
        return txn
    return pipe.top_alerts(1)[0].txn_id


# --------------------------------------------------------------------------- #
def cmd_generate(args, cfg) -> None:
    ds, _ = _build(args.seed, n_employees=args.employees, days=args.days)
    if args.json:
        print(json.dumps(ds.summary(), indent=2))
        return
    R.banner()
    R.dataset_summary(ds.summary())
    if args.save:
        path = cfg.data_dir / f"tenant_seed{args.seed}.json"
        path.write_text(ds.model_dump_json(), encoding="utf-8")
        R.info(f"saved tenant to {path}")


def cmd_fraud(args, cfg) -> None:
    ds, pipe = _build(args.seed)
    if args.json:
        print(json.dumps({
            "metrics": pipe.holdout_metrics.to_dict(),
            "rings": [r.to_dict() for r in pipe.rings()],
            "top_alerts": [a.to_dict() for a in pipe.top_alerts(args.top)],
        }, indent=2))
        return
    R.rule("Fraud intelligence")
    R.fraud_metrics(pipe.holdout_metrics.to_dict(), pipe.model.feature_importances)
    R.alerts_table(pipe.top_alerts(args.top), show_truth=True)
    R.rings_table(pipe.rings())


def cmd_investigate(args, cfg) -> None:
    from .agents import FraudInvestigator
    ds, pipe = _build(args.seed)
    txn_id = _pick_txn(pipe, args.txn, args.top)
    report = FraudInvestigator(ds, pipe).investigate(txn_id)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return
    R.rule("Autonomous fraud investigation")
    R.investigation(report.to_dict())


def cmd_orchestrate(args, cfg) -> None:
    from .agents import ComptrollerOrchestrator
    ds, pipe = _build(args.seed)
    backend = _resolve_backend(args.backend, cfg)
    txn_id = _pick_txn(pipe, args.txn, args.top)
    decision = ComptrollerOrchestrator(ds, pipe, backend).handle_transaction(txn_id)
    if args.json:
        print(json.dumps(decision.to_dict(), indent=2))
        return
    R.rule("Comptroller orchestrator")
    R.orchestrator_decision(decision.to_dict())


def cmd_eval(args, cfg) -> None:
    from .eval import EvalHarness, build_tasks
    from .llm import build_backends
    ds, pipe = _build(args.seed)
    tasks = build_tasks(args.tasks.split(",") if args.tasks else None)
    backends = build_backends(cfg)
    if not args.json:
        R.rule("Financial-correctness eval")
        R.info(f"{len(tasks)} tasks x {len(backends)} backends "
               f"({'LIVE Claude' if cfg.has_live_models else 'offline + simulated'}), "
               f"{args.limit} cases/task")
    report = EvalHarness(ds, pipe, seed=args.seed).run(
        tasks, backends, limit_per_task=args.limit)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return
    R.leaderboard(report)
    if args.task_detail:
        for t in report.tasks:
            R.task_detail(report, t)


def cmd_demo(args, cfg) -> None:
    from .demo import run_demo
    run_demo(seed=args.seed, eval_limit=args.limit)


def cmd_serve(args, cfg) -> None:
    import uvicorn
    R.info(f"serving Comptroller API on http://{args.host}:{args.port} (docs at /docs)")
    uvicorn.run("comptroller.api.app:app", host=args.host, port=args.port, reload=False)


# --------------------------------------------------------------------------- #
def build_parser(cfg) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="comptroller", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="build a synthetic Stripe tenant")
    g.add_argument("--seed", type=int, default=cfg.seed)
    g.add_argument("--employees", type=int, default=None)
    g.add_argument("--days", type=int, default=None)
    g.add_argument("--save", action="store_true")
    g.add_argument("--json", action="store_true")
    g.set_defaults(func=cmd_generate)

    f = sub.add_parser("fraud", help="train & show fraud intelligence")
    f.add_argument("--seed", type=int, default=cfg.seed)
    f.add_argument("--top", type=int, default=8)
    f.add_argument("--json", action="store_true")
    f.set_defaults(func=cmd_fraud)

    i = sub.add_parser("investigate", help="autonomous fraud investigation")
    i.add_argument("txn", nargs="?", default=None)
    i.add_argument("--top", action="store_true", help="investigate the top alert")
    i.add_argument("--seed", type=int, default=cfg.seed)
    i.add_argument("--json", action="store_true")
    i.set_defaults(func=cmd_investigate)

    o = sub.add_parser("orchestrate", help="run the full Comptroller agent")
    o.add_argument("txn", nargs="?", default=None)
    o.add_argument("--top", action="store_true")
    o.add_argument("--backend", default="offline",
                   help="offline | claude-opus-4-8 | claude-sonnet-4-6 | claude-haiku-4-5")
    o.add_argument("--seed", type=int, default=cfg.seed)
    o.add_argument("--json", action="store_true")
    o.set_defaults(func=cmd_orchestrate)

    e = sub.add_parser("eval", help="multi-model financial-correctness leaderboard")
    e.add_argument("--seed", type=int, default=cfg.seed)
    e.add_argument("--limit", type=int, default=40, help="cases per task")
    e.add_argument("--tasks", default=None, help="comma-separated subset")
    e.add_argument("--task-detail", action="store_true")
    e.add_argument("--json", action="store_true")
    e.set_defaults(func=cmd_eval)

    d = sub.add_parser("demo", help="end-to-end narrated showcase")
    d.add_argument("--seed", type=int, default=cfg.seed)
    d.add_argument("--limit", type=int, default=30, help="eval cases per task")
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("serve", help="launch the FastAPI service")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    _utf8_stdout()
    cfg = load_config()
    parser = build_parser(cfg)
    args = parser.parse_args(argv)
    args.func(args, cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
