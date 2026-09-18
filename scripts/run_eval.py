"""Run the financial-correctness leaderboard and optionally persist it.

    python scripts/run_eval.py [--seed 7] [--limit 40] [--tasks a,b] [--detail] [--save]

With ANTHROPIC_API_KEY set, the simulated rows are replaced by live Claude Opus 4.8 /
Sonnet 4.6 / Haiku 4.5 and the board grades real model output.
"""
from __future__ import annotations

import argparse
import json
import sys

from comptroller import reporting as R
from comptroller.config import load_config
from comptroller.data import generate_tenant
from comptroller.eval import EvalHarness, build_tasks
from comptroller.fraud import FraudPipeline
from comptroller.llm import build_backends


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=40, help="cases per task")
    ap.add_argument("--tasks", default=None, help="comma-separated subset")
    ap.add_argument("--detail", action="store_true", help="per-task breakdown")
    ap.add_argument("--save", action="store_true", help="write JSON to artifacts/")
    args = ap.parse_args()

    cfg = load_config()
    R.banner()
    R.rule("Financial-correctness leaderboard")
    R.info(f"live models: {cfg.has_live_models} | "
           f"{'Claude Opus/Sonnet/Haiku' if cfg.has_live_models else 'offline + simulated'}")

    ds = generate_tenant(seed=args.seed)
    pipe = FraudPipeline(ds, seed=args.seed)
    tasks = build_tasks(args.tasks.split(",") if args.tasks else None)
    report = EvalHarness(ds, pipe, seed=args.seed).run(
        tasks, build_backends(cfg), limit_per_task=args.limit)

    R.leaderboard(report)
    if args.detail:
        for t in report.tasks:
            R.task_detail(report, t)

    if args.save:
        path = cfg.artifact_dir / f"leaderboard_seed{args.seed}.json"
        path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        R.info(f"saved leaderboard to {path}")


if __name__ == "__main__":
    main()
