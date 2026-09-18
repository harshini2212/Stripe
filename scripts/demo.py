"""End-to-end Comptroller showcase.

    python scripts/demo.py [--seed 7] [--limit 30]
"""
from __future__ import annotations

import argparse
import sys

from comptroller.demo import run_demo


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--limit", type=int, default=30, help="eval cases per task")
    args = ap.parse_args()
    run_demo(seed=args.seed, eval_limit=args.limit)


if __name__ == "__main__":
    main()
