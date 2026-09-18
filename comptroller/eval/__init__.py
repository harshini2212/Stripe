"""Financial-correctness evaluation harness and multi-model leaderboard."""
from .scorers import numeric_within, set_prf
from .tasks import EvalCase, EvalTask, build_tasks
from .harness import BackendTaskScore, EvalHarness, EvalReport

__all__ = [
    "numeric_within",
    "set_prf",
    "EvalCase",
    "EvalTask",
    "build_tasks",
    "BackendTaskScore",
    "EvalHarness",
    "EvalReport",
]
