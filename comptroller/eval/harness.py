"""The evaluation runner and multi-model leaderboard.

Builds golden cases once per task (so every backend is graded on identical inputs),
runs each agent through each backend, scores against held-out ground truth, and
aggregates into per-(task, backend) scores with bootstrap confidence intervals plus a
cross-task leaderboard. Cost and latency are tracked so the board answers not just
"which model is most correct" but "at what price".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..domain import Dataset
from ..llm.base import Backend
from .tasks import EvalCase, EvalTask


@dataclass
class CaseResult:
    task: str
    backend: str
    case_id: str
    correct: bool
    metrics: dict[str, Any]
    latency_ms: float
    cost_usd: float
    ok: bool
    error: str | None = None


@dataclass
class BackendTaskScore:
    task: str
    backend: str
    kind: str
    n: int
    accuracy: float
    ci_low: float
    ci_high: float
    aux: dict[str, float]
    latency_ms: float
    cost_usd: float
    error_rate: float

    def to_dict(self) -> dict:
        return {
            "task": self.task, "backend": self.backend, "kind": self.kind, "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "ci95": [round(self.ci_low, 4), round(self.ci_high, 4)],
            "aux": {k: round(v, 4) for k, v in self.aux.items()},
            "latency_ms": round(self.latency_ms, 1),
            "cost_usd": round(self.cost_usd, 6),
            "error_rate": round(self.error_rate, 4),
        }


@dataclass
class EvalReport:
    backends: list[tuple[str, str]]  # (name, kind)
    tasks: list[str]
    scores: list[BackendTaskScore]
    results: list[CaseResult] = field(default_factory=list)
    seed: int = 7

    def score_for(self, task: str, backend: str) -> BackendTaskScore | None:
        for s in self.scores:
            if s.task == task and s.backend == backend:
                return s
        return None

    def leaderboard(self) -> list[dict]:
        """Cross-task macro leaderboard, best overall first."""
        rows = []
        for name, kind in self.backends:
            per_task = {t: (self.score_for(t, name).accuracy if self.score_for(t, name) else None)
                        for t in self.tasks}
            accs = [a for a in per_task.values() if a is not None]
            cost = sum(self.score_for(t, name).cost_usd for t in self.tasks
                       if self.score_for(t, name))
            lat = np.mean([self.score_for(t, name).latency_ms for t in self.tasks
                           if self.score_for(t, name)]) if accs else 0.0
            rows.append({
                "backend": name, "kind": kind,
                "overall_accuracy": round(float(np.mean(accs)), 4) if accs else 0.0,
                "by_task": {t: (round(a, 4) if a is not None else None)
                            for t, a in per_task.items()},
                "total_cost_usd": round(cost, 6),
                "avg_latency_ms": round(float(lat), 1),
            })
        rows.sort(key=lambda r: r["overall_accuracy"], reverse=True)
        return rows

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "backends": [{"name": n, "kind": k} for n, k in self.backends],
            "tasks": self.tasks,
            "leaderboard": self.leaderboard(),
            "scores": [s.to_dict() for s in self.scores],
        }


class EvalHarness:
    def __init__(self, dataset: Dataset, pipeline, *, seed: int = 7):
        self.dataset = dataset
        self.pipeline = pipeline
        self.seed = seed

    def run(self, tasks: list[EvalTask], backends: list[Backend], *,
            limit_per_task: int = 40, bootstrap: int = 1000) -> EvalReport:
        # Build cases ONCE per task so every backend sees identical inputs.
        cases_by_task: dict[str, list[EvalCase]] = {}
        for i, task in enumerate(tasks):
            rng = np.random.default_rng(self.seed + 1000 * (i + 1))
            cases_by_task[task.name] = task.build_cases(
                self.dataset, self.pipeline, rng, limit_per_task)

        results: list[CaseResult] = []
        scores: list[BackendTaskScore] = []
        for task in tasks:
            cases = cases_by_task[task.name]
            for backend in backends:
                task_results: list[CaseResult] = []
                for case in cases:
                    res = backend.run(task.agent, case.inputs)
                    if res.ok:
                        metrics = task.score(res.data, case.expected)
                    else:
                        metrics = {"correct": False}
                    cr = CaseResult(
                        task.name, backend.name, case.case_id,
                        bool(metrics.get("correct", False)), metrics,
                        res.latency_ms, res.usage.cost_usd, res.ok, res.error)
                    task_results.append(cr)
                    results.append(cr)
                scores.append(self._aggregate(task.name, backend, task_results, bootstrap))

        return EvalReport(
            backends=[(b.name, b.kind) for b in backends],
            tasks=[t.name for t in tasks],
            scores=scores, results=results, seed=self.seed)

    # ---- aggregation ---------------------------------------------------------
    def _aggregate(self, task: str, backend: Backend, rows: list[CaseResult],
                   bootstrap: int) -> BackendTaskScore:
        n = len(rows)
        correct = np.array([1.0 if r.correct else 0.0 for r in rows])
        acc = float(correct.mean()) if n else 0.0
        lo, hi = self._bootstrap_ci(correct, bootstrap)
        aux = self._aux(rows)
        latency = float(np.mean([r.latency_ms for r in rows])) if n else 0.0
        cost = float(sum(r.cost_usd for r in rows))
        err_rate = float(np.mean([0.0 if r.ok else 1.0 for r in rows])) if n else 0.0
        return BackendTaskScore(task, backend.name, backend.kind, n, acc, lo, hi,
                                aux, latency, cost, err_rate)

    def _bootstrap_ci(self, correct: np.ndarray, b: int) -> tuple[float, float]:
        if len(correct) == 0 or b <= 0:
            return (0.0, 0.0)
        rng = np.random.default_rng(self.seed)
        idx = rng.integers(0, len(correct), size=(b, len(correct)))
        means = correct[idx].mean(axis=1)
        return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

    @staticmethod
    def _aux(rows: list[CaseResult]) -> dict[str, float]:
        aux: dict[str, float] = {}
        m = [r.metrics for r in rows]
        if any("viol_tp" in x for x in m):
            tp = sum(x.get("viol_tp", 0) for x in m)
            fp = sum(x.get("viol_fp", 0) for x in m)
            fn = sum(x.get("viol_fn", 0) for x in m)
            prec = tp / (tp + fp) if (tp + fp) else 1.0
            rec = tp / (tp + fn) if (tp + fn) else 1.0
            aux["violation_f1"] = (2 * prec * rec / (prec + rec)) if (prec + rec) else 1.0
            aux["approval_accuracy"] = float(np.mean([x.get("approval_ok", False) for x in m]))
        if any("abs_error_usd" in x for x in m):
            aux["mae_usd"] = float(np.mean([x.get("abs_error_usd", 0.0) for x in m]))
            aux["ties_accuracy"] = float(np.mean([x.get("ties_ok", False) for x in m]))
        return aux
