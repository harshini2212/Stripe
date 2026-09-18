"""The eval harness and leaderboard."""
from comptroller.eval import EvalHarness, build_tasks
from comptroller.llm import AnalyticalBackend, SimulatedBackend


def _report(dataset, pipeline):
    backends = [AnalyticalBackend(), SimulatedBackend("weak-sim", skill=0.8)]
    return EvalHarness(dataset, pipeline, seed=7).run(
        build_tasks(), backends, limit_per_task=15, bootstrap=100)


def test_analytical_is_the_reference(dataset, pipeline):
    report = _report(dataset, pipeline)
    board = report.leaderboard()
    top = board[0]
    assert top["backend"] == "offline-heuristic"
    assert top["overall_accuracy"] >= 0.95


def test_simulated_underperforms_reference(dataset, pipeline):
    report = _report(dataset, pipeline)
    board = {r["backend"]: r["overall_accuracy"] for r in report.leaderboard()}
    assert board["offline-heuristic"] >= board["weak-sim"]


def test_leaderboard_is_sorted(dataset, pipeline):
    board = _report(dataset, pipeline).leaderboard()
    accs = [r["overall_accuracy"] for r in board]
    assert accs == sorted(accs, reverse=True)


def test_policy_violation_f1_perfect_for_engine(dataset, pipeline):
    report = _report(dataset, pipeline)
    s = report.score_for("policy_audit", "offline-heuristic")
    assert s.aux["violation_f1"] >= 0.99


def test_tieout_engine_is_exact(dataset, pipeline):
    report = _report(dataset, pipeline)
    s = report.score_for("tieout", "offline-heuristic")
    assert s.aux["mae_usd"] < 0.01
    assert s.accuracy >= 0.99


def test_scores_have_valid_confidence_intervals(dataset, pipeline):
    for s in _report(dataset, pipeline).scores:
        assert 0.0 <= s.ci_low <= s.accuracy <= s.ci_high <= 1.0 or s.n == 0
