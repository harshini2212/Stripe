"""Deterministic scoring primitives."""
from comptroller.eval.scorers import numeric_within, set_prf


def test_numeric_within_tolerance():
    assert numeric_within(100.00, 100.004)
    assert not numeric_within(100.00, 100.02)
    assert not numeric_within("oops", 1.0)


def test_set_prf_exact():
    r = set_prf(["a", "b"], ["a", "b"])
    assert r["exact"] and r["f1"] == 1.0


def test_set_prf_partial():
    r = set_prf(["a"], ["a", "b"])
    assert not r["exact"]
    assert r["tp"] == 1 and r["fn"] == 1 and r["fp"] == 0
    assert r["recall"] == 0.5 and r["precision"] == 1.0


def test_set_prf_empty_sets_are_perfect():
    r = set_prf([], [])
    assert r["exact"] and r["f1"] == 1.0
