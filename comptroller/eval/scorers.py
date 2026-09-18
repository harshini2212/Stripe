"""Deterministic scoring primitives used by the eval tasks.

Financial correctness needs more than string-equality: set membership for policy
violations, numeric tolerance for dollar tieouts, and exact matching for categorical
decisions. These helpers keep the task definitions terse.
"""
from __future__ import annotations

from typing import Iterable


def exact(a, b) -> bool:
    return a == b


def numeric_within(pred: float, expected: float, tol: float = 0.005) -> bool:
    """True if two dollar figures agree within ``tol`` (default half a cent)."""
    try:
        return abs(float(pred) - float(expected)) <= tol
    except (TypeError, ValueError):
        return False


def set_prf(pred: Iterable[str], expected: Iterable[str]) -> dict:
    """Precision / recall / F1 / exact-match for a predicted vs expected set."""
    p, e = set(pred), set(expected)
    tp = len(p & e)
    fp = len(p - e)
    fn = len(e - p)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall,
            "f1": f1, "exact": p == e}
