"""The fraud-scoring ML ensemble.

Two complementary learners are blended:

* **Isolation Forest** (unsupervised) — flags transactions that are anomalous in
  feature space even if we've never labelled their pattern. Catches novel fraud.
* **Gradient-boosted trees** (supervised) — learns the *interactions* that separate
  fraud from legitimate activity (e.g. a geo jump is benign on a trusted device but
  damning on a brand-new one), with class-imbalance weighting so rare fraud isn't
  drowned out.

The blended probability feeds risk banding, the causal explainer, and the eval
harness. Everything is deterministic given a seed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


@dataclass
class FraudMetrics:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    n: int
    n_fraud: int

    def to_dict(self) -> dict:
        return {
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "threshold": round(self.threshold, 4),
            "n": self.n,
            "n_fraud": self.n_fraud,
        }


class FraudModel:
    """Blended anomaly + supervised fraud scorer."""

    def __init__(self, random_state: int = 7, blend: float = 0.7):
        self.random_state = random_state
        self.blend = blend  # weight on the supervised classifier
        self.scaler = StandardScaler()
        self.iforest = IsolationForest(
            n_estimators=120, contamination="auto", random_state=random_state, n_jobs=-1)
        self.clf = GradientBoostingClassifier(
            n_estimators=160, max_depth=3, learning_rate=0.09,
            subsample=0.9, random_state=random_state)
        self.columns = list(FEATURE_COLUMNS)
        self._anom_min = 0.0
        self._anom_max = 1.0
        self.threshold = 0.5
        self.fitted = False

    # ---- internals -----------------------------------------------------------
    def _matrix(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X[self.columns].to_numpy(dtype=float)
        return np.asarray(X, dtype=float)

    def _anomaly(self, Xs: np.ndarray) -> np.ndarray:
        raw = -self.iforest.score_samples(Xs)  # higher = more anomalous
        rng = (self._anom_max - self._anom_min) or 1.0
        return np.clip((raw - self._anom_min) / rng, 0.0, 1.0)

    # ---- API -----------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "FraudModel":
        Xm = self._matrix(X)
        y = np.asarray(y, dtype=int)
        Xs = self.scaler.fit_transform(Xm)
        self.iforest.fit(Xs)
        raw = -self.iforest.score_samples(Xs)
        self._anom_min, self._anom_max = float(raw.min()), float(raw.max())
        # Inverse-frequency sample weights to counter class imbalance.
        pos = max(int(y.sum()), 1)
        neg = max(len(y) - pos, 1)
        w = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))
        self.clf.fit(Xs, y, sample_weight=w)
        self.fitted = True
        # Pick the F1-optimal probability threshold on the training data.
        self.threshold = self._best_threshold(self.predict_proba(X), y)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(self._matrix(X))
        clf_p = self.clf.predict_proba(Xs)[:, 1]
        anom = self._anomaly(Xs)
        return self.blend * clf_p + (1 - self.blend) * anom

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float | None = None) -> np.ndarray:
        thr = self.threshold if threshold is None else threshold
        return (self.predict_proba(X) >= thr).astype(int)

    @staticmethod
    def _best_threshold(scores: np.ndarray, y: np.ndarray) -> float:
        if y.sum() == 0:
            return 0.5
        prec, rec, thr = precision_recall_curve(y, scores)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
        # precision_recall_curve returns thresholds of len-1; align.
        best = int(np.argmax(f1[:-1])) if len(thr) else 0
        return float(thr[best]) if len(thr) else 0.5

    def evaluate(self, X: pd.DataFrame, y: np.ndarray | pd.Series,
                 threshold: float | None = None) -> FraudMetrics:
        y = np.asarray(y, dtype=int)
        scores = self.predict_proba(X)
        thr = self.threshold if threshold is None else threshold
        preds = (scores >= thr).astype(int)
        roc = float(roc_auc_score(y, scores)) if y.sum() and y.sum() < len(y) else float("nan")
        pr = float(average_precision_score(y, scores)) if y.sum() else float("nan")
        p, r, f1, _ = precision_recall_fscore_support(
            y, preds, average="binary", zero_division=0)
        return FraudMetrics(roc, pr, float(p), float(r), float(f1), thr, len(y), int(y.sum()))

    @property
    def feature_importances(self) -> dict[str, float]:
        if not self.fitted:
            return {}
        return {c: float(v) for c, v in sorted(
            zip(self.columns, self.clf.feature_importances_),
            key=lambda kv: kv[1], reverse=True)}
