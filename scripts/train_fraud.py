"""Train and cross-validate the fraud ensemble; report metrics, drivers and rings.

    python scripts/train_fraud.py [--seed 7] [--folds 5]

Uses stratified k-fold CV for a robust ROC-AUC / PR-AUC estimate (more honest than a
single split), then trains on all data and prints feature importances and detected
fraud rings.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

from comptroller import reporting as R
from comptroller.data import generate_tenant
from comptroller.fraud import EntityGraph, FraudModel, build_feature_frame
from comptroller.fraud.pipeline import FraudPipeline


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    R.banner()
    R.rule("Fraud ensemble — cross-validated training")
    ds = generate_tenant(seed=args.seed)
    graph = EntityGraph(ds)
    frame = build_feature_frame(ds, graph)
    y = frame["y"].to_numpy(dtype=int)
    R.info(f"{len(frame):,} transactions | {int(y.sum())} fraud "
           f"({y.mean():.2%}) | {args.folds}-fold stratified CV")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rocs, prs, precs, recs, f1s = [], [], [], [], []
    for tr, te in skf.split(frame, y):
        model = FraudModel(random_state=args.seed).fit(frame.iloc[tr], y[tr])
        m = model.evaluate(frame.iloc[te], y[te])
        rocs.append(m.roc_auc); prs.append(m.pr_auc)
        precs.append(m.precision); recs.append(m.recall); f1s.append(m.f1)

    def ms(x):
        return f"{np.nanmean(x):.3f} ± {np.nanstd(x):.3f}"

    R.fraud_metrics({
        "roc_auc": float(np.nanmean(rocs)), "pr_auc": float(np.nanmean(prs)),
        "precision": float(np.nanmean(precs)), "recall": float(np.nanmean(recs)),
        "f1": float(np.nanmean(f1s)), "n": len(frame), "n_fraud": int(y.sum()),
        "threshold": 0.5,
    })
    R.console().print(
        f"[dim]CV spread — ROC-AUC {ms(rocs)} | PR-AUC {ms(prs)} | "
        f"precision {ms(precs)} | recall {ms(recs)} | F1 {ms(f1s)}[/dim]")

    pipe = FraudPipeline(ds, seed=args.seed)
    R.fraud_metrics(pipe.holdout_metrics.to_dict(), pipe.model.feature_importances)
    R.rings_table(pipe.rings())


if __name__ == "__main__":
    main()
