"""Agents: deterministic engines, the orchestrator, and the investigator."""
from comptroller.agents import (
    CategorizationAgent,
    ComptrollerOrchestrator,
    FraudInvestigator,
    PolicyAuditAgent,
)
from comptroller.agents import inputs as build
from comptroller.agents.categorization import recoverable_from_name
from comptroller.llm import AnalyticalBackend


def _clean_txn(dataset):
    for t in dataset.card_transactions:
        if (not t.ground_truth.is_fraud and t.mcc != "5999"
                and t.ground_truth.true_category):
            return t
    raise AssertionError("no suitable transaction")


def test_categorization_clean_case(dataset):
    t = _clean_txn(dataset)
    out = CategorizationAgent().solve(build.categorization_inputs(t, dataset))
    assert out["category"] == t.ground_truth.true_category.value


def test_categorization_recovers_miscoded(dataset):
    target = None
    for t in dataset.card_transactions:
        m = dataset.merchant_index()[t.merchant_id]
        if (not t.ground_truth.is_fraud and t.mcc != "5999"
                and recoverable_from_name(m.name)):
            target = t
            break
    assert target is not None
    out = CategorizationAgent().solve(build.categorization_inputs(target, dataset, miscode=True))
    # MCC was hidden (5999); the engine recovers the category from the merchant name.
    assert out["category"] == target.ground_truth.true_category.value


def test_categorization_coerce_handles_junk(dataset):
    t = _clean_txn(dataset)
    out = CategorizationAgent().coerce({"category": "NONSENSE", "confidence": "x"},
                                       build.categorization_inputs(t, dataset))
    assert out["category"] == "other"
    assert isinstance(out["confidence"], float)


def test_policy_engine_matches_ground_truth(dataset):
    agent = PolicyAuditAgent()
    checked = 0
    for t in dataset.card_transactions:
        if t.ground_truth.is_fraud:
            continue
        out = agent.solve(build.policy_inputs(t, dataset))
        expected = sorted(v.value for v in t.ground_truth.policy_violations)
        assert sorted(out["violations"]) == expected
        checked += 1
        if checked >= 200:
            break
    assert checked > 0


def test_analytical_backend_runs(dataset):
    t = _clean_txn(dataset)
    res = AnalyticalBackend().run(CategorizationAgent(), build.categorization_inputs(t, dataset))
    assert res.ok and "category" in res.data


def test_orchestrator_produces_decision(dataset, pipeline):
    top = pipeline.top_alerts(1)[0]
    dec = ComptrollerOrchestrator(dataset, pipeline, AnalyticalBackend()).handle_transaction(
        top.txn_id)
    assert dec.txn_id == top.txn_id
    assert dec.recommended_actions
    assert len(dec.trace) >= 4
    assert dec.is_fraud is True  # top alert should trip the fraud path


def test_investigator_produces_report(dataset, pipeline):
    top = pipeline.top_alerts(1)[0]
    rep = FraudInvestigator(dataset, pipeline).investigate(top.txn_id)
    assert rep.recommended_actions
    assert rep.steps  # tool calls were recorded
    assert rep.narrative
