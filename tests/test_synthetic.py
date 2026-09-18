"""Synthetic tenant: determinism, structure, and rule-engine consistency."""
from comptroller.data import generate_tenant
from comptroller.domain.policy import evaluate_policy


def test_generation_is_deterministic():
    a = generate_tenant(seed=7)
    b = generate_tenant(seed=7)
    assert a.summary() == b.summary()
    assert [t.id for t in a.card_transactions[:50]] == [t.id for t in b.card_transactions[:50]]
    assert a.card_transactions[0].amount_cents == b.card_transactions[0].amount_cents


def test_different_seeds_differ():
    assert generate_tenant(seed=1).summary() != generate_tenant(seed=2).summary()


def test_summary_is_well_formed(dataset):
    s = dataset.summary()
    assert s["card_transactions"] > 1000
    assert s["fraud_transactions"] > 0
    assert s["policy_violations"] > 0
    assert s["disputes"] > 0
    assert s["total_card_spend_usd"] > 0


def test_ground_truth_matches_rule_engine(dataset):
    """Every non-fraud transaction's labelled violations equal the canonical engine's."""
    cards = dataset.card_index()
    mismatches = 0
    for t in dataset.card_transactions:
        if t.ground_truth.is_fraud:
            continue
        is_dup = "duplicate_spend" in [v.value for v in t.ground_truth.policy_violations]
        v = evaluate_policy(
            amount_cents=t.amount_cents, category=t.ground_truth.true_category,
            has_receipt=t.has_receipt, is_weekend=t.ts.weekday() >= 5,
            card_per_txn_limit_cents=cards[t.card_id].per_txn_limit_cents,
            policy=dataset.policy, is_duplicate=is_dup)
        if sorted(x.value for x in v) != sorted(x.value for x in t.ground_truth.policy_violations):
            mismatches += 1
    assert mismatches == 0


def test_disputes_have_outcomes(dataset):
    for d in dataset.disputes:
        assert d.ground_truth.dispute_should_win in (True, False)
