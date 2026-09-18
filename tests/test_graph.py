"""Entity graph + fraud-ring detection."""
from comptroller.fraud import EntityGraph


def test_rings_detected_with_shared_infrastructure(pipeline):
    rings = pipeline.rings()
    assert len(rings) >= 1
    for r in rings:
        assert len(r.card_ids) >= 2
        assert r.shared_devices >= 1 or r.shared_ips >= 1
        assert r.n_txns > 0


def test_office_ips_excluded_from_rings(pipeline):
    """Single-metro corporate egress IPs (172.16.x) must not appear as ring links."""
    for r in pipeline.rings():
        for ip in r.ip_ids:
            assert not ip.startswith("172.16."), ip


def test_txn_graph_features_present(dataset):
    g = EntityGraph(dataset)
    feats = g.txn_graph_features()
    sample = next(iter(feats.values()))
    for key in ("device_card_fanout", "ip_card_fanout", "ip_cross_metro",
                "ring_component_size", "in_suspected_ring"):
        assert key in sample


def test_ring_exposure_is_scoped(pipeline, dataset):
    """Ring exposure reflects suspicious activity, not all card spend."""
    total_spend = sum(t.amount_cents for t in dataset.card_transactions)
    for r in pipeline.rings():
        assert r.total_exposure_cents < total_spend
