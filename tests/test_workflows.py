"""Receipt Autopilot workflow + document generation.

These use the deterministic simulation path (``client=None``) so the suite stays fast
and reproducible regardless of whether a live key is present — the live vision path is
exercised manually / via the endpoint.
"""
from comptroller.documents import build_sample_receipts
from comptroller.workflows import ReceiptAutopilot


def test_receipts_render_real_pngs(dataset):
    receipts = build_sample_receipts(dataset, seed=7)
    assert len(receipts) >= 5
    for r in receipts:
        assert r.png[:4] == b"\x89PNG"        # genuine PNG image
        assert r.charged_amount > 0


def test_autopilot_matches_and_flags(dataset, pipeline):
    auto = ReceiptAutopilot(dataset, pipeline, client=None)
    by_anomaly = {}
    for r in build_sample_receipts(dataset, seed=7):
        res = auto.process(r.png, "image/png", known=r)
        assert res["ok"]
        assert res["match"]["status"] == "matched"
        assert res["match"]["txn_id"] == r.txn_id          # matched the right transaction
        by_anomaly[r.anomaly] = res

    # clean receipts auto-approve with no flags
    clean = by_anomaly["clean"]
    assert clean["verdict"] == "auto_approve"
    assert not clean["flags"]

    # amount mismatch is caught and ties_out is false
    mm = by_anomaly["amount_mismatch"]
    assert any(f["type"] == "amount_mismatch" for f in mm["flags"])
    assert mm["reconciliation"]["ties_out"] is False
    assert mm["verdict"] in ("needs_review", "reject")

    # alcohol / personal items are caught
    pi = by_anomaly["personal_items"]
    assert any(f["type"] == "personal_items" for f in pi["flags"])


def test_autopilot_upload_without_vision_is_graceful(dataset, pipeline):
    """An uploaded image with no vision client returns a clear message, not a crash."""
    auto = ReceiptAutopilot(dataset, pipeline, client=None)
    res = auto.process(b"\x89PNG_not_a_real_image", "image/png", known=None)
    assert res["ok"] is False and "ANTHROPIC_API_KEY" in res["message"]
