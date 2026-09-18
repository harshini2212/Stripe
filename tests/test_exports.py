"""Scheduled exports: serializers, cadence math, the clock-driven runner, email
delivery, and the API."""
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient

from comptroller.api import app
from comptroller.exports import delivery as D
from comptroller.exports import serializers as S
from comptroller.exports.scheduler import ScheduleStore, next_run_after

client = TestClient(app)
NOW = datetime(2026, 6, 24, 8, 0, tzinfo=timezone.utc)


def _sched(**kw):
    base = dict(id="exp_x", name="Test export", dataset="cards", recipient="f@stripe.com",
                cadence="daily", filters={})
    base.update(kw)
    return SimpleNamespace(**base)


# ---- serializers ---------------------------------------------------------- #
def test_transaction_rows_header_and_filter(dataset, pipeline):
    all_rows = list(S.transaction_rows(dataset, pipeline, {}))
    assert all_rows[0] == S.TXN_HEADER
    fraud_rows = list(S.transaction_rows(dataset, pipeline, {"fraud": True}))
    # filtering to fraud-only never grows the set and drops at least one legit row
    assert 1 < len(fraud_rows) < len(all_rows)
    # every data row carries the 12 columns of the header
    assert all(len(r) == len(S.TXN_HEADER) for r in fraud_rows[1:])


def test_dict_rows_match_column_map():
    items = [{"id": "INV-1", "vendor": "Acme", "amount_usd": 10.0, "due": "2026-01-01",
              "po_id": "PO-1", "bank_account": "****1234", "status": "open", "anomaly": "clean"}]
    rows = list(S.dict_rows("invoices", items))
    assert rows[0][0] == "invoice_id" and rows[0][1] == "vendor"
    assert rows[1][0] == "INV-1" and rows[1][2] == 10.0


def test_csv_has_bom_and_lowercase_booleans():
    csv_text = S.rows_to_csv([["a", "b"], [True, None]])
    assert csv_text.startswith("﻿")          # Excel-friendly UTF-8 BOM
    assert "true," in csv_text and csv_text.rstrip().endswith(",")  # None -> empty cell


def test_materialize_counts_data_rows_only(dataset, pipeline):
    text, n = S.materialize(S.transaction_rows(dataset, pipeline, {"fraud": True}))
    assert text.startswith("﻿")
    assert n == text.replace("﻿", "").strip().count("\n")  # header excluded from n


# ---- cadence math --------------------------------------------------------- #
def test_next_run_alignment():
    dt = datetime(2026, 6, 24, 10, 30, tzinfo=timezone.utc)  # a Wednesday, 10:30
    assert next_run_after("hourly", dt) == datetime(2026, 6, 24, 11, 0, tzinfo=timezone.utc)
    nd = next_run_after("daily", dt)
    assert nd.hour == 8 and nd.day == 25                       # 08:00 already passed -> tomorrow
    nw = next_run_after("weekly", dt)
    assert nw.weekday() == 0 and nw > dt                       # next Monday
    nm = next_run_after("monthly", dt)
    assert nm.day == 1 and nm.month == 7                       # first of next month


# ---- the runner ----------------------------------------------------------- #
def _store(tmp_path, clock):
    runs = []

    def runner(sched):
        runs.append(sched.id)
        return ("col\r\nx\r\ny\r\n", 2)

    # force outbox delivery so the suite never depends on (or hits) a real SMTP server
    return ScheduleStore(runner, tmp_path / "schedules.json", tmp_path / "runs",
                         clock=clock, delivery=D.Delivery(D.SMTPConfig()))


def test_run_one_writes_artifact_and_advances(tmp_path):
    t = datetime(2026, 6, 24, 7, 0, tzinfo=timezone.utc)
    store = _store(tmp_path, lambda: t)
    s = store.create(name="Daily cards", dataset="cards", cadence="daily", recipient="f@stripe.com")
    run = store.run_one(s)
    assert run["ok"] and run["rows"] == 2 and run["bytes"] > 0
    assert store.run_path(s.id, run["id"]).exists()
    # the run is stamped and the next firing is scheduled strictly in the future
    assert s.last_run_at == t.isoformat()
    assert datetime.fromisoformat(s.next_run_at) > t


def test_run_due_only_fires_past_due_enabled(tmp_path):
    now = {"t": datetime(2026, 6, 24, 7, 0, tzinfo=timezone.utc)}
    store = _store(tmp_path, lambda: now["t"])
    due = store.create(name="hourly", dataset="people", cadence="hourly", recipient="f@stripe.com")
    store.update(due.id, enabled=True)
    paused = store.create(name="paused", dataset="cards", cadence="hourly", recipient="f@stripe.com")
    store.update(paused.id, enabled=False)
    # jump past both next-run times
    now["t"] = now["t"].replace(hour=9)
    fired = store.run_due()
    assert len(fired) == 1                          # only the enabled one fired
    assert store.get(due.id).runs and not store.get(paused.id).runs


def test_invalid_dataset_or_cadence_rejected(tmp_path):
    store = _store(tmp_path, lambda: datetime(2026, 6, 24, tzinfo=timezone.utc))
    for kwargs in ({"dataset": "nope", "cadence": "daily"}, {"dataset": "cards", "cadence": "yearly"}):
        try:
            store.create(name="x", recipient="f@stripe.com", **kwargs)
            assert False, "should have raised"
        except ValueError:
            pass


def test_delete_removes_run_artifacts(tmp_path):
    t = datetime(2026, 6, 24, tzinfo=timezone.utc)
    store = _store(tmp_path, lambda: t)
    s = store.create(name="temp", dataset="cards", cadence="daily", recipient="f@stripe.com")
    run = store.run_one(s)
    artifact = store.run_path(s.id, run["id"])
    assert artifact.exists()
    store.delete(s.id)
    assert not artifact.exists() and not (tmp_path / "runs" / s.id).exists()


def test_store_persists_across_reload(tmp_path):
    t = datetime(2026, 6, 24, tzinfo=timezone.utc)
    s1 = _store(tmp_path, lambda: t)
    sched = s1.create(name="keep", dataset="invoices", cadence="weekly", recipient="f@stripe.com")
    s2 = ScheduleStore(lambda x: ("", 0), tmp_path / "schedules.json", tmp_path / "runs", clock=lambda: t)
    assert s2.get(sched.id) is not None and s2.get(sched.id).name == "keep"


# ---- dataset CSV endpoints ------------------------------------------------ #
def test_dataset_csv_endpoints_stream_attachments():
    for ds in ("transactions", "cards", "people", "invoices"):
        r = client.get(f"/api/{ds}.csv", params={"seed": 7})
        assert r.status_code == 200
        assert "attachment" in r.headers["content-disposition"]
        assert r.text.startswith("﻿") and "\r\n" in r.text


def test_transactions_csv_full_export_beats_display_cap():
    capped = client.get("/api/transactions", params={"seed": 7, "limit": 250}).json()
    csv_text = client.get("/api/transactions.csv", params={"seed": 7}).text
    data_rows = csv_text.replace("﻿", "").strip().count("\n")  # minus header
    assert data_rows == capped["count"] and data_rows > capped["shown"]


# ---- schedule API --------------------------------------------------------- #
def test_schedule_api_create_run_download_delete():
    created = client.post("/api/exports/schedules", json={
        "name": "Weekly flagged spend", "dataset": "transactions", "cadence": "weekly",
        "recipient": "controller@stripe.com", "filters": {"flagged": True}}).json()
    sid = created["id"]
    assert created["next_run_at"] and created["enabled"] is True
    # it shows up in the list
    listing = client.get("/api/exports/schedules").json()["schedules"]
    assert any(s["id"] == sid for s in listing)
    # run on demand -> a downloadable artifact
    run = client.post(f"/api/exports/schedules/{sid}/run").json()
    assert run["ok"] and run["rows"] > 0
    dl = client.get(f"/api/exports/schedules/{sid}/runs/{run['id']}/download")
    assert dl.status_code == 200 and dl.text.startswith("﻿")
    # pause via PATCH
    assert client.patch(f"/api/exports/schedules/{sid}", json={"enabled": False}).json()["enabled"] is False
    # clean up
    assert client.delete(f"/api/exports/schedules/{sid}").json()["deleted"] == sid


def test_schedule_api_rejects_bad_dataset():
    r = client.post("/api/exports/schedules", json={
        "name": "bad", "dataset": "ledger", "cadence": "daily", "recipient": "x@stripe.com"})
    assert r.status_code == 400


# ---- email delivery ------------------------------------------------------- #
def test_build_email_is_multipart_with_csv_attachment():
    msg = D.build_email("comptroller@stripe.com", _sched(name="Weekly cards"),
                        b"card_id,spend\r\nC1,100\r\n", "cards.csv", NOW, 1)
    assert msg["To"] == "f@stripe.com" and "Weekly cards" in msg["Subject"]
    attachments = [p for p in msg.walk() if p.get_content_disposition() == "attachment"]
    assert len(attachments) == 1 and attachments[0].get_filename() == "cards.csv"


def test_outbox_delivery_writes_real_eml():
    res = D.Delivery(D.SMTPConfig()).deliver(_sched(), b"a,b\r\n1,2\r\n", "x.csv", NOW, 1)
    assert res.channel == "outbox" and res.ok
    assert res.eml_bytes.startswith(b"From:") or b"Subject:" in res.eml_bytes
    assert "Test export" in res.subject


def test_smtp_delivery_calls_smtplib(monkeypatch):
    sent = []

    class FakeSMTP:
        def __init__(self, host, port, timeout=0):
            self.host, self.port = host, port

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def ehlo(self):
            pass

        def starttls(self):
            sent.append(("starttls",))

        def login(self, user, pw):
            sent.append(("login", user))

        def send_message(self, msg):
            sent.append(("send", msg["To"], msg["Subject"]))

    monkeypatch.setattr(D.smtplib, "SMTP", FakeSMTP)
    cfg = D.SMTPConfig(host="smtp.test", port=587, username="u", password="p", use_tls=True)
    res = D.Delivery(cfg).deliver(_sched(recipient="cfo@stripe.com"), b"x\r\n1\r\n", "c.csv", NOW, 1)
    assert res.channel == "smtp" and res.ok
    assert ("starttls",) in sent and ("login", "u") in sent
    assert any(s[0] == "send" and s[1] == "cfo@stripe.com" for s in sent)


def test_smtp_failure_keeps_eml_and_reports_error(monkeypatch):
    class BoomSMTP:
        def __init__(self, *a, **k):
            raise OSError("connection refused")

    monkeypatch.setattr(D.smtplib, "SMTP", BoomSMTP)
    cfg = D.SMTPConfig(host="smtp.test", port=587)
    res = D.Delivery(cfg).deliver(_sched(), b"x\r\n1\r\n", "c.csv", NOW, 1)
    assert res.channel == "smtp" and res.ok is False and res.error
    assert res.eml_bytes  # the message is still preserved for retry / download


def test_run_one_attaches_delivery_record(tmp_path):
    store = _store(tmp_path, lambda: NOW)
    s = store.create(name="Daily people", dataset="people", cadence="daily", recipient="f@stripe.com")
    run = store.run_one(s)
    assert run["eml"] is True and run["delivery"]["channel"] == "outbox"
    assert store.run_path(s.id, run["id"], "eml").exists()


def test_api_email_preview_and_eml_download():
    created = client.post("/api/exports/schedules", json={
        "name": "Email preview test", "dataset": "cards", "cadence": "daily",
        "recipient": "controller@stripe.com"}).json()
    sid = created["id"]
    run = client.post(f"/api/exports/schedules/{sid}/run").json()
    assert run["eml"] is True and run["delivery"] is not None
    preview = client.get(f"/api/exports/schedules/{sid}/runs/{run['id']}/email").json()
    assert preview["to"] == "controller@stripe.com" and "Email preview test" in preview["subject"]
    assert preview["attachment"]["filename"].endswith(".csv") and preview["attachment"]["bytes"] > 0
    eml = client.get(f"/api/exports/schedules/{sid}/runs/{run['id']}/download", params={"kind": "eml"})
    assert eml.status_code == 200 and eml.headers["content-type"].startswith("message/rfc822")
    assert client.delete(f"/api/exports/schedules/{sid}").json()["deleted"] == sid
