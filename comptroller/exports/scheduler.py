"""In-process scheduled exports: persistent schedules + a clock-driven runner.

A schedule says *export <dataset> matching <filters> every <cadence> to <recipient>*.
``run_due(now)`` fires every schedule whose ``next_run_at`` has passed: it calls the
injected ``runner`` to produce the CSV, writes that to the run store as an outbox drop,
records a run, and advances ``next_run_at``. Delivery here is a downloadable file; a real
deployment swaps the body of ``run_one`` (or the ``runner``) for SMTP / Stripe notifications
without touching the scheduling logic.

The clock is injectable so tests are fully deterministic — no sleeping, no wall-clock.
"""
from __future__ import annotations

import json
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .serializers import DATASETS

CADENCES = ("hourly", "daily", "weekly", "monthly")

# A runner takes a schedule and returns (csv_text, data_row_count).
Runner = Callable[["ExportSchedule"], "tuple[str, int]"]
Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def next_run_after(cadence: str, dt: datetime) -> datetime:
    """First firing strictly after ``dt``.

    ``hourly`` rolls to the top of the next hour. ``daily`` / ``weekly`` / ``monthly`` align
    to 08:00 UTC, with weekly landing on Monday and monthly on the 1st — so "every Monday
    morning" actually lands on a Monday, not "7 days from whenever you clicked save".
    """
    if cadence == "hourly":
        return dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    eight = dt.replace(hour=8, minute=0, second=0, microsecond=0)
    if cadence == "weekly":
        days = (0 - eight.weekday()) % 7  # Monday == 0
        nxt = eight + timedelta(days=days)
        while nxt <= dt:
            nxt += timedelta(days=7)
        return nxt
    if cadence == "monthly":
        year, month = (dt.year + 1, 1) if dt.month == 12 else (dt.year, dt.month + 1)
        return dt.replace(year=year, month=month, day=1, hour=8, minute=0, second=0, microsecond=0)
    # daily (default)
    nxt = eight
    while nxt <= dt:
        nxt += timedelta(days=1)
    return nxt


@dataclass
class ExportSchedule:
    id: str
    name: str
    dataset: str
    cadence: str
    recipient: str
    filters: dict = field(default_factory=dict)
    seed: int = 7
    fmt: str = "csv"
    enabled: bool = True
    created_at: str = ""
    last_run_at: str | None = None
    next_run_at: str | None = None
    runs: list = field(default_factory=list)

    def public(self) -> dict:
        d = asdict(self)
        d["runs"] = d["runs"][:10]
        return d


class ScheduleStore:
    """Thread-safe registry of schedules, persisted to JSON, with run artifacts on disk."""

    def __init__(self, runner: Runner, store_path: Path, runs_dir: Path,
                 clock: Clock = _utcnow, delivery: Any = None) -> None:
        from .delivery import Delivery
        self._runner = runner
        self._path = Path(store_path)
        self._runs_dir = Path(runs_dir)
        self._clock = clock
        self._delivery = delivery or Delivery()  # defaults to outbox (.eml) unless SMTP_* is set
        self._lock = threading.RLock()
        self._schedules: dict[str, ExportSchedule] = {}
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # ---- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            for d in json.loads(self._path.read_text("utf-8")):
                self._schedules[d["id"]] = ExportSchedule(**d)
        except Exception:
            pass  # a corrupt store should never wedge the service

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(s) for s in self._schedules.values()]
        self._path.write_text(json.dumps(payload, indent=2), "utf-8")

    # ---- CRUD --------------------------------------------------------------
    def list(self) -> list[ExportSchedule]:
        with self._lock:
            return sorted(self._schedules.values(), key=lambda s: s.created_at)

    def get(self, sid: str) -> ExportSchedule | None:
        return self._schedules.get(sid)

    def create(self, *, name: str, dataset: str, cadence: str, recipient: str,
               filters: dict | None = None, seed: int = 7) -> ExportSchedule:
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")
        if cadence not in CADENCES:
            raise ValueError(f"unknown cadence {cadence!r}; choose from {CADENCES}")
        with self._lock:
            now = self._clock()
            sid = "exp_" + uuid.uuid4().hex[:8]
            sched = ExportSchedule(
                id=sid, name=name or f"Export {dataset}", dataset=dataset, cadence=cadence,
                recipient=recipient or "finance@stripe.com", filters=dict(filters or {}), seed=seed,
                created_at=now.isoformat(), next_run_at=next_run_after(cadence, now).isoformat())
            self._schedules[sid] = sched
            self._save()
            return sched

    def update(self, sid: str, **changes: Any) -> ExportSchedule | None:
        with self._lock:
            sched = self._schedules.get(sid)
            if sched is None:
                return None
            for key in ("name", "cadence", "recipient", "enabled", "filters", "seed"):
                if changes.get(key) is not None:
                    setattr(sched, key, changes[key])
            if changes.get("cadence"):
                sched.next_run_at = next_run_after(sched.cadence, self._clock()).isoformat()
            self._save()
            return sched

    def delete(self, sid: str) -> bool:
        with self._lock:
            if sid not in self._schedules:
                return False
            del self._schedules[sid]
            self._save()
        shutil.rmtree(self._runs_dir / sid, ignore_errors=True)  # drop this schedule's outbox
        return True

    # ---- running -----------------------------------------------------------
    def run_one(self, sched: ExportSchedule, now: datetime | None = None) -> dict:
        """Generate + deliver one export, record the run, advance ``next_run_at``."""
        now = now or self._clock()
        run_id = "run_" + uuid.uuid4().hex[:8]
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        filename = f"{sched.dataset}-{stamp}.csv"
        try:
            text, rows = self._runner(sched)  # runner is pure CSV generation — kept outside the lock
            data = text.encode("utf-8")
            target_dir = self._runs_dir / sched.id
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / f"{run_id}.csv").write_bytes(data)
            res = self._delivery.deliver(sched, data, filename, now, rows)  # build + send/drop the email
            if res.eml_bytes:
                (target_dir / f"{run_id}.eml").write_bytes(res.eml_bytes)
            run = {"id": run_id, "at": now.isoformat(), "rows": rows, "bytes": len(data),
                   "recipient": sched.recipient, "filename": filename, "ok": True, "error": None,
                   "eml": bool(res.eml_bytes),
                   "delivery": {"channel": res.channel, "ok": res.ok, "detail": res.detail,
                                "subject": res.subject, "to": res.to, "error": res.error}}
        except Exception as exc:  # a failing export records the failure, never crashes the loop
            run = {"id": run_id, "at": now.isoformat(), "rows": 0, "bytes": 0,
                   "recipient": sched.recipient, "filename": filename, "ok": False, "error": str(exc),
                   "eml": False, "delivery": None}
        with self._lock:
            sched.runs.insert(0, run)
            del sched.runs[25:]
            sched.last_run_at = now.isoformat()
            sched.next_run_at = next_run_after(sched.cadence, now).isoformat()
            self._save()
        return run

    def run_due(self, now: datetime | None = None) -> list[dict]:
        """Fire every enabled schedule whose ``next_run_at`` has passed."""
        now = now or self._clock()
        with self._lock:
            due = [s for s in self._schedules.values()
                   if s.enabled and s.next_run_at
                   and datetime.fromisoformat(s.next_run_at) <= now]
        return [self.run_one(s, now) for s in due]

    def run_now(self, sid: str, now: datetime | None = None) -> dict | None:
        sched = self.get(sid)
        return None if sched is None else self.run_one(sched, now)

    def run_path(self, sid: str, run_id: str, ext: str = "csv") -> Path | None:
        path = self._runs_dir / sid / f"{run_id}.{ext.lstrip('.')}"
        return path if path.exists() else None
