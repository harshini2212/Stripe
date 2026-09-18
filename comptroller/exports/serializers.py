"""CSV serialization for every exportable dataset.

Column order mirrors the on-screen data grids so a downloaded CSV matches exactly what
the user sorted and saw. Booleans render lowercase, ``None`` renders empty, and every
writer emits a UTF-8 BOM so Excel opens the file as UTF-8 instead of mojibake.

Two shapes of producer:
  * ``transaction_rows`` filters the raw transaction stream (the dataset with a row cap
    in the UI, so the export must re-derive the full matching set).
  * ``dict_rows`` projects an already-aggregated list of dicts (cards / people / invoices)
    through a fixed column map — same dicts the JSON endpoints return.
"""
from __future__ import annotations

import csv
import io
from typing import Any, Iterable, Iterator

DATASETS = ("transactions", "cards", "people", "invoices")

# header label -> key in the dict returned by the matching /api/<dataset> endpoint
COLUMNS: dict[str, list[tuple[str, str]]] = {
    "cards": [("card_id", "card_id"), ("employee", "employee"), ("department", "department"),
              ("type", "type"), ("last4", "last4"), ("status", "status"),
              ("per_txn_limit_usd", "per_txn_limit_usd"), ("monthly_limit_usd", "monthly_limit_usd"),
              ("spend_usd", "spend_usd"), ("txns", "txns")],
    "people": [("name", "name"), ("department", "department"), ("role", "role"),
               ("email", "email"), ("spend_usd", "spend_usd"), ("flagged", "flagged")],
    "invoices": [("invoice_id", "id"), ("vendor", "vendor"), ("amount_usd", "amount_usd"),
                 ("due", "due"), ("po_id", "po_id"), ("bank_account", "bank_account"),
                 ("status", "status"), ("anomaly", "anomaly")],
}

TXN_HEADER = ["txn_id", "date", "merchant", "employee", "department", "category",
              "amount_usd", "channel", "has_receipt", "flags", "is_fraud", "fraud_score"]


def transaction_rows(ds: Any, pipe: Any, filters: dict | None = None) -> Iterator[list[Any]]:
    """Yield the header then one row per transaction matching ``filters``.

    Supported filters (all optional): ``q`` (merchant/id substring), ``category``,
    ``department``, ``flagged`` (policy violation or fraud), ``fraud``, ``employee_id``.
    The predicates are identical to the ``/api/transactions`` JSON endpoint so the CSV and
    the table never disagree.
    """
    f = filters or {}
    q = (f.get("q") or "").lower()
    category = f.get("category")
    department = f.get("department")
    flagged = bool(f.get("flagged"))
    fraud = bool(f.get("fraud"))
    employee_id = f.get("employee_id")

    scores = pipe.scores()
    mi = ds.merchant_index()
    emp = ds.employee_index()
    emp_dept = {e.id: e.department for e in ds.employees}

    yield list(TXN_HEADER)
    for t in ds.card_transactions:
        cat = (t.ground_truth.true_category.value if t.ground_truth.true_category else "other")
        if employee_id and t.employee_id != employee_id:
            continue
        if category and cat != category:
            continue
        if department and emp_dept.get(t.employee_id) != department:
            continue
        viol = [v.value for v in t.ground_truth.policy_violations]
        if flagged and not (viol or t.ground_truth.is_fraud):
            continue
        if fraud and not t.ground_truth.is_fraud:
            continue
        name = mi[t.merchant_id].name
        if q and q not in name.lower() and q not in t.id.lower():
            continue
        yield [t.id, t.ts.date().isoformat(), name,
               emp[t.employee_id].name if t.employee_id in emp else t.employee_id,
               emp_dept.get(t.employee_id, "?"), cat, round(t.amount, 2), t.channel.value,
               t.has_receipt, " ".join(viol), bool(t.ground_truth.is_fraud),
               round(float(scores.get(t.id, 0.0)), 3)]


def dict_rows(dataset: str, items: Iterable[dict]) -> Iterator[list[Any]]:
    """Project pre-aggregated dicts (cards / people / invoices) through ``COLUMNS``."""
    cols = COLUMNS[dataset]
    yield [h for h, _ in cols]
    for it in items:
        yield [it.get(k) for _, k in cols]


def _cell(v: Any) -> Any:
    if isinstance(v, bool):
        return "true" if v else "false"
    return "" if v is None else v


def _write(buf: io.StringIO, writer: Any, row: list[Any]) -> None:
    writer.writerow([_cell(c) for c in row])


def rows_to_csv(rows: Iterable[list[Any]], bom: bool = True) -> str:
    """Materialize all rows into one CSV string (header included)."""
    buf = io.StringIO()
    if bom:
        buf.write("﻿")
    writer = csv.writer(buf)
    for row in rows:
        _write(buf, writer, row)
    return buf.getvalue()


def stream_csv(rows: Iterable[list[Any]], bom: bool = True) -> Iterator[str]:
    """Yield CSV one row at a time so a large export never buffers whole in memory."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    if bom:
        buf.write("﻿")
    for row in rows:
        _write(buf, writer, row)
        chunk = buf.getvalue()
        buf.seek(0)
        buf.truncate(0)
        yield chunk


def materialize(rows: Iterable[list[Any]], bom: bool = True) -> tuple[str, int]:
    """Return ``(csv_text, data_row_count)`` — the count excludes the header row."""
    collected = list(rows)
    return rows_to_csv(collected, bom=bom), max(0, len(collected) - 1)
