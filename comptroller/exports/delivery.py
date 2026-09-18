"""Email delivery for scheduled exports.

Every run builds one real, standards-compliant MIME message (``build_email``) with the
CSV attached. How it leaves the building depends on configuration:

  * **smtp**  — if ``SMTP_HOST`` is set, the message is sent over SMTP via ``smtplib``
    (STARTTLS / SSL / auth all supported). This is a real send.
  * **outbox** — otherwise the exact same message is written to the run's outbox as a
    ``.eml`` file: a genuine RFC-822 email you can open in Outlook/Apple Mail or forward.
    This keeps the demo and tests credential-free while exercising the real message path.

Swapping providers never touches the scheduler — it just calls ``Delivery.deliver``.
"""
from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from email.utils import format_datetime, make_msgid
from typing import Any


@dataclass
class SMTPConfig:
    host: str | None = None
    port: int = 587
    username: str | None = None
    password: str | None = None
    sender: str = "comptroller@stripe.com"
    use_tls: bool = True       # STARTTLS on a plain connection
    use_ssl: bool = False      # implicit TLS (SMTPS, usually port 465)

    @classmethod
    def from_env(cls) -> "SMTPConfig":
        def flag(name: str, default: bool) -> bool:
            return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")

        return cls(
            host=os.environ.get("SMTP_HOST") or None,
            port=int(os.environ.get("SMTP_PORT", "587")),
            username=os.environ.get("SMTP_USERNAME") or None,
            password=os.environ.get("SMTP_PASSWORD") or None,
            sender=os.environ.get("SMTP_FROM", "comptroller@stripe.com"),
            use_tls=flag("SMTP_STARTTLS", True),
            use_ssl=flag("SMTP_SSL", False),
        )

    @property
    def configured(self) -> bool:
        return bool(self.host)


@dataclass
class DeliveryResult:
    channel: str          # "smtp" | "outbox"
    ok: bool
    detail: str           # human-readable summary for the UI
    subject: str
    to: str
    body: str
    eml_bytes: bytes      # the full RFC-822 message (always produced)
    error: str | None = None


def _filters_summary(filters: dict[str, Any] | None) -> str:
    if not filters:
        return "none"
    parts = [k if v is True else f"{k}={v}" for k, v in filters.items() if v not in (None, "", False)]
    return ", ".join(parts) or "none"


def build_email(sender: str, sched: Any, csv_bytes: bytes, filename: str,
                now: datetime, rows: int) -> EmailMessage:
    """Construct the MIME email for one export run (CSV attached). Pure + deterministic
    except for the Message-ID domain, which is irrelevant to delivery."""
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = sched.recipient
    msg["Subject"] = f"[Comptroller] {sched.name} — {now:%b %d, %Y}"
    msg["Date"] = format_datetime(now)
    msg["Message-ID"] = make_msgid(domain="comptroller.stripe.com")
    msg["X-Comptroller-Schedule"] = sched.id
    msg["X-Comptroller-Dataset"] = sched.dataset
    msg.set_content(
        f"Your scheduled export is attached.\n\n"
        f"  Schedule : {sched.name}\n"
        f"  Dataset  : {sched.dataset}\n"
        f"  Filters  : {_filters_summary(sched.filters)}\n"
        f"  Rows     : {rows:,}\n"
        f"  Cadence  : {sched.cadence}\n"
        f"  Generated: {now:%Y-%m-%d %H:%M UTC}\n\n"
        f"— Comptroller, automated finance exports for Stripe\n")
    msg.add_attachment(csv_bytes, maintype="text", subtype="csv", filename=filename)
    return msg


def _send_smtp(cfg: SMTPConfig, msg: EmailMessage) -> None:
    if cfg.use_ssl:
        server: smtplib.SMTP = smtplib.SMTP_SSL(cfg.host, cfg.port, timeout=20)
    else:
        server = smtplib.SMTP(cfg.host, cfg.port, timeout=20)
    with server:
        server.ehlo()
        if cfg.use_tls and not cfg.use_ssl:
            server.starttls()
            server.ehlo()
        if cfg.username:
            server.login(cfg.username, cfg.password or "")
        server.send_message(msg)


class Delivery:
    """Selects a transport from config and delivers a run's email."""

    def __init__(self, config: SMTPConfig | None = None) -> None:
        self.cfg = config or SMTPConfig.from_env()

    def deliver(self, sched: Any, csv_bytes: bytes, filename: str,
                now: datetime, rows: int) -> DeliveryResult:
        msg = build_email(self.cfg.sender, sched, csv_bytes, filename, now, rows)
        eml = msg.as_bytes()
        subject, to = msg["Subject"], sched.recipient
        body = msg.get_body(preferencelist=("plain",)).get_content()  # type: ignore[union-attr]

        if self.cfg.configured:
            try:
                _send_smtp(self.cfg, msg)
                return DeliveryResult("smtp", True,
                                      f"Sent via {self.cfg.host}:{self.cfg.port} to {to}",
                                      subject, to, body, eml)
            except Exception as exc:
                # never lose an export to a flaky mail server — the .eml is still written
                return DeliveryResult("smtp", False,
                                      f"SMTP send failed ({self.cfg.host}:{self.cfg.port}) — kept .eml",
                                      subject, to, body, eml, error=str(exc))
        return DeliveryResult("outbox", True, f"Wrote .eml to outbox for {to}",
                              subject, to, body, eml)
