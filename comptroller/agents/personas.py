"""Personas — the role lenses the same ledger is viewed through.

Permission is enforced by *which tools the agent receives* per persona (see
``agent_tools.allowed_tools``), not by prompting. An Investor-mode agent literally does
not possess the tool to open an individual employee's transactions.
"""
from __future__ import annotations

from enum import Enum


class Persona(str, Enum):
    EMPLOYEE = "employee"
    FINANCE = "finance"
    EXECUTIVE = "executive"
    INVESTOR = "investor"

    @property
    def tier(self) -> str:
        return {
            Persona.EMPLOYEE: "self · read-only",
            Persona.FINANCE: "full access · read + write",
            Persona.EXECUTIVE: "company-wide · read + forecast",
            Persona.INVESTOR: "aggregates · read-only diligence",
        }[self]

    @property
    def label(self) -> str:
        return {
            Persona.EMPLOYEE: "Employee",
            Persona.FINANCE: "Finance / Admin",
            Persona.EXECUTIVE: "Executive (CFO)",
            Persona.INVESTOR: "Investor / Board",
        }[self]


# Which tabs each persona may open. The frontend reads this to show/hide tabs.
PERSONA_TABS: dict[Persona, list[str]] = {
    Persona.EMPLOYEE: ["my_spend", "receipts"],
    Persona.FINANCE: ["forensics", "policy_studio", "ap", "receipts", "close", "card_issuance"],
    Persona.EXECUTIVE: ["forensics", "runway", "treasury", "investor_room"],
    Persona.INVESTOR: ["investor_room", "runway"],
}

TAB_TITLES = {
    "my_spend": "My Spend",
    "forensics": "Spend Forensics",
    "policy_studio": "Policy Studio",
    "ap": "Bill Pay / AP",
    "receipts": "Receipt Autopilot",
    "runway": "Runway Lab",
    "investor_room": "Investor Room",
    "close": "Month-End Close",
    "card_issuance": "Issue Card",
    "treasury": "Treasury",
}
