"""Company research on live SEC EDGAR data.

Ported from the tieout/Hebbia project's EDGAR ingestion, slimmed to the
`companyfacts` JSON API (no arelle): ticker search -> CIK -> XBRL facts ->
an analyst-ready profile with annual series, ratios, and accounting tie-outs.
"""
from .edgar import EdgarClient, build_company_profile

__all__ = ["EdgarClient", "build_company_profile"]
