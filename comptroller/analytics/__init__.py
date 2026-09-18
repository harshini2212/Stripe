"""Financial-operations analytics: treasury forecasting, underwriting, spend & AP intel.

Each module takes a :class:`~comptroller.domain.models.Dataset` and produces decision-grade
output for one Stripe surface. Together with the fraud + eval stack they form the full
acquisition-grade platform.
"""
from .forecasting import CashForecast, TreasuryForecaster
from .underwriting import CreditAssessment, Underwriter
from .spend import SpendIntelligence
from .ap import APIntelligence
from .registry import model_registry

__all__ = [
    "CashForecast",
    "TreasuryForecaster",
    "CreditAssessment",
    "Underwriter",
    "SpendIntelligence",
    "APIntelligence",
    "model_registry",
]
