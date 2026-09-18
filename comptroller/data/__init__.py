"""Synthetic Stripe tenant generation and golden eval datasets."""
from .synthetic import GenSpec, generate_tenant
from .geo import CITIES, haversine_km

__all__ = ["GenSpec", "generate_tenant", "CITIES", "haversine_km"]
