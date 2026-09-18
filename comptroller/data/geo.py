"""Lightweight geography helpers powering geo-velocity ("impossible travel") signals.

We keep a tiny city -> (lat, lon) table instead of a heavy geocoding dependency.
``haversine_km`` gives great-circle distance so the fraud feature layer can compute
implied travel speed between consecutive transactions on the same card.
"""
from __future__ import annotations

import math

# City -> (lat, lon). Covers the home metros used by the tenant generator plus a
# handful of "far away" metros used to plant impossible-travel fraud.
CITIES: dict[str, tuple[float, float]] = {
    "San Francisco, CA": (37.7749, -122.4194),
    "New York, NY": (40.7128, -74.0060),
    "Austin, TX": (30.2672, -97.7431),
    "Seattle, WA": (47.6062, -122.3321),
    "Chicago, IL": (41.8781, -87.6298),
    "Denver, CO": (39.7392, -104.9903),
    "Boston, MA": (42.3601, -71.0589),
    "Los Angeles, CA": (34.0522, -118.2437),
    "Miami, FL": (25.7617, -80.1918),
    "Atlanta, GA": (33.7490, -84.3880),
    # Far metros used for fraud bursts (often cross-border / high-risk).
    "Lagos, NG": (6.5244, 3.3792),
    "Bucharest, RO": (44.4268, 26.1025),
    "Singapore, SG": (1.3521, 103.8198),
    "London, GB": (51.5074, -0.1278),
    "Sao Paulo, BR": (-23.5505, -46.6333),
}

HOME_METROS: tuple[str, ...] = (
    "San Francisco, CA",
    "New York, NY",
    "Austin, TX",
    "Seattle, WA",
    "Chicago, IL",
    "Denver, CO",
    "Boston, MA",
    "Los Angeles, CA",
    "Atlanta, GA",
)

FAR_METROS: tuple[str, ...] = (
    "Lagos, NG",
    "Bucharest, RO",
    "Singapore, SG",
    "London, GB",
    "Sao Paulo, BR",
    "Miami, FL",
)


def haversine_km(a: str, b: str) -> float:
    """Great-circle distance in km between two known city labels.

    Unknown labels return 0.0 (treated as "co-located") so the feature layer
    degrades gracefully rather than raising.
    """
    if a == b:
        return 0.0
    pa, pb = CITIES.get(a), CITIES.get(b)
    if pa is None or pb is None:
        return 0.0
    lat1, lon1 = math.radians(pa[0]), math.radians(pa[1])
    lat2, lon2 = math.radians(pb[0]), math.radians(pb[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(h)))
