"""Shared fixtures. The tenant + trained pipeline are session-scoped (they're the
expensive objects) so the suite stays fast."""
import pytest

from comptroller.data import generate_tenant
from comptroller.fraud import FraudPipeline


@pytest.fixture(scope="session")
def dataset():
    return generate_tenant(seed=7)


@pytest.fixture(scope="session")
def pipeline(dataset):
    return FraudPipeline(dataset, seed=7)
