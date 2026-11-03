import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from test_data_generator import generate  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full kamplengde (ca. 30 s). Hopp over med -m 'not slow'")


@pytest.fixture(scope="session")
def short_match():
    return generate(seed=7, mode="short")


@pytest.fixture(scope="session")
def full_match():
    return generate(seed=7, mode="full")


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT
