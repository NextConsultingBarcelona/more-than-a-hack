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


# --- possession-pipelinen (src/): syntetisk kamp i de ekte filformatene -----------------------
for _extra in ("evaluation", "starter", "tools"):
    _path = str(REPO_ROOT / _extra)
    if _path not in sys.path:
        sys.path.insert(0, _path)


@pytest.fixture(scope="session")
def fake_match(tmp_path_factory):
    """Skrevet falsk kamp: {'dir', 'sim', 'paths'}. Ca. 13 min video, 10 min spill."""
    import make_fake_match

    out = str(tmp_path_factory.mktemp("fake_match"))
    sim = make_fake_match.simulate(seed=7, half_s=300.0)
    paths = make_fake_match.write(sim, out)
    return {"dir": out, "sim": sim, "paths": paths}


@pytest.fixture(scope="session")
def fake_labels(fake_match):
    from src import pipeline

    sim = fake_match["sim"]
    return pipeline.label_match(fake_match["dir"], sim["blocks"], sim["n_frames"], use_cache=False)
