"""Tester for Sportec-fasiten (external_validation/build_dataset.py).

Regel (arrangøren, løse baller): laget som sist hadde ballen beholder den.
Kjør:  python -m pytest tests/test_ground_truth.py -q
"""
import numpy as np

from external_validation.build_dataset import fill_owner, make_gt


def obj(*values):
    return np.array(values, dtype=object)


def test_los_ball_beholder_forrige_lag():
    owner = obj("H", None, None, "A")
    period = np.array([1, 1, 1, 1])
    assert list(fill_owner(owner, period)) == ["H", "H", "H", "A"]


def test_starten_av_omgangen_bruker_neste_eier():
    owner = obj(None, None, "A", "H")
    period = np.array([1, 1, 1, 1])
    assert list(fill_owner(owner, period)) == ["A", "A", "A", "H"]


def test_fyller_ikke_paa_tvers_av_omganger():
    # 1. omgang slutter med H, 2. omgang starter uten eier og får neste eier (A)
    owner = obj("H", None, None, "A")
    period = np.array([1, 1, 2, 2])
    assert list(fill_owner(owner, period)) == ["H", "H", "A", "A"]


def test_dod_ball_er_fortsatt_D():
    clean = dict(owner=obj("H", None, None, "A"),
                 alive=np.array([True, False, True, True]),
                 period=np.array([1, 1, 1, 1]))
    gt = make_gt(clean, n_total=4, idx=np.arange(4))
    assert list(gt["label"]) == ["H", "D", "H", "A"]