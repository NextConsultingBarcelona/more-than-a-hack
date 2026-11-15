"""Tester for Sportec-fasiten (external_validation/build_dataset.py).

Regel (arrangøren, løse baller): laget som sist hadde ballen beholder den.
Kjør:  python -m pytest tests/test_ground_truth.py -q
"""
import numpy as np

from external_validation.build_dataset import controlled_owner, fill_owner, make_gt


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
    gt = make_gt(clean, n_total=4, idx=np.arange(4), min_controlled_s=0)
    assert list(gt["label"]) == ["H", "D", "H", "A"]

# --- kontrollert ballbesittelse (arrangørens regel) -------------------------------------------
def lab(text):
    return np.array(list(text), dtype=object)


def test_kort_bytte_mellom_samme_lag_er_ikke_et_bytte():
    assert "".join(controlled_owner(lab("HHHHAAHHHH"), 3)) == "HHHHHHHHHH"


def test_langt_bytte_beholdes():
    assert "".join(controlled_owner(lab("HHHHAAAHHHH"), 3)) == "HHHHAAAHHHH"


def test_perioder_ved_dodball_og_x_rores_ikke():
    for text in ("DDAHHHH", "HHHADDD", "XXAHHHH", "HHHADHH"):
        assert "".join(controlled_owner(lab(text), 3)) == text


def test_korteste_forst():
    # A (1) forsvinner først; da er H (2) ikke lenger mellom to A-perioder
    assert "".join(controlled_owner(lab("AAAAHHAHHHHH"), 3)) == "AAAAHHHHHHHH"


def test_make_gt_bruker_regelen():
    clean = dict(owner=obj("H", "H", "A", "H", "H"), alive=np.ones(5, dtype=bool),
                 period=np.ones(5, dtype=int))
    gt = make_gt(clean, n_total=5, idx=np.arange(5), min_controlled_s=2 / 25)
    assert list(gt["label"]) == ["H"] * 5
