"""Unit tests of the state model on hand-built arrays."""

import numpy as np

from src.config import Config
from src.possession import A, D, H, enforce_durations, runs_of, side_of, viterbi

CFG = Config()


def _decode(llr, dead, cfg=CFG):
    half = 0.5 * np.asarray(llr, dtype=float)
    return viterbi(half, -half, np.asarray(dead, dtype=float), cfg.c_switch, cfg.c_die, cfg.c_restart)


def test_viterbi_follows_clear_evidence_and_switches_on_time():
    llr = np.concatenate([np.full(200, 0.5), np.full(200, -0.5)])
    states = _decode(llr, np.full(400, -0.3))
    assert states[0] == H and states[-1] == A
    starts, _, _ = runs_of(states)
    assert list(starts) == [0, 200]          # Viterbi is not late, unlike a causal filter


def test_viterbi_holds_through_silence():
    """Ball missing = no evidence = the last team keeps the ball (loose-ball rule)."""
    llr = np.concatenate([np.full(100, 0.5), np.zeros(300), np.full(100, 0.5)])
    assert np.all(_decode(llr, np.zeros(500)) == H)


def test_single_touch_is_not_a_possession():
    llr = np.full(500, 0.4)
    llr[240:250] = -0.5                      # 0.4 s of the other team on the ball
    assert np.all(_decode(llr, np.full(500, -0.2)) == H)


def test_sustained_control_is_a_possession():
    llr = np.full(600, 0.4)
    llr[200:350] = -0.5                      # 6 s
    states = _decode(llr, np.full(600, -0.2))
    assert list(runs_of(states)[2]) == [H, A, H]


def test_stoppage_splits_a_possession():
    dead = np.full(900, -0.3)
    dead[300:600] = 0.5
    states = _decode(np.full(900, 0.3), dead)
    starts, ends, vals = runs_of(states)
    assert list(vals) == [H, D, H]
    assert abs(starts[1] - 300) <= 2 and abs(ends[1] - 600) <= 2


def test_enforce_durations():
    s = np.array([H] * 100 + [A] * 10 + [H] * 100 + [D] * 20 + [H] * 100 + [D] * 200 + [A] * 3 + [H] * 50,
                 dtype=np.int8)
    out = enforce_durations(s, CFG)
    starts, ends, vals = runs_of(out)
    assert list(vals) == [H, D, H]           # touch absorbed, blip stoppage removed, 3-frame spell merged
    assert starts[1] == 330 and ends[1] == 530


def test_side_of_handles_spelling_and_na():
    names = ("FC Rovers", "Athletic Town")
    assert side_of("Rovers", names) == 1
    assert side_of("atletic town", names) == -1
    assert side_of("N/A", names) == 0
    assert side_of(None, names) == 0
