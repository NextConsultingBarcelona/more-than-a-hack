"""Tester for falske data-generatoren (v1).

Kjør:  python -m pytest tests -q            (alt, ca. 1 min)
       python -m pytest tests -q -m "not slow"  (bare kort modus, noen sekunder)
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_data_generator import FULL, SHORT, generate
from test_data_generator.config import FPS, N_SLOTS, SLOT_NAMES
from test_data_generator.writer import write_match

MATCHES = [
    pytest.param("short_match", id="short"),
    pytest.param("full_match", id="full", marks=pytest.mark.slow),
]


@pytest.fixture(params=MATCHES)
def match(request):
    return request.getfixturevalue(request.param)


def runs(labels):
    """Sammenhengende like etiketter: liste av (etikett, start, slutt_eksklusiv)."""
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(labels)]
    return [(labels[s], s, e) for s, e in zip(starts, ends)]


def eval_mask(tl):
    m = np.zeros(tl.n_frames, bool)
    m[tl.h1_start:tl.h1_end + 1] = True
    m[tl.h2_start:tl.h2_end + 1] = True
    return m


# ---------------------------------------------------------------- 1. Format

def test_full_timeline_matches_public_boundaries():
    assert FULL.n_frames == 199_319
    assert (FULL.h1_start, FULL.h1_end) == (11_185, 90_651)
    assert (FULL.h2_start, FULL.h2_kickoff, FULL.h2_end) == (110_808, 113_742, 190_197)
    assert FULL.h2_kickoff - FULL.h2_start == 2_934


def test_tracking_shape_and_columns(match):
    t = match.tracking
    assert len(t) == match.timeline.n_frames
    expected = ["frame_id", "ball_x", "ball_y"] + [f"{s}_{a}" for s in SLOT_NAMES for a in ("x", "y", "s")]
    assert list(t.columns) == expected
    assert len(t.columns) == 3 + 3 * N_SLOTS
    assert t["frame_id"].tolist() == list(range(1, len(t) + 1))


def test_slot_teams(match):
    st = match.slot_teams
    assert list(st.columns) == ["slot", "team"]
    assert st["slot"].tolist() == list(SLOT_NAMES)
    assert st["team"].value_counts().to_dict() == {"H": 23, "A": 23}


def test_truth_format(match):
    tr = match.truth
    assert list(tr.columns) == ["frame", "label"]
    assert len(tr) == match.timeline.n_frames
    assert tr["frame"].tolist() == list(range(len(tr)))
    assert set(tr["label"]) <= {"H", "A", "D", "X"}


# ---------------------------------------------------------------- 2. Kampstruktur i fasiten

def test_x_exactly_outside_evaluated_frames(match):
    lab = match.truth["label"].to_numpy()
    em = eval_mask(match.timeline)
    assert (lab[~em] == "X").all()
    assert not (lab[em] == "X").any()


def test_second_half_dead_before_kickoff(match):
    tl = match.timeline
    lab = match.truth["label"].to_numpy()
    assert (lab[tl.h2_start:tl.h2_kickoff] == "D").all()
    assert lab[tl.h2_kickoff] in ("H", "A")


def test_kickoffs_alternate(match):
    tl = match.timeline
    lab = match.truth["label"].to_numpy()
    assert lab[tl.h1_start] in ("H", "A")
    assert lab[tl.h2_kickoff] == ("A" if lab[tl.h1_start] == "H" else "H")


# ---------------------------------------------------------------- 3. Possession-logikk

def test_possession_durations(match):
    tl = match.timeline
    lab = match.truth["label"].to_numpy()
    half_ends = {tl.h1_end + 1, tl.h2_end + 1}
    poss = [(l, s, e) for l, s, e in runs(lab) if l in ("H", "A")]
    assert len(poss) > 0
    for l, s, e in poss:
        dur = (e - s) / FPS
        assert dur <= 30.0 + 1e-9, (l, s, e)
        if e not in half_ends:
            assert dur >= 5.0 - 1e-9, (l, s, e)


def test_possessions_match_events(match):
    """Hver H/A-sekvens i fasiten = nøyaktig én possession i hendelseslista."""
    lab = match.truth["label"].to_numpy()
    poss_runs = [(s, e) for l, s, e in runs(lab) if l in ("H", "A")]
    ev = match.events[match.events["kind"] == "possession"]
    assert poss_runs == list(zip(ev["start"], ev["end"]))


def test_reasonable_number_of_possessions(match):
    ev = match.events
    n = int((ev["kind"] == "possession").sum())
    played_min = match.truth["label"].isin(["H", "A", "D"]).sum() / FPS / 60
    per_90 = n / played_min * 90
    assert 200 <= per_90 <= 500, per_90


def test_loose_ball_keeps_label(match):
    """Løs ball: laget som sist hadde ballen beholder etiketten til det andre laget har kontroll."""
    lab = match.truth["label"].to_numpy()
    loose = match.events[match.events["kind"].str.startswith("loose")]
    assert len(loose) > 0
    for row in loose.itertuples():
        assert (lab[row.start:row.end] == row.label).all()
        if row.kind == "loose_kept":
            assert lab[row.end] == row.label  # samme lag vinner den tilbake
        else:
            assert lab[row.end] == ("A" if row.label == "H" else "H")  # bytte først ved kontroll


def test_restart_owner(match):
    """Etter en død ball starter possession hos restart-eieren."""
    tl = match.timeline
    lab = match.truth["label"].to_numpy()
    half_ends = {tl.h1_end + 1, tl.h2_end + 1}
    ev = match.events.reset_index(drop=True)
    dead = ev[ev["kind"] == "dead"]
    assert len(dead) > 0
    for i, row in dead.iterrows():
        assert (lab[row.start:row.end] == "D").all()
        if row.end not in half_ends:
            assert lab[row.end] == row.owner, row
        if row.detail in ("corner", "goal_kick"):
            prev = ev[(ev["kind"] == "possession") & (ev["end"] == row.start)].iloc[0]
            if row.detail == "corner":
                assert row.owner == prev.label  # forsvarer siste berøring
            else:
                assert row.owner != prev.label  # angriper siste berøring


def test_restart_types_present(full_match):
    kinds = set(full_match.events.loc[full_match.events["kind"] == "dead", "detail"])
    assert {"throw_in", "free_kick", "goal_kick", "corner", "before_kickoff"} <= kinds


# ---------------------------------------------------------------- 4. Ball

def test_ball_missing_rate(match):
    em = match.debug["eval_mask"]
    missing = match.tracking["ball_x"].isna().to_numpy()
    rate = missing[em].mean()
    assert 0.35 <= rate <= 0.45, rate
    assert missing[~em].all()  # ingen ball utenfor spilletid
    assert (match.tracking["ball_x"].isna() == match.tracking["ball_y"].isna()).all()


@pytest.mark.slow
def test_ball_missing_more_in_duels_and_near_goal(full_match):
    d = full_match.debug
    em, miss = d["eval_mask"], full_match.tracking["ball_x"].isna().to_numpy()
    calm = em & ~d["duel"] & ~d["near_goal"]
    assert miss[d["duel"]].mean() > miss[calm].mean()
    assert miss[d["near_goal"]].mean() > miss[calm].mean()


def test_ball_coordinates_in_range(match):
    b = match.tracking[["ball_x", "ball_y"]].dropna()
    assert b.min().min() >= -0.02 and b.max().max() <= 1.02


# ---------------------------------------------------------------- 5. Slots / spillere

def test_active_slots(match):
    em = match.debug["eval_mask"]
    x_cols = [f"{s}_x" for s in SLOT_NAMES]
    active = match.tracking[x_cols].notna().sum(axis=1).to_numpy()[em]
    assert 20 <= np.median(active) <= 23
    assert active.min() >= 16
    assert active.max() <= 26


def test_slot_values_in_range(match):
    t = match.tracking
    xy = t[[f"{s}_{a}" for s in SLOT_NAMES for a in ("x", "y")]].to_numpy()
    sp = t[[f"{s}_s" for s in SLOT_NAMES]].to_numpy()
    # Litt utenfor 0-1 er lov: innkast-taker står utenfor sidelinja, som i ekte data
    assert np.nanmin(xy) >= -0.03 and np.nanmax(xy) <= 1.03
    assert np.nanmin(sp) >= 0.0 and np.nanmax(sp) < 12.0
    # x, y og s mangler alltid samtidig
    for s in SLOT_NAMES:
        nx = t[f"{s}_x"].isna()
        assert (nx == t[f"{s}_y"].isna()).all() and (nx == t[f"{s}_s"].isna()).all()


def test_slots_appear_and_change(full_match):
    """Flere enn 11 slots per lag blir brukt i løpet av kampen (identitet bytter slot)."""
    t, st = full_match.tracking, full_match.slot_teams
    used = {s for s in SLOT_NAMES if t[f"{s}_x"].notna().any()}
    for team in ("H", "A"):
        team_used = used & set(st.loc[st["team"] == team, "slot"])
        assert len(team_used) > 11


# ---------------------------------------------------------------- 6. Determinisme

def test_same_seed_same_data():
    a, b = generate(seed=123, mode="short"), generate(seed=123, mode="short")
    pd.testing.assert_frame_equal(a.tracking, b.tracking)
    pd.testing.assert_frame_equal(a.truth, b.truth)
    pd.testing.assert_frame_equal(a.events, b.events)


def test_different_seed_different_data():
    a, b = generate(seed=1, mode="short"), generate(seed=2, mode="short")
    assert not a.truth["label"].equals(b.truth["label"])


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_same_seed_byte_identical_files(tmp_path):
    p1 = write_match(generate(seed=5, mode="short"), tmp_path / "run1")
    p2 = write_match(generate(seed=5, mode="short"), tmp_path / "run2")
    for key in p1:
        assert _sha(p1[key]) == _sha(p2[key]), key


def test_writer_refuses_data_folder(tmp_path, short_match):
    with pytest.raises(ValueError):
        write_match(short_match, tmp_path / "data" / "x")


# ---------------------------------------------------------------- 7. Arrangørens validator

def _validator(repo_root):
    root = Path(os.environ.get("MTAH_REPO_DIR", repo_root))
    return root / "evaluation" / "validate_submission.py"


@pytest.mark.parametrize("mode", [pytest.param("short", id="short"),
                                  pytest.param("full", id="full", marks=pytest.mark.slow)])
def test_truth_passes_organizer_validator(mode, repo_root, tmp_path, request):
    script = _validator(repo_root)
    if not script.is_file():
        pytest.skip(f"Fant ikke {script}. Kjør testene fra lagets repo (kopi av arrangørens).")
    match = request.getfixturevalue(f"{mode}_match")
    csv = tmp_path / "fake_truth.csv"
    match.truth.to_csv(csv, index=False, lineterminator="\n")
    res = subprocess.run([sys.executable, str(script), str(csv), "--rows", str(len(match.truth))],
                         cwd=script.parent.parent, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert res.stdout.startswith("OK"), res.stdout
