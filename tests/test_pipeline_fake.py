"""End-to-end tests on a synthetic match written in the real file formats."""

import filecmp
import os

import numpy as np

import possession_metrics as pm
from src import io as mio
from src import pipeline
from src.possession import runs_of


def _score(sim, labels):
    return pm.evaluate(pm.label_codes(sim["truth"]), pm.label_codes(labels))


def test_parser_roundtrip(fake_match):
    sim, paths = fake_match["sim"], fake_match["paths"]
    trk = mio.load_tracking(paths["atd_meta"], paths["atd_raw"], sim["n_frames"])
    assert trk.x.shape == sim["x"].shape
    assert np.array_equal(trk.team, sim["slot_team"])
    assert np.array_equal(np.isnan(trk.x), np.isnan(sim["x"]))
    assert np.nanmax(np.abs(trk.x - sim["x"])) < 1e-5
    assert np.array_equal(np.isnan(trk.ball[:, 0]), np.isnan(sim["ball"][:, 0]))


def test_parser_slow_path_and_cache(tmp_path):
    meta = tmp_path / "m.xml"
    meta.write_text("<M><Team id='a'><Name>Home</Name></Team><Team id='b'><Name>Away</Name></Team>"
                    "<Player teamId='a'/><Player teamId='b'/><Player teamId='b'/></M>")
    raw = tmp_path / "r.txt"
    raw.write_text("1:0.1,0.2,1.5;NaN;0.3,0.4,2.0:0.5,0.5,0\n"       # bare NaN -> per-group parse
                   "2:NaN,NaN,NaN;0.6,0.7,0.1;NaN,NaN,NaN:NaN,NaN,NaN\n"
                   "garbage line\n")
    for _ in range(2):                                                # second pass reads the cache
        trk = mio.load_tracking(str(meta), str(raw), cache_dir=str(tmp_path / "cache"))
        assert trk.x.shape == (2, 3) and list(trk.team) == [0, 1, 1]
        assert np.isnan(trk.x[0, 1]) and abs(trk.x[0, 2] - 0.3) < 1e-6 and abs(trk.s[0, 0] - 1.5) < 1e-6
        assert np.isnan(trk.ball[1]).all() and abs(trk.ball[0, 0] - 0.5) < 1e-6
        assert abs(trk.y[1, 1] - 0.7) < 1e-6


def test_output_shape_and_labels(fake_match, fake_labels):
    sim = fake_match["sim"]
    labels, _ = fake_labels
    assert len(labels) == sim["n_frames"]
    graded = np.zeros(len(labels), dtype=bool)
    for first, last in sim["blocks"]:
        graded[first:last + 1] = True
    assert set(labels[graded]) <= {"H", "A", "D"}      # never abstain where it is scored
    assert set(labels[~graded]) == {"X"}


def test_beats_the_starter_baseline(fake_match, fake_labels):
    import baseline

    sim, paths = fake_match["sim"], fake_match["paths"]
    ours = _score(sim, fake_labels[0])
    codes = baseline.nearest_player_labels(paths["atd_raw"], baseline.team_of_each_track(paths["atd_meta"]),
                                           sim["n_frames"], (105.3, 68.0))
    floor = _score(sim, baseline.smooth(codes, 50))
    assert ours["spell_f1_mean"] > 0.6
    assert ours["spell_f1_mean"] > floor["spell_f1_mean"] + 0.1
    assert ours["macro_f1"] > floor["macro_f1"]
    # the possession count is in the right range: no flicker, no collapse
    assert 0.6 * ours["n_gt_spells"] <= ours["n_pred_spells"] <= 1.4 * ours["n_gt_spells"]


def test_kickoffs_and_pre_kickoff_dead(fake_match, fake_labels):
    sim = fake_match["sim"]
    labels, _ = fake_labels
    second_block_first = sim["blocks"][1][0]
    k2 = sim["kickoffs"][1]
    assert np.all(labels[second_block_first:k2 - 25] == "D")
    for k in sim["kickoffs"]:
        alive = np.flatnonzero(labels[k:k + 100] != "D")
        assert len(alive) and int(alive[0]) <= 25                # live within 1 s of the kick-off


def test_restarts_are_found(fake_match, fake_labels):
    sim = fake_match["sim"]
    labels, _ = fake_labels
    starts, _, vals = runs_of(labels)
    pred = starts[1:][(vals[:-1] == "D") & (vals[1:] != "D")]
    close = [np.min(np.abs(pred - r)) <= 25 for r in sim["restarts"]]
    assert np.mean(close) >= 0.7


def test_tag_polarity_is_learned_not_assumed(fake_labels):
    polarity = fake_labels[1]["tag_polarity"]
    assert polarity["progression"] > 0.2                  # Team = the side with the ball
    assert polarity["defending in middle third"] < -0.2   # Team = the side without it


def test_deterministic_bytes(fake_match, tmp_path):
    meta = fake_match["paths"]["meta"]
    outs = []
    for name in ("a.csv", "b.csv"):
        out = str(tmp_path / name)
        pipeline.main(["--data-dir", fake_match["dir"], "--meta", meta, "--out", out, "--no-cache"])
        outs.append(out)
    assert filecmp.cmp(outs[0], outs[1], shallow=False)
    frame = pm.read_label_csv(outs[0])
    assert pm.validate_submission(frame, fake_match["sim"]["n_frames"]) is None
    assert not os.path.exists(os.path.join(fake_match["dir"], "cache"))
