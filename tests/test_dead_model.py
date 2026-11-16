"""The learned dead-ball detector (src/dead_model.py), on synthetic data only."""

import os

import numpy as np

import possession_metrics as pm
from src import dead_model as dm
from src import io as mio
from src import pipeline
from src.config import Config


def _toy_model():
    """Two hand-built trees on two features.

    tree 0: f0 <= 1.0 ? -2 : (f1 <= 5.0 ? 3 : 4)   missing f0 goes right, missing f1 goes left
    tree 1: f1 <= 0.0 ? 10 : 20                     missing f1 goes right
    """
    feature = np.array([0, -1, 1, -1, -1, 1, -1, -1], dtype=np.int16)
    number = np.array([1.0, -2.0, 5.0, 3.0, 4.0, 0.0, 10.0, 20.0])
    left = np.array([1, 0, 3, 0, 0, 6, 0, 0], dtype=np.int32)
    right = np.array([2, 0, 4, 0, 0, 7, 0, 0], dtype=np.int32)
    nan_left = np.array([False, False, True, False, False, False, False, False])
    return {"feature": feature, "number": number, "left": left, "right": right,
            "nan_left": nan_left, "roots": np.array([0, 5], dtype=np.int32),
            "baseline": np.float64(0.5)}


def test_tree_walk_including_missing_values():
    nan = np.nan
    X = np.array([[0.0, -1.0], [1.0, 1.0], [2.0, 5.0], [2.0, 6.0], [nan, nan], [nan, 7.0], [3.0, nan]],
                 dtype=np.float32)
    want = 0.5 + np.array([-2 + 10, -2 + 20, 3 + 20, 4 + 20, 3 + 20, 4 + 20, 3 + 20.0])
    assert np.array_equal(dm.raw_scores(X, _toy_model()), want)


def test_two_state_viterbi_ignores_flicker_and_keeps_real_stoppages():
    score = np.full(1000, -3.0)
    score[300:500] = 3.0            # a real stoppage
    score[700:703] = 7.0            # three noisy frames: 21 < the 24 it costs to go in and out
    score[380:384] = -5.0           # a dip inside the stoppage: cheaper to sit through
    path = dm.two_state_viterbi(score, cost=12.0)
    assert dm._runs(path) == [(300, 500)]
    assert dm.two_state_viterbi(np.zeros(0), 5.0).size == 0


def test_stoppages_respect_blocks_and_minimum_durations():
    cfg = Config()
    raw = np.full(3000, -6.0)
    raw[1000:1400] = 6.0            # 16 s stoppage in the first block
    raw[1150:1160] = -6.0           # ten live-looking frames inside it: closed
    raw[1700:1720] = 8.0            # 0.8 s: too short to be a stoppage
    raw[2100:2600] = 6.0            # runs past the end of the second block
    blocks = [(500, 1900), (2000, 2499)]
    mask = dm.stoppages(raw, blocks, cfg)
    assert dm._runs(mask) == [(1000, 1400), (2100, 2500)]
    assert not mask[:500].any() and not mask[2500:].any()


def test_features_ignore_slot_order_direction_of_play_and_team_naming(fake_match):
    sim, paths = fake_match["sim"], fake_match["paths"]
    trk = mio.load_tracking(paths["atd_meta"], paths["atd_raw"], sim["n_frames"])
    lo = sim["blocks"][0][0]
    cut = mio.Tracking(trk.x[lo:lo + 1500], trk.y[lo:lo + 1500], trk.s[lo:lo + 1500],
                       trk.ball[lo:lo + 1500], trk.team, trk.names)
    X, names = dm.features(cut, (105.3, 68.0))
    assert X.shape[1] == len(names) > 200 and X.dtype == np.float32

    perm = np.random.default_rng(0).permutation(len(cut.team))
    shuffled = mio.Tracking(cut.x[:, perm], cut.y[:, perm], cut.s[:, perm], cut.ball,
                            cut.team[perm], cut.names)
    Y, _ = dm.features(shuffled, (105.3, 68.0))
    assert np.array_equal(np.isnan(X), np.isnan(Y)) and np.nanmax(np.abs(X - Y)) < 1e-3

    mirrored = mio.Tracking(1 - cut.x, 1 - cut.y, cut.s, 1 - cut.ball, 1 - cut.team, cut.names)
    Z, _ = dm.features(mirrored, (105.3, 68.0))
    assert np.array_equal(np.isnan(X), np.isnan(Z)) and np.nanmax(np.abs(X - Z)) < 1e-3


def test_model_file_matches_the_feature_code_and_passes_the_data_guard(fake_match):
    assert os.path.getsize(dm.MODEL_PATH) < 1_000_000          # tools/check_no_data.py limit
    model = dm.load_model()                                    # plain arrays, no pickle
    sim, paths = fake_match["sim"], fake_match["paths"]
    trk = mio.load_tracking(paths["atd_meta"], paths["atd_raw"], sim["n_frames"])
    cut = mio.Tracking(trk.x[:400], trk.y[:400], trk.s[:400], trk.ball[:400], trk.team, trk.names)
    _, names = dm.features(cut, (105.3, 68.0))
    assert names == [str(s) for s in model["names"]]
    assert int(model["feature"].max()) < len(names)


def test_evidence_follows_the_detector_and_keeps_pre_kick_off_dead():
    cfg = Config()
    n = 6000
    blocks = [(0, n - 1)]
    raw = np.full(n, -6.0)
    for start in range(600, n - 400, 700):
        raw[start:start + 250] = 6.0                            # a 10 s stoppage every 28 s
    prekick = np.zeros(n, dtype=bool)
    prekick[:100] = True
    own = np.linspace(-1, 1, n)
    out, used = dm.evidence(own, prekick, raw, blocks, cfg)
    assert used
    assert np.all(out[:100] == cfg.dm_weight)                   # before the kick-off: dead
    assert np.all(out[600:850] == cfg.dm_weight) and np.all(out[100:600] == -cfg.dm_weight)


def test_implausible_detector_output_is_not_used():
    cfg = Config()
    n = 6000
    blocks = [(0, n - 1)]
    own = np.linspace(-1, 1, n)
    prekick = np.zeros(n, dtype=bool)
    for raw in (np.full(n, 6.0), np.full(n, -6.0)):             # all dead / never dead
        out, used = dm.evidence(own, prekick, raw, blocks, cfg)
        assert not used and out is own


def test_detector_is_used_on_the_fake_match_and_agrees_with_its_truth(fake_match):
    sim = fake_match["sim"]
    labels, diag = pipeline.label_match(fake_match["dir"], sim["blocks"], sim["n_frames"],
                                        use_cache=False)
    assert diag["dead_model_used"]
    truth = pm.label_codes(sim["truth"])
    ours = pm.label_codes(labels)
    graded = np.zeros(len(truth), dtype=bool)
    for first, last in sim["blocks"]:
        graded[first:last + 1] = True
    dead_truth = sim["truth"] == "D"
    assert np.mean((labels[graded] == "D") == dead_truth[graded]) > 0.9
    assert pm.evaluate(truth, ours)["spell_f1_mean"] > 0.6


def test_the_state_models_own_evidence_is_still_available(fake_match):
    sim = fake_match["sim"]
    labels, diag = pipeline.label_match(fake_match["dir"], sim["blocks"], sim["n_frames"],
                                        cfg=Config(dead_model=False), use_cache=False)
    assert not diag["dead_model_used"]
    assert set(labels[sim["blocks"][0][0]:sim["blocks"][0][1] + 1]) <= {"H", "A", "D"}
    assert not Config().hand_weighted().dead_model            # --hand-weighted: the earlier model


def test_pipeline_falls_back_to_its_own_evidence_if_the_detector_fails(fake_match, monkeypatch):
    sim = fake_match["sim"]

    def boom(*args, **kwargs):
        raise RuntimeError("no detector today")

    monkeypatch.setattr(dm, "scores", boom)
    labels, diag = pipeline.label_match(fake_match["dir"], sim["blocks"], sim["n_frames"],
                                        use_cache=False)
    assert not diag["dead_model_used"] and "RuntimeError" in diag["dead_model_problem"]
    own, _ = pipeline.label_match(fake_match["dir"], sim["blocks"], sim["n_frames"],
                                  cfg=Config(dead_model=False), use_cache=False)
    assert np.array_equal(labels, own)
