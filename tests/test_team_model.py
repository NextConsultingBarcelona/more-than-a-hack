"""The second home-vs-away model (src/team_model.py) on synthetic data only."""
import os

import numpy as np
import pytest

from src import deadball, features, pipeline, possession, team_model
from src import io as mio
from src.config import Config

PITCH = (105.0, 68.0)


def _load(fake_match):
    sim, paths = fake_match["sim"], fake_match["paths"]
    return mio.load_tracking(paths["atd_meta"], paths["atd_raw"], sim["n_frames"]), sim


def _features(trk, sim):
    cfg = Config()
    feats = features.compute(trk, PITCH, cfg)
    own, _ = deadball.compute(feats, [], sim["blocks"], cfg)
    ball_llr, _ = possession.ball_evidence(feats, cfg)
    hint = np.zeros(feats.n)                      # the run-time hint is not part of this test
    shape = team_model.shape_stats(trk, PITCH)
    return team_model.features(feats, ball_llr, hint, own, sim["blocks"], shape, cfg)


def _graded(sim):
    graded = np.zeros(sim["n_frames"], dtype=bool)
    for first, last in sim["blocks"]:
        graded[first + 150:last + 1 - 150] = True          # away from the edges of a half
    return graded


def test_helpers_on_small_series():
    e = np.array([0.0, 1.0, 0.0, 0.0, -1.0, 0.0])
    sign, since = team_model._last(e)
    assert list(sign) == [0.0, 1.0, 1.0, 1.0, -1.0, -1.0]
    assert list(since) == [team_model.CAP, 0.0, 1.0, 2.0, 0.0, 1.0]
    sign, until = team_model._next(e)
    assert list(sign) == [1.0, 1.0, -1.0, -1.0, -1.0, 0.0]
    assert list(until) == [1.0, 0.0, 2.0, 1.0, 0.0, team_model.CAP]
    a = np.arange(6, dtype=float)
    assert list(team_model._box_sum(a, -1, 1)) == [1.0, 3.0, 6.0, 9.0, 12.0, 9.0]
    assert list(team_model._shift(a, 2, -1.0)) == [2.0, 3.0, 4.0, 5.0, -1.0, -1.0]
    assert list(team_model._shift(a, -2, -1.0)) == [-1.0, -1.0, 0.0, 1.0, 2.0, 3.0]


def test_swapping_the_teams_mirrors_the_features(fake_match):
    trk, sim = _load(fake_match)
    swapped = type(trk)(trk.x, trk.y, trk.s, trk.ball, (1 - trk.team).astype(trk.team.dtype), trk.names)
    anti, sym = _features(trk, sim)
    anti_s, sym_s = _features(swapped, sim)
    g = _graded(sim)
    assert np.allclose(anti_s[g], -anti[g], atol=1e-6)
    assert np.allclose(sym_s[g], sym[g], atol=1e-6)


def _own_features(trk, sim):
    """Only what this module adds (the first model's features are tested in test_learned.py)."""
    cfg = Config()
    feats = features.compute(trk, PITCH, cfg)
    a1, s1 = team_model.local_features(feats, cfg)
    a2, s2 = team_model.collective_features(feats, team_model.shape_stats(trk, PITCH), sim["blocks"])
    return np.column_stack([a1, a2]), np.column_stack([s1, s2])


def test_features_do_not_depend_on_the_order_of_the_slots(fake_match):
    trk, sim = _load(fake_match)
    order = np.random.default_rng(3).permutation(trk.x.shape[1])
    shuffled = type(trk)(trk.x[:, order], trk.y[:, order], trk.s[:, order], trk.ball,
                         trk.team[order], trk.names)
    anti, sym = _own_features(trk, sim)
    anti_p, sym_p = _own_features(shuffled, sim)
    g = _graded(sim)
    assert np.allclose(anti_p[g], anti[g], atol=1e-6)
    assert np.allclose(sym_p[g], sym[g], atol=1e-6)


def test_model_file_matches_the_feature_code_and_passes_the_data_guard(fake_match):
    trk, sim = _load(fake_match)
    anti, sym = _features(trk, sim)
    model = team_model.load_model()
    assert int(model["n_features"]) == anti.shape[1] + sym.shape[1]
    assert int(model["feature"].max()) < anti.shape[1] + sym.shape[1]
    assert os.path.getsize(team_model.MODEL_PATH) < 1_000_000       # tools/check_no_data.py limit
    z = team_model.logit_from(anti, sym, model)
    assert np.allclose(team_model.logit_from(-anti, sym, model), -z)   # exactly antisymmetric
    with pytest.raises(ValueError):
        team_model.logit_from(anti[:, :-1], sym, model)


def test_second_model_is_used_on_the_fake_match(fake_match):
    sim = fake_match["sim"]
    labels, diag = pipeline.label_match(fake_match["dir"], sim["blocks"], sim["n_frames"], use_cache=False)
    again, _ = pipeline.label_match(fake_match["dir"], sim["blocks"], sim["n_frames"], use_cache=False)
    assert diag["team_model_used"] and diag["team_model_problem"] is None
    assert np.array_equal(labels, again)                               # deterministic
    graded = np.zeros(len(labels), dtype=bool)
    for first, last in sim["blocks"]:
        graded[first:last + 1] = True
    assert set(labels[graded]) <= {"H", "A", "D"} and set(labels[~graded]) == {"X"}


def test_the_first_model_alone_is_still_available(fake_match):
    sim = fake_match["sim"]
    run = lambda cfg: pipeline.label_match(fake_match["dir"], sim["blocks"], sim["n_frames"], cfg=cfg,
                                           use_cache=False)
    labels, diag = run(Config().tagged())
    assert not diag["team_model_used"] and not diag["dead_model_used"]
    assert np.array_equal(labels, run(Config(dead_model=False, team_model=False))[0])
    assert not Config().hand_weighted().team_model               # --hand-weighted: the earlier model
    _, diag = run(Config().hand_weighted())
    assert not diag["team_model_used"]


def test_pipeline_keeps_the_first_model_if_the_second_fails(fake_match, monkeypatch):
    sim = fake_match["sim"]

    def broken(*args, **kwargs):
        raise RuntimeError("no model today")

    first, _ = pipeline.label_match(fake_match["dir"], sim["blocks"], sim["n_frames"],
                                    cfg=Config(team_model=False), use_cache=False)
    monkeypatch.setattr(team_model, "logit", broken)
    labels, diag = pipeline.label_match(fake_match["dir"], sim["blocks"], sim["n_frames"], use_cache=False)
    assert not diag["team_model_used"] and "RuntimeError" in diag["team_model_problem"]
    assert np.array_equal(labels, first)
