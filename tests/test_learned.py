"""The learned evidence (src/learned.py) on synthetic data only."""

import numpy as np

import possession_metrics as pm
from src import deadball, features, learned, pipeline, possession
from src import io as mio
from src.config import Config


def test_predict_trees_walks_a_hand_built_tree():
    # x0 <= 1 ? (x1 <= 5 ? 10 : 20) : 30, plus a constant second tree
    tree = dict(f=[0, 1, -1, -1, -1], t=[1.0, 5.0, 0.0, 0.0, 0.0], l=[1, 2, 2, 3, 4],
                r=[4, 3, 2, 3, 4], v=[0.0, 0.0, 10.0, 20.0, 30.0], depth=2)
    stump = dict(f=[-1], t=[0.0], l=[0], r=[0], v=[0.5], depth=0)
    x = np.array([[0.0, 0.0], [1.0, 6.0], [2.0, 0.0]])
    assert list(learned.predict_trees(dict(trees=[tree, stump]), x)) == [10.5, 20.5, 30.5]


def _evidence(trk, sim):
    cfg = Config()
    feats = features.compute(trk, (105.0, 68.0), cfg)
    dead_hand, _ = deadball.compute(feats, [], sim["blocks"], cfg)
    ball_llr, _ = possession.ball_evidence(feats, cfg)
    return feats, ball_llr, dead_hand


def test_swapping_the_teams_mirrors_the_team_features(fake_match):
    sim, paths = fake_match["sim"], fake_match["paths"]
    trk = mio.load_tracking(paths["atd_meta"], paths["atd_raw"], sim["n_frames"])
    swapped = trk._replace(team=(1 - trk.team).astype(trk.team.dtype)) if hasattr(trk, "_replace") \
        else type(trk)(trk.x, trk.y, trk.s, trk.ball, (1 - trk.team).astype(trk.team.dtype), trk.names)
    out = []
    for t in (trk, swapped):
        feats, ball_llr, dead_hand = _evidence(t, sim)
        out.append(learned.team_features(feats, ball_llr, np.zeros(feats.n), dead_hand, sim["blocks"]))
    (anti, sym), (anti_s, sym_s) = out
    graded = np.zeros(sim["n_frames"], dtype=bool)
    for first, last in sim["blocks"]:
        graded[first:last + 1] = True
    assert np.allclose(anti_s[graded], -anti[graded], atol=1e-6)
    assert np.allclose(sym_s[graded], sym[graded], atol=1e-6)


def test_learned_pipeline_on_a_fake_match(fake_match):
    sim = fake_match["sim"]
    cfg = Config().with_(learned=True)
    labels, _ = pipeline.label_match(fake_match["dir"], sim["blocks"], sim["n_frames"], cfg=cfg,
                                     use_cache=False)
    again, _ = pipeline.label_match(fake_match["dir"], sim["blocks"], sim["n_frames"], cfg=cfg,
                                    use_cache=False)
    assert np.array_equal(labels, again)                       # deterministic
    graded = np.zeros(len(labels), dtype=bool)
    for first, last in sim["blocks"]:
        graded[first:last + 1] = True
    assert set(labels[graded]) <= {"H", "A", "D"}
    assert set(labels[~graded]) == {"X"}
    res = pm.evaluate(pm.label_codes(sim["truth"]), pm.label_codes(labels))
    assert res["macro_f1"] > 0.6                               # sensible, not a tuned target


def test_no_live_ball_before_the_second_half_kickoff(tmp_path):
    """Also when the ball is hidden on the spot before the kick (seed 1 of the fake match)."""
    import make_fake_match

    for seed in (1, 7):
        out = str(tmp_path / "m{}".format(seed))
        sim = make_fake_match.simulate(seed=seed, half_s=300.0)
        make_fake_match.write(sim, out)
        labels, _ = pipeline.label_match(out, sim["blocks"], sim["n_frames"], use_cache=False)
        first, kick = sim["blocks"][1][0], sim["kickoffs"][1]
        assert np.all(labels[first:kick] == "D")
        alive = np.flatnonzero(labels[kick:kick + 100] != "D")
        assert len(alive) and int(alive[0]) <= 25             # live within 1 s of the kick
