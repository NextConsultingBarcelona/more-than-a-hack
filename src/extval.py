"""Adapter so the possession model runs in external_validation/sweep.py (open Sportec data).

    {"model": "src.extval:predict", "grid": {"c_switch": [4, 6, 8], "r_ctrl": [1.5, 2.0]}}

Every keyword is a field of src.config.Config, so the sweep tunes exactly the numbers the real
pipeline uses. The validation data has no Smart Tagging, so tag evidence is absent there.
"""

import numpy as np

from . import dead_model, deadball, features, possession, team_model
from .config import Config
from .io import AWAY, HOME, Tracking

PITCH = (105.0, 68.0)
_cache = {}


def _tracking(tracking, slots):
    """Tracking arrays and graded blocks from the validation dataframes (cached per dataframe)."""
    key = id(tracking)
    if _cache.get("key") != key:
        names = slots["slot"].tolist()
        team = np.where(slots["team"].to_numpy() == "H", HOME, AWAY).astype(np.int8)
        cols = {axis: tracking[[s + "_" + axis for s in names]].to_numpy(dtype=np.float32)
                for axis in "xys"}
        ball = tracking[["ball_x", "ball_y"]].to_numpy(dtype=np.float32)
        trk = Tracking(cols["x"], cols["y"], cols["s"], ball, team, ("home", "away"))
        # rows outside the halves carry no tracking at all: those are the ungraded X rows
        live = np.isfinite(cols["x"]).any(axis=1) | np.isfinite(ball).any(axis=1)
        edges = np.diff(np.concatenate(([0], live.astype(np.int8), [0])))
        blocks = list(zip(np.flatnonzero(edges == 1).tolist(),
                          (np.flatnonzero(edges == -1) - 1).tolist()))
        _cache.update(key=key, trk=trk, blocks=blocks)
    return _cache["trk"], _cache["blocks"]


def _scores(trk, pitch, fps, model_path):
    """The detector's per-frame log-odds, computed once per dataframe and model file."""
    key = (_cache.get("key"), tuple(pitch), model_path)
    if _cache.get("scores_key") != key:
        model = dead_model.load_model(model_path) if model_path else None
        _cache.update(scores_key=key, scores=dead_model.scores(trk, tuple(pitch), fps, model))
    return _cache["scores"]


def predict(tracking, slots, pitch=PITCH, model_path=None, team_path=None, **params):
    """One label (H/A/D/X) per row of `tracking`. params override src.config.Config fields.

    With the stored models (model_path=None, team_path=None) the Sportec matches include matches
    the models were trained on: use predict_lomo for scores that mean something.
    """
    cfg = Config().with_(**params)
    trk, blocks = _tracking(tracking, slots)
    feats = features.compute(trk, tuple(pitch), cfg)
    dead_llr, info = deadball.compute(feats, [], blocks, cfg)
    stops, used = None, False
    if cfg.dead_model:
        raw = _scores(trk, pitch, cfg.fps, model_path)
        stops, used = dead_model.evidence(dead_llr, info["prekick"], raw, blocks, cfg)
    team = None
    if cfg.team_model and cfg.learned:
        team = {"shape": team_model.shape_stats(trk, tuple(pitch)),
                "model": team_model.load_model(team_path) if team_path else None}
    labels, diag = possession.decode(feats, dead_llr, [], trk.names, blocks, cfg,
                                     dead_evidence=stops if used else None, team=team)
    if team is not None and not diag.get("team_model_used"):
        raise RuntimeError("the second home-vs-away model was not used: {}".format(
            diag.get("team_model_problem")))
    return labels


def _lomo_file(folder, n_rows, what, need):
    """The model file of the validation match with n_rows rows, or None for the stored model."""
    import json
    import os

    index_path = os.path.join(folder, "index.json")
    if not os.path.isfile(index_path):
        if need:
            raise IOError("no leave-one-out {} in {}".format(what, folder))
        return None
    with open(index_path) as handle:
        index = json.load(handle)
    folds = index.get("folds", index)
    if str(n_rows) in folds:
        return os.path.join(folder, folds[str(n_rows)] + ".npz")
    if str(n_rows) in index.get("unseen", {}):
        return None                      # the stored model has not seen this match
    raise KeyError("no leave-one-out {} for a match with {} rows".format(what, n_rows))


def predict_lomo(tracking, slots, pitch=PITCH, **params):
    """predict() with models that have not seen this match (leave-one-match-out).

    The detector files are written by dead_training/train.py --lomo into ext_data/dead_lomo/ and
    the files of the second home-vs-away model by team_training/train.py into
    ext_data/team_lomo/, one per match, each with an index.json that maps the number of rows of
    a validation match to its match id (sweep.py does not pass the match id). The first model
    (learned_model.json) is the stored one: on the five tuning matches it has seen the match.
    """
    import os

    cfg = Config().with_(**params)
    root = os.path.join(dead_model.ROOT, "ext_data")
    dead_file = _lomo_file(os.path.join(root, "dead_lomo"), len(tracking), "detectors: run "
                           "dead_training/train.py --lomo", cfg.dead_model)
    team_file = _lomo_file(os.path.join(root, "team_lomo"), len(tracking), "team models: run "
                           "team_training/train.py", cfg.team_model and cfg.learned)
    return predict(tracking, slots, pitch, dead_file, team_file, **params)
