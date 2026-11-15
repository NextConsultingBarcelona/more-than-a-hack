"""Adapter so the possession model runs in external_validation/sweep.py (open Sportec data).

    {"model": "src.extval:predict", "grid": {"c_switch": [4, 6, 8], "r_ctrl": [1.5, 2.0]}}

Every keyword is a field of src.config.Config, so the sweep tunes exactly the numbers the real
pipeline uses. The validation data has no Smart Tagging, so tag evidence is absent there.
"""

import numpy as np

from . import deadball, features, possession
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


def predict(tracking, slots, pitch=PITCH, **params):
    """One label (H/A/D/X) per row of `tracking`. params override src.config.Config fields."""
    cfg = Config().with_(**params)
    trk, blocks = _tracking(tracking, slots)
    feats = features.compute(trk, tuple(pitch), cfg)
    dead_llr, _ = deadball.compute(feats, [], blocks, cfg)
    labels, _ = possession.decode(feats, dead_llr, [], trk.names, blocks, cfg)
    return labels
