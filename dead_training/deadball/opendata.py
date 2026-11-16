"""Loader for the open matches the dead-ball detector is trained on. Never used on the competition match.

Sportec / IDSSE (Bassek et al. 2025, CC-BY 4.0): Bundesliga matches, TRACAB tracking with a
per-frame ball status (dead / alive) and ball possession. Packed by dead_training/pack_sportec.py.
"""
from pathlib import Path

import numpy as np

from .frames import Frames

MISSING = -32768


def load_sportec(path):
    z = np.load(path, allow_pickle=False)

    def xy(a):
        a = a.astype(np.float32)
        a[a == MISSING] = np.nan
        return a / 100.0

    h, a, b = xy(z["home_xy"]), xy(z["away_xy"]), xy(z["ball_xyz"])
    keep_h = ~np.isnan(h[..., 0]).all(0)
    keep_a = ~np.isnan(a[..., 0]).all(0)
    h, a = h[:, keep_h], a[:, keep_a]
    dead = z["ball_dead"] == 1
    bx, by = b[:, 0].copy(), b[:, 1].copy()
    # While the ball is dead TRACAB freezes it at a placeholder position. That is not an
    # observation, so it is removed here; degrade.py decides what a camera would have seen.
    bx[dead] = np.nan
    by[dead] = np.nan
    pitch = tuple(float(v) for v in z["pitch_m"])
    return Frames(
        px=np.concatenate([h[..., 0], a[..., 0]], 1), py=np.concatenate([h[..., 1], a[..., 1]], 1),
        team=np.r_[np.zeros(h.shape[1], np.int8), np.ones(a.shape[1], np.int8)],
        bx=bx, by=by, pitch=pitch, fps=int(round(float(z["fps"]))), dead=dead,
        owner=z["owner"].astype(np.int8), period=z["period"].astype(np.int8),
        name="sportec_" + str(z["match_id"]),
        meta={"bz": b[:, 2], "source": "sportec"})


def load_all(folder):
    """Every packed match in the folder, in a fixed (sorted) order."""
    return [load_sportec(p) for p in sorted(Path(folder).glob("sportec_*.npz"))]
