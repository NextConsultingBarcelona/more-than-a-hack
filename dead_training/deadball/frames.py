"""Common in-memory tracking format used by every step of the dead-ball module.

Positions are metres with the origin at the centre spot, x along the pitch length. Missing
observations are NaN. Players are *slots*: a column is not a person, only a team membership,
which is all automatic tracking offers.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

FPS = 25


@dataclass
class Frames:
    px: np.ndarray                 # (n, S) float32 metres
    py: np.ndarray                 # (n, S)
    team: np.ndarray               # (S,) int8, 0 = home, 1 = away
    bx: np.ndarray                 # (n,) float32, NaN where the ball is not observed
    by: np.ndarray
    pitch: tuple = (105.0, 68.0)   # length, width in metres
    fps: int = FPS
    # labels, only for open matches used in training and validation
    dead: Optional[np.ndarray] = None      # (n,) bool
    owner: Optional[np.ndarray] = None     # (n,) int8: 1 home, 2 away, 0 unknown
    period: Optional[np.ndarray] = None    # (n,) int8, 0 = not in play
    name: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.bx)

    def slice(self, lo, hi):
        cut = lambda a: None if a is None else a[lo:hi]
        return Frames(self.px[lo:hi], self.py[lo:hi], self.team, self.bx[lo:hi], self.by[lo:hi],
                      self.pitch, self.fps, cut(self.dead), cut(self.owner), cut(self.period),
                      self.name, dict(self.meta))


def runs(mask):
    """[(start, end)) for every maximal run of True in a boolean array."""
    m = np.asarray(mask, bool)
    if m.size == 0:
        return np.zeros((0, 2), int)
    d = np.diff(np.r_[False, m, False].astype(np.int8))
    return np.c_[np.flatnonzero(d == 1), np.flatnonzero(d == -1)]
