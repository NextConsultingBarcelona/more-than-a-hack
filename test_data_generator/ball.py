"""Sann ballbane per frame (meter), interpolert mellom nøkkelpunktene fra possession.py.

Utenfor vurderte frames (før kamp, pause, etter kamp) er ballen NaN.
Dette er den SANNE banen. tracker.py lager hullene som måleutstyret ville hatt.
"""

import numpy as np

from .config import FPS


def true_ball(keyframes, n_frames, eval_mask, dead_mask, rng):
    kf = np.array(keyframes, dtype=float)
    # Ved like frames: behold siste nøkkelpunkt
    frames = kf[:, 0]
    keep = np.r_[frames[1:] != frames[:-1], True]
    kf = kf[keep]

    all_frames = np.arange(n_frames)
    x = np.interp(all_frames, kf[:, 0], kf[:, 1])
    y = np.interp(all_frames, kf[:, 0], kf[:, 2])

    # Litt «dribleuro» når ballen er i spill (glatt, ca. 0,3 m)
    alive = eval_mask & ~dead_mask
    noise = rng.normal(0.0, 0.3, size=(n_frames, 2))
    kernel = np.ones(FPS) / FPS
    for k in range(2):
        noise[:, k] = np.convolve(noise[:, k], kernel, mode="same") * np.sqrt(FPS)
    x = x + np.where(alive, noise[:, 0], 0.0)
    y = y + np.where(alive, noise[:, 1], 0.0)

    ball = np.stack([x, y], axis=1)
    ball[~eval_mask] = np.nan
    return ball
