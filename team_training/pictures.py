"""Other pictures of the ball during LIVE play, for training and stress tests of the
home-vs-away part. Open Sportec data only; never reads the competition match.

The validation data (external_validation/build_dataset.py) removes the ball in blocks of 0.5 to
5 s, more often in duels and near goal, but regardless of how high the ball is, and adds no
position noise. The organiser says the ball is lost in "occlusions, crowded boxes and balls in
the air" (DATA.md), and automatic tracking is not exact. The pictures below keep the same share
of the ball missing in live play and leave stoppage frames exactly as the validation data has
them:

    built    the validation data as built
    air      the ball is not seen while it is more than 1.5 m above the ground (plus 0.2 s on
             each side); blocks as in the validation data fill up to the same share missing
    jitter   as built, plus position noise: 0.25 m on the ball, independent per frame, and
             0.25 m on every player slot, correlated over about a second
    airjit   'air' plus the noise of 'jitter'
    jitter50 as 'jitter' with twice the noise (0.5 m); used only as a test, never in training

The ball's height comes from the packed open matches (dead_training/pack_sportec.py).
"""
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = ["built", "air", "jitter", "airjit", "jitter50"]
NOISE_M = {"jitter": 0.25, "airjit": 0.25, "jitter50": 0.5}
PITCH = (105.0, 68.0)
AIR_M = 1.5


def _blocks(missing, live, weight, rng, target, lo, hi):
    """Add gaps of lo..hi frames, starting on live frames with probability ~ weight, up to target."""
    idx = np.flatnonzero(live)
    p = weight[idx] / weight[idx].sum()
    for _ in range(10000):
        if missing[live].sum() >= target * len(idx):
            break
        starts = rng.choice(idx, size=200, p=p)
        lengths = rng.integers(lo, hi + 1, size=200)
        for s, ln in zip(starts, lengths):
            seg = slice(s, min(s + ln, len(missing)))
            missing[seg] |= live[seg]
            if missing[live].sum() >= target * len(idx):
                break
    return missing


def _ar1(n, cols, sigma, tau, rng):
    """Stationary AR(1) noise [n, cols]: standard deviation sigma, time constant tau frames."""
    a = np.exp(-1.0 / tau)
    e = rng.standard_normal((n, cols)) * sigma * np.sqrt(1 - a * a)
    out = np.empty((n, cols))
    out[0] = rng.standard_normal(cols) * sigma
    for t in range(1, n):
        out[t] = a * out[t - 1] + e[t]
    return out


def true_ball(match, packed_dir, layout):
    """(ball [T, 2] in the validation data's coordinates, height [T] in m) of every frame."""
    z = np.load(Path(packed_dir) / "sportec_{}.npz".format(match), allow_pickle=False)
    n_total, idx = layout({"period": z["period"]})
    b = z["ball_xyz"].astype(np.float32) / 100.0
    length, width = [float(v) for v in z["pitch_m"]]
    ball = np.full((n_total, 2), np.nan, dtype=np.float32)
    height = np.full(n_total, np.nan, dtype=np.float32)
    ball[idx, 0] = b[:, 0] / length + 0.5
    ball[idx, 1] = 0.5 - b[:, 1] / width
    height[idx] = b[:, 2]
    return ball, height


def make(match, picture, tracking, slots, labels, packed_dir, layout):
    """The validation match `tracking` (a DataFrame as build_dataset.py writes it) with another
    picture of the live ball. -> a new DataFrame; 'built' returns the input unchanged."""
    if picture == "built":
        return tracking
    if picture not in NAMES:
        raise ValueError("unknown picture {}".format(picture))
    rng = np.random.default_rng([11, zlib.crc32((match + picture).encode())])
    out = tracking.copy()
    ball = out[["ball_x", "ball_y"]].to_numpy(dtype=np.float32).copy()
    x_cols = [c for c in out.columns if c.endswith("_x") and c != "ball_x"]
    y_cols = [c[:-2] + "_y" for c in x_cols]
    live = np.isin(labels, ["H", "A"])

    if picture in ("air", "airjit"):
        true, height = true_ball(match, packed_dir, layout)
        if len(true) != len(ball):
            raise ValueError("{}: the packed match and the validation data differ in length".format(match))
        target = float(np.isnan(ball[live, 0]).mean())               # the same share missing
        xs = out[x_cols].to_numpy(dtype=np.float32)
        ys = out[y_cols].to_numpy(dtype=np.float32)
        d = np.sqrt(((xs - true[:, None, 0]) * PITCH[0]) ** 2 + ((ys - true[:, None, 1]) * PITCH[1]) ** 2)
        d = np.where(np.isnan(d), np.inf, d)
        team = dict(zip(slots["slot"], slots["team"]))
        is_home = np.array([str(team[c[:-2]]).upper().startswith("H") for c in x_cols])
        duel = (d[:, is_home].min(axis=1) < 2.5) & (d[:, ~is_home].min(axis=1) < 2.5)
        goal = np.minimum(np.hypot(true[:, 0] * PITCH[0], (true[:, 1] - 0.5) * PITCH[1]),
                          np.hypot((1 - true[:, 0]) * PITCH[0], (true[:, 1] - 0.5) * PITCH[1]))
        weight = 1.0 + 2.0 * duel + 1.0 * (np.nan_to_num(goal, nan=np.inf) < 20.0)
        up = np.nan_to_num(height, nan=0.0) > AIR_M
        up = np.convolve(up.astype(float), np.ones(11), mode="same") > 0
        missing = _blocks(up & live, live, weight, rng, target, 12, 125)
        ball[live] = true[live]
        ball[missing & live] = np.nan

    if picture in NOISE_M:
        sigma = NOISE_M[picture]
        ball = ball + (rng.standard_normal(ball.shape) * sigma / np.array(PITCH)).astype(np.float32)
        noise = _ar1(len(out), 2 * len(x_cols), sigma, 25.0, rng).astype(np.float32)
        out[x_cols] = out[x_cols].to_numpy(dtype=np.float32) + noise[:, :len(x_cols)] / PITCH[0]
        out[y_cols] = out[y_cols].to_numpy(dtype=np.float32) + noise[:, len(x_cols):] / PITCH[1]

    out["ball_x"], out["ball_y"] = ball[:, 0], ball[:, 1]
    return out
