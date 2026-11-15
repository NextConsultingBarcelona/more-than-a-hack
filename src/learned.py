"""Learned per-frame evidence: gradient-boosted trees trained on open matches, run in numpy.

Two models, trained by external_validation/train_emissions.py on the open Sportec/IDSSE matches
(never on the provided match): dead-vs-alive, and home-vs-away given that the ball is alive.
Their log-odds replace the hand-weighted evidence as input to the same Viterbi decoder
(Borghesi et al. 2023 for learned per-frame evidence, Kim et al. 2026 for structured decoding).

The trees are stored as plain arrays in learned_model.json and evaluated here with numpy only,
so the pipeline needs no extra dependency and gives the same output on every machine.

Team features are built so that swapping home and away negates the "antisymmetric" block and
leaves the "symmetric" block unchanged. Training uses both orientations, so the model cannot
prefer a side.
"""

import json
from pathlib import Path

import numpy as np

from .features import rolling_mean, shifted

MODEL_PATH = Path(__file__).with_name("learned_model.json")
MISSING = -999.0
_model_cache = {}


def _fill(a, value=MISSING):
    return np.where(np.isfinite(a), a, value).astype(np.float64)


def attack_direction(feats, blocks):
    """+1 where the home team attacks towards +x, -1 otherwise, constant within a block.

    A team's centroid (goalkeeper included) lies on its own side of the opponent's centroid
    almost all the time, so the mean over a half gives the sides without any kick-off logic.
    """
    cx_h, cx_a = feats.team_feats[:, 0], feats.team_feats[:, 1]
    out = np.ones(feats.n)
    for first, last in blocks:
        diff = cx_a[first:last + 1] - cx_h[first:last + 1]
        diff = diff[np.isfinite(diff)]
        if len(diff) and diff.mean() < 0:
            out[first:last + 1] = -1.0
    return out


def _context(a, fps=25):
    """Centred, past-only and future-only means of a signal at a few time scales."""
    cols = [rolling_mean(a, h) for h in (6, 12, 25, 50, 100)]
    for h in (25, 50):
        mean = rolling_mean(a, h)
        cols += [shifted(mean, -h), shifted(mean, h)]      # the h frames before / after
    return cols


def team_features(feats, ball_llr, hint, dead_hand, blocks):
    """(antisymmetric [T, Fa], symmetric [T, Fs]) features for home-vs-away."""
    tf = feats.team_feats
    mid = feats.pitch[0] / 2.0
    direction = attack_direction(feats, blocks)
    ok = feats.ball_ok
    d_h, d_a = np.minimum(feats.d_home, 50.0), np.minimum(feats.d_away, 50.0)
    known = ok & (np.maximum(d_h, d_a) < 50.0)
    gap = np.where(known, np.clip(d_a - d_h, -10.0, 10.0), 0.0)
    vcom = direction * (tf[:, 3] + tf[:, 4]) / 2.0
    anti = [ball_llr] + _context(ball_llr) + [gap] + _context(gap)[2:5] + [
        hint, rolling_mean(hint, 50),
        vcom, direction * rolling_mean((tf[:, 3] + tf[:, 4]) / 2.0, 50),
        direction * (tf[:, 7] + tf[:, 8]) / 2.0,           # one second ahead
        direction * (tf[:, 9] + tf[:, 10]) / 2.0,          # one second back
        direction * ((tf[:, 0] + tf[:, 1]) / 2.0 - mid),   # where the block of players stands
        tf[:, 5], tf[:, 6],                                # spread home minus away, x and y
        np.where(ok, direction * (feats.ball_x - mid), np.nan),
    ]
    anti = np.column_stack([_fill(c, 0.0) for c in anti])

    sym = [
        np.where(known, np.minimum(d_h, d_a), np.nan),
        np.where(ok, feats.ball_speed, np.nan),
        ok.astype(float), feats.ball_obs.astype(float), np.minimum(feats.gap_len, 250),
        rolling_mean(ok.astype(float), 50),
        direction * (tf[:, 1] - tf[:, 0]),
        direction * (tf[:, 3] - tf[:, 4]),
        rolling_mean(np.abs(ball_llr), 25),
        dead_hand, feats.med_speed, feats.frac_slow,
    ]
    sym = np.column_stack([_fill(c) for c in sym])
    return anti, sym


def dead_features(feats, ball_llr, dead_hand):
    """Features [T, F] for dead-vs-alive."""
    n = feats.n
    t = np.arange(n)
    obs = feats.ball_obs
    last = np.maximum.accumulate(np.where(obs, t, -1))
    nxt = np.minimum.accumulate(np.where(obs, t, 10 * n)[::-1])[::-1]
    since = np.where(last >= 0, np.minimum(t - last, 500), 500)
    until = np.minimum(nxt - t, 500)
    d_near = np.minimum(np.minimum(feats.d_home, feats.d_away), 50.0)
    cols = [
        feats.oob, np.clip(feats.inside, -5.0, 35.0), feats.ball_still.astype(float),
        feats.ball_ok.astype(float), obs.astype(float), np.minimum(feats.gap_len, 500),
        since, until, np.where(feats.ball_ok, feats.ball_speed, np.nan),
        np.where(feats.ball_ok, d_near, np.nan),
        feats.med_speed, feats.frac_slow, feats.kick_form.astype(float), feats.n_tracked,
        rolling_mean(feats.med_speed, 75), rolling_mean(feats.med_speed, 150),
        rolling_mean(feats.frac_slow, 75),
        rolling_mean(obs.astype(float), 75),
        rolling_mean(np.abs(ball_llr), 25), rolling_mean(np.abs(ball_llr), 100),
        dead_hand, rolling_mean(dead_hand, 150),
    ] + _context(dead_hand) + [feats.dead_feats[:, k] for k in range(feats.dead_feats.shape[1])]
    return np.column_stack([_fill(c) for c in cols])


# ------------------------------------------------------------------------------------- trees
def predict_trees(model, X):
    """Raw score (log-odds) of a tree ensemble stored as arrays. X [T, F] float64."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    rows = np.arange(len(X))
    out = np.zeros(len(X))
    for tree in model["trees"]:
        feat = np.asarray(tree["f"], dtype=np.int64)
        thr = np.asarray(tree["t"], dtype=np.float64)
        left = np.asarray(tree["l"], dtype=np.int64)
        right = np.asarray(tree["r"], dtype=np.int64)
        value = np.asarray(tree["v"], dtype=np.float64)
        node = np.zeros(len(X), dtype=np.int64)
        for _ in range(int(tree["depth"])):
            f = feat[node]
            go_left = X[rows, np.maximum(f, 0)] <= thr[node]
            node = np.where(f < 0, node, np.where(go_left, left[node], right[node]))
        out += value[node]
    return out


def load_model(path=None):
    path = Path(path) if path else MODEL_PATH
    key = str(path)
    if key not in _model_cache:
        _model_cache[key] = json.loads(path.read_text(encoding="utf-8"))
    return _model_cache[key]


def emissions(feats, ball_llr, hint, dead_hand, blocks, model=None):
    """(team log-odds home-vs-away [T], dead log-odds dead-vs-alive [T]), priors removed."""
    model = model or load_model()
    anti, sym = team_features(feats, ball_llr, hint, dead_hand, blocks)
    team = predict_trees(model["team"], np.column_stack([anti, sym]))
    # average with the mirrored match, so the output is exactly antisymmetric
    team = 0.5 * (team - predict_trees(model["team"], np.column_stack([-anti, sym])))
    dead = predict_trees(model["dead"], dead_features(feats, ball_llr, dead_hand))
    return team, dead - model["dead"]["prior_logit"]
