"""Second home-vs-away model. numpy only at run time.

The first model (learned.py) sees the ball and the nearest player of each team through running
means. This one adds what those means blur:

  * the local shape of "who is closer to the ball, and how close", sampled 0.1 to 2 s before and
    after the frame;
  * touches (a kink in the ball's motion next to a player) and clear contacts (one team at the
    ball, the other not): which team had the last one and will have the next one, how long ago
    and how soon, and how many each team has in the seconds around the frame;
  * what was seen on both sides of a gap in the ball track;
  * the two teams as collectives, oriented towards the goal each attacks: how high the block and
    its back and front lines stand, how wide and deep it is, how fast it moves and in which
    direction, how many players are in the opponent's half. A team that has the ball spreads
    and moves up; a team without it closes up and drops.

It is trained on open Sportec matches only (external_validation/train_team.py), on ownership
labels in which an uncontrolled touch does not count as a change of possession, which is how
the organiser defines a change (CHALLENGE.md). Nothing here is fitted at run time and nothing
comes from the competition match.

Every extra feature is either antisymmetric (changes sign when home and away are swapped) or
symmetric (does not), and the model's output is averaged with the mirrored match, so it is
exactly antisymmetric. No feature uses player identity or slot order. The arithmetic added
here is limited to operations that IEEE 754 rounds exactly (+, -, *, /, sqrt, comparisons,
sorting).
"""
import os

import numpy as np

from . import dead_model, learned
from .config import Config
from .io import AWAY, HOME

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "team_trees.npz")
CAP = 250                      # frames; an event further away than this is "long ago"
FAR = 50.0                     # m; distance used where no distance is known


# ------------------------------------------------------------------------------ small helpers
def _shift(a, k, fill=0.0):
    """a[t + k]; k > 0 looks ahead. Frames beyond the ends get `fill`."""
    out = np.full(len(a), fill, dtype=np.float64)
    if k > 0:
        out[:-k] = a[k:]
    elif k < 0:
        out[-k:] = a[:k]
    else:
        out[:] = a
    return out


def _last(e):
    """(sign of the last non-zero event at or before t, frames since it, capped)."""
    n = len(e)
    t = np.arange(n)
    idx = np.maximum.accumulate(np.where(e != 0, t, -1))
    sign = np.where(idx >= 0, e[np.maximum(idx, 0)], 0.0)
    since = np.where(idx >= 0, np.minimum(t - idx, CAP), CAP).astype(np.float64)
    return sign, since


def _next(e):
    sign, until = _last(e[::-1])
    return sign[::-1], until[::-1]


def _box_sum(a, lo, hi):
    """Sum of a[t + lo .. t + hi], inclusive; lo and hi may be negative."""
    n = len(a)
    c = np.concatenate(([0.0], np.cumsum(a, dtype=np.float64)))
    t = np.arange(n)
    return c[np.clip(t + hi + 1, 0, n)] - c[np.clip(t + lo, 0, n)]


def _centred_mean(a, w):
    """Mean over the w frames centred on t (fewer at the ends), ignoring NaN; 0 where empty."""
    n = len(a)
    fin = np.isfinite(a)
    c = np.concatenate(([0.0], np.cumsum(np.where(fin, a, 0.0), dtype=np.float64)))
    k = np.concatenate(([0.0], np.cumsum(fin, dtype=np.float64)))
    t = np.arange(n)
    i0, i1 = np.clip(t - w // 2, 0, n), np.clip(t + w // 2 + 1, 0, n)
    cnt = k[i1] - k[i0]
    return np.where(cnt > 0, (c[i1] - c[i0]) / np.maximum(cnt, 1.0), 0.0)


def _events(e):
    """Features of a signed event series e (+1 home, -1 away, 0 nothing)."""
    last_sign, since = _last(e)
    next_sign, until = _next(e)
    anti = [last_sign, next_sign, last_sign * (CAP - since) / CAP, next_sign * (CAP - until) / CAP]
    sym = [since, until]
    size = np.abs(e)
    for lo, hi in ((-25, -1), (-75, -1), (1, 25), (1, 75), (1, 150)):
        anti.append(_box_sum(e, lo, hi))
        sym.append(_box_sum(size, lo, hi))
    return anti, sym


# ----------------------------------------------------------------------- the ball and its gaps
def local_features(feats, cfg=None):
    """(antisymmetric [T, Fa], symmetric [T, Fs]) from the ball and the nearest player of each
    team, all taken from the Features the pipeline already has."""
    cfg = cfg or Config()
    d_h = np.minimum(np.where(np.isfinite(feats.d_home), feats.d_home, FAR), FAR)
    d_a = np.minimum(np.where(np.isfinite(feats.d_away), feats.d_away, FAR), FAR)
    ok = np.asarray(feats.ball_ok, dtype=bool) & (np.maximum(d_h, d_a) < FAR)
    gap = np.where(ok, np.clip(d_a - d_h, -10.0, 10.0), 0.0)         # > 0: home is closer
    near = np.where(ok, np.minimum(d_h, d_a), FAR)
    speed = np.where(np.isfinite(feats.ball_speed) & ok, feats.ball_speed, 0.0)
    dv = np.where(np.isfinite(feats.ball_dv) & ok, feats.ball_dv, 0.0)
    anti, sym = [], []

    for k in (-50, -25, -12, -6, -3, 3, 6, 12, 25, 50):               # the local shape
        anti.append(_shift(gap, k))
        sym.append(_shift(near, k, FAR))
    for k in (-12, -6, -3, 3, 6, 12):
        sym.append(_shift(speed, k))

    peak = dv >= cfg.touch_dv                                         # touches, as in possession.py
    for k in (1, 2):
        peak &= (dv >= np.roll(dv, k)) & (dv >= np.roll(dv, -k))
    peak &= ok & (near <= cfg.touch_r) & (np.abs(gap) > 0.3)
    a, s = _events(np.where(peak, np.sign(gap), 0.0))
    anti += a
    sym += s + [dv]

    for radius, clear in ((0.8, 0.4), (1.5, 0.7)):                    # one team clearly at the ball
        contact = np.where(ok & (near < radius) & (np.abs(gap) > clear), np.sign(gap), 0.0)
        a, s = _events(contact)
        anti += a
        sym += s

    n = len(gap)                                                      # both sides of a ball gap
    t = np.arange(n)
    last = np.maximum.accumulate(np.where(ok, t, -1))
    nxt = np.minimum.accumulate(np.where(ok, t, 10 * n + CAP)[::-1])[::-1]
    seen_before, seen_after = last >= 0, nxt < n
    i0, i1 = np.maximum(last, 0), np.minimum(nxt, n - 1)
    anti += [np.where(seen_before, gap[i0], 0.0), np.where(seen_after, gap[i1], 0.0)]
    sym += [np.where(seen_before, np.minimum(t - last, CAP), CAP).astype(np.float64),
            np.where(seen_after, np.minimum(nxt - t, CAP), CAP).astype(np.float64),
            np.where(seen_before, near[i0], FAR), np.where(seen_after, near[i1], FAR)]
    return np.column_stack(anti), np.column_stack(sym)


# ------------------------------------------------------------------- the teams as collectives
def shape_stats(trk, pitch):
    """Numbers about each team on each frame that do not depend on the direction of play.
    -> {HOME: {...}, AWAY: {...}} of [T] float64 arrays. u is the distance from the halfway line
    along the pitch, in metres; lo0 <= lo1 <= lo2 are the three smallest u of the team, hi0 >= hi1
    >= hi2 the three largest."""
    length, width = pitch
    x = np.asarray(trk.x, dtype=np.float64) * length - length / 2.0
    y = np.asarray(trk.y, dtype=np.float64) * width
    s = np.asarray(trk.s, dtype=np.float64)
    team = np.asarray(trk.team)
    out = {}
    for side in (HOME, AWAY):
        cols = np.flatnonzero(team == side)
        if len(cols) < 2:
            raise ValueError("the team model needs at least two track slots per team")
        u, v, sp = x[:, cols], y[:, cols], s[:, cols]
        ok = np.isfinite(u) & np.isfinite(v)
        n = ok.sum(axis=1)
        nn = np.maximum(n, 1).astype(np.float64)
        rows = np.arange(len(u))
        up = np.sort(np.where(ok, u, np.inf), axis=1)                 # ascending, missing last
        width_cols = up.shape[1]

        def low(k):
            return np.where(n > k, up[rows, min(k, width_cols - 1)], np.nan)

        def high(k):
            return np.where(n > k, up[rows, np.maximum(n - 1 - k, 0)], np.nan)

        cu = np.where(n > 0, np.where(ok, u, 0.0).sum(axis=1) / nn, np.nan)
        cv = np.where(n > 0, np.where(ok, v, 0.0).sum(axis=1) / nn, np.nan)
        cu0, cv0 = np.where(n > 0, cu, 0.0), np.where(n > 0, cv, 0.0)
        sd_u = np.sqrt(np.where(ok, (u - cu0[:, None]) ** 2, 0.0).sum(axis=1) / nn)
        sd_v = np.sqrt(np.where(ok, (v - cv0[:, None]) ** 2, 0.0).sum(axis=1) / nn)
        ok_s = ok & np.isfinite(sp)
        ns = np.maximum(ok_s.sum(axis=1), 1).astype(np.float64)
        out[side] = dict(
            n=n.astype(np.float64), cu=cu, sd_u=sd_u, sd_v=sd_v,
            lo0=low(0), lo1=low(1), lo2=low(2), hi0=high(0), hi1=high(1), hi2=high(2),
            mean_s=np.where(ok_s, sp, 0.0).sum(axis=1) / ns,
            fast=(ok_s & (sp > 4.0)).sum(axis=1) / ns,
            right=(ok & (u > 0)).sum(axis=1) / nn, left=(ok & (u < 0)).sum(axis=1) / nn)
    return out


def _rate(a):
    """Change per second of a series over 10 frames, 0 where it is not known."""
    out = np.zeros(len(a))
    out[5:-5] = (a[10:] - a[:-10]) * 2.5
    return np.where(np.isfinite(out), out, 0.0)


def collective_features(feats, shape, blocks):
    """(antisymmetric [T, Fa] = home minus away, symmetric [T, Fs] = home plus away), each team
    seen along its own direction of attack (+ is towards the opponent's goal)."""
    direction = learned.attack_direction(feats, blocks)               # +1: home attacks towards +x
    oriented = {}
    for side, sign in ((HOME, direction), (AWAY, -direction)):
        st = shape[side]
        fwd = sign > 0
        back1 = np.where(fwd, st["lo1"], -st["hi1"])                  # the last line (keeper excluded)
        back2 = np.where(fwd, st["lo2"], -st["hi2"])
        front1 = np.where(fwd, st["hi0"], -st["lo0"])
        front2 = np.where(fwd, st["hi1"], -st["lo1"])
        centre = sign * st["cu"]
        oriented[side] = [
            centre, st["sd_u"], st["sd_v"], back1, back2, front1, front2, st["mean_s"], st["fast"],
            _rate(centre), _rate(front2), _rate(back1),
            np.where(fwd, st["right"], st["left"]),                   # share in the opponent's half
            front1 - back1,
        ]
    anti, sym = [], []
    for h, a in zip(oriented[HOME], oriented[AWAY]):
        diff = np.where(np.isfinite(h - a), h - a, 0.0)
        total = np.where(np.isfinite(h + a), h + a, 0.0)
        change = np.zeros(len(diff))
        change[25:-25] = diff[50:] - diff[:-50]
        anti += [diff, _centred_mean(diff, 25), _centred_mean(diff, 75), change]
        sym.append(total)
    return np.column_stack(anti), np.column_stack(sym)


# ------------------------------------------------------------------------------------- model
def features(feats, ball_llr, hint, dead_hand, blocks, shape, cfg=None):
    """(antisymmetric, symmetric) feature blocks: the first model's features, then ours."""
    anti, sym = learned.team_features(feats, ball_llr, hint, dead_hand, blocks)
    a1, s1 = local_features(feats, cfg)
    a2, s2 = collective_features(feats, shape, blocks)
    return np.column_stack([anti, a1, a2]), np.column_stack([sym, s1, s2])


def load_model(path=None):
    return dead_model.load_model(path or MODEL_PATH)


def logit_from(anti, sym, model):
    """Log-odds home-vs-away [T], averaged with the mirrored match (exactly antisymmetric)."""
    if anti.shape[1] + sym.shape[1] != int(model["n_features"]):
        raise ValueError("the stored team model was trained on {} features, the code makes {}"
                         .format(int(model["n_features"]), anti.shape[1] + sym.shape[1]))
    plain = dead_model.raw_scores(np.column_stack([anti, sym]), model)
    mirrored = dead_model.raw_scores(np.column_stack([-anti, sym]), model)
    return 0.5 * (plain - mirrored)


def logit(feats, ball_llr, hint, dead_hand, blocks, shape, cfg=None, model=None):
    anti, sym = features(feats, ball_llr, hint, dead_hand, blocks, shape, cfg)
    return logit_from(anti, sym, model if model is not None else load_model())
