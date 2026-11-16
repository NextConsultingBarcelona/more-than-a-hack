"""Learned dead-ball detector. numpy only at run time.

A gradient-boosted tree ensemble scores every frame with the log-odds that the ball is dead, from
about 280 collective features of the players and the ball. A two-state Viterbi pass with a
switching cost turns the scores into stoppages, and the stoppages are handed to the H/A/D state
model in possession.py as its dead-ball evidence. deadball.py still locates the kick-off of each
half, and the state model's own dead-ball evidence (learned.py) takes over if the detector's
output is implausible or the detector fails.

Why a separate detector: the dead-ball evidence in deadball.py and learned.py was tuned and
trained on the validation data as built, and the open Sportec data freezes the ball at the point
where play stopped. On that data "the ball is lying still" marks a stoppage almost by
construction. An automatic tracker need not show a still ball -- it may show nothing, a ball
being carried, or a ball lying at the restart point. This detector is trained with all of those
pictures and reads the players as much as the ball (NEXT.md, section 5.8).

Everything here is collective and mirror-symmetric: no feature uses player identity, slot order
or the direction of play.

The ensemble in models/dead_trees.npz was trained by us on open Sportec/IDSSE matches only
(dead_training/). Nothing in it comes from the competition match, and nothing is fitted at run
time. The arithmetic below is limited to operations that IEEE 754 rounds exactly (+, -, *, /,
sqrt, comparisons), so the stoppages should not depend on the machine they are computed on.
"""

import os
import warnings

import numpy as np

from .config import Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "dead_trees.npz")
CAP_S = 20.0
_models = {}


# --------------------------------------------------------------------------------- tree ensemble
def load_model(path=MODEL_PATH):
    """The ensemble as flat arrays (cached per path)."""
    if path not in _models:
        with np.load(path, allow_pickle=False) as z:
            _models[path] = {k: z[k] for k in z.files}
    return _models[path]


def raw_scores(X, model):
    """Log-odds of "dead" per row of X [n, F] (float32, NaN = missing).

    Each tree is walked for all rows at once. A node sends a row left when its feature is at most
    the node's threshold, a missing feature goes the way the node learned in training. Tree
    outputs are added in a fixed order.
    """
    n = X.shape[0]
    feature, number = model["feature"], model["number"]
    left, right, nan_left = model["left"], model["right"], model["nan_left"]
    raw = np.full(n, float(model["baseline"]), dtype=np.float64)
    rows_all = np.arange(n)
    for root in model["roots"]:
        node = np.full(n, int(root), dtype=np.int32)
        rows = rows_all
        while len(rows):
            cur = node[rows]
            feat = feature[cur]
            inner = feat >= 0                       # leaves carry feature -1
            rows, cur, feat = rows[inner], cur[inner], feat[inner]
            if not len(rows):
                break
            x = X[rows, feat]
            miss = np.isnan(x)
            go_left = np.where(miss, nan_left[cur], x <= number[cur])
            node[rows] = np.where(go_left, left[cur], right[cur])
        raw += number[node]                         # a leaf's number is its value
    return raw


# -------------------------------------------------------------------------------------- features
def _hyp(a, b):
    return np.sqrt(a * a + b * b)


def _box(a, w):
    """Centred moving mean over w (odd) frames, edge value repeated. float64."""
    a = np.asarray(a, dtype=np.float64)
    h = w // 2
    pad = np.concatenate([np.full(h, a[0]), a, np.full(h, a[-1])])
    c = np.cumsum(pad)
    out = c[w - 1:].copy()
    out[1:] -= c[:-w]
    return out / w


def _nanmean_filter(a, w):
    """Centred moving mean over w frames that ignores NaN; NaN where the window is empty."""
    ok = ~np.isnan(a)
    num = _box(np.where(ok, a, 0.0), w)
    den = _box(ok.astype(np.float64), w)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-9, num / den, np.nan).astype(np.float32)


def _shift(a, k):
    """a[t + k], edge-padded."""
    if k == 0:
        return a
    out = np.empty_like(a)
    if k > 0:
        out[:-k] = a[k:]
        out[-k:] = a[-1]
    else:
        out[-k:] = a[:k]
        out[:-k] = a[0]
    return out


def _velocity(x, h, fps):
    """Central difference over +-h frames; NaN if either end is missing."""
    v = np.full_like(x, np.nan)
    v[h:-h] = (x[2 * h:] - x[:-2 * h]) * (fps / (2.0 * h))
    return v


def _since(flag):
    """Frames since flag was last True (0 where True), and the index of that frame."""
    n = len(flag)
    idx = np.where(flag, np.arange(n), -10 ** 9)
    last = np.maximum.accumulate(idx)
    return np.arange(n) - last, last


def _until(flag):
    d, last = _since(flag[::-1])
    n = len(flag)
    return d[::-1], (n - 1 - last)[::-1]


def _player_features(px, py, team, pitch, fps):
    L, W = pitch
    n = px.shape[0]
    vis = ~np.isnan(px)
    vx, vy = _velocity(px, 3, fps), _velocity(py, 3, fps)
    sp = _hyp(vx, vy)
    bad = sp > 12.0                       # slot swaps and tracker jumps, not running
    sp[bad] = np.nan
    vx[bad] = np.nan
    vy[bad] = np.nan
    acc = np.abs(_velocity(sp, 3, fps))
    f = {}
    f["n_vis"] = vis.sum(1).astype(np.float32)
    f["sp_mean"] = np.nanmean(sp, 1)
    q = np.nanpercentile(sp, [25, 50, 90], axis=1)
    f["sp_p25"], f["sp_med"], f["sp_p90"] = q[0], q[1], q[2]
    f["sp_max"] = np.nanmax(np.where(np.isnan(sp), -1, sp), 1)
    f["sp_std"] = np.nanstd(sp, 1)
    nsp = (~np.isnan(sp)).sum(1).clip(1)
    f["frac_still"] = (sp < 1.0).sum(1) / nsp
    f["frac_walk"] = (sp < 2.5).sum(1) / nsp
    f["frac_run"] = (sp > 5.0).sum(1) / nsp
    f["acc_mean"] = np.nanmean(acc, 1)
    mvx, mvy = np.nanmean(vx, 1), np.nanmean(vy, 1)
    f["coherence"] = _hyp(mvx, mvy) / (f["sp_mean"] + 0.1)
    f["flow_x"] = np.abs(mvx)
    cx, cy = np.nanmean(px, 1), np.nanmean(py, 1)
    f["cx_abs"], f["cy_abs"] = np.abs(cx), np.abs(cy)
    f["sx"], f["sy"] = np.nanstd(px, 1), np.nanstd(py, 1)
    tc = []
    for t in (0, 1):
        m = team == t
        tc.append((np.nanmean(px[:, m], 1), np.nanmean(py[:, m], 1), np.nanstd(px[:, m], 1),
                   np.nanmean(sp[:, m], 1), np.nanmean(np.sign(px[:, m]), 1)))
    f["team_dx"] = np.abs(tc[0][0] - tc[1][0])
    f["team_dy"] = np.abs(tc[0][1] - tc[1][1])
    f["team_sx_min"] = np.fmin(tc[0][2], tc[1][2])
    f["team_sx_max"] = np.fmax(tc[0][2], tc[1][2])
    f["team_sp_min"] = np.fmin(tc[0][3], tc[1][3])
    f["team_sp_max"] = np.fmax(tc[0][3], tc[1][3])
    f["half_sep"] = np.abs(tc[0][4] - tc[1][4]) / 2.0      # 1 = each team in its own half
    ay, ax = np.abs(py), np.abs(px)
    f["y_out"] = np.nanmax(np.where(vis, ay, -1), 1) - W / 2
    f["x_out"] = np.nanmax(np.where(vis, ax, -1), 1) - L / 2
    f["n_out"] = ((ay > W / 2 + 0.3) | (ax > L / 2 + 0.3)).sum(1).astype(np.float32)
    f["n_touch"] = (ay > W / 2 - 3.0).sum(1).astype(np.float32)
    inbox = (ax > L / 2 - 16.5) & (ay < 20.16)
    bl, br = (inbox & (px < 0)).sum(1), (inbox & (px > 0)).sum(1)
    f["box_max"] = np.maximum(bl, br).astype(np.float32)
    f["box_min"] = np.minimum(bl, br).astype(np.float32)
    f["x_p10_p90"] = np.nanpercentile(px, 90, axis=1) - np.nanpercentile(px, 10, axis=1)
    # distance from every player to the nearest opponent, in chunks to keep memory flat
    hm, am = team == 0, team == 1
    mh = np.full((n, int(hm.sum())), 1e3, np.float32)
    ma = np.full((n, int(am.sum())), 1e3, np.float32)
    for i in range(0, n, 20000):
        j = min(n, i + 20000)
        d = _hyp(px[i:j, hm][:, :, None] - px[i:j, am][:, None, :],
                 py[i:j, hm][:, :, None] - py[i:j, am][:, None, :])
        d = np.where(np.isnan(d), 1e3, d)
        mh[i:j], ma[i:j] = d.min(2), d.min(1)
    both = np.concatenate([np.where(mh > 500, np.nan, mh), np.where(ma > 500, np.nan, ma)], 1)
    f["opp_mean"] = np.nanmean(both, 1)
    f["opp_p25"] = np.nanpercentile(both, 25, axis=1)
    f["n_marked"] = (both < 2.0).sum(1).astype(np.float32)
    return {k: np.asarray(v, np.float32) for k, v in f.items()}


def _ball_features(px, py, bx, by, pitch, fps):
    L, W = pitch
    n = len(bx)
    vis = ~np.isnan(bx)
    visf = vis.astype(np.float32)
    f = {}
    f["b_vis"] = visf
    f["b_ax"], f["b_ay"] = np.abs(bx), np.abs(by)
    out = np.fmax(np.abs(bx) - L / 2, np.abs(by) - W / 2)        # > 0 outside the pitch
    f["b_out"] = out
    f["b_out_x"] = np.abs(bx) - L / 2
    f["b_out_y"] = np.abs(by) - W / 2
    v1x, v1y = _velocity(bx, 1, fps), _velocity(by, 1, fps)
    v3x, v3y = _velocity(bx, 3, fps), _velocity(by, 3, fps)
    sp1, sp3 = _hyp(v1x, v1y), _hyp(v3x, v3y)
    spd = np.where(np.isnan(sp3), sp1, sp3)
    f["b_speed"] = spd
    # velocity component towards the nearest boundary, and the time to reach it
    ox = np.where(np.isnan(v3x), v1x, v3x) * np.sign(bx)
    oy = np.where(np.isnan(v3y), v1y, v3y) * np.sign(by)
    f["b_vout_x"], f["b_vout_y"] = ox, oy
    tx = np.where(ox > 0.5, (L / 2 - np.abs(bx)) / ox, np.nan)
    ty = np.where(oy > 0.5, (W / 2 - np.abs(by)) / oy, np.nan)
    f["b_t_exit"] = np.clip(np.fmin(tx, ty), -2, 10)
    d = _hyp(px - bx[:, None], py - by[:, None])
    dn = np.where(np.isnan(d), 1e3, d)
    ds = np.sort(dn, 1)[:, :2]
    f["b_d1"] = np.where(vis & (ds[:, 0] < 500), ds[:, 0], np.nan)
    f["b_d2"] = np.where(vis & (ds[:, 1] < 500), ds[:, 1], np.nan)
    f["b_n5"] = np.where(vis, (dn < 5).sum(1), np.nan)
    f["b_n15"] = np.where(vis, (dn < 15).sum(1), np.nan)
    f["b_corner"] = _hyp(np.abs(bx) - L / 2, np.abs(by) - W / 2)
    f["b_centre"] = _hyp(bx, by)
    f["b_goalarea"] = _hyp(np.abs(bx) - (L / 2 - 5.5), np.fmax(np.abs(by) - 9.16, 0))
    # how long the ball has stayed within 0.7 m of where it is now
    still = np.zeros(n, np.float32)
    for k in (5, 12, 25, 50, 100, 200):
        for sgn in (-1, 1):
            ok = _hyp(_shift(bx, sgn * k) - bx, _shift(by, sgn * k) - by) < 0.7
            still += np.where(ok, 1.0, 0.0).astype(np.float32)
    f["b_still"] = np.where(vis, still, np.nan)
    for name, k in (("b_move_p1", -25), ("b_move_f1", 25), ("b_move_p3", -75), ("b_move_f3", 75)):
        f[name] = _hyp(_shift(bx, k) - bx, _shift(by, k) - by)
    # what was seen before and after each gap in the ball track
    since, last = _since(vis)
    until, nxt = _until(vis)
    f["g_since"] = np.clip(since / fps, 0, CAP_S).astype(np.float32)
    f["g_until"] = np.clip(until / fps, 0, CAP_S).astype(np.float32)
    f["g_len"] = np.clip((since + until) / fps, 0, 2 * CAP_S).astype(np.float32)
    li, ni = np.clip(last, 0, n - 1), np.clip(nxt, 0, n - 1)
    has_l, has_n = last >= 0, nxt < n
    for key in ("b_out", "b_speed", "b_t_exit", "b_d1", "b_n5", "b_still", "b_ax", "b_ay"):
        f["gl_" + key[2:]] = np.where(has_l, f[key][li], np.nan)
        f["gn_" + key[2:]] = np.where(has_n, f[key][ni], np.nan)
    jump = _hyp(bx[ni] - bx[li], by[ni] - by[li])
    f["g_jump"] = np.where(has_l & has_n, jump, np.nan)
    f["g_jump_speed"] = np.where(since + until > 0, jump / ((since + until) / fps + 1e-6), np.nan)
    for w_s in (1, 3, 8):
        w = int(w_s * fps) | 1
        f["b_vis_{}".format(w_s)] = _box(visf, w)
        f["b_speed_{}".format(w_s)] = _nanmean_filter(spd, w)
        f["b_out_{}".format(w_s)] = _nanmean_filter(out, w)
    w = 2 * fps + 1
    for k, name in ((-50, "p2"), (50, "f2")):
        f["b_vis_" + name] = _shift(_box(visf, w), k // 2)
        f["b_speed_" + name] = _shift(_nanmean_filter(spd, w), k // 2)
    return {k: np.asarray(v, np.float32) for k, v in f.items()}


CONTEXT = ("sp_mean", "sp_med", "sp_p90", "sp_p25", "frac_still", "frac_walk", "frac_run",
           "acc_mean", "coherence", "flow_x", "half_sep", "box_max", "y_out", "n_out", "sx",
           "team_dx", "opp_mean", "n_marked", "x_p10_p90", "team_sp_min", "team_sp_max")


def _context_features(f, fps):
    """The player features over 1 s to 20 s windows, before and after the frame."""
    out = {}
    for k in CONTEXT:
        a = f[k]
        for w_s in (1, 3, 8, 20):
            out["{}_m{}".format(k, w_s)] = _nanmean_filter(a, int(w_s * fps) | 1)
        m2 = _nanmean_filter(a, 2 * fps + 1)
        out[k + "_p2"] = _shift(m2, -fps)
        out[k + "_f2"] = _shift(m2, fps)
        m6 = _nanmean_filter(a, 6 * fps + 1)
        out[k + "_p6"] = _shift(m6, -3 * fps)
        out[k + "_f6"] = _shift(m6, 3 * fps)
        out[k + "_d2"] = out[k + "_f2"] - out[k + "_p2"]
    return out


def features(trk, pitch, fps=25):
    """(X float32 [T, F], feature names) from a Tracking with normalised 0-1 coordinates."""
    L, W = float(pitch[0]), float(pitch[1])
    px = (trk.x.astype(np.float32) - np.float32(0.5)) * np.float32(L)
    py = (trk.y.astype(np.float32) - np.float32(0.5)) * np.float32(W)
    bx = (trk.ball[:, 0].astype(np.float32) - np.float32(0.5)) * np.float32(L)
    by = (trk.ball[:, 1].astype(np.float32) - np.float32(0.5)) * np.float32(W)
    gone = np.isnan(bx) | np.isnan(by)
    bx[gone] = np.nan
    by[gone] = np.nan
    gone = np.isnan(px) | np.isnan(py)
    px[gone] = np.nan
    py[gone] = np.nan
    team = np.asarray(trk.team)
    if min(int((team == 0).sum()), int((team == 1).sum())) < 2:
        raise ValueError("the dead-ball detector needs at least two track slots per team")
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # frames without any tracking
        f = _player_features(px, py, team, (L, W), fps)
        f.update(_context_features(f, fps))
        f.update(_ball_features(px, py, bx, by, (L, W), fps))
    names = sorted(f)
    X = np.stack([f[k] for k in names], 1).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X, names


# -------------------------------------------------------------------------------------- decoding
def two_state_viterbi(score, cost):
    """Best alive/dead path: maximises the summed score of the dead frames minus `cost` per switch."""
    n = len(score)
    if n == 0:
        return np.zeros(0, dtype=bool)
    s = np.asarray(score, dtype=np.float64).tolist()
    b0, b1 = bytearray(n), bytearray(n)
    alive, dead = 0.0, s[0]
    for t in range(1, n):
        if alive >= dead - cost:
            na, b0[t] = alive, 0
        else:
            na, b0[t] = dead - cost, 1
        if dead >= alive - cost:
            nd, b1[t] = dead, 1
        else:
            nd, b1[t] = alive - cost, 0
        alive, dead = na, nd + s[t]
    path = np.zeros(n, dtype=bool)
    state = 1 if dead > alive else 0
    back = (b0, b1)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[state][t]
    return path


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def scores(trk, pitch, fps=25, model=None):
    """Log-odds of "dead" for every frame [T]. The expensive step; depends on no Config value."""
    model = model if model is not None else load_model()
    X, names = features(trk, pitch, fps)
    if names != [str(s) for s in model["names"]]:
        raise ValueError("dead-ball model and feature code do not match")
    return raw_scores(X, model)


def stoppages(raw, blocks, cfg=None):
    """Dead mask [T] from the log-odds. Frames outside the graded blocks are False."""
    cfg = cfg or Config()
    mask = np.zeros(len(raw), dtype=bool)
    for first, last in blocks:
        score = np.clip(raw[first:last + 1], -cfg.dm_clip, cfg.dm_clip) + cfg.dm_bias
        d = two_state_viterbi(score, cfg.dm_cost)
        for a, b in _runs(~d):                      # a live gap too short to be a possession
            if b - a < cfg.frames(cfg.dm_min_alive_s) and a > 0 and b < len(d):
                d[a:b] = True
        for a, b in _runs(d):
            if b - a < cfg.frames(cfg.dm_min_dead_s):
                d[a:b] = False
        mask[first:last + 1] = d
    return mask


def plausible(mask, blocks, cfg=None):
    """Label-free guard: is this a believable amount of dead ball for a football match?

    The ranges are several times wider than anything seen on the open matches. Outside them
    the detector has met data it does not understand and the state model keeps its own
    dead-ball evidence.
    """
    cfg = cfg or Config()
    frames = sum(last - first + 1 for first, last in blocks)
    if frames == 0:
        return False
    dead = sum(int(mask[first:last + 1].sum()) for first, last in blocks)
    stops = sum(len(_runs(mask[first:last + 1])) for first, last in blocks)
    share = dead / float(frames)
    per_90 = stops / (frames / float(cfg.fps) / 60.0) * 90.0
    return (cfg.dm_share_lo <= share <= cfg.dm_share_hi) and (cfg.dm_stops_lo <= per_90 <= cfg.dm_stops_hi)


def evidence(own_llr, prekick, raw, blocks, cfg=None):
    """Dead-ball evidence for possession.decode: the detector's stoppages as a hard +-weight.

    own_llr and prekick come from deadball.compute, raw from scores(). Graded frames before a
    half's kick-off stay dead whatever the detector says. If the detector's output is not
    plausible, own_llr is returned unchanged and the caller keeps the state model's own
    evidence. -> (evidence [T], used_model).
    """
    cfg = cfg or Config()
    mask = stoppages(raw, blocks, cfg)
    if not plausible(mask, blocks, cfg):
        return own_llr, False
    graded = np.zeros(len(mask), dtype=bool)
    for first, last in blocks:
        graded[first:last + 1] = True
    out = np.where(mask | prekick, cfg.dm_weight, -cfg.dm_weight).astype(np.float64)
    out[~graded] = 0.0
    return out, True
