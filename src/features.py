"""Per-frame features from the tracking. Everything is a numpy array of length T (video frames).

Distances are in metres. The ball is cleaned (tracker jumps removed) and short gaps are linearly
interpolated; `ball_obs` says where it was really detected and `ball_ok` where a position exists
at all. No feature assumes the ball is present: where it is not, ball features are NaN / inf and
the player-only features carry on.
"""

import warnings
from dataclasses import dataclass

import numpy as np

from .config import Config
from .io import AWAY, HOME


@dataclass
class Features:
    n: int
    pitch: tuple
    ball_x: np.ndarray       # m, NaN where not ball_ok
    ball_y: np.ndarray
    ball_obs: np.ndarray     # bool, detected by the tracker (after cleaning)
    ball_ok: np.ndarray      # bool, detected or interpolated
    gap_len: np.ndarray      # frames, length of the ball gap this frame sits in (0 if observed)
    ball_speed: np.ndarray   # m/s, NaN where unknown
    ball_dv: np.ndarray      # m/s, change of ball velocity across the frame (touch detector)
    ball_still: np.ndarray   # bool, ball detected and lying still for about a second
    oob: np.ndarray          # m outside the pitch (0 inside or unknown)
    inside: np.ndarray       # m inside the nearest line (negative outside, NaN unknown)
    d_home: np.ndarray       # m, nearest home player to the ball (inf if unknown)
    d_away: np.ndarray
    med_speed: np.ndarray    # m/s, median player speed, smoothed over 1 s
    frac_slow: np.ndarray    # share of tracked players slower than 1 m/s, smoothed
    n_tracked: np.ndarray
    kick_raw: np.ndarray     # bool, both teams entirely in their own halves on this frame
    kick_form: np.ndarray    # bool, the same, sustained for about 2 s
    ball_center: np.ndarray  # bool, ball within 2 m of the centre spot
    team_feats: np.ndarray   # [T, F] player-only team-shape features for the ball-free hint
    dead_feats: np.ndarray   # [T, F] player-only tempo features for the ball-free dead hint


# ------------------------------------------------------------------------------- small helpers
def rolling_mean(a, half):
    """Centred mean over +/- half frames, ignoring NaN. NaN where the window has no data."""
    a = np.asarray(a, dtype=np.float64)
    present = np.isfinite(a)
    cv = np.concatenate(([0.0], np.cumsum(np.where(present, a, 0.0))))
    cp = np.concatenate(([0.0], np.cumsum(present)))
    i = np.arange(len(a))
    lo = np.maximum(0, i - half)
    hi = np.minimum(len(a), i + half + 1)
    n = cp[hi] - cp[lo]
    out = np.full(len(a), np.nan)
    ok = n > 0
    out[ok] = (cv[hi] - cv[lo])[ok] / n[ok]
    return out


def shifted(a, k):
    """a[t + k] with NaN padding (k may be negative)."""
    out = np.full(a.shape, np.nan, dtype=np.float64)
    if k == 0:
        out[:] = a
    elif k > 0:
        out[:-k] = a[k:]
    else:
        out[-k:] = a[:k]
    return out


def _row_stats(v):
    """Row-wise count, mean and std ignoring NaN, without warnings on empty rows."""
    fin = np.isfinite(v)
    cnt = fin.sum(axis=1)
    safe = np.maximum(cnt, 1)
    mean = np.where(fin, v, 0.0).sum(axis=1) / safe
    var = (np.where(fin, v - mean[:, None], 0.0) ** 2).sum(axis=1) / safe
    mean = np.where(cnt > 0, mean, np.nan)
    std = np.where(cnt > 1, np.sqrt(var), np.nan)
    return cnt, mean, std


# ---------------------------------------------------------------------------------------- ball
def clean_ball(bx, by, cfg):
    """Drop detections that sit in short segments bounded by physically impossible jumps."""
    bx = bx.astype(np.float64).copy()
    by = by.astype(np.float64).copy()
    obs = np.isfinite(bx) & np.isfinite(by)
    idx = np.flatnonzero(obs)
    if len(idx) >= 2:
        dt = np.diff(idx) / float(cfg.fps)
        speed = np.hypot(np.diff(bx[idx]), np.diff(by[idx])) / dt
        seg = np.concatenate(([0], np.cumsum(speed > cfg.ball_vmax)))
        size = np.bincount(seg)
        drop = idx[size[seg] < cfg.ball_min_seg]
        if len(drop) < len(idx):          # never drop everything
            bx[drop] = np.nan
            by[drop] = np.nan
            obs[drop] = False
    bx[~obs] = np.nan
    by[~obs] = np.nan
    return bx, by, obs


def interp_ball(bx, by, obs, max_gap):
    """Linear interpolation across gaps of at most max_gap frames. Returns x, y, ok, gap_len."""
    n = len(bx)
    t = np.arange(n)
    idx = np.flatnonzero(obs)
    gap_len = np.zeros(n, dtype=np.int64)
    if len(idx) == 0:
        gap_len[:] = n
        return bx.copy(), by.copy(), obs.copy(), gap_len
    prev = np.maximum.accumulate(np.where(obs, t, -1))
    nxt = np.minimum.accumulate(np.where(obs, t, n)[::-1])[::-1]
    inner = (prev >= 0) & (nxt < n)
    gap = np.where(inner, nxt - prev - 1, n)
    gap_len[~obs] = gap[~obs]
    fill = ~obs & inner & (gap <= max_gap)
    ok = obs | fill
    fx = np.interp(t, idx, bx[idx])
    fy = np.interp(t, idx, by[idx])
    fx[~ok] = np.nan
    fy[~ok] = np.nan
    return fx, fy, ok, gap_len


# ---------------------------------------------------------------------------------------- main
def compute(trk, pitch, cfg=None):
    cfg = cfg or Config()
    length, width = pitch
    fps = float(cfg.fps)
    X = trk.x.astype(np.float32) * np.float32(length)
    Y = trk.y.astype(np.float32) * np.float32(width)
    n = X.shape[0]
    home = trk.team == HOME
    away = trk.team == AWAY

    # ball
    bx, by, obs = clean_ball(trk.ball[:, 0] * length, trk.ball[:, 1] * width, cfg)
    fx, fy, ok, gap_len = interp_ball(bx, by, obs, cfg.ball_max_gap)
    # Detection noise of a decimetre is metres per second after differencing at 25 fps, so the
    # positions are smoothed first (Vidal-Codina et al. use a 7-frame Savitzky-Golay filter).
    raw_x, raw_y = fx, fy
    fx = np.where(ok, rolling_mean(raw_x, 2), np.nan)
    fy = np.where(ok, rolling_mean(raw_y, 2), np.nan)
    k = 3
    vx = (shifted(fx, k) - shifted(fx, -k)) * fps / (2 * k)
    vy = (shifted(fy, k) - shifted(fy, -k)) * fps / (2 * k)
    speed = np.hypot(vx, vy)
    k = 5
    dvx = (shifted(fx, k) - 2 * fx + shifted(fx, -k)) * fps / k
    dvy = (shifted(fy, k) - 2 * fy + shifted(fy, -k)) * fps / k
    dv = np.hypot(dvx, dvy)

    # still = the detected ball stays within a few decimetres for about a second
    half_s = int(fps) // 2
    seen = rolling_mean(obs.astype(float), half_s)
    spread = np.sqrt(
        np.maximum(rolling_mean(raw_x ** 2, half_s) - rolling_mean(raw_x, half_s) ** 2, 0.0)
        + np.maximum(rolling_mean(raw_y ** 2, half_s) - rolling_mean(raw_y, half_s) ** 2, 0.0))
    ball_still = obs & (seen >= 0.8) & (np.nan_to_num(spread, nan=np.inf) < cfg.still_spread)

    out_x = np.maximum(np.maximum(-fx, fx - length), 0.0)
    out_y = np.maximum(np.maximum(-fy, fy - width), 0.0)
    oob = np.where(ok, np.hypot(out_x, out_y), 0.0)
    inside = np.minimum(np.minimum(fx, length - fx), np.minimum(fy, width - fy))

    # nearest player of each team
    with np.errstate(invalid="ignore"):
        dist = np.hypot(X - fx[:, None].astype(np.float32), Y - fy[:, None].astype(np.float32))
    dist[~np.isfinite(dist)] = np.inf
    d_home = dist[:, home].min(axis=1).astype(np.float64) if home.any() else np.full(n, np.inf)
    d_away = dist[:, away].min(axis=1).astype(np.float64) if away.any() else np.full(n, np.inf)
    del dist

    # player speeds: delivered value, else from positions
    k = 2
    comp = np.full(X.shape, np.nan, dtype=np.float32)
    comp[k:-k] = np.hypot(X[2 * k:] - X[:-2 * k], Y[2 * k:] - Y[:-2 * k]) * (fps / (2 * k))
    sp = np.where(np.isfinite(trk.s), trk.s, comp).astype(np.float32)
    sp[~np.isfinite(X)] = np.nan
    sp[sp > 12.0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(sp, axis=1)
    fin = np.isfinite(sp)
    cnt = fin.sum(axis=1)
    slow = (np.where(fin, sp, np.inf) < 1.0).sum(axis=1) / np.maximum(cnt, 1)
    slow = np.where(cnt > 0, slow, np.nan)
    half = int(round(fps / 2))
    med_speed = rolling_mean(med, half)
    frac_slow = rolling_mean(slow, half)

    # team shape
    n_h, cx_h, sx_h = _row_stats(X[:, home])
    n_a, cx_a, sx_a = _row_stats(X[:, away])
    _, _, sy_h = _row_stats(Y[:, home])
    _, _, sy_a = _row_stats(Y[:, away])
    n_tracked = np.isfinite(X).sum(axis=1)

    # kick-off formation: every tracked player on his own side of the halfway line
    tol = 1.5
    mid = length / 2.0
    miss = ~np.isfinite(X)
    left = (X < mid + tol) | miss
    right = (X > mid - tol) | miss
    form_a = left[:, home].all(axis=1) & right[:, away].all(axis=1)
    form_b = right[:, home].all(axis=1) & left[:, away].all(axis=1)
    kick_raw = (form_a | form_b) & (n_h >= 7) & (n_a >= 7)
    kick_form = rolling_mean(kick_raw.astype(float), int(fps)) >= 0.9
    ball_center = ok & (np.hypot(fx - mid, fy - width / 2.0) < 2.0)

    # player-only features for the ball-free possession hint
    cx_h, cx_a = rolling_mean(cx_h, half), rolling_mean(cx_a, half)
    vx_h = (shifted(cx_h, half) - shifted(cx_h, -half)) * fps / (2 * half)
    vx_a = (shifted(cx_a, half) - shifted(cx_a, -half)) * fps / (2 * half)
    dsx = rolling_mean(sx_h - sx_a, half)
    dsy = rolling_mean(sy_h - sy_a, half)
    lag = int(fps)
    cols = [cx_h, cx_a, cx_h - cx_a, vx_h, vx_a, dsx, dsy,
            shifted(vx_h, lag), shifted(vx_a, lag), shifted(vx_h, -lag), shifted(vx_a, -lag)]
    if cfg.hint_rich:
        _, sp_h, _ = _row_stats(sp[:, home])
        _, sp_a, _ = _row_stats(sp[:, away])
        _, cy_h, _ = _row_stats(Y[:, home])
        _, cy_a, _ = _row_stats(Y[:, away])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ext_h = np.nanmax(X[:, home], axis=1) - np.nanmin(X[:, home], axis=1)
            ext_a = np.nanmax(X[:, away], axis=1) - np.nanmin(X[:, away], axis=1)
        dsp = rolling_mean(sp_h - sp_a, half)
        dext = rolling_mean(ext_h - ext_a, half)
        dcy = rolling_mean(cy_h - cy_a, half)
        cols += [dsp, dext, dcy, vx_h - vx_a, dsx * (cx_h - cx_a), dsy * dsy,
                 shifted(vx_h, 2 * lag), shifted(vx_a, 2 * lag),
                 shifted(vx_h, -2 * lag), shifted(vx_a, -2 * lag),
                 shifted(dsx, lag), shifted(dsx, -lag), shifted(dsp, lag), shifted(dsp, -lag)]
    team_feats = np.column_stack(cols)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sd_speed = rolling_mean(np.nanstd(sp, axis=1), half)
        p90 = rolling_mean(np.nanpercentile(sp, 90, axis=1), half)
    dead_feats = np.column_stack([
        med_speed, frac_slow, sd_speed, p90, np.abs(vx_h), np.abs(vx_a),
        shifted(med_speed, lag), shifted(med_speed, -lag),
        shifted(med_speed, 3 * lag), shifted(med_speed, -3 * lag),
        shifted(p90, 2 * lag), shifted(p90, -2 * lag)])

    return Features(
        n=n, pitch=(length, width), ball_x=fx, ball_y=fy, ball_obs=obs, ball_ok=ok,
        gap_len=gap_len, ball_speed=speed, ball_dv=dv, ball_still=ball_still, oob=oob, inside=inside,
        d_home=d_home, d_away=d_away, med_speed=med_speed, frac_slow=frac_slow,
        n_tracked=n_tracked, kick_raw=kick_raw, kick_form=kick_form, ball_center=ball_center,
        team_feats=team_feats, dead_feats=dead_feats)
