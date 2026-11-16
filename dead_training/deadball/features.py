"""Training-time per-frame features for dead-ball detection.

The run-time code in src/dead_model.py is a numpy-only rewrite of this file with the same
definitions (dead_training/check_features.py compares the two). Everything is collective (no
player identity) and mirror-symmetric (no assumption about which team attacks which way),
because track slots are not people and the direction of play is not given. Ball features are
NaN-aware.
"""
import warnings

import numpy as np
from scipy.ndimage import uniform_filter1d

FPS = 25
CAP_S = 20.0


def _nanmean_filter(a, w):
    """Centred moving nan-mean over w frames (axis 0)."""
    ok = ~np.isnan(a)
    num = uniform_filter1d(np.where(ok, a, 0.0).astype(np.float64), w, axis=0, mode="nearest")
    den = uniform_filter1d(ok.astype(np.float64), w, axis=0, mode="nearest")
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


def _velocity(x, h=3, fps=FPS):
    """Central difference over +-h frames; NaN if either end is missing."""
    v = np.full_like(x, np.nan)
    v[h:-h] = (x[2 * h:] - x[:-2 * h]) * (fps / (2.0 * h))
    return v


def _since(flag):
    """Frames since flag was last True (0 where True, large before the first)."""
    n = len(flag)
    idx = np.where(flag, np.arange(n), -10 ** 9)
    last = np.maximum.accumulate(idx)
    return np.arange(n) - last, last


def _until(flag):
    d, last = _since(flag[::-1])
    n = len(flag)
    return d[::-1], (n - 1 - last)[::-1]


def player_features(fr):
    px, py, team = fr.px, fr.py, fr.team
    L, W = fr.pitch
    fps = fr.fps
    n = fr.n
    vis = ~np.isnan(px)
    vx, vy = _velocity(px, 3, fps), _velocity(py, 3, fps)
    sp = np.hypot(vx, vy)
    bad = sp > 12.0                      # slot swaps and tracker jumps, not running
    sp[bad] = np.nan
    vx[bad] = np.nan
    vy[bad] = np.nan
    acc = np.abs(_velocity(sp, 3, fps))
    f = {}
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN frames outside play
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
        f["coherence"] = np.hypot(mvx, mvy) / (f["sp_mean"] + 0.1)
        f["flow_x"] = np.abs(mvx)
        cx, cy = np.nanmean(px, 1), np.nanmean(py, 1)
        f["cx_abs"], f["cy_abs"] = np.abs(cx), np.abs(cy)
        f["sx"], f["sy"] = np.nanstd(px, 1), np.nanstd(py, 1)
        tc = []
        for t in (0, 1):
            m = team == t
            tx, ty = np.nanmean(px[:, m], 1), np.nanmean(py[:, m], 1)
            tc.append((tx, ty, np.nanstd(px[:, m], 1), np.nanmean(sp[:, m], 1),
                       np.nanmean(np.sign(px[:, m]), 1)))
        f["team_dx"] = np.abs(tc[0][0] - tc[1][0])
        f["team_dy"] = np.abs(tc[0][1] - tc[1][1])
        f["team_sx_min"] = np.fmin(tc[0][2], tc[1][2])
        f["team_sx_max"] = np.fmax(tc[0][2], tc[1][2])
        f["team_sp_min"] = np.fmin(tc[0][3], tc[1][3])
        f["team_sp_max"] = np.fmax(tc[0][3], tc[1][3])
        f["half_sep"] = np.abs(tc[0][4] - tc[1][4]) / 2.0     # 1 = each team in its own half
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
        # nearest opponent distances, in chunks to keep memory flat on a full match
        hm, am = team == 0, team == 1
        mh = np.empty((n, int(hm.sum())), np.float32)
        ma = np.empty((n, int(am.sum())), np.float32)
        for i in range(0, n, 20000):
            j = min(n, i + 20000)
            dx = px[i:j, hm][:, :, None] - px[i:j, am][:, None, :]
            dy = py[i:j, hm][:, :, None] - py[i:j, am][:, None, :]
            d = np.hypot(dx, dy)
            d = np.where(np.isnan(d), 1e3, d)
            mh[i:j], ma[i:j] = d.min(2), d.min(1)
        mh = np.where(mh > 500, np.nan, mh)
        ma = np.where(ma > 500, np.nan, ma)
        both = np.concatenate([mh, ma], 1)
        f["opp_mean"] = np.nanmean(both, 1)
        f["opp_p25"] = np.nanpercentile(both, 25, axis=1)
        f["n_marked"] = (both < 2.0).sum(1).astype(np.float32)
    return {k: np.asarray(v, np.float32) for k, v in f.items()}, (vx, vy, sp)


def ball_features(fr):
    px, py = fr.px, fr.py
    bx, by = fr.bx, fr.by
    L, W = fr.pitch
    fps = fr.fps
    n = fr.n
    vis = ~np.isnan(bx)
    f = {}
    f["b_vis"] = vis.astype(np.float32)
    f["b_ax"], f["b_ay"] = np.abs(bx), np.abs(by)
    out = np.fmax(np.abs(bx) - L / 2, np.abs(by) - W / 2)        # > 0 outside the pitch
    f["b_out"] = out
    f["b_out_x"] = np.abs(bx) - L / 2
    f["b_out_y"] = np.abs(by) - W / 2
    # speed from neighbouring visible samples (1 and 3 frame baselines)
    v1x, v1y = _velocity(bx, 1, fps), _velocity(by, 1, fps)
    v3x, v3y = _velocity(bx, 3, fps), _velocity(by, 3, fps)
    sp1, sp3 = np.hypot(v1x, v1y), np.hypot(v3x, v3y)
    spd = np.where(np.isnan(sp3), sp1, sp3)
    f["b_speed"] = spd
    # outward velocity: component towards the nearest boundary
    ox = np.where(np.isnan(v3x), v1x, v3x) * np.sign(bx)
    oy = np.where(np.isnan(v3y), v1y, v3y) * np.sign(by)
    f["b_vout_x"], f["b_vout_y"] = ox, oy
    with np.errstate(invalid="ignore", divide="ignore"):
        tx = np.where(ox > 0.5, (L / 2 - np.abs(bx)) / ox, np.nan)
        ty = np.where(oy > 0.5, (W / 2 - np.abs(by)) / oy, np.nan)
    f["b_t_exit"] = np.clip(np.fmin(tx, ty), -2, 10)
    d = np.hypot(px - bx[:, None], py - by[:, None])
    dn = np.where(np.isnan(d), 1e3, d)
    ds = np.sort(dn, 1)[:, :2]
    f["b_d1"] = np.where(vis & (ds[:, 0] < 500), ds[:, 0], np.nan)
    f["b_d2"] = np.where(vis & (ds[:, 1] < 500), ds[:, 1], np.nan)
    f["b_n5"] = np.where(vis, (dn < 5).sum(1), np.nan)
    f["b_n15"] = np.where(vis, (dn < 15).sum(1), np.nan)
    f["b_corner"] = np.hypot(np.abs(bx) - L / 2, np.abs(by) - W / 2)
    f["b_centre"] = np.hypot(bx, by)
    f["b_goalarea"] = np.hypot(np.abs(bx) - (L / 2 - 5.5), np.fmax(np.abs(by) - 9.16, 0))
    # how long the ball has stayed within 0.7 m of where it is now (visible frames only)
    still = np.zeros(n, np.float32)
    for k in (5, 12, 25, 50, 100, 200):
        for sgn in (-1, 1):
            ok = np.hypot(_shift(bx, sgn * k) - bx, _shift(by, sgn * k) - by) < 0.7
            still += np.where(ok, 1.0, 0.0).astype(np.float32)
    f["b_still"] = np.where(vis, still, np.nan)
    for name, k in (("b_move_p1", -25), ("b_move_f1", 25), ("b_move_p3", -75), ("b_move_f3", 75)):
        f[name] = np.hypot(_shift(bx, k) - bx, _shift(by, k) - by)
    # gap-aware: defined on every frame
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
    jump = np.hypot(bx[ni] - bx[li], by[ni] - by[li])
    f["g_jump"] = np.where(has_l & has_n, jump, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["g_jump_speed"] = np.where(since + until > 0, jump / ((since + until) / fps + 1e-6),
                                     np.nan)
    # window summaries
    for w_s in (1, 3, 8):
        w = int(w_s * fps) | 1
        f[f"b_vis_{w_s}"] = uniform_filter1d(vis.astype(np.float32), w, mode="nearest")
        f[f"b_speed_{w_s}"] = _nanmean_filter(spd, w)
        f[f"b_out_{w_s}"] = _nanmean_filter(out, w)
    for k, name in ((-50, "p2"), (50, "f2")):
        w = 2 * fps + 1
        f["b_vis_" + name] = _shift(uniform_filter1d(vis.astype(np.float32), w, mode="nearest"),
                                    k // 2)
        f["b_speed_" + name] = _shift(_nanmean_filter(spd, w), k // 2)
    return {k: np.asarray(v, np.float32) for k, v in f.items()}


CONTEXT = ("sp_mean", "sp_med", "sp_p90", "sp_p25", "frac_still", "frac_walk", "frac_run",
           "acc_mean", "coherence", "flow_x", "half_sep", "box_max", "y_out", "n_out", "sx",
           "team_dx", "opp_mean", "n_marked", "x_p10_p90", "team_sp_min", "team_sp_max")


def context_features(f, fps=FPS):
    """Multi-scale temporal context: this is offline, so future frames are fair game."""
    out = {}
    for k in CONTEXT:
        a = f[k]
        for w_s in (1, 3, 8, 20):
            out[f"{k}_m{w_s}"] = _nanmean_filter(a, int(w_s * fps) | 1)
        m2 = _nanmean_filter(a, 2 * fps + 1)
        out[f"{k}_p2"] = _shift(m2, -fps)          # mean over the previous 2 s
        out[f"{k}_f2"] = _shift(m2, fps)           # mean over the next 2 s
        m6 = _nanmean_filter(a, 6 * fps + 1)
        out[f"{k}_p6"] = _shift(m6, -3 * fps)
        out[f"{k}_f6"] = _shift(m6, 3 * fps)
        out[f"{k}_d2"] = out[f"{k}_f2"] - out[f"{k}_p2"]
    return out


def build(fr, use_ball=True):
    """-> (X float32 (n, F), names)."""
    f, _ = player_features(fr)
    f.update(context_features(f, fr.fps))
    if use_ball:
        f.update(ball_features(fr))
    names = sorted(f)
    X = np.stack([f[k] for k in names], 1).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X, names
