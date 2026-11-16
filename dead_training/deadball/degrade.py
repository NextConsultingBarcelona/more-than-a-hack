"""Make clean open tracking look like automatic tracking before training on it.

The open matches are manually corrected tracking: the ball is present on every live frame and
every player is tracked without gaps. The competition tracking is automatic: per the organiser's
public documentation the ball is missing on about 41 % of assessed frames (crowds, duels, balls
in the air), tracks drop out, and positions are noisier. A model trained on the clean version
would lean on signals automatic tracking does not have. No number here was measured on the
competition match.

What an automatic tracker shows while the ball is dead is not documented, and the open data do
not record it: Sportec freezes the ball where play stopped. Each stoppage therefore gets one of
several plausible pictures at random -- nothing, a ball lying at the restart point, a ball moved
there, a ball with the player taking the restart -- so the model cannot rely on any single one.
"""
import numpy as np

from .frames import Frames, runs


def _gaps(n, rng, rate_per_s, mean_len_s, fps):
    """Boolean mask of random gaps: Poisson starts, geometric lengths."""
    m = np.zeros(n, bool)
    k = rng.poisson(n / fps * rate_per_s)
    if k == 0:
        return m
    starts = rng.integers(0, n, k)
    lens = rng.geometric(1.0 / max(1.0, mean_len_s * fps), k)
    for s, l in zip(starts, lens):
        m[s:s + l] = True
    return m


def simulate_dead_ball(fr, rng, mode="mix"):
    """Ball positions a camera tracker might report during stoppages. Returns bx, by copies."""
    bx, by = fr.bx.copy(), fr.by.copy()
    fps = fr.fps
    n = fr.n
    p_exit = rng.uniform(0.4, 1.0)
    _, p_rest, p_move, p_carry = rng.dirichlet([1.0, 1.5, 0.6, 1.0])
    if mode != "mix":                    # stress tests: one picture for every stoppage
        p_exit = 0.0 if mode == "none" else p_exit
        p_rest, p_move, p_carry = {"none": (0, 0, 0), "rest": (1, 0, 0), "move": (0, 1, 0),
                                   "carry": (0, 0, 1)}[mode]
    for s, e in runs(fr.dead):
        L = e - s
        # 1. the ball keeps rolling for a moment after the whistle or the line
        if s >= 3 and not np.isnan(bx[s - 1]) and not np.isnan(bx[s - 3]) and rng.random() < p_exit:
            vx = (bx[s - 1] - bx[s - 3]) / 2.0
            vy = (by[s - 1] - by[s - 3]) / 2.0
            T = min(L, int(rng.uniform(0.2, 2.5) * fps))
            t = np.arange(1, T + 1)
            tau = rng.uniform(0.4, 1.5) * fps
            decay = tau * (1 - np.exp(-t / tau))
            bx[s:s + T] = bx[s - 1] + vx * decay
            by[s:s + T] = by[s - 1] + vy * decay
        if e >= n or np.isnan(fr.bx[e]):
            continue
        qx, qy = fr.bx[e], fr.by[e]
        u = rng.random()
        if u < p_rest:
            # 2a. the ball lies at the restart spot for the last part of the stoppage
            T = max(1, int(L * (rng.uniform(0.05, 1.0) if mode == 'mix' else 0.9)))
            bx[e - T:e], by[e - T:e] = qx, qy
        elif u < p_rest + p_move and L > 2 * fps and s >= 1 and not np.isnan(fr.bx[s - 1]):
            # 2b. the ball is carried or played back to the restart spot, then lies there
            rest = int(rng.uniform(0.2, 0.6) * L)
            speed = rng.uniform(1.5, 9.0) / fps
            dist = np.hypot(qx - fr.bx[s - 1], qy - fr.by[s - 1])
            T = int(min(L - rest, max(2, dist / speed)))
            a = e - rest - T
            w = np.linspace(0, 1, T, endpoint=False)
            bx[a:a + T] = fr.bx[s - 1] + w * (qx - fr.bx[s - 1])
            by[a:a + T] = fr.by[s - 1] + w * (qy - fr.by[s - 1])
            bx[e - rest:e], by[e - rest:e] = qx, qy
        elif u < p_rest + p_move + p_carry:
            # 2c. the ball is in the hands or at the feet of the player who will restart
            j = int(np.nanargmin(np.hypot(fr.px[e] - qx, fr.py[e] - qy)))
            T = max(1, int(L * (rng.uniform(0.2, 1.0) if mode == 'mix' else 0.9)))
            ox, oy = rng.normal(0, 0.4, 2)
            bx[e - T:e] = fr.px[e - T:e, j] + ox
            by[e - T:e] = fr.py[e - T:e, j] + oy
    return bx, by


def degrade(fr, seed, ball_dead=True, dead_mode="mix", dead_hide=None):
    """Copy of an open match made to look like automatic tracking. Deterministic in (match, seed)."""
    rng = np.random.default_rng(seed)
    n, S = fr.px.shape
    fps = fr.fps

    # ---- ball
    if ball_dead and fr.dead is not None:
        bx, by = simulate_dead_ball(fr, rng, dead_mode)
    else:
        bx, by = fr.bx.copy(), fr.by.copy()
    d = np.hypot(fr.px - bx[:, None], fr.py - by[:, None])
    d = np.where(np.isnan(d), 1e3, d)
    crowd = (d < 2.5).sum(1)
    bz = fr.meta.get("bz")
    if bz is not None:
        high = np.nan_to_num(bz) > rng.uniform(1.2, 2.5)
    else:
        sp = np.r_[0, np.hypot(np.diff(bx), np.diff(by))] * fps
        high = (np.nan_to_num(sp) > 14) & (d.min(1) > 4)
    base = rng.uniform(0.04, 0.2)
    p_hide = base + rng.uniform(0.3, 0.8) * high + rng.uniform(0.15, 0.5) * (crowd >= 2) \
        + rng.uniform(0.0, 0.25) * (crowd == 1)
    # hiding is bursty: draw it on a slow grid and hold it
    hold = max(1, int(rng.uniform(0.3, 1.2) * fps))
    u = np.repeat(rng.random(n // hold + 1), hold)[:n]
    hide = u < np.clip(p_hide, 0, 0.97)
    hide |= _gaps(n, rng, rng.uniform(0.02, 0.08), rng.uniform(1.0, 5.0), fps)
    if fr.dead is not None:
        hide |= fr.dead & (np.repeat(rng.random(n // (2 * fps) + 1), 2 * fps)[:n]
                           < (rng.uniform(0.0, 0.35) if dead_hide is None else dead_hide))
    sig = rng.uniform(0.05, 0.3)
    bx = bx + rng.normal(0, sig, n).astype(np.float32)
    by = by + rng.normal(0, sig, n).astype(np.float32)
    # false detections: the ball jumps onto something else for a few frames
    k = rng.poisson(n / fps * rng.uniform(0.0, 0.03))
    for s in rng.integers(0, n, k):
        l = int(rng.integers(1, 8))
        bx[s:s + l] = rng.uniform(-fr.pitch[0] / 2, fr.pitch[0] / 2)
        by[s:s + l] = rng.uniform(-fr.pitch[1] / 2, fr.pitch[1] / 2)
        hide[s:s + l] = False
    bx[hide] = np.nan
    by[hide] = np.nan

    # ---- players: correlated noise and track dropouts
    sig = rng.uniform(0.05, 0.25)
    rho = 0.9
    e = rng.normal(0, sig * np.sqrt(1 - rho ** 2), (n, S, 2)).astype(np.float32)
    from scipy.signal import lfilter
    noise = lfilter([1.0], [1.0, -rho], e, axis=0).astype(np.float32)
    px = fr.px + noise[..., 0]
    py = fr.py + noise[..., 1]
    rate, mlen = rng.uniform(0.01, 0.04), rng.uniform(1.0, 4.0)
    for s in range(S):
        g = _gaps(n, rng, rate, mlen, fps)
        px[g, s] = np.nan
        py[g, s] = np.nan
    return Frames(px=px.astype(np.float32), py=py.astype(np.float32), team=fr.team,
                  bx=bx.astype(np.float32), by=by.astype(np.float32), pitch=fr.pitch, fps=fps,
                  dead=fr.dead, owner=fr.owner, period=fr.period, name=fr.name,
                  meta={k: v for k, v in fr.meta.items() if k != "bz"})
