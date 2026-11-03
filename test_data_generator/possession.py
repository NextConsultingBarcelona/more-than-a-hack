"""Simulerer «den sanne kampen»: hvem som har ballen og hvor ballen er (nøkkelpunkter).

Resultat:
- segments: sammenhengende intervaller med etikett H/A (possession) eller D (død ball)
- loose:    løs-ball-vinduer (etiketten beholdes av laget som sist hadde ballen)
- keyframes: (frame, x_m, y_m) for ballen; ball.py interpolerer mellom dem

Possession-regler (fra arrangørens offentlige definisjon):
- I spill har alltid ett lag ballen; løs ball beholdes av laget som sist hadde den.
- Bytte skjer først når det andre laget har kontroll (slutten av løs-ball-vinduet).
- Død ball er alltid D. Hver restart har en tydelig eier.
"""

import numpy as np

from .config import FPS, PITCH_LENGTH as L, PITCH_WIDTH as W


def other(team):
    return "A" if team == "H" else "H"


def attack_dir(team, half):
    """+1 hvis laget angriper mot x = L, -1 mot x = 0. Hjemmelaget angriper +x i 1. omgang."""
    d = 1 if team == "H" else -1
    return d if half == 1 else -d


def _sec(rng, lo_hi):
    lo, hi = lo_hi
    return int(round(rng.uniform(lo, hi) * FPS))


def _clip_pitch(p, margin=2.0):
    return np.array([np.clip(p[0], margin, L - margin), np.clip(p[1], margin, W - margin)])


def _walk(rng, params, t0, t1, pos, d, keyframes):
    """Ballen går mellom berøringer (pasning/dribling) fra t0 til t1. Returnerer posisjon ved t1."""
    t = t0
    while t < t1:
        gap = _sec(rng, params.touch_gap_s)
        t_next = min(t + max(gap, 1), t1)
        frac = (t_next - t) / max(gap, 1)
        step = np.array([d * rng.normal(4.0, 9.0), rng.normal(0.0, 10.0)]) * frac
        pos = _clip_pitch(pos + step)
        keyframes.append((t_next, pos[0], pos[1]))
        t = t_next
    return pos


def _possession_frames(rng, params):
    s = rng.lognormal(np.log(params.possession_median_s), params.possession_sigma)
    s = float(np.clip(s, params.possession_min_s, params.possession_max_s))
    return int(round(s * FPS))


def _choose_out(rng, params, pos, team, d):
    """Velg type død ball, eier, punkt der ballen går ut og restart-punkt."""
    goal_line_x = L if d > 0 else 0.0
    near_byline = abs(goal_line_x - pos[0]) <= params.byline_reach_m

    if params.p_goal > 0 and near_byline and rng.random() < params.p_goal:
        out_pt = np.array([goal_line_x + 0.5 * d, W / 2])
        return "kickoff_after_goal", other(team), out_pt, np.array([L / 2, W / 2])

    probs = np.array([params.p_throw_in, params.p_byline if near_byline else 0.0, params.p_free_kick])
    kind = ["throw_in", "byline", "free_kick"][rng.choice(3, p=probs / probs.sum())]

    if kind == "throw_in":
        last_touch = team if rng.random() < 0.5 else other(team)
        side_y = 0.0 if pos[1] < W / 2 else W
        out_pt = np.array([pos[0], side_y + (-0.5 if side_y == 0 else 0.5)])
        spot = np.array([pos[0], side_y + (-0.3 if side_y == 0 else 0.3)])
        return "throw_in", other(last_touch), out_pt, spot

    if kind == "byline":
        out_y = float(np.clip(pos[1] + rng.normal(0, 8), 1.0, W - 1.0))
        out_pt = np.array([goal_line_x + 0.5 * d, out_y])
        if rng.random() < 0.6:  # angriper siste berøring -> utspark til forsvarer
            spot = np.array([goal_line_x - 5.5 * d, W / 2 + rng.uniform(-9.0, 9.0)])
            return "goal_kick", other(team), out_pt, spot
        corner_y = 0.0 if out_y < W / 2 else W  # forsvarer siste berøring -> corner
        return "corner", team, out_pt, np.array([goal_line_x, corner_y])

    owner = team if rng.random() < 0.6 else other(team)  # laget som ble felt
    return "free_kick", owner, pos.copy(), pos.copy()


def _dead_frames(rng, params, restart):
    ranges = {
        "throw_in": params.dead_throw_in_s,
        "goal_kick": params.dead_goal_kick_s,
        "corner": params.dead_corner_s,
        "free_kick": params.dead_free_kick_s,
        "kickoff_after_goal": params.dead_kickoff_after_goal_s,
    }
    return _sec(rng, ranges[restart])


def _simulate_half(rng, params, half, kickoff, end, kick_team, segments, loose, keyframes):
    t = kickoff
    team = kick_team
    pos = np.array([L / 2, W / 2])
    restart = "kickoff"
    keyframes.append((t, pos[0], pos[1]))

    while t < end:
        dur = _possession_frames(rng, params)
        p_end = t + dur
        if p_end >= end:
            p_end, reason = end, "half_end"
        else:
            p = np.array([params.p_turnover, params.p_out])
            reason = ["turnover", "out"][rng.choice(2, p=p / p.sum())]
        d = attack_dir(team, half)
        dur = p_end - t

        # Løs ball som samme lag vinner tilbake (ingen etikettbytte)
        # (plassert slik at den aldri overlapper et eventuelt løs-ball-vindu på slutten)
        if dur >= 10 * FPS and rng.random() < params.p_loose_kept:
            lw = _sec(rng, (params.loose_min_s, params.loose_max_s))
            a = t + int(rng.integers(2 * FPS, dur - 7 * FPS))
            loose.append(dict(half=half, start=a, end=a + lw, label=team, kind="loose_kept",
                              detail="", owner=team))

        segments.append(dict(half=half, start=t, end=p_end, label=team, kind="possession",
                             detail=f"{restart}->{reason}", owner=team))

        if reason == "half_end":
            pos = _walk(rng, params, t, p_end, pos, d, keyframes)
            break

        if reason == "turnover":
            lw = min(_sec(rng, (params.loose_min_s, params.loose_max_s)), max(dur - FPS, 1))
            pos = _walk(rng, params, t, p_end - lw, pos, d, keyframes)
            pos = _clip_pitch(pos + rng.normal(0, 6, size=2))  # styrt/blokkert ball
            keyframes.append((p_end, pos[0], pos[1]))
            loose.append(dict(half=half, start=p_end - lw, end=p_end, label=team,
                              kind="loose_turnover", detail="", owner=other(team)))
            team, restart, t = other(team), "turnover", p_end
            continue

        # reason == "out": ballen går ut / blåses av -> D
        leg = min(int(params.final_leg_s * FPS), max(dur - FPS, 1))
        pos = _walk(rng, params, t, p_end - leg, pos, d, keyframes)
        restart, owner, out_pt, spot = _choose_out(rng, params, pos, team, d)
        keyframes.append((p_end, out_pt[0], out_pt[1]))
        dead_end = min(p_end + _dead_frames(rng, params, restart), end)
        segments.append(dict(half=half, start=p_end, end=dead_end, label="D", kind="dead",
                             detail=restart, owner=owner))
        if dead_end - p_end > FPS:
            keyframes.append((p_end + FPS, spot[0], spot[1]))
        keyframes.append((dead_end, spot[0], spot[1]))
        team, pos, t = owner, spot.copy(), dead_end


def simulate_match(timeline, params, rng):
    segments, loose, keyframes = [], [], []
    first = "H" if rng.random() < 0.5 else "A"
    for half, eval_start, kickoff, end in timeline.halves():
        kick_team = first if half == 1 else other(first)
        if kickoff > eval_start:
            segments.append(dict(half=half, start=eval_start, end=kickoff, label="D", kind="dead",
                                 detail="before_kickoff", owner=kick_team))
            keyframes.append((eval_start, L / 2, W / 2))
        _simulate_half(rng, params, half, kickoff, end, kick_team, segments, loose, keyframes)
    return segments, loose, keyframes
