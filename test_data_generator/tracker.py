"""«Måleutstyret»: gjør den sanne kampen om til det trackingen ville sett.

- Ballen mangler ca. 40 % av vurderte frames, oftere i dueller og nær mål.
- Spillere forsvinner kort (dropout) og dukker opp igjen på en NY slot av samme lag.
  Slots frigjøres og gjenbrukes (FIFO), så slots dukker opp, forsvinner og bytter identitet.
- Av og til en duplikat-slot (samme spiller sett to ganger) noen sekunder.
- 46 slots totalt (23 per lag). Ubrukte slots er NaN.

v2: identitetsbytter mellom spillere, støy i posisjoner, slots som hopper mellom lag.
"""

from collections import deque

import numpy as np

from .config import FPS, N_SLOTS, PITCH_LENGTH as L, PLAYERS_PER_TEAM, SLOTS_PER_TEAM


def ball_missing_mask(ball, players, eval_mask, params, rng):
    """Bool-maske: True der ballen IKKE er sett. Kalibrerer startrate mot målandelen."""
    n = len(ball)
    with np.errstate(invalid="ignore"):
        dist = np.hypot(players[..., 0] - ball[:, None, 0], players[..., 1] - ball[:, None, 1])
        d_home = np.nanmin(np.where(np.isnan(dist[:, :PLAYERS_PER_TEAM]), np.inf, dist[:, :PLAYERS_PER_TEAM]), axis=1)
        d_away = np.nanmin(np.where(np.isnan(dist[:, PLAYERS_PER_TEAM:]), np.inf, dist[:, PLAYERS_PER_TEAM:]), axis=1)
        duel = (d_home < params.duel_radius_m) & (d_away < params.duel_radius_m)
        bx = ball[:, 0]
        near_goal = (bx < params.near_goal_frac * L) | (bx > (1 - params.near_goal_frac) * L)
    duel &= eval_mask
    near_goal &= eval_mask

    weight = 1.0 + params.ball_duel_weight * duel + params.ball_goal_weight * near_goal
    u = rng.random(n)
    lo, hi = params.ball_gap_s
    lengths = rng.integers(int(lo * FPS), int(hi * FPS) + 1, size=n)

    def build(p0):
        m = np.zeros(n, bool)
        cur_end = -1
        for s in np.flatnonzero(eval_mask & (u < p0 * weight)):
            if s < cur_end:
                continue
            cur_end = s + lengths[s]
            m[s:cur_end] = True
        return m

    target = params.ball_missing_target
    a, b = 0.0, 0.2
    for _ in range(30):  # bisection, deterministisk
        mid = 0.5 * (a + b)
        if build(mid)[eval_mask].mean() < target:
            a = mid
        else:
            b = mid
    mask = build(0.5 * (a + b)) | ~eval_mask
    return mask, duel, near_goal


def _poisson_intervals(rng, n, rate_per_min, dur_s):
    """Ikke-overlappende intervaller (start, slutt) med gitt rate og varighet."""
    out = []
    if rate_per_min <= 0:
        return out
    mean_gap = 60.0 * FPS / rate_per_min
    t = int(rng.exponential(mean_gap))
    while t < n:
        d = int(rng.integers(int(dur_s[0] * FPS), int(dur_s[1] * FPS) + 1))
        out.append((t, min(t + d, n)))
        t = t + d + int(rng.exponential(mean_gap)) + 1
    return out


def assign_slots(players, speed, params, rng):
    """Returnerer slot_xy (n, 46, 2) meter, slot_s (n, 46) og lag per slot (liste med 'H'/'A')."""
    n = players.shape[0]
    slot_xy = np.full((n, N_SLOTS, 2), np.nan)
    slot_s = np.full((n, N_SLOTS), np.nan)
    slot_team = ["H"] * SLOTS_PER_TEAM + ["A"] * SLOTS_PER_TEAM

    for t_idx in range(2):
        team_slots = list(range(t_idx * SLOTS_PER_TEAM, (t_idx + 1) * SLOTS_PER_TEAM))
        team_players = list(range(t_idx * PLAYERS_PER_TEAM, (t_idx + 1) * PLAYERS_PER_TEAM))

        # Hendelser: (frame, rekkefølge, type, spiller, slutt)
        events = []
        for p in team_players:
            for s, e in _poisson_intervals(rng, n, params.player_dropout_per_min, params.player_dropout_s):
                events.append((s, 0, "drop", p, e))
        for s, e in _poisson_intervals(rng, n, params.duplicate_per_min, params.duplicate_s):
            events.append((s, 1, "dup", int(rng.choice(team_players)), e))
        events.sort()

        free = deque(team_slots[PLAYERS_PER_TEAM:])
        current = {p: team_slots[i] for i, p in enumerate(team_players)}
        since = {p: 0 for p in team_players}
        pending = []  # (frame, type, spiller/slot)

        def flush(upto):
            pending.sort()
            while pending and pending[0][0] <= upto:
                f, kind, a, b = pending.pop(0)
                if kind == "back":  # spiller tilbake på ny slot
                    slot = free.popleft()
                    current[a] = slot
                    since[a] = f
                else:  # duplikat ferdig
                    free.append(b)

        for f, _, kind, p, e in events:
            flush(f)
            if kind == "drop":
                if current[p] is None:
                    continue
                slot = current[p]
                slot_xy[since[p]:f, slot] = players[since[p]:f, p]
                slot_s[since[p]:f, slot] = speed[since[p]:f, p]
                free.append(slot)
                current[p] = None
                pending.append((e, "back", p, None))
            else:  # duplikat
                if not free:
                    continue
                slot = free.popleft()
                offset = rng.normal(0, 0.7, size=2)
                slot_xy[f:e, slot] = players[f:e, p] + offset
                slot_s[f:e, slot] = speed[f:e, p]
                pending.append((e, "dupend", p, slot))
        flush(n)
        for p in team_players:
            if current[p] is not None:
                slot = current[p]
                slot_xy[since[p]:, slot] = players[since[p]:, p]
                slot_s[since[p]:, slot] = speed[since[p]:, p]

    return slot_xy, slot_s, slot_team
