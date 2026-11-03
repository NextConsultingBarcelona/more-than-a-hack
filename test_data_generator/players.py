"""Sanne spillerposisjoner (meter) for 22 spillere. Indeks 0-10 = H, 11-21 = A.

Enkel modell, simulert ved sim_hz (5 Hz) og interpolert til 25 fps:
- Hver spiller sikter mot sin formasjonsposisjon, forskjøvet etter ballen.
- Laget med ballen skyver fram, laget uten trekker sammen.
- Nærmeste spiller på ballførende lag går til ballen, nærmeste motspiller presser.
- Fartsgrense og glatting gir realistisk fart (maks ca. 8 m/s).
- Lagene bytter side i 2. omgang.
"""

import numpy as np

from .config import FPS, PITCH_LENGTH as L, PITCH_WIDTH as W, PLAYERS_PER_TEAM

# Formasjoner for et lag som angriper mot +x (egen keeper ved x = 0)
_F442 = [(5, 34), (20, 10), (18, 26), (18, 42), (20, 58),
         (38, 10), (36, 26), (36, 42), (38, 58), (48, 28), (48, 40)]
_F433 = [(5, 34), (20, 10), (18, 26), (18, 42), (20, 58),
         (35, 18), (33, 34), (35, 50), (48, 12), (49, 34), (48, 56)]

LABEL_CODE = {"H": 0, "A": 1, "D": 2, "X": 3}


def _formation(half):
    home = np.array(_F442, dtype=float)
    away = np.array(_F433, dtype=float)
    if half == 1:  # H angriper +x, A angriper -x
        away[:, 0] = L - away[:, 0]
    else:
        home[:, 0] = L - home[:, 0]
    return np.vstack([home, away])


def true_players(timeline, labels, owners, ball, params, rng):
    """labels: kode per frame (0 H, 1 A, 2 D, 3 X). owners: eier ved D (0/1, -1 ukjent)."""
    n = timeline.n_frames
    step = FPS // params.sim_hz
    dt = 1.0 / params.sim_hz
    sample = np.arange(0, n, step)
    if sample[-1] != n - 1:
        sample = np.r_[sample, n - 1]

    form = {1: _formation(1), 2: _formation(2)}
    team_of = np.r_[np.zeros(PLAYERS_PER_TEAM, int), np.ones(PLAYERS_PER_TEAM, int)]
    is_gk = np.zeros(2 * PLAYERS_PER_TEAM, bool)
    is_gk[[0, PLAYERS_PER_TEAM]] = True
    # retning for hvert lag: +1 / -1
    dirs = {1: np.array([1.0, -1.0]), 2: np.array([-1.0, 1.0])}

    pos = form[1].copy()
    vel = np.zeros_like(pos)
    jitter = np.zeros_like(pos)
    out = np.empty((len(sample), 2 * PLAYERS_PER_TEAM, 2))
    centre = np.array([L / 2, W / 2])

    for i, f in enumerate(sample):
        half = 1 if f <= timeline.h1_end else 2
        lab = labels[f]
        b = ball[f] if not np.isnan(ball[f, 0]) else centre
        jitter = 0.95 * jitter + rng.normal(0, 0.6, size=jitter.shape)
        vmax = np.full(len(pos), params.vmax_player)

        if lab == 3:  # utenfor spill: rolig mot formasjon
            target = form[half].copy()
            vmax[:] = params.vmax_idle
        else:
            base = form[half]
            shift = np.where(is_gk[:, None], [0.15, 0.2], [0.55, 0.3]) * (b - centre)
            target = base + shift + jitter
            if lab in (0, 1):
                d = dirs[half]
                push = np.where(team_of == lab, 6.0, -4.0) * d[team_of]
                target[:, 0] += np.where(is_gk, 0.0, push)
            target[:, 0] = np.clip(target[:, 0], 1.0, L - 1.0)
            target[:, 1] = np.clip(target[:, 1], 1.0, W - 1.0)

            chase_team = lab if lab in (0, 1) else owners[f]
            if chase_team in (0, 1):
                dist = np.hypot(*(pos - b).T)
                own = np.flatnonzero(team_of == chase_team)
                c = own[np.argmin(dist[own])]
                target[c] = b
                vmax[c] = params.vmax_chaser
                if lab in (0, 1):  # bare press når ballen er i spill
                    opp = np.flatnonzero(team_of != chase_team)
                    o = opp[np.argmin(dist[opp])]
                    target[o] = b + np.array([1.5, 0.0]) * -dirs[half][team_of[o]]
                    vmax[o] = params.vmax_chaser

        delta = target - pos
        dist = np.hypot(*delta.T)[:, None]
        want = np.where(dist > 1e-9, delta / np.maximum(dist, 1e-9), 0.0) * np.minimum(dist / 1.0, vmax[:, None])
        vel = 0.5 * vel + 0.5 * want
        speed = np.hypot(*vel.T)[:, None]
        vel = np.where(speed > vmax[:, None], vel * vmax[:, None] / np.maximum(speed, 1e-9), vel)
        pos = pos + vel * dt
        out[i] = pos

    # Interpoler til 25 fps
    frames = np.arange(n)
    full = np.empty((n, 2 * PLAYERS_PER_TEAM, 2))
    for p in range(2 * PLAYERS_PER_TEAM):
        for k in range(2):
            full[:, p, k] = np.interp(frames, sample, out[:, p, k])

    # Fart (m/s) fra posisjonsendring, glattet over 5 frames
    d = np.diff(full, axis=0, prepend=full[:1])
    speed = np.hypot(d[..., 0], d[..., 1]) * FPS
    kernel = np.ones(5) / 5
    for p in range(speed.shape[1]):
        speed[:, p] = np.convolve(speed[:, p], kernel, mode="same")
    return full, speed
