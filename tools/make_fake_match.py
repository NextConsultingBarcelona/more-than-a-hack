"""A synthetic match written in the real file formats, with its own ground truth.

For tests and for anyone (including an LLM assistant) who must not look at the real data: the
output has the same layout as the gated bundle -- tactical ATD txt + metadata xml + Smart Tagging
xml -- so the whole pipeline runs on it unchanged.

    python tools/make_fake_match.py --out test_data/fake_match --seed 0

It is a toy, not a football simulator: 22 players on formation anchors, a ball that is carried,
passed, intercepted, tackled away and kicked out, restarts after stoppages, and the defects the
real tracking has -- ball drop-outs (more in flight and during stoppages), false ball detections,
position noise, and track slots that die and come back under another slot number.

Also written: truth.csv (frame,label) and fake_meta.json (n_frames and the graded blocks).
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, asset_paths  # noqa: E402

FPS = 25
N_SLOTS = 46
META_NAMES = ("FC Rovers", "Athletic Town")      # names in the tracking metadata
TAG_NAMES = ("Rovers", "Atletic Town")           # the tagging spells them differently, like the real one
BASE_X = np.array([0.06] + [0.22] * 4 + [0.40] * 4 + [0.58] * 2)
BASE_Y = np.array([0.5, 0.15, 0.38, 0.62, 0.85, 0.15, 0.38, 0.62, 0.85, 0.35, 0.65])


def simulate(seed=0, half_s=240.0, pitch=None):
    """Run the toy match. Returns a dict of arrays and tag instances (see write())."""
    rng = np.random.default_rng(seed)
    L, W = pitch or GAMES[GAME]["pitch"]
    dt = 1.0 / FPS
    team_of = np.array([0] * 11 + [1] * 11)

    def sec(s):
        return int(round(s * FPS))

    pre, ht, pre2, post = sec(15), sec(20), sec(14), sec(8)
    half = sec(half_s)
    k1 = pre                                   # first-half kick-off = first graded frame
    b2 = k1 + half + ht                        # second half is graded from here ...
    k2 = b2 + pre2                             # ... but kicks off here
    n = k2 + half + post
    blocks = [(k1, k1 + half - 1), (b2, k2 + half - 1)]

    P = np.zeros((22, 2))
    V = np.zeros((22, 2))
    pos = np.full((n, 22, 2), np.nan)
    spd = np.full((n, 22), np.nan)
    ball = np.full((n, 2), np.nan)
    ball_seen = np.zeros(n, dtype=bool)        # physically visible (before random drop-outs)
    in_flight = np.zeros(n, dtype=bool)
    truth = np.full(n, "X", dtype="<U1")
    restarts, kickoffs = [], []

    def anchors(direction, ball_x, attacking, kickoff=False):
        out = np.zeros((22, 2))
        for team in (0, 1):
            d = direction[team]
            bn = ball_x / L if d > 0 else 1.0 - ball_x / L
            x = BASE_X * L * 0.85
            if kickoff:
                x = np.minimum(BASE_X * L * 0.8, L / 2 - 2.5)
            else:
                shift = (bn - 0.5) * 0.55 * L + (5.0 if attacking == team else -3.0)
                scale = np.ones(11)
                scale[0] = 0.15
                x = np.clip(x + shift * scale, 2.0, L - 2.0)
            if d < 0:
                x = L - x
            out[team * 11:team * 11 + 11, 0] = x
            out[team * 11:team * 11 + 11, 1] = BASE_Y * W
        return out

    def step(targets, vmax):
        nonlocal P, V
        want = (targets - P) / 0.6
        norm = np.linalg.norm(want, axis=1)
        want *= (np.minimum(norm, vmax) / np.maximum(norm, 1e-9))[:, None]
        V += (want - V) * 0.15
        P += V * dt

    def wander(f0, f1, direction, gather_from=None, graded=False):
        """Not play: players drift (or walk to their kick-off spots from gather_from on)."""
        nonlocal P
        spots = anchors(direction, L / 2, None, kickoff=True)
        offs = rng.normal(0, 8.0, (22, 2))
        for f in range(f0, f1):
            if gather_from is not None and f >= gather_from:
                step(spots, np.full(22, 2.8))
            else:
                if (f - f0) % sec(4) == 0:
                    offs = rng.normal(0, 8.0, (22, 2))
                step(np.clip(spots + offs, 1.0, [L - 1.0, W - 1.0]), np.full(22, 1.3))
            pos[f], spd[f] = P, np.linalg.norm(V, axis=1)
            if gather_from is not None and f >= f1 - sec(3):
                ball[f] = (L / 2, W / 2)
                ball_seen[f] = True
            if graded:
                truth[f] = "D"

    def play(f0, f1, direction, kick_team):
        """One half of live play from the kick-off at f0 to f1 (exclusive)."""
        nonlocal P
        b = np.array([L / 2, W / 2])
        owner_team = kick_team
        carrier = kick_team * 11 + 9
        P[carrier] = b + (-0.5 * direction[kick_team], 0.0)
        mode, timer = "carry", sec(0.4)
        flight = None                          # (start, end, frames, elapsed, controller, goes_out)
        dead_until, spot, taker, exit_f = 0, None, None, 0

        def start_pass(f):
            nonlocal mode, flight, carrier
            me, team = carrier, team_of[carrier]
            d = direction[team]
            near_goal = (b[0] > L - 10) if d > 0 else (b[0] < 10)
            if rng.random() < (0.5 if near_goal else 0.07):
                side = rng.integers(3)
                if side == 0:
                    end = np.array([L + 5.0 if d > 0 else -5.0, rng.uniform(5, W - 5)])
                else:
                    end = np.array([np.clip(b[0] + d * rng.uniform(5, 25), 2, L - 2),
                                    W + 5.0 if side == 1 else -5.0])
                controller, goes_out = None, True
            else:
                mates = [i for i in range(22) if team_of[i] == team and i != me]
                dist = np.linalg.norm(P[mates] - b, axis=1)
                fwd = (P[mates, 0] - b[0]) * d
                wgt = np.exp(-((dist - 18.0) / 12.0) ** 2) * (1.0 + 0.5 * (fwd > 0)) + 1e-6
                controller = mates[rng.choice(len(mates), p=wgt / wgt.sum())]
                end = P[controller] + V[controller] * 0.4 + rng.normal(0, 1.0, 2)
                if rng.random() < 0.18:
                    opp = [i for i in range(22) if team_of[i] != team]
                    controller = opp[int(np.argmin(np.linalg.norm(P[opp] - end, axis=1)))]
                    end = P[controller] + rng.normal(0, 0.7, 2)
                end = np.clip(end, 1.0, [L - 1.0, W - 1.0])
                goes_out = False
            frames = max(3, int(np.linalg.norm(end - b) / rng.uniform(12, 20) * FPS))
            flight = [b.copy(), end, frames, 0, controller, goes_out]
            mode, carrier = "pass", None

        for f in range(f0, f1):
            targets = anchors(direction, float(np.clip(b[0], 0, L)), owner_team)
            vmax = np.full(22, 4.5)
            live = mode != "dead"
            if mode == "carry":
                team = team_of[carrier]
                goal = np.array([L if direction[team] > 0 else 0.0, W / 2])
                head = goal - P[carrier]
                head = head / max(np.linalg.norm(head), 1e-9)
                head = head + (0.0, 0.5 * np.sin(f / 17.0 + carrier))
                targets[carrier] = P[carrier] + head * 10.0
                vmax[carrier] = 4.8
                opp = np.flatnonzero(team_of != team)
                presser = opp[int(np.argmin(np.linalg.norm(P[opp] - b, axis=1)))]
                targets[presser], vmax[presser] = b, 6.5
            elif mode in ("pass", "loose") and flight[4] is not None:
                targets[flight[4]], vmax[flight[4]] = flight[1], 7.5
            elif mode == "dead":
                vmax[:] = 1.6
                targets[taker], vmax[taker] = spot, 3.5
            step(targets, vmax)

            if mode == "carry":
                v = V[carrier]
                b = P[carrier] + 0.6 * v / max(np.linalg.norm(v), 1e-9)
                timer -= 1
                if np.linalg.norm(P[presser] - b) < 1.2 and rng.random() < 0.015:
                    carrier, owner_team = presser, team_of[presser]
                    timer = sec(rng.uniform(1.0, 3.5))
                elif timer <= 0:
                    start_pass(f)
            elif mode == "pass":
                flight[3] += 1
                b = flight[0] + (flight[1] - flight[0]) * (flight[3] / flight[2])
                in_flight[f] = True
                outside = b[0] < 0 or b[0] > L or b[1] < 0 or b[1] > W
                if flight[5] and outside:
                    mode, live = "dead", False
                    exit_f = f
                    owner_team = 1 - owner_team
                    spot = np.clip(b, 0.3, [L - 0.3, W - 0.3])
                    mates = np.flatnonzero(team_of == owner_team)
                    taker = mates[int(np.argmin(np.linalg.norm(P[mates] - spot, axis=1)))]
                    wait = max(rng.uniform(6, 14), np.linalg.norm(P[taker] - spot) / 3.0 + 3.0)
                    dead_until = f + sec(wait)
                elif flight[3] >= flight[2] and not flight[5]:
                    mode = "loose"
            if mode == "loose":
                b = flight[1].copy()
                if np.linalg.norm(P[flight[4]] - b) < 1.0:
                    carrier, owner_team = flight[4], team_of[flight[4]]
                    mode, timer = "carry", sec(rng.uniform(1.0, 3.5))
            elif mode == "dead" and f > exit_f:
                live = False
                if f - exit_f <= sec(0.6):
                    b = b + (flight[1] - flight[0]) / flight[2]        # rolls on behind the line
                at_spot = np.linalg.norm(P[taker] - spot) < 1.0
                if f >= dead_until and at_spot:
                    carrier, mode, timer, live = taker, "carry", sec(0.3), True
                    b = spot.copy()
                    restarts.append(f)

            pos[f], spd[f] = P, np.linalg.norm(V, axis=1)
            if live:
                truth[f] = "HA"[owner_team]
                ball[f], ball_seen[f] = b, True
            else:
                truth[f] = "D"
                if f - exit_f <= sec(0.6):
                    ball[f], ball_seen[f] = b, True
                elif f >= dead_until - sec(2) and np.linalg.norm(P[taker] - spot) < 1.0:
                    ball[f], ball_seen[f] = spot, True

    P[:] = anchors({0: 1, 1: -1}, L / 2, None, kickoff=True) + rng.normal(0, 6.0, (22, 2))
    d1, d2 = {0: 1, 1: -1}, {0: -1, 1: 1}
    wander(0, k1, d1, gather_from=0)
    kickoffs.append((k1, 0, "1st Half"))
    play(k1, k1 + half, d1, kick_team=0)
    wander(k1 + half, b2, d2)
    wander(b2, k2, d2, gather_from=b2 + sec(2), graded=True)
    kickoffs.append((k2, 1, "2nd Half"))
    play(k2, k2 + half, d2, kick_team=1)
    wander(k2 + half, n, d2)

    # ---- what the tracker reports -------------------------------------------------------------
    obs_pos = pos + rng.normal(0, 0.08, pos.shape)
    obs_spd = np.abs(spd + rng.normal(0, 0.1, spd.shape))
    seen = np.zeros(n, dtype=bool)
    visible = True
    u = rng.random(n)
    for f in range(n):
        lose = 0.036 if in_flight[f] else 0.012
        visible = (u[f] >= lose) if visible else (u[f] < 0.03)
        seen[f] = visible and ball_seen[f]
    obs_ball = np.where(seen[:, None], ball + rng.normal(0, 0.12, ball.shape), np.nan)
    for f in np.flatnonzero(rng.random(n) < 0.002):            # false detections, 2 frames long
        obs_ball[f:f + 2] = rng.uniform((0, 0), (L, W))

    slot_team = np.array([0] * 23 + [1] * 23)[rng.permutation(N_SLOTS)]
    x = np.full((n, N_SLOTS), np.nan)
    y = np.full((n, N_SLOTS), np.nan)
    s = np.full((n, N_SLOTS), np.nan)
    free = {t: list(np.flatnonzero(slot_team == t)) for t in (0, 1)}
    slot = np.array([free[team_of[i]].pop(0) for i in range(22)])
    back_at = np.zeros(22, dtype=int)
    breaks = rng.random((n, 22)) < 1.0 / (40.0 * FPS)
    gaps = rng.integers(10, 21, (n, 22))
    for f in range(n):
        for i in np.flatnonzero(breaks[f]):
            if f >= back_at[i]:
                pool = free[team_of[i]]
                pool.append(slot[i])
                slot[i] = pool.pop(0)
                back_at[i] = f + gaps[f, i]
        on = back_at <= f
        x[f, slot[on]], y[f, slot[on]], s[f, slot[on]] = obs_pos[f, on, 0], obs_pos[f, on, 1], obs_spd[f, on]

    # ---- Smart Tagging ------------------------------------------------------------------------
    tags = []
    for frame, team, half_name in kickoffs:
        tags.append(("SET PIECES", frame - sec(2), frame + sec(3),
                     {"Team": TAG_NAMES[team], "Type": "Kick Off (Start)", "Half": half_name}))
    for frame in restarts:
        tags.append(("SET PIECES", frame - sec(2.5), frame + sec(2.5),
                     {"Team": "N/A", "Type": "Throw-in"}))
    change = np.flatnonzero(truth[1:] != truth[:-1]) + 1
    for a, b_ in zip(np.concatenate(([0], change)), np.concatenate((change, [n]))):
        if truth[a] in "HA" and b_ - a >= sec(5) and rng.random() < 0.7:
            team = "HA".index(truth[a])
            if rng.random() < 0.1:
                team = 1 - team
            tags.append(("PROGRESSION", a + sec(1), b_, {"Team": TAG_NAMES[team]}))
            tags.append(("DEFENDING IN MIDDLE THIRD", a + sec(1), b_, {"Team": TAG_NAMES[1 - team]}))
    tags.sort(key=lambda t: t[1])

    return {"x": x / L, "y": y / W, "s": s, "ball": obs_ball / (L, W), "slot_team": slot_team,
            "truth": truth, "blocks": blocks, "tags": tags, "n_frames": n,
            "kickoffs": [k[0] for k in kickoffs], "restarts": restarts}


def _group(a, b, c):
    if not np.isfinite(a):
        return "NaN,NaN,NaN"
    return "{:.5f},{:.5f},{:.3f}".format(a, b, c)


def write(sim, out_dir, game=GAME):
    """Write the bundle under out_dir in the real layout. Returns the paths used."""
    paths = asset_paths(game, out_dir)
    os.makedirs(os.path.dirname(paths["atd_raw"]), exist_ok=True)

    ids = ("FIFATMA", "FIFATMB")
    lines = ["<?xml version='1.0' encoding='utf-8'?>", "<MainMetadata><Metadata>", "<Teams>"]
    lines += ["<Team id=\"{}\"><Name>{}</Name></Team>".format(i, nm) for i, nm in zip(ids, META_NAMES)]
    lines += ["</Teams>", "<Players>"]
    for k, team in enumerate(sim["slot_team"]):
        lines.append("<Player id=\"P{0}\" teamId=\"{1}\"><Name>Track {0}</Name>"
                     "<ShirtNumber>{0}</ShirtNumber></Player>".format(k + 1, ids[team]))
    lines += ["</Players>", "</Metadata></MainMetadata>"]
    with open(paths["atd_meta"], "w") as handle:
        handle.write("\n".join(lines) + "\n")

    x, y, s, ball = sim["x"], sim["y"], sim["s"], sim["ball"]
    with open(paths["atd_raw"], "w") as handle:
        for f in range(sim["n_frames"]):
            players = ";".join(_group(x[f, k], y[f, k], s[f, k]) for k in range(x.shape[1]))
            handle.write("{}:{}:{}\n".format(f + 1, players, _group(ball[f, 0], ball[f, 1], 0.0)))

    rows = ["<?xml version='1.0' encoding='utf-8'?>", "<file><ALL_INSTANCES>"]
    for k, (code, a, b, labels) in enumerate(sim["tags"]):
        body = "".join("<label><group>{}</group><text>{}</text></label>".format(g, t)
                       for g, t in labels.items())
        rows.append("<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end><code>{}</code>{}"
                    "</instance>".format(k + 1, a / FPS, b / FPS, code, body))
    rows.append("</ALL_INSTANCES></file>")
    with open(paths["smart_tagging"], "w") as handle:
        handle.write("\n".join(rows) + "\n")

    truth_path = os.path.join(out_dir, "truth.csv")
    with open(truth_path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(sim["truth"]))
    meta_path = os.path.join(out_dir, "fake_meta.json")
    with open(meta_path, "w") as handle:
        json.dump({"n_frames": sim["n_frames"], "blocks": sim["blocks"],
                   "kickoffs": sim["kickoffs"]}, handle)
    return {"truth": truth_path, "meta": meta_path, **paths}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="directory to write the fake bundle into")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--half-seconds", type=float, default=240.0)
    args = parser.parse_args()
    if os.path.abspath(args.out).startswith(os.path.join(ROOT, "data")):
        raise SystemExit("refusing to write fake data into data/ -- that is where the real match lives")
    written = write(simulate(args.seed, args.half_seconds), args.out)
    print("fake match written to {}".format(args.out))
    print("truth: {}".format(written["truth"]))
