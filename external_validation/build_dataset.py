"""Bygg valideringsdata fra ÅPNE Sportec/IDSSE-kamper (CC-BY 4.0).

Steg:
  1. Last inn kampen med kloppy (only_alive=False).
  2. Lag fasit per frame: D = død ball, ellers H/A etter ball-eiende lag (hjemmelag = H).
  3. Gjør dataene dårligere så de ligner konkurransedataene (fast seed).
  4. Skriv ext_data/<kamp>/tracking.parquet, slots.csv, gt.csv og info.json.

Kjør fra repo-mappen:
  python external_validation/build_dataset.py                 # én kamp (J03WPY)
  python external_validation/build_dataset.py --matches all   # alle 7
  python external_validation/build_dataset.py --selftest      # kunstige data, ingen nedlasting

Kilde: Bassek et al. (2025), Scientific Data, doi:10.1038/s41597-025-04505-y
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Innlasting
# ---------------------------------------------------------------------------
def load_sportec(match_id):
    """Last en Sportec-kamp og gjør den om til enkle numpy-tabeller ("clean")."""
    from kloppy import sportec

    print("  Laster {} med kloppy (første gang lastes filen ned, kan ta noen minutter) ...".format(match_id))
    ds = sportec.load_open_tracking_data(match_id=match_id, only_alive=False, coordinates="kloppy")
    meta = ds.metadata
    pd_ = meta.pitch_dimensions
    length = getattr(pd_, "pitch_length", None) or 105.0
    width = getattr(pd_, "pitch_width", None) or 68.0

    home = [t for t in meta.teams if str(t.ground).lower().endswith("home")][0]

    frames = sorted(ds.frames, key=lambda f: (f.period.id, f.frame_id))
    n = len(frames)

    player_ids, player_team, pidx = [], [], {}
    for team in meta.teams:
        for p in team.players:
            pidx[p.player_id] = len(player_ids)
            player_ids.append(p.player_id)
            player_team.append("H" if team.team_id == home.team_id else "A")

    period = np.empty(n, dtype=np.int8)
    alive = np.zeros(n, dtype=bool)
    owner = np.empty(n, dtype=object)
    ball = np.full((n, 2), np.nan)
    px = np.full((n, len(player_ids)), np.nan)
    py = np.full((n, len(player_ids)), np.nan)

    for i, f in enumerate(frames):
        period[i] = f.period.id
        alive[i] = str(f.ball_state).lower().endswith("alive")
        owner[i] = "H" if (f.ball_owning_team is not None and f.ball_owning_team.team_id == home.team_id) else "A"
        if f.ball_coordinates is not None:
            ball[i] = (f.ball_coordinates.x, f.ball_coordinates.y)
        for player, pdata in f.players_data.items():
            j = pidx.get(player.player_id)
            if j is not None and pdata.coordinates is not None:
                px[i, j] = pdata.coordinates.x
                py[i, j] = pdata.coordinates.y

    return dict(period=period, alive=alive, owner=owner, ball=ball, px=px, py=py,
                player_team=np.array(player_team), length=float(length), width=float(width),
                source="Sportec/IDSSE {} (CC-BY 4.0)".format(match_id))


def make_selftest(match_id, minutes_per_half=4.0):
    """Kunstige data i samme form som load_sportec - for å teste uten nedlasting."""
    rng = np.random.default_rng(zlib.crc32(match_id.encode()))
    n_half = int(minutes_per_half * 60 * C.FPS)
    n = 2 * n_half
    period = np.repeat(np.array([1, 2], dtype=np.int8), n_half)
    n_players = 26  # 22 + noen innbyttere
    team = np.array(["H"] * 13 + ["A"] * 13)
    # Tilfeldig gange for spillerne
    steps = rng.normal(0, 0.0015, size=(n, n_players, 2))
    pos = np.clip(rng.uniform(0.1, 0.9, size=(1, n_players, 2)) + np.cumsum(steps, axis=0), 0, 1)
    px, py = pos[:, :, 0].copy(), pos[:, :, 1].copy()
    # Innbyttere: spiller 11-12 og 24-25 er bare med i 2. omgang
    for j in (11, 12, 24, 25):
        px[:n_half, j] = np.nan
        py[:n_half, j] = np.nan
    for j in (9, 10, 22, 23):
        px[n_half:, j] = np.nan
        py[n_half:, j] = np.nan
    # Possessions på 5-30 s med dødballer innimellom
    alive = np.zeros(n, dtype=bool)
    owner = np.empty(n, dtype=object)
    ball = np.full((n, 2), np.nan)
    t, current = 0, "H"
    while t < n:
        length = int(rng.uniform(5, 30) * C.FPS)
        stop = min(t + length, n)
        is_dead = rng.random() < 0.25
        alive[t:stop] = not is_dead
        owner[t:stop] = current
        cand = np.flatnonzero((team == current) & ~np.isnan(px[t]))
        carrier = rng.choice(cand) if len(cand) else 0
        ball[t:stop, 0] = np.nan_to_num(px[t:stop, carrier], nan=0.5)
        ball[t:stop, 1] = np.nan_to_num(py[t:stop, carrier], nan=0.5)
        if rng.random() < 0.6:
            current = "A" if current == "H" else "H"
        t = stop
    return dict(period=period, alive=alive, owner=owner, ball=ball, px=px, py=py,
                player_team=team, length=105.0, width=68.0,
                source="SELFTEST (kunstige data)")


# ---------------------------------------------------------------------------
# 2. Fasit
# ---------------------------------------------------------------------------
def layout(clean):
    """Plasser omgangene i én tidslinje med X-rammer rundt: pre | 1. omg | pause | 2. omg | post.

    Returnerer (total lengde, indekser i den nye tidslinjen for hver rad i clean)."""
    period = clean["period"]
    idx = np.empty(len(period), dtype=np.int64)
    pos = C.PRE_FRAMES
    periods = sorted(set(period.tolist()))
    for k, p in enumerate(periods):
        rows = np.flatnonzero(period == p)
        idx[rows] = pos + np.arange(len(rows))
        pos += len(rows)
        if k < len(periods) - 1:
            pos += C.HALFTIME_FRAMES
    return pos + C.POST_FRAMES, idx


def make_gt(clean, n_total, idx):
    labels = np.full(n_total, "X", dtype=object)
    labels[idx] = np.where(clean["alive"], clean["owner"], "D")
    return pd.DataFrame({"frame": np.arange(n_total), "label": labels})


# ---------------------------------------------------------------------------
# 3. Gjør dataene dårligere
# ---------------------------------------------------------------------------
def _runs(mask):
    """[(start, stop)] for sammenhengende True-sekvenser."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _speed(x, y, period, length, width):
    """Fart i m/s fra posisjon (0-1), regnet innenfor hver omgang og glattet."""
    out = np.full(x.shape, np.nan)
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        dx = np.diff(x[rows], axis=0) * length
        dy = np.diff(y[rows], axis=0) * width
        v = np.sqrt(dx ** 2 + dy ** 2) * C.FPS
        v = np.vstack([v[:1], v])
        v = pd.DataFrame(v).rolling(C.SPEED_SMOOTH_FRAMES, center=True, min_periods=1).mean().values
        out[rows] = v
    return out


def remove_ball(clean, rng):
    """Fjern ballen i blokker til ca. BALL_MISSING_TARGET mangler. Oftere i duell og nær mål."""
    bx, by = clean["ball"][:, 0], clean["ball"][:, 1]
    L, W = clean["length"], clean["width"]
    team = clean["player_team"]
    dist = np.sqrt(((clean["px"] - bx[:, None]) * L) ** 2 + ((clean["py"] - by[:, None]) * W) ** 2)
    dist = np.where(np.isnan(dist), np.inf, dist)
    near_h = dist[:, team == "H"].min(axis=1) < C.DUEL_RADIUS_M
    near_a = dist[:, team == "A"].min(axis=1) < C.DUEL_RADIUS_M
    duel = near_h & near_a
    goal_dist = np.minimum(np.hypot(bx * L, (by - 0.5) * W), np.hypot((1 - bx) * L, (by - 0.5) * W))
    near_goal = np.nan_to_num(goal_dist, nan=np.inf) < C.NEAR_GOAL_M

    weight = 1.0 + (C.DUEL_WEIGHT - 1.0) * duel + (C.NEAR_GOAL_WEIGHT - 1.0) * near_goal
    prob = weight / weight.sum()

    n = len(bx)
    period = clean["period"]
    period_end = np.empty(n, dtype=np.int64)
    for p in np.unique(period):
        rows = np.flatnonzero(period == p)
        period_end[rows] = rows[-1] + 1

    missing = np.isnan(bx).copy()
    target = int(round(C.BALL_MISSING_TARGET * n))
    lo, hi = (int(s * C.FPS) for s in C.BALL_GAP_SECONDS)
    while missing.sum() < target:
        starts = rng.choice(n, size=200, p=prob)
        lengths = rng.integers(lo, hi + 1, size=200)
        for s, ln in zip(starts, lengths):
            e = min(s + ln, period_end[s])
            missing[s:e] = True
            if missing.sum() >= target:
                break
    ball = clean["ball"].copy()
    ball[missing] = np.nan
    return ball, float(missing.mean()), float(duel.mean()), float(near_goal.mean())


def assign_slots(clean, rng):
    """Anonyme slots: hull i sporene, og noen ganger ny slot etter et hull."""
    px, py, period = clean["px"], clean["py"], clean["period"]
    n, n_players = px.shape
    L, W = clean["length"], clean["width"]
    speed = _speed(px, py, period, L, W)

    # Fordel slotene tilfeldig mellom lagene (lag per slot er fast, som i metadata)
    order = rng.permutation(C.N_SLOTS)
    half = C.N_SLOTS // 2
    slot_team = np.empty(C.N_SLOTS, dtype=object)
    slot_team[order[:half]] = "H"
    slot_team[order[half:]] = "A"

    # Lag segmenter (sammenhengende tilstedeværelse) etter å ha lagt inn tilfeldige hull
    segments = []
    lo, hi = (int(s * C.FPS) for s in C.PLAYER_GAP_SECONDS)
    for j in range(n_players):
        present = ~np.isnan(px[:, j])
        n_present = int(present.sum())
        if n_present == 0:
            continue
        n_gaps = rng.poisson(C.PLAYER_GAPS_PER_MINUTE * n_present / (60 * C.FPS))
        if n_gaps:
            starts = rng.choice(np.flatnonzero(present), size=n_gaps, replace=False)
            for s in starts:
                present[s:s + rng.integers(lo, hi + 1)] = False
        # Del også ved omgangsskifte
        boundary = np.concatenate(([False], np.diff(period) != 0))
        for s, e in _runs(present):
            cuts = [s] + [k for k in np.flatnonzero(boundary[s:e]) + s if k > s] + [e]
            for a, b in zip(cuts[:-1], cuts[1:]):
                segments.append((a, b, j))
    segments.sort()

    busy = np.zeros((C.N_SLOTS, n), dtype=bool)
    last_slot = {}
    out_x = np.full((n, C.N_SLOTS), np.nan)
    out_y = np.full((n, C.N_SLOTS), np.nan)
    out_s = np.full((n, C.N_SLOTS), np.nan)
    dropped = switches = 0
    for s, e, j in segments:
        team = clean["player_team"][j]
        prev = last_slot.get(j)
        keep = prev is not None and rng.random() >= C.SLOT_SWITCH_PROB and not busy[prev, s:e].any()
        if keep:
            slot = prev
        else:
            free = [k for k in np.flatnonzero(slot_team == team) if not busy[k, s:e].any()]
            if not free:
                dropped += 1
                continue
            slot = int(rng.choice(free))
            if prev is not None and slot != prev:
                switches += 1
        busy[slot, s:e] = True
        last_slot[j] = slot
        out_x[s:e, slot] = px[s:e, j]
        out_y[s:e, slot] = py[s:e, j]
        out_s[s:e, slot] = speed[s:e, j]
    stats = dict(segments=len(segments), slot_switches=switches, dropped_segments=dropped,
                 mean_active_slots=float((~np.isnan(out_x)).sum(axis=1).mean()))
    return out_x, out_y, out_s, slot_team, stats


# ---------------------------------------------------------------------------
# 4. Output
# ---------------------------------------------------------------------------
def build_match(match_id, selftest=False):
    t0 = time.time()
    clean = make_selftest(match_id) if selftest else load_sportec(match_id)
    rng = np.random.default_rng([C.SEED, zlib.crc32(match_id.encode())])  # likt på alle maskiner

    n_total, idx = layout(clean)
    gt = make_gt(clean, n_total, idx)
    ball, ball_missing, duel_share, goal_share = remove_ball(clean, rng)
    sx, sy, ss, slot_team, slot_stats = assign_slots(clean, rng)

    cols = {"frame_id": np.arange(1, n_total + 1)}
    for name, values in (("ball_x", ball[:, 0]), ("ball_y", ball[:, 1])):
        full = np.full(n_total, np.nan, dtype=np.float32)
        full[idx] = values
        cols[name] = full
    for k in range(C.N_SLOTS):
        name = C.SLOT_NAME.format(k)
        for suffix, arr in (("_x", sx), ("_y", sy), ("_s", ss)):
            full = np.full(n_total, np.nan, dtype=np.float32)
            full[idx] = arr[:, k]
            cols[name + suffix] = full
    tracking = pd.DataFrame(cols)
    slots = pd.DataFrame({"slot": [C.SLOT_NAME.format(k) for k in range(C.N_SLOTS)], "team": slot_team})

    out = C.OUT_DIR / match_id
    out.mkdir(parents=True, exist_ok=True)
    tracking.to_parquet(out / "tracking.parquet", index=False)
    slots.to_csv(out / "slots.csv", index=False)
    gt.to_csv(out / "gt.csv", index=False)

    counts = gt["label"].value_counts().to_dict()
    info = dict(match_id=match_id, source=clean["source"], seed=C.SEED, n_frames=int(n_total),
                pitch_m=[clean["length"], clean["width"]],
                labels={k: int(v) for k, v in counts.items()},
                ball_missing_share=round(ball_missing, 3),
                duel_share=round(duel_share, 3), near_goal_share=round(goal_share, 3),
                slots=slot_stats, seconds=round(time.time() - t0, 1))
    (out / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("  OK {}: {} frames, ball mangler {:.0%}, {:.1f} aktive slots i snitt, {} slot-bytter -> {}".format(
        match_id, n_total, ball_missing, slot_stats["mean_active_slots"], slot_stats["slot_switches"], out))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", nargs="+", default=None,
                    help="Kamp-ID-er, eller 'all'. Standard: {}".format(C.DEFAULT_MATCHES))
    ap.add_argument("--selftest", action="store_true", help="Bruk kunstige data (ingen nedlasting)")
    args = ap.parse_args()

    if args.selftest:
        matches = args.matches or ["SELFTEST1", "SELFTEST2"]
    elif args.matches in (None, []):
        matches = C.DEFAULT_MATCHES
    elif args.matches == ["all"]:
        matches = C.ALL_MATCHES
    else:
        matches = args.matches
        unknown = [m for m in matches if m not in C.ALL_MATCHES]
        if unknown:
            sys.exit("Ukjent kamp-ID: {}. Gyldige: {}".format(unknown, C.ALL_MATCHES))

    print("Bygger {} kamp(er) til {}".format(len(matches), C.OUT_DIR))
    for m in matches:
        build_match(m, selftest=args.selftest)
    print("Ferdig.")


if __name__ == "__main__":
    main()
