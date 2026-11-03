"""Setter alt sammen til DataFrames på samme nivå som lasteren returnerer (v1)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import ball as ball_mod
from . import players as players_mod
from . import possession, tracker
from .config import FPS, MODES, PITCH_LENGTH as L, PITCH_WIDTH as W, SLOT_NAMES, Params, Timeline


@dataclass
class GeneratedMatch:
    """Resultatet av én generering.

    tracking   : frame_id (1..N), ball_x, ball_y, <slot>_x, <slot>_y, <slot>_s  (0-1, m/s, NaN)
    slot_teams : slot, team (H/A)
    truth      : frame (0..N-1), label (H/A/D/X)  - samme format som innleveringen
    events     : segmentliste (possession/dead/løs ball) for QA og tester
    timeline   : kampstrukturen som ble brukt
    seed       : seed som ble brukt
    debug      : sanne verdier (ballbane i meter, duell-/nær-mål-flagg) - kun for tester
    """

    tracking: pd.DataFrame
    slot_teams: pd.DataFrame
    truth: pd.DataFrame
    events: pd.DataFrame
    timeline: Timeline
    seed: int
    debug: dict


def generate(seed=42, mode="short", params=None, timeline=None):
    """Lag en syntetisk kamp. Samme seed + mode + params gir identisk resultat."""
    params = params or Params()
    tl = timeline or MODES[mode]
    rng = np.random.default_rng(seed)
    n = tl.n_frames

    # 1) Sann kamp: possessions, dødballer, ballens nøkkelpunkter
    segments, loose, keyframes = possession.simulate_match(tl, params, rng)

    labels = np.full(n, "X", dtype="<U1")
    owners = np.full(n, -1, dtype=int)
    dead = np.zeros(n, bool)
    for s in segments:
        labels[s["start"]:s["end"]] = s["label"]
        if s["label"] == "D":
            dead[s["start"]:s["end"]] = True
            owners[s["start"]:s["end"]] = 0 if s["owner"] == "H" else 1
    eval_mask = np.zeros(n, bool)
    eval_mask[tl.h1_start:tl.h1_end + 1] = True
    eval_mask[tl.h2_start:tl.h2_end + 1] = True

    # 2) Sann ballbane og spillere
    ball_m = ball_mod.true_ball(keyframes, n, eval_mask, dead, rng)
    codes = np.zeros(n, dtype=int)
    for lab, code in players_mod.LABEL_CODE.items():
        codes[labels == lab] = code
    players_m, speed = players_mod.true_players(tl, codes, owners, ball_m, params, rng)

    # 3) Måleutstyr: hull i ball, slots med dropout/duplikater
    missing, duel, near_goal = tracker.ball_missing_mask(ball_m, players_m, eval_mask, params, rng)
    slot_xy, slot_s, slot_team = tracker.assign_slots(players_m, speed, params, rng)

    # 4) DataFrames (koordinater normalisert til 0-1)
    scale = np.array([L, W])
    ball_obs = np.where(missing[:, None], np.nan, ball_m) / scale
    cols = {"frame_id": np.arange(1, n + 1), "ball_x": ball_obs[:, 0], "ball_y": ball_obs[:, 1]}
    for j, name in enumerate(SLOT_NAMES):
        cols[f"{name}_x"] = slot_xy[:, j, 0] / L
        cols[f"{name}_y"] = slot_xy[:, j, 1] / W
        cols[f"{name}_s"] = slot_s[:, j]
    tracking = pd.DataFrame(cols)

    slot_teams = pd.DataFrame({"slot": list(SLOT_NAMES), "team": slot_team})
    truth = pd.DataFrame({"frame": np.arange(n), "label": labels.astype(str)})

    events = pd.DataFrame(segments + loose, columns=["half", "start", "end", "label", "kind", "detail", "owner"])
    events = events.sort_values(["start", "kind"], kind="stable").reset_index(drop=True)
    events["duration_s"] = (events["end"] - events["start"]) / FPS

    debug = {"ball_true_m": ball_m, "ball_missing": missing, "duel": duel, "near_goal": near_goal,
             "eval_mask": eval_mask, "players_true_m": players_m}
    return GeneratedMatch(tracking, slot_teams, truth, events, tl, seed, debug)
