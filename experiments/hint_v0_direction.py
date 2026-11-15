"""Hint v0: "the whole pitch moving one way" + "everyone standing still".

The idea, in football terms:
  * everyone (nearly) standing still for a while  -> D  (play is stopped)
  * the players, on average, moving toward Athletic's goal -> H  (Barcelona attacking)
  * the players, on average, moving toward Barcelona's goal -> A  (Athletic attacking)
  * moving neither way clearly -> keep the previous answer

No ball, no model, no training. It is a deliberately crude first rung: the point is to measure
how much these two hints carry on their own before they go into the HMM as evidence.

Run it LOCALLY, never through Claude (the match data must not reach a cloud LLM):

    python experiments/hint_v0_direction.py
    python evaluation/validate_submission.py evaluation/results/hint_v0.csv
    python evaluation/run_evaluation.py

The public 5 minutes are a sanity check only. Do not tune the constants below on them -- that is
against RULES.md and picks the wrong values anyway. Tuning happens on the external games.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from baseline import AWAY, HOME, team_of_each_track, write_submission  # noqa: E402

# ---------------------------------------------------------------- constants (starting guesses)
FPS = 25
KICKOFF_FRAMES = (11185, 113742)  # video frames of the two kickoffs (FAQ.md, load_atd.py)

STILL_SPEED = 1.2        # m/s: mean player speed below this counts as "standing still"
MIN_DEAD_S = 2.0         # s: must stay still this long before it counts as dead ball
DIRECTION_EPS = 0.3      # m/s: mean x-velocity needed to call a direction at all
MIN_POSSESSION_S = 2.0   # s: shorter H/A runs are absorbed into the run before them
SMOOTH_S = 1.0           # s: half-width of the moving average on both signals
MAX_PLAYER_SPEED = 12.0  # m/s: faster than this between two frames = a track jump, ignored

RESULTS = ROOT / "evaluation" / "results"


# ---------------------------------------------------------------- reading
def read_players(raw_path, n_frames, n_slots):
    """x, y (0-1) and speed (m/s) per frame and track slot. NaN where the slot is empty."""
    x = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    y = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    s = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1  # ATD frame_id counts from 1, video frames from 0
            if not 0 <= row < n_frames:
                continue
            for slot, group in enumerate(parts[1].split(";")):
                if slot >= n_slots or not group or group[0] == "N":
                    continue
                fields = group.split(",")
                x[row, slot] = float(fields[0])
                y[row, slot] = float(fields[1])
                if len(fields) > 2:
                    s[row, slot] = float(fields[2])
    return x, y, s


# ---------------------------------------------------------------- small numpy helpers
def nanmean_rows(a):
    """Row-wise mean ignoring NaN; NaN where a row is all NaN. No RuntimeWarning spam."""
    count = np.sum(~np.isnan(a), axis=1)
    total = np.nansum(a, axis=1)
    out = np.full(a.shape[0], np.nan)
    ok = count > 0
    out[ok] = total[ok] / count[ok]
    return out


def rolling_nanmean(a, half):
    """Centred moving average over +/- half frames, ignoring NaN."""
    values = np.nan_to_num(a)
    present = (~np.isnan(a)).astype(np.float64)
    cv = np.concatenate(([0.0], np.cumsum(values)))
    cp = np.concatenate(([0.0], np.cumsum(present)))
    i = np.arange(len(a))
    lo = np.maximum(0, i - half)
    hi = np.minimum(len(a), i + half + 1)
    n = cp[hi] - cp[lo]
    out = np.full(len(a), np.nan)
    ok = n > 0
    out[ok] = (cv[hi] - cv[lo])[ok] / n[ok]
    return out


def runs(labels):
    """(start, end_exclusive, label) for every maximal run of equal labels."""
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(labels)]))
    return [(int(a), int(b), labels[a]) for a, b in zip(starts, ends)]


# ---------------------------------------------------------------- the hints
def x_velocity(x, length):
    """Per-slot x-velocity in m/s (central difference). Track jumps become NaN."""
    vx = np.full_like(x, np.nan)
    vx[1:-1] = (x[2:] - x[:-2]) * length * FPS / 2.0
    vx[np.abs(vx) > MAX_PLAYER_SPEED] = np.nan
    return vx


def home_attacks_positive_x(x, team, kickoff):
    """True if Barcelona attacks toward x = 1 in the half that starts at `kickoff`.

    At a kickoff every player stands in his own half, so the side the home team stands on is the
    side it defends. Uses the first second after the kickoff frame.
    """
    window = x[kickoff:kickoff + FPS]
    home_x = np.nanmedian(window[:, team == HOME])
    return home_x < 0.5


def label(x, s, team, n_frames, length):
    labels = np.full(n_frames, "X", dtype="<U1")
    vx = x_velocity(x, length)
    half = int(SMOOTH_S * FPS)

    mean_speed = rolling_nanmean(nanmean_rows(s), half)
    mean_vx = rolling_nanmean(nanmean_rows(vx), half)

    for (_, first, last), kickoff in zip(half_video_blocks(GAME), KICKOFF_FRAMES):
        sign = 1.0 if home_attacks_positive_x(x, team, kickoff) else -1.0
        toward_away_goal = mean_vx[first:last + 1] * sign      # > 0 means toward Athletic's goal
        speed = mean_speed[first:last + 1]

        block = np.full(last - first + 1, "X", dtype="<U1")

        # 1. dead ball: still for at least MIN_DEAD_S
        still = np.nan_to_num(speed, nan=0.0) < STILL_SPEED
        for a, b, is_still in runs(still):
            if is_still and (b - a) >= MIN_DEAD_S * FPS:
                block[a:b] = "D"
        # the stretch before the second-half kickoff is always dead (FAQ.md)
        if kickoff > first:
            block[:kickoff - first] = "D"

        # 2. direction while alive; hold the previous answer when it is unclear
        current = "X"
        for k in range(len(block)):
            if block[k] == "D":
                current = "X"   # a new possession starts after every stoppage
                continue
            v = toward_away_goal[k]
            if not np.isnan(v) and v > DIRECTION_EPS:
                current = "H"
            elif not np.isnan(v) and v < -DIRECTION_EPS:
                current = "A"
            block[k] = current

        # 3. absorb very short H/A runs into whatever came before them
        min_len = int(MIN_POSSESSION_S * FPS)
        previous = None
        for a, b, lab in runs(block):
            if lab in ("H", "A") and (b - a) < min_len and previous in ("H", "A"):
                block[a:b] = previous
            else:
                previous = block[a]

        labels[first:last + 1] = block
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(RESULTS / "hint_v0.csv"))
    args = parser.parse_args()

    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    team = team_of_each_track(meta_path)
    n_frames = source_frames("tactical")
    length = GAMES[GAME]["pitch"][0]

    print("reading tracking (a minute or two) ...")
    x, _, s = read_players(raw_path, n_frames, len(team))
    labels = label(x, s, team, n_frames, length)

    write_submission(args.out, labels)
    print("wrote {}".format(args.out))
    print("next:  python evaluation/validate_submission.py {}".format(args.out))
    print("       python evaluation/run_evaluation.py")


if __name__ == "__main__":
    main()
