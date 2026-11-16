"""What Sportec's ownership field means, measured on the clean open matches.

    python team_training/ownership.py

Open data only. Three facts behind the training labels of the second home-vs-away model
(NEXT.md 5.9):

1. Ownership changes when the other team touches the ball: at the frame where it changes, how
   far is the ball from the nearest player of the new team and of the old team?
2. How many possessions there are per match, and how long they are, when a team spell shorter
   than T seconds between two spells of the other team does not count (controlled_owner in
   external_validation/build_dataset.py). T = 0 is the field as it is.
3. How much of live play the ball spends in the air (the validation data as built removes the
   ball regardless of its height; team_training/pictures.py has a picture that does not).

Needs dead_training/pack_sportec.py and external_validation/build_dataset.py --matches all.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "external_validation")]

import build_dataset as B  # noqa: E402
import config as C  # noqa: E402
import train_emissions as T  # noqa: E402
from score import load_gt  # noqa: E402

PACKED = ROOT / "ext_data" / "dead_training" / "sportec"


def nearest(players, ball):
    d2 = (players[..., 0] - ball[:, None, 0]) ** 2 + (players[..., 1] - ball[:, None, 1]) ** 2
    return np.sqrt(np.nanmin(np.where(np.isfinite(d2), d2, np.inf), axis=1))


def main():
    new, old, high = [], [], []
    for match in T.TUNING:
        z = np.load(PACKED / "sportec_{}.npz".format(match), allow_pickle=False)
        home = z["home_xy"].astype(np.float32) / 100.0
        away = z["away_xy"].astype(np.float32) / 100.0
        for a in (home, away):
            a[a[..., 0] <= -300] = np.nan                      # the packed files' missing value
        ball = z["ball_xyz"].astype(np.float32) / 100.0
        owner = z["owner"].astype(int)
        live = (z["ball_dead"] == 0) & (z["period"] > 0) & (owner > 0)
        dist = {1: nearest(home, ball), 2: nearest(away, ball)}
        flips = np.flatnonzero(live[1:] & live[:-1] & (owner[1:] != owner[:-1])) + 1
        new += [dist[owner[t]][t] for t in flips]
        old += [dist[owner[t - 1]][t] for t in flips]
        high.append(float((ball[live, 2] > 1.5).mean()))
    new, old = np.array(new), np.array(old)
    print("Changes of ownership during live play, five tuning matches: {}".format(len(new)))
    print("  ball to the nearest player of the NEW team: median {:.2f} m, within 1 m {:.1f} %, within 2 m {:.1f} %".format(
        np.median(new), 100 * (new < 1).mean(), 100 * (new < 2).mean()))
    print("  ball to the nearest player of the OLD team: median {:.2f} m, within 1 m {:.1f} %".format(
        np.median(old), 100 * (old < 1).mean()))

    print("Share of live play with the ball more than 1.5 m above the ground: {:.1f} %".format(
        100 * np.mean(high)))
    print("\nPossessions per match and their length, by the shortest spell that counts (T):")
    for t in (0.0, 1.0, 2.0, 3.0):
        count, mean, short = [], [], []
        for match in T.TUNING:
            lab = load_gt(match)["label"].astype(str).str.strip().str.upper().to_numpy()
            lab = B.controlled_owner(lab, int(round(t * C.FPS))).astype(str)
            edges = np.flatnonzero(lab[1:] != lab[:-1]) + 1
            starts, ends = np.r_[0, edges], np.r_[edges, len(lab)]
            keep = np.isin(lab[starts], ["H", "A"])
            dur = (ends - starts)[keep] / C.FPS
            count.append(len(dur))
            mean.append(dur.mean())
            short.append((dur < 2).mean())
        print("  T = {:.0f} s: {:.0f} possessions per match, mean {:.1f} s, {:.0f} % under 2 s".format(
            t, np.mean(count), np.mean(mean), 100 * np.mean(short)))


if __name__ == "__main__":
    main()
