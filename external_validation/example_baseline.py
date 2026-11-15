"""Enkel eksempelmodell: nærmeste slot til ballen + flertallsstemme i et vindu.

Samme idé som arrangørens baselines. Formålet er å vise FORMATET Oskars modell
må ha, og å teste at hele valideringen henger sammen - ikke å være god.

Formatet sweep.py forventer:
    def predict(tracking, slots, **params) -> liste/array med én etikett (H/A/D/X) per rad i tracking

Kan også kjøres alene og skrive CSV-er som score.py kan lese:
  python external_validation/example_baseline.py --window 50
  python external_validation/score.py ext_data/preds/baseline_w50
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402


def predict(tracking, slots, window=50):
    """tracking: dataframe (kapittel 4b). slots: dataframe med kolonnene slot, team."""
    names = slots["slot"].tolist()
    team = slots["team"].to_numpy()
    bx = tracking["ball_x"].to_numpy(dtype=float)
    by = tracking["ball_y"].to_numpy(dtype=float)
    sx = tracking[[s + "_x" for s in names]].to_numpy(dtype=float)
    sy = tracking[[s + "_y" for s in names]].to_numpy(dtype=float)

    dist = np.hypot((sx - bx[:, None]) * 105.0, (sy - by[:, None]) * 68.0)
    dist = np.where(np.isnan(dist), np.inf, dist)
    nearest = dist.argmin(axis=1)
    has_ball = np.isfinite(dist.min(axis=1))

    is_h = (has_ball & (team[nearest] == "H")).astype(float)
    is_a = (has_ball & (team[nearest] == "A")).astype(float)
    roll = dict(window=int(window), center=True, min_periods=1)
    votes_h = pd.Series(is_h).rolling(**roll).sum().to_numpy()
    votes_a = pd.Series(is_a).rolling(**roll).sum().to_numpy()

    labels = np.full(len(tracking), "X", dtype=object)
    labels[votes_h > votes_a] = "H"
    labels[votes_a > votes_h] = "A"
    return labels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--matches", nargs="+", default=None)
    args = ap.parse_args()

    from score import built_matches
    matches = args.matches or built_matches()
    out = C.PREDS_DIR / "baseline_w{}".format(args.window)
    out.mkdir(parents=True, exist_ok=True)
    for m in matches:
        tracking = pd.read_parquet(C.OUT_DIR / m / "tracking.parquet")
        slots = pd.read_csv(C.OUT_DIR / m / "slots.csv")
        labels = predict(tracking, slots, window=args.window)
        pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(out / "{}.csv".format(m), index=False)
        print("  skrev {}".format(out / "{}.csv".format(m)))


if __name__ == "__main__":
    main()
