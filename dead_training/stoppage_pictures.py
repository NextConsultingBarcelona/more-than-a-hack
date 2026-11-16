"""How the pipeline holds up when the ball is shown differently during stoppages.

Open Sportec data only; never reads the competition match.

The validation data (external_validation/build_dataset.py) keeps Sportec's ball through
stoppages. Sportec freezes the ball where play stopped and moves it to the restart point on the
restart frame: a placeholder, not something a camera saw. This script leaves every live frame
exactly as the validation data has it and gives the stoppage frames another picture:

    frozen   the validation data as built
    rest     the ball lies at the restart point
    none     no ball
    carry    the ball is with the player who takes the restart
    mix      one of the above at random per stoppage

The validation data's own missing-ball mask is applied on top, so the ball is gone on the same
frames in every picture. Each match is then labelled three ways with the code the pipeline runs:
with the state model's own dead-ball evidence (dead_model=False), with the stoppage detector
(dead_model=True), and with the true stoppages (an upper bound for the dead-ball part).

No model is scored on a match it has seen. The two tree models of src/learned.py are refitted
leave-one-match-out on the five tuning matches; on the two held-out matches the stored file is
used, which has not seen them. The detector of each match is the leave-one-match-out one from
dead_training/train.py --lomo.

Needs external_validation/build_dataset.py --matches all, dead_training/pack_sportec.py and
dead_training/train.py --lomo to have been run.

    python dead_training/stoppage_pictures.py              # about 40 minutes on one core
    python dead_training/stoppage_pictures.py --jobs 2
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "external_validation"), str(HERE)]

import build_dataset as B  # noqa: E402
import config as C  # noqa: E402
import train_emissions as T  # noqa: E402
from score import load_gt, score_labels  # noqa: E402

from deadball import degrade, opendata  # noqa: E402
from src import dead_model, deadball, extval, features, learned, possession  # noqa: E402
from src.config import Config  # noqa: E402
from src.io import Tracking  # noqa: E402

PICTURES = ["frozen", "rest", "none", "carry", "mix"]
ALL = T.TUNING + T.HELD_OUT
_WORK = {}


def ball_for(match, picture, packed):
    """Ball [T, 2] in the validation data's coordinates, with another picture during stoppages."""
    ball = pd.read_parquet(C.OUT_DIR / match / "tracking.parquet",
                           columns=["ball_x", "ball_y"]).to_numpy(dtype=np.float32).copy()
    if picture == "frozen":
        return ball
    fr = opendata.load_sportec(packed / "sportec_{}.npz".format(match))   # ball removed while dead
    n_total, idx = B.layout({"period": fr.period})
    if n_total != len(ball):
        raise ValueError("{}: the packed match and the validation data differ in length".format(match))
    rng = np.random.default_rng([7, zlib.crc32((match + picture).encode())])
    bx, by = degrade.simulate_dead_ball(fr, rng, picture)
    length, width = fr.pitch
    shown = np.stack([bx / length + 0.5, 0.5 - by / width], 1).astype(np.float32)[fr.dead]
    rows = idx[fr.dead]
    shown[np.isnan(ball[rows, 0])] = np.nan           # the validation data's own missing-ball mask
    ball[rows] = shown
    return ball


def evaluate(job):
    picture, match = job
    packed, lomo = _WORK["packed"], _WORK["lomo"]
    tracking = pd.read_parquet(C.OUT_DIR / match / "tracking.parquet")
    slots = pd.read_csv(C.OUT_DIR / match / "slots.csv")
    extval._cache.clear()
    trk, blocks = extval._tracking(tracking, slots)
    trk = Tracking(trk.x, trk.y, trk.s, ball_for(match, picture, packed), trk.team, trk.names)
    gt = load_gt(match)
    truth = gt["label"].astype(str).str.strip().str.upper().to_numpy()
    graded = np.zeros(len(truth), dtype=bool)
    for first, last in blocks:
        graded[first:last + 1] = True

    cfg = Config()
    feats = features.compute(trk, extval.PITCH, cfg)
    own, info = deadball.compute(feats, [], blocks, cfg)
    # the tree models of learned.py that have not seen this match
    learned._model_cache[str(learned.MODEL_PATH)] = _WORK["models"][match]
    raw = dead_model.scores(trk, extval.PITCH, cfg.fps,
                            dead_model.load_model(str(lomo / "{}.npz".format(match))))
    stops, used = dead_model.evidence(own, info["prekick"], raw, blocks, cfg)
    perfect = np.where(truth == "D", cfg.dm_weight, -cfg.dm_weight)
    perfect[~graded] = 0.0

    out = dict(picture=picture, match=match, held_out=match in T.HELD_OUT, detector_used=bool(used))
    for name, c, evidence in (("own", cfg.with_(dead_model=False), None),
                              ("detector", cfg, stops if used else None),
                              ("perfect", cfg, perfect)):
        labels, _ = possession.decode(feats, own, [], trk.names, blocks, c, dead_evidence=evidence)
        res = score_labels(gt, labels)
        out["spellF1_" + name], out["macroF1_" + name] = res["spell_f1"], res["macro_f1"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--packed", type=Path, default=ROOT / "ext_data" / "dead_training" / "sportec")
    ap.add_argument("--lomo-dir", type=Path, default=ROOT / "ext_data" / "dead_lomo")
    ap.add_argument("--pictures", nargs="*", default=PICTURES, choices=PICTURES)
    ap.add_argument("--matches", nargs="*", default=ALL, choices=ALL)
    ap.add_argument("--jobs", type=int, default=1, help="processes (does not change the result)")
    args = ap.parse_args()

    t0 = time.time()
    tuning = [T.prepare(m) for m in T.TUNING]
    models = {m: json.loads(Path(learned.MODEL_PATH).read_text(encoding="utf-8")) for m in T.HELD_OUT}
    for m in tuning:
        if m["match"] in args.matches:
            models[m["match"]] = T.fit([o for o in tuning if o is not m], 150)
    del tuning
    print("tree models of learned.py, leave-one-match-out: {:.0f} s".format(time.time() - t0), flush=True)

    _WORK.update(packed=args.packed, lomo=args.lomo_dir, models=models)
    jobs = [(p, m) for p in args.pictures for m in args.matches]
    import multiprocessing as mp
    if args.jobs > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(args.jobs) as pool:
            rows = pool.map(evaluate, jobs, chunksize=1)
    else:
        rows = [evaluate(job) for job in jobs]
    table = pd.DataFrame(rows)

    pd.set_option("display.width", 200)
    fmt = dict(index=False, float_format=lambda v: "{:.4f}".format(v))
    cols = ["spellF1_own", "spellF1_detector", "spellF1_perfect", "macroF1_own", "macroF1_detector"]
    print("\nMean over the matches, per picture of the ball during stoppages:")
    print(table.groupby("picture", sort=False)[cols].mean().reset_index().to_string(**fmt))
    print("\nHeld-out matches only:")
    print(table[table["held_out"]].groupby("picture", sort=False)[cols[:3]].mean().reset_index()
          .to_string(**fmt))
    ahead = int((table["spellF1_detector"] > table["spellF1_own"] + 5e-5).sum())
    level = int((np.abs(table["spellF1_detector"] - table["spellF1_own"]) <= 5e-5).sum())
    print("\nDetector ahead on {} of {} match-and-picture combinations, level on {}; "
          "used on {}.".format(ahead, len(table), level, int(table["detector_used"].sum())))
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.RESULTS_DIR / "stoppage_pictures_{}.csv".format(time.strftime("%Y%m%d-%H%M%S"))
    table.to_csv(out, index=False)
    print("Per match: {}   total time {:.0f} s".format(out, time.time() - t0))


if __name__ == "__main__":
    main()
