"""The versions of the pipeline side by side on the open Sportec matches.

    python team_training/compare.py --jobs 2            # other pictures of the ball during stoppages
    python team_training/compare.py --live --jobs 2     # other pictures of the ball during live play

Open data only; never reads the competition match. Each match is labelled four ways with the
code the pipeline runs:

    tagged     the model first tagged phase1-submission (Config().tagged())
    detector   that model with the stoppage detector (dead_model.py)
    final      the detector and the second home-vs-away model (team_model.py): the default
    true       'final' with the true stoppages: an upper bound for the dead-ball part

and scored against Sportec's ownership as it is (T0) and with uncontrolled touches removed
(T2: a team spell shorter than 2 s between two spells of the other team does not count, the
organiser's definition of a change of possession).

No model is scored on a match it has seen. The two tree models of src/learned.py are refitted
leave-one-match-out on the five tuning matches; the second home-vs-away model of a tuning
match is the one from team_training/train.py that was trained on the other four; on the two
held-out matches the stored files of both are used, which have not seen them. The detector of
each match is the leave-one-match-out one from dead_training/train.py --lomo.

Needs external_validation/build_dataset.py --matches all, dead_training/pack_sportec.py,
dead_training/train.py --lomo and team_training/train.py to have been run. Detector scores and
the refitted first models are kept in ext_data/cache/, so a second run is quicker.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / "external_validation"), str(ROOT / "dead_training"), str(HERE)]

import stoppage_pictures as SP  # noqa: E402  (dead_training/)
import pictures  # noqa: E402

B, C, T = SP.B, SP.C, SP.T
from score import load_gt, score_labels  # noqa: E402
from src import dead_model, deadball, extval, features, learned, possession, team_model  # noqa: E402
from src.config import Config  # noqa: E402
from src.io import Tracking  # noqa: E402

ALL = T.TUNING + T.HELD_OUT
VERSIONS = ["tagged", "detector", "final", "true"]
EVAL_T = (0.0, 2.0)
_WORK = {}


def first_models(cache):
    """{match: tree models of learned.py that have not seen the match}."""
    models = {m: json.loads(Path(learned.MODEL_PATH).read_text(encoding="utf-8")) for m in T.HELD_OUT}
    missing = [m for m in T.TUNING if not (cache / "learned_{}.json".format(m)).is_file()]
    if missing:
        tuning = [T.prepare(m) for m in T.TUNING]
        for m in tuning:
            path = cache / "learned_{}.json".format(m["match"])
            path.write_text(json.dumps(T.fit([o for o in tuning if o is not m], 150), separators=(",", ":")))
    for m in T.TUNING:
        models[m] = json.loads((cache / "learned_{}.json".format(m)).read_text(encoding="utf-8"))
    return models


def evaluate(job):
    kind, picture, match = job
    w = _WORK
    tracking = pd.read_parquet(C.OUT_DIR / match / "tracking.parquet")
    slots = pd.read_csv(C.OUT_DIR / match / "slots.csv")
    gt = load_gt(match)
    truth = gt["label"].astype(str).str.strip().str.upper().to_numpy()
    if kind == "live":
        tracking = pictures.make(match, picture, tracking, slots, truth, w["packed"], B.layout)
    extval._cache.clear()
    trk, blocks = extval._tracking(tracking, slots)
    if kind == "stoppage":
        trk = Tracking(trk.x, trk.y, trk.s, SP.ball_for(match, picture, w["packed"]), trk.team, trk.names)
    graded = np.zeros(len(truth), dtype=bool)
    for first, last in blocks:
        graded[first:last + 1] = True

    cfg = Config()
    feats = features.compute(trk, extval.PITCH, cfg)
    own, info = deadball.compute(feats, [], blocks, cfg)
    learned._model_cache[str(learned.MODEL_PATH)] = w["first"][match]

    raw_path = w["cache"] / "detector_{}_{}_{}.npy".format(kind, picture, match)
    if raw_path.is_file():
        raw = np.load(raw_path)
    else:
        raw = dead_model.scores(trk, extval.PITCH, cfg.fps,
                                dead_model.load_model(str(w["dead_lomo"] / "{}.npz".format(match))))
        np.save(raw_path, raw)
    stops, used = dead_model.evidence(own, info["prekick"], raw, blocks, cfg)
    perfect = np.where(truth == "D", cfg.dm_weight, -cfg.dm_weight)
    perfect[~graded] = 0.0

    fold = w["team_lomo"] / "{}.npz".format(match)
    if match in T.TUNING and not fold.is_file():
        raise IOError("no leave-one-out team model for {}: run team_training/train.py".format(match))
    # the second model's log-odds once, with the inputs possession.decode gives it
    ball_llr, _ = possession.ball_evidence(feats, cfg)
    hint = possession.hint_evidence(feats, ball_llr, own, blocks, cfg)
    team = {"logit": team_model.logit(
        feats, ball_llr, hint, own, blocks, team_model.shape_stats(trk, extval.PITCH), cfg,
        team_model.load_model(str(fold)) if match in T.TUNING else team_model.load_model())}
    first = {}                               # the first models' log-odds once per match, too
    emissions = learned.emissions
    learned.emissions = lambda *a, **k: first["z"] if "z" in first else first.setdefault("z", emissions(*a, **k))

    out = dict(kind=kind, picture=picture, match=match, held_out=match in T.HELD_OUT,
               detector_used=bool(used))
    labelled = {t: gt.assign(label=B.controlled_owner(truth, int(round(t * C.FPS))).astype(str))
                for t in EVAL_T}
    for name, c, evidence, tm in (("tagged", cfg.tagged(), None, None),
                                  ("detector", cfg.with_(team_model=False), stops if used else None, None),
                                  ("final", cfg, stops if used else None, team),
                                  ("true", cfg, perfect, team)):
        labels, diag = possession.decode(feats, own, [], trk.names, blocks, c,
                                         dead_evidence=evidence, team=tm)
        if tm is not None and not diag["team_model_used"]:
            raise RuntimeError("second model not used: {}".format(diag["team_model_problem"]))
        for t in EVAL_T:
            res = score_labels(labelled[t], labels)
            out["spellF1_T{:g}_{}".format(t, name)] = res["spell_f1"]
            out["macroF1_T{:g}_{}".format(t, name)] = res["macro_f1"]
    learned.emissions = emissions
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true", help="pictures of the live ball instead of stoppages")
    ap.add_argument("--packed", type=Path, default=ROOT / "ext_data" / "dead_training" / "sportec")
    ap.add_argument("--dead-lomo", type=Path, default=ROOT / "ext_data" / "dead_lomo")
    ap.add_argument("--team-lomo", type=Path, default=ROOT / "ext_data" / "team_lomo")
    ap.add_argument("--cache", type=Path, default=ROOT / "ext_data" / "cache")
    ap.add_argument("--pictures", nargs="*")
    ap.add_argument("--matches", nargs="*", default=ALL, choices=ALL)
    ap.add_argument("--jobs", type=int, default=1, help="processes (does not change the result)")
    args = ap.parse_args()

    kind = "live" if args.live else "stoppage"
    names = args.pictures or (pictures.NAMES if args.live else SP.PICTURES)
    args.cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _WORK.update(packed=args.packed, dead_lomo=args.dead_lomo, team_lomo=args.team_lomo,
                 cache=args.cache, first=first_models(args.cache))
    jobs = [(kind, p, m) for p in names for m in args.matches]
    import multiprocessing as mp
    if args.jobs > 1 and "fork" in mp.get_all_start_methods():
        with mp.get_context("fork").Pool(args.jobs) as pool:
            rows = pool.map(evaluate, jobs, chunksize=1)
    else:
        rows = [evaluate(job) for job in jobs]
    table = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    fmt = dict(index=False, float_format=lambda v: "{:.4f}".format(v))
    for t in EVAL_T:
        cols = ["spellF1_T{:g}_{}".format(t, v) for v in VERSIONS]
        short = dict(zip(cols, VERSIONS))
        what = "Sportec's ownership as it is" if t == 0 else "uncontrolled touches under {:g} s removed".format(t)
        print("\nspellF1 against {} (T{:g}), mean over the matches:".format(what, t))
        print(table.groupby("picture", sort=False)[cols].mean().rename(columns=short).reset_index().to_string(**fmt))
        print("held-out matches only:")
        print(table[table["held_out"]].groupby("picture", sort=False)[cols].mean().rename(columns=short)
              .reset_index().to_string(**fmt))
        a, b = table["spellF1_T{:g}_final".format(t)], table["spellF1_T{:g}_tagged".format(t)]
        print("final ahead of tagged on {} of {} match-and-picture combinations, level on {}".format(
            int((a > b + 5e-5).sum()), len(table), int((np.abs(a - b) <= 5e-5).sum())))
    cols = ["macroF1_T0_{}".format(v) for v in VERSIONS]
    print("\nmacroF1 (T0):")
    print(table.groupby("picture", sort=False)[cols].mean().rename(columns=dict(zip(cols, VERSIONS)))
          .reset_index().to_string(**fmt))
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.RESULTS_DIR / "compare_{}_{}.csv".format(kind, time.strftime("%Y%m%d-%H%M%S"))
    table.to_csv(out, index=False)
    print("\nPer match: {}   total time {:.0f} s".format(out, time.time() - t0))


if __name__ == "__main__":
    main()
