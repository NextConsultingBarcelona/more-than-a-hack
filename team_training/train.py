"""Train the second home-vs-away model (src/team_model.py) on open Sportec matches.

    python team_training/train.py                 # leave-one-match-out models + the stored model
    python team_training/train.py --lomo-only

Needs external_validation/build_dataset.py --matches all and dead_training/pack_sportec.py to
have been run, and the packages in external_validation/requirements-validation.txt (LightGBM).
Never reads the competition match.

Labels. Sportec's ownership field changes on every touch by the other team (at the frame where
ownership changes, a player of the new team is within 1 m of the ball in 85 % of cases). The
organiser counts only controlled possession: "a simple touch of the ball without control" is
not a change (CHALLENGE.md). The model is therefore trained on the ownership field with
uncontrolled touches removed: a team spell shorter than CONTROLLED_S that lies between two
spells of the other team belongs to the other team (external_validation/build_dataset.py,
controlled_owner). CONTROLLED_S = 2 s is the organiser's "a player keeps the ball for a few
seconds"; it was not fitted to anything.

Protocol. As for the first model (train_emissions.py): the five tuning matches, both
orientations (home and away swapped, antisymmetric features negated). Each match is used in
three pictures of the live ball (team_training/pictures.py): as built, with the ball missing
while it is in the air, and with position noise; every tenth live frame of each. The stored
model has not seen the two held-out matches. For scores on the tuning matches, one model per
match, trained on the other four, is written to ext_data/team_lomo/.
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
sys.path[:0] = [str(ROOT), str(ROOT / "external_validation"), str(HERE)]

import build_dataset as B  # noqa: E402
import config as C  # noqa: E402
import train_emissions as T  # noqa: E402
from score import load_gt  # noqa: E402

from src import deadball, extval, features, possession, team_model  # noqa: E402
from src.config import Config  # noqa: E402

import pictures  # noqa: E402

CONTROLLED_S = 2.0
N_TREES = 300
PARAMS = dict(T.LGB_PARAMS, num_leaves=31)
PICTURES = ["built", "air", "jitter"]
STRIDE = 10


def features_of(match, tracking, slots):
    """(antisymmetric, symmetric) feature blocks of one validation match, with the run-time code."""
    extval._cache.clear()
    trk, blocks = extval._tracking(tracking, slots)
    cfg = Config()
    feats = features.compute(trk, extval.PITCH, cfg)
    own, _ = deadball.compute(feats, [], blocks, cfg)
    ball_llr, _ = possession.ball_evidence(feats, cfg)
    hint = possession.hint_evidence(feats, ball_llr, own, blocks, cfg)
    extval._cache.clear()
    return team_model.features(feats, ball_llr, hint, own, blocks,
                               team_model.shape_stats(trk, extval.PITCH), cfg)


def prepare(match, packed_dir, picture_names=PICTURES):
    """Training rows of one match: every STRIDE-th live frame of each picture, both orientations."""
    tracking = pd.read_parquet(C.OUT_DIR / match / "tracking.parquet")
    slots = pd.read_csv(C.OUT_DIR / match / "slots.csv")
    labels = load_gt(match)["label"].astype(str).str.strip().str.upper().to_numpy()
    controlled = B.controlled_owner(labels, int(round(CONTROLLED_S * C.FPS))).astype(str)
    rows = np.flatnonzero(np.isin(controlled, ["H", "A"]))[::STRIDE]
    home = (controlled[rows] == "H").astype(float)
    xs, ys = [], []
    for name in picture_names:
        anti, sym = features_of(match, pictures.make(match, name, tracking, slots, labels,
                                                     packed_dir, B.layout), slots)
        xs += [np.column_stack([anti[rows], sym[rows]]), np.column_stack([-anti[rows], sym[rows]])]
        ys += [home, 1.0 - home]
    return dict(match=match, n=len(labels), x=np.vstack(xs), y=np.concatenate(ys))


def to_flat(booster, n_features, meta):
    """The LightGBM trees as the flat arrays src/dead_model.raw_scores walks."""
    feature, number, left, right, roots = [], [], [], [], []
    for info in booster.dump_model()["tree_info"]:
        nodes = []

        def walk(node):
            i = len(nodes)
            nodes.append(None)
            if "split_feature" not in node:
                nodes[i] = (-1, float(node["leaf_value"]), 0, 0)
                return i
            if node["decision_type"] != "<=":
                raise ValueError("unsupported split type {}".format(node["decision_type"]))
            lo, hi = walk(node["left_child"]), walk(node["right_child"])
            nodes[i] = (int(node["split_feature"]), float(node["threshold"]), lo, hi)
            return i

        walk(info["tree_structure"])
        offset = len(feature)
        roots.append(offset)
        for f, v, lo, hi in nodes:
            feature.append(f)
            number.append(v)
            left.append(lo + offset if f >= 0 else 0)
            right.append(hi + offset if f >= 0 else 0)
    return dict(feature=np.array(feature, np.int16), number=np.array(number, np.float64),
                left=np.array(left, np.int32), right=np.array(right, np.int32),
                nan_left=np.zeros(len(feature), dtype=bool), roots=np.array(roots, np.int32),
                baseline=np.float64(0.0), n_features=np.int32(n_features),
                meta=np.array(json.dumps(meta)))


def fit(matches):
    import lightgbm as lgb

    x, y = np.vstack([m["x"] for m in matches]), np.concatenate([m["y"] for m in matches])
    booster = lgb.train(PARAMS, lgb.Dataset(x, y), num_boost_round=N_TREES)
    meta = dict(train=[m["match"] for m in matches], rows=int(len(y)), controlled_s=CONTROLLED_S,
                pictures=PICTURES, trees=N_TREES, params=PARAMS, stride=STRIDE,
                lightgbm=lgb.__version__)
    model = to_flat(booster, x.shape[1], meta)
    from src import dead_model
    diff = np.abs(dead_model.raw_scores(x[:5000], model) - booster.predict(x[:5000], raw_score=True))
    assert diff.max() < 1e-9, "the numpy trees differ from LightGBM: {}".format(diff.max())
    return model


def save(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **model)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "team_trees.npz")
    ap.add_argument("--lomo-dir", type=Path, default=ROOT / "ext_data" / "team_lomo")
    ap.add_argument("--packed", type=Path, default=ROOT / "ext_data" / "dead_training" / "sportec",
                    help="the packed open matches (dead_training/pack_sportec.py), for the ball's height")
    ap.add_argument("--lomo-only", action="store_true")
    ap.add_argument("--no-lomo", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    tuning = [prepare(m, args.packed) for m in T.TUNING]
    print("features of {} matches in {} pictures: {:.0f} s, {} per frame, {} rows".format(
        len(tuning), len(PICTURES), time.time() - t0, tuning[0]["x"].shape[1],
        sum(len(m["y"]) for m in tuning)), flush=True)
    if not args.no_lomo:
        index = {}
        for m in tuning:
            save(fit([o for o in tuning if o is not m]), args.lomo_dir / "{}.npz".format(m["match"]))
            index[str(m["n"])] = m["match"]
            print("  without {}: {:.0f} s".format(m["match"], time.time() - t0), flush=True)
        unseen = {str(len(load_gt(m))): m for m in T.HELD_OUT}
        (args.lomo_dir / "index.json").write_text(json.dumps(dict(folds=index, unseen=unseen), indent=1))
        print("leave-one-match-out models in {}".format(args.lomo_dir))
    if not args.lomo_only:
        model = fit(tuning)
        save(model, args.out)
        print("stored model: {} ({} trees, {} nodes, {} rows), {:.0f} s".format(
            args.out, len(model["roots"]), len(model["feature"]), json.loads(str(model["meta"]))["rows"],
            time.time() - t0))


if __name__ == "__main__":
    main()
