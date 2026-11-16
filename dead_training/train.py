"""Train the dead-ball detector on the seven open Sportec matches and write models/dead_trees.npz.

    python -m pip install -r dead_training/requirements-training.txt
    python dead_training/pack_sportec.py      # download and pack the matches (about an hour)
    python dead_training/train.py             # features, fit, export (about 20 minutes, 5 GB RAM)
    python dead_training/train.py --lomo      # also seven leave-one-match-out detectors

Reads open Sportec matches only. Every random step has a fixed seed. Gradient boosting sums
floating-point numbers in parallel, so a retrained detector is equivalent to the one in the
repository but not guaranteed to be bit-identical; the submission is reproduced from the stored
file, not from a retraining.

--lomo writes ext_data/dead_lomo/<match>.npz, each trained on the other six matches, and the
index.json that src.extval.predict_lomo needs. Those are what the scores in NEXT.md are
computed with: the stored detector has seen all seven matches.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "external_validation")]
import config as C                                        # noqa: E402  (external_validation/config.py)
from deadball import dataset, model, opendata             # noqa: E402
from export_trees import export                            # noqa: E402


def load_rows(files):
    parts = [np.load(f) for f in files]
    sizes = [len(z["y"]) for z in parts]
    X = np.empty((sum(sizes), parts[0]["X"].shape[1]), np.float32)
    y = np.empty(sum(sizes), bool)
    i = 0
    for z, k in zip(parts, sizes):
        X[i:i + k], y[i:i + k] = z["X"], z["y"]
        i += k
    return X, y


def fit(files, names, feature_names):
    X, y = load_rows(files)
    clf = model.fit(X, y)
    return {"clf": clf, "features": feature_names, "train": names, "n_rows": int(len(y)),
            "seeds": list(dataset.TRAIN_SEEDS), "params": dict(model.PARAMS)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=ROOT / "ext_data" / "dead_training" / "sportec")
    ap.add_argument("--cache", type=Path, default=ROOT / "ext_data" / "dead_training" / "cache")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "dead_trees.npz")
    ap.add_argument("--lomo", action="store_true", help="also train the leave-one-match-out detectors")
    ap.add_argument("--lomo-dir", type=Path, default=ROOT / "ext_data" / "dead_lomo")
    a = ap.parse_args()

    matches = opendata.load_all(a.data)
    if not matches:
        sys.exit("no packed matches in {}: run dead_training/pack_sportec.py first".format(a.data))
    files, rows, feature_names = {}, {}, None
    for fr in matches:                                    # sorted by name: a fixed order
        files[fr.name] = [dataset.build_match(fr, seed, a.cache, full=False)
                          for seed in dataset.TRAIN_SEEDS]
        rows[fr.name] = fr.n + C.PRE_FRAMES + C.HALFTIME_FRAMES + C.POST_FRAMES
        feature_names = [str(s) for s in np.load(files[fr.name][0])["names"]]
        print("features", fr.name, flush=True)
    names = sorted(files)
    del matches

    t = time.time()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    export(fit([f for n in names for f in files[n]], names, feature_names), a.out)
    print("wrote {} ({:.0f} s)".format(a.out, time.time() - t), flush=True)

    if a.lomo:
        folder = a.lomo_dir
        folder.mkdir(parents=True, exist_ok=True)
        index = {}
        for test in names:
            t = time.time()
            rest = [n for n in names if n != test]
            match_id = test.replace("sportec_", "")
            export(fit([f for n in rest for f in files[n]], rest, feature_names),
                   folder / (match_id + ".npz"))
            index[str(rows[test])] = match_id
            print("leave-one-out detector for {} ({:.0f} s)".format(match_id, time.time() - t), flush=True)
        (folder / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
