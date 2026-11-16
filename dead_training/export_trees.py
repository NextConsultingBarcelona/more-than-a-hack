"""Write a fitted scikit-learn HistGradientBoostingClassifier as flat numpy arrays.

    python dead_training/export_trees.py <model.joblib> models/dead_trees.npz

The run-time code (src/dead_model.py) then needs numpy only. A leaf has feature -1 and its
value in `number`; an inner node has its split feature and its threshold in `number`.
"""
import json
import sys

import joblib
import numpy as np


def export(bundle, path):
    clf = bundle["clf"]
    feature, number, left, right, nan_left, roots = [], [], [], [], [], []
    offset = 0
    for stage in clf._predictors:
        (tree,) = stage                              # binary: one tree per iteration
        nodes = tree.nodes
        if nodes["is_categorical"].any():
            raise ValueError("categorical splits are not supported")
        leaf = nodes["is_leaf"].astype(bool)
        roots.append(offset)
        feature.append(np.where(leaf, -1, nodes["feature_idx"]).astype(np.int16))
        number.append(np.where(leaf, nodes["value"], nodes["num_threshold"]).astype(np.float64))
        left.append(np.where(leaf, 0, nodes["left"].astype(np.int64) + offset).astype(np.int32))
        right.append(np.where(leaf, 0, nodes["right"].astype(np.int64) + offset).astype(np.int32))
        nan_left.append(nodes["missing_go_to_left"].astype(bool))
        offset += len(nodes)
    meta = {k: bundle[k] for k in ("train", "n_rows", "seeds", "params") if k in bundle}
    np.savez_compressed(
        path, feature=np.concatenate(feature), number=np.concatenate(number),
        left=np.concatenate(left), right=np.concatenate(right), nan_left=np.concatenate(nan_left),
        roots=np.array(roots, np.int32), baseline=np.float64(np.ravel(clf._baseline_prediction)[0]),
        names=np.array(bundle["features"]), meta=np.array(json.dumps(meta)))


if __name__ == "__main__":
    export(joblib.load(sys.argv[1]), sys.argv[2])
    print("wrote", sys.argv[2])
