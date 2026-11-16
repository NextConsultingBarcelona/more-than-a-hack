"""Frame classifier: P(ball is dead) from the features in features.py."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

SEED = 20261117
PARAMS = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200,
              l2_regularization=1.0, max_bins=255, early_stopping=False, random_state=SEED)


def fit(X, y, sample_weight=None, **over):
    p = dict(PARAMS)
    p.update(over)
    clf = HistGradientBoostingClassifier(**p)
    clf.fit(X, y, sample_weight=sample_weight)
    return clf


def predict(clf, X, chunk=200_000):
    out = np.empty(len(X), np.float32)
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = clf.predict_proba(X[i:i + chunk])[:, 1]
    return out
