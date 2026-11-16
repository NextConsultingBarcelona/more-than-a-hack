"""Training sets from the open matches, degraded to look like automatic tracking."""
from pathlib import Path

import numpy as np

from . import degrade, features

TRAIN_SEEDS = (0, 1)
EVAL_SEED = 1000


def train_rows(dead, period, rng, step=3, near=50):
    """Every `step`-th in-play frame plus all frames within `near` of a dead/alive boundary."""
    n = len(dead)
    keep = np.zeros(n, bool)
    keep[rng.integers(0, step)::step] = True
    edges = np.flatnonzero(np.diff(dead.astype(np.int8)) != 0) + 1
    for e in edges:
        keep[max(0, e - near):e + near] = True
    if period is not None:
        keep &= period > 0
    return np.flatnonzero(keep)


def match_seed(name, seed):
    return (sum(ord(c) * (i + 1) for i, c in enumerate(name)) * 1000 + seed) % (2 ** 31)


def build_match(fr, seed, cache_dir, full):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{fr.name}_s{seed}{'_full' if full else ''}.npz"
    if p.exists():
        return p
    g = degrade.degrade(fr, match_seed(fr.name, seed))
    X, names = features.build(g)
    if full:
        rows = np.arange(fr.n)
    else:
        rows = train_rows(fr.dead, fr.period, np.random.default_rng(match_seed(fr.name, seed) + 7))
    extra = {}
    if full:
        extra = dict(bx=g.bx, by=g.by, px=g.px, py=g.py, team=g.team, owner=fr.owner,
                     period=fr.period, pitch=np.array(fr.pitch, np.float32))
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(tmp, X=X[rows], y=fr.dead[rows], rows=rows, names=np.array(names), **extra)
    tmp.replace(p)
    return p
