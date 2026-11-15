"""Tracking -> features -> dead-ball evidence -> possession model -> submission CSV.

    python -m src.pipeline --team-name NEXT --no-cache   # real data: run by a human, never by Claude
    python -m src.pipeline --data-dir test_data/fake_match --meta test_data/fake_match/fake_meta.json --out test_data/fake_match/pred.csv

Deterministic: no random numbers anywhere, so the same data always gives the same bytes.
Prints label counts and possession counts of its own output only -- never the tracking.
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import games  # noqa: E402

from . import deadball, features, possession  # noqa: E402
from . import io as mio  # noqa: E402
from .config import Config  # noqa: E402


def label_match(data_dir=None, blocks=None, n_frames=None, cfg=None, game=games.GAME,
                use_cache=True):
    """Labels [T] of 'H'/'A'/'D'/'X' for the match under data_dir, plus diagnostics."""
    cfg = cfg or Config()
    paths = games.asset_paths(game, data_dir)
    for key in ("atd_meta", "atd_raw"):
        if not os.path.isfile(paths[key]):
            raise IOError("match data not found: {} (see DATA.md)".format(paths[key]))
    if n_frames is None:
        n_frames = games.source_frames("tactical", game)
    if blocks is None:
        blocks = [(first, last) for _, first, last in games.half_video_blocks(game)]
    cache_dir = os.path.join(data_dir or games.DATA_DIR, "cache") if use_cache else None

    trk = mio.load_tracking(paths["atd_meta"], paths["atd_raw"], n_frames, cache_dir)
    tags = mio.load_tags(paths["smart_tagging"], cfg.fps) \
        if os.path.isfile(paths["smart_tagging"]) else []
    feats = features.compute(trk, games.GAMES[game]["pitch"], cfg)
    dead_llr, _ = deadball.compute(feats, tags, blocks, cfg)
    return possession.decode(feats, dead_llr, tags, trk.names, blocks, cfg)


def summary(labels, fps=25):
    """Counts of the output labels and of its possessions (maximal H or A runs)."""
    out = {lab: int(np.sum(labels == lab)) for lab in "HADX"}
    starts, ends, vals = possession.runs_of(labels)
    spells = [(e - s) for s, e, v in zip(starts, ends, vals) if v in ("H", "A")]
    out["possessions"] = len(spells)
    out["mean_possession_s"] = round(float(np.mean(spells)) / fps, 2) if spells else 0.0
    out["stoppages"] = int(np.sum(vals == "D"))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--team-name", default="NEXT")
    parser.add_argument("--out", help="default: evaluation/results/<team-name>.csv")
    parser.add_argument("--data-dir", help="default: games.DATA_DIR ($MTAH_DATA_DIR or ./data)")
    parser.add_argument("--meta", help="json with n_frames and blocks (fake matches only)")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    blocks = n_frames = None
    if args.meta:
        with open(args.meta) as handle:
            meta = json.load(handle)
        n_frames, blocks = meta["n_frames"], [tuple(b) for b in meta["blocks"]]
    labels, _ = label_match(args.data_dir, blocks, n_frames, use_cache=not args.no_cache)
    out = args.out or os.path.join(ROOT, "evaluation", "results", "{}.csv".format(args.team_name))
    mio.write_submission(out, labels)
    print("wrote {}".format(out))
    print(summary(labels))


if __name__ == "__main__":
    main()
