"""Tracking -> features -> dead-ball detector -> possession model -> submission CSV.

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

from . import dead_model, deadball, features, possession, team_model  # noqa: E402
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
    pitch = games.GAMES[game]["pitch"]
    feats = features.compute(trk, pitch, cfg)
    dead_llr, info = deadball.compute(feats, tags, blocks, cfg)
    stops, used_model = None, False
    problem = "detector switched off" if not cfg.dead_model else "output not plausible"
    if cfg.dead_model:
        try:
            raw = dead_model.scores(trk, pitch, cfg.fps)
            stops, used_model = dead_model.evidence(dead_llr, info["prekick"], raw, blocks, cfg)
        except Exception as exc:       # the pipeline's own dead-ball evidence still gives a valid file
            problem = "detector failed with {}".format(type(exc).__name__)
    team, team_problem = None, "switched off" if cfg.learned else "hand-weighted evidence"
    if cfg.team_model and cfg.learned:
        try:
            team = {"shape": team_model.shape_stats(trk, pitch), "model": team_model.load_model()}
        except Exception as exc:       # the first home-vs-away model still gives a valid file
            team_problem = "failed with {}".format(type(exc).__name__)
    labels, diag = possession.decode(feats, dead_llr, tags, trk.names, blocks, cfg,
                                     dead_evidence=stops if used_model else None, team=team)
    diag["dead_model_used"] = used_model
    diag["dead_model_problem"] = None if used_model else problem
    if not diag.get("team_model_used", False):
        diag["team_model_used"] = False
        if team is None or not diag.get("team_model_problem"):
            diag["team_model_problem"] = team_problem
    return labels, diag


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
    parser.add_argument("--hand-weighted", action="store_true",
                        help="the earlier model: hand-weighted evidence instead of the learned one")
    parser.add_argument("--no-detector", action="store_true",
                        help="dead ball from the state model's own evidence, not the trained detector")
    parser.add_argument("--no-team-model", action="store_true",
                        help="home-vs-away evidence from the first model alone (learned.py)")
    parser.add_argument("--tagged", action="store_true",
                        help="the model first tagged phase1-submission: neither the detector nor "
                             "the second home-vs-away model")
    args = parser.parse_args(argv)

    blocks = n_frames = None
    if args.meta:
        with open(args.meta) as handle:
            meta = json.load(handle)
        n_frames, blocks = meta["n_frames"], [tuple(b) for b in meta["blocks"]]
    cfg = Config().hand_weighted() if args.hand_weighted else Config()
    if args.no_detector:
        cfg = cfg.with_(dead_model=False)
    if args.no_team_model:
        cfg = cfg.with_(team_model=False)
    if args.tagged:
        cfg = cfg.tagged()
    labels, diag = label_match(args.data_dir, blocks, n_frames, cfg=cfg,
                               use_cache=not args.no_cache)
    out = args.out or os.path.join(ROOT, "evaluation", "results", "{}.csv".format(args.team_name))
    mio.write_submission(out, labels)
    print("wrote {}".format(out))
    print("dead ball from: {}".format(
        "trained detector" if diag["dead_model_used"]
        else "the state model's own evidence ({})".format(diag["dead_model_problem"])))
    print("home/away from: {}".format(
        "second model" if diag["team_model_used"]
        else "first model ({})".format(diag["team_model_problem"])))
    print(summary(labels))


if __name__ == "__main__":
    main()
