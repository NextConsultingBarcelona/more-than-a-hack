"""Kommandolinje:  python -m test_data_generator --seed 42 --mode short

Skriver til test_data/ i repo-roten som standard.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from .build import generate
from .writer import write_match

REPO_ROOT = Path(__file__).resolve().parent.parent


def summary(match):
    t = match.truth["label"].to_numpy()
    tl = match.timeline
    lines = [f"seed={match.seed}  frames={len(t)}  (1. omg {tl.h1_start}-{tl.h1_end}, "
             f"2. omg {tl.h2_start}-{tl.h2_end}, avspark {tl.h2_kickoff})"]
    for lab in "HADX":
        c = int((t == lab).sum())
        lines.append(f"  {lab}: {c:>7}  {100 * c / len(t):5.1f} %")
    ev = match.events
    pos = ev[ev["kind"] == "possession"]
    dead = ev[ev["kind"] == "dead"]
    lines.append(f"  possessions: {len(pos)}  (median {pos['duration_s'].median():.1f} s)")
    lines.append(f"  dødballer:   {len(dead)}  " + ", ".join(
        f"{k}={v}" for k, v in dead["detail"].value_counts().sort_index().items()))
    lines.append(f"  løse baller: {int((ev['kind'] == 'loose_kept').sum())} beholdt, "
                 f"{int((ev['kind'] == 'loose_turnover').sum())} med bytte")
    em = match.debug["eval_mask"]
    lines.append(f"  ball mangler: {100 * match.debug['ball_missing'][em].mean():.1f} % av vurderte frames")
    slot_x = match.tracking[[c for c in match.tracking.columns if c.endswith("_x") and c != "ball_x"]]
    active = slot_x.notna().sum(axis=1).to_numpy()[em]
    lines.append(f"  aktive slots: median {int(np.median(active))}, min {active.min()}, maks {active.max()}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generer FALSKE fotballdata (syntetisk kamp med kjent fasit).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["short", "full"], default="short")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "test_data",
                    help="mappe for output (standard: test_data/ i repo-roten)")
    args = ap.parse_args(argv)

    t0 = time.time()
    match = generate(seed=args.seed, mode=args.mode)
    paths = write_match(match, args.out)
    print(summary(match))
    print(f"\nSkrev ({time.time() - t0:.0f} s):")
    for p in paths.values():
        print(f"  {p}")


if __name__ == "__main__":
    main()
