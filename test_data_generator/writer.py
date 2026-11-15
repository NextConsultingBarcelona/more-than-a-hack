"""Skriver generert kamp til disk, deterministisk (samme seed -> byte-identiske filer)."""

import os
from pathlib import Path

from .config import SLOT_NAMES

FILES = {
    "tracking": "fake_tracking.csv.gz",
    "slot_teams": "fake_slot_teams.csv",
    "truth": "fake_truth.csv",
    "events": "fake_events.csv",
}


def _check_out_dir(out_dir):
    """Sperre: generatoren skal aldri skrive i en mappe som heter data/ eller i MTAH_DATA_DIR."""
    out = Path(out_dir).resolve()
    if "data" in [p.lower() for p in out.parts]:
        raise ValueError(f"Nekter å skrive til {out}: stien inneholder en mappe som heter 'data'. "
                         "Falske data skal i test_data/.")
    real = os.environ.get("MTAH_DATA_DIR")
    if real:
        real_path = Path(real).resolve()
        if out == real_path or real_path in out.parents:
            raise ValueError(f"Nekter å skrive til {out}: det er inni MTAH_DATA_DIR (ekte data).")
    return out


def write_match(match, out_dir):
    out = _check_out_dir(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tracking = match.tracking.copy()
    xy_cols = ["ball_x", "ball_y"] + [f"{s}_{a}" for s in SLOT_NAMES for a in ("x", "y")]
    s_cols = [f"{s}_s" for s in SLOT_NAMES]
    tracking[xy_cols] = tracking[xy_cols].round(4)  # ca. 1 cm
    tracking[s_cols] = tracking[s_cols].round(2)

    paths = {k: out / v for k, v in FILES.items()}
    # mtime=0 -> gzip-headeren får ikke tidsstempel, så fila blir byte-identisk ved ny kjøring
    tracking.to_csv(paths["tracking"], index=False, lineterminator="\n",
                    compression={"method": "gzip", "mtime": 0})
    match.slot_teams.to_csv(paths["slot_teams"], index=False, lineterminator="\n")
    match.truth.to_csv(paths["truth"], index=False, lineterminator="\n")
    ev = match.events.copy()
    ev["duration_s"] = ev["duration_s"].round(2)
    ev.to_csv(paths["events"], index=False, lineterminator="\n")
    return paths
