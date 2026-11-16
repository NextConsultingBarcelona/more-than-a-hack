"""Download the seven open Sportec/IDSSE matches and pack each into one small file.

Open data (Bundesliga 2022/23, CC-BY 4.0, Bassek et al. 2025, Sci Data 12, 195). Not the
competition match.

    python dead_training/pack_sportec.py          # writes ext_data/dead_training/sportec/

One file per match, about 12 MB. Takes 5-10 minutes and 3-6 GB of memory per match; a match
that is already packed is skipped, so the script can simply be started again.

Per file (positions in cm as int16, origin at the centre spot, -32768 = missing):
    home_xy, away_xy  (n_frames, n_players, 2)
    ball_xyz          (n_frames, 3)
    ball_dead         (n_frames,) 1 = dead ball, 0 = alive, 255 = unknown
    owner             (n_frames,) 1 = home, 2 = away, 0 = unknown
    period, frame_id, time_s, pitch_m, fps, home, away, match_id
"""
import argparse
import gc
import sys
from pathlib import Path

import numpy as np

MATCHES = ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]
MISSING = np.iinfo(np.int16).min


def _cm(v):
    if v is None or v != v:
        return MISSING
    return int(np.clip(round(float(v) * 100.0), -32767, 32767))


def pack(ds, match_id):
    """TrackingDataset (coordinates='sportec', metres) -> dict of compact arrays."""
    from kloppy.domain import BallState, Ground

    home, away = ds.metadata.teams
    assert home.ground == Ground.HOME, "expected the home team first"
    idx = {}
    for t, team in enumerate((home, away)):
        for j, p in enumerate(team.players):
            idx[p.player_id] = (t, j)
    n = len(ds.records)
    xy = [np.full((n, len(home.players), 2), MISSING, np.int16),
          np.full((n, len(away.players), 2), MISSING, np.int16)]
    ball = np.full((n, 3), MISSING, np.int16)
    dead = np.full(n, 255, np.uint8)
    owner = np.zeros(n, np.uint8)
    period = np.zeros(n, np.uint8)
    frame_id = np.zeros(n, np.int32)
    time_s = np.zeros(n, np.float32)

    for i, fr in enumerate(ds.records):
        period[i] = fr.period.id
        frame_id[i] = fr.frame_id
        time_s[i] = fr.timestamp.total_seconds()
        if fr.ball_state is not None:
            dead[i] = 0 if fr.ball_state == BallState.ALIVE else 1
        if fr.ball_owning_team is not None:
            owner[i] = 1 if fr.ball_owning_team.team_id == home.team_id else 2
        b = fr.ball_coordinates
        if b is not None:
            ball[i] = (_cm(b.x), _cm(b.y), _cm(getattr(b, "z", None)))
        for player, pd_ in fr.players_data.items():
            c = pd_.coordinates
            if c is None:
                continue
            t, j = idx[player.player_id]
            xy[t][i, j] = (_cm(c.x), _cm(c.y))

    dims = ds.metadata.pitch_dimensions
    pitch = (getattr(dims, "pitch_length", None), getattr(dims, "pitch_width", None))
    return dict(
        home_xy=xy[0], away_xy=xy[1], ball_xyz=ball, ball_dead=dead, owner=owner,
        period=period, frame_id=frame_id, time_s=time_s,
        pitch_m=np.array([np.nan if v is None else float(v) for v in pitch], np.float32),
        fps=np.float32(ds.metadata.frame_rate),
        home=str(home.name), away=str(away.name), match_id=match_id,
        license="CC-BY 4.0, Bassek et al. 2025, doi:10.1038/s41597-025-04505-y",
    )


def summary(d):
    known = d["ball_dead"] != 255
    return (f"{len(d['period'])} frames, {d['fps']:.0f} fps, "
            f"dead ball {100 * (d['ball_dead'][known] == 1).mean():.0f} %, "
            f"pitch {d['pitch_m'][0]:.1f} x {d['pitch_m'][1]:.1f} m")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "ext_data" / "dead_training" / "sportec")
    ap.add_argument("--matches", nargs="*", default=MATCHES)
    args = ap.parse_args()

    from kloppy import sportec

    args.out.mkdir(parents=True, exist_ok=True)
    failed = []
    for m in args.matches:
        dst = args.out / f"sportec_{m}.npz"
        if dst.exists():
            print(f"{m}: already packed, skipping")
            continue
        print(f"{m}: downloading and reading (several minutes) ...", flush=True)
        try:
            ds = sportec.load_open_tracking_data(match_id=m, coordinates="sportec", only_alive=False)
            d = pack(ds, m)
            del ds
            tmp = dst.with_name(dst.stem + ".tmp.npz")
            np.savez_compressed(tmp, **d)
            tmp.replace(dst)
            print(f"{m}: {d['home']} - {d['away']}, {summary(d)}, "
                  f"{dst.stat().st_size / 1e6:.0f} MB -> {dst}", flush=True)
            del d
        except Exception as e:  # carry on with the next match
            print(f"{m}: FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append(m)
        gc.collect()
    if failed:
        print("Failed:", " ".join(failed), "- run the script again to retry.")
        sys.exit(1)
    print(f"Done. The files are in {args.out}/")


if __name__ == "__main__":
    main()
