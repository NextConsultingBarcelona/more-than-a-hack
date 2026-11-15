"""Fast parsing of the ATD, its metadata and the Smart Tagging into numpy, with an on-disk cache.

Everything is indexed by video frame (ATD `frame_id - 1`), which is also the submission row.
Nothing here prints data: the functions return arrays and the caller decides what to do with them.
"""

import hashlib
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

HOME, AWAY = 0, 1
CACHE_VERSION = 1


@dataclass
class Tracking:
    x: np.ndarray        # [T, S] float32, normalised 0-1, NaN where the slot is empty
    y: np.ndarray        # [T, S]
    s: np.ndarray        # [T, S] speed in m/s as delivered, NaN where missing
    ball: np.ndarray     # [T, 2] float32, normalised, NaN where the ball is not tracked
    team: np.ndarray     # [S] int8, HOME or AWAY
    names: tuple         # (home name, away name)

    @property
    def n_frames(self):
        return self.x.shape[0]


def load_teams(meta_path):
    """(team per track slot, (home name, away name)). First team in the file is the home side."""
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    home_id = teams[0].get("id")
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    team = np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)
    names = tuple((t.findtext("Name") or "").strip() for t in teams)
    return team, names


def _parse_group(group):
    fields = group.split(",")
    out = [np.nan, np.nan, np.nan]
    for i in range(min(3, len(fields))):
        try:
            out[i] = float(fields[i])
        except ValueError:
            pass
    return out


def parse_raw(raw_path, n_slots, n_frames=None):
    """x, y, s [T, S] and ball [T, 2] from "<frame>:<x,y,s>;...:<ball>" lines.

    The common case (every slot written as three comma-separated numbers) goes through one
    numpy text parse per line; anything else falls back to a per-group parse.
    """
    ids, players, balls = [], [], []
    want = 3 * n_slots
    with open(raw_path) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            row = np.fromstring(parts[1].replace(";", ","), sep=",", dtype=np.float64) \
                if parts[1] else np.empty(0)
            if row.size != want:
                groups = [g for g in parts[1].split(";") if g != ""]
                row = np.full(want, np.nan)
                for slot, group in enumerate(groups[:n_slots]):
                    row[3 * slot:3 * slot + 3] = _parse_group(group)
            ids.append(int(parts[0]))
            players.append(row)
            balls.append(_parse_group(parts[2])[:2])

    ids = np.asarray(ids, dtype=np.int64) - 1          # frame_id counts from 1
    if n_frames is None:
        n_frames = int(ids.max()) + 1 if len(ids) else 0
    keep = (ids >= 0) & (ids < n_frames)
    ids = ids[keep]

    data = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    if len(ids):
        data[ids] = np.asarray(players, dtype=np.float32)[keep].reshape(-1, n_slots, 3)
        ball[ids] = np.asarray(balls, dtype=np.float32)[keep]
    return data[:, :, 0].copy(), data[:, :, 1].copy(), data[:, :, 2].copy(), ball


def _cache_key(paths, n_frames):
    h = hashlib.sha1()
    h.update("v{}|{}".format(CACHE_VERSION, n_frames).encode())
    for path in paths:
        stat = os.stat(path)
        h.update("{}|{}|{}".format(os.path.basename(path), stat.st_size, stat.st_mtime_ns).encode())
    return h.hexdigest()[:16]


def load_tracking(meta_path, raw_path, n_frames=None, cache_dir=None):
    """The ATD as a Tracking. With cache_dir, the parsed arrays are stored there as .npz."""
    team, names = load_teams(meta_path)
    cache = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, "atd_{}.npz".format(
            _cache_key((meta_path, raw_path), n_frames)))
        if os.path.isfile(cache):
            with np.load(cache) as z:
                return Tracking(z["x"], z["y"], z["s"], z["ball"], team, names)
    x, y, s, ball = parse_raw(raw_path, len(team), n_frames)
    if cache:
        np.savez(cache, x=x, y=y, s=s, ball=ball)
    return Tracking(x, y, s, ball, team, names)


def load_tags(path, fps=25):
    """Smart Tagging instances as [{code, start, end, tags}], frames half-open, text lower-cased."""
    out = []
    for instance in ET.parse(path).iter("instance"):
        start = float(instance.findtext("start"))
        end = float(instance.findtext("end"))
        out.append({
            "code": (instance.findtext("code") or "").strip().lower(),
            "start": int(round(start * fps)),
            "end": int(round(end * fps)),
            "tags": {(label.findtext("group") or "").strip().lower():
                     (label.findtext("text") or "").strip().lower()
                     for label in instance.findall("label")},
        })
    return out


def write_submission(path, labels):
    """frame,label CSV. Plain '\\n' line endings so the same labels always give the same bytes."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.write("".join("{},{}\n".format(i, label) for i, label in enumerate(labels)))
    return path
