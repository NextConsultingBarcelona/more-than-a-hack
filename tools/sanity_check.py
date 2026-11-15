"""Logical sanity check of a submission CSV. Prints only OK / FEIL per check, never the labels.

    python tools/sanity_check.py                       # evaluation/results/NEXT.csv
    python tools/sanity_check.py path/to/file.csv
    python tools/sanity_check.py pred.csv --meta fake_meta.json   # fake matches

Checks (submission checklist B3-B8): no BOM, '\\n' line endings, header, row count, frame order,
allowed labels, X exactly outside the assessed halves, and D on the assessed frames before the
second-half kick-off. Also prints the SHA-256 of the file, for the submission log.

On the real file this is run by a human. The output contains no match data.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import games  # noqa: E402

# Video frame of the second-half kick-off, from the organiser's public documentation
# (FAQ.md, starter/load_atd.py). Assessed frames before it are dead ball.
SECOND_HALF_KICKOFF = 113742


def early_start(labels, first, last, fps=25):
    """Coarse description of where the non-D frames of a pre-kick-off span lie. No numbers."""
    wrong = [i for i in range(first, last + 1) if labels[i] != "D"]
    if not wrong:
        return ""
    lead = last + 1 - wrong[0]                       # frames from the earliest one to the kick-off
    for seconds, text in ((1, "bare siste sekund"), (5, "siste 5 s"), (30, "siste 30 s")):
        if lead <= seconds * fps:
            break
    else:
        text = "tidligere enn 30 s"
    whole = wrong == list(range(wrong[0], last + 1))
    return "{} for avspark, {}".format(
        text, "sammenhengende fram til avspark" if whole else "i flere biter")


def check(path, n_frames, blocks, dead_spans):
    """[(name, ok)] for the file at path. blocks and dead_spans are inclusive (first, last)."""
    raw = Path(path).read_bytes()
    results = [("ingen BOM", not raw.startswith(b"\xef\xbb\xbf")),
               ("linjeskift er \\n", b"\r" not in raw)]
    lines = raw.decode("utf-8-sig").replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    results.append(("header er frame,label", bool(lines) and lines[0] == "frame,label"))
    rows = [line.split(",") for line in lines[1:]]
    results.append(("{} rader".format(n_frames), len(rows) == n_frames))
    well_formed = all(len(row) == 2 for row in rows)
    results.append(("frame er 0..N-1 i rekkefolge",
                    well_formed and all(row[0] == str(i) for i, row in enumerate(rows))))
    labels = [row[1] for row in rows] if well_formed else []
    results.append(("bare H, A, D, X", well_formed and set(labels) <= set("HADX")))
    if len(labels) != n_frames:
        return results

    graded = [False] * n_frames
    for first, last in blocks:
        for i in range(first, last + 1):
            graded[i] = True
    results.append(("ingen X i omgangene", all(lab != "X" for lab, g in zip(labels, graded) if g)))
    results.append(("bare X utenfor omgangene",
                    all(lab == "X" for lab, g in zip(labels, graded) if not g)))
    for first, last in dead_spans:
        where = early_start(labels, first, last)
        results.append(("D for avspark ({}-{}){}".format(first, last, ": ikke-D " + where if where else ""),
                        not where))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="?", default=str(ROOT / "evaluation" / "results" / "NEXT.csv"))
    parser.add_argument("--meta", help="json with n_frames and blocks (fake matches only)")
    args = parser.parse_args(argv)

    path = Path(args.csv)
    if not path.is_file():
        print("FEIL  finner ikke {}".format(path))
        return 1
    if args.meta:
        meta = json.loads(Path(args.meta).read_text())
        n_frames, blocks, dead_spans = meta["n_frames"], [tuple(b) for b in meta["blocks"]], []
    else:
        n_frames = games.source_frames("tactical")
        blocks = [(first, last) for _, first, last in games.half_video_blocks(games.GAME)]
        dead_spans = [(blocks[1][0], SECOND_HALF_KICKOFF - 1)]

    results = check(path, n_frames, blocks, dead_spans)
    for name, ok in results:
        print("{}  {}".format("OK  " if ok else "FEIL", name))
    print("sha256  {}".format(hashlib.sha256(path.read_bytes()).hexdigest()))
    failed = sum(not ok for _, ok in results)
    print("ALT OK" if not failed else "FEIL: {} sjekk(er) feilet".format(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
