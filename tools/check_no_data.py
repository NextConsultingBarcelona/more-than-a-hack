#!/usr/bin/env python3
"""Stopper commit av kampdata.

Bruk:  python tools/check_no_data.py          (sjekker det som er klart for commit)
       python tools/check_no_data.py --all    (sjekker alle filer git sporer)
Exit 1 = funnet noe som ikke skal i repoet.
"""
import re
import subprocess
import sys
from pathlib import PurePosixPath

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Eksakte repo-relative stier (med /) som et MENNESKE har vurdert og godkjent.
ALLOWLIST = set()

VIDEO = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv"}
DATA_EXT = {".csv", ".txt", ".xml", ".json", ".dat", ".parquet", ".npy", ".npz",
            ".pkl", ".pickle", ".h5", ".hdf5", ".feather", ".zip", ".gz", ".7z", ".tar"}
NAME_RE = re.compile(r"(tracking|homograph|smart[_\- ]?tagging|(?<![a-z])atd(?![a-z])|epts)", re.I)
FORBIDDEN_DIRS = ("data/", "evaluation/gt/")
FORBIDDEN_PARTS = {"real_data", "raw_data"}
SAFE_PREFIXES = ("test_data/", "test_data_generator/")
DATA_LIMIT = 1_000_000      # 1 MB for csv/txt/json/xml osv.
ANY_LIMIT = 5_000_000       # 5 MB for alt


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, check=True).stdout


def staged_files():
    out = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    return [p for p in out.decode("utf-8", "replace").split("\0") if p]


def all_files():
    out = git("ls-files", "-z")
    return [p for p in out.decode("utf-8", "replace").split("\0") if p]


def size_of(path, use_index):
    try:
        if use_index:
            return int(git("cat-file", "-s", ":" + path).decode().strip())
        return int(git("cat-file", "-s", "HEAD:" + path).decode().strip())
    except Exception:
        return 0


def problems(path, use_index):
    p = path.replace("\\", "/")
    low = p.lower()
    if p in ALLOWLIST:
        return []
    out = []
    ext = PurePosixPath(low).suffix
    if low.startswith(FORBIDDEN_DIRS) or FORBIDDEN_PARTS & set(low.split("/")):
        out.append("ligger i en datamappe")
    if ext in VIDEO:
        out.append("er en videofil")
    if ext == ".xml":
        out.append("xml-filer (tracking/smart tagging) skal ikke i repoet")
    if ext in DATA_EXT and NAME_RE.search(PurePosixPath(low).name) \
            and not low.startswith(SAFE_PREFIXES):
        out.append("navnet ligner tracking/homografi/smart tagging-data")
    size = size_of(path, use_index)
    if ext in DATA_EXT and size > DATA_LIMIT:
        out.append("datafil på %.1f MB (grense 1 MB)" % (size / 1e6))
    elif size > ANY_LIMIT:
        out.append("stor fil på %.1f MB (grense 5 MB)" % (size / 1e6))
    return out


def main():
    use_all = "--all" in sys.argv
    try:
        files = all_files() if use_all else staged_files()
    except Exception as e:
        print("check_no_data: kunne ikke spørre git (%s). Stoppet." % e, file=sys.stderr)
        return 1
    bad = []
    for f in files:
        reasons = problems(f, use_index=not use_all)
        if reasons:
            bad.append((f, reasons))
    if not bad:
        print("check_no_data: OK (%d filer sjekket)" % len(files))
        return 0
    print("\nCOMMIT STOPPET: disse filene ser ut som kampdata eller data-avledet innhold:\n",
          file=sys.stderr)
    for f, reasons in bad:
        print("  %s\n      -> %s" % (f, "; ".join(reasons)), file=sys.stderr)
    print("\nFjern dem fra commit:  git restore --staged <fil>\n"
          "Er det falske testdata? Legg dem i test_data/ (ikke i git).\n"
          "Er du helt sikker på at en fil er trygg? Spør Morten. Bare mennesker kan\n"
          "legge den i ALLOWLIST i tools/check_no_data.py.\n", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
