#!/usr/bin/env python3
"""Setter opp sperrene på DENNE maskinen. Kjør én gang etter at du har hentet repoet:

    python tools/setup_guards.py        (på Mac kan det hete python3)

Gjør dette:
  1. slår på git-sperren (pre-commit) for repoet
  2. gjør pre-commit kjørbar (Mac)
  3. (valgfritt) husker hvor ekte data ligger, slik at Claude-sperren kan blokkere den mappen
  4. sjekker at alt henger sammen
"""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
ok = True


def say(status, msg):
    global ok
    if status == "FEIL":
        ok = False
    print("[%s] %s" % (status, msg))


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def main():
    if sys.version_info < (3, 8):
        say("FEIL", "Python 3.8 eller nyere trengs. Du har %s" % sys.version.split()[0])
    else:
        say("OK", "Python %s" % sys.version.split()[0])

    if git("rev-parse", "--is-inside-work-tree").returncode != 0:
        say("FEIL", "Dette er ikke en git-mappe. Kjør fra repoet.")
        return finish()

    git("config", "core.hooksPath", ".githooks")
    cfg = git("config", "--get", "core.hooksPath").stdout.strip()
    say("OK" if cfg == ".githooks" else "FEIL", "git core.hooksPath = %r" % cfg)

    hook = ROOT / ".githooks" / "pre-commit"
    if not hook.is_file():
        say("FEIL", ".githooks/pre-commit mangler")
    elif os.name != "nt":
        hook.chmod(hook.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        say("OK", "pre-commit er gjort kjørbar")
    else:
        say("OK", "pre-commit (Windows trenger ikke kjørbar-flagg lokalt)")

    try:
        json.loads((ROOT / ".claude" / "settings.json").read_text(encoding="utf-8"))
        say("OK", ".claude/settings.json er gyldig JSON")
    except Exception as e:
        say("FEIL", ".claude/settings.json: %s" % e)

    if (ROOT / "data").exists():
        say("ADVARSEL", "Det finnes en mappe data/ inni repoet. Ekte data skal ligge UTENFOR repoet.")

    if os.environ.get("MTAH_DATA_DIR"):
        say("ADVARSEL", "MTAH_DATA_DIR er satt i DENNE terminalen. Ikke start Claude Code herfra.")

    local = ROOT / ".claude" / "data_dirs.local.txt"
    if not local.exists():
        print("\nHvor ligger de ekte kampdataene på din maskin? (mappen UTENFOR repoet)")
        print("Dette lagres kun lokalt, aldri i git, og brukes til å SPERRE mappen for Claude.")
        ans = input("Lim inn mappestien, eller trykk Enter for å hoppe over: ").strip().strip('"')
        if ans:
            local.write_text("# Lokal fil, ikke i git. En mappesti per linje.\n" + ans + "\n",
                             encoding="utf-8")
            say("OK", "Lagret i .claude/data_dirs.local.txt")
        else:
            say("ADVARSEL", "Hoppet over. Kjør skriptet på nytt når dataene er lastet ned.")
    else:
        say("OK", ".claude/data_dirs.local.txt finnes allerede")
    return finish()


def finish():
    print()
    if ok:
        print("Ferdig. Neste steg: python tools/test_guards.py")
        return 0
    print("Noe feilet. Les [FEIL]-linjene over, og si fra til Morten.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
