#!/usr/bin/env python3
"""Automatisk test av sperrene. Bruker bare dummy-filer i en midlertidig mappe.

    python tools/test_guards.py

Del A: Claude-hooken (.claude/hooks/guard_paths.py)
Del B: git pre-commit (tools/check_no_data.py + .githooks/pre-commit)
Det er IKKE ekte data involvert noe sted.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
HOOK = REPO / ".claude" / "hooks" / "guard_paths.py"
failures = []


def report(name, passed, detail=""):
    print("[%s] %s%s" % ("OK" if passed else "FEIL", name, ("  -> " + detail) if detail else ""))
    if not passed:
        failures.append(name)


def run_hook(project, tool, tool_input, extra_env=None, clear_data_env=True):
    env = dict(os.environ)
    if clear_data_env:
        env.pop("MTAH_DATA_DIR", None)
    env["CLAUDE_PROJECT_DIR"] = str(project)
    env.update(extra_env or {})
    payload = {"tool_name": tool, "tool_input": tool_input, "cwd": str(project)}
    return subprocess.run([sys.executable, str(HOOK)], input=json.dumps(payload), text=True,
                          capture_output=True, env=env).returncode


def part_a():
    print("\n== Del A: Claude-hook ==")
    tmp = Path(tempfile.mkdtemp(prefix="mtah_test_"))
    try:
        proj = tmp / "repo"
        (proj / "data").mkdir(parents=True)
        (proj / "evaluation" / "gt").mkdir(parents=True)
        (proj / "tools").mkdir()
        (proj / "src").mkdir()
        (proj / "data" / "dummy.txt").write_text("DUMMY-HEMMELIG-12345")
        (proj / "evaluation" / "gt" / "dummy.csv").write_text("a,b")
        (proj / "CLAUDE.md").write_text("x")
        (proj / "src" / "ok.py").write_text("print(1)")
        outside = tmp / "ekte_data_utenfor"
        outside.mkdir()
        (outside / "dummy.txt").write_text("DUMMY")
        (proj / ".claude").mkdir()
        (proj / ".claude" / "data_dirs.local.txt").write_text(str(outside) + "\n")

        d = str(proj / "data" / "dummy.txt")
        cases = [
            ("Read data/dummy.txt", "Read", {"file_path": d}, 2),
            ("Read relativ sti data/dummy.txt", "Read", {"file_path": "data/dummy.txt"}, 2),
            ("Read evaluation/gt", "Read", {"file_path": str(proj / "evaluation/gt/dummy.csv")}, 2),
            ("Read datamappe utenfor repo", "Read", {"file_path": str(outside / "dummy.txt")}, 2),
            ("Read videofil", "Read", {"file_path": str(proj / "src" / "kamp.mp4")}, 2),
            ("Read vanlig kode", "Read", {"file_path": str(proj / "src" / "ok.py")}, 0),
            ("Grep inni data/", "Grep", {"pattern": "x", "path": str(proj / "data")}, 2),
            ("Grep fra rot (data/ finnes)", "Grep", {"pattern": "x"}, 2),
            ("Grep i src/", "Grep", {"pattern": "x", "path": str(proj / "src")}, 0),
            ("Glob data/**", "Glob", {"pattern": "data/**"}, 2),
            ("Edit CLAUDE.md", "Edit", {"file_path": str(proj / "CLAUDE.md")}, 2),
            ("Write .claude/settings.json", "Write", {"file_path": str(proj / ".claude/settings.json")}, 2),
            ("Write vanlig fil", "Write", {"file_path": str(proj / "src" / "ny.py")}, 0),
            ("Bash cat data/dummy.txt", "Bash", {"command": "cat data/dummy.txt"}, 2),
            ("Bash type data\\dummy.txt", "Bash", {"command": "type data\\dummy.txt"}, 2),
            ("Bash ls data", "Bash", {"command": "ls data"}, 2),
            ("Bash cd data && cat dummy.txt", "Bash", {"command": "cd data && cat dummy.txt"}, 2),
            ("Bash python -c open('data/..')", "Bash",
             {"command": "python -c \"print(open('data/dummy.txt').read())\""}, 2),
            ("Bash Get-Content PowerShell", "Bash", {"command": "Get-Content .\\data\\dummy.txt"}, 2),
            ("Bash absolutt sti til data", "Bash", {"command": "cat " + d}, 2),
            ("Bash datamappe utenfor repo", "Bash", {"command": "cat " + str(outside / "dummy.txt")}, 2),
            ("Bash echo $MTAH_DATA_DIR", "Bash", {"command": "echo $MTAH_DATA_DIR"}, 2),
            ("Bash sed -i CLAUDE.md", "Bash", {"command": "sed -i s/a/b/ CLAUDE.md"}, 2),
            ("Bash echo > CLAUDE.md", "Bash", {"command": "echo hei > CLAUDE.md"}, 2),
            ("Bash git commit --no-verify", "Bash", {"command": "git commit --no-verify -m x"}, 2),
            ("Bash git add -f data/x", "Bash", {"command": "git add -f src/x"}, 2),
            ("Bash vanlig: ls src", "Bash", {"command": "ls src"}, 0),
            ("Bash vanlig: python -m pytest", "Bash", {"command": "python -m pytest -q"}, 0),
            ("Bash vanlig: cat CLAUDE.md (lese er ok)", "Bash", {"command": "cat CLAUDE.md"}, 0),
            ("Bash vanlig: git commit -m 'update data loader'", "Bash",
             {"command": "git commit -m 'update data loader'"}, 0),
            ("Bash test_data er lov", "Bash", {"command": "ls test_data/"}, 0),
        ]
        for name, tool, ti, want in cases:
            got = run_hook(proj, tool, ti)
            report(name, got == want, "" if got == want else "forventet %d, fikk %d" % (want, got))

        got = run_hook(proj, "Bash", {"command": "ls src"}, extra_env={"MTAH_DATA_DIR": str(outside)},
                       clear_data_env=False)
        report("Bash nektes når MTAH_DATA_DIR er satt", got == 2, "" if got == 2 else "fikk %d" % got)

        env = dict(os.environ)
        env["CLAUDE_PROJECT_DIR"] = str(proj)
        got = subprocess.run([sys.executable, str(HOOK)], input="ikke json", text=True,
                             capture_output=True, env=env).returncode
        report("Ugyldig input -> blokkert (fail closed)", got == 2, "" if got == 2 else "fikk %d" % got)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def git(cwd, *args, env=None):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env)


def part_b():
    print("\n== Del B: git pre-commit ==")
    if shutil.which("git") is None:
        report("git er installert", False, "installer git og kjør på nytt")
        return
    tmp = Path(tempfile.mkdtemp(prefix="mtah_gittest_"))
    try:
        (tmp / "tools").mkdir()
        (tmp / ".githooks").mkdir()
        shutil.copy(REPO / "tools" / "check_no_data.py", tmp / "tools" / "check_no_data.py")
        hook = tmp / ".githooks" / "pre-commit"
        shutil.copy(REPO / ".githooks" / "pre-commit", hook)
        if os.name != "nt":
            hook.chmod(0o755)
        git(tmp, "init", "-q")
        git(tmp, "config", "core.hooksPath", ".githooks")
        git(tmp, "config", "user.name", "test")
        git(tmp, "config", "user.email", "test@example.com")
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"

        def try_commit(files, label, should_pass):
            for rel, content in files.items():
                p = tmp / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, bytes):
                    p.write_bytes(content)
                else:
                    p.write_text(content)
                git(tmp, "add", "-f", rel)
            r = git(tmp, "commit", "-q", "-m", label, env=env)
            passed = (r.returncode == 0) == should_pass
            report(label, passed, "" if passed else (r.stderr.strip()[-200:] or r.stdout.strip()[-200:]))
            if r.returncode != 0:
                git(tmp, "reset", "-q")
                for rel in files:
                    try:
                        (tmp / rel).unlink()
                    except OSError:
                        pass

        try_commit({"README.md": "hei"}, "vanlig fil committes", True)
        try_commit({"data/dummy.txt": "DUMMY-HEMMELIG-12345"}, "data/dummy.txt stoppes (git add -f)", False)
        try_commit({"evaluation/gt/dummy.csv": "a,b"}, "evaluation/gt/ stoppes", False)
        try_commit({"src/kamp.mp4": b"x"}, "videofil stoppes", False)
        try_commit({"src/homography_frames.csv": "a,b"}, "homografi-csv stoppes", False)
        try_commit({"src/stor.csv": "a,b\n" * 400_000}, "stor csv (>1 MB) stoppes", False)
        try_commit({"src/liten.csv": "a,b\n1,2\n"}, "liten csv er ok", True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    part_a()
    part_b()
    print()
    if failures:
        print("FEILET (%d): %s" % (len(failures), "; ".join(failures)))
        return 1
    print("Alle sperretester bestått.")
    print("Husk den manuelle testen med Claude Code selv: docs/SPERRETEST.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
