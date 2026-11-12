#!/usr/bin/env python3
"""PreToolUse-hook for Claude Code: hindrer at kampdata og låste filer berøres.

Exit 2 = BLOKKER (Claude Code viser stderr til Claude).  Exit 0 = ingen innvending.
Kun standardbibliotek. Fungerer på Windows og Mac. Ikke vanntett mot bevisst omgåelse
(f.eks. kodede stinavn) - den stopper uhell.
"""
import json
import os
import re
import shlex
import sys
from pathlib import Path

try:
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Relative til prosjektroten. Kampdata skal ligge UTENFOR repoet, men disse sperres uansett.
REL_FORBIDDEN = ["data", "evaluation/gt"]
# Filer Claude ikke skal kunne endre (verktøy som redigerer filer).
PROTECTED_EDIT = ["CLAUDE.md", ".claude", ".githooks",
                  "tools/check_no_data.py", "tools/setup_guards.py", "tools/test_guards.py"]
# Bash/PowerShell: skriving mot disse stoppes.
PROTECTED_BASH = ["claude.md", ".claude/", ".githooks"]

VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv"}
DATA_EXT = {".csv", ".txt", ".xml", ".json", ".dat", ".parquet", ".npy", ".npz",
            ".pkl", ".pickle", ".h5", ".hdf5", ".feather", ".zip", ".gz", ".7z", ".tar"}
NAME_RE = re.compile(r"(tracking|homograph|smart[_\- ]?tagging|(?<![a-z])atd(?![a-z])|epts)", re.I)
SAFE_DIRS = ("test_data", "test_data_generator")  # falske data: navnemønstre gjelder ikke her

FILE_CMDS = {"cat", "type", "head", "tail", "more", "less", "ls", "dir", "tree", "find",
             "cd", "pushd", "cp", "copy", "mv", "move", "xxd", "od", "strings", "wc", "du",
             "stat", "file", "grep", "rg", "awk", "sed", "open", "start", "explorer", "code",
             "get-content", "gc", "get-childitem", "gci", "select-string", "sls",
             "set-location", "sl", "import-csv", "xargs", "tar", "unzip", "zip"}
WRITE_RE = re.compile(
    r"(>|\bsed\s+-i|\btee\b|\brm\b|\bdel\b|\berase\b|\bmv\b|\bmove\b|\bren\b|\brename\b|"
    r"\bcp\b|\bcopy\b|\btruncate\b|set-content|add-content|out-file|remove-item|move-item|"
    r"copy-item|new-item|\bchmod\b|\battrib\b|\bicacls\b)")
GIT_BYPASS_RE = re.compile(
    r"(--no-verify|core\.hookspath|git\s+add\s+(\S+\s+)*(-f\b|--force\b)|git\s+commit\s+\S*\s*-n\b)")


def block(why):
    sys.stderr.write(
        "BLOKKERT av MTAH-sperren: " + why + "\n"
        "Ikke prøv å omgå dette. Forklar for brukeren hva du ville gjort, og spør.\n"
        "Reglene står i CLAUDE.md. Kampdata må aldri til Claude.\n")
    sys.exit(2)


def norm(s):
    return s.replace("\\", "/").lower()


def real(p):
    return norm(os.path.realpath(os.path.expanduser(os.path.expandvars(p))))


def inside(path_n, root_n):
    root_n = root_n.rstrip("/")
    return path_n == root_n or path_n.startswith(root_n + "/")


def project_root(inp):
    base = os.environ.get("CLAUDE_PROJECT_DIR") or inp.get("cwd") or os.getcwd()
    return Path(base).resolve()


def forbidden_roots(root):
    roots = [real(str(root / rel)) for rel in REL_FORBIDDEN]
    local = root / ".claude" / "data_dirs.local.txt"
    if local.is_file():
        for line in local.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                roots.append(real(line))
    env = os.environ.get("MTAH_DATA_DIR")
    if env:
        roots.append(real(env))
    return sorted(set(roots))


def path_variants(abs_n):
    """Ulike skrivemåter av samme absolutte sti slik de kan stå i en kommando."""
    v = {abs_n}
    m = re.match(r"^([a-z]):/(.*)$", abs_n)
    if m:  # c:/users/x -> /c/users/x (Git Bash)
        v.add("/%s/%s" % (m.group(1), m.group(2)))
    home = norm(os.path.expanduser("~"))
    if abs_n.startswith(home + "/"):
        v.add("~/" + abs_n[len(home) + 1:])
    return v


def is_datalike_name(name_n):
    ext = os.path.splitext(name_n)[1]
    if ext in VIDEO_EXT:
        return True
    return ext in DATA_EXT and bool(NAME_RE.search(os.path.basename(name_n)))


def under_safe_dir(path_n):
    return any(("/" + d + "/") in ("/" + path_n) or path_n.startswith(d + "/") for d in SAFE_DIRS)


def resolve_input_path(raw, cwd):
    p = os.path.expanduser(os.path.expandvars(raw))
    if not os.path.isabs(p):
        p = os.path.join(cwd, p)
    return real(p)


def check_file_tool(tool, ti, root, cwd, roots):
    raw_paths = [ti.get(k) for k in ("file_path", "notebook_path", "path") if ti.get(k)]
    is_search = tool in ("Grep", "Glob")
    if is_search and not raw_paths:
        raw_paths = [cwd]
    for raw in raw_paths:
        p = resolve_input_path(raw, cwd)
        for r in roots:
            if inside(p, r):
                block("%s mot en forbudt datamappe (%s)." % (tool, raw))
            if is_search and inside(r, p) and os.path.exists(r):
                block("%s fra en mappe som inneholder en datamappe. Søk i en konkret undermappe "
                      "(f.eks. tools/, docs/, starter/)." % tool)
        if tool == "Read" and is_datalike_name(p) and not under_safe_dir(p):
            block("Filen ser ut som kampdata (video/tracking/homografi/smart tagging): %s" % raw)
        if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
            for prot in PROTECTED_EDIT:
                if inside(p, real(str(root / prot))):
                    block("%s er låst. Bare mennesker endrer den." % prot)
    for key in ("pattern", "glob"):
        val = ti.get(key)
        if isinstance(val, str) and is_search:
            t = norm(val)
            if re.search(r"(?<![\w.\-])(data|gt)/", t) or "evaluation/gt" in t:
                block("Søkemønsteret peker mot en datamappe.")


def check_shell(command, root, roots):
    if os.environ.get("MTAH_DATA_DIR"):
        block("MTAH_DATA_DIR er satt i miljøet Claude Code kjører i. Avslutt Claude, åpne en terminal "
              "uten den variabelen, og start på nytt. (Ekte data kjøres i egen terminal uten Claude.)")
    text = norm(command)
    text_alt = re.sub(r"(?<![\w.])/([a-z])/", r"\1:/", text)

    if "mtah_data_dir" in text:
        block("Kommandoen refererer til MTAH_DATA_DIR (stien til ekte data).")
    for r in roots:
        for variant in path_variants(r):
            if variant in text or variant in text_alt:
                block("Kommandoen peker mot en forbudt datamappe.")
    if re.search(r"(?<![\w.\-])data/", text) or re.search(r"(?<![\w.\-])gt/", text) \
            or "evaluation/gt" in text:
        block("Kommandoen peker mot data/ eller evaluation/gt/.")

    # Bare mappenavn som argument til fil-kommandoer: "cd data", "ls gt"
    for seg in re.split(r"[;&|\n]+", command):
        try:
            toks = shlex.split(seg, posix=False)
        except ValueError:
            toks = seg.split()
        toks = [t.strip("\"'") for t in toks]
        while toks and re.match(r"^\w+=", toks[0]):
            toks = toks[1:]
        if not toks:
            continue
        first = os.path.basename(norm(toks[0]))
        first = first[:-4] if first.endswith(".exe") else first
        if first in FILE_CMDS:
            for t in toks[1:]:
                t_n = norm(t).rstrip("/")
                if t_n in ("data", "gt", "evaluation/gt", "./data", "./gt"):
                    block("Kommandoen peker mot en forbudt datamappe (%s)." % t)

    for tok in re.findall(r"[^\s\"'=;|&<>()]+", text):
        if is_datalike_name(tok) and "test_data" not in tok:
            block("Kommandoen peker mot en fil som ser ut som kampdata: %s" % tok)

    if GIT_BYPASS_RE.search(text):
        block("Git-sperrer skal ikke omgås (--no-verify, core.hooksPath, add -f).")
    if any(p in text for p in PROTECTED_BASH) and WRITE_RE.search(text):
        block("Kommandoen ser ut til å endre en låst fil (CLAUDE.md, .claude/ eller .githooks/).")


def main():
    try:
        inp = json.load(sys.stdin)
    except Exception:
        block("Kunne ikke lese hook-input. Sperren feiler lukket.")
    tool = inp.get("tool_name", "")
    ti = inp.get("tool_input") or {}
    root = project_root(inp)
    cwd = inp.get("cwd") or str(root)
    roots = forbidden_roots(root)
    if tool in ("Read", "Edit", "Write", "MultiEdit", "NotebookEdit", "Grep", "Glob"):
        check_file_tool(tool, ti, root, cwd, roots)
    elif tool in ("Bash", "PowerShell"):
        check_shell(ti.get("command", "") or "", root, roots)
    sys.exit(0)


if __name__ == "__main__":
    main()
