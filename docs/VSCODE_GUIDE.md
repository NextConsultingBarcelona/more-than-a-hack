# Slik legger du sperrene inn i repoet (VS Code + GitHub, steg for steg)

Skrevet for deg som er ny på git og VS Code. Alt fungerer likt på Windows og Mac der ikke annet står.

## 0. Før du begynner
- **Sett repoet til privat først** (se steg 7), før du pusher noe. Repoet er public nå.
- Du trenger: VS Code, Git, Python 3.8+ og tilgang til repoet på GitHub.
  Windows: installer «Git for Windows» fra git-scm.com (Claude Code trenger den også).
- Ord du trenger: **repo** = prosjektmappen; **commit** = lagre en ferdig endring i historikken;
  **push** = sende commits til GitHub; **gren (branch)** = en egen arbeidskopi uten å røre hovedgrenen `main`.

## 1. Hent repoet til maskinen (hopp over hvis du allerede har det)
1. Åpne VS Code.
2. Trykk `Ctrl+Shift+P` (Mac: `Cmd+Shift+P`). Skriv `Git: Clone` og trykk Enter.
3. Lim inn `https://github.com/NextConsultingBarcelona/more-than-a-hack` og trykk Enter.
4. Velg en mappe å lagre i. **Ikke** i OneDrive/iCloud. Eksempel: `C:\Prosjekter` eller `~/Prosjekter`.
5. Klikk **Open** når VS Code spør. Logg inn på GitHub hvis du blir bedt om det.

## 2. Pakk ut sperre-pakken utenfor repoet
Pakk ut `mtah-guards.zip` til f.eks. Nedlastinger. Ikke inni repomappen.

## 3. Åpne terminalen i VS Code
Meny **Terminal → New Terminal**. Panelet nederst viser en sti. Den skal slutte på repomappen.

## 4. Lag en gren (så vi ikke rører main)
1. Klikk på grennavnet nederst til venstre i VS Code (står `main`).
2. Velg **Create new branch...**
3. Skriv `chat-a-claude-oppsett` og trykk Enter.
   [AVKLARES: branch + PR eller rett på main. Inntil Philip svarer: bruk gren.]

## 5. Legg filene inn
I terminalen (bytt ut stien med der du pakket ut; på Mac bruk `python3`):

    python "C:\Users\DITTNAVN\Downloads\mtah-guards\install.py" .

Punktumet på slutten betyr «denne mappen». Du får en liste over kopierte filer.
Finnes `CLAUDE.md` fra før, lages `CLAUDE.md.NY`. Slå sammen for hånd og slett `.NY`-filen.

## 6. Sett opp og test sperrene
    python tools/setup_guards.py
    python tools/test_guards.py

Første spør om datamappen. Lim inn stien til der ekte data ligger (utenfor repoet) eller trykk Enter.
Siste skal ende med `Alle sperretester bestått.` Gjør deretter den manuelle testen i `docs/SPERRETEST.md`.

## 7. Sett repoet til privat (krever admin på GitHub)
GitHub → repoet → **Settings** → **General** → helt nederst **Danger Zone** → **Change visibility** → **Make private**.
Be Oskar eller Philip gjøre det hvis du ikke ser knappen.

## 8. Commit
1. Klikk **Source Control**-ikonet i venstre sidefelt (tredje ikon, eller `Ctrl+Shift+G` / `Cmd+Shift+G`).
2. Du ser en liste over endrede filer. Hold musepekeren over **Changes** og klikk **+** (Stage All Changes).
3. **Bare Mac, én gang:** skriv i terminalen `git update-index --chmod=+x .githooks/pre-commit`
   (gjør at sperren kjører på Mac-maskinene til de andre).
4. Skriv en melding i boksen øverst, f.eks. `Legg til Claude-sperrer og CLAUDE.md`.
5. Klikk **Commit**. Pre-commit-sperren kjører nå av seg selv. Stopper den noe: les meldingen.

## 9. Push (send til GitHub)
Klikk **Publish Branch** (første gang) eller **Sync Changes**. Åpne repoet på GitHub: du får
«Compare & pull request». Velg den og be Philip eller Oskar godkjenne.

## 10. Alle andre i laget
Etter at de har hentet endringene (`git pull`): kjør `python tools/setup_guards.py` og
`python tools/test_guards.py`. Det må gjøres én gang per maskin.

## Daglige regler
- Ekte data ligger **utenfor** repoet. Sett `MTAH_DATA_DIR` **bare** i terminalen der du kjører kode på ekte data:
  Windows PowerShell `$env:MTAH_DATA_DIR = "C:\MTAH-data"`, Mac `export MTAH_DATA_DIR=~/MTAH-data`.
  Aldri som permanent miljøvariabel, og aldri i en terminal der du starter Claude.
- Start Claude Code **i repomappen**, så den finner `CLAUDE.md` og `.claude/settings.json`.
- Står du fast: stopp og spør Morten før du prøver noe nytt.
