# Overlevering fra Morten (15. nov 2026)

Morten har eksamen fra 16. nov og har svært begrenset tid. Oskar og Martin tar over resten. **Fristen er 17. nov** (laget sikter på å levere 16. nov). Alt som rører ekte kampdata, kjøres i en vanlig terminal **uten Claude** (R1).

## 1. Hva jeg har laget og hvor det ligger

| Hva | Hvor | Status |
|---|---|---|
| CLAUDE.md + sperrer | `CLAUDE.md`, `.claude/settings.json`, `.claude/hooks/guard_paths.py`, `.githooks/pre-commit`, `tools/` | På `main` |
| Generator for falske testdata | `test_data_generator/`, `tests/` (output i `test_data/`, ikke i git) | Grenen `test-data-generator`, PR til `main` i dag |
| Ekstern validering (Sportec) | `external_validation/` (README forklarer alt) | Grenen `ekstern-validering`, PR til `main` i dag |
| Sjekkliste før innlevering | `docs/SUBMISSION_CHECKLIST.md` | Samme PR |
| Teknisk dokument (utkast) | `docs/NEXT.md`: plassholdere `[OSKAR FYLLER INN]`, `[MARTIN FYLLER INN]` og `[SJEKK]` | Samme PR |
| Svar fra arrangøren | `docs/organizer_answers.md` | Samme PR |

## 2. Oskar: slik tuner du parametere (obligatorisk etter R9)

```
python -m pip install -r requirements.txt -r external_validation/requirements-validation.txt
python external_validation/build_dataset.py --selftest      # sjekker oppsettet (1 sek)
python external_validation/build_dataset.py --matches all   # laster ned 7 Sportec-kamper
python external_validation/sweep.py min_grid.json           # tester alle kombinasjoner
```

- `min_grid.json`: kopi av `external_validation/grids/example_baseline.json`, med `"model": "<modul>:predict"` og `"grid": {"switch_penalty": [...], ...}`.
- Modellen må ha formen `predict(tracking, slots, **params)` og gi én etikett per rad.
- **Du får:** en tabell per parameterkombinasjon med snitt spellF1, spredning, minimum, snitt macroF1 og spellF1 per kamp. Den lagres i `ext_data/results/` (ikke i git). Velg verdier som er gode på de fleste kampene. Tallene kan limes rett inn i `NEXT.md` seksjon 5.6, siden det er åpne data.
- **Aldri** velg parametere ut fra de 5 offentlige minuttene eller videoen. Commit og tag parameterne **før** 5-minutterssjekken.

## 3. Hva som gjenstår, og hvem som eier det

| Oppgave | Eier |
|---|---|
| Tuning av possession-modellen på Sportec (punkt 2) | **Oskar** |
| Reproduksjonssjekk fra ren klon, Windows + Mac, samme SHA-256 (sjekkliste C1–C7) | **Martin** |
| Ferdigstille `docs/NEXT.md` (egne plassholdere, deretter alle `[SJEKK]`) | **Oskar + Martin** |
| Gå gjennom sjekklisten, samt innlevering via Google-skjemaet | **Martin** |

## 4. E-post til arrangøren

Sendt **12. nov** (lagendring 5 → 3 personer + gråsonene G1–G5). **Besvart 15. nov**, se `docs/organizer_answers.md` avsnitt 2:
- **Ingen nye avklaringer.** G1–G5 er fortsatt åpne, og de trygge valgene gjelder: ingen tall fra ekte data til sky-AI, og Claude leser ikke 5-minuttersfasiten.
- **Lagendringen bekreftes hvis vi går videre** til neste fase. Sjekklistepunkt A2 er derfor ikke helt lukket.
- **«Alt som ikke er eksplisitt tillatt, er ikke tillatt.»** Er dere i tvil om noe: ikke gjør det.

Nye svar legges inn i `docs/organizer_answers.md` (uten data). [SJEKK: hvem følger med på innboksen mens Morten har eksamen?]

## 5. Kjente punkter (fra QA)

- **Sperre-hooken matcher stier som tekst.** Den stopper uhell, ikke bevisst omgåelse (f.eks. kodede stinavn).
- **VS Code varsler «Property onFailure is not allowed»** i `.claude/settings.json` (varselet vises på linje 29, nøkkelen står på linje 31). Det er bare skjemaet til VS Code som ikke kjenner nøkkelen. Hooken virker likevel. Ikke endre filen.
- **`tools/check_no_data.py` har sin egen liste** over sperrede mapper (`FORBIDDEN_DIRS = ("data/", "evaluation/gt/")`). Listen mangler `evaluation/results/` og `ext_data/`. Dette dekkes i dag bare av `.gitignore` (`ext_data/`, `*.parquet`, `evaluation/results/*.csv` unntatt `baseline_*`). Anbefaling: et menneske legger begge mappene til i listen. Filen er låst for Claude.
- **`requirements.txt` er låst** med `==` (15. nov, Python 3.13.5). C1 må fortsatt verifiseres fra ren klon på Windows.
- **Arrangørens 5-minuttersfasit og baseline-svarene** (`evaluation/gt/`, `evaluation/results/baseline_*.csv`) ligger i repoet fra den første kopien av arrangørens repo. Det er arrangørens egne offentlige filer, så det er OK etter D5. Ikke commit nye filer der.

