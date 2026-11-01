# PROJECT_MAP – levende kart (Claude oppdaterer denne)

> Regler for denne filen (se også `CLAUDE.md`):
> - Oppdateres i **samme commit** som koden den beskriver.
> - Kun info fra kode og offentlig dokumentasjon. **Aldri funn fra ekte data.**
> - Avklaringer føres inn her først når et menneske har sagt svaret i chatten.

Sist oppdatert: 1. november 2026

## Åpne avklaringer
| Tema | Status | Svar | Hvem |
|---|---|---|---|
| Python-versjon | [AVKLARES: Python-versjon] | | Philip |
| Pakkehåndtering (pip/conda, requirements.txt) | [AVKLARES: pip eller conda] | | Philip |
| Branches + PR eller rett på main | [AVKLARES: branch-strategi] | | Philip |
| Repo privat frem til innlevering | [AVKLARES: ok?] | | Philip/Oskar |
| Gråsone G1–G4 (se kontekstdokumentet) | Sendes samlet til arrangør | | Morten |

## Struktur
```
CLAUDE.md                 låste regler (mennesker)
.claude/                  Claude Code-sperrer (settings.json, hooks/)
.githooks/                pre-commit som stopper datafiler
docs/                     dokumentasjon (dette kartet, sperretest, guide)
tools/                    sjekk- og oppsettskript for sperrene
test_data_generator/      lager falske data (ikke laget ennå)
test_data/                falske data, ikke i git
starter/ evaluation/      kopi av arrangørens repo
```
Legg til nye mapper og moduler her når de opprettes.

## Moduler
| Modul | Formål | Eier | Status |
|---|---|---|---|
| (ingen ennå) | | | |

## Kommandoer
| Hva | Kommando |
|---|---|
| Sett opp sperrer (én gang per maskin) | `python tools/setup_guards.py` |
| Test sperrene | `python tools/test_guards.py` |
| Sjekk at ingen datafiler er sporet i git | `python tools/check_no_data.py --all` |

På Mac kan kommandoen være `python3`. [AVKLARES: Python-versjon og kommando]
