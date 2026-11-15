# PROJECT_MAP – levende kart (Claude oppdaterer denne)

> Regler for denne filen (se også `CLAUDE.md`):
> - Oppdateres i **samme commit** som koden den beskriver.
> - Kun info fra kode og offentlig dokumentasjon. **Aldri funn fra ekte data.**
> - Avklaringer føres inn her først når et menneske har sagt svaret i chatten.

Sist oppdatert: 15. november 2026

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
test_data_generator/      lager falske data
external_validation/      tuning på åpne Sportec-kamper (eneste lovlige tuning, R9)
src/                      possession-pipelinen (se Moduler)
tests/                    pytest, kun falske data
test_data/                falske data, ikke i git
starter/ evaluation/      kopi av arrangørens repo
```
Legg til nye mapper og moduler her når de opprettes.

## Moduler
| Modul | Formål | Eier | Status |
|---|---|---|---|
| `src/io.py` | Leser ATD, metadata og Smart Tagging til numpy (med cache), skriver innleverings-CSV | Oskar | Testet på falske data |
| `src/features.py` | Per-frame-trekk: renset og glattet ball, nærmeste spiller per lag, ballfart, spillerfart, avsparkformasjon | Oskar | Testet på falske data |
| `src/deadball.py` | Evidens for død ball per frame (ball ute, ball i ro, avspark, dødballtagger, spillerfart) | Oskar | Testet på falske data |
| `src/possession.py` | H/A/D-tilstandsmodell: lag-evidens, Viterbi, minstevarighet, grensejustering, igangsetter etter stopp | Oskar | Testet på falske data, tunet på Sportec |
| `src/selftrain.py` | Logistisk regresjon på algoritmiske pseudo-etiketter (ingen menneskelige etiketter) | Oskar | Testet på falske data |
| `src/config.py` | Alle parametere samlet. Fire er valgt på 5 Sportec-kamper (grid: `external_validation/grids/next_hmm.json`), 2 kamper holdt utenfor | Oskar | Tunet 15. nov |
| `src/pipeline.py` | Limer alt sammen og skriver innleverings-CSV-en | Oskar | Ikke kjørt på ekte data |
| `src/extval.py` | Adapter `predict(tracking, slots, **params)` for `external_validation/sweep.py` | Oskar | Testet på falske data |
| `tools/make_fake_match.py` | Syntetisk kamp i de ekte filformatene (ATD + Smart Tagging) med fasit, for testene av `src/` | Oskar | I bruk i testene |

## Kommandoer
| Hva | Kommando |
|---|---|
| Sett opp sperrer (én gang per maskin) | `python tools/setup_guards.py` |
| Test sperrene | `python tools/test_guards.py` |
| Sjekk at ingen datafiler er sporet i git | `python tools/check_no_data.py --all` |
| Kjør testene (falske data) | `python -m pytest -q` |
| Tun possession-modellen på Sportec | `python external_validation/sweep.py external_validation/grids/next_hmm.json` |
| Lag innleveringen (ekte data, kun mennesker, uten Claude) | `python -m src.pipeline --team-name next --no-cache` |

På Mac kan kommandoen være `python3`. [AVKLARES: Python-versjon og kommando]
