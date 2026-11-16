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
experiments/              løse eksperimenter, ikke del av pipelinen
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
| `src/learned.py` | Lært evidens: trekk per frame + to tre-modeller (død/levende, hjemme/borte) kjørt i ren numpy fra `src/learned_model.json`. Trent på Sportec, aldri på konkurransekampen | Oskar | Testet på falske data, valgt på Sportec 15. nov |
| `src/dead_model.py` | Stoppdetektor: 279 kollektive trekk per frame + tre-ensemble i `models/dead_trees.npz`, kjørt i ren numpy. Stoppene går inn i tilstandsmodellen som dødball-evidens; ved urimelig utdata brukes modellens egen evidens. Trent på Sportec (`dead_training/`), aldri på konkurransekampen | Martin | Testet på falske data, validert på Sportec 16. nov |
| `src/team_model.py` | Andre hjemme/borte-modell: 151 trekk i tillegg til den første modellens 36 (lokal form rundt ballen, berøringer og kontakter, ballhull, lagene som kollektiv) + trær i `models/team_trees.npz`, kjørt i ren numpy. Trent på Sportec med eierskap der ukontrollerte berøringer ikke teller (`team_training/`). Feiler den, brukes den første modellen | Martin | Testet på falske data, validert på Sportec 16. nov |
| `src/selftrain.py` | Logistisk regresjon på algoritmiske pseudo-etiketter (ingen menneskelige etiketter) | Oskar | Testet på falske data |
| `src/config.py` | Alle parametere samlet. Fire er valgt på 5 Sportec-kamper (grid: `external_validation/grids/next_hmm.json`), 2 kamper holdt utenfor | Oskar | Tunet 15. nov |
| `src/pipeline.py` | Limer alt sammen og skriver innleverings-CSV-en | Oskar | Ikke kjørt på ekte data |
| `src/extval.py` | Adapter `predict(tracking, slots, **params)` for `external_validation/sweep.py` | Oskar | Testet på falske data |
| `tools/sanity_check.py` | Logisk sjekk av innleverings-CSV-en (format, X bare utenfor omgangene, D før avspark i 2. omgang), skriver bare OK/FEIL og SHA-256 | Oskar | Testet på falske data |
| `experiments/hint_v0_direction.py` | Grovt eksperiment (spillerretning + stillstand, uten ball). Ikke del av pipelinen | Oskar | Ikke testet |
| `tools/make_fake_match.py` | Syntetisk kamp i de ekte filformatene (ATD + Smart Tagging) med fasit, for testene av `src/` | Oskar | I bruk i testene |

## Kommandoer
| Hva | Kommando |
|---|---|
| Sett opp sperrer (én gang per maskin) | `python tools/setup_guards.py` |
| Test sperrene | `python tools/test_guards.py` |
| Sjekk at ingen datafiler er sporet i git | `python tools/check_no_data.py --all` |
| Kjør testene (falske data) | `python -m pytest -q` |
| Tren lært evidens + velg dekoderverdier på Sportec (skriver `src/learned_model.json`) | `python external_validation/train_emissions.py --jobs 6` |
| Stoppdetektoren mot modellens egen dødball-evidens, med ulike bilder av ballen under stopp (NEXT.md 5.8) | `python dead_training/stoppage_pictures.py --jobs 2` (krever `pack_sportec.py` og `train.py --lomo` først, se `dead_training/README.md`) |
| De tre versjonene side om side (tagget, med detektor, med detektor og andre H/A-modell), mot Sportecs eierskap som det er og uten ukontrollerte berøringer (NEXT.md 5.9) | `python team_training/compare.py --jobs 2` og `python team_training/compare.py --live --jobs 2` (se `team_training/README.md`) |
| Tidligere tuning av håndvektet modell (`learned=False`) | `python external_validation/sweep.py external_validation/grids/next_hmm.json` |
| Forsøk: tuning mot «kontrollert ballbesittelse» (ikke tatt i bruk) | sett `MIN_CONTROLLED_S = 3.0` i `external_validation/config.py`, bygg på nytt, kjør `sweep.py` med `grids/next_hmm_controlled.json` |
| Lag innleveringen (ekte data, kun mennesker, uten Claude) | `python -m src.pipeline --team-name NEXT --no-cache` |
| Samme, uten stoppdetektoren og den andre H/A-modellen (versjonen som først ble tagget `phase1-submission`) | `python -m src.pipeline --team-name NEXT --no-cache --tagged` |
| Samme, med den tidligere håndvektede modellen (reserve) | `python -m src.pipeline --team-name NEXT --no-cache --hand-weighted` |
| Sjekk innleverings-CSV-en (skriver bare OK/FEIL + SHA-256) | `python tools/sanity_check.py evaluation/results/NEXT.csv` |
| Falsk kamp i ekte filformat, for feilsøking | `python tools/make_fake_match.py --out test_data/fake_match --seed 0` |

Avhengigheter er låst med `==` i `requirements.txt` (pipeline + scorer), `requirements-dev.txt`
(tester) og `external_validation/requirements-validation.txt` (Sportec), testet med Python 3.13.5.
`Makefile` er bare snarveier (`make test`, `make fake`, `make submit`) til kommandoene over.

På Mac kan kommandoen være `python3`. [AVKLARES: Python-versjon og kommando]
