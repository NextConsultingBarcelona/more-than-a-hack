# Ekstern validering (Sportec/IDSSE)

Den eneste lovlige måten å velge parametere på (regel R9, se `docs/organizer_answers.md`).
Bruker **bare åpne data**, aldri konkurransedataene.

**Datakilde:** Sportec/IDSSE, Bundesliga 2022/23, 7 kamper, CC-BY 4.0.
Bassek, M., Rein, R., Weber, H. et al. (2025). *An integrated dataset of spatiotemporal and event data in elite soccer.* Scientific Data 12, 195. doi:10.1038/s41597-025-04505-y
(Må krediteres i det tekniske dokumentet.)

## Hva skriptene gjør

| Skript | Gjør |
|---|---|
| `build_dataset.py` | Laster kamp(er) med kloppy, lager fasit (D / H / A), gjør dataene dårligere og skriver `ext_data/<kamp>/` |
| `score.py` | Scorer en mappe med prediksjons-CSV-er med arrangørens `evaluation/possession_metrics.py` (spellF1 + macroF1) |
| `sweep.py` | Tester mange parameterverdier og lager én tabell med spellF1 per kamp |
| `example_baseline.py` | Enkel eksempelmodell som viser formatet `predict(tracking, slots, **params)` |
| `config.py` | Alle innstillinger (kamper, seed, hvor dårlig dataene gjøres) |

Output per kamp i `ext_data/<kamp>/` (stoppes av `.gitignore`, skal aldri committes):

- `tracking.parquet`: `frame_id`, `ball_x`, `ball_y`, `sNN_x`, `sNN_y`, `sNN_s` (46 slots, 0–1, fart i m/s)
- `slots.csv`: `slot,team` (H/A)
- `gt.csv`: `frame,label` (H/A/D, og X mellom omgangene)
- `info.json`: seed, andel uten ball, antall slot-bytter osv.

## Installer (én gang)

Fra repo-mappen, i terminalen i VS Code:

```
python -m pip install -r requirements.txt -r external_validation/requirements-validation.txt
```

(På Mac kan det hete `python3` i stedet for `python`.)

## Kjør

```
python external_validation/build_dataset.py --selftest                      # 1 sek, kunstige data – sjekker at alt virker
python external_validation/build_dataset.py                                 # J03WPY (første gang: nedlasting, noen minutter)
python external_validation/build_dataset.py --matches all                   # alle 7 kamper
python external_validation/sweep.py external_validation/grids/example_baseline.json
```

Nedlastede filer caches av kloppy i `~/kloppy_cache` (utenfor repoet).
En hel kamp krever noen GB RAM under innlastingen.

## For Oskar: test din egen modell

1. Lag en funksjon `predict(tracking, slots, **params)` som returnerer én etikett per rad i `tracking`.
2. Kopier `grids/example_baseline.json`, og endre `model` (f.eks. `possession.hmm:predict`) og `grid`:
   ```json
   {"model": "possession.hmm:predict",
    "grid": {"switch_penalty": [2, 5, 10, 20], "dead_speed": [0.5, 1.0]}}
   ```
3. Kjør `python external_validation/sweep.py min_fil.json`.
4. Tabellen sorteres etter snitt spellF1 og lagres i `ext_data/results/`.

**Les tallene riktig:**

- Velg verdien som er god på de fleste kampene (høyt snitt, lav spredning, OK minimum), ikke den som vinner på én kamp.
- Tallene sammenligner parametere med hverandre. De er **ikke** et anslag på konkurransescoren: Sportecs possession-felt (DFL) er definert litt annerledes enn arrangørens (løse baller m.m.).
- `X`-radene mellom omgangene har NaN i all tracking. Modellen må tåle det.

## Kjente forenklinger i v1

- Ballen fjernes i blokker på 0,5–5 s til 41 % mangler. Hull starter 3x oftere i duell og 2x oftere nær mål. Tallene kommer fra offentlig dokumentasjon og er ikke målt på kampdataene.
- Ingen duplikat-slots og ingen støy på posisjonene.
- Pausen er 20 s (ikke ekte lengde). Den skal bare skille omgangene.
