# Falske data-generator (v1)

Lager en **syntetisk** fotballkamp med kjent fasit, så laget kan teste koden uten å røre de ekte kampdataene.

> **Formål:** teste at koden virker (format, 199 319 rader, ingen krasj, at H/A/D-logikken oppfører seg).
> **Ikke** for å trene, tune eller velge modell. Det gjøres på åpne data fra andre kamper.
>
> Bygget kun fra offentlig dokumentasjon (DATA.md, CHALLENGE.md) og generelle fotballfakta.
> Ingenting er kalibrert mot den ekte kampen. Generatoren leser aldri `data/` eller `evaluation/gt/`.

## Kjøre

```bash
python -m test_data_generator --seed 42 --mode short   # ca. 10 min kamp, noen sekunder
python -m test_data_generator --seed 42 --mode full    # 199 319 frames, ca. 30 s
python -m pytest tests -q                              # alle tester (ca. 15 s)
python -m pytest tests -q -m "not slow"                # bare kort modus
```

Fra kode (uten filer):

```python
from test_data_generator import generate
m = generate(seed=42, mode="short")
m.tracking      # frame_id, ball_x, ball_y, s00_x, s00_y, s00_s, ... s45_s
m.slot_teams    # slot, team (H/A)
m.truth         # frame, label  (samme format som innleveringen)
m.events        # possessions, dødballer og løse baller (for QA)
```

## Output i `test_data/`

| Fil | Innhold |
|---|---|
| `fake_tracking.csv.gz` | `frame_id` (1…N), `ball_x/y` (0–1), `<slot>_x/_y` (0–1), `<slot>_s` (m/s). NaN = ikke sett |
| `fake_slot_teams.csv` | `slot,team` – 23 H og 23 A |
| `fake_truth.csv` | `frame,label` (H/A/D/X), samme kontrakt som innleveringen |
| `fake_events.csv` | Én rad per possession / død ball / løs ball, med start, slutt, eier og type |

Video-frame = `frame_id − 1`, akkurat som i DATA.md.

## Kampstruktur

| Modus | Frames | 1. omgang | 2. omgang (vurdert) | Avspark 2. omg |
|---|---|---|---|---|
| `full` | 199 319 | 11 185 – 90 651 | 110 808 – 190 197 | 113 742 |
| `short` | 16 800 | 250 – 7 749 | 8 750 – 16 549 | 9 050 |

Utenfor vurderte frames: etikett `X`, ball NaN, spillere står i/går til formasjon.

## Hvordan kampen lages

1. **`possession.py` – sann kamp.** Possessions på 5–30 s (lognormal, median 12 s). Hver slutter med
   - *bytte*: 1–3 s løs ball, etiketten beholdes til det andre laget har kontroll
   - *ball ut / frispark*: D i 4–15 s, så restart med fast eier:
     - innkast → motstanderen av siste berører
     - corner → angrepslaget (forsvarer siste berøring)
     - utspark → forsvarslaget (angriper siste berøring)
     - frispark → laget som ble felt
   - I tillegg: av og til løs ball midt i en possession som *samme* lag vinner tilbake (ingen bytte).
2. **`ball.py`** – ballen går mellom berøringer hvert 1–3 s og står stille på restart-punktet under D.
3. **`players.py`** – 2 × 11 spillere i formasjon (4-4-2 og 4-3-3), forskjøvet etter ballen, en går til ballen og en presser. Maks ca. 8 m/s. Side-bytte i pausen.
4. **`tracker.py` – måleutstyret.** Ballen mangler ca. 40 % av vurderte frames (oftere i dueller og nær mål). Spillere faller ut i 1–5 s og kommer tilbake på en *ny* slot. Av og til en duplikat-slot. Gir ca. 20–24 aktive slots.

Alle parametre ligger i `config.py` (`Params`). Samme seed + samme parametre = byte-identiske filer.

## Kjente forenklinger (v1)

- Ingen mål som standard (`Params(p_goal=...)` slår det på).
- Ballen beveger seg i rette linjer mellom berøringer, ingen høyde.
- Slot-navnene `s00`–`s45` er plassholdere. Sjekk mot arrangørens `starter/load_atd.py` og endre `SLOT_NAMES` i `config.py` hvis den bruker andre navn.
- Ingen identitetsbytter mellom spillere, ingen slots som hopper mellom lag.

## v2 (planlagt)

Rå filer i EPTS-format (XML-metadata + rå tekst) med Metricas åpne Sample Game 3 og `starter/load_atd.py` som formatmal, slik at testen går gjennom den ekte lasteren. Pluss Smart Tagging, homografi, identitetsbytter og dupliserte slots som hopper. Simuleringen (`possession.py`, `ball.py`, `players.py`) trenger ikke endres – bare en ny skriver ved siden av `writer.py`.
