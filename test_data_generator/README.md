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

## Bruk i din egen kode

Kort oppsummert: **lag en falsk kamp → kjør koden din på den → sjekk at svaret har riktig format og oppfører seg fornuftig.**

### Regel 1: Ikke skriv kolonnenavn rett inn i koden

Slot-navnene i de ekte dataene kommer fra arrangørens metadata (lest via kloppy i `starter/load_atd.py`) og er ukjente før dataene kommer. Navnene her (`s00`–`s45`) er plassholdere.

Finn derfor kolonnene via slot→lag-tabellen, ikke ved å skrive `"s05_x"`:

```python
# Feil – virker bare på falske data
hjemme_x = tracking[["s00_x", "s01_x", "s02_x"]]

# Riktig – virker på både falske og ekte data
hjemme_slots = slot_teams.loc[slot_teams["team"] == "H", "slot"]
hjemme_x = tracking[[f"{s}_x" for s in hjemme_slots]]
```

Samme gjelder antall: ikke anta 46 slots eller 22 aktive spillere. Les det fra dataene.

### Regel 2: Husk forskyvningen på 1

`frame_id` i tracking starter på **1**. Rader i innleveringen (`frame`) starter på **0**.

```python
video_frame = tracking["frame_id"] - 1
```

### Eksempel A: Hent en falsk kamp direkte i Python (raskest)

Ingen filer, alt i minnet:

```python
from test_data_generator import generate

m = generate(seed=42, mode="short")   # ca. 10 min kamp, noen sekunder
tracking   = m.tracking                # det lasteren vår skal returnere
slot_teams = m.slot_teams
fasit      = m.truth                   # frame,label
```

### Eksempel B: Les filene i `test_data/`

Lag filene først med `python -m test_data_generator --seed 42 --mode short`, så:

```python
import pandas as pd

tracking   = pd.read_csv("test_data/fake_tracking.csv.gz")
slot_teams = pd.read_csv("test_data/fake_slot_teams.csv")
fasit      = pd.read_csv("test_data/fake_truth.csv", dtype={"label": str})
```

### Eksempel C: Test hele kjeden fra data til innlevering

Bytt ut `min_modell` med lagets egen funksjon. Den skal ta inn tracking og slot-tabell og gi én etikett (H/A/D/X) per rad.

```python
from test_data_generator import generate
import pandas as pd

m = generate(seed=42, mode="short")
etiketter = min_modell(m.tracking, m.slot_teams)        # din kode

svar = pd.DataFrame({"frame": m.tracking["frame_id"] - 1, "label": etiketter})
svar.to_csv("test_data/mitt_svar.csv", index=False)
```

Sjekk så formatet med arrangørens validator. Husk `--rows` i kort modus:

```bash
python evaluation/validate_submission.py test_data/mitt_svar.csv --rows 16800
```

Du kan også sammenligne med den falske fasiten for å se at logikken oppfører seg, for eksempel at modellen ikke svarer H hele kampen:

```python
treff = (svar["label"].to_numpy() == m.truth["label"].to_numpy()).mean()
print(f"Lik falsk fasit: {treff:.0%}")
```

> **Viktig:** Dette tallet er kun en røyktest. Bruk det **aldri** til å velge eller tune modell. Den falske kampen ligner ikke nok på ekte fotball.

### Eksempel D: Egen pytest-test for din modul

Legg en fil i `tests/`, for eksempel `tests/test_min_modul.py`:

```python
from test_data_generator import generate

def test_min_modell_gir_riktig_format():
    m = generate(seed=1, mode="short")
    etiketter = min_modell(m.tracking, m.slot_teams)
    assert len(etiketter) == len(m.tracking)
    assert set(etiketter) <= {"H", "A", "D", "X"}
```

Kjør med `python -m pytest tests -q`.

### Ikke gjør dette

- Ikke bruk falske data til trening, tuning eller modellvalg.
- Ikke legg falske filer i `data/`. De hører hjemme i `test_data/`, som er sperret i `.gitignore` (generatoren nekter også å skrive til en mappe som heter `data`).
- Ikke commit noe fra `test_data/`.

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
- Slot-navnene `s00`–`s45` er plassholdere. `starter/load_atd.py` setter ikke navnene selv. Den bruker kloppy, som henter dem fra metadata i de ekte dataene. Se regel 1 under «Bruk i din egen kode».
- kloppy gir trolig flere kolonner enn vi lager (for eksempel `ball_z` og `period_id`). Vurderes i v2 mot Metricas åpne Sample Game 3.
- Ingen identitetsbytter mellom spillere, ingen slots som hopper mellom lag.

## v2 (planlagt)

Rå filer i EPTS-format (XML-metadata + rå tekst) med Metricas åpne Sample Game 3 og `starter/load_atd.py` som formatmal, slik at testen går gjennom den ekte lasteren. Pluss Smart Tagging, homografi, identitetsbytter og dupliserte slots som hopper. Simuleringen (`possession.py`, `ball.py`, `players.py`) trenger ikke endres – bare en ny skriver ved siden av `writer.py`.
