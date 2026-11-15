# NEXT – Ball Possession Detection, More than a Hack 2027 (Phase 1)

> **DRAFT (15 Nov 2026, Morten).** Before submission, remove this box and resolve every tag:
> - `[OSKAR FYLLER INN: …]` – possession model (Oskar)
> - `[MARTIN FYLLER INN: …]` – dead-ball model, pipeline, reproduction (Martin)
> - `[SJEKK: …]` – something Morten was unsure about; verify and then delete the tag
>
> Target length when finished: 4–6 pages. Save as `NEXT.md` (or export to `NEXT.pdf`) with the team name **identical** to the registration (checklist A3).
>
> ⚠️ **No competition-match data in this document:** no plots, screenshots, tables or numbers computed from the competition match (checklist G4, open question U4). Figures and numbers from Sportec or synthetic data are fine.

| | |
|---|---|
| **Team** | NEXT [SJEKK: identical to the registration, character for character] |
| **Members** | Oskar Rändel (captain, possession model) · Martin (dead-ball model, pipeline, submission) · Morten (external validation, QA, documentation) |
| **Repository** | https://github.com/NextConsultingBarcelona/more-than-a-hack |
| **Commit / tag** | `[MARTIN FYLLER INN: commit hash]` / `phase1-submission` |
| **Submitted file** | `NEXT.csv` (SHA-256: `[MARTIN FYLLER INN: hash from checklist B6]`) |

---

## Summary

[OSKAR FYLLER INN: 4–6 sentences. What the system does (tracking → per-frame cues → HMM → Viterbi → H/A/D sequence), the one key idea (a switching penalty that prevents flickering, because spellF1 rewards whole possessions), and that all parameters were tuned on open Sportec matches only. No score on the competition match.]

---

## 1. Approach – what we built and why

### 1.1 Problem framing

Each frame of the match must be labelled `H` (home in possession), `A` (away in possession), `D` (dead ball) or `X` (no answer). The ranking metric, spellF1, does not count frames: it counts *possession spells* (maximal runs of `H` or `A` within a half) and matches them one-to-one against the ground truth by temporal overlap (IoU), averaging F1 over IoU thresholds 0.3–0.7. A single wrong frame inside a spell therefore splits it in two: `H H A H H` is three spells instead of one, and two of them will not be matched. This pushes us towards a **sequence model** that decides on whole spells, rather than a per-frame classifier.

### 1.2 Possession model (H/A): Hidden Markov Model + Viterbi

[OSKAR FYLLER INN:
- hidden states (`H`, `A`, `D`, and any sub-states, e.g. separate restart states)
- emissions: which cues, and how they are combined into a likelihood per state
- transitions: the H↔A switching penalty and why it is the key parameter
- Viterbi decoding: per half? how frames outside the assessed periods are labelled
- small state diagram (made from scratch or on synthetic data)
- how the H/A model and Martin's dead-ball model are combined (one HMM, or D first and then H/A?)]

### 1.3 Dead-ball model (D) and transitions

[MARTIN FYLLER INN:
- which cues indicate dead ball (ball speed, ball outside the pitch, players slowing down / stopping, Smart Tagging?)
- how the start and end frame of a dead-ball period are found (precise transitions)
- how the owner of a restart (kick-off, throw-in, corner, goal kick, free kick) is decided
- how the second-half kick-off is handled (assessed from the start of the half, but the ball is dead until kick-off)]

### 1.4 Cues ("hints") from tracking

[OSKAR FYLLER INN + MARTIN FYLLER INN: one line per cue actually used in the final pipeline: what it is and why it indicates possession or dead ball. Candidates from our plan:
- team share among the k players nearest the ball
- distance from the ball to the nearest player of each team
- ball speed and acceleration
- ball outside the pitch boundaries
- players slowing down / stopping (typical of dead ball)
- behaviour when the ball is missing (about 41 % of assessed frames, according to the organiser's DATA.md): what the model relies on instead]

### 1.5 Why this approach

The HMM is interpretable, has few free parameters (so it can be tuned on a handful of external matches), is deterministic (so the submitted file can be reproduced exactly), and needs no labelled data from the competition match. Alternatives considered:
- **Nearest player + majority vote in a window** (the organiser's baseline). Simple, but changing the window size alone changes spellF1 by an order of magnitude while macroF1 barely moves – the problem is flickering, which a sequence model addresses directly.
- **Temporal convolutional networks** (Borghesi et al., 2023). Strong per-frame results, but they need many labelled matches and are trained for per-frame accuracy, not whole spells. We have no labelled training data from the competition match and chose not to start with deep learning.

[OSKAR FYLLER INN: confirm or adjust; add anything else that was tried.]

---

## 2. Data sources – what we used, how, and what we dropped

### 2.1 Organiser-provided data

| Source | Used? | How it was used |
|---|---|---|
| Tracking data (ATD, EPTS format) | Yes | Main input for all cues. [MARTIN FYLLER INN: loader (e.g. the organiser's `starter/load_atd.py` via kloppy), and how NaN, slots that appear/disappear and duplicate slots are handled] |
| Smart Tagging | [MARTIN FYLLER INN: Yes / No] | [MARTIN FYLLER INN: e.g. extra cue for dead-ball phases. Note that it is automatic tagging, not ground truth, and that team tags are missing (`N/A`) on throw-ins] |
| Homography | [SJEKK: probably No] | [SJEKK: not needed if all cues come from pitch coordinates in the tracking] |
| Tactical video | [SJEKK: probably No] | [SJEKK: if not used as input, write "Not used as model input." Either way:] Our labels were **never compared with the video** (rule R8). Team members watched the video only to understand what counts as possession, which the organiser explicitly allows. |
| Broadcast video | [SJEKK: probably No] | |
| Public ground truth (first 5 minutes) | Pipeline check only | Used only to confirm that the pipeline runs end-to-end and produces a valid file. **Not** used to select models or parameters (see sections 5.7 and 9). |

### 2.2 Data we deliberately did not use

- **No event data** and no other human-annotated data about the competition match (team rule, stricter than the official rules).
- **No manual labelling** of the competition match for any purpose, not even for testing (team rule, stricter than the official rules, which allow a small amount of hand-labelled training data).
- **No full ground truth** and nothing derived from it.
- **No organiser baseline outputs** (`evaluation/results/baseline_*.csv`) were opened or used: they are labels for the competition match.

### 2.3 External and synthetic data

| Data | Role |
|---|---|
| **Sportec/IDSSE open tracking data** (Bundesliga 2022/23, CC-BY 4.0) | The *only* data used to tune or select parameters (section 5) |
| **Synthetic match from our own generator** (`test_data_generator/`) | Software testing only: correct format, 199 319 rows, no crashes, sensible H/A/D behaviour. Built only from the organiser's public documentation (DATA.md, CHALLENGE.md) and general football knowledge; nothing in it is calibrated against the competition match. **Never** used for training, tuning or model selection |

### 2.4 Tried and dropped

[OSKAR FYLLER INN + MARTIN FYLLER INN: short list of what was tried and why it was dropped. Every evaluation behind a decision must come from the external validation in section 5, not from the competition match.]

---

## 3. Possession rules we implemented

| Organiser's rule | Our implementation |
|---|---|
| When the ball is in play, one team always has possession (no neutral state) | [OSKAR FYLLER INN: e.g. no neutral state in the HMM; `X` used only where …] |
| **Loose balls** (50/50, ball rolling after a tackle): the last team in possession keeps it | [OSKAR FYLLER INN: e.g. high self-transition probability; switch only when cues for the other team persist for ≥ N frames] |
| Possession changes only when a player of the other team has **control** | [OSKAR FYLLER INN: how "control" is approximated from tracking (distance, ball speed relative to the player, duration)] |
| One touch without control, a deflected pass, or a won header without subsequent control is **not** a change | [OSKAR FYLLER INN] |
| **Restarts** (kick-off, throw-in, corner, goal kick, free kick) have a clear owner | [MARTIN FYLLER INN: how the owner of a restart is determined] |
| Dead ball is always `D` | [MARTIN FYLLER INN: cues and thresholds for dead ball, and for its start and end] |
| Two halves; the second half is assessed from frame 110 808 while kick-off is at 113 742 | [MARTIN FYLLER INN: how half boundaries are handled. The frame numbers are the official boundaries from the organiser's public CHALLENGE.md and must be read from there or from the data, never hand-coded per frame (rule R2)] [SJEKK: frames before kick-off in the second half must come out as `D` – checklist B7] |
| Use of `X` | [OSKAR FYLLER INN + MARTIN FYLLER INN: whether and where we output `X`. Note that on spellF1 an `X` is forward-filled with our previous answer in the same half: `X` inside a possession costs nothing on spellF1 (only on macroF1), but `X` across a team change is expensive] |

---

## 4. Limitations – where the model fails and what we would fix with more time

### 4.1 Known so far (external validation and data)

- **Domain gap.** Sportec data are corrected tracking with the ball almost always present. Our degradation (section 5.3) imitates the competition data only in broad terms: the ball-missing share and gap behaviour are taken from public documentation and general football knowledge, not measured on the competition match.
- **Simplified degradation (v1).** No duplicate slots, no noise on positions, and a 20-second half-time break instead of a real one (it only separates the halves). The model may therefore meet noisier data in the competition match than in validation.
- **Different possession definitions.** Sportec's per-frame possession field follows the DFL definition, which differs from the organiser's (e.g. for loose balls). Sportec scores therefore compare parameter values with each other; they are **not** an estimate of the competition score.
- **Ball-owner gaps in the Sportec ground truth.** When the ball is alive but Sportec has no ball-owning team for a frame, the label builder keeps the team that last had the ball in the same half (at the very start of a half: the next known owner), following the organiser's loose-ball rule (`fill_owner` in `external_validation/build_dataset.py`, tested in `tests/test_ground_truth.py`). On match J03WPY this never occurs (0 of 81,460 alive frames); the other six matches have not been checked.

### 4.2 Model limitations

[OSKAR FYLLER INN + MARTIN FYLLER INN: based on Sportec validation only, e.g.
- long periods with the ball missing (duels, near goal)
- short possessions (quick one-twos, interceptions) that the switching penalty may smooth away
- exact first/last frame of dead-ball periods at restarts]

### 4.3 With more time

[OSKAR FYLLER INN + MARTIN FYLLER INN: 3–5 concrete items, e.g. more open matches for tuning, learned emissions, explicit restart-type states, more realistic degradation (duplicate slots, position noise).]

---

## 5. External validation and tuning on Sportec matches

All training, tuning and calibration was done on **external open data only** (organiser's e-mail to all participants, ~5 Nov 2026; our rule R9). This section is the evidence that our parameters were chosen legitimately. The code is in `external_validation/` and can be re-run by anyone; it never reads competition data.

### 5.1 Dataset

Sportec/IDSSE open tracking data, Bundesliga 2022/23 (Bassek et al., 2025), licence CC-BY 4.0. 25 fps, the same frame rate as the competition match. Seven full matches, loaded with `kloppy` 3.19.1 (`sportec.load_open_tracking_data(match_id=..., only_alive=False)`). The match IDs are listed in `external_validation/config.py`.

| Match ID | Role |
|---|---|
| J03WPY | [OSKAR FYLLER INN: tuning / held-out test] |
| J03WMX | [OSKAR FYLLER INN] |
| J03WN1 | [OSKAR FYLLER INN] |
| J03WOH | [OSKAR FYLLER INN] |
| J03WOY | [OSKAR FYLLER INN] |
| J03WQQ | [OSKAR FYLLER INN] |
| J03WR9 | [OSKAR FYLLER INN] |

### 5.2 Ground truth on Sportec

Labels are derived mechanically from two per-frame fields in the Sportec tracking, in the same spirit as the organiser's ground truth (which is derived from corrected tracking, not from video or events):

- ball state *dead* → `D`
- ball state *alive* → the ball-owning team: home team → `H`, away team → `A`. A frame with no ball-owning team keeps the previous owner in the same half (the next known owner at the very start of a half), following the organiser's loose-ball rule

The two halves are placed on one timeline with `X` frames before the match, between the halves and after the match, so that the file has the same structure as the competition file and the organiser's scorer treats each half separately. All tracking values are NaN on these `X` frames. Differences from the organiser's definition are described in section 4.1.

### 5.3 Degradation to resemble the competition data

The Sportec data are corrected and the ball is almost always present. `external_validation/build_dataset.py` makes them resemble the competition tracking. All settings are in `external_validation/config.py`; every random step uses a fixed seed (42, combined with a checksum of the match ID), so the degraded data are identical on every machine.

| Step | What is done | Setting | Source |
|---|---|---|---|
| Ball removed | The ball is removed in blocks of 0.5–5 s until 41 % of in-play and dead-ball frames have no ball | `BALL_MISSING_TARGET = 0.41`, `BALL_GAP_SECONDS = (0.5, 5.0)` | 41 %: organiser's public DATA.md. Block length: football knowledge |
| More gaps in duels | A gap is 3× more likely to start in a duel (players from both teams within 2.5 m of the ball) | `DUEL_WEIGHT = 3.0`, `DUEL_RADIUS_M = 2.5` | Public documentation + football knowledge |
| More gaps near goal | A gap is 2× more likely to start when the ball is within 20 m of a goal | `NEAR_GOAL_WEIGHT = 2.0`, `NEAR_GOAL_M = 20.0` | Public documentation + football knowledge |
| Player identity removed | Players are placed in 46 anonymous track slots; each slot has a fixed team (as in the organiser's metadata) | `N_SLOTS = 46` | Organiser's DATA.md |
| Slots disappear | On average one gap per player every second minute, 1–8 s long | `PLAYER_GAPS_PER_MINUTE = 0.5`, `PLAYER_GAP_SECONDS = (1.0, 8.0)` | Football / tracking knowledge |
| Slots swap identity | After a gap, a player gets a new slot with probability 0.3 | `SLOT_SWITCH_PROB = 0.3` | Football / tracking knowledge |
| Speed per slot | Speed in m/s from positions, smoothed over 5 frames | `SPEED_SMOOTH_FRAMES = 5` | Matches the competition format (position + speed per slot) |

None of these values were measured on, or adjusted to, the competition data.

### 5.4 Metric and protocol

- **Metric:** the organiser's own scorer, `evaluation/possession_metrics.py`, imported directly (not copied), so Sportec scores are computed exactly as the competition will be. We report spellF1 (`spell_f1_mean`, primary) and macroF1 (tie-break).
- **Tools:** `external_validation/sweep.py` runs a model over a grid of parameter values on every built Sportec match and produces one table with, per parameter combination: mean spellF1, its spread (standard deviation) across matches, minimum spellF1, mean macroF1, and spellF1 per match.
- **Selection rule:** choose values that are good on *most* matches (high mean, low spread, acceptable minimum), not the value that wins on one match.
- **Baseline:** `external_validation/example_baseline.py` implements the organiser's baseline idea (nearest slot to the ball + majority vote in a window) on the same degraded Sportec matches, for comparison.
- **Protocol:** [OSKAR FYLLER INN: e.g. tune on 4 matches, report on 3 held-out matches, or leave-one-match-out.]

### 5.5 Parameters and how each was set

Every free parameter in the final pipeline must appear here with its source. Allowed sources: tuned on Sportec, taken from literature, or set from football knowledge (with justification). "Chosen from the score on the public 5 minutes" or "looked right on video" are **not** allowed sources.

| Parameter | What it controls | Final value | Source | Search range / method |
|---|---|---|---|---|
| H↔A switching penalty | How strong and persistent the evidence must be before possession changes team | [OSKAR FYLLER INN] | Tuned on Sportec | [OSKAR FYLLER INN] |
| Live→dead / dead→live transition probabilities | How easily the model enters and leaves `D` | [OSKAR/MARTIN FYLLER INN] | Tuned on Sportec | [OSKAR/MARTIN FYLLER INN] |
| Dead-ball ball-speed threshold | Ball speed below which a stopped ball suggests dead ball | [MARTIN FYLLER INN] | [MARTIN FYLLER INN] | [MARTIN FYLLER INN] |
| Out-of-bounds margin | Distance outside the lines before "ball out" is assumed (tracking noise) | [MARTIN FYLLER INN] | [MARTIN FYLLER INN] | [MARTIN FYLLER INN] |
| Number of nearest players (k) | Size of the neighbourhood for the team-share cue | [OSKAR FYLLER INN] | Literature (Borghesi et al. 2023) / tuned | [OSKAR FYLLER INN] |
| Emission weights / temperature | Relative weight of each cue in the state likelihood | [OSKAR FYLLER INN] | Tuned on Sportec | [OSKAR FYLLER INN] |
| Handling of missing ball | Weight of player-based cues when the ball is NaN | [OSKAR FYLLER INN] | | |
| `X` policy (if used) | When the model outputs `X` instead of a label | [OSKAR FYLLER INN] | | |
| [OSKAR/MARTIN FYLLER INN: any other parameter] | | | | |

### 5.6 Results on Sportec

[OSKAR FYLLER INN: table with spellF1 and macroF1 per held-out match and on average, for our model and the baseline (copy from the `sweep.py` output in `ext_data/results/`). Sportec results only. One or two sentences of interpretation, remembering that these numbers compare parameters and are not an estimate of the competition score.]

### 5.7 Freezing parameters

[OSKAR FYLLER INN + MARTIN FYLLER INN: confirm that the final parameters were committed and tagged (`<tag>`, `<date>`) **before** the pipeline was run on the public 5 minutes, and that nothing was changed afterwards based on that run.]

---

## 6. Reproduction – how to run it

> Must be **identical** to the README (checklist G6).

**Requirements:** Python [MARTIN FYLLER INN: x.y.z], Windows or macOS [SJEKK: tested on both – checklist C5]. Dependencies pinned in [MARTIN FYLLER INN: lock file]. [SJEKK: `requirements.txt` currently lists `numpy` and `pandas` without versions, and `requirements-dev.txt` uses `>=`. Every package, including indirect ones, must be pinned with `==` – checklist C1.]

```bash
git clone https://github.com/NextConsultingBarcelona/more-than-a-hack.git
cd more-than-a-hack
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS: source .venv/bin/activate
pip install -r requirements.txt   # [MARTIN FYLLER INN: or the lock-file command]
```

**Data:** The match data are **not** included in the repository. Place the organiser's data in a folder of your choice, outside the repository, and set:

```bash
# macOS:   export MTAH_DATA_DIR=/path/to/data
# Windows: set MTAH_DATA_DIR=C:\path\to\data
```

**Run (one command):**

```bash
python [MARTIN FYLLER INN: run script].py   # writes NEXT.csv
```

[MARTIN FYLLER INN: expected runtime; that the output is deterministic (fixed seeds, no tuning at run time); SHA-256 of the expected output.]

**Reproducing the external validation (optional, open data only):**

```bash
python -m pip install -r external_validation/requirements-validation.txt
python external_validation/build_dataset.py --selftest        # ~1 s, synthetic data, checks the setup
python external_validation/build_dataset.py --matches all     # downloads the 7 Sportec matches via kloppy
python external_validation/sweep.py <grid file>.json          # [OSKAR FYLLER INN: the grid file used for the final parameters]
```

The Sportec files are downloaded and cached by kloppy outside the repository; generated files are written to `ext_data/`, which is git-ignored. Loading one match needs a few GB of RAM.

---

## 7. External data, models and tools

Required by the rules (R7): everything external that we used, with licence and credit. Nothing private.

**Datasets**

| Dataset | Use | Licence | Credit |
|---|---|---|---|
| Sportec/IDSSE open tracking data, Bundesliga 2022/23 (match IDs J03WPY, J03WMX, J03WN1, J03WOH, J03WOY, J03WQQ, J03WR9) | Tuning and validation (section 5) | CC-BY 4.0 | Bassek, M., Rein, R., Weber, H. et al. (2025). *Scientific Data* 12, 195. doi:10.1038/s41597-025-04505-y |
| Metrica Sports Sample Data (Sample Game 3) | [SJEKK: delete this row if it was not used. Our generator v1 is built from the organiser's documentation only] | No formal licence ("use responsibly, cite the source") | Metrica Sports, github.com/metrica-sports/sample-data |

**Pre-trained models:** None. [OSKAR/MARTIN: confirm]

**Software libraries**

| Library | Version | Licence | Use |
|---|---|---|---|
| kloppy | 3.19.1 | [SJEKK: BSD-3-Clause?] | Loading Sportec data (and EPTS data, if used) |
| numpy | [MARTIN FYLLER INN] | BSD-3-Clause [SJEKK] | |
| pandas | [MARTIN FYLLER INN] | BSD-3-Clause [SJEKK] | |
| pyarrow | [MARTIN FYLLER INN] | Apache-2.0 [SJEKK] | Parquet files in the external validation |
| [OSKAR/MARTIN FYLLER INN: e.g. scipy, hmmlearn] | | | |
| Organiser's starter and evaluation code (`starter/`, `evaluation/`) | – | MIT | Data loaders; scoring in the external validation |

**Development tools:** Claude Code (Anthropic) was used as a coding assistant on synthetic and open (Sportec) data only. No competition data were shared with it. The repository contains technical guards for this: deny rules in `.claude/settings.json`, a pre-tool hook (`.claude/hooks/guard_paths.py`) that blocks the assistant from reading the data folders, and a pre-commit check (`tools/check_no_data.py`) that stops data files from being committed. [SJEKK: keep or shorten after the organiser's answer on open question U5.]

---

## 8. References

- Bassek, M., Rein, R., Weber, H. et al. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195. doi:10.1038/s41597-025-04505-y
- Borghesi, M., et al. (2023). Using Temporal Convolutional Networks to Estimate Ball Possession in Soccer Games. *Expert Systems with Applications*. [SJEKK: volume/pages, or delete if not cited]
- More than a Hack 2027, official repository: https://github.com/metrica-sports/more-than-a-hack-2027
- kloppy: https://github.com/PySport/kloppy

---

## 9. Compliance statement

> Keep only statements that **all three members** have confirmed (checklist E1–E7). [SJEKK: Oskar and Martin confirm each line before submission.]

- **Labels come from the algorithm (R2).** Labels are produced entirely by the code in the repository. No manual labelling, no manual correction of output and no frame-specific rules were used.
- **External tuning only (R9).** All training, tuning and calibration were done on external open data (Sportec/IDSSE) only. No parameter was selected using the public ground truth or the video.
- **Public ground truth (R3).** The public 5-minute ground truth was used only to verify that the pipeline runs end-to-end. No full ground truth, and nothing derived from it, was used.
- **No video comparison (R8).** Our labels were never compared with the video, neither manually nor by "checking whether it looks right".
- **No human-made data about this match.** No event data, hand-labelled data or other human-annotated data about this match were used.
- **No cloud AI (R1).** No competition data, excerpts or outputs derived from them were shared with cloud-based AI services.
- **No data in the repository (R5).** No competition data are included in the repository or its history.
- **Reproducible (R4).** One command reproduces the submitted file exactly (section 6).
- **Everything external is listed (R7).** All external data, models and libraries are listed in section 7.
