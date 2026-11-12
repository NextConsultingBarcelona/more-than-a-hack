# NEXT – Ball Possession Detection, More than a Hack 2027 (Phase 1)

> ✏️ **How to use this template:** Every block that starts with ✏️ is guidance. Replace it with your text and **delete the guidance before submitting**. Save the final file as `NEXT.md` or `NEXT.pdf`, with the team name **identical** to the registration (checklist A3). Target length: **4–6 pages**. Sections 1–6 map to the six requirements in the organiser's brief; sections 5 and 9 are our additions because of rules R8/R9.
>
> ⚠️ **No match data in this document:** no plots, screenshots, tables or numbers computed from the competition match (checklist G4, open question U4). Figures from Sportec data or synthetic data are fine.

| | |
|---|---|
| **Team** | NEXT |
| **Members** | Oskar Rändel (captain, possession model) · Martin (dead-ball model, pipeline, submission) · Morten (external validation, QA, documentation) |
| **Repository** | https://github.com/NextConsultingBarcelona/more-than-a-hack |
| **Commit / tag** | `<commit hash>` / `phase1-submission` |
| **Submitted file** | `NEXT.csv` (SHA-256: `<hash>`) |

---

## Summary

> ✏️ 4–6 sentences for a busy jury member. What the system does (tracking → per-frame cues → HMM → Viterbi → H/A/D sequence), the one key idea (a switching penalty that prevents flickering, because spellF1 rewards whole possessions), and how it was validated (external validation on open Sportec matches only). Do **not** report a score on the competition match.

---

## 1. Approach – what we built and why

> ✏️ Explain the method so that someone who has not seen our code can follow it. Suggested structure:
>
> **1.1 Problem framing.** Per-frame classification into H/A/D, but evaluated mainly by spellF1 (whole possession spells). Explain why this pushes us towards a *sequence* model rather than per-frame classification: a single wrong frame inside a spell splits it in two (e.g. `H H A H H` = three spells instead of one).
>
> **1.2 Model: Hidden Markov Model + Viterbi decoding.**
> - Hidden states: `H`, `A`, `D` (state if you used sub-states, e.g. separate "restart" states).
> - **Emissions** = cues computed from tracking per frame (see 1.3), and how they are combined into a likelihood per state (e.g. a logistic score, or product of independent terms).
> - **Transitions** = probability of staying in a state vs. switching. The **switching penalty** between H and A is the key parameter: the model only changes team when the cues are strong for long enough.
> - **Viterbi** finds the single most likely H/A/D sequence for each half. State whether halves are decoded separately and how frames outside the assessed periods are labelled.
> - Include a small diagram of the three states and allowed transitions (made from scratch or on synthetic data).
>
> **1.3 Cues ("hints") from tracking.** One short line per cue: what it is and why it indicates possession or dead ball. Examples from our plan:
> - team share among the 5 players nearest the ball
> - distance from ball to nearest player of each team
> - ball speed and acceleration
> - ball outside the pitch boundaries
> - players slowing down / stopping (typical of dead ball)
> - behaviour when the ball is missing (~41 % of assessed frames): what the model relies on instead
>
> **1.4 Why this approach.** 2–4 sentences: interpretable, few parameters (can be tuned on little external data), deterministic (exact reproducibility), and no labelled data from the competition match is needed. Mention alternatives considered (e.g. nearest-player baseline with majority voting, temporal convolutional networks as in Borghesi et al. 2023) and why we did not choose them.

---

## 2. Data sources – what we used, how, and what we dropped

> ✏️ One row per data source the organiser provided. Be explicit about **how** each one was used, and whether it was used at all.

| Source | Used? | How it was used |
|---|---|---|
| Tracking data (ATD, EPTS format) | ✏️ Yes | ✏️ Main input for all cues. Loaded with `<loader>`. Handling of NaN, slot changes and duplicate slots |
| Smart Tagging | ✏️ Yes / No | ✏️ e.g. as an extra cue for dead-ball phases; note that team tags are missing (`N/A`) on throw-ins |
| Homography | ✏️ Yes / No | ✏️ |
| Tactical video | ✏️ Yes / No | ✏️ If used as input: how. State explicitly that labels were **never compared with the video** (R8) |
| Broadcast video | ✏️ Yes / No | ✏️ |
| Public ground truth (first 5 minutes) | Pipeline check only | ✏️ Used only to confirm that the pipeline runs end-to-end and produces a valid file. **Not** used to select models or parameters (see section 9) |

> ✏️ **2.1 Tried and dropped.** Short list: what we tried, why it was dropped (e.g. "Smart Tagging intervals as hard dead-ball boundaries – too coarse at restart boundaries on Sportec validation"). Any evaluation behind a decision must come from the external validation in section 5, not from the competition match.

---

## 3. Possession rules we implemented

> ✏️ For each rule from the organiser's definition, state **how the code implements it**. One short paragraph or table row each. The jury will compare this with the rules, so be concrete (thresholds, durations, which cue drives it).

| Organiser's rule | Our implementation |
|---|---|
| When the ball is in play, one team always has possession (no neutral state) | ✏️ e.g. no neutral state in the HMM; `X` used only where … |
| **Loose balls** (50/50, ball rolling after a tackle): last team in possession keeps it | ✏️ e.g. high self-transition probability; switch only when cues for the other team persist for ≥ N frames |
| Possession changes only when a player of the other team has **control** | ✏️ How "control" is approximated from tracking (distance, ball speed relative to player, duration) |
| One touch without control, a deflected pass, or a won header without subsequent control is **not** a change | ✏️ |
| **Restarts** (kick-off, throw-in, corner, goal kick, free kick) have a clear owner | ✏️ How the owner of a restart is determined |
| Dead ball is always `D` | ✏️ Cues and thresholds for detecting dead ball, and for its start and end |
| Two halves; second half assessed from frame 110 808, kick-off at 113 742 | ✏️ How half boundaries are handled (taken from the organiser's documentation, not hand-coded per frame) |
| Use of `X` | ✏️ Whether and where we output `X`, and why (note that on spellF1 an `X` is filled with our previous answer in the same half) |

---

## 4. Limitations – where the model fails and what we would fix with more time

> ✏️ Honest and specific. Base the failure analysis on **Sportec validation** (section 5), not on the competition match. Suggested points:
> - Situations with the ball missing for long periods (duels, near goal)
> - Short possessions (quick one-twos, interceptions) that the switching penalty may smooth away
> - Restart boundaries: exact first/last frame of dead ball
> - Domain gap: Sportec data are corrected tracking; even after degradation they differ from the competition data
> - Possible mismatch between Sportec's possession definition and the organiser's (see 5.2)
>
> **With more time:** 3–5 concrete items, e.g. more open matches for tuning, learned emissions, explicit restart-type states.

---

## 5. External validation and tuning on Sportec matches

> ✏️ This section documents that **all training, tuning and calibration was done on external data only** (organiser's e-mail, ~5 Nov; our rule R9). It is the evidence that our parameters were chosen legitimately, so be precise.

### 5.1 Dataset

> ✏️ Sportec/IDSSE open tracking data, Bundesliga 2022/23 (Bassek et al., 2025), CC-BY 4.0. 25 fps, same as the competition match. List the match IDs used and their role:

| Match ID | Role (tuning / held-out test) |
|---|---|
| J03WPY | ✏️ |
| J03WMX | ✏️ |
| J03WN1 | ✏️ |
| J03WOH | ✏️ |
| J03WOY | ✏️ |
| J03WPF | ✏️ |
| J03WQF | ✏️ |

> ✏️ Loaded with `kloppy` (`sportec.load_open_tracking_data(match_id=...)`, version `<x.y.z>`).

### 5.2 Ground truth on Sportec

> ✏️ How H/A/D labels were derived mechanically from the Sportec per-frame fields (ball alive/dead and ball-owning team), mapping home team → `H`, away → `A`, ball dead → `D`.
> **Note any differences** between Sportec's possession definition and the organiser's rules (e.g. how loose balls are handled) and what we did about them. If we did not check this, say so in section 4.

### 5.3 Degradation to match the competition data

> ✏️ The Sportec data are corrected, with the ball almost always present. Describe exactly what was done to make them resemble the competition tracking, with parameters and **random seed**:
> - ball removed on ~41 % of frames (state how: random frames, random segments, more often in duels/near goal)
> - player identity removed (anonymous track slots)
> - slots disappearing, reappearing and swapping identity
> - noise added to positions (if any)
>
> The 41 % figure comes from the organiser's public documentation (DATA.md), not from our own analysis of the match data.

### 5.4 Metric and protocol

> ✏️ Metric: the organiser's spellF1 (primary) and macroF1 (tie-break), using their public evaluation code `<file name>`. Protocol, e.g. tune on 4 matches, report on 3 held-out matches; or leave-one-match-out. Include a comparison with a simple baseline (nearest player + majority vote) on the same Sportec matches.

### 5.5 Parameters and how each was set

> ✏️ **Every** free parameter in the final pipeline must appear here, with its source. Allowed sources: tuned on Sportec, taken from literature, or set from football knowledge (with justification). "Chosen from the score on the public 5 minutes" or "looked right on video" are **not** allowed sources.

| Parameter | What it controls | Final value | Source | Search range / method |
|---|---|---|---|---|
| H↔A switching penalty (transition probability) | How strong and persistent the evidence must be before possession changes team | ✏️ | Tuned on Sportec | ✏️ e.g. grid search, range … |
| Live→dead / dead→live transition probabilities | How easily the model enters and leaves `D` | ✏️ | Tuned on Sportec | ✏️ |
| Dead-ball ball-speed threshold | Ball speed below which a stopped ball suggests dead ball | ✏️ | ✏️ | ✏️ |
| Out-of-bounds margin | Distance outside the lines before "ball out" is assumed (tracking noise) | ✏️ | ✏️ | ✏️ |
| Number of nearest players (k) | Size of neighbourhood for the team-share cue | ✏️ 5 | Literature (Borghesi et al. 2023) / tuned | ✏️ |
| Emission weights / temperature | Relative weight of each cue in the state likelihood | ✏️ | Tuned on Sportec | ✏️ |
| Handling of missing ball | Weight of player-based cues when the ball is NaN | ✏️ | ✏️ | ✏️ |
| `X` policy (if used) | When the model outputs `X` instead of a label | ✏️ | ✏️ | ✏️ |
| ✏️ *(add any other)* | | | | |

### 5.6 Results on Sportec

> ✏️ Table with spellF1 and macroF1 per held-out match and on average, for our model and the baseline. These are Sportec results only and can be shown freely. One or two sentences of interpretation.

### 5.7 Freezing parameters

> ✏️ State that the final parameters were committed and tagged (`<tag>`, `<date>`) **before** the pipeline was run on the public 5 minutes, and that nothing was changed afterwards based on that run.

---

## 6. Reproduction – how to run it

> ✏️ Must be **identical** to the README (checklist G6).

**Requirements:** Python `<x.y.z>`, Windows or macOS (tested on both). Dependencies are pinned in `<lock file>`.

```bash
git clone https://github.com/NextConsultingBarcelona/more-than-a-hack.git
cd more-than-a-hack
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS: source .venv/bin/activate
pip install -r requirements.txt   # ✏️ or the lock-file command
```

**Data:** The match data are **not** included in the repository. Place the organiser's data in a folder of your choice and set:

```bash
# macOS:   export MTAH_DATA_DIR=/path/to/data
# Windows: set MTAH_DATA_DIR=C:\path\to\data
```

**Run (one command):**

```bash
python <run script>.py   # ✏️ writes NEXT.csv
```

> ✏️ State: expected runtime, that the output is deterministic (fixed seeds, no tuning at run time), and the SHA-256 of the expected output. Optional: separate command to reproduce the Sportec tuning (section 5), which downloads the open data via kloppy.

---

## 7. External data, models and tools

> ✏️ Required by the rules (R7): **everything** external that we used, with licence and credit. Nothing private.

**Datasets**

| Dataset | Use | Licence | Credit |
|---|---|---|---|
| Sportec/IDSSE open tracking data, Bundesliga 2022/23 | Tuning and validation (section 5) | CC-BY 4.0 | Bassek, M., et al. (2025). *Scientific Data*. doi:10.1038/s41597-025-04505-y |
| Metrica Sports Sample Data (Sample Game 3) | ✏️ Format reference / testing, or delete row | No formal licence ("use responsibly, cite the source") | Metrica Sports, github.com/metrica-sports/sample-data |

**Pre-trained models:** ✏️ None. *(If any are used: name, source, licence.)*

**Software libraries**

| Library | Version | Licence | Use |
|---|---|---|---|
| kloppy | ✏️ | ✏️ check | Loading Sportec and EPTS data |
| numpy | ✏️ | ✏️ | |
| pandas | ✏️ | ✏️ | |
| ✏️ … | | | |
| Organiser's starter code (`starter/`) | – | MIT | Data loaders |

**Development tools:** ✏️ e.g. "Claude Code was used as a coding assistant on synthetic and open data only; no competition data were shared with it." *(Pending open question U5 – keep or remove after the organiser's answer.)*

---

## 8. References

> ✏️ Keep only those actually cited.

- Bassek, M., et al. (2025). [Title]. *Scientific Data*. doi:10.1038/s41597-025-04505-y
- Borghesi, M., et al. (2023). Using Temporal Convolutional Networks to Estimate Ball Possession in Soccer Games. *Expert Systems with Applications*.
- More than a Hack 2027, official repository: https://github.com/metrica-sports/more-than-a-hack-2027
- ✏️ kloppy documentation, other sources

---

## 9. Compliance statement

> ✏️ Short and factual. Only keep statements that **all three members** have confirmed (checklist E1–E7).

- Labels are produced entirely by the algorithm in the repository. No manual labelling, manual correction of output or frame-specific rules were used.
- All training, tuning and calibration were done on external open data (Sportec/IDSSE) only. No parameter was selected using the public ground truth or the video.
- The public 5-minute ground truth was used only to verify that the pipeline runs end-to-end.
- Our labels were never compared with the video.
- No full ground truth, event data or other human-annotated data about this match were used.
- No competition data were shared with cloud-based AI services.
- No competition data are included in the repository or its history.
