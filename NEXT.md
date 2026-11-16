# NEXT – Ball Possession Detection, More than a Hack 2027 (Phase 1)

| | |
|---|---|
| **Team** | NEXT |
| **Members** | Oskar Rändel (captain, possession model) · Martin (dead-ball model, pipeline, submission) · Morten (external validation, QA, documentation) |
| **Repository** | https://github.com/NextConsultingBarcelona/more-than-a-hack |
| **Commit / tag** | the commit tagged `phase1-submission` |
| **Submitted file** | `NEXT.csv` (SHA-256: `f12d15ce52ebeba71c6fa5c245e19bf061c4360a4b5e3ddebea4214a1d63ae6f`) |

---

## Summary

Our system turns the automatic tracking into one label per frame in four steps: per-frame cues from players and ball, evidence for "home vs away" and for "dead ball", a three-state hidden Markov model (`H`, `A`, `D`) decoded with Viterbi, and a clean-up that snaps each boundary to the event it stands for. The key idea is that a frame with nothing to say (ball in flight, ball missing) carries no evidence, so the model keeps the last team, and a switching cost means a single touch cannot change possession. This is the organiser's loose-ball rule expressed as a model, and it is what spellF1 rewards: whole possessions, not frames. Because the ball is missing on about 41 % of assessed frames, the model also learns, without any labels, how team shape and movement relate to possession on the frames where the ball *is* seen, and uses that where it is not. The per-frame evidence that goes into the decoder comes from two small gradient-boosted models trained on five open Sportec matches, with our hand-built cues as their inputs; decoder settings were chosen on the same five matches and checked on two held-out ones (spellF1 0.58 against 0.24 for the nearest-player baseline). Nothing was trained or tuned on the competition match and we report no score on it.

---

## 1. Approach – what we built and why

### 1.1 Problem framing

Each frame of the match must be labelled `H` (home in possession), `A` (away in possession), `D` (dead ball) or `X` (no answer). The ranking metric, spellF1, does not count frames: it counts *possession spells* (maximal runs of `H` or `A` within a half) and matches them one-to-one against the ground truth by temporal overlap (IoU), averaging F1 over IoU thresholds 0.3–0.7. A single wrong frame inside a spell therefore splits it in two: `H H A H H` is three spells instead of one, and two of them will not be matched. This pushes us towards a **sequence model** that decides on whole spells, rather than a per-frame classifier.

### 1.2 Possession model (H/A): Hidden Markov Model + Viterbi

**States.** Three hidden states, `H`, `A` and `D`, in **one** model. There are no sub-states and no "contested" state, because the ground truth has none.

```
            c_switch = 2.0
        H <----------------> A
         \                  /
  c_die   \                /   c_die = 10       (live -> dead)
  = 10     v              v
            +---- D -----+     c_restart = 10   (dead -> live, to either team)
```

Costs are negative log-probabilities per frame; staying in a state is free.

**Emissions.** Two per-frame log-likelihood ratios are computed (`src/possession.py`, `src/deadball.py`):

- `team_llr` (home vs away) = ball proximity + touches + ball-free hint + Smart Tagging prior (section 1.4). State `H` scores `+team_llr/2`, state `A` scores `-team_llr/2`.
- `dead_llr` (dead vs alive) from the dead-ball cues (section 1.3). State `D` scores `dead_llr`.

A frame with no evidence scores 0 for all three states, so the decoder simply stays where it is.

**Learned evidence (final model).** The two ratios above are not fed to the decoder directly. They are inputs to two gradient-boosted tree models trained on open Sportec matches (section 5), one for dead-vs-alive and one for home-vs-away given a live ball (`src/learned.py`):

- *Inputs* are the hand-built evidence of sections 1.3 and 1.4 (without Smart Tagging) and the features behind it, each also averaged over windows from a quarter of a second to four seconds, and over the second or two before and after the frame: ball proximity and touches, the ball-free hint, ball speed, position and gaps, team centroids, their movement along the pitch and team spread, player tempo.
- *Home/away symmetry.* Team features are oriented by each team's attacking direction, taken per half from which side its centroid stands on. The model is trained on every match in both orientations and evaluated as the average of a match and its mirror image, so it cannot favour a side.
- *Output.* The models' log-odds, scaled by 0.4 and capped at ±6 per frame, replace `team_llr` and `dead_llr` in the decoder. The Smart Tagging terms (team prior, set pieces, the stretch before kick-off) are added on top unchanged, because Sportec has no Smart Tagging to learn them from.
- *Why trees, and why only as evidence.* Borghesi et al. (2023) show that learned per-frame evidence beats rule-based cues, and Kim et al. (2026) that structured decoding matters more than the per-frame model. We kept our decoder and clean-up and replaced only the weighting of the cues. The trees are stored as plain arrays in `src/learned_model.json` and evaluated with numpy, so the pipeline needs no machine-learning library and is deterministic.

**Transitions.** The H↔A switching cost is the parameter that defines "controlled" possession: an opponent spell inside a possession must collect more evidence than two switches cost, which a deflection or a single touch cannot. After switching the ball-free hint on, it was the most influential setting in the external validation (section 5.6).

**Decoding.** Viterbi is run separately on each assessed half, so nothing leaks across half-time. Frames outside the two halves are written as `X`; inside them the model never abstains (section 3).

**After decoding** (same file, in this order):

1. *Minimum durations.* A stoppage shorter than 2 s is treated as a tracking blip; a team spell shorter than 1 s between two spells of the other team is a touch, not a possession.
2. *Boundary refinement.* A team→team boundary moves to the first frame the new team controls the ball after the old team's last control, so a pass in flight still belongs to the passer. A live→dead boundary moves to the first frame the ball is seen outside the pitch. A dead→live boundary moves back over a fast in-play ball or forward over a ball still lying on its spot.
3. *Restart owner.* If the other team was clearly on the ball as play restarted, it owns the frames from the restart until the following team first controls the ball (section 3).

### 1.3 Dead-ball model (D) and transitions

`D` is a state of the same HMM, not a separate model run first. Its evidence `dead_llr` is a sum of:

- **ball outside the pitch** by more than 0.75 m (strongest cue);
- **kick-off formation**: both teams entirely in their own halves for about two seconds (restart after a goal);
- **ball lying still** (detected, and staying within 0.35 m for about a second) counts for dead; a **fast ball inside the pitch** counts for alive;
- **ball missing for more than 3 s** counts weakly for dead;
- **Smart Tagging**: the 3 s before a tagged set piece, and the span between a tagged goal and the kick-off tag that follows it;
- **player tempo**: a small logistic model on median player speed and the share of slow players. It is fitted at run time, without labels, on the frames the cues above are sure about (section 2.5). Its contribution is capped so that slow players alone need several seconds to create a stoppage.

**Start and end frames.** Viterbi places the boundaries; refinement then snaps live→dead to the ball leaving the pitch and dead→live to the ball leaving its spot (section 1.2).

**Restart owner.** The team whose player is on the still ball in the last 2 s of the stoppage, or the team named by a set-piece tag when that tag type has been measured to name the taker (section 1.4).

**Second-half kick-off.** Assessed frames before the kick-off get an overwhelming dead-ball term (`w_prekick`), so they come out as `D`. The kick-off frame is not hard-coded: the Smart Tagging "Kick Off (Start)" tag gives a window, and inside it the kick is pinned by the formation breaking and, where the ball is seen, by the ball leaving the centre spot.

### 1.4 Cues ("hints") from tracking

Cues for **which team** (all in `src/features.py` and `src/possession.py`):

| Cue | What it is | Why |
|---|---|---|
| Ball proximity | Difference between the distance from the ball to the nearest home and nearest away player, weighted by how close the nearer one is (possession zone of 2 m) and by the ball being slower than about 10 m/s | The player on the ball has it; a fast ball passing a player is not under his control (Vidal-Codina et al. 2022; Borghesi et al. 2023 on earlier rule-based systems) |
| Touch | A change of ball velocity of at least 3 m/s next to a player | A kick or a reception is the moment possession is exercised (Link & Hoernig 2017) |
| Ball-free hint | Logistic model on team centroids, their velocities and team spread, fitted per half at run time on frames where ball proximity is confident (section 2.5) | The ball is missing on ~41 % of assessed frames; player trajectories alone carry possession information (Kim et al. 2026) |
| Smart Tagging phases | Weak prior from the `Team` tag of each phase | Free evidence, but automatic. Whether a code's tag names the team *with* or *without* the ball is measured against the ball evidence, not assumed; codes that agree with neither get weight 0 |

Cues for **dead ball** are listed in section 1.3.

**Ball handling.** Detections in short segments bounded by impossible jumps (> 45 m/s) are dropped; gaps of up to 1 s are linearly interpolated and count half; positions are smoothed over 5 frames before speeds are computed.

**When the ball is missing.** Ball cues are silent, the hint and the Smart Tagging prior speak, and otherwise the decoder holds the last team. We did **not** use a "team share among the k nearest players" cue from our original plan; the nearest player of each team was enough.

### 1.5 Why this approach

The design combines four published approaches: the possession zone and kick detection of Link & Hoernig (2017) and Vidal-Codina et al. (2022) as cues, per-frame evidence learned from labelled open matches as in Borghesi et al. (2023), and structured decoding of whole possessions as argued by Kim et al. (2026). The HMM is interpretable, is deterministic (so the submitted file can be reproduced exactly), and needs no labelled data from the competition match. It has many settings (section 5.5), but only a handful change the result, so they can be chosen on a few external matches. Alternatives considered:
- **Nearest player + majority vote in a window** (the organiser's baseline). Simple, but changing the window size alone changes spellF1 by an order of magnitude while macroF1 barely moves – the problem is flickering, which a sequence model addresses directly.
- **Temporal convolutional networks** (Borghesi et al., 2023). Strong per-frame results, but they need many labelled matches and are trained for per-frame accuracy, not whole spells. We have no labelled training data from the competition match and chose not to start with deep learning.

- **Rule-based event detection** (Link & Hoernig 2017; Vidal-Codina et al. 2022). We took their cues (possession zone, kick detection) but not their hard per-frame decisions, which assume a ball on every frame.
- **Neural sequence models with structured decoding** (Kim et al. 2026). Their result that constrained Viterbi decoding matters more than the backbone supports the HMM; their backbone needs labelled matches with reliable tracking, which we do not have for this data source.

---

## 2. Data sources – what we used, how, and what we dropped

### 2.1 Organiser-provided data

| Source | Used? | How it was used |
|---|---|---|
| Tracking data (ATD, EPTS format), tactical | Yes | Main input for all cues. Read by our own parser (`src/io.py`, numpy only, no kloppy), indexed by video frame (`frame_id − 1`). Missing values stay NaN and every feature is NaN-aware. Track slots are never treated as persons: all cues are per team (nearest player, centroid, spread), so slots that appear, vanish or swap do not matter. Duplicate slots are not removed |
| Smart Tagging | Yes | Four uses: (1) the kick-off tag anchors the search for each half's kick-off; (2) the seconds before a tagged set piece, and the span from a tagged goal to its kick-off, count for dead ball; (3) phase `Team` tags give a weak team prior with measured polarity (section 1.4); (4) a set-piece `Team` tag can name the restart owner. It is treated as automatic evidence, not ground truth; `N/A` team tags (all throw-ins) are ignored. Team names are matched to home/away by string similarity, since the tagging and the tracking metadata spell them differently |
| Homography | No | Not needed: all cues come from pitch coordinates in the tracking |
| Tactical video | No | Not used as model input. Our labels were **never compared with the video** (rule R8). Team members watched the video only to understand what counts as possession, which the organiser explicitly allows |
| Broadcast video and its tracking | No | Not used: it is on its own timeline and we did not attempt the alignment |
| Public ground truth (first 5 minutes) | Pipeline check only | Used only to confirm that the pipeline runs end-to-end and produces a valid file. **Not** used to select models or parameters (see sections 5.7 and 9). |

### 2.2 Data we deliberately did not use

- **No event data** and no other human-annotated data about the competition match (team rule, stricter than the official rules).
- **No manual labelling** of the competition match for any purpose, not even for testing (team rule, stricter than the official rules, which allow a small amount of hand-labelled training data).
- **No full ground truth** and nothing derived from it.
- **No organiser baseline outputs** (`evaluation/results/baseline_*.csv`) were opened or used: they are labels for the competition match.

### 2.3 External and synthetic data

| Data | Role |
|---|---|
| **Sportec/IDSSE open tracking data** (Bundesliga 2022/23, CC-BY 4.0) | The *only* data used to train models and to tune or select parameters (section 5). **Training labels:** the dataset's own per-frame ball state and ball-owning team, published by the DFL with the tracking (Bassek et al. 2025); we labelled nothing ourselves. Five matches, about 730,000 frames, every fifth frame used for training |
| **Synthetic match from our own generator** (`test_data_generator/`) | Software testing only: correct format, 199 319 rows, no crashes, sensible H/A/D behaviour. Built only from the organiser's public documentation (DATA.md, CHALLENGE.md) and general football knowledge; nothing in it is calibrated against the competition match. **Never** used for training, tuning or model selection |
| **Synthetic match in the real file formats** (`tools/make_fake_match.py`) | Software testing of `src/` only: a toy match written as ATD text, metadata XML and Smart Tagging XML, with its own ground truth, so the whole pipeline is tested end to end. Built from the public documentation only. Not used for tuning; where it disagreed with the Sportec validation (section 2.4), Sportec decided |

### 2.4 Tried and dropped

Every decision below rests on the Sportec validation in section 5.

- **Hand-weighted evidence as decoder input.** This was our model until 15 Nov: spellF1 0.501 on the tuning matches and 0.514 on the held-out ones. Replacing the hand-set weights by the learned evidence of section 1.2 gave 0.546 (leave-one-match-out) and 0.582, with macroF1 up from 0.87 to 0.90, so the learned evidence is the final model. `learned=False` in `src/config.py` gives the earlier model back.
- **Hand-set parameters without the ball-free hint.** The first version scored spellF1 0.35 on the tuning matches. Our synthetic match had suggested the hint was harmful, so it was off; Sportec showed it adds about 0.09, so it is on. This is why synthetic data are used for testing only.
- **Richer hint features** (team speed difference, team extent, longer lags) – no gain (0.505 vs 0.503), dropped (`hint_rich = False`).
- **Iterated self-training** (re-fitting the hint on the decoder's own output for one or two more rounds) – worse (0.49–0.50), dropped (`hint_rounds = 0`).
- **A ball-free dead/alive hint re-fitted on the decoder's output** – worse (0.47–0.48), dropped (`dead_iter = False`).
- **Re-tuning against "controlled possession" labels.** Sportec's ownership field changes on brief touches (40–50 % of its possessions are shorter than 3 s), while the organiser's definition counts only controlled possession (CHALLENGE.md). We built a variant of the Sportec labels in which a team spell shorter than 3 s lying between two spells of the other team does not count (the organiser's "a few seconds" of control; `MIN_CONTROLLED_S` in `external_validation/config.py`), and re-ran a 64-combination grid over `c_switch`, `min_poss_s`, `min_edge_s` and `touch_bonus` (`external_validation/grids/next_hmm_controlled.json`). Under those labels the unchanged model scores 0.585 on the tuning matches and 0.585 on the held-out ones (window-125 baseline: 0.26). The best re-tuned setting gained 0.02 on the tuning matches, and on the held-out matches it improved one and worsened the other. That is inside the noise band, so the parameters were **not** changed.
- **Team share among the k nearest players** – never needed; not implemented.
- **Homography, video and broadcast tracking** – not attempted.

### 2.5 What is fitted at run time, and on what

The pipeline fits three small things on the match it is labelling. None of them uses a human label, the ground truth or the video; all are deterministic.

| What | Fitted on | Target |
|---|---|---|
| Ball-free team hint (logistic regression, per half) | Team-shape features of the tracking | The sign of the ball-proximity evidence on frames where it is confident (pseudo-labels from our own algorithm) |
| Player-tempo dead-ball term (logistic regression) | Median player speed, share of slow players | Frames our dead-ball cues are sure about (ball outside the pitch, kick-off formation) vs frames with a fast ball inside the pitch |
| Smart Tagging polarity (one number per code) | Tagged phases | Agreement between the phase's `Team` tag and the ball-proximity evidence |

RULES.md explicitly allows this on the provided match ("Learning without human labels – unsupervised or self-supervised methods, or pseudo-labels produced by an algorithm, provided no person decided any of them"). The same fitting runs unchanged on each Sportec match in the validation.

---

## 3. Possession rules we implemented

| Organiser's rule | Our implementation |
|---|---|
| When the ball is in play, one team always has possession (no neutral state) | The HMM has exactly three states, `H`, `A`, `D`. There is no neutral or contested state, and inside the assessed halves the model never outputs `X` |
| **Loose balls** (50/50, ball rolling after a tackle): the last team in possession keeps it | A frame without evidence scores 0 for every state and staying is free, so the decoder keeps the last team. This covers a ball in flight, a ball rolling away from everyone, and a ball the tracker has lost |
| Possession changes only when a player of the other team has **control** | "Control" on a frame = ball within 2 m of a player, that team clearly nearer than the other, ball slower than about 10 m/s. A change needs enough such frames to outweigh the switching cost (3.0), and a spell shorter than 1 s between two spells of the other team is removed. The boundary is then placed on the first frame of that control, so the pass before it stays with the passer |
| One touch without control, a deflected pass, or a won header without subsequent control is **not** a change | A touch adds evidence for about five frames; it cannot pay for switching away and back (2 × 3.0), so the surrounding team keeps the ball. A deflection that does not slow the ball contributes even less, because fast-ball frames are down-weighted |
| **Restarts** (kick-off, throw-in, corner, goal kick, free kick) have a clear owner | The decoder picks the team that the evidence after the restart favours. If the *other* team was on the still ball during the last 2 s of the stoppage (at least 10 control frames, four times more than the opponent), or a set-piece tag of a measured-reliable type names it, that team owns the frames from the restart until the following team first controls the ball. Throw-ins carry no team tag and often no ball, so there the owner is whoever the evidence favours |
| Dead ball is always `D` | `D` is a state of the HMM with the cues of section 1.3. Stoppages shorter than 2 s are removed as tracking blips. Ownership of a dead ball is never output |
| Two halves; the second half is assessed from frame 110 808 while kick-off is at 113 742 | The assessed spans are read from the organiser's `games.half_video_blocks()` (11 185–90 651 and 110 808–190 197); each half is decoded on its own. The kick-off frame is **not** hard-coded: assessed frames before the detected kick-off are forced to `D` (section 1.3). No other frame number appears in the code. `tools/sanity_check.py` checks this on the submitted file (checklist B7); see section 4.2 for the known deviation in the last seconds before the kick-off. |
| Use of `X` | `X` only outside the two assessed halves, where it is never scored. Inside them we never abstain: on spellF1 an `X` is forward-filled with our previous label, so it can never score better than writing that label ourselves, and it costs macroF1 |

---

## 4. Limitations – where the model fails and what we would fix with more time

### 4.1 Known so far (external validation and data)

- **Domain gap.** Sportec data are corrected tracking with the ball almost always present. Our degradation (section 5.3) imitates the competition data only in broad terms: the ball-missing share and gap behaviour are taken from public documentation and general football knowledge, not measured on the competition match.
- **Simplified degradation (v1).** No duplicate slots, no noise on positions, and a 20-second half-time break instead of a real one (it only separates the halves). The model may therefore meet noisier data in the competition match than in validation.
- **Different possession definitions.** Sportec's per-frame possession field follows the DFL definition, which differs from the organiser's (e.g. for loose balls). Sportec scores therefore compare parameter values with each other; they are **not** an estimate of the competition score.
- **Ball-owner gaps in the Sportec ground truth.** When the ball is alive but Sportec has no ball-owning team for a frame, the label builder keeps the team that last had the ball in the same half (at the very start of a half: the next known owner), following the organiser's loose-ball rule (`fill_owner` in `external_validation/build_dataset.py`, tested in `tests/test_ground_truth.py`). On match J03WPY this never occurs (0 of 81,460 alive frames); the other six matches have not been checked.

### 4.2 Model limitations

Based on the Sportec validation only.

- **Many possessions are still not matched.** On the held-out matches macroF1 is 0.90 but spellF1 is 0.58: most frames are right, yet many spells are split, merged or misplaced. We did not break this down by error type, so we cannot say how much is dead-ball timing, short spells or team errors.
- **Weaker than the hand-weighted evidence on our synthetic test match.** On three synthetic matches from `tools/make_fake_match.py` the learned evidence scored below the hand-weighted evidence on two and predicted up to 40 % too many possessions. The synthetic match was built around the hand-built cues and is not used for any choice (section 2.3), but it shows that the learned evidence can over-segment on tracking unlike its training data. `python -m src.pipeline --hand-weighted` produces the earlier model's file.
- **Second-half kick-off a few seconds early.** The organiser's FAQ states that the second half does not kick off until frame 113,742 and that every assessed frame before it is `D`. Our detected kick-off (Smart Tagging window, formation breaking, ball leaving the centre spot) falls within the last five seconds before that frame, so the file goes live slightly early there. We did not hard-code the published frame: RULES.md does not allow constants tied to specific frames of the game. The effect is at most about 125 of 158,857 assessed frames and the start of one possession.
- **Trained on one data source.** The learned evidence has seen five Bundesliga matches from one provider, degraded by our own rules (section 5.3). How well it transfers to the competition tracking cannot be measured without labels.
- **Decoder settings are on a plateau.** Across the 192 decoder settings searched with the learned evidence, the best dozen lie within 0.004 spellF1 of each other, far inside the scorer's noise band (0.04).
- **Short possessions.** The switching cost and the 1 s minimum remove brief spells by design. That matches the organiser's "controlled possession", but a genuine quick interception followed by an immediate loss is lost too.
- **Long ball gaps.** With no ball, a change of possession can only be seen through team movement, which lags by a second or two; the boundary is then late.
- **Restarts without a visible ball** (most throw-ins): the owner is inferred from what happens afterwards, so a restart that is lost at once is given to the wrong team.
- **Untested parts.** Sportec has no Smart Tagging, so everything that uses it (kick-off anchoring, set-piece cues, team prior) is tested on synthetic data only, and its weights are design choices, not tuned values.
- **Tracking noise.** The validation data have no position noise and no duplicate slots; the possession radius and the touch threshold may be too tight for noisier tracking.

### 4.3 With more time

1. **An error breakdown on Sportec** (dead-ball timing vs short spells vs team errors) before changing anything else.
2. **More and more varied training data** for the learned evidence (other open datasets, several degradation seeds per match), and a sequence model in place of per-frame trees.
3. **More realistic degradation** in the validation (position noise, duplicate slots, false ball detections) so that radii and thresholds can be tuned.
4. **Explicit restart states** per set-piece type, so that the restart owner is part of the decoding rather than a correction after it.
5. **The broadcast tracking** as a second ball source for the frames where the tactical tracking has none.

---

## 5. External validation and tuning on Sportec matches

All training, tuning and calibration was done on **external open data only** (organiser's e-mail to all participants, ~5 Nov 2026; our rule R9). This section is the evidence that our parameters were chosen legitimately. The code is in `external_validation/` and can be re-run by anyone; it never reads competition data.

### 5.1 Dataset

Sportec/IDSSE open tracking data, Bundesliga 2022/23 (Bassek et al., 2025), licence CC-BY 4.0. 25 fps, the same frame rate as the competition match. Seven full matches, loaded with `kloppy` 3.19.1 (`sportec.load_open_tracking_data(match_id=..., only_alive=False)`). The match IDs are listed in `external_validation/config.py`.

| Match ID | Role |
|---|---|
| J03WPY | Tuning |
| J03WMX | Tuning |
| J03WN1 | Tuning |
| J03WOH | Tuning |
| J03WOY | Tuning |
| J03WQQ | Held-out check |
| J03WR9 | Held-out check |

The split was fixed before any score was computed.

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
- **Protocol:** all choices were made on the five tuning matches. The two held-out matches were scored once, after the values had been written into `src/config.py`, and nothing was changed afterwards. The model is called through `src/extval.py`, a thin adapter around the same code the competition pipeline runs; the only difference is that Sportec has no Smart Tagging, so that evidence is absent in validation.
- **What was run:** about ten sweeps on the tuning matches (transition costs, evidence strengths, hint settings, dead-ball weights, durations, refinement windows, and the three extensions in section 2.4). The first two exploratory sweeps used a parallel driver that calls the same `predict` and the same scorer; everything reported here comes from `sweep.py`. The grid that documents the final choice is `external_validation/grids/next_hmm.json`.

### 5.5 Parameters and how each was set

Every free parameter in the final pipeline must appear here with its source. Allowed sources: tuned on Sportec, taken from literature, or set from football knowledge (with justification). "Chosen from the score on the public 5 minutes" or "looked right on video" are **not** allowed sources.

All values live in `src/config.py`; names below are the field names there. Three kinds of source:

- **Tuned** – chosen on the five Sportec tuning matches.
- **Prior, checked** – set from literature or football knowledge, then varied on Sportec; the prior was kept because no alternative was better by more than the noise band.
- **Prior** – set from literature or football knowledge and not varied (either Sportec cannot test it, or it is a guard value).

**Learned evidence and decoder (final model), chosen on Sportec**

| Parameter | What it controls | Final value | Search range |
|---|---|---|---|
| `learned` | Decoder input from the trained models instead of the hand-weighted evidence | on | on / off |
| `learned_team_scale`, `learned_dead_scale` | How much of each frame's log-odds counts (neighbouring frames are near-copies) | 0.4, 0.4 | 0.05 – 0.4 each |
| `c_switch` | H↔A switching cost | 2.0 | 2 – 6 |
| `c_die`, `c_restart` | Cost of entering / leaving `D` | 10, 10 | 4 – 10 (kept equal) |
| `learned_clip` | Cap on one frame's log-odds | 6.0 | not varied (3 gave the same score) |
| Trees | 150 trees of at most 15 leaves per model, learning rate 0.06, at least 200 frames per leaf | – | not varied |

The 192 combinations are in `GRID` in `external_validation/train_emissions.py`; the best was taken automatically. The tables below describe the hand-weighted evidence, which is still computed and is now the main input of the trained models. Its parameters keep the values found there, except `c_switch`, `c_die` and `c_restart`, which were re-chosen as above.

**Tuned on Sportec (hand-weighted evidence)**

| Parameter | What it controls | Final value | Search range |
|---|---|---|---|
| `c_switch` | H↔A switching cost: how much evidence a change of team needs | 3.0 | 0.25 – 7; 2 and 3 are equal within noise, 3 was taken as the less twitchy |
| `use_hint` | Whether the ball-free team hint is used | on | on / off (+0.09 with it on) |
| `touch_bonus` | Evidence per frame around a detected touch | 0.6 | 0 – 1.0 |
| `dead_clip` | Cap on the player-tempo dead-ball term | 0.08 | 0.05 – 0.4 |

**Prior, checked on Sportec**

| Parameter | What it controls | Final value | Prior from | Range checked |
|---|---|---|---|---|
| `c_die`, `c_restart` | Cost of entering / leaving `D` | 6.5, 6.0 | Stoppages are rare events (one per ~30 s of play) | 2 – 10 |
| `r_ctrl` | Radius of a player's possession zone | 2.0 m | Vidal-Codina et al. use 0.5–1 m on manual-grade tracking; widened for automatic tracking | 1.0 – 3.0 |
| `v_ctrl` | Ball speed above which a ball is passing by, not controlled | 10 m/s | Slow-ball condition of earlier rule-based systems | 6 – 16 |
| `beta`, `s_gap` | Strength and distance scale of the proximity evidence | 0.5, 1.0 m | Design | 0.25 – 2; 0.5 – 2 |
| `touch_dv` | Velocity change that counts as a touch | 3 m/s | Kick detection, Link & Hoernig (4 m/s² on smoothed data) | 2 – 5 |
| `ball_max_gap`, `interp_weight` | Longest ball gap that is interpolated, and its weight | 25 frames, 0.5 | Design | 0 – 125; 0.25 – 1 |
| `hint_shrink`, `hint_clip`, `hint_on_ball` | Scale, cap and on-ball weight of the hint | 0.15, 0.3, 0.3 | Neighbouring frames are near-copies, so each counts for little | 0.15 – 0.6; 0.3 – 1.2; 0.3 – 1 |
| `w_still`, `w_fast`, `w_missing`, `long_gap` | Dead-ball weights: still ball, fast ball in play, long ball gap and its length | 0.4, 0.3, 0.1, 75 frames | Design | 0 – 1; 0 – 1; 0 – 0.3; 25 – 150 |
| `w_kickform` | Dead-ball weight of the kick-off formation | 1.0 | Laws of the game (Vidal-Codina's kick-off trigger) | on / off |
| `dead_selftrain` | Player-tempo term fitted at run time instead of a fixed threshold | on | – | on / off |
| `min_poss_s`, `min_edge_s`, `min_dead_s` | Minimum durations | 1.0, 0.4, 2.0 s | Organiser's "controlled possession"; a real stoppage lasts seconds | 0.3 – 2; 0.2 – 1; 1 – 4 |
| `refine`, `refine_team_s`, `refine_dead_s` | Boundary refinement and its search windows | on, 2.0 s, 1.0 s | Design | on / off; 0.5 – 4; 0.5 – 2 |
| `use_restart` | Restart-owner correction | on | Organiser's rule on restarts | on / off |
| `hint_rich`, `hint_rounds`, `dead_iter` | The three dropped extensions (section 2.4) | off, 0, off | – | on / off; 0 – 2 |

**Prior, not varied**

| Parameter | What it controls | Final value | Source |
|---|---|---|---|
| `ball_vmax`, `ball_min_seg` | Ball-jump filter | 45 m/s, 5 detections | A football does not exceed ~35 m/s in play |
| `s_ctrl`, `s_v`, `touch_r`, `touch_half` | Softness of the radius and speed limits; touch distance and width | 0.5 m, 2 m/s, 2.5 m, ±2 frames | Design |
| `hint_conf`, `hint_stride`, `hint_min_frames` | Which frames become pseudo-labels for the hint | 0.3, every 5th, ≥ 400 per class | Design |
| `oob_margin`, `w_oob` | Ball-out margin and weight | 0.75 m, 1.5 | Margin for tracking noise; strongest dead-ball cue |
| `still_spread` | Spread below which a detected ball counts as lying still | 0.35 m | Design |
| `dead_min_frames`, `dead_shrink` | Fitting of the player-tempo term | ≥ 300 per class, 0.25 | Design |
| `v0_players`, `s_players`, `w_players` | Fallback player-speed threshold if the tempo term cannot be fitted | 1.6 m/s, 0.5, 0.1 | Football knowledge |
| `use_tags`, `tag_gamma`, `tag_min_agree`, `tag_min_frames` | Smart Tagging team prior | on, 0.03, 0.2, 250 | Design; **cannot be tested on Sportec** (no Smart Tagging). `tag_gamma` is set so that a 10 s phase alone cannot create a possession |
| `w_setpiece`, `setpiece_pre_s`, `w_goal`, `w_prekick` | Smart Tagging dead-ball cues and the forced `D` before kick-off | 0.3, 3 s, 0.5, 8.0 | Design; cannot be tested on Sportec |
| `restart_look_s`, `restart_min_frames`, `restart_max_s` | Restart-owner correction | 2 s, 10 frames, 6 s | Design |

**`X` policy:** not a parameter. `X` is written outside the assessed halves only.

No value was chosen from the public 5-minute ground truth, from the video, or from any measurement on the competition match.

### 5.6 Results on Sportec

Sportec results only. "Baseline" is `example_baseline.py` with a window of 50 frames.

**Final model (learned evidence)**, from `external_validation/train_emissions.py`

| | spellF1 | macroF1 | Per match |
|---|---|---|---|
| Tuning matches, leave-one-match-out (each match scored by models trained on the other four) | 0.546 (spread 0.030) | 0.890 | lowest 0.515 |
| Held-out matches, models trained on the five tuning matches, scored with the chosen decoder | **0.582** | **0.899** | J03WQQ 0.595, J03WR9 0.570 |
| Hand-weighted evidence, same matches | 0.501 / 0.514 | 0.862 / 0.869 | tables below |

**Hand-weighted evidence (the model until 15 Nov)**, from `sweep.py`. "Ours" below is that model (`learned=False`, `c_switch` 3, `c_die` 6.5, `c_restart` 6).

**Held-out matches** (not used for any choice)

| Match | Baseline spellF1 | Ours spellF1 | Baseline macroF1 | Ours macroF1 |
|---|---|---|---|---|
| J03WQQ | 0.240 | 0.506 | 0.364 | – |
| J03WR9 | 0.236 | 0.522 | 0.359 | – |
| **Mean** | **0.238** | **0.514** | **0.361** | **0.869** |

(`sweep.py` reports macroF1 for our model as a mean over the matches only.)

**Tuning matches**

| Match | Baseline spellF1 | Ours spellF1 |
|---|---|---|
| J03WPY | 0.212 | 0.526 |
| J03WMX | 0.227 | 0.448 |
| J03WN1 | 0.224 | 0.491 |
| J03WOH | 0.205 | 0.540 |
| J03WOY | 0.217 | 0.501 |
| **Mean** | **0.217** | **0.501** (spread 0.036, macroF1 0.862) |

**What moved the score** (mean spellF1 on the tuning matches)

| Change | spellF1 |
|---|---|
| Hand-set starting values, hint off, `c_switch` 6 | 0.350 |
| + ball-free hint on | 0.438 |
| + `c_switch` 3 | 0.480 |
| + `touch_bonus` 0.6 | 0.494 |
| + `dead_clip` 0.08 (final) | 0.501 |

On the held-out matches, with the other final values in place: 0.383 (hint off, `c_switch` 6), 0.412 (hint off, `c_switch` 3), 0.487 (hint on, `c_switch` 6) and 0.514 (final).

The held-out score is as good as the tuning score, so the choice did not overfit the five tuning matches. The baseline's low macroF1 is partly because it never outputs `D`. These numbers compare configurations with each other on Sportec's possession definition; they are **not** an estimate of our score on the competition match.

### 5.7 Freezing parameters

All parameters and both trained models were chosen on Sportec data only, as described in sections 5.4–5.6: the hand-weighted evidence with the grid in `external_validation/grids/next_hmm.json`, and the learned evidence and its decoder settings with `external_validation/train_emissions.py`, under an adoption rule fixed beforehand (better on the tuning matches and on both held-out matches). The submitted file is the one produced by the commit tagged `phase1-submission`.

---

## 6. Reproduction – how to run it

> Must be **identical** to the README (checklist G6).

**Requirements:** Python 3.13 (tested with 3.13.5), macOS (tested) or Windows (the code uses no OS-specific paths or shell scripts, but the byte-identical file has so far been verified on macOS only). All dependencies, including indirect ones, are pinned with `==` in `requirements.txt`.

```bash
git clone https://github.com/NextConsultingBarcelona/more-than-a-hack.git
cd more-than-a-hack
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Data:** The match data are **not** included in the repository. Place the organiser's data in a folder of your choice, outside the repository, and set:

```bash
# macOS:   export MTAH_DATA_DIR=/path/to/data
# Windows: set MTAH_DATA_DIR=C:\path\to\data
```

**Run (one command):**

```bash
python -m src.pipeline --team-name NEXT --no-cache   # writes NEXT.csv into the organiser's results folder
```

This is the same command as in the README, which also gives the two commands that check the file. The pipeline uses no random numbers, so the same input gives the same bytes; a test checks this on synthetic data (`tests/test_pipeline_fake.py`).

Expected runtime: about 15 seconds on a laptop, without cache. The output is deterministic: there are no random numbers and nothing is tuned at run time. Expected SHA-256 of `NEXT.csv`: `f12d15ce52ebeba71c6fa5c245e19bf061c4360a4b5e3ddebea4214a1d63ae6f` (macOS, Python 3.13.5).

**Reproducing the external validation (optional, open data only):**

```bash
python -m pip install -r external_validation/requirements-validation.txt
python external_validation/build_dataset.py --selftest        # ~1 s, synthetic data, checks the setup
python external_validation/build_dataset.py --matches all     # downloads the 7 Sportec matches via kloppy
python external_validation/sweep.py external_validation/grids/next_hmm.json --matches J03WPY J03WMX J03WN1 J03WOH J03WOY
```

python external_validation/train_emissions.py --jobs 6    # trains the learned evidence, searches the decoder, writes src/learned_model.json
```

The `sweep.py` command reproduces the grid behind the hand-weighted model (32 combinations, a few minutes); the last command reproduces the final model and its scores (about two minutes).

```text If the kloppy download times out (the tracking files are about 370 MB each), re-run it; the cache keeps finished files.

The Sportec files are downloaded and cached by kloppy outside the repository; generated files are written to `ext_data/`, which is git-ignored. Loading one match needs a few GB of RAM.

---

## 7. External data, models and tools

Required by the rules (R7): everything external that we used, with licence and credit. Nothing private.

**Datasets**

| Dataset | Use | Licence | Credit |
|---|---|---|---|
| Sportec/IDSSE open tracking data, Bundesliga 2022/23 (match IDs J03WPY, J03WMX, J03WN1, J03WOH, J03WOY, J03WQQ, J03WR9) | Tuning and validation (section 5) | CC-BY 4.0 | Bassek, M., Rein, R., Weber, H. et al. (2025). *Scientific Data* 12, 195. doi:10.1038/s41597-025-04505-y |

**Pre-trained models:** No third-party pre-trained model is used. Our own two tree models, trained by us on the Sportec matches above, are stored in the repository as `src/learned_model.json` (about 0.3 MB) and can be re-created with `python external_validation/train_emissions.py`. The small logistic regressions of section 2.5 are fitted at run time and not stored.

**Software libraries**

| Library | Version | Licence | Use |
|---|---|---|---|
| LightGBM | 4.7.0 | MIT | Training the two tree models on Sportec data (`external_validation/train_emissions.py`). **Not** needed to produce the submission: the trained trees are stored as arrays and evaluated with numpy |
| kloppy | 3.19.1 | BSD-3-Clause | Loading Sportec data in the external validation only. The competition pipeline does not use it |
| numpy | 2.5.4 | BSD-3-Clause | The whole competition pipeline (`src/`): parsing, features, logistic regression, Viterbi. No scipy, scikit-learn or HMM library is used |
| pandas | 3.1.1 | BSD-3-Clause | The organiser's scorer and the external validation; not needed to produce the submission |
| pyarrow | 26.0.1 | Apache-2.0 | Parquet files in the external validation |
| Organiser's code (`games.py`, `evaluation/`, `starter/`) | – | MIT | `games.py` for file paths and the assessed spans; `evaluation/` for scoring in the external validation and in tests; `starter/baseline.py` in one test as a comparison on synthetic data |

**Development tools:** Claude Code (Anthropic) was used as a coding assistant on synthetic and open (Sportec) data: it wrote most of the code and ran the Sportec experiments. It was never given the competition data and never ran the pipeline on the competition match.

---

## 8. References

- Bassek, M., Rein, R., Weber, H. et al. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195. doi:10.1038/s41597-025-04505-y
- Borghesi, M., Costa, L. D., Morra, L., & Lamberti, F. (2023). Using Temporal Convolutional Networks to estimate ball possession in soccer games. *Expert Systems with Applications* 223, 119780. doi:10.1016/j.eswa.2023.119780
- Kim, H., Lee, K., Seo, S., Ko, S.-K., Yoon, J., & Park, C. (2026). PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories. *KDD '26*. doi:10.1145/3770855.3818152
- Link, D., & Hoernig, M. (2017). Individual ball possession in soccer. *PLoS ONE* 12(7), e0179953. doi:10.1371/journal.pone.0179953
- Vidal-Codina, F., Evans, N., El Fakir, B., & Billingham, J. (2022). Automatic event detection in football using tracking data. arXiv:2202.00804
- More than a Hack 2027, official repository: https://github.com/metrica-sports/more-than-a-hack-2027
- kloppy: https://github.com/PySport/kloppy

---

## 9. Compliance statement

- **Labels come from the algorithm (R2).** Labels are produced entirely by the code in the repository. No manual labelling, no manual correction of output and no frame-specific rules were used.
- **External training and tuning only (R9).** Every parameter and both trained models were set from literature, football knowledge or external open data (Sportec/IDSSE). No parameter or model was selected using the competition match, the public ground truth or the video. At run time the pipeline fits three small models on the match's own tracking using pseudo-labels from our own algorithm, with no human label, no ground truth and no video (section 2.5), which RULES.md allows as learning without human labels.
- **Public ground truth (R3).** The public 5-minute ground truth was used only as a pipeline check: that the pipeline runs end-to-end and that its output is valid and not broken. No full ground truth, and nothing derived from it, was used.
- **No video comparison (R8).** Our labels were never compared with the video, neither manually nor by "checking whether it looks right".
- **No human-made data about this match.** No event data, hand-labelled data or other human-annotated data about this match were used.
- **No cloud AI (R1).** No competition data or excerpts of it were shared with cloud-based AI services.
- **No data in the repository (R5).** No competition data are included in the repository or its history.
- **Reproducible (R4).** One command reproduces the submitted file exactly (section 6).
- **Everything external is listed (R7).** All external data, models and libraries are listed in section 7.
