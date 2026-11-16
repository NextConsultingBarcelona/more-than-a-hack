# NEXT – Ball Possession Detection, More than a Hack 2027 (Phase 1)

| | |
|---|---|
| **Team** | NEXT |
| **Members** | Oskar Rändel (captain, possession model) · Martin (dead-ball model, pipeline, submission) · Morten (external validation, QA, documentation) |
| **Repository** | https://github.com/NextConsultingBarcelona/more-than-a-hack |
| **Commit / tag** | the commit tagged `phase1-submission` [SJEKK: Oskar – the tag has to be moved to the commit that includes the stoppage detector] |
| **Submitted file** | `NEXT.csv` (SHA-256: [SJEKK: Oskar/Martin – SHA-256 of the file produced with the stoppage detector. `f12d15ce52ebeba71c6fa5c245e19bf061c4360a4b5e3ddebea4214a1d63ae6f` is the file without it, which `--no-detector` reproduces]) |

---

## Summary

Our system turns the automatic tracking into one label per frame in four steps: per-frame cues from players and ball, evidence for "home vs away" and for "dead ball", a three-state hidden Markov model (`H`, `A`, `D`) decoded with Viterbi, and a clean-up that snaps each boundary to the event it stands for. The key idea is that a frame with nothing to say (ball in flight, ball missing) carries no evidence, so the model keeps the last team, and a switching cost means a single touch cannot change possession. This is the organiser's loose-ball rule expressed as a model, and it is what spellF1 rewards: whole possessions, not frames. Because the ball is missing on about 41 % of assessed frames, the model also learns, without any labels, how team shape and movement relate to possession on the frames where the ball *is* seen, and uses that where it is not. The per-frame evidence that goes into the decoder comes from two small gradient-boosted models trained on five open Sportec matches, with our hand-built cues as their inputs; decoder settings were chosen on the same five matches and checked on two held-out ones (spellF1 0.58 against 0.24 for the nearest-player baseline). Stoppages are found by a separate detector, trained on the seven open matches with several pictures of the ball during stoppages, because open tracking data freeze the ball while play is stopped and an automatic tracker need not. It lifts the held-out score to 0.60, and the score stays within 0.03 when the ball is shown differently during stoppages (sections 1.3 and 5.8). A second home-vs-away model, trained on the same open matches with ownership labels in which an uncontrolled touch is not a change of possession, adds local detail around the ball and the two teams as collectives. With both, the held-out score is 0.627 against Sportec's ownership as it is and 0.686 when uncontrolled touches are not counted, which is the organiser's definition (section 5.9). Nothing was trained or tuned on the competition match and we report no score on it.

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
- *Output.* The models' log-odds, scaled by 0.4 and capped at ±6 per frame, replace `team_llr` and `dead_llr` in the decoder. The Smart Tagging terms (team prior, set pieces, the stretch before kick-off) are added on top unchanged, because Sportec has no Smart Tagging to learn them from. In the final pipeline the dead-vs-alive evidence is in turn replaced by the stoppage detector of section 1.3; the dead-vs-alive model remains as its fallback.
- *Why trees, and why only as evidence.* Borghesi et al. (2023) show that learned per-frame evidence beats rule-based cues, and Kim et al. (2026) that structured decoding matters more than the per-frame model. We kept our decoder and clean-up and replaced only the weighting of the cues. The trees are stored as plain arrays in `src/learned_model.json` and evaluated with numpy, so the pipeline needs no machine-learning library and is deterministic.

**Second home-vs-away model (final model).** In the final pipeline the home-vs-away log-odds come from a second tree model (`src/team_model.py`, `models/team_trees.npz`). It sees the 36 inputs of the first model and 151 more, all from the same tracking:

- *Local shape.* Which team is closer to the ball and how close, at ten offsets from 2 s before to 2 s after the frame, and the ball's speed at six. The first model sees these only as running means.
- *Touches and contacts.* A touch is a peak in the ball's change of velocity next to a player (section 1.4); a clear contact is one team within 0.8 m (or 1.5 m) of the ball and clearly nearer than the other. For each kind: which team had the last one and will have the next one, how long ago and how soon, and how many each team has in the 1 to 6 s around the frame.
- *Ball gaps.* How long the ball has been unseen and how soon it is seen again, and who was nearer on each side of the gap.
- *The teams as collectives*, each seen along its own direction of attack: where the block and its back and front lines stand, how wide and deep it is, its mean speed and the share of its players above 4 m/s, how fast the block and its lines move forward, and the share of its players in the opponent's half. These enter as home minus away, with means over 1 and 3 s and the change over 2 s. A team that has the ball spreads and moves up; a team without it closes up and drops.

It has 300 trees of at most 31 leaves, is trained on ownership labels in which an uncontrolled touch does not count (section 5.9), and is made exactly antisymmetric in the same way as the first model. Nothing in it is fitted at run time. The decoder is unchanged, with one exception in the clean-up below. If the second model fails, the first model's log-odds are used and the run says so.

**Transitions.** The H↔A switching cost is the parameter that defines "controlled" possession: an opponent spell inside a possession must collect more evidence than two switches cost, which a deflection or a single touch cannot. After switching the ball-free hint on, it was the most influential setting in the external validation (section 5.6).

**Decoding.** Viterbi is run separately on each assessed half, so nothing leaks across half-time. Frames outside the two halves are written as `X`; inside them the model never abstains (section 3).

**After decoding** (same file, in this order):

1. *Minimum durations.* A stoppage shorter than 2 s is treated as a tracking blip; a team spell shorter than 1 s between two spells of the other team is a touch, not a possession.
2. *Boundary refinement.* A team→team boundary moves to the first frame the new team controls the ball after the old team's last control, so a pass in flight still belongs to the passer. A live→dead boundary moves to the first frame the ball is seen outside the pitch. A dead→live boundary moves back over a fast in-play ball or forward over a ball still lying on its spot. With the second home-vs-away model the team→team boundary is left where the model puts it (`tm_refine_team_s = 0`): its local features place the change better than the snap does (section 5.9).
3. *Restart owner.* If the other team was clearly on the ball as play restarted, it owns the frames from the restart until the following team first controls the ball (section 3).

### 1.3 Dead-ball model (D) and transitions

`D` is a state of the same HMM. The hand-built dead-ball evidence `dead_llr` is a sum of:

- **ball outside the pitch** by more than 0.75 m (strongest cue);
- **kick-off formation**: both teams entirely in their own halves for about two seconds (restart after a goal);
- **ball lying still** (detected, and staying within 0.35 m for about a second) counts for dead; a **fast ball inside the pitch** counts for alive;
- **ball missing for more than 3 s** counts weakly for dead;
- **Smart Tagging**: the 3 s before a tagged set piece, and the span between a tagged goal and the kick-off tag that follows it;
- **player tempo**: a small logistic model on median player speed and the share of slow players. It is fitted at run time, without labels, on the frames the cues above are sure about (section 2.5). Its contribution is capped so that slow players alone need several seconds to create a stoppage.

**Stoppage detector (final model).** In the final pipeline the decoder's dead-ball evidence is neither `dead_llr` nor the dead-vs-alive model of section 1.2, but the output of a separate detector (`src/dead_model.py`):

1. *Features (279 per frame).* The players as a collective: the distribution of their speeds and accelerations, how coherent the movement is, team shape, players beyond the lines and in the penalty areas, whether each team is in its own half, distance to the nearest opponent. The same over windows of 1 to 20 s before and after the frame. And the ball where it is seen: position relative to the lines, speed, how long it has been lying still, distance to the nearest players, and what was seen before and after each gap in the ball track. No feature uses player identity, slot order or the direction of play.
2. *Classifier.* 300 gradient-boosted trees give the log-odds that the ball is dead on each frame. They were trained on the seven open Sportec matches (section 5.8), are stored in `models/dead_trees.npz` and are evaluated with numpy alone. Nothing in the detector is fitted at run time.
3. *Stoppages.* A two-state Viterbi pass with a switching cost turns the log-odds into stoppages. Stoppages under 2 s are dropped, and live gaps under 1 s inside a stoppage are closed.
4. *Into the HMM.* The stoppages enter the H/A/D model as hard evidence (`dm_weight`), so `D` in the output is where the detector found a stoppage. The team evidence, the duration rules, the boundary refinement and the restart owner of section 1.2 run around it unchanged; `dead_llr` is still an input of the home-vs-away model, exactly as in its training.

*Why a separate detector.* Sportec keeps a placeholder ball through every stoppage: frozen where play stopped, then moved to the restart point on the restart frame. On such data "the ball is lying still" marks a stoppage almost by construction, and both `dead_llr` and the dead-vs-alive model of section 1.2 were tuned and trained on it. An automatic tracker need not show that. When we changed only what the ball does during stoppages in the validation data, the model without the detector fell from 0.556 to between 0.394 and 0.529 spellF1, and the model with it stayed between 0.538 and 0.568 (section 5.8). The detector was therefore trained with several such pictures.

*Fallback.* If the detector's output is not plausible for a football match (dead ball on less than 8 % or more than 70 % of the assessed frames, or fewer than 30 or more than 220 stoppages per 90 minutes), or the detector fails, the decoder uses the dead-vs-alive evidence of section 1.2 instead and the run says so. The check uses no labels, and its limits are far outside anything seen on the open matches (dead ball 41 to 52 %, 92 to 113 stoppages per match).

**Start and end frames.** The detector places the boundaries; refinement can then move one by up to 1 s, to the ball leaving the pitch (live→dead) or to the ball leaving its spot (dead→live) (section 1.2).

**Restart owner.** The team whose player is on the still ball in the last 2 s of the stoppage, or the team named by a set-piece tag when that tag type has been measured to name the taker (section 1.4).

**Second-half kick-off.** Assessed frames before the kick-off are forced to `D`, whatever the detector says (in the fallback through an overwhelming dead-ball term, `w_prekick`). The kick-off frame is not hard-coded: the Smart Tagging "Kick Off (Start)" tag gives a window, and inside it the kick is pinned by the formation breaking and, where the ball is seen, by the ball leaving the centre spot.

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
| Smart Tagging | Yes | Four uses: (1) the kick-off tag anchors the search for each half's kick-off; (2) in the fallback dead-ball evidence only (section 1.3), the seconds before a tagged set piece and the span from a tagged goal to its kick-off count for dead ball; (3) phase `Team` tags give a weak team prior with measured polarity (section 1.4); (4) a set-piece `Team` tag can name the restart owner. It is treated as automatic evidence, not ground truth; `N/A` team tags (all throw-ins) are ignored. Team names are matched to home/away by string similarity, since the tagging and the tracking metadata spell them differently |
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
| **Sportec/IDSSE open tracking data** (Bundesliga 2022/23, CC-BY 4.0) | The *only* data used to train models and to tune or select parameters (section 5). **Training labels:** the dataset's own per-frame ball state and ball-owning team, published by the DFL with the tracking (Bassek et al. 2025); we labelled nothing ourselves. Five matches, about 730,000 frames, every fifth frame used for training. The stoppage detector is trained on all seven matches with the same ball-state label (section 5.8), the second home-vs-away model on the five tuning matches (section 5.9) |
| **Synthetic match from our own generator** (`test_data_generator/`) | Software testing only: correct format, 199 319 rows, no crashes, sensible H/A/D behaviour. Built only from the organiser's public documentation (DATA.md, CHALLENGE.md) and general football knowledge; nothing in it is calibrated against the competition match. **Never** used for training, tuning or model selection |
| **Synthetic match in the real file formats** (`tools/make_fake_match.py`) | Software testing of `src/` only: a toy match written as ATD text, metadata XML and Smart Tagging XML, with its own ground truth, so the whole pipeline is tested end to end. Built from the public documentation only. Not used for tuning; where it disagreed with the Sportec validation (section 2.4), Sportec decided |

### 2.4 Tried and dropped

Every decision below rests on the Sportec validation in section 5.

- **Hand-weighted evidence as decoder input.** This was our model until 15 Nov: spellF1 0.501 on the tuning matches and 0.514 on the held-out ones. Replacing the hand-set weights by the learned evidence of section 1.2 gave 0.546 (leave-one-match-out) and 0.582, with macroF1 up from 0.87 to 0.90, so the learned evidence is the final model. `learned=False` in `src/config.py` gives the earlier model back.
- **The trained dead-vs-alive model as the decoder's dead-ball evidence.** This was the final model until 16 Nov. On the validation data as built it scores 0.545 on the tuning matches and 0.584 on the held-out ones, against 0.555 and 0.600 with the stoppage detector; with another picture of the ball during stoppages it scores 0.39 to 0.53 and the detector 0.54 to 0.56 (section 5.8). Kept as the fallback; `dead_model=False` in `src/config.py`, or `--no-detector`, gives it back.
- **The first home-vs-away model alone.** This was the home-vs-away evidence until 16 Nov. With the stoppage detector it scores 0.555 on the tuning matches and 0.600 on the held-out ones, against 0.581 and 0.627 with the second model (section 5.9). Kept as the fallback; `team_model=False` in `src/config.py`, or `--no-team-model`, gives it back.
- **Other changes to the home-vs-away part**, tried on the tuning matches with the true stoppages given (exploratory runs; the scripts are not in the repository). Transition costs and scales beyond the grid of section 5.5, in which the chosen values lie on the edge: no change (0.545 → 0.547). Training the first model on labels without uncontrolled touches: no change. The first model's inputs with 300 trees of 31 leaves: +0.005. Twice as many leaves or twice as many trees for the second model: at most +0.007. A plain "the last team within 1 m of the ball" rule on the clean ball: 85 % of live frames right, against 90 % for the first model on the same data, because in two dimensions a ball that flies over a player looks like a touch.
- **Metrica's public sample matches as extra training data for the detector.** An early version of the detector was trained on them as well. They come without a formal licence, so they were dropped and the detector was retrained on Sportec alone. The stored detector and its decoding values do not depend on them. [SJEKK: Oskar and Martin – keep or delete this bullet]
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
| Dead ball is always `D` | `D` is a state of the HMM, and its evidence is the stoppage detector of section 1.3. Stoppages shorter than 2 s are removed as tracking blips. Ownership of a dead ball is never output |
| Two halves; the second half is assessed from frame 110 808 while kick-off is at 113 742 | The assessed spans are read from the organiser's `games.half_video_blocks()` (11 185–90 651 and 110 808–190 197); each half is decoded on its own. The kick-off frame is **not** hard-coded: assessed frames before the detected kick-off are forced to `D` (section 1.3). No other frame number appears in the code. `tools/sanity_check.py` checks this on the submitted file (checklist B7); see section 4.2 for the known deviation in the last seconds before the kick-off. |
| Use of `X` | `X` only outside the two assessed halves, where it is never scored. Inside them we never abstain: on spellF1 an `X` is forward-filled with our previous label, so it can never score better than writing that label ourselves, and it costs macroF1 |

---

## 4. Limitations – where the model fails and what we would fix with more time

### 4.1 Known so far (external validation and data)

- **Domain gap.** Sportec data are corrected tracking with the ball almost always present. Our degradation (section 5.3) imitates the competition data only in broad terms: the ball-missing share and gap behaviour are taken from public documentation and general football knowledge, not measured on the competition match.
- **Simplified degradation (v1).** No duplicate slots, no noise on positions, and a 20-second half-time break instead of a real one (it only separates the halves). The model may therefore meet noisier data in the competition match than in validation.
- **The ball during stoppages.** Sportec keeps a placeholder ball through every stoppage: frozen where play stopped, then moved to the restart point on the restart frame. Our degradation leaves it in (apart from the 41 % that is removed everywhere), so the validation as built favours any method that reads a still ball as a stoppage. Section 5.8 varies this picture. What the competition tracking shows during stoppages we do not know.
- **Different possession definitions.** Sportec's per-frame possession field follows the DFL definition, which differs from the organiser's (e.g. for loose balls). We measured one difference: the field changes whenever the other team touches the ball, while the organiser counts only controlled possession (section 5.9). Since 16 Nov we therefore report every score twice, against the field as it is (T0) and with team spells under 2 s between two spells of the other team not counted (T2). Which of them is nearer the competition's ground truth we cannot measure. Sportec scores compare versions and parameter values with each other; they are **not** an estimate of the competition score.
- **Ball-owner gaps in the Sportec ground truth.** When the ball is alive but Sportec has no ball-owning team for a frame, the label builder keeps the team that last had the ball in the same half (at the very start of a half: the next known owner), following the organiser's loose-ball rule (`fill_owner` in `external_validation/build_dataset.py`, tested in `tests/test_ground_truth.py`). On match J03WPY this never occurs (0 of 81,460 alive frames); the other six matches have not been checked.

### 4.2 Model limitations

Based on the Sportec validation only.

- **Many possessions are still not matched.** On the held-out matches macroF1 is 0.91 but spellF1 is 0.62 (0.68 when uncontrolled touches are not counted): most frames are right, yet many spells are split, merged or misplaced. With the true stoppages the same team decoding reaches 0.68 (section 5.9), so about 0.05 of the gap is dead ball. Most of the rest is short possessions: against the field as it is, 31 % of Sportec's possessions last under 2 s and we match few of them, by design (see "Short possessions" below).
- **Heavy position noise.** With 0.5 m of noise on the ball and the players, every version falls to about 0.44 on Sportec, and the gain from the detector and the second home-vs-away model is gone (section 5.9). At 0.25 m both still help. We do not know how noisy the competition tracking is.
- **The detector's training pictures are our own.** It has seen seven matches from one provider, and the pictures of the ball during stoppages that it was trained with are constructions, not observations of an automatic tracker. Section 5.8 shows that it does not depend on which of them is true, not that it covers everything a tracker might do.
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

1. **A finer error breakdown on Sportec** (short spells vs team errors) before changing anything else. Dead ball is now separated out (section 5.8), and it is the smaller part.
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
- **Protocol:** all choices were made on the five tuning matches. The two held-out matches were scored once, after the values had been written into `src/config.py`, and nothing was changed afterwards. The model is called through `src/extval.py`, a thin adapter around the same code the competition pipeline runs; the only difference is that Sportec has no Smart Tagging, so that evidence is absent in validation. The stoppage detector is trained on all seven matches, so it is scored leave-one-match-out (section 5.8).
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

**Stoppage detector, chosen on Sportec**

| Parameter | What it controls | Final value | Search range |
|---|---|---|---|
| `dead_model` | Stoppages from the detector instead of the decoder's own dead-ball evidence | on | on / off |
| `dm_cost` | Switching cost in the detector's own alive/dead Viterbi pass | 70 | 20 – 250; flat between 70 and 150 |
| `dm_bias` | Shift of the detector's log-odds per frame towards "play goes on" | −3 | 0 – −6; flat between −2 and −3 |
| `dm_clip`, `dm_min_dead_s`, `dm_min_alive_s`, `dm_weight` | Detector decoding: cap on the log-odds per frame, shortest stoppage, shortest live gap inside a stoppage, weight of its stoppages in the HMM | 8, 2 s, 1 s, 30 | Design, not varied; 2 s is the same floor as `min_dead_s` |
| `dm_share_lo`, `dm_share_hi`, `dm_stops_lo`, `dm_stops_hi` | Plausibility check on the detector's output (section 1.3) | 8 %, 70 %, 30 and 220 per 90 min | Guard values, far outside the open matches (41 to 52 % dead ball, 92 to 113 stoppages per match) |
| Trees | 300 trees of at most 63 leaves, learning rate 0.08, at least 200 frames per leaf | – | not varied |

**Second home-vs-away model, chosen on Sportec**

| Parameter | What it controls | Final value | How it was set |
|---|---|---|---|
| `team_model` | Home-vs-away log-odds from the second model instead of the first | on | on / off (section 5.9) |
| `tm_refine_team_s` | Search window for moving a team→team boundary to the first touch of the new team, when the second model is used | 0 (off) | 0, 1, 2 s on the tuning matches: 0 is best against every label definition tried (T = 0 to 3 s) |
| Training label | Shortest team spell between two spells of the other team that counts as a possession | 2 s | The organiser's "a player keeps the ball for a few seconds"; not fitted. Trained with 0 s the model is 0.010 better against the field as it is and 0.013 worse at T2 |
| Trees | 300 trees of at most 31 leaves; other settings as the first model | – | 63 leaves: no gain; 600 trees: +0.007 |

`dm_cost` and `dm_bias` were chosen on the five tuning matches on 15 Nov, with the hand-weighted evidence as the rest of the model, and were not re-chosen when the learned evidence replaced it (section 5.8). With the detector in use, `learned_dead_scale` acts only in the fallback, and `c_die` and `c_restart` have no practical effect, because the detector's stoppages are hard evidence.

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
| The same models, with stoppages from the detector, leave-one-match-out (section 5.8) | 0.555 / 0.600 | 0.896 / 0.905 | J03WQQ 0.618, J03WR9 0.582 |
| **Final pipeline:** the detector and the second home-vs-away model, leave-one-match-out (section 5.9) | 0.581 / **0.627** | 0.905 / **0.914** | J03WQQ 0.620, J03WR9 0.634 |

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

All parameters and both trained models were chosen on Sportec data only, as described in sections 5.4–5.6: the hand-weighted evidence with the grid in `external_validation/grids/next_hmm.json`, and the learned evidence and its decoder settings with `external_validation/train_emissions.py`, under an adoption rule fixed beforehand (better on the tuning matches and on both held-out matches). The stoppage detector and the second home-vs-away model were added on 16 Nov on the evidence of sections 5.8 and 5.9 alone, and each meets the same adoption rule. They were built and validated on Sportec and synthetic data, without access to the competition match or to any score on it. [SJEKK: Oskar and Martin – the pipeline without the detector had by then been run on the competition match. Confirm that nothing from that run, and no comparison of the versions on the public 5 minutes, played any part in the decision to add the detector and the second model, then delete this tag.] The submitted file is the one produced by the commit tagged `phase1-submission`.

### 5.8 The stoppage detector: training and validation

**Training data.** The seven Sportec matches of section 5.1, 853 338 frames after subsampling. The label is Sportec's own per-frame ball status; we labelled nothing. No frame of the competition match, of its public ground truth or of its video is involved. Code and details: `dead_training/`.

**Making the training data look like automatic tracking.** Each match is copied twice with different seeds. In every copy the ball is removed on roughly 30 to 50 % of the live frames (more in crowds and when it is in the air), player tracks get gaps and correlated position noise, and false ball detections are inserted. Sportec's frozen stoppage ball is a placeholder, not an observation, so it is removed, and each stoppage gets one of several pictures at random: no ball, a ball that rolls on for a moment, a ball lying at the restart point, a ball moved to the restart point, or a ball with the player who takes the restart. The detector therefore cannot lean on any single one of them.

**Protocol.** Leave-one-match-out: every match is scored with a detector trained on the other six, so each number below is on a match the detector has not seen. The two tree models of section 1.2 are handled the same way: on the five tuning matches each match is scored with models refitted on the other four, and on the two held-out matches with the stored models. The detector's two decoding values (`dm_cost`, `dm_bias`) were chosen on the five tuning matches on 15 Nov, as the pair with the highest mean spellF1 over the five ball pictures below (24 pairs compared on the stoppages alone, six of them end to end), at that time with the hand-weighted evidence as the rest of the model. They were not re-chosen for the learned evidence. The stored detector is trained on all seven matches with the same settings. Everything in this section comes from `python dead_training/stoppage_pictures.py`.

**Result.** The live frames are left exactly as section 5.3 builds them, and the same 41 % of ball frames are missing; only the ball during stoppages differs between the rows. Mean spellF1 of the whole pipeline over all seven matches:

| Ball during stoppages | Without the detector | With the detector (final) | True stoppages, same team decoding |
|---|---|---|---|
| Frozen where play stopped (validation as built) | 0.556 | 0.568 | 0.600 |
| Lying at the restart point | 0.529 | 0.561 | 0.595 |
| Not visible | 0.479 | 0.561 | 0.596 |
| With the player who takes the restart | 0.394 | 0.538 | 0.591 |
| A random mix per stoppage: not visible, lying at or moved to the restart point, or with the taker | 0.478 | 0.558 | 0.596 |

On the validation data as built the detector scores 0.555 against 0.545 on the tuning matches and 0.600 against 0.584 on the held-out ones (macroF1 0.898 against 0.892 over all seven). Per match, without → with: J03WPY 0.579 → 0.587, J03WMX 0.517 → 0.535, J03WN1 0.539 → 0.547, J03WOH 0.579 → 0.579, J03WOY 0.510 → 0.526, J03WQQ 0.595 → 0.618, J03WR9 0.573 → 0.582. Across the 35 match-and-picture combinations the detector is ahead on 34 and level on one. The pictures other than the first are our own constructions, and the detector was trained with pictures of the same kind, so the table shows robustness to them, not to everything an automatic tracker might do. [SJEKK: Oskar – the numbers in this section were computed on a rebuild of the validation data from the same matches and seeds, with positions rounded to 1 cm; the model without the detector scores 0.545 and 0.584 there, against 0.546 and 0.582 in section 5.6. Rerun `dead_training/stoppage_pictures.py` on our own `ext_data/` and replace the numbers if they differ.]

**Synthetic matches** (not used for any choice, section 2.3). On five matches from `tools/make_fake_match.py` the pipeline scores 0.79 with the detector and 0.77 without it; the detector is ahead on four and behind on one.

**Run-time implementation.** `src/dead_model.py` is a numpy-only rewrite of the training-time feature code, and it walks the stored trees itself. On a full match its log-odds are identical to scikit-learn's (zero difference on 142 345 frames), and `dead_training/check_features.py` checks the features against the training-time code. The run-time arithmetic is limited to operations that IEEE 754 rounds exactly (section 6).

### 5.9 The second home-vs-away model: training and validation

**What Sportec's ownership field means.** At the frame where ownership changes, a player of the new team is within 1 m of the ball in 85 % of cases and within 2 m in 99.8 % (1 771 changes in the five tuning matches, clean data). The field changes whenever the other team touches the ball. The organiser counts only controlled possession: a simple touch, a deflection or an aerial duel without control is not a change (CHALLENGE.md). We therefore score against two versions of the labels: the field as it is (**T0**), and with a team spell shorter than 2 s between two spells of the other team not counted (**T2**; `controlled_owner` in `external_validation/build_dataset.py`). T2 has 332 possessions per match with a mean of 9.3 s, T0 458 with a mean of 6.7 s. The 2 s is the organiser's "a player keeps the ball for a few seconds". It was not fitted to anything, and nothing here uses the competition match or its ground truth. `python team_training/ownership.py` prints these numbers.

**Training.** The five tuning matches, each in three pictures of the live ball (`team_training/pictures.py`): as built; with the ball not seen while it is more than 1.5 m above the ground; and with position noise of 0.25 m on the ball and on the players. The validation data as built removes the ball regardless of its height and adds no noise, while the organiser says the ball is lost in the air (DATA.md), and in the open matches it is above 1.5 m in a fifth of live play. Every tenth live frame of each picture, in both orientations: 230 070 rows. Label: T2. `python team_training/train.py`.

**Protocol.** As in sections 5.4 and 5.8. On the five tuning matches every model is leave-one-match-out: the second model of a match is trained on the other four, the first models are refitted on the other four, the detector on the other six. On the two held-out matches the stored first and second models are used, which have not seen them. Every choice about the second model (its feature groups, the size of the trees, the T2 label, the three training pictures, leaving team→team boundaries where the model puts them) was made on the tuning matches with the true stoppages given. The held-out matches were first scored with the finished model. Everything below comes from `python team_training/compare.py` (stoppage pictures) and `python team_training/compare.py --live`.

**Result.** Mean spellF1 of the whole pipeline over all seven matches. "Tagged" is the model of section 5.6, "detector" adds the stoppage detector, "final" adds the second home-vs-away model. First with the ball shown differently during stoppages (section 5.8):

| Ball | T0: tagged | T0: detector | T0: final | T2: tagged | T2: detector | T2: final |
|---|---|---|---|---|---|---|
| Frozen where play stopped (validation as built) | 0.556 | 0.568 | 0.594 | 0.638 | 0.649 | 0.682 |
| Lying at the restart point | 0.529 | 0.561 | 0.592 | 0.605 | 0.642 | 0.680 |
| Not visible | 0.479 | 0.561 | 0.588 | 0.548 | 0.641 | 0.677 |
| With the player who takes the restart | 0.394 | 0.538 | 0.570 | 0.445 | 0.616 | 0.658 |
| A random mix per stoppage | 0.478 | 0.558 | 0.590 | 0.544 | 0.638 | 0.679 |

On the validation data as built the final pipeline scores 0.581 against 0.545 on the tuning matches and 0.627 against 0.584 on the held-out ones at T0, and 0.680 against 0.635 and 0.686 against 0.646 at T2 (macroF1 at T0 over all seven: 0.907 against 0.892). Per match at T0, tagged → final: J03WPY 0.579 → 0.590, J03WMX 0.517 → 0.561, J03WN1 0.539 → 0.544, J03WOH 0.579 → 0.619, J03WOY 0.510 → 0.589, J03WQQ 0.595 → 0.620, J03WR9 0.573 → 0.634. With the true stoppages the final team decoding reaches 0.631 (T0) and 0.723 (T2).

The second model's part of the gain is uneven. Against the detector version it adds +0.027 to +0.064 at T0 on J03WMX, J03WOH, J03WOY, J03WR9, and between -0.003 and +0.003 on J03WPY, J03WN1, J03WQQ; at T2 it is -0.014 on J03WQQ. Seven matches are few, and one match moves by about 0.04 from noise alone (section 5.4), so the mean (+0.026 at T0, +0.033 at T2) is the number to read.

Then with the ball missing differently during live play (stoppage frames as built):

| Ball | T0: tagged | T0: detector | T0: final | T2: tagged | T2: detector | T2: final |
|---|---|---|---|---|---|---|
| As built: blocks of 0.5 to 5 s, regardless of height | 0.556 | 0.568 | 0.594 | 0.638 | 0.649 | 0.682 |
| Not seen above 1.5 m | 0.519 | 0.534 | 0.561 | 0.618 | 0.633 | 0.659 |
| As built, with 0.25 m position noise | 0.472 | 0.539 | 0.572 | 0.539 | 0.617 | 0.651 |
| Not seen above 1.5 m, with 0.25 m noise | 0.443 | 0.520 | 0.545 | 0.519 | 0.611 | 0.642 |
| As built, with 0.5 m position noise (never used in training) | 0.440 | 0.423 | 0.444 | 0.501 | 0.480 | 0.505 |

The last row is a test beyond anything used in training: at 0.5 m of noise every version falls to about 0.44, the detector version is behind the tagged one (0.423 against 0.440) and the final pipeline is level with it (0.444). With the true stoppages the team decoding still reaches 0.597, so at that noise level the loss is in the dead-ball part, in every version.

Across the 63 combinations of match and picture in the two tables, the final pipeline is ahead of the tagged one on 60 at T0 and 60 at T2, and ahead of the detector version on 58 and 59. Leaving the 0.5 m row aside, its mean stays between 0.545 and 0.594 at T0; the tagged pipeline falls to 0.394. The pictures other than the first of each table are our own constructions, and the detector and the second model were trained with pictures of the same kind (the second model with the first three of the live ones), so the tables show robustness to them, not to everything an automatic tracker might do. [SJEKK: Oskar – as in section 5.8, these numbers were computed on a rebuild of the validation data with positions rounded to 1 cm. Rerun `team_training/compare.py` on our own `ext_data/` and replace them if they differ.]

**Synthetic matches** (not used for any choice, section 2.3). On five matches from `tools/make_fake_match.py` the final pipeline scores 0.78, the detector version 0.79 and the tagged one 0.77; the second model is ahead of the detector version on two matches and behind on three. The hand-weighted model of 15 Nov scores 0.87 there. The synthetic matches are generated by simple rules, which favours hand-set rules over anything learned from real football, but they are the one test we have on data that is not Sportec, and on it the second model shows no gain.

**What the organiser publishes about the ground truth.** DATA.md states that the public five minutes hold 17 possessions with a mean of 14.2 s. We read that as consistent with a definition that counts fewer and longer possessions than Sportec's field does (mean 6.7 s). We did not fit the 2 s, or anything else, to it, and we did not open the public ground truth for this. [SJEKK: Oskar and Martin – keep or delete this paragraph; it is here so that the document says everything that informed the choice of T2.]

**Run-time implementation.** `src/team_model.py` builds its features from the arrays the pipeline already has, and the training script uses the same functions, so training and run time cannot drift apart. The trees are stored as flat arrays and walked by the code of section 5.8; `train.py` checks them against LightGBM on 5 000 rows (largest difference below 1e-9). The arithmetic added by this module is limited to operations that IEEE 754 rounds exactly.

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

Expected runtime: about 70 seconds on a laptop, without cache [SJEKK: Oskar/Martin – time of the run with the detector and the second model; it was about 15 seconds without them]. The output is deterministic: there are no random numbers and nothing is tuned at run time. Expected SHA-256 of `NEXT.csv`: [SJEKK: Oskar/Martin – SHA-256 of the file produced with the detector] (macOS, Python 3.13.5).

The run prints which dead-ball evidence was used (`dead ball from: trained detector`, or the state model's own evidence if the detector's output failed its plausibility check, section 1.3), which home-vs-away model was used (`home/away from: second model`, or the first model if the second failed) and counts of its own output labels. It prints no tracking data. The detector's arithmetic is limited to operations that IEEE 754 rounds exactly (+, −, ×, ÷, square root, comparisons), so it should not depend on the machine. [SJEKK: Oskar/Martin – say where the file with the detector was verified, as in the Requirements line above] Switches give the earlier models back: `--tagged` is the model first tagged `phase1-submission`, without the stoppage detector and the second home-vs-away model (SHA-256 `f12d15ce52ebeba71c6fa5c245e19bf061c4360a4b5e3ddebea4214a1d63ae6f`); `--no-detector` and `--no-team-model` switch off one of them; `--hand-weighted` is the model before the learned evidence.

**Reproducing the external validation (optional, open data only):**

```bash
python -m pip install -r external_validation/requirements-validation.txt
python external_validation/build_dataset.py --selftest        # ~1 s, synthetic data, checks the setup
python external_validation/build_dataset.py --matches all     # downloads the 7 Sportec matches via kloppy
python external_validation/sweep.py external_validation/grids/next_hmm.json --matches J03WPY J03WMX J03WN1 J03WOH J03WOY
python external_validation/train_emissions.py --jobs 6    # trains the learned evidence, searches the decoder, writes src/learned_model.json
```

The `sweep.py` command reproduces the grid behind the hand-weighted model (32 combinations, a few minutes); the last command reproduces the learned evidence and its scores (about two minutes).

If the kloppy download times out (the tracking files are about 370 MB each), re-run it; the cache keeps finished files.

**Retraining the stoppage detector and the second home-vs-away model, and reproducing sections 5.8 and 5.9 (optional, open data only):** see `dead_training/README.md` and `team_training/README.md`. The submission is reproduced from the stored files `models/dead_trees.npz` and `models/team_trees.npz`, not from a retraining.

The Sportec files are downloaded and cached by kloppy outside the repository; generated files are written to `ext_data/`, which is git-ignored. Loading one match needs a few GB of RAM.

---

## 7. External data, models and tools

Required by the rules (R7): everything external that we used, with licence and credit. Nothing private.

**Datasets**

| Dataset | Use | Licence | Credit |
|---|---|---|---|
| Sportec/IDSSE open tracking data, Bundesliga 2022/23 (match IDs J03WPY, J03WMX, J03WN1, J03WOH, J03WOY, J03WQQ, J03WR9) | Training, tuning and validation (section 5) | CC-BY 4.0 | Bassek, M., Rein, R., Weber, H. et al. (2025). *Scientific Data* 12, 195. doi:10.1038/s41597-025-04505-y |

**Pre-trained models:** No third-party pre-trained model is used. Our own two tree models, trained by us on the Sportec matches above, are stored in the repository as `src/learned_model.json` (about 0.3 MB) and can be re-created with `python external_validation/train_emissions.py`. The stoppage detector, also trained by us on those matches, is stored as `models/dead_trees.npz` (about 0.4 MB); the code that made it is in `dead_training/`. The second home-vs-away model is stored as `models/team_trees.npz` (about 0.2 MB); the code that made it is in `team_training/`. The small logistic regressions of section 2.5 are fitted at run time and not stored.

**Software libraries**

| Library | Version | Licence | Use |
|---|---|---|---|
| LightGBM | 4.7.0 | MIT | Training the two tree models (`external_validation/train_emissions.py`) and the second home-vs-away model (`team_training/train.py`) on Sportec data. **Not** needed to produce the submission: the trained trees are stored as arrays and evaluated with numpy |
| scikit-learn, scipy, joblib | 1.9.1, 1.18.1, 1.6.0 | BSD-3-Clause | Training the stoppage detector on Sportec data (`dead_training/`). **Not** needed to produce the submission: the trained trees are stored as arrays and evaluated with numpy |
| kloppy | 3.19.1 | BSD-3-Clause | Loading Sportec data in the external validation only. The competition pipeline does not use it |
| numpy | 2.5.4 | BSD-3-Clause | The whole competition pipeline (`src/`): parsing, features, the stored tree ensembles, logistic regression, Viterbi. No scipy, scikit-learn or HMM library is used at run time |
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
