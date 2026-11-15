"""Every tunable number of the pipeline, in one frozen dataclass.

The defaults start from the literature (Link & Hoernig 2017, Vidal-Codina et al. 2022, Morra et
al. 2020) and football priors. Four of them (use_hint, c_switch, touch_bonus, dead_clip) were then
chosen on five open Sportec/IDSSE matches with external_validation/sweep.py and the grid in
external_validation/grids/next_hmm.json; two more matches were held out as a check. Nothing is
tuned on the provided match or its public ground truth (RULES.md; docs/organizer_answers.md R9).

On 15 Nov the hand-weighted evidence was replaced as decoder input by two gradient-boosted
models trained on the same five matches (learned.py, external_validation/train_emissions.py); the
hand-weighted evidence is now one of their inputs. The decoder scales and transition costs were
chosen leave-one-match-out on those five matches and checked on the two held-out ones.

Evidence weights are log-likelihood ratios per frame. Transition costs are negative log
probabilities per frame, so `c_switch = 6` means "a possession change is a 1-in-400-frames event".
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    fps: int = 25

    # --- ball cleaning (features.py) -----------------------------------------------------------
    ball_vmax: float = 45.0        # m/s between two detections: faster = a tracker jump
    ball_min_seg: int = 5          # detections; shorter jump-bounded segments are dropped as blips
    ball_max_gap: int = 25         # frames; shorter ball gaps are linearly interpolated

    # --- ball-based team evidence (possession.py) ----------------------------------------------
    r_ctrl: float = 2.0            # m, radius of a player's possession zone (Vidal-Codina: 0.5-1 m
    s_ctrl: float = 0.5            #    on manual-grade tracking; wider here for automatic tracking)
    s_gap: float = 1.0             # m, scale of the home/away distance gap
    v_ctrl: float = 10.0           # m/s, a ball faster than this is passing by, not controlled
    s_v: float = 2.0
    beta: float = 0.5              # max |llr| per frame from proximity
    interp_weight: float = 0.5     # weight of frames where the ball is interpolated
    touch_dv: float = 3.0          # m/s change of ball velocity that counts as a touch (Link: kick)
    touch_r: float = 2.5           # m, the toucher must be this close
    touch_bonus: float = 0.6       # llr per frame, over +/- touch_half frames around the touch
    touch_half: int = 2

    # --- ball-free hint, self-trained on confident ball frames ---------------------------------
    use_hint: bool = True          # +0.09 spellF1 on the Sportec tuning matches
    hint_conf: float = 0.3         # |ball llr| needed for a frame to become a pseudo-label
    hint_stride: int = 5           # subsampling of pseudo-labels (neighbouring frames are copies)
    hint_min_frames: int = 400     # per class and half, else the hint is off for that half
    hint_shrink: float = 0.15      # frames are heavily autocorrelated: do not count each in full
    hint_clip: float = 0.3
    hint_on_ball: float = 0.3      # weight of the hint on frames that do have a ball
    hint_rich: bool = False        # more team-shape features and longer lags
    hint_rounds: int = 0           # extra rounds retrained on the decoder's own output
    dead_iter: bool = False        # in those rounds, also retrain a ball-free dead/alive hint
    dead_iter_shrink: float = 0.1
    dead_iter_clip: float = 0.3

    # --- Smart Tagging team prior, polarity calibrated against the ball evidence ---------------
    use_tags: bool = True
    tag_gamma: float = 0.03        # a 10 s phase alone must not be able to create a possession
    tag_min_agree: float = 0.2
    tag_min_frames: int = 250

    # --- dead ball evidence (deadball.py) ------------------------------------------------------
    oob_margin: float = 0.75       # m outside the lines before the ball counts as out
    w_oob: float = 1.5
    still_spread: float = 0.35     # m, positional spread of a ball that is lying still
    w_still: float = 0.4
    v_fast: float = 3.0
    w_fast: float = 0.3
    long_gap: int = 75             # frames without a ball before absence itself hints at dead
    w_missing: float = 0.1
    w_kickform: float = 1.0        # both teams entirely in their own halves
    w_setpiece: float = 0.3        # the seconds before a tagged restart
    setpiece_pre_s: float = 3.0
    w_goal: float = 0.5            # between a tagged goal and the kick-off that follows it
    w_prekick: float = 8.0         # graded frames before the half's kick-off
    dead_selftrain: bool = True
    dead_min_frames: int = 300
    dead_shrink: float = 0.25
    dead_clip: float = 0.08        # slow players alone need several seconds to make a stoppage
    v0_players: float = 1.6        # m/s, fallback: median player speed at the dead/alive boundary
    s_players: float = 0.5
    w_players: float = 0.1

    # --- learned evidence (learned.py), trained on open Sportec matches ------------------------
    # Leave-one-match-out on the five tuning matches: spellF1 0.546 against 0.501 for the
    # hand-weighted evidence; on the two held-out matches 0.582 against 0.514
    # (external_validation/train_emissions.py). learned=False gives the earlier model back.
    learned: bool = True           # use the trained models instead of the hand-weighted evidence
    learned_team_scale: float = 0.4    # neighbouring frames are near-copies: count each in part
    learned_dead_scale: float = 0.4
    learned_clip: float = 6.0      # cap on the log-odds of a single frame

    # --- state model ---------------------------------------------------------------------------
    # Chosen with the learned evidence (train_emissions.py). With learned=False the values
    # tuned for the hand-weighted evidence were 3.0 / 6.5 / 6.0 (grids/next_hmm.json).
    c_switch: float = 2.0          # H <-> A
    c_die: float = 10.0            # H/A -> D
    c_restart: float = 10.0        # D -> H/A
    min_poss_s: float = 1.0        # a shorter spell between two spells of the other team is a touch
    min_edge_s: float = 0.4        # a shorter spell next to a stoppage is noise
    min_dead_s: float = 2.0

    # --- restart possessor: after a stoppage the taker's team has the ball until the other controls
    use_restart: bool = True
    restart_look_s: float = 2.0    # how far back into the stoppage to look for the taker
    restart_min_frames: int = 10   # frames the taker's team must be on the ball
    restart_max_s: float = 6.0     # the taker's spell must end within this, else leave it alone

    # --- boundary refinement -------------------------------------------------------------------
    refine: bool = True
    refine_team_s: float = 2.0     # search window for the first touch of the new team
    refine_dead_s: float = 1.0     # search window for ball-out and restart snaps

    def hand_weighted(self):
        """The model as it was before the learned evidence: hand-weighted evidence and the
        transition costs tuned for it (external_validation/grids/next_hmm.json)."""
        return replace(self, learned=False, c_switch=3.0, c_die=6.5, c_restart=6.0)

    def with_(self, **changes):
        return replace(self, **changes)

    def frames(self, seconds):
        return max(1, int(round(seconds * self.fps)))
