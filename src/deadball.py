"""Per-frame dead-ball evidence: log p(obs | dead) - log p(obs | alive), length T.

Positive means "looks like a stoppage". It is evidence for the state model in possession.py, not
a decision: a frame with nothing to say (ball missing, players at ordinary speed) scores 0 and the
state model keeps whatever it had.

Sources, strongest first:
  * graded frames before the half's kick-off (Smart Tagging kick-off tag + formation + ball)
  * ball seen outside the pitch
  * both teams entirely in their own halves for a couple of seconds (kick-off after a goal)
  * ball lying still / ball moving fast inside the pitch
  * the seconds before a tagged set piece, and the span between a tagged goal and its kick-off
  * player speeds, calibrated without labels: a logistic model is self-trained on the frames
    the sources above are sure about and applied everywhere (Borghesi et al. 2023 found player
    velocities to carry most of the dead/alive signal beyond the ball itself)
"""

import numpy as np

from .config import Config
from .selftrain import selftrain_llr

KICKOFF_START = "kick off (start)"
KICKOFF_GOAL = "kick off (after goal)"


def _in_blocks(n, blocks):
    mask = np.zeros(n, dtype=bool)
    for first, last in blocks:
        mask[first:last + 1] = True
    return mask


def kickoff_frame(feats, tags, first, last, cfg):
    """First live frame of the half that is graded from `first`, or `first` if unknown.

    The kick-off tag gives a window of a few seconds around the restart. Inside it the kick is
    pinned by the formation breaking (players may only cross the halfway line once the ball is
    played) and, where the ball is seen, by the ball leaving the centre spot.
    """
    reach = cfg.frames(360)
    cands = [t for t in tags
             if t["code"] == "set pieces" and t["tags"].get("type") == KICKOFF_START
             and t["end"] > first - cfg.frames(30) and t["start"] < min(last, first + reach)]
    if not cands:
        return first
    tag = min(cands, key=lambda t: t["start"])
    lo = max(first, tag["start"])
    hi = min(last + 1, max(tag["end"], lo + 1) + cfg.frames(2))
    form = feats.kick_raw[lo:hi]
    if not form.any():
        return lo
    start = int(np.argmax(form))
    # first frame after `start` from which the formation stays broken for 5 frames
    broken = ~form
    run = np.convolve(broken.astype(int), np.ones(5, dtype=int))[4:]   # sum over [i, i+5)
    after = np.flatnonzero((run >= 5) & (np.arange(len(form)) > start))
    stop = int(after[0]) if len(after) else len(form)
    centre = np.flatnonzero(feats.ball_center[lo + start:lo + stop])
    if len(centre):
        return lo + start + int(centre[-1]) + 1
    return lo + stop


def compute(feats, tags, blocks, cfg=None):
    """dead_llr [T] and a dict of the pseudo-label masks used (for tests and diagnostics)."""
    cfg = cfg or Config()
    n = feats.n
    graded = _in_blocks(n, blocks)
    llr = np.zeros(n)

    obs, ok = feats.ball_obs, feats.ball_ok
    speed = np.where(np.isfinite(feats.ball_speed), feats.ball_speed, np.nan)

    out = obs & (feats.oob > cfg.oob_margin)
    llr[out] += cfg.w_oob

    still = feats.ball_still
    llr[still & ~out] += cfg.w_still

    fast = obs & (np.nan_to_num(speed, nan=0.0) > cfg.v_fast) \
        & (np.nan_to_num(feats.inside, nan=-1.0) > 1.0)
    llr[fast] -= cfg.w_fast

    llr[~ok & (feats.gap_len > cfg.long_gap)] += cfg.w_missing
    llr[feats.kick_form] += cfg.w_kickform

    # Smart Tagging: restarts and goals
    prekick = np.zeros(n, dtype=bool)
    for first, last in blocks:
        prekick[first:kickoff_frame(feats, tags, first, last, cfg)] = True
    pre = cfg.frames(cfg.setpiece_pre_s)
    kickoffs = sorted(t["start"] for t in tags
                      if t["code"] == "set pieces" and t["tags"].get("type") == KICKOFF_GOAL)
    for t in tags:
        if t["code"] == "set pieces" and t["tags"].get("type") != KICKOFF_START:
            llr[max(0, t["start"] - pre):max(0, t["start"])] += cfg.w_setpiece
        elif t["code"] == "goals":
            nxt = [k for k in kickoffs if t["start"] <= k <= t["end"] + cfg.frames(180)]
            if nxt and nxt[0] > t["end"]:
                llr[max(0, t["end"]):nxt[0]] += cfg.w_goal

    # player speeds
    hand = cfg.w_players * np.clip(
        (cfg.v0_players - np.nan_to_num(feats.med_speed, nan=cfg.v0_players)) / cfg.s_players,
        -1.5, 1.5)
    dead_anchor = graded & (out | feats.kick_form | prekick)
    alive_anchor = graded & ~dead_anchor & ~still & obs \
        & (np.nan_to_num(speed, nan=0.0) > 5.0) & (np.nan_to_num(feats.inside, nan=-1.0) > 2.0)
    players = hand
    if cfg.dead_selftrain:
        x = np.column_stack([feats.med_speed, feats.frac_slow])
        z = selftrain_llr(x, dead_anchor | alive_anchor, dead_anchor.astype(float),
                          cfg.dead_min_frames, stride=5)
        if z is not None:
            players = np.clip(cfg.dead_shrink * z, -cfg.dead_clip, cfg.dead_clip)
    llr += players

    llr[prekick] += cfg.w_prekick
    llr[~graded] = 0.0
    return llr, {"dead_anchor": dead_anchor, "alive_anchor": alive_anchor, "prekick": prekick}
