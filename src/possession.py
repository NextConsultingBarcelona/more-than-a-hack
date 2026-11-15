"""H / A / D state model: evidence -> Viterbi -> duration clean-up -> boundary refinement.

The ranking metric matches whole possessions by temporal IoU, so the model is built around
possessions as intervals, not frames:

1. Team evidence (log-likelihood ratio home vs away per frame)
   * ball proximity: the team of the nearest player, weighted by how close he is and by the ball
     being slow enough to be under control (possession zone of Vidal-Codina et al. 2022, slow-ball
     condition of Morra et al. 2020). A ball in flight or a ball that is missing says nothing.
   * touches: a change of ball velocity next to a player (kick detection, Link & Hoernig 2017).
   * a ball-free hint from team shape and movement, self-trained per half on the frames where
     the ball evidence is confident (PathCRF, Kim et al. 2026: player trajectories alone carry
     possession; Borghesi et al. 2023: team membership + positions are what the team label needs).
   * Smart Tagging phases, with the meaning of each code's Team tag learned from its agreement
     with the ball evidence rather than assumed.
2. A three-state HMM decoded with Viterbi. "Nothing to say" frames keep the current state, which
   is exactly the labelling rule: a loose or contested ball stays with the last team. The switch
   cost is what makes a single touch or deflection not a change of possession.
3. Clean-up of spells shorter than a controlled possession, then each boundary is snapped to the
   event it stands for: the first touch of the new team, the ball crossing the line, the restart.

On graded frames the output never abstains: for spellF1 an X is a forward-fill of the previous
label, so it can never beat writing that label, and it costs macroF1.
"""

from difflib import SequenceMatcher

import numpy as np

from .config import Config
from .selftrain import selftrain_llr

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ------------------------------------------------------------------------------------ evidence
def ball_evidence(feats, cfg):
    """(llr home-vs-away [T], control masks {H: bool[T], A: bool[T]}) from ball and players."""
    cap = 50.0
    d_h = np.minimum(feats.d_home, cap)
    d_a = np.minimum(feats.d_away, cap)
    ok = feats.ball_ok & (np.maximum(d_h, d_a) < cap)
    gap = d_a - d_h                                     # > 0: home is closer
    near = np.minimum(d_h, d_a)
    speed = np.where(np.isfinite(feats.ball_speed), feats.ball_speed, 0.0)

    close = _sigmoid((cfg.r_ctrl - near) / cfg.s_ctrl)
    slow = _sigmoid((cfg.v_ctrl - speed) / cfg.s_v)
    llr = cfg.beta * np.tanh(gap / cfg.s_gap) * close * slow
    llr = np.where(feats.ball_obs, llr, llr * cfg.interp_weight)

    # touches: local maxima of the velocity change, next to a player
    dv = np.where(np.isfinite(feats.ball_dv), feats.ball_dv, -1.0)
    peak = dv >= cfg.touch_dv
    for k in (1, 2):
        peak &= (dv >= np.roll(dv, k)) & (dv >= np.roll(dv, -k))
    peak &= ok & (near <= cfg.touch_r) & (np.abs(gap) > 0.3)
    width = 2 * cfg.touch_half + 1
    touch = np.convolve(np.where(peak, np.sign(gap), 0.0), np.ones(width), mode="same")
    llr = llr + cfg.touch_bonus * np.clip(touch, -1.0, 1.0)
    llr = np.where(ok, llr, 0.0)

    under_control = ok & (speed < cfg.v_ctrl)
    ctrl = {H: under_control & (d_h <= cfg.r_ctrl) & (d_h + 0.3 < d_a),
            A: under_control & (d_a <= cfg.r_ctrl) & (d_a + 0.3 < d_h)}
    return llr, ctrl


def side_of(name, names):
    """+1 if a Smart Tagging team name is the home side, -1 away, 0 if it is neither (N/A)."""
    name = (name or "").strip().lower()
    if name in ("", "n/a", "na", "none"):
        return 0
    ratio = [SequenceMatcher(None, name, (n or "").strip().lower()).ratio() for n in names]
    if max(ratio) < 0.5 or abs(ratio[0] - ratio[1]) < 0.1:
        return 0
    return 1 if ratio[0] > ratio[1] else -1


def tag_evidence(tags, names, ball_llr, graded, cfg):
    """Team prior from Smart Tagging phases [T], and the learned polarity of each code.

    Whether a code's Team tag names the team with the ball ("build up") or the team without it
    ("defending in middle third") is not assumed: it is measured as the agreement between the
    tag and the confident ball evidence, and codes that do not agree either way get weight 0.
    """
    n = len(ball_llr)
    signed = {}
    for t in tags:
        side = side_of(t["tags"].get("team"), names)
        if side == 0 or t["end"] <= t["start"]:
            continue
        arr = signed.setdefault(t["code"], np.zeros(n))
        arr[max(0, t["start"]):min(n, t["end"])] += side
    conf = graded & (np.abs(ball_llr) >= cfg.hint_conf)
    ref = np.sign(ball_llr)
    total = np.zeros(n)
    polarity = {}
    for code in sorted(signed):
        s = np.clip(signed[code], -1.0, 1.0)
        m = conf & (s != 0)
        count = int(m.sum())
        agree = float((s[m] * ref[m]).mean()) if count else 0.0
        weight = agree if (count >= cfg.tag_min_frames and abs(agree) >= cfg.tag_min_agree) else 0.0
        polarity[code] = weight
        total += weight * s
    return cfg.tag_gamma * np.clip(total, -1.5, 1.5), polarity


def hint_evidence(feats, ball_llr, dead_llr, blocks, cfg):
    """Ball-free team hint [T], self-trained per half on confident ball frames."""
    hint = np.zeros(feats.n)
    for first, last in blocks:
        sl = slice(first, last + 1)
        conf = feats.ball_obs[sl] & (np.abs(ball_llr[sl]) >= cfg.hint_conf) & (dead_llr[sl] < 0.2)
        z = selftrain_llr(feats.team_feats[sl], conf, (ball_llr[sl] > 0).astype(float),
                          cfg.hint_min_frames, stride=cfg.hint_stride)
        if z is None:
            continue
        z = np.clip(cfg.hint_shrink * z, -cfg.hint_clip, cfg.hint_clip)
        hint[sl] = np.where(feats.ball_ok[sl], cfg.hint_on_ball * z, z)
    return hint


# ------------------------------------------------------------------------------------- decoder
def viterbi(e_h, e_a, e_d, c_switch, c_die, c_restart):
    """Most likely H/A/D path for per-frame scores and constant transition costs. int8 [T].

    A plain Python loop on purpose: three states and scalar floats run faster than per-frame
    numpy calls (well under a second per half), and exact ties resolve to "stay".
    """
    n = len(e_h)
    if n == 0:
        return np.zeros(0, dtype=np.int8)
    eh, ea, ed = np.asarray(e_h).tolist(), np.asarray(e_a).tolist(), np.asarray(e_d).tolist()
    bh, ba, bd = bytearray(n), bytearray(n), bytearray(n)
    h, a, d = eh[0], ea[0], ed[0]
    for t in range(1, n):
        vh, ph = h, 0
        x = a - c_switch
        if x > vh:
            vh, ph = x, 1
        x = d - c_restart
        if x > vh:
            vh, ph = x, 2
        va, pa = a, 1
        x = h - c_switch
        if x > va:
            va, pa = x, 0
        x = d - c_restart
        if x > va:
            va, pa = x, 2
        vd, pd = d, 2
        x = h - c_die
        if x > vd:
            vd, pd = x, 0
        x = a - c_die
        if x > vd:
            vd, pd = x, 1
        bh[t], ba[t], bd[t] = ph, pa, pd
        h, a, d = vh + eh[t], va + ea[t], vd + ed[t]
    back = (bh, ba, bd)
    state = 0 if (h >= a and h >= d) else (1 if a >= d else 2)
    path = bytearray(n)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[state][t]
    return np.frombuffer(bytes(path), dtype=np.uint8).astype(np.int8)


def runs_of(states):
    """(starts, ends_exclusive, values) of the maximal constant runs."""
    if len(states) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, states.dtype)
    change = np.flatnonzero(states[1:] != states[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(states)]))
    return starts, ends, states[starts]


def enforce_durations(states, cfg):
    """Remove spells too short to be a controlled possession or a real stoppage.

    Shortest offender first, until none is left:
      * a stoppage shorter than min_dead_s is a tracking blip: play continued (previous team)
      * a team spell shorter than min_poss_s between two spells of the same other label is a
        touch, not a possession: it takes the surrounding label
      * a team spell shorter than min_edge_s anywhere else joins its live neighbour
    """
    s = states.copy()
    min_dead, min_poss, min_edge = (cfg.frames(cfg.min_dead_s), cfg.frames(cfg.min_poss_s),
                                    cfg.frames(cfg.min_edge_s))
    while True:
        starts, ends, vals = runs_of(s)
        m = len(vals)
        if m <= 1:
            return s
        best, fill = None, None
        for i in range(m):
            length = ends[i] - starts[i]
            prev = vals[i - 1] if i > 0 else None
            nxt = vals[i + 1] if i < m - 1 else None
            target = None
            if vals[i] == D:
                if length < min_dead:
                    target = prev if prev is not None else nxt
            elif prev is not None and nxt is not None and prev == nxt:
                if length < min_poss:
                    target = prev
            elif length < min_edge:
                live = [v for v in (prev, nxt) if v is not None and v != D]
                others = [v for v in (prev, nxt) if v is not None]
                target = live[0] if live else others[0]
            if target is not None and (best is None or length < ends[best] - starts[best]):
                best, fill = i, target
        if best is None:
            return s
        s[starts[best]:ends[best]] = fill


def refine_boundaries(states, ctrl, feats, offset, cfg):
    """Snap every boundary to the event it stands for. `offset` is the block's first frame.

      team -> team   the first frame the new team controls the ball after the old team's last
                     control: a pass in flight still belongs to the passer
      live -> dead   the first frame the ball is seen outside the pitch
      dead -> live   back over a fast in-play ball just before the boundary, forward over a
                     ball still lying on its spot just after it
    A boundary never moves past the middle of either neighbouring run.
    """
    s = states.copy()
    n = len(s)
    sl = slice(offset, offset + n)
    c = {H: ctrl[H][sl], A: ctrl[A][sl]}
    out = (feats.ball_obs & (feats.oob > cfg.oob_margin))[sl]
    speed = np.nan_to_num(feats.ball_speed[sl], nan=0.0)
    fast_in = feats.ball_ok[sl] & (speed >= cfg.v_fast) & (np.nan_to_num(feats.inside[sl], nan=-1) > 0)
    still = feats.ball_still[sl]
    w_team, w_dead = cfg.frames(cfg.refine_team_s), cfg.frames(cfg.refine_dead_s)

    starts, ends, vals = runs_of(states)
    for i in range(len(vals) - 1):
        b = int(ends[i])
        old, new = int(vals[i]), int(vals[i + 1])
        lo_lim = b - (b - int(starts[i])) // 2
        hi_lim = b + (int(ends[i + 1]) - b) // 2
        nb = b
        if old != D and new != D:
            lo, hi = max(lo_lim, b - w_team), min(hi_lim, b + w_team)
            before = np.flatnonzero(c[old][lo:b])
            last_old = lo + int(before[-1]) if len(before) else lo - 1
            cand = np.flatnonzero(c[new][last_old + 1:hi + 1])
            if len(cand):
                nb = last_old + 1 + int(cand[0])
        elif new == D:
            lo, hi = max(lo_lim, b - w_dead), min(hi_lim, b + w_dead)
            cand = np.flatnonzero(out[lo:hi + 1])
            if len(cand):
                nb = lo + int(cand[0])
        else:
            lo, hi = max(lo_lim, b - w_dead), min(hi_lim, b + w_dead)
            if b > lo and fast_in[b - 1]:
                slow = np.flatnonzero(~fast_in[lo:b])
                nb = lo + int(slow[-1]) + 1 if len(slow) else lo
            elif b < n and still[b]:
                moving = np.flatnonzero(~still[b:hi + 1])
                nb = b + int(moving[0]) if len(moving) else b
        nb = min(max(nb, lo_lim), hi_lim)
        if nb < b:
            s[nb:b] = new
        elif nb > b:
            s[b:nb] = old
    return s


def restart_possessor(states, ctrl, offset, set_piece_side, cfg):
    """Give the first seconds after a stoppage to the team that took the restart.

    "Whenever play restarts after a stoppage, there is a clear possessor" (CHALLENGE.md). When a
    restart is lost at once -- a long goal kick, a contested throw-in -- the decoder sees only
    the team that ends up with the ball. If the other team was clearly on the ball as play
    restarted (or a calibrated set-piece tag names it), it owns [restart, first control of the
    team that follows).
    """
    s = states.copy()
    n = len(s)
    c = {H: ctrl[H][offset:offset + n], A: ctrl[A][offset:offset + n]}
    look, grace = cfg.frames(cfg.restart_look_s), cfg.frames(0.4)
    starts, ends, vals = runs_of(states)
    for i in range(1, len(vals)):
        if vals[i - 1] != D or vals[i] == D:
            continue
        b, new = int(starts[i]), int(vals[i])
        taker = 1 - new
        lo = max(int(starts[i - 1]), b - look)
        n_taker = int(c[taker][lo:b + grace].sum())
        n_new = int(c[new][lo:b + grace].sum())
        tagged = set_piece_side is not None and set_piece_side[offset + b] == (1 if taker == H else -1)
        if not tagged and not (n_taker >= cfg.restart_min_frames and n_taker >= 4 * n_new):
            continue
        hi = min(int(ends[i]) - 1, b + cfg.frames(cfg.restart_max_s))
        first_new = np.flatnonzero(c[new][b + (0 if tagged else grace):hi])
        if not len(first_new):
            continue
        cut = b + (0 if tagged else grace) + int(first_new[0])
        if cut - b >= cfg.frames(cfg.min_edge_s):
            s[b:cut] = taker
    return s


# ---------------------------------------------------------------------------------------- main
def decode_states(team_llr, dead_llr, ctrl, feats, blocks, set_piece_side, cfg):
    """Viterbi and the clean-up steps, one int8 state array per block."""
    out = []
    for first, last in blocks:
        sl = slice(first, last + 1)
        half_llr = 0.5 * team_llr[sl]
        states = viterbi(half_llr, -half_llr, dead_llr[sl], cfg.c_switch, cfg.c_die, cfg.c_restart)
        states = enforce_durations(states, cfg)
        if cfg.refine:
            states = refine_boundaries(states, ctrl, feats, first, cfg)
            states = enforce_durations(states, cfg)
        if cfg.use_restart:
            states = restart_possessor(states, ctrl, first, set_piece_side, cfg)
        out.append(states)
    return out


def learned_llr(z_team, z_dead, tag_team, tag_dead, graded, cfg):
    """Decoder inputs from the learned log-odds, plus the Smart Tagging terms."""
    clip = cfg.learned_clip
    team = cfg.learned_team_scale * np.clip(z_team, -clip, clip) + tag_team
    dead = cfg.learned_dead_scale * np.clip(z_dead, -clip, clip) + tag_dead
    return np.where(graded, team, 0.0), np.where(graded, dead, 0.0)


def _labels(decoded, blocks, n):
    labels = np.full(n, "X", dtype="<U1")
    for (first, last), states in zip(blocks, decoded):
        labels[first:last + 1] = LETTERS[states]
    return labels


def decode(feats, dead_llr, tags, names, blocks, cfg=None):
    """Labels 'H' / 'A' / 'D' / 'X' for every video frame, and a dict of diagnostics.

    blocks: [(first, last)] graded spans in video frames, inclusive. Frames outside are 'X'.
    """
    cfg = cfg or Config()
    n = feats.n
    graded = np.zeros(n, dtype=bool)
    for first, last in blocks:
        graded[first:last + 1] = True

    ball_llr, ctrl = ball_evidence(feats, cfg)
    team_llr = ball_llr.copy()
    polarity = {}
    set_piece_side = None
    if cfg.use_tags and tags:
        tag_llr, polarity = tag_evidence(tags, names, ball_llr, graded, cfg)
        team_llr += tag_llr
        if polarity.get("set pieces", 0.0) > 0:        # the Team tag was measured to be the taker
            set_piece_side = np.zeros(n, dtype=np.int8)
            pad = cfg.frames(cfg.setpiece_pre_s)
            for t in tags:
                side = side_of(t["tags"].get("team"), names)
                if t["code"] == "set pieces" and side:
                    set_piece_side[max(0, t["start"] - pad):min(n, t["end"] + pad)] = side
    if cfg.learned:
        # Evidence from the models trained on open matches (learned.py). Their inputs are the
        # tag-free hand evidence, exactly as in training; what Smart Tagging adds (team prior,
        # set pieces, the stretch before kick-off) is put back on top.
        from . import deadball, learned
        dead_hand, _ = deadball.compute(feats, [], blocks, cfg)
        hint = hint_evidence(feats, ball_llr, dead_hand, blocks, cfg) if cfg.use_hint \
            else np.zeros(n)
        z_team, z_dead = learned.emissions(feats, ball_llr, hint, dead_hand, blocks)
        team_llr, dead_llr = learned_llr(z_team, z_dead, team_llr - ball_llr,
                                         dead_llr - dead_hand, graded, cfg)
        decoded = decode_states(team_llr, dead_llr, ctrl, feats, blocks, set_piece_side, cfg)
        return _labels(decoded, blocks, n), {"ball_llr": ball_llr, "team_llr": team_llr,
                                             "tag_polarity": polarity}

    hint = hint_evidence(feats, ball_llr, dead_llr, blocks, cfg) if cfg.use_hint else None
    if cfg.use_hint:
        team_llr += hint

    def run(team, dead):
        return decode_states(team, dead, ctrl, feats, blocks, set_piece_side, cfg)

    decoded = run(team_llr, dead_llr)
    base_llr = team_llr - hint if cfg.use_hint else team_llr
    for _ in range(cfg.hint_rounds if cfg.use_hint else 0):
        # self-training: the decoder's own output becomes the pseudo-labels of the next round
        hint2, dead2 = np.zeros(n), dead_llr.copy()
        for (first, last), states in zip(blocks, decoded):
            sl = slice(first, last + 1)
            z = selftrain_llr(feats.team_feats[sl], states != D, (states == H).astype(float),
                              cfg.hint_min_frames, stride=cfg.hint_stride)
            if z is not None:
                z = np.clip(cfg.hint_shrink * z, -cfg.hint_clip, cfg.hint_clip)
                hint2[sl] = np.where(feats.ball_ok[sl], cfg.hint_on_ball * z, z)
            if cfg.dead_iter:
                z = selftrain_llr(feats.dead_feats[sl], np.ones(len(states), dtype=bool),
                                  (states == D).astype(float), cfg.hint_min_frames,
                                  stride=cfg.hint_stride)
                if z is not None:
                    dead2[sl] += np.clip(cfg.dead_iter_shrink * z, -cfg.dead_iter_clip, cfg.dead_iter_clip)
        team_llr = base_llr + hint2
        decoded = run(team_llr, dead2)

    return _labels(decoded, blocks, n), {"ball_llr": ball_llr, "team_llr": team_llr,
                                         "tag_polarity": polarity}
