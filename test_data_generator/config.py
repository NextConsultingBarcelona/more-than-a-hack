"""Konstanter og justerbare parametre. Alle «knotter» ligger her.

Kilder:
- Omgangsgrenser og antall frames: offentlig dokumentasjon (CHALLENGE.md / kontekstdok. kap. 2).
- Format: DATA.md (kontekstdok. kap. 4b).
- Realisme-tall: generelle fotballfakta. IKKE målt fra den ekte kampen.
"""

from dataclasses import dataclass

FPS = 25
PITCH_LENGTH = 105.0  # meter (arrangøren bruker ca. 105,3 x 68,0)
PITCH_WIDTH = 68.0

N_SLOTS = 46
SLOTS_PER_TEAM = 23
PLAYERS_PER_TEAM = 11

# Plassholder-navn på track slots. Sjekk mot arrangørens offentlige starter/load_atd.py
# og bytt her hvis den bruker andre navn. Resten av koden bruker bare denne lista.
SLOT_NAMES = tuple(f"s{i:02d}" for i in range(N_SLOTS))

LABELS = ("H", "A", "D", "X")


@dataclass(frozen=True)
class Timeline:
    """Kampstruktur i video-frames (0-indeksert). Slutt-verdier er inklusive."""

    n_frames: int
    h1_start: int  # første vurderte frame i 1. omgang (= avspark)
    h1_end: int  # siste vurderte frame i 1. omgang
    h2_start: int  # første vurderte frame i 2. omgang
    h2_kickoff: int  # avspark 2. omgang (frames h2_start .. h2_kickoff-1 er D)
    h2_end: int  # siste vurderte frame i 2. omgang

    def halves(self):
        """(omgang, første vurderte frame, avspark, slutt eksklusiv) for hver omgang."""
        return [
            (1, self.h1_start, self.h1_start, self.h1_end + 1),
            (2, self.h2_start, self.h2_kickoff, self.h2_end + 1),
        ]


# Samme grenser som den ekte kampen (offentlig dokumentert).
FULL = Timeline(
    n_frames=199_319,
    h1_start=11_185,
    h1_end=90_651,
    h2_start=110_808,
    h2_kickoff=113_742,
    h2_end=190_197,
)

# Kort modus: ca. 10 min spill (2 x 5 min) for raske tester. Samme struktur.
SHORT = Timeline(
    n_frames=16_800,
    h1_start=250,
    h1_end=7_749,
    h2_start=8_750,
    h2_kickoff=9_050,
    h2_end=16_549,
)

MODES = {"full": FULL, "short": SHORT}


@dataclass(frozen=True)
class Params:
    """Justerbare realisme-parametre. Endringer her endrer output for samme seed."""

    # Possession-varighet (sekunder): lognormal, klippet
    possession_median_s: float = 12.0
    possession_sigma: float = 0.5
    possession_min_s: float = 5.0
    possession_max_s: float = 30.0

    # Hvordan en possession slutter (normaliseres mot hverandre)
    p_turnover: float = 0.45
    p_out: float = 0.40
    # Sannsynlighet for en løs ball midt i possession som SAMME lag vinner tilbake
    p_loose_kept: float = 0.15
    loose_min_s: float = 1.0
    loose_max_s: float = 3.0

    # Type død ball når ballen går ut / blåses av
    p_throw_in: float = 0.50
    p_byline: float = 0.30  # corner eller utspark (bare hvis ballen er nær dødlinja)
    p_free_kick: float = 0.20
    byline_reach_m: float = 30.0
    p_goal: float = 0.0  # mål av som standard (se README)

    # Hvor lenge ballen er død (sekunder) per restart-type
    dead_throw_in_s: tuple = (4.0, 10.0)
    dead_goal_kick_s: tuple = (6.0, 15.0)
    dead_corner_s: tuple = (8.0, 15.0)
    dead_free_kick_s: tuple = (5.0, 15.0)
    dead_kickoff_after_goal_s: tuple = (25.0, 45.0)

    # Ballbane
    touch_gap_s: tuple = (1.0, 3.0)
    final_leg_s: float = 1.5

    # Måleutstyr: ball
    ball_missing_target: float = 0.40
    ball_gap_s: tuple = (1.0, 5.0)
    ball_duel_weight: float = 4.0
    ball_goal_weight: float = 3.0
    duel_radius_m: float = 3.0
    near_goal_frac: float = 0.15

    # Måleutstyr: spillere / slots
    player_dropout_per_min: float = 0.25  # per spiller
    player_dropout_s: tuple = (1.0, 5.0)
    duplicate_per_min: float = 0.3  # per lag
    duplicate_s: tuple = (1.0, 4.0)

    # Spillerbevegelse
    sim_hz: int = 5
    vmax_chaser: float = 8.0
    vmax_player: float = 6.0
    vmax_idle: float = 2.0
