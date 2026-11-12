"""Alle innstillinger for ekstern validering, samlet på ett sted.

Bare ÅPNE data (Sportec/IDSSE, CC-BY 4.0). Aldri konkurransedata.

Tallene for "gjør dataene dårligere" er hentet fra arrangørens OFFENTLIGE
dokumentasjon (DATA.md) og generell fotballkunnskap - IKKE målt på kampdataene.
Ikke juster dem ut fra målinger på de ekte dataene (se G1 og R9).
"""
from pathlib import Path

# --- Stier (fungerer på Windows og Mac) ------------------------------------
VALIDATION_DIR = Path(__file__).resolve().parent
REPO_ROOT = VALIDATION_DIR.parent
EVALUATION_DIR = REPO_ROOT / "evaluation"          # arrangørens possession_metrics.py
OUT_DIR = REPO_ROOT / "ext_data"                   # output, stoppes av .gitignore
PREDS_DIR = OUT_DIR / "preds"
RESULTS_DIR = OUT_DIR / "results"

# --- Kamper -----------------------------------------------------------------
# Gyldige ID-er i kloppy 3.19.1 (sjekket i kloppy-kildekoden).
# NB: J03WPF og J03WQF finnes ikke i datasettet - riktige ID-er er J03WQQ og J03WR9.
ALL_MATCHES = ["J03WPY", "J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WQQ", "J03WR9"]
DEFAULT_MATCHES = ["J03WPY"]   # start med én kamp

# --- Generelt ---------------------------------------------------------------
SEED = 42
FPS = 25

# Rammer med X (ikke vurdert) rundt omgangene, som i konkurransen
# (før kamp, pause, etter kamp). All tracking er NaN her.
PRE_FRAMES = 250        # 10 s
HALFTIME_FRAMES = 500   # 20 s (kortere enn ekte pause - holder for å skille omgangene)
POST_FRAMES = 250       # 10 s

# --- Format (kapittel 4b) ---------------------------------------------------
N_SLOTS = 46
SLOT_NAME = "s{:02d}"   # kolonner blir s00_x, s00_y, s00_s, ... Endre hvis generatoren bruker et annet navn.

# --- Gjør dataene dårligere -------------------------------------------------
# Ballen
BALL_MISSING_TARGET = 0.41        # andel av spilletiden uten ball (DATA.md: ca. 41 %)
BALL_GAP_SECONDS = (0.5, 5.0)     # lengde på hvert hull (min, maks)
DUEL_RADIUS_M = 2.5               # duell = spiller fra BEGGE lag nærmere ballen enn dette
DUEL_WEIGHT = 3.0                 # hvor mye oftere hull starter i duell
NEAR_GOAL_M = 20.0                # "nær mål" = ballen nærmere et mål enn dette
NEAR_GOAL_WEIGHT = 2.0            # hvor mye oftere hull starter nær mål

# Spillere / slots
PLAYER_GAPS_PER_MINUTE = 0.5      # i snitt ett hull per spiller annethvert minutt
PLAYER_GAP_SECONDS = (1.0, 8.0)   # lengde på hvert hull
SLOT_SWITCH_PROB = 0.3            # sjanse for at spilleren får NY slot etter et hull
SPEED_SMOOTH_FRAMES = 5           # glatting av farten (m/s)
