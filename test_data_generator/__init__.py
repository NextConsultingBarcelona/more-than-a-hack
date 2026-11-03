"""Generator for FALSKE (syntetiske) fotballdata til More than a Hack 2027.

Formål: teste at koden vår virker (format, antall rader, ingen krasj, at H/A/D-logikken
oppfører seg). IKKE for å trene, tune eller velge modell.

Bygget kun fra offentlig formatbeskrivelse (DATA.md / lagets kontekstdokument, kap. 4b)
og generelle fotballfakta. Leser aldri ekte kampdata, og skal aldri peke mot data/.

Bruk fra kode:
    from test_data_generator import generate
    match = generate(seed=42, mode="short")
    match.tracking, match.slot_teams, match.truth, match.events

Bruk fra terminal:
    python -m test_data_generator --seed 42 --mode short
"""

from .build import GeneratedMatch, generate
from .config import FULL, SHORT, Params, Timeline

__all__ = ["generate", "GeneratedMatch", "Params", "Timeline", "FULL", "SHORT"]
__version__ = "1.0.0"
