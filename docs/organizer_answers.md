# Svar fra arrangøren

> Skriftlige avklaringer fra arrangørene av More than a Hack 2027.
> Filen inneholder **ingen kampdata**, bare regeltolkninger.
> Vedlikeholdes av Morten (QA / kontakt med arrangør).

---

## 1. Trening, tuning og bruk av video (e-post til alle deltakere, ca. 5. november 2026)

| Ikke lov | Lov |
|---|---|
| Sjekke output mot videoen | Automatiske sjekker som bare bruker de utleverte dataene |
| Håndmerking, menneskelaget data eller annen manuelt laget data om kampen | Se på dataene og videoen (som menneske) for å forstå hva som teller som possession |
| Laste opp de utleverte dataene til AI-tjenester | Bruke video, tracking, Smart Tagging og annen generert data som input, så lenge algoritmen ikke er trent på fasiten eller videoen |
| Trene, tilpasse eller kalibrere algoritmen på fasiten eller videoen | Levere filer med mellomsteg |
| | Bruke ethvert datasett vi har rettigheter til (sjekk lisensen) |

### Slik følger laget dette opp

| # | Regel | I praksis |
|---|---|---|
| R8 | Ikke sjekk output mot videoen | Ingen sammenligning av våre H/A/D-etiketter med videoen, verken manuelt eller ved å «se om det stemmer» |
| R9 | Trening, tilpasning og kalibrering KUN på eksterne data | Alle parametere (f.eks. HMM-ens straff for lagbytte, terskler for dødball) velges med `external_validation/` på åpne Sportec/IDSSE-kamper. Aldri på fasiten eller videoen |

- De 5 offentlige minuttene brukes **bare** til å sjekke at pipelinen virker. Å velge parametere ut fra scoren der regnes som kalibrering og er ikke lov.
- Ekstern validering er obligatorisk hvis vi tuner noen parametere.

---

## 2. Ubesvarte spørsmål

| # | Spørsmål | Til vi har svar |
|---|---|---|
| G1 | Kan aggregerte tall fra ekte data deles med en sky-AI? | Nei |
| G2 | Kan Claude lese den offentlige 5-minutters fasiten i arrangørens repo? | Nei |
| G3 | Er vilkårsdokumentet konfidensielt? | Behandles som konfidensielt |

Nye svar legges til som egne avsnitt med dato og kilde.
