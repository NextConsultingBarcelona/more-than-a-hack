# Sjekkliste før innlevering – More than a Hack 2027, Phase 1

> **Eier:** Morten (QA)  ·  **Brukes:** ved kodefrys 14. nov og ved endelig innlevering 16. nov  ·  **Offisiell frist:** 17. nov 2026
> **Grunnlag:** `claude/MTAH2027_kontekst_og_regler.md` (R1–R9, kapittel 5) + arrangørens RULES.md / SUBMISSION.md. Vilkårene du signerte er den endelige autoriteten.

## Slik bruker du listen

1. Gå gjennom seksjonene **i rekkefølge**. Seksjon B–D forutsetter at koden er frosset.
2. Kryss av bare når punktet er **verifisert**, ikke når noen sier at det er gjort. Skriv initialer og dato i kolonnen «Sjekket».
3. **Alt som rører ekte kampdata kjøres i en vanlig terminal uten Claude** (R1). Det gjelder spesielt seksjon B og C. Ikke lim inn output fra disse kommandoene i Claude, heller ikke tall (G1 er ikke besvart).
4. Ett punkt som ikke er OK = **stopp**. Rett feilen, og kjør seksjonen på nytt fra start.

**Symboler:** 🔴 = kan gi diskvalifisering · 🟡 = kan gi feil score eller avvist innlevering · ⚪ = god praksis

---

## A. Lag og påmelding

| ✓ | # | Punkt | Slik sjekker du | Regel | Eier | Sjekket |
|---|---|---|---|---|---|---|
| ☐ | A1 | 🔴 Laget har **minst 3** personer, og alle (Oskar, Martin, Morten) er påmeldt og har signert vilkårene | Bekreftelses-e-post fra hver av de tre | R6 | Morten | |
| ☐ | A2 | 🔴 Arrangøren har bekreftet lagendringen (5 → 3, Philip og Kristian ut) | Skriftlig svar på e-posten fra 12. nov, lagret i `docs/organizer_answers.md` | G5 | Morten | |
| ☐ | A3 | 🟡 **Lagnavnet er tegn-for-tegn identisk** i påmeldingen (alle tre), Google-skjemaet og filnavnene `NEXT.csv` og `NEXT.pdf` / `.md` | Kopier lagnavnet fra påmeldingsbekreftelsen og lim det inn, ikke skriv det på nytt. Sjekk store/små bokstaver, mellomrom, bindestrek og æøå | Kap. 5 | Morten | |
| ☐ | A4 | ⚪ Lagnavnet står skrevet ned ett sted som fasit (f.eks. øverst i README) | Les README | – | Morten | |

## B. CSV-filen (kjøres uten Claude)

| ✓ | # | Punkt | Slik sjekker du | Regel | Eier | Sjekket |
|---|---|---|---|---|---|---|
| ☐ | B1 | 🟡 Filen heter `NEXT.csv` | Se A3 | Kap. 5 | Martin | |
| ☐ | B2 | 🟡 `validate_submission.py` er godkjent på **nøyaktig den filen som lastes opp**, ikke en kopi eller en eldre versjon | Kjør validatoren, og regn deretter ut SHA-256 av samme fil (se B6). Last opp akkurat den filen, uten å åpne eller lagre den i Excel/Numbers imellom | Kap. 5 | Martin + Morten | |
| ☐ | B3 | 🟡 Header er nøyaktig `frame,label` | Validatoren | Kap. 2 | Martin | |
| ☐ | B4 | 🟡 **199 319 datarader** (frame 0 … 199 318), i stigende rekkefølge, ingen hull, ingen duplikater | Validatoren (+ 199 320 linjer inkl. header hvis du teller selv) | Kap. 2 | Martin | |
| ☐ | B5 | 🟡 Bare etikettene `H`, `A`, `D`, `X` forekommer | Validatoren | Kap. 2 | Martin | |
| ☐ | B6 | 🟡 Sjekksum er notert: `python -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" NEXT.csv` | Lim sjekksummen inn i innleveringsloggen nederst. Den brukes i C4 | R4 | Morten | |
| ☐ | B7 | 🟡 Logisk sanity check (automatisk, ingen visuell sjekk): frames 110 808–113 741 er `D` (vurdert, men før avspark i 2. omgang) | Et testskript i repoet som bare skriver `OK`/`FEIL` | Kap. 2 | Martin | |
| ☐ | B8 | ⚪ Linjeskift er `\n` (ikke `\r\n`), og filen har ingen BOM | Se C5 – viktig fordi laget bruker både Windows og Mac | R4 | Martin | |

## C. Reproduksjon fra ren klon (kjøres uten Claude)

| ✓ | # | Punkt | Slik sjekker du | Regel | Eier | Sjekket |
|---|---|---|---|---|---|---|
| ☐ | C1 | 🔴 Alle avhengigheter er **låst til eksakte versjoner** (lock-fil eller `requirements.txt` med `==` for alle pakker, også indirekte), og Python-versjonen er angitt | Les filen: ingen `>=`, `~=` eller pakker uten versjon | R4 | Martin | |
| ☐ | C2 | 🔴 **Én kommando** lager CSV-en (f.eks. `python run.py`), uten manuelle steg, notebooks eller redigering av filer | Les README: kjøreinstruksjonen har én kommando etter installasjon | R4 | Martin | |
| ☐ | C3 | 🔴 Alle tilfeldige steg har **fast seed**, og alle parametere ligger fast i kode/konfig (ikke regnet ut på nytt ved kjøring) | Søk i koden etter `random`, `np.random`, `sample`, `shuffle` uten seed | R4, R9 | Oskar | |
| ☐ | C4 | 🔴 **Ren klon gjenskaper identisk fil:** ny mappe → `git clone` av `main` → nytt virtuelt miljø → installer fra lock-filen → sett `MTAH_DATA_DIR` → kjør kommandoen → SHA-256 er **lik** B6 | Sammenlign sjekksummene. Ulik sjekksum = ikke godkjent, også om bare én rad er ulik | R4 | Morten | |
| ☐ | C5 | 🔴 C4 er gjort på **både Windows og Mac**, med samme sjekksum | Én person på hvert OS. Vanlige feilkilder: `\r\n`-linjeskift, sortering av filer i mapper, flyttall-avrunding | R4 | Morten + én til | |
| ☐ | C6 | 🟡 Commit-hashen som ble brukt i C4 er den samme som er `main` på GitHub ved innlevering | `git rev-parse HEAD` i den rene klonen, sammenlign med GitHub | R4 | Morten | |
| ☐ | C7 | ⚪ README forklarer hvor dataene skal ligge (`MTAH_DATA_DIR`) uten å inneholde data | Les README | R4, R5 | Morten | |

## D. Data og hemmeligheter i repoet

| ✓ | # | Punkt | Slik sjekker du | Regel | Eier | Sjekket |
|---|---|---|---|---|---|---|
| ☐ | D1 | 🔴 **Ingen kampdata i nåværende filer** (tracking, Smart Tagging, homografi, video, fasit, utdrag, klipp, bilder) | `git ls-files` og se etter `.xml`, `.txt`, `.csv`, `.json`, `.mp4`, `.png`, `.parquet`, `.pkl`, `.npy` | R5 | Morten | |
| ☐ | D2 | 🔴 **Ingen kampdata i git-historikken**, på noen gren | `git log --all --name-only --pretty=format:` og se etter de samme filtypene. Finnes noe: **stopp og ikke gjør repoet public** – historikken må skrives om før det | R5 | Morten | |
| ☐ | D3 | 🔴 Ingen **avledede** data fra kampen: våre egne output-CSV-er, mellomfiler, cache, plott eller figurer laget fra ekte data | Samme kommandoer som D1/D2. Se også uklart punkt U4 | R5 | Morten | |
| ☐ | D4 | 🔴 Ingen Jupyter-notebooks med **lagrede output-celler** fra ekte data | Åpne hver `.ipynb` og sjekk at «outputs» er tomme | R5 | Morten | |
| ☐ | D5 | 🔴 `evaluation/gt/` (5-min-fasiten) er ikke committet av oss | `git log --all -- evaluation/gt/` skal være tom (eller bare vise arrangørens egne commits) | R3, R5 | Morten | |
| ☐ | D6 | 🟡 Ingen logger med dataverdier (`*.log`, tracebacks, utskrifter) | `git ls-files` | R5 | Morten | |
| ☐ | D7 | 🟡 Ingen passord, tokens eller private e-poster | Søk etter `token`, `password`, `@gmail`, `api_key` | – | Morten | |
| ☐ | D8 | ⚪ Sperrene er fortsatt på plass: `.gitignore` (data/, test_data/, evaluation/gt/), `.githooks/`, `.claude/settings.json` | Les filene | Kap. 7 | Morten | |
| ☐ | D9 | 🔴 Repoet `NextConsultingBarcelona/more-than-a-hack` er **public** når skjemaet sendes – og først etter at D1–D7 er OK | Åpne repo-lenken i et inkognito-vindu (ikke innlogget) | Kap. 5 | Morten | |

## E. Metode og regler

| ✓ | # | Punkt | Slik sjekker du | Regel | Eier | Sjekket |
|---|---|---|---|---|---|---|
| ☐ | E1 | 🔴 **Ingen håndmerking** av denne kampen, for noe formål (heller ikke «litt for testing») | Spør Oskar og Martin direkte, og noter svaret | R2 + lagregel | Morten | |
| ☐ | E2 | 🔴 **Ingen håndretting** av output og ingen **frame-spesifikke regler** (f.eks. `if frame == 45210`, eller lister med frame-intervaller skrevet inn for hånd) | Søk i koden etter tall i størrelsesorden 10 000–199 318. Unntak: de offisielle omgangsgrensene fra CHALLENGE.md (11 185, 90 651, 110 808, 190 197) – sjekk at de er hentet fra dokumentasjonen og står med kommentar | R2 | Morten | |
| ☐ | E3 | 🔴 **Ingen parametere valgt ut fra scoren på de 5 offentlige minuttene** | Hver parameter har en kilde: Sportec-tuning, litteratur eller fotballfaglig begrunnelse (se tabell i teknisk dokument). Anbefaling: commit og tag parameterne **før** noen kjører 5-min-sjekken | R3, R9 | Oskar + Morten | |
| ☐ | E4 | 🔴 **Ingenting er sammenlignet med videoen** – verken manuelt eller ved å «se om det stemmer» | Spør alle tre direkte. Å se video for å forstå possession-begrepet er lov | R8 | Morten | |
| ☐ | E5 | 🔴 All trening, tuning og kalibrering er gjort **bare på eksterne data** (Sportec), og koden for dette ligger i repoet | Tuning-skriptet finnes og leser bare Sportec-data via kloppy | R9 | Morten | |
| ☐ | E6 | 🔴 Ingen eventdata eller annen menneskelaget data om denne kampen | Les listen over datakilder i koden | R3 + lagregel | Morten | |
| ☐ | E7 | 🔴 Ingen kampdata er gitt til Claude eller annen sky-AI i løpet av prosjektet | Spør alle tre direkte. Er svaret «kanskje»: stopp og avklar med arrangøren | R1 | Morten | |

## F. Ekstern data, modeller og kreditering

| ✓ | # | Punkt | Slik sjekker du | Regel | Eier | Sjekket |
|---|---|---|---|---|---|---|
| ☐ | F1 | 🔴 **Sportec/IDSSE** er kreditert i teknisk dokument og README: Bundesliga 2022/23, lisens **CC-BY 4.0**, med referansen **Bassek et al. (2025), *Scientific Data*, doi:10.1038/s41597-025-04505-y**, og hvilke kamp-ID-er som er brukt | Les begge dokumentene | R7, G4 | Morten | |
| ☐ | F2 | 🟡 Metrica Sample Data (hvis brukt) er oppgitt som kilde | Les dokumentet | R7, G4 | Morten | |
| ☐ | F3 | 🟡 Arrangørens starter-kode (MIT) beholder lisensteksten der den er kopiert | Se etter `LICENSE` / kommentar i kopierte filer | – | Martin | |
| ☐ | F4 | 🟡 Alle biblioteker vi bygger på er listet (kloppy, numpy, pandas osv.), med lisens sjekket | Sammenlign med lock-filen | R7 | Morten | |
| ☐ | F5 | 🔴 **Ingen private modeller eller interne datasett** – alt kan hentes av arrangøren | Hver ekstern ressurs har en offentlig lenke | R7 | Morten | |
| ☐ | F6 | 🟡 Står det eksplisitt «ingen forhåndstrente modeller» hvis vi ikke bruker noen? | Les dokumentet | R7 | Morten | |

## G. Teknisk dokument

| ✓ | # | Punkt | Slik sjekker du | Regel | Eier | Sjekket |
|---|---|---|---|---|---|---|
| ☐ | G1 | 🟡 Filen heter `NEXT.pdf` eller `NEXT.md` | Se A3 | Kap. 5 | Morten | |
| ☐ | G2 | 🟡 Dekker **alle seks krav i kapittel 5**: (1) tilnærming og hvorfor, (2) datakilder – også hva vi prøvde og droppet, (3) possession-reglene vi implementerte (løse baller, dueller, restarter), (4) begrensninger og hva vi ville fikset, (5) reproduksjon, (6) all ekstern data og alle modeller | Kryss av hvert krav mot overskriftene i malen | Kap. 5 | Morten | |
| ☐ | G3 | 🟡 Ekstern validering er beskrevet: hvilke Sportec-kamper, hvordan de ble «gjort dårligere», og hvilke parametere som ble tunet | Egen seksjon i dokumentet | R9 | Morten | |
| ☐ | G4 | 🔴 Inneholder **ingen data** fra kampen (ingen plott, skjermbilder eller tabeller med verdier fra ekte data) | Les dokumentet. Se uklart punkt U4 | R5 | Morten | |
| ☐ | G5 | ⚪ «Noen få sider» – ikke en avhandling | Sidetall | Kap. 5 | Morten | |
| ☐ | G6 | 🟡 Kjøreinstruksjonen i dokumentet er **identisk** med README (samme kommando) | Sammenlign | R4 | Morten | |
| ☐ | G7 | ⚪ Lest korrektur av både Oskar og Martin | Godkjenning i Slack | – | Morten | |

## H. Selve innleveringen (Google-skjemaet)

| ✓ | # | Punkt | Slik sjekker du | Regel | Eier | Sjekket |
|---|---|---|---|---|---|---|
| ☐ | H1 | 🟡 **Én person** fyller ut skjemaet for laget (ett svar per lag) | Avtal hvem på forhånd | Kap. 5 | Martin | |
| ☐ | H2 | 🟡 Lagnavn er limt inn fra påmeldingen (A3) | – | Kap. 5 | Martin | |
| ☐ | H3 | 🟡 Opplastet CSV har samme SHA-256 som B6 | Regn ut sjekksum på den filen du faktisk velger i opplastingsdialogen | R4 | Martin | |
| ☐ | H4 | 🟡 Repo-lenken peker til riktig repo og fungerer uten innlogging | Inkognito-vindu | Kap. 5 | Martin | |
| ☐ | H5 | 🟡 Teknisk dokument er lastet opp | – | Kap. 5 | Martin | |
| ☐ | H6 | 🟡 Kvittering på e-post er mottatt og videresendt til laget | – | – | Martin | |
| ☐ | H7 | ⚪ Levert **16. nov** (ett døgns margin). Endringer etterpå: husk at siste svar før fristen teller | – | Kap. 5 | Martin | |

## I. Etter innlevering

| ✓ | # | Punkt | Regel |
|---|---|---|---|
| ☐ | I1 | Ingen nye commits på `main` som endrer output etter innlevering (eller: tag innleveringscommiten, f.eks. `phase1-submission`) | R4 |
| ☐ | I2 | Ekte kampdata slettes eller lagres i henhold til vilkårene | R5 |

---

## Uklart i reglene – avklar før innlevering

Jeg har **ikke** antatt noe om disse. De bør med i neste e-post til arrangøren, eller sjekkes i SUBMISSION.md / vilkårene:

| # | Spørsmål | Hvorfor det betyr noe |
|---|---|---|
| U1 | **Klokkeslett og tidssone for fristen 17. nov** | Dokumentet sier bare dato. Barcelona (CET) og Norge er samme sone, men «slutten av dagen» kan være UTC eller AoE |
| U2 | Må **tuning-steget** (Sportec) også kunne gjenskapes med én kommando, eller holder det at den endelige CSV-en gjenskapes? | R4 nevner bare CSV-en. R7/R9 krever at metoden er forklart. Trygt valg: begge deler i repoet, men hovedkommandoen bruker faste parametere |
| U3 | Skal vår egen **output-CSV ligge i repoet**? | R5 forbyr avledet data som kan gjenskape kampdataene. Etikettene alene gjør trolig ikke det, men det er ikke avklart. Trygt valg inntil svar: **ikke** commit den |
| U4 | Kan teknisk dokument inneholde **figurer eller tall fra ekte data** (f.eks. en possession-tidslinje)? | Dokumentet sendes til arrangøren, men kan bli offentlig. Trygt valg inntil svar: bare figurer fra Sportec- eller falske data |
| U5 | Er det krav om å oppgi bruk av **AI-kodeassistent** (Claude Code) i dokumentet? | R7 sier «alle eksterne data og modeller». Det er uklart om det gjelder verktøy som skrev kode, ikke modeller som lager etiketter. Trygt og ærlig valg: nevn det kort |
| U6 | Er det greit at repoet var **public allerede fra 1. nov**, før innlevering? | Ikke et regelbrudd så lenge ingen data er committet, men det gjør D2 enda viktigere |
| U7 | G1–G5 fra regeldokumentet | Fortsatt ubesvart per 12. nov |

---

## Innleveringslogg

| Felt | Verdi |
|---|---|
| Lagnavn (limt inn fra påmelding) | NEXT (bekreft mot påmeldingen, se A3) |
| Commit-hash (`main`) | |
| SHA-256 av `NEXT.csv` | |
| SHA-256 fra ren klon – Windows | |
| SHA-256 fra ren klon – Mac | |
| `validate_submission.py` godkjent (dato, av hvem) | |
| Repo satt til public (dato, av hvem) | |
| Skjema sendt (dato, kl., av hvem) | |
| Kvittering mottatt | |

**Godkjent for innlevering:** Oskar ☐  Martin ☐  Morten ☐
