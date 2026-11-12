# CLAUDE.md – LÅST FIL (kun mennesker endrer den)

Claude: du kan ikke og skal ikke endre denne filen, `.claude/` eller `.githooks/`.
Er en sperre i veien: stopp og si fra. Ikke omgå den.

## Prosjekt
Lag i More than a Hack 2027 (Phase 1). Oppgave: for hver frame (25 fps) i én
fotballkamp (FC Barcelona – Athletic Club) si hvem som har ballen: `H`, `A`, `D` eller `X`.
Leveranser: CSV `frame,label` (199 319 rader), offentlig kode-repo, teknisk dokument.
Kart over repoet, moduler, kommandoer og åpne avklaringer: **`docs/PROJECT_MAP.md`**.

## Absolutte regler (brudd kan gi diskvalifisering)
1. **Kampdata skal aldri til Claude.** Aldri åpne, lese, liste, søke i eller printe
   innhold i `data/`, `evaluation/gt/`, mappen `MTAH_DATA_DIR` peker på, eller filer som ser ut
   som kampdata (video, tracking, homografi, smart tagging, fasit). Gjelder også utdrag,
   plott, skjermbilder, feilmeldinger med dataverdier, aggregerte tall fra ekte data og
   CSV-er laget fra dem. Du kjører aldri kode på ekte data.
2. **Etikettene kommer fra en algoritme.** Ingen håndmerking, ingen håndretting av output,
   ingen regler for enkeltframes eller hardkodede tidsrom i denne kampen, ingen eventdata
   eller annen menneskelaget data om kampen. (Omgangsgrensene fra den offentlige
   oppgavebeskrivelsen er greit som navngitte konstanter.)
3. **Aldri commit data** eller noe avledet av den. Repoet blir offentlig.
   Ikke bruk `git add -f` eller `--no-verify`. Ikke push uten at et menneske ber om det.
4. **Nøyaktig reproduksjon:** fast seed, låste versjoner av avhengigheter, ingen manuelle
   steg. Én kommando skal gi identisk CSV.
5. **Trening og tuning bare på åpne data fra andre kamper** (sjekk lisens først). De 5
   offentlige minuttene brukes ikke til å velge modell.
6. Settes `MTAH_DATA_DIR` i miljøet ditt: stopp og si fra. Claude skal startes i en
   terminal uten den.

## Arbeidsflyt
1. Les `docs/PROJECT_MAP.md`.
2. Skriv kode og tester.
3. Test på **falske data** i `test_data/` (laget av `test_data_generator/`).
4. Mennesker kjører på ekte data i **egen terminal uten Claude** og limer aldri inn output.
5. Feil på ekte data? Gjenskap den på falske data og vis Claude den versjonen.
6. Lager eller flytter du moduler eller kommandoer: oppdater `docs/PROJECT_MAP.md` i samme
   commit. Bare info fra kode og offentlig dokumentasjon, aldri funn fra ekte data.

Kode: Python, `pathlib`, må fungere på både Windows og Mac. Ingen shell-skript, ingen
OS-spesifikke stier. Datasti hentes kun fra `MTAH_DATA_DIR`.

Du kan lese: offentlig dokumentasjon og kode (README, DATA.md, RULES.md, `starter/*.py`) og `docs/`.

## Usikker? Stopp og spør
Er du usikker på om noe er innenfor reglene, eller en fil/mappe kan inneholde kampdata:
ikke åpne den, ikke gjett. Forklar hva du ville gjort og spør. Regelansvarlig: Morten.
