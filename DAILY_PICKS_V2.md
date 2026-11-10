# Tennis Oracle 2.0 — Daily Picks (MVP, sperimentale)

Questo branch aggiunge un motore **pre-match** che può creare, se disponibili:
- **Singola:** 1 evento con quota >= 1.40.
- **3×3:** esattamente 3 eventi, quota totale 2.70–3.45.
- **Multipla 5+:** da 3 a 5 eventi differenti, quota totale >= 5.
- **Bomba 7+:** da 3 a 5 eventi differenti, quota totale >= 7.

La quota di una multipla è il **prodotto**, non la somma, delle quote.
L'assenza di selezioni è un risultato valido e non produce numeri inventati.

## Fonte e limiti

L'endpoint GET /daily_picks legge il calendario pre-match ESPN e le quote
ricavate dal connettore The Odds API preesistente. Richiede:
- ODDS_API_KEY definita come variabile d'ambiente
- modelli joblib preesistenti disponibili e caricabili
- identità dei giocatori riconosciute senza ambiguità
- almeno 20 incontri storici e 5 sulla superficie per entrambi i giocatori
- quote valide >= 1.40, incontro futuro e non ancora iniziato
- per il mercato games: esclusivamente Over/Under 22.5 su best-of-three

**Non vengono analizzati i best-of-five ATP Slam** in questo MVP.
Questo impedisce il riutilizzo improprio del modello 22.5 in gare BO5.
I mercati di ace, handicap, tie-break e set non sono inclusi
fino alla verifica di quote e modello specifici.

La scheda mostra stime non certificate dei modelli e un *quality score*
che misura solo la disponibilità dei dati. Non equivale a probabilità di
successo. Le quote possono cambiare o non essere puntabili.
Mancano ancora backtesting walk-forward corretto, calibrazione fuori campione,
settlement automatico verificato e confronto di closing line.

**Non sono stati verificati rendimenti o vantaggi sui bookmaker.**

## Test e utilizzo

    python -m unittest discover -s tests -p "test_*.py" -v

Il motore puro (tennis_daily_picks.py) è testabile senza rete e senza
servizi a pagamento. L'endpoint usa quote live solo se sono disponibili.
Non attivare alcun servizio a pagamento senza un'approvazione separata.

## Sicurezza

Il token Telegram precedentemente scritto in tennis_config.py è stato
rimosso dalla versione corrente, ma **resta nella storia Git**.
Revocarlo con BotFather e impostare BOT_TOKEN come secret d'ambiente.
Controllare anche fork, copie e vecchie immagini di deploy.

## Next steps

1. Ripristinare fonti storiche con licenza compatibile e aggiornamenti affidabili.
2. Eliminare leakage nelle feature e costruire training/test temporali veri.
3. Confrontare calibrazione e log loss con baseline Elo.
4. Conservare snapshot di quote/timestamp e settlement delle combinazioni.
5. Solo dopo validare e ampliare i mercati e rimuovere l'etichetta sperimentale.

## Storico cartaceo (opzionale)

Impostare ORACLE_DB_PATH verso un file SQLite **su volume persistente** per
conservare automaticamente la prima proposta di ogni categoria per giorno
UTC. Richieste successive non riscrivono la quota originariamente registrata.

- GET /daily_picks/history: storico e ROI su puntata simulata di 1 unità;
  disabilitato senza ORACLE_DB_PATH.
- Per registrare manualmente gli esiti, senza modificare le quote storiche:

    python tennis_daily_tracker.py settle --db /path/oracle.sqlite --id 2026-11-10:single --outcomes win

Per le multiple usare una voce win/loss/void per ogni selezione, nello
stesso ordine di emissione, separata da virgole. Le liquidazioni sono
esplicitamente contrassegnate come **manuali**: il ROI non prova la
correttezza degli esiti né rappresenta operazioni realmente piazzate.
Un disco temporaneo della piattaforma cancellerà lo storico: non
considerarlo persistente senza volume adeguato.

Historical sources: the original Sackmann URLs return 404 in 2026. The
loader now defaults to the archived CC BY-NC-SA non-commercial mirror
(Aneeshers/tennis-sackmann-archive). This is NOT a live results feed;
inspect freshness and licensing before training and any commercial use.

## Chronological feature engine (in development)
\`tennis_temporal.py\` builds point-in-time player features per *calendar
day*: no same-day match outcomes feed a prediction for another fixture
that day. ATP/WTA histories are isolated and potential legacy numeric ID
collisions are excluded from the backward-compatible player lookup.
Train/calibration/test masks are strictly date-disjoint. The scoring
models in this experimental version are still legacy until the trainer
and API are explicitly switched over to newly validated artifacts.
