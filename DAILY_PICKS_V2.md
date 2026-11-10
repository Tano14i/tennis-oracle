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


## Chronological model training / held-out evaluation

The retraining entry point now calls the point-in-time features: previous
dates only, deterministic player order, no winner/loser double rows.
Training and calibration use separate periods.

Run:
    python tennis_trainer.py
    python tennis_backtest.py

The independent backtest re-fits NEW models on pre-2024 data, calibrates
only on later pre-2024 data and evaluates untouched matches from 2024.
It reports Brier, log loss, accuracy, and a historical rank baseline.
It deliberately leaves ROI blank without actual timestamped bookmaker odds.
Calibration scores are NOT independent test scores.

Models created by the new trainer carry model_manifest.json. Daily Picks
fails closed if this manifest is absent/legacy or the historical data is
over 21 days old. The known archival mirror is too stale for trustworthy
current-day signals until a fresh licensed data feed is available.


## One-off archival holdout run
The workflow temporal_holdout_validation.yml triggers on creation or
manual dispatch. It evaluates historical archived data with 50 trees/model
to check the complete pipeline. This is a development smoke evaluation,
NOT a benchmark of the default 250-tree production model. It uploads a
JSON artifact only if the sources and processing succeed.


## UI freshness and source notice
The health response now exposes daily_picks_ready and a reason when
models/data/quote access are unavailable. The frontend explicitly shows
a historical-data warning instead of claiming fully operational daily picks.
Historical attribution and non-commercial licensing are visible in-app.


## Performance breakdowns and stale-data deployment blocking
The held-out JSON now includes by_year and by_tour accuracy/Brier breakdowns
for each trained market to expose drift. The monthly trainer fails if its
latest historical event is older than 21 days. A manual archival research
override exists via ORACLE_ALLOW_STALE_RESEARCH=1, but does not override
the Daily Picks freshness gate and must never be set in production.


## Candidate fresh feed not approved
External Valuebetennis 2026 CSV (semicolon-delimited, CC BY 4.0)
provided 77,908 rows and the expected odds columns. However, the
2026-11-10 UTC audit found four future-dated match records already
containing final scores: https://github.com/Tano14i/tennis-oracle/actions/runs/42785830368
The source is **quarantined**, not used for model training, quote
history or Daily Picks. A standalone quality scanner is provided in
tennis_source_audit.py with unit tests, but it does not import data.


## Verified milestone: 2026-11-10

- Latest tested branch: 41 offline unit tests passed:
  https://github.com/Tano14i/tennis-oracle/actions/runs/42791023562
- Full-size 250-tree chronological holdout finished successfully:
  https://github.com/Tano14i/tennis-oracle/actions/runs/42784958627
- Winner model, unseen 2024–2026 matches: 41,100 evaluated;
  accuracy **65.30%** vs rank-only accuracy **62.54%**;
  Brier **0.21515** vs base-rate Brier **0.24990**.
- Historical evaluation only. **Not a real-money win rate**; this
  evaluation does not include contemporaneous bookmaker prices or ROI.
- Last historical archive match in this test: **2026-06-01**. It is
  too stale for live daily suggestions under the freshness gate.
- The investigated 2026 external source was rejected after detecting
  four future-dated matches already carrying final scores:
  https://github.com/Tano14i/tennis-oracle/actions/runs/42785830368
- The source is NOT imported into training or price history.
- Deployment is NOT performed by this branch. Do not merge/deploy
  before a clean and licensed current results source, refreshed models,
  live API checks and credential rotation.
