# Tennis Oracle 2.0 — Daily Picks (MVP, sperimentale)

Questo branch aggiunge un motore **pre-match** che può creare, se disponibili:
- **Singola:** 1 evento con quota >= 1.40.
- **3×3:** esattamente 3 eventi, quota totale 2.75–3.45.
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
Mancano ancora backtesting walk-forward corretto, calibratura fuori campione,
storico persistente con settlement e confronto di closing line.

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
