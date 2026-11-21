# Tennis Oracle 2.0 — Checkpoint sperimentale (10 novembre 2026)

> Ricerca, NON pubblicato. Nessun pronostico operativo / nessuna evidenza di ROI.
> I dataset originali comprendono contenuti di terzi con diritti di
> redistribuzione da chiarire prima di qualsiasi uso commerciale.

## 1. Identita ATP/WTA: collegamento sicuro

Il nuovo feed pubblica identificativi giocatore opachi, non i numeri
dell'archivio Sackmann. Il nuovo file tennis_hf_crosswalk.py NON li
tratta come intercambiabili.

- 1.304 identita giocate nel feed, 1.042 nell'archivio storico 2025-26
- 878 corrispondenze univoche per cognome composto e iniziali
- 826 identita con conferma incrociata tramite partite storiche
- 7.366 partite in accordo su vincitore, avversario, punteggio e finestra
  temporale del torneo
- 52 identita soltanto nominali: escluse
- 4 collisioni tra identificativi: escluse
- 2.340 partite con entrambe le identita confermate, su 2.659
  giocate dopo il 1° giugno 2026 (87,99%)

Sackmann usa la DATA DI INIZIO DEL TORNEO: e' sbagliato confrontarla
con il giorno esatto del match nel nuovo feed. Le corrispondenze usano
finestra [-2,+21] giorni dal tournament start e punteggio verificato.

Report verificato:
https://github.com/Tano14i/tennis-oracle/actions/runs/42881720160

## 2. Confronto out-of-sample (finestra 2 giugno–8 novembre 2026)

Il modello di ricerca e' una Random Forest 250 alberi addestrata con
feature a disponibilita' temporale (nessun risultato futuro incluso).
Calibrazione successiva all'addestramento, ma PRIMA del test.

- Training: 5.636 match, fino al 22 febbraio 2026
- Calibrazione: 1.507 match, 23 febbraio–1° giugno 2026
- Holdout: 2.198 match, 2 giugno–8 novembre 2026
- Campione comune al mercato di chiusura: 2.197

| Indicatore | Tennis Oracle research | Probabilita' di chiusura |
|---|---:|---:|
| Accuracy sul campione comune | 61,994% | 66,682% |
| Brier (minore meglio) | 0,22637 | 0,20695 |
| Log loss (minore meglio) | 0,64337 | 0,59907 |

Le probabilita' di chiusura sono una **baseline retrospettiva**, NON
quote disponibili alla generazione di un segnale. Restano fuori dalle
feature e dalla calibrazione del modello. Nessun ROI/CLV calcolabile.

Report testato:
https://github.com/Tano14i/tennis-oracle/actions/runs/42882446503

**Decisione: niente deploy.** Il nuovo modello NON e' competitivo con
il mercato sul campione valutato. Il precedente backtest al 65,3% e'
su coorte e periodo differenti: NON confrontarlo come prova di miglioramento.

## 3. Diagnosi dati mancanti

Solo 392 incontri del holdout su 2.198 (17,83%) forniscono entrambe
le classifiche nel dataset. Sul sottoinsieme con rank disponibili
l'accuracy del ranking-only e' 63,52%.

E' stato implementato un riporto dei ranking (tennis_rank_history.py)
rigorosamente causale: puo' riutilizzare un rank realmente osservato
in una giornata PRECEDENTE, massimo 70 giorni di anzianita', senza
contaminare un'altra partita dello stesso giorno.

Esito esperimento: su 9.341 incontri idonei ci sono soltanto
784 campi ranking realmente osservati, su 18.682 campi totali.
Il riporto ha recuperato **ZERO** campi ulteriori. La copertura
insufficiente e' un vincolo della fonte, non si puo' risolvere
inventando ranking o appoggiandosi ai risultati futuri.

Report esplorativo POST-HOC (stesso test osservato, NON una nuova
validazione indipendente):
https://github.com/Tano14i/tennis-oracle/actions/runs/42883364157

## 4. Blocco rilascio e prossimo intervento

Il live Daily Picks richiede ancora un dataset aggiornato e autorizzato,
un addestramento su feature coerenti e quote realmente osservabili prima
della partita. Le quattro categorie rimangono nel branch PR#1,
**non pubblicate**. I nuovi script di ricerca non scrivono in models/,
non cambiano API e non dispiegano la webapp.

Priorita':
1. reperire un feed di ranking ATP/WTA *storico e point-in-time*, con
   termini di utilizzo verificati; oppure sperimentare rating Elo costruiti
   unicamente dai risultati antecedenti;
2. integrare solo dati con identita univoche, senza confondere i
   sample universi e senza prezzi di chiusura come feature;
3. predisporre la prossima finestra temporale intatta (dopo l'8
   novembre 2026) per non ottimizzare ripetutamente sullo stesso
   campione di verifica;
4. solo dopo, raccogliere quote prematch timestamped e fare paper
   tracking reale, confrontando le closing line per il medesimo evento;
5. risolvere licenze, controlli endpoint e token Telegram trapelato
   nello storico Git PRIMA di un deployment.

Suite automatica: 67 test passati dopo l'ultimo run di ricerca.


## 5. Elo cronologico per superficie — benchmark post-hoc

Implemented in tennis_elo.py and tennis_elo_research.py; no production integration.
Pre-match probability uses exclusively dates BEFORE each match date.
All fixtures on the same calendar day are scored before that day's Elo updates.
Separate ATP/WTA and Hard/Clay/Grass states; fixed rating scale 400, K global 28,
K surface 36, maximum blended surface weight 65%. No market odds as features.

Historical sample: 2,197 common June 2–November 8, 2026 fixtures:

| Predictor | Accuracy | Brier (lower better) | Log loss |
|---|---:|---:|---:|
| Random Forest previous research | 61.994% | 0.22637 | 0.64337 |
| Elo raw | 62.767% | 0.22163 | 0.63258 |
| Elo pretest calibrated | 62.995% | 0.22238 | 0.63410 |
| Closing market consensus | 66.682% | 0.20695 | 0.59907 |

Elo slightly outperformed Random Forest on this reused historical cohort but
underperformed the market. Logistic calibration improved accuracy but worsened
Brier and log loss against raw Elo. Neither accuracy nor closing comparison
establishes achievable betting ROI or model value against bookmakers.

CAUTION: This very same June-November 2026 holdout was inspected before
developing Elo. It is consequently POST-HOC DIAGNOSTIC, NOT an independent
prospective test of this model family. No parameter tuning on the inspected
sample is justified. The next untouched period must begin after Nov 8, 2026.

Result: https://github.com/Tano14i/tennis-oracle/actions/runs/42907317350
Unit test CI: 78 passed:
https://github.com/Tano14i/tennis-oracle/actions/runs/42907331514

Deployment remains blocked: provenance and commercial reuse rights for the
external data are not independently approved, live prematch odds are not
validated, and no future independent record has been demonstrated.
