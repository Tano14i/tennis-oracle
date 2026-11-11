# Tennis Oracle 2.0 — Historical holdout report

Source: https://github.com/Tano14i/tennis-oracle/actions/runs/42783695638

Date: 2026-11-10 UTC. This was a **50-tree research smoke run**,
not the final production-size model. Underlying data are from an
archival noncommercial (CC BY-NC-SA) Sackmann mirror.

## Data windows

- 61,915 training matches, through November 21, 2022
- 15,990 probability calibration matches, Nov 22, 2022–Nov 27, 2023
- 41,100 untouched test matches, Jan 1, 2024–June 1, 2026

All features reflect strictly earlier match dates. Calibration and test
dates are not part of model training. There are NO bookmaker price
snapshots, so ROI and profitability cannot be assessed.

## Held-out metrics

| Market | Tested | Accuracy | Brier | Pretest-frequency baseline Brier |
| --- | ---: | ---: | ---: | ---: |
| Winner | 41,100 | 65.21% | 0.21527 | 0.24990 |
| Aces over 10.5 | 39,150 | 76.99% | 0.15685 | 0.20767 |
| Games over 22.5 | 39,852 | 58.67% | 0.23949 | 0.24301 |
| Both win set | 39,852 | 65.68% | 0.22399 | 0.22537 |
| Sets over 2.5 | 39,852 | 65.69% | 0.22404 | 0.22530 |
| At least one tiebreak | 39,852 | 69.69% | 0.20606 | 0.21182 |

Winner ranking-only baseline accuracy: 62.54%; ML +2.67 pp.
Aces positive rate: 29.43%; always-negative accuracy ~70.57%.

## Interpretation

These numbers are encouraging for winner and aces, but do not show
a profitable advantage over bookmakers. Gains for games/sets are small.
Current archived results end on 2026-06-01; stale models must not
produce November 2026 recommendations. No financial ROI is invented.

Needed: size-250 model replication, year-by-year stability, independent
quote snapshots, properly licensed fresh results, and paper settlement.


## Full 250-tree holdout, verified GitHub Actions run

Execution: https://github.com/Tano14i/tennis-oracle/actions/runs/42784401193

| Market | Evaluated | Accuracy | Brier |
| --- | ---: | ---: | ---: |
| Winner | 41,100 | 65.30% | 0.21515 |
| Aces over 10.5 | 39,150 | 77.06% | 0.15655 |
| Games over 22.5 | 39,852 | 58.70% | 0.23937 |
| Both players win set | 39,852 | 65.69% | 0.22382 |
| Sets over 2.5 | 39,852 | 65.72% | 0.22378 |
| At least one tiebreak | 39,852 | 69.70% | 0.20594 |

## Results by tour and calendar year

Execution: https://github.com/Tano14i/tennis-oracle/actions/runs/42784958627

Winner accuracy by year: 2024 **64.67%** (16,650 matches);
2025 **65.08%** (16,299); 2026 to June **67.00%** (8,151).
Winner by tour: ATP **65.76%** (7,188), Challenger **65.31%**
(27,370), WTA **64.72%** (6,542).

Aces over 10.5 accuracy by tour: ATP **69.25%**, Challenger
**76.10%**, WTA **88.12%**. Crucial: WTA aces-over event occurs
on only **12.15%** of evaluated matches, so always choosing under
would achieve about **87.85%** accuracy. Accordingly, a high WTA
ace accuracy is NOT evidence of a large improvement over a naive strategy.

Games Over 22.5 accuracy by tour: ATP **55.61%**, Challenger
**58.71%**, WTA **61.49%**. Overall the Brier improvement versus
pre-test prevalence is small. No odds or ROI were used to compute
these metrics; do not use them as evidence of profitable betting.

## Data-integrity warning: alternative recent feed QUARANTINED

Candidate CC BY 4.0 seasonal CSV from Valuebetennis provides recent
matches and bookmaker odds, but automated audit at 2026-11-10 08:50 UTC
found **4 rows labeled as completed matches with final scores dated
later the same day (18:00–18:30 UTC)**. Such future-dated settlements
contradict the provider's own 'only settled matches' description.
Action result: https://github.com/Tano14i/tennis-oracle/actions/runs/42785830368

**Do not merge or ingest this feed** until the publisher explains the
timestamps and records are independently validated. The research
mirror in use for historical evaluation remains stale at June 2026.
