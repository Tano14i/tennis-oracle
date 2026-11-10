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
