---
title: Tennis Oracle API
emoji: 🎾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Tennis Oracle 2.0 — Experimental Daily Picks

Research-only tennis predictions from point-in-time historical
match statistics, calibration and verified market quote comparisons.

Four daily categories: Singola (>=1.40), 3x3 (3 events around 3.00),
Multipla (>=5.00), Bomba (>=7.00). Combined odds are multiplied,
never summed. Absence of suitable data returns no selection.

## Backend endpoints

- GET /health — model load status
- POST /analyze — per-match prediction/explanation
- GET /daily_picks — four source-verified pre-match research categories
- GET /daily_picks/history — optional SQLite paper ledger

Training: python tennis_trainer.py
Independent chronological backtest: python tennis_backtest.py
CI: python -m unittest discover -s tests -p "test_*.py" -v

The new Daily Picks route requires a temporal-v2 manifest, recent
historical data and market quotes. No actual service deployment is
performed by this draft branch.

First held-out evaluation (50 trees, 41,100 test matches):
Winner 65.21% accuracy vs ranking-only 62.54%; Aces over 10.5 76.99%.
No trustworthy ROI without timestamped historical bookmaker quotes.

See BACKTEST_V2_REPORT.md and DAILY_PICKS_V2.md for evidence and caveats.

## Safety and licensing

This repository uses a Sackmann archive (CC BY-NC-SA 4.0):
non-commercial research use only. Archive freshness is NOT guaranteed.
Revoke any Telegram token previously published in Git history with
BotFather; configure new secrets through environment variables.
No proven betting edge or guaranteed results; no paid services enabled.
