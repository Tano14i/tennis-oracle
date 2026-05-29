---
title: Tennis Oracle API
emoji: 🎾
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# Tennis Oracle API

Backend Flask per Tennis Oracle — analisi ML su 6 mercati tennis (ATP/WTA).

## Endpoints

- `GET /health` — stato API e numero giocatori nel dataset
- `POST /analyze` — analisi ML per una partita

### Esempio /analyze

```json
{
  "p1": "Sinner",
  "p2": "Alcaraz",
  "surface": "Clay",
  "tour": "ATP",
  "round": "SF",
  "best_of": 3
}
```

## Mercati

| Mercato | Lift |
|---|---|
| Winner | +0.054 |
| Both set | +0.016 |
| Games over 22.5 | +0.053 |
| Aces over 10.5 | +0.087 |
