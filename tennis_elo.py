"""Research-only pre-match Elo ratings, general and surface-specific.

Point-in-time guarantee: rating features for ALL matches on date D use
ONLY results dated strictly before D. Same-day matches update the ratings
as a single batch AFTER every prediction on that date has been frozen.

No price, odds, bookmaker, ranking, result after D or future feature.
Fixed hyperparameters are declared here, never selected on test results.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

START = 1500.0
GLOBAL_K = 28.0
SURFACE_K = 36.0
ELO_SCALE = 400.0
SURFACE_WEIGHT_MAX = 0.65
SURFACE_PRIOR_MATCHES = 12.0
SURFACES = {"Hard", "Clay", "Grass"}


def win_probability(p1_rating, p2_rating):
    """Elo logistic probability for first player."""
    return 1.0 / (1.0 + 10.0 ** ((p2_rating - p1_rating) / ELO_SCALE))


def _safe_player_id(value):
    try:
        if pd.isna(value):
            raise ValueError("Missing player")
        pid = int(value)
        if pid <= 0:
            raise ValueError("Invalid player")
        return pid
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Player identifiers must be positive integer IDs")


def pre_match_elo(frame):
    """Compute one independent pre-match Elo probability per historical match.

    Accepts winner/loser ordered outcomes but ALWAYS predicts the
    lower numeric ID first (fixed outcome-independent orientation).
    Expects tour, surface, winner_id, loser_id, tourney_date, source_id.

    Returns DataFrame with original row index preserved and pre-match
    columns. Unknown, unfinished and invalid matches are refused.
    """
    required = {"tour", "surface", "winner_id", "loser_id",
                "tourney_date", "source_id"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing Elo fields: {required - set(frame.columns)}")
    if frame.empty:
        raise ValueError("No historical matches")
    df = frame.copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    if df["tourney_date"].isna().any():
        raise ValueError("Unknown match dates")
    if df["source_id"].isna().any() or df["source_id"].duplicated().any():
        raise ValueError("Each historical match must have a unique source_id")
    if not df["surface"].isin(SURFACES).all():
        raise ValueError("Unrecognized surface; do not silently assume Hard")
    if df["tour"].isna().any():
        raise ValueError("Each match must belong to a tour")

    global_rating = defaultdict(lambda: START)
    surface_rating = defaultdict(lambda: START)
    global_count = Counter()
    surface_count = Counter()
    out = {}

    # Always sort calendar days, irrespective of CSV row order.
    for day, group in df.groupby(df["tourney_date"].dt.normalize(), sort=True):
        pending = []
        for idx, match in group.iterrows():
            wid = _safe_player_id(match["winner_id"])
            lid = _safe_player_id(match["loser_id"])
            if wid == lid:
                raise ValueError("Winner and loser cannot be identical")
            p1, p2 = min(wid, lid), max(wid, lid)
            tour, surface = str(match["tour"]), str(match["surface"])
            a, b = (tour, p1), (tour, p2)
            sa, sb = (tour, surface, p1), (tour, surface, p2)
            gp = win_probability(global_rating[a], global_rating[b])
            sp = win_probability(surface_rating[sa], surface_rating[sb])
            played = min(surface_count[sa], surface_count[sb])
            w = SURFACE_WEIGHT_MAX * played / (played + SURFACE_PRIOR_MATCHES)
            probability = (1.0 - w) * gp + w * sp
            result = int(p1 == wid)
            out[idx] = {
                "source_id": match["source_id"],
                "tour": tour,
                "surface": surface,
                "date": day,
                "p1_id": p1,
                "p2_id": p2,
                "y_p1_wins": result,
                "elo_p1_probability": probability,
                "elo_global_probability": gp,
                "elo_surface_probability": sp,
                "elo_surface_weight": w,
                "p1_global_matches_before": global_count[a],
                "p2_global_matches_before": global_count[b],
                "p1_surface_matches_before": surface_count[sa],
                "p2_surface_matches_before": surface_count[sb],
            }
            # Freeze all expectations for this entire day.
            pending.append((a, b, sa, sb, result, gp, sp))
        for a, b, sa, sb, result, gp, sp in pending:
            global_rating[a] += GLOBAL_K * (result - gp)
            global_rating[b] += GLOBAL_K * ((1 - result) - (1 - gp))
            surface_rating[sa] += SURFACE_K * (result - sp)
            surface_rating[sb] += SURFACE_K * ((1 - result) - (1 - sp))
            global_count[a] += 1
            global_count[b] += 1
            surface_count[sa] += 1
            surface_count[sb] += 1

    # Preserve original order, and use values from prior-day only.
    return pd.DataFrame.from_dict(out, orient="index").reindex(df.index)


def elo_configuration():
    return {
        "start_rating": START,
        "global_k": GLOBAL_K,
        "surface_k": SURFACE_K,
        "scale": ELO_SCALE,
        "max_surface_weight": SURFACE_WEIGHT_MAX,
        "surface_prior_matches": SURFACE_PRIOR_MATCHES,
        "same_day_results_used_for_prematch": False,
        "parameters_tuned_on_2026_test": False,
    }
