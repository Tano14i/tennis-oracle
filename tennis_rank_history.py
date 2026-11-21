"""Point-in-time tennis ranking carry-forward for sparse historical feeds.

A missing pre-match rank may use only the last *observed* rank on a prior
calendar date, up to max_age_days. Matches on the SAME day cannot supply
one another's rank. An imputed rank is never treated as a new observation.

This module is for offline experimental preprocessing. It makes no calls to
bookmakers, results APIs, prediction services or any network connection.
"""
from __future__ import annotations

from collections import defaultdict
import math
import pandas as pd


def _rank(value):
    try:
        x = float(value)
        return x if math.isfinite(x) and 0 < x <= 5000 else None
    except (TypeError, ValueError):
        return None


def carry_forward_prior_ranks(frame, max_age_days=70):
    """Fill missing winner/loser ranks using prior-date observations only.

    Args:
       frame: DataFrame with tour, tourney_date, winner_id, loser_id,
              winner_rank, loser_rank.
       max_age_days: upper bound on previous observation staleness.
    Returns:
       copied DataFrame and report. No mutation of caller's data.

    Same-day results are never used as rank inputs for another match that
    day, even if the data arrives in different row order.
    """
    if max_age_days < 1:
        raise ValueError("max_age_days must be positive")
    columns = {"tour", "tourney_date", "winner_id", "loser_id",
               "winner_rank", "loser_rank"}
    if not columns.issubset(frame.columns):
        raise ValueError(f"Missing ranking history columns: {columns - set(frame.columns)}")
    output = frame.copy().reset_index(drop=True)
    output["tourney_date"] = pd.to_datetime(output["tourney_date"], errors="coerce")
    if output["tourney_date"].isna().any():
        raise ValueError("Unknown match dates cannot be imputed safely")
    dates = output["tourney_date"].dt.normalize()
    last_observed = {}  # (tour, player_id) -> (rank, date_of_real_observation)
    stats = {
        "rows": len(output), "original_known_rank_fields": 0,
        "filled_from_prior_rank": 0, "not_filled": 0,
        "source_rank_conflicts_same_day": 0,
        "max_age_days": int(max_age_days),
    }
    for day, indices in output.groupby(dates, sort=True).groups.items():
        daily = list(indices)
        seen_today = defaultdict(set)
        # Construct today's *features* from prior-date state first.
        for idx in daily:
            row = output.loc[idx]
            tour = str(row["tour"])
            for side in ("winner", "loser"):
                column = side + "_rank"
                pid = row[side + "_id"]
                key = (tour, str(pid))
                found = _rank(row[column])
                if found is not None:
                    stats["original_known_rank_fields"] += 1
                    seen_today[key].add(found)
                    continue
                previous = last_observed.get(key)
                if previous and 0 <= (day - previous[1]).days <= max_age_days:
                    output.at[idx, column] = previous[0]
                    stats["filled_from_prior_rank"] += 1
                else:
                    stats["not_filled"] += 1
        # Only *genuinely observed* new rankings may update future matches.
        for key, reported_ranks in seen_today.items():
            if len(reported_ranks) == 1:
                last_observed[key] = (next(iter(reported_ranks)), day)
            else:
                stats["source_rank_conflicts_same_day"] += 1
    return output, stats
