"""Chronological, point-in-time features for Tennis Oracle.

No match outcome, ace count, head-to-head or form from a match on date D
can affect a feature for ANY other match on date D. Results are applied
only once the entire UTC tournament date has been featurized.

This module does NOT train classifiers. It produces reproducible features
and as-of-latest snapshots for later model-fitting and prediction.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import isfinite
import numpy as np
import pandas as pd

from tennis_models import build_row, FEATURE_COLS_BASE, FEATURE_COLS_WINNER


def _n(value, default=None):
    try:
        value = float(value)
        return value if isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _ratio(a, b, default):
    a, b = _n(a), _n(b)
    if a is None or b is None or b <= 0:
        return default
    return a / b


def _mean(values, default):
    clean = [x for x in values if x is not None and isfinite(x)]
    return float(sum(clean) / len(clean)) if clean else default


@dataclass
class PlayerHistory:
    matches: int = 0
    wins: int = 0
    opponent_rank_sum: float = 0
    surface: dict = field(default_factory=dict)
    rolling: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_results: deque = field(default_factory=lambda: deque(maxlen=5))
    weighted_results: deque = field(default_factory=lambda: deque(maxlen=10))
    streak: int = 0
    rank: float = 300
    points: float = 0
    age: float = 25
    height: float = 185

    def snapshot(self):
        r = list(self.rolling)
        recent = list(self.recent_results)
        weighted = list(self.weighted_results)
        if weighted:
            weights = [w for _, w in weighted]
            weighted_rate = sum(result * weight for result, weight in weighted) / max(1e-9, sum(weights))
        else:
            weighted_rate = .5
        return {
            "p_matches": self.matches,
            "p_win_rate": self.wins / self.matches if self.matches else .5,
            "p_recent_wr": _mean(recent, .5),
            "p_weighted_wr": weighted_rate,
            "p_avg_opp_rank": self.opponent_rank_sum / self.matches if self.matches else 150,
            "p_streak": self.streak,
            "p_aces_avg": _mean([x[0] for x in r], 5),
            "p_df_avg": _mean([x[1] for x in r], 3),
            "p_1st_in": _mean([x[2] for x in r], .60),
            "p_1st_won": _mean([x[3] for x in r], .70),
            "p_2nd_won": _mean([x[4] for x in r], .50),
            "p_rank": self.rank, "p_pts": self.points,
            "p_age": self.age, "p_ht": self.height,
        }

    def surface_snapshot(self):
        out = {}
        for surface in ("Hard", "Clay", "Grass"):
            wins, games = self.surface.get(surface, (0, 0))
            out[f"wr_{surface.lower()}"] = wins / games if games else .5
            out[f"n_{surface.lower()}"] = games
        return out

    def observe(self, row, side, win, surface):
        self.matches += 1
        self.wins += int(win)
        opp_side = "loser" if side == "winner" else "winner"
        rank = _n(row.get(f"{side}_rank"), 300)
        opp_rank = _n(row.get(f"{opp_side}_rank"), 300)
        self.rank = rank
        self.points = _n(row.get(f"{side}_rank_points"), 0)
        self.age = _n(row.get(f"{side}_age"), 25)
        self.height = _n(row.get(f"{side}_ht"), 185)
        self.opponent_rank_sum += opp_rank

        wins, games = self.surface.get(surface, (0, 0))
        self.surface[surface] = (wins + int(win), games + 1)
        self.recent_results.append(int(win))
        weight = max(.01, (301 - min(max(opp_rank, 1), 300)) / 300)
        self.weighted_results.append((int(win), weight))
        self.streak = (max(0, self.streak) + 1) if win else (min(0, self.streak) - 1)

        prefix = "w_" if win else "l_"
        first_in = _n(row.get(prefix + "1stIn"))
        points = _n(row.get(prefix + "svpt"))
        second_pts = (points - first_in) if points is not None and first_in is not None else None
        self.rolling.append((
            _n(row.get(prefix + "ace")),
            _n(row.get(prefix + "df")),
            _ratio(first_in, points, None),
            _ratio(row.get(prefix + "1stWon"), first_in, None),
            _ratio(row.get(prefix + "2ndWon"), second_pts, None),
        ))


def build_temporal_frame(df):
    """Return time-valid per-match rows and latest feature snapshots.

    DataFrame must contain winner/loser ids, tour, tourney_date, and enriched
    targets. Matching tour is included in H2H/state keys to prevent ATP/WTA
    IDs from colliding. Each prediction uses ONLY earlier UTC dates.
    """
    if df.empty:
        raise ValueError("Empty dataset")
    needed = {"tourney_date", "winner_id", "loser_id", "tour"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Dataset missing columns: {needed-set(df.columns)}")
    frame = df.copy()
    frame["tourney_date"] = pd.to_datetime(frame["tourney_date"], errors="coerce")
    frame = frame.dropna(subset=["tourney_date", "winner_id", "loser_id", "tour"])
    if frame.empty:
        raise ValueError("No valid dated matches")
    frame = frame.sort_values("tourney_date", kind="mergesort").reset_index(drop=True)
    state = {}
    h2h = {}   # (tour, smaller_player_id, larger_player_id) -> [small wins, large wins]
    feat_base, feat_winner, y_winner, dates, rows = [], [], [], [], []

    for day, subset in frame.groupby(frame["tourney_date"].dt.normalize(), sort=True):
        matches = subset.to_dict("records")
        # Snapshot all participating players BEFORE applying any result on day D.
        today_features = []
        for row in matches:
            try:
                wid, lid = int(row["winner_id"]), int(row["loser_id"])
            except (TypeError, ValueError):
                continue
            if wid == lid:
                continue
            tour = str(row["tour"])
            surface = str(row.get("surface") or "Hard")
            p1_is_winner = wid < lid
            p1 = wid if p1_is_winner else lid
            p2 = lid if p1_is_winner else wid
            one = state.get((tour, wid), PlayerHistory())
            two = state.get((tour, lid), PlayerHistory())
            snapshots = {wid: one.snapshot(), lid: two.snapshot()}
            surfaces = {wid: one.surface_snapshot(), lid: two.surface_snapshot()}

            # Training uses deterministic player order, never 'winner first'.
            feat = build_row(row, snapshots, surfaces, flip=not p1_is_winner)
            key = (tour, min(wid, lid), max(wid, lid))
            previous = h2h.get(key, (0, 0))
            total = sum(previous)
            feat["h2h_total"] = total
            feat["h2h_p1_win_rate"] = previous[0] / total if total else .5
            feat_base.append({k: feat[k] for k in FEATURE_COLS_BASE})
            feat_winner.append({k: feat[k] for k in FEATURE_COLS_WINNER})
            y_winner.append(int(p1_is_winner))
            dates.append(day)
            rows.append(row)
            today_features.append((row, wid, lid, tour, surface))

        # Results become visible only for future dates, never same-day matches.
        for row, wid, lid, tour, surface in today_features:
            state.setdefault((tour, wid), PlayerHistory()).observe(row, "winner", True, surface)
            state.setdefault((tour, lid), PlayerHistory()).observe(row, "loser", False, surface)
            key = (tour, min(wid, lid), max(wid, lid))
            prev = h2h.get(key, (0, 0))
            h2h[key] = (
                prev[0] + int(wid == key[1]),
                prev[1] + int(wid == key[2]),
            )

    if not rows:
        raise ValueError("No distinct-player match rows")
    last_stats = {}
    last_surface = {}
    # These two unqualified maps are for legacy API player IDs.
    # The trainer must not merge collisions silently.
    duplicate_tour_ids = set()
    for (tour, pid), player in state.items():
        if pid in last_stats:
            duplicate_tour_ids.add(pid)
        else:
            last_stats[pid] = player.snapshot()
            last_surface[pid] = player.surface_snapshot()
    if duplicate_tour_ids:
        for pid in duplicate_tour_ids:
            last_stats.pop(pid, None)
            last_surface.pop(pid, None)

    raw = pd.DataFrame(rows).reset_index(drop=True)
    return {
        "raw": raw,
        "dates": pd.to_datetime(pd.Series(dates)),
        "X_base": pd.DataFrame(feat_base, columns=FEATURE_COLS_BASE).fillna(0),
        "X_winner": pd.DataFrame(feat_winner, columns=FEATURE_COLS_WINNER).fillna(0),
        "y_winner": np.asarray(y_winner, dtype=int),
        "player_stats": last_stats,
        "surface_stats": last_surface,
        "colliding_ids": sorted(duplicate_tour_ids),
    }


def temporal_model_sets(feature_pack, train_mask, calibration_mask, n_estimators=250):
    """Fit RF ONLY on earlier training dates, then sigmoid-calibrate later data.

    Masks must be disjoint and strictly time-separated. No random CV, no
    winner/loser duplicate rows and no reuse of holdout data for training.
    Returns estimators plus calibration-only diagnostics (not test scores).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.metrics import brier_score_loss, log_loss

    dates = feature_pack["dates"]
    train_mask = np.asarray(train_mask, dtype=bool)
    calibration_mask = np.asarray(calibration_mask, dtype=bool)
    if len(train_mask) != len(dates) or len(calibration_mask) != len(dates):
        raise ValueError("Masks must cover the entire temporal frame")
    if (train_mask & calibration_mask).any():
        raise ValueError("Training/calibration overlap")
    if not train_mask.any() or not calibration_mask.any():
        raise ValueError("Empty train or calibration window")
    if dates[train_mask].max() >= dates[calibration_mask].min():
        raise ValueError("Calibration must occur strictly AFTER training")

    from tennis_models import TARGET_COLS, ALL_MARKETS
    trained, diag = {}, {}
    for market in ALL_MARKETS:
        X = feature_pack["X_winner"] if market == "winner" else feature_pack["X_base"]
        if market == "winner":
            y = feature_pack["y_winner"]
            eligible = np.ones(len(y), dtype=bool)
        else:
            col = TARGET_COLS[market]
            if col not in feature_pack["raw"].columns:
                continue
            y_raw = pd.to_numeric(feature_pack["raw"][col], errors="coerce")
            y = y_raw.fillna(-1).to_numpy().astype(int)
            eligible = y_raw.notna().to_numpy()
            eligible &= feature_pack["raw"]["best_of"].eq(3).to_numpy()
            if market == "aces_over":
                for col2 in ("w_ace", "l_ace"):
                    if col2 not in feature_pack["raw"]:
                        eligible[:] = False
                    else:
                        eligible &= feature_pack["raw"][col2].notna().to_numpy()

        fit_idx, cal_idx = train_mask & eligible, calibration_mask & eligible
        if fit_idx.sum() < 200 or cal_idx.sum() < 80:
            continue
        if len(np.unique(y[fit_idx])) < 2 or len(np.unique(y[cal_idx])) < 2:
            continue
        forest = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=8,
            min_samples_leaf=10, random_state=42, n_jobs=-1,
        )
        forest.fit(X.loc[fit_idx], y[fit_idx])
        # FrozenEstimator does NOT refit the forest on calibration data.
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(forest), method="sigmoid"
        )
        calibrated.fit(X.loc[cal_idx], y[cal_idx])
        prob = calibrated.predict_proba(X.loc[cal_idx])[:, 1]
        trained[market] = calibrated
        diag[market] = {
            "train_samples": int(fit_idx.sum()),
            "calibration_samples": int(cal_idx.sum()),
            "calibration_brier": round(float(brier_score_loss(y[cal_idx], prob)), 5),
            "calibration_log_loss": round(float(log_loss(y[cal_idx], prob)), 5),
            "label_rate": round(float(y[cal_idx].mean()), 4),
        }
    return trained, diag


def chronological_masks(pack, cutoff, calibration_fraction=0.20):
    """Train/calibrate using dates strictly before cutoff; reserve future for test."""
    date = pack["dates"]
    pre = sorted(pd.to_datetime(date[date < pd.Timestamp(cutoff)]).dt.normalize().unique())
    if len(pre) < 10:
        raise ValueError("Insufficient separate dates")
    cut = pre[max(1, int(len(pre) * (1.0 - calibration_fraction)))]
    train = (date < cut).to_numpy()
    cal = ((date >= cut) & (date < pd.Timestamp(cutoff))).to_numpy()
    test = (date >= pd.Timestamp(cutoff)).to_numpy()
    return train, cal, test, str(pd.Timestamp(cut).date())
