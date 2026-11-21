"""Conservative ATP/WTA player-ID crosswalk between two third-party archives.

RESEARCH ONLY. No fuzzy matches, no modification of production model inputs.
A plausible same-name mapping is NOT approved until a historical fixture with
both players, winner, score and tournament-date window confirms the identity.

Jeff Sackmann tourney_date is a *tournament start*, not the match start.
New feed start_time_utc is the actual match start: NEVER exact-date join.

The new provider's opaque IDs remain separate from integer historical IDs.
This tool must not be used to authorize upstream licenses or live picks.
"""
from __future__ import annotations

import csv
import io
import json
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from tennis_hf_audit import NAMES, download as download_hf, audit_csv, _date as utc_date

LEGACY_BASE = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main"
LEGACY_FILES = (
    ("ATP", "atp/atp_matches_2025.csv"),
    ("ATP", "atp/atp_matches_2026.csv"),
    ("WTA", "wta/wta_matches_2025.csv"),
    ("WTA", "wta/wta_matches_2026.csv"),
)

def clean(value):
    ascii_text = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = "".join(c for c in ascii_text if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text.lower()))

def source_signature(name):
    """Surname and one or more given-name initials, e.g. De Minaur A."""
    raw = str(name or "").strip()
    match = re.fullmatch(r"(.+?)\s+((?:[A-Za-z]\.\s*){1,4})", raw)
    if not match:
        return None
    surname = clean(match.group(1))
    initials = "".join(re.findall(r"[A-Za-z]", match.group(2))).lower()
    return (surname, initials) if surname and initials else None

def historical_signatures(name):
    """Enumerate safe suffix surname candidates without guessing surname parts."""
    tokens = clean(name).split()
    if len(tokens) < 2:
        return set()
    return {
        (" ".join(tokens[i:]), "".join(token[0] for token in tokens[:i]))
        for i in range(1, len(tokens)) if all(tokens[:i])
    }

def score_key(raw):
    """Score sets only. Exclude incomplete/retires to prevent false evidence."""
    score = str(raw or "").strip()
    if not score or re.search(r"\b(RET|W/O|DEF|ABN|ABD)\b", score, re.I):
        return None
    tokens = score.split()
    if len(tokens) < 2:
        return None
    out = []
    for token in tokens:
        m = re.fullmatch(r"(\d{1,2})-(\d{1,2})(?:\(\d{1,2}\))?", token)
        if not m:
            return None
        out.append((int(m.group(1)), int(m.group(2))))
    return tuple(out)

def legacy_day(raw):
    s = str(raw or "").strip()
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        return None

def historic_id(value):
    try:
        n = int(value)
        return n if n > 0 else None
    except (ValueError, TypeError):
        return None

def parse_csv(text):
    return list(csv.DictReader(io.StringIO(text.lstrip("\ufeff"))))

def build_crosswalk(hf_by_tour, legacy_by_tour, now=None):
    """Return conservative verified mapping and an inspectable diagnostic report.

    The inputs are lists of CSV dictionaries grouped by ATP/WTA. Only a
    unique surname+initials name match AND a separate independent overlap
    match can be verified. None of the generated output is training data.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Timezone-aware UTC audit time required")

    source_names = defaultdict(set)
    source_rows = []
    legacy_sigs = defaultdict(set)
    legacy_rows = []
    legacy_identity_names = defaultdict(set)
    for tour in ("ATP", "WTA"):
        for row in hf_by_tour.get(tour, []):
            start = utc_date(row.get("start_time_utc"))
            if start is None or start > now or not row.get("score"):
                continue
            source_rows.append((tour, row, start.date()))
            for side in ("winner", "loser"):
                pid = str(row.get(side + "_id") or "").strip()
                nm = str(row.get(side + "_name") or "").strip()
                if pid and nm:
                    source_names[(tour, pid)].add(nm)
        for row in legacy_by_tour.get(tour, []):
            w, l = historic_id(row.get("winner_id")), historic_id(row.get("loser_id"))
            if w is None or l is None or w == l:
                continue
            legacy_rows.append((tour, row))
            for side, pid in (("winner", w), ("loser", l)):
                nm = str(row.get(side + "_name") or "").strip()
                if nm:
                    legacy_identity_names[(tour, pid)].add(nm)
    for (tour, lid), names in legacy_identity_names.items():
        for name in names:
            for signature in historical_signatures(name):
                legacy_sigs[(tour, signature)].add(lid)

    candidates = {}
    statuses = Counter()
    for source_key, names in source_names.items():
        tour, source_pid = source_key
        choices = []
        for name in names:
            signature = source_signature(name)
            choices.append(legacy_sigs.get((tour, signature), set()) if signature else set())
        common = set.intersection(*choices) if choices else set()
        if len(common) == 1:
            candidates[source_key] = next(iter(common))
            statuses["unique_name_candidate"] += 1
        elif not common:
            statuses["unmatched_or_format"] += 1
        else:
            statuses["ambiguous_name"] += 1

    # Enforce bidirectional one-to-one identity across both seasons.
    reverse = defaultdict(list)
    for source_key, legacy_id in candidates.items():
        reverse[(source_key[0], legacy_id)].append(source_key)
    conflicts = {source_key for sources in reverse.values() if len(sources) > 1
                 for source_key in sources}
    for key in conflicts:
        candidates.pop(key, None)
    statuses["conflicting_source_ids"] = len(conflicts)

    historic_matches = defaultdict(list)
    for tour, row in legacy_rows:
        wid, lid = historic_id(row.get("winner_id")), historic_id(row.get("loser_id"))
        score = score_key(row.get("score"))
        start = legacy_day(row.get("tourney_date"))
        if wid and lid and score and start:
            historic_matches[(tour, wid, lid, score)].append(start)

    verified_keys = set()
    match_examples = []
    match_count = 0
    ambiguous_fixture_count = 0
    overlapping_possible = 0
    for tour, row, played in source_rows:
        winner_key = (tour, str(row.get("winner_id") or "").strip())
        loser_key = (tour, str(row.get("loser_id") or "").strip())
        if winner_key not in candidates or loser_key not in candidates:
            continue
        # Keep possible crosswalk coverage separate from historical validation.
        overlapping_possible += 1
        score = score_key(row.get("score"))
        if not score:
            continue
        dates = historic_matches.get((
            tour, candidates[winner_key], candidates[loser_key], score
        ), [])
        # Tournament start is usually 0..14 days before a match, occasionally
        # within two days after a provider start due to timezone/corrections.
        evidence = [d for d in dates if -2 <= (played - d).days <= 21]
        if len(evidence) == 1:
            match_count += 1
            verified_keys.update((winner_key, loser_key))
            if len(match_examples) < 6:
                match_examples.append({
                    "tour": tour, "winner": row.get("winner_name"),
                    "loser": row.get("loser_name"), "score": row.get("score"),
                    "played_date": played.isoformat(),
                    "historical_tournament_start": evidence[0].isoformat(),
                })
        elif len(evidence) > 1:
            ambiguous_fixture_count += 1

    verified = {k: candidates[k] for k in verified_keys}
    named_only = len(candidates) - len(verified)
    coverage = defaultdict(lambda: {"matches": 0, "both_name_candidates": 0,
                                    "both_overlap_verified": 0})
    cutoff = date(2026, 6, 2)
    for tour, row, played in source_rows:
        if played < cutoff or played.year != 2026:
            continue
        group = coverage[tour]
        group["matches"] += 1
        wk = (tour, str(row.get("winner_id") or "").strip())
        lk = (tour, str(row.get("loser_id") or "").strip())
        if wk in candidates and lk in candidates:
            group["both_name_candidates"] += 1
        if wk in verified and lk in verified:
            group["both_overlap_verified"] += 1

    return {
        "type": "RESEARCH_CROSSWALK_ONLY",
        "checked_at_utc": now.isoformat(),
        "source_players_seen": len(source_names),
        "historical_players_seen": len(legacy_identity_names),
        "name_candidates_after_one_to_one": len(candidates),
        "overlap_verified_players": len(verified),
        "name_only_unverified_players": named_only,
        "name_conflicts_removed": len(conflicts),
        "source_matches_with_both_candidates": overlapping_possible,
        "independent_exact_fixture_overlaps": match_count,
        "ambiguous_fixture_overlap": ambiguous_fixture_count,
        "after_2026_06_01": dict(coverage),
        "diagnostic_statuses": dict(statuses),
        "evidence_examples": match_examples,
        "verified_crosswalk": {
            tour + ":" + sid: {"tour": tour, "source_id": sid, "legacy_id": lid}
            for (tour, sid), lid in sorted(verified.items())
        },
        "approved_for_model_training": False,
        "license_provenance_verified": False,
        "reason": "Unverified redistribution rights; name-only identities not usable.",
    }

def _download_legacy(path):
    if path not in {item[1] for item in LEGACY_FILES}:
        raise ValueError("Historical path is not allowlisted")
    req = urllib.request.Request(
        LEGACY_BASE + "/" + path,
        headers={"User-Agent": "TennisOracleCrosswalkResearch/1.0"}
    )
    with urllib.request.urlopen(req, timeout=40) as response:
        data = response.read(10_000_001)
    if len(data) > 10_000_000:
        raise ValueError("Historical CSV exceeds bounded download")
    return data.decode("utf-8-sig")

def run(now=None):
    now = now or datetime.now(timezone.utc)
    hf_by_tour = {"ATP": [], "WTA": []}
    legacy_by_tour = {"ATP": [], "WTA": []}
    source_integrity = {}
    for filename in NAMES:
        text, sha = download_hf(filename)
        audit = audit_csv(text, now)
        if not audit["accepted"]:
            raise ValueError("Source audit failed for " + filename)
        source_integrity[filename] = sha
        tour = "ATP" if filename.startswith("atp_") else "WTA"
        hf_by_tour[tour].extend(parse_csv(text))
    for tour, path in LEGACY_FILES:
        legacy_by_tour[tour].extend(parse_csv(_download_legacy(path)))
    result = build_crosswalk(hf_by_tour, legacy_by_tour, now)
    result["source_sha256_by_filename"] = source_integrity
    # Flat crosswalk output deliberately separated from training artifacts.
    mapping = result.pop("verified_crosswalk")
    with open("hf_verified_crosswalk_research.json", "w", encoding="utf-8") as output:
        json.dump({"type": "RESEARCH_DO_NOT_DEPLOY", "mapping": mapping},
                  output, indent=2, ensure_ascii=False)
    with open("hf_crosswalk_report.json", "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if len(mapping) < 100 or result["independent_exact_fixture_overlaps"] < 100:
        raise ValueError("Crosswalk evidence insufficient: quarantined.")
    return result

if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print("CROSSWALK QUARANTINED:", type(exc).__name__, str(exc)[:250])
        raise SystemExit(1) from exc
