"""
tennis_schedule.py
Recupera il palinsesto tennis del giorno da ESPN usando i groupings.
"""

import requests
from datetime import datetime, timezone

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

ESPN_ENDPOINTS = {
    "atp":        "ATP",
    "wta":        "WTA",
}

_player_cache = {}


def resolve_player_name(player_id, tour_slug):
    if player_id in _player_cache:
        return _player_cache[player_id]
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/tennis/" + tour_slug + "/athletes/" + str(player_id)
        r = requests.get(url, headers=HEADERS, timeout=8)
        if r.status_code == 200:
            data = r.json()
            name = data.get("athlete", {}).get("displayName", "")
            if name:
                _player_cache[player_id] = name
                return name
    except Exception:
        pass
    return ""


def _extract_name(competitor, tour_slug):
    athlete = competitor.get("athlete", {})
    if isinstance(athlete, dict):
        name = athlete.get("displayName") or athlete.get("fullName") or ""
        if name:
            return name
    team = competitor.get("team", {})
    if isinstance(team, dict):
        name = team.get("displayName") or team.get("name") or ""
        if name:
            return name
    name = competitor.get("displayName") or competitor.get("name") or ""
    if name:
        return name
    pid = competitor.get("id") or (team.get("id") if isinstance(team, dict) else None)
    if pid:
        resolved = resolve_player_name(str(pid).split("-")[0], tour_slug)
        if resolved:
            return resolved
    return ""


def _is_female_name(name):
    """Euristica semplice per rilevare nomi femminili nei risultati ESPN."""
    female_indicators = [
        "kessler", "zarazua", "shnaider", "fernandez", "parks",
        "muchova", "zakharova", "swiatek", "gauff", "sabalenka",
        "rybakina", "pegula", "keys", "navarro", "andreeva",
        "paolini", "jabeur", "ostapenko", "azarenka", "kerber",
        "halep", "kvitova", "pliskova", "kontaveit", "vondrousova",
    ]
    n = name.lower()
    return any(w in n for w in female_indicators)


def get_matches_today():
    """
    Recupera le partite tennis di oggi da ESPN.
    Usa endpoint ATP e WTA separati e assegna il tour corretto.
    """
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    today_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    matches = []

    for tour_slug, tour_label in ESPN_ENDPOINTS.items():
        try:
            url = "https://site.api.espn.com/apis/site/v2/sports/tennis/" + tour_slug + "/scoreboard?dates=" + today
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code != 200:
                continue

            data = r.json()
            events = data.get("events", [])

            for ev in events:
                tournament = ev.get("name", "")
                groupings = ev.get("groupings", [])

                for group in groupings:
                    competitions = group.get("competitions", [])
                    for comp in competitions:
                        comp_date = comp.get("date", "")[:10]
                        status_type = comp.get("status", {}).get("type", {})
                        status_state = status_type.get("state", "")
                        status_desc = status_type.get("description", "")

                        # Includi scheduled E in-progress E recent (partite di oggi)
                        if comp_date != today_date:
                            continue

                        competitors = comp.get("competitors", [])
                        if len(competitors) < 2:
                            continue

                        p1 = _extract_name(competitors[0], tour_slug)
                        p2 = _extract_name(competitors[1], tour_slug)
                        if not p1 or not p2:
                            continue

                        # Determina tour reale dal genere dei giocatori
                        # ESPN a volte mescola ATP/WTA nello stesso endpoint
                        actual_tour = tour_label
                        if tour_slug == "atp" and (_is_female_name(p1) or _is_female_name(p2)):
                            actual_tour = "WTA"

                        round_obj = comp.get("round", {})
                        if isinstance(round_obj, dict):
                            round_str = round_obj.get("displayName", "")
                        else:
                            round_str = str(round_obj) if round_obj else ""

                        if not round_str:
                            notes = comp.get("notes", [])
                            round_str = notes[0].get("text", "")[:30] if notes else ""

                        match_time = ""
                        try:
                            dt = datetime.fromisoformat(comp.get("date", "").replace("Z", "+00:00"))
                            match_time = dt.strftime("%H:%M UTC")
                        except Exception:
                            pass

                        surface = detect_surface(tournament)

                        matches.append({
                            "p1": p1,
                            "p2": p2,
                            "tournament": tournament,
                            "surface": surface,
                            "tour": actual_tour,
                            "round": round_str,
                            "time": match_time,
                            "status": status_desc,
                            "state": status_state,
                            "id": comp.get("id", ""),
                        })

        except Exception as e:
            print("Errore ESPN " + tour_slug + ": " + str(e))
            continue

    # Deduplicazione per coppia giocatori
    seen = set()
    unique = []
    for m in matches:
        key = tuple(sorted([m["p1"].lower(), m["p2"].lower()]))
        if key not in seen:
            seen.add(key)
            unique.append(m)

    # Ordina per orario
    unique.sort(key=lambda x: x.get("time", ""))
    return unique


def detect_surface(text):
    text = text.lower()
    if any(w in text for w in ["clay", "roland", "paris", "barcelona", "madrid",
                                "rome", "monte-carlo", "hamburg", "lyon", "geneva",
                                "estoril", "bucharest", "munich", "belgrade",
                                "marrakech", "houston", "bogota"]):
        return "Clay"
    if any(w in text for w in ["grass", "wimbledon", "halle", "queen", "eastbourne",
                                "hertogenbosch", "nottingham", "bad homburg", "mallorca"]):
        return "Grass"
    return "Hard"


def detect_tour(text):
    text = text.lower()
    if "wta" in text or "women" in text:
        return "WTA"
    if "itf" in text:
        return "ITF"
    if "challenger" in text:
        return "Challenger"
    return "ATP"


def filter_matches(matches, tours=None):
    if not tours:
        return matches
    return [m for m in matches if m.get("tour") in tours]


def group_by_tournament(matches):
    grouped = {}
    for m in matches:
        key = m["tour"] + " — " + m["tournament"]
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(m)
    return grouped


if __name__ == "__main__":
    print("Test recupero palinsesto ESPN...")
    matches = get_matches_today()
    print("Partite trovate: " + str(len(matches)))
    grouped = group_by_tournament(matches)
    for tournament, ms in grouped.items():
        print("\n" + tournament + " (" + str(len(ms)) + " partite)")
        for m in ms:
            print("  " + m["tour"] + " | " + m["p1"] + " vs " + m["p2"] + " | " + m["round"] + " | " + m["time"])
