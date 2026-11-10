"""Isolated research audit for a third-party HF tennis source: never trains models."""
from __future__ import annotations
import csv, io, math, json, urllib.request, hashlib
from datetime import datetime, timezone

NAMES = ("atp_matches_2025.csv","wta_matches_2025.csv",
         "atp_matches_2026.csv","wta_matches_2026.csv")
BASE = "https://huggingface.co/datasets/minmaxxxer/atp-wta-matches-with-odds/resolve/main/"
REQUIRED = {"match_id","start_time_utc","winner_id","loser_id","winner_name",
            "loser_name","score","winner_close_prob","winner_close_fair_odds",
            "winner_best_close_odds","loser_best_close_odds"}

def _date(x):
    try:
        d = datetime.fromisoformat(str(x).strip().replace("Z","+00:00"))
        return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)
    except (TypeError,ValueError):
        return None

def _float(x):
    try:
        v=float(x)
        return v if math.isfinite(v) else None
    except (TypeError,ValueError):
        return None

def audit_csv(content, now):
    if now.tzinfo is None:
        raise ValueError("UTC-aware timestamp required")
    reader=csv.DictReader(io.StringIO(content.lstrip("\ufeff")))
    missing=sorted(REQUIRED-set(reader.fieldnames or []))
    out={"rows":0,"missing_columns":missing,"duplicates":0,"invalid_players":0,
         "invalid_dates":0,"future_settled":0,"odds_mismatch":0,
         "closing_pairs":0,"latest_start_utc":None,"accepted":False}
    if missing:
        return out
    seen=set()
    latest=None
    for r in reader:
        out["rows"]+=1
        mid=(r.get("match_id") or "").strip()
        if not mid or mid in seen:
            out["duplicates"]+=1
        seen.add(mid)
        p=(r.get("winner_id") or "").strip()
        q=(r.get("loser_id") or "").strip()
        if not p or not q or p==q or not r.get("winner_name") or not r.get("loser_name"):
            out["invalid_players"]+=1
        dt=_date(r.get("start_time_utc"))
        if dt is None:
            out["invalid_dates"]+=1
        else:
            latest=max(latest,dt) if latest else dt
            if r.get("score") and dt>now:
                out["future_settled"]+=1
        probability=_float(r.get("winner_close_prob"))
        fair=_float(r.get("winner_close_fair_odds"))
        if probability is not None and fair and fair>1 and abs(probability-1/fair)>.08:
            out["odds_mismatch"]+=1
        a=_float(r.get("winner_best_close_odds"))
        b=_float(r.get("loser_best_close_odds"))
        if a and b and 1<a<150 and 1<b<150:
            out["closing_pairs"]+=1
    out["latest_start_utc"]=latest.isoformat() if latest else None
    out["accepted"]=out["rows"]>=100 and not any(out[k] for k in
        ("duplicates","invalid_players","invalid_dates","future_settled","odds_mismatch"))
    return out

def download(name):
    if name not in NAMES:
        raise ValueError("Filename not allowlisted")
    req=urllib.request.Request(BASE+name,headers={"User-Agent":"TennisOracleAudit/1.0"})
    with urllib.request.urlopen(req,timeout=35) as r:
        data=r.read(5_000_001)
    if len(data)>5_000_000:
        raise ValueError("File exceeds size limit")
    return data.decode("utf-8-sig"),hashlib.sha256(data).hexdigest()

def run(now=None):
    now=now or datetime.now(timezone.utc)
    report={"source":"minmaxxxer/atp-wta-matches-with-odds",
            "mode":"QUARANTINE_AUDIT_ONLY","published_license":"CC BY 4.0 (unverified claim)",
            "provenance_verified":False,"safe_to_merge_player_ids":False,"files":[]}
    known=set()
    cross_duplicates=0
    for name in NAMES:
        data,sha=download(name)
        result=audit_csv(data,now)
        result.update(filename=name,sha256=sha)
        for row in csv.DictReader(io.StringIO(data)):
            mid=(row.get("match_id") or "").strip()
            if mid in known:
                cross_duplicates+=1
            known.add(mid)
        report["files"].append(result)
    recent=[_date(f["latest_start_utc"]) for f in report["files"]
            if "_2026" in f["filename"] and f["latest_start_utc"]]
    latest=max(recent) if recent else None
    age=(now-latest).days if latest else None
    report["cross_file_duplicates"]=cross_duplicates
    report["latest_result_utc"]=latest.isoformat() if latest else None
    report["latest_age_days"]=age
    report["passed_automated"]=all(f["accepted"] for f in report["files"]) and (
        cross_duplicates==0 and age is not None and 0<=age<=21)
    report["decision"]="AUDIT_PASS_NOT_AUTHORIZED_TO_TRAIN" if report["passed_automated"] else "REJECT_SOURCE"
    return report

if __name__=="__main__":
    try:
        result=run()
    except Exception as exc:
        result={"mode":"QUARANTINE_AUDIT_ONLY","passed_automated":False,
                "decision":"FETCH_FAILURE","error":f"{type(exc).__name__}: {str(exc)[:250]}"}
    with open("hf_source_audit.json","w",encoding="utf-8") as f:
        json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result.get("passed_automated") else 1)
