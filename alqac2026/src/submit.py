"""
Leaderboard submission client for ALQAC 2026.

Constraints stated by the organizers:
  - API rate limit: 1 request / 5 seconds per team.
  - Leaderboard submissions: max 3 / day per team.
  - Auth via the team secret token (kept in .env, NEVER committed).

The EXACT endpoint path and JSON payload schema are on the api-docs page
(https://alqac2026-leaderboard.ngrok.app/api-docs). Confirm them, then fill in
ENDPOINT and the payload shape below. This file does not hardcode any token.
"""
from __future__ import annotations
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

RATE_LIMIT_SECONDS = 5          # 1 request / 5s
DAILY_SUBMISSION_LIMIT = 3      # 3 submissions / day
_COUNTER = Path(".submit_count.json")  # local guard, git-ignored


def _check_daily_limit() -> int:
    today = date.today().isoformat()
    data = {}
    if _COUNTER.exists():
        data = json.loads(_COUNTER.read_text())
    used = data.get(today, 0)
    if used >= DAILY_SUBMISSION_LIMIT:
        sys.exit(f"Daily submission limit reached ({used}/{DAILY_SUBMISSION_LIMIT}). Try tomorrow.")
    return used


def _bump_daily(used: int):
    today = date.today().isoformat()
    _COUNTER.write_text(json.dumps({today: used + 1}))


def submit(predictions_path: str, dry_run: bool = True):
    load_dotenv()
    token = os.getenv("ALQAC_TEAM_TOKEN")
    base = os.getenv("ALQAC_API_BASE", "").rstrip("/")
    if not token or "xxxx" in token:
        sys.exit("Set ALQAC_TEAM_TOKEN in .env (do NOT commit it).")

    preds = json.loads(Path(predictions_path).read_text(encoding="utf-8"))

    # Each entry carries the outcome label AND the gathered case_evidence (chunk_ids):
    #   {"case_id": "...", "label": "A_WIN", "case_evidence": ["..._chunk_2", ...]}
    # case_evidence feeds the 20% Penalized Case Recall score.
    #
    # NOTE: the EXACT leaderboard submission method is NOT in the retrieval API docs.
    # Confirm on the dashboard (/about). Two possibilities:
    #   (a) Web upload of this JSON on https://alqac2026-leaderboard.ngrok.app, or
    #   (b) an API POST -> fill ENDPOINT + payload + auth below.
    ENDPOINT = f"{base}/submit"          # <-- CONFIRM on dashboard
    headers = {"X-API-Key": token, "Content-Type": "application/json"}  # <-- CONFIRM auth
    payload = {"results": preds}         # <-- CONFIRM key name & shape

    if dry_run:
        print(f"[dry-run] {len(preds)} entries; sample:")
        print(json.dumps(preds[:2], ensure_ascii=False, indent=2))
        miss = [p["case_id"] for p in preds if not p.get("case_evidence")]
        if miss:
            print(f"[warn] {len(miss)} entries have NO case_evidence (recall=0 for those).")
        print(f"[dry-run] would POST to {ENDPOINT} — CONFIRM this is correct!")
        return

    used = _check_daily_limit()
    time.sleep(RATE_LIMIT_SECONDS)       # respect 1 req / 5s
    r = requests.post(ENDPOINT, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    _bump_daily(used)
    print("Submitted. Server response:", r.text[:500])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", help="path to predictions JSON")
    ap.add_argument("--go", action="store_true", help="actually submit (omit for dry-run)")
    a = ap.parse_args()
    submit(a.predictions, dry_run=not a.go)
