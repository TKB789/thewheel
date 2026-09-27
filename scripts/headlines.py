#!/usr/bin/env python3
"""
Headline collector.

Yahoo only shows the latest few dozen headlines per stock and keeps no history,
so this runs with every update, scores each new headline's tone, and appends it
to data/headlines/<TICKER>.csv. research.py tests the tone against the price once
enough days have built up.

Tone is VADER's "compound" score (-1 very negative ... +1 very positive), a
standard free sentiment scorer. It reads general English, not finance jargon,
so "beats estimates" scores as neutral-to-positive and "cuts guidance" may be
missed. Treat it as a rough reading.
"""
from __future__ import annotations

import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "headlines"
WATCHLIST = ROOT / "watchlist.txt"
FIELDS = ["id", "published_utc", "provider", "title", "score"]


def parse_item(it: dict):
    """Handle both the current and the older yfinance news formats."""
    c = it.get("content") if isinstance(it.get("content"), dict) else None
    if c:
        title = c.get("title") or ""
        summary = c.get("summary") or c.get("description") or ""
        pub = c.get("pubDate") or c.get("displayTime")
        prov = (c.get("provider") or {}).get("displayName", "")
        uid = it.get("id") or c.get("id") or title
        try:
            when = datetime.fromisoformat(str(pub).replace("Z", "+00:00")) if pub else None
        except ValueError:
            when = None
    else:
        title = it.get("title") or ""
        summary = it.get("summary") or ""
        ts = it.get("providerPublishTime")
        when = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
        prov = it.get("publisher", "")
        uid = it.get("uuid") or title
    if not title or when is None:
        return None
    return {"id": str(uid), "published_utc": when.astimezone(timezone.utc).isoformat(timespec="minutes"),
            "provider": prov, "title": " ".join(title.split()), "text": f"{title}. {summary}".strip()}


def fetch_news(ticker):
    import yfinance as yf
    t = yf.Ticker(ticker)
    try:
        items = t.get_news(count=50)
    except Exception:
        items = t.news
    return [x for x in (parse_item(i) for i in (items or [])) if x]


def main():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    tickers = []
    for line in WATCHLIST.read_text().splitlines():
        s = line.split("#")[0].strip().upper()
        if s and s not in tickers:
            tickers.append(s)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tk in tickers:
        path = OUT_DIR / f"{tk}.csv"
        rows = []
        if path.exists():
            with path.open(newline="") as fh:
                rows = list(csv.DictReader(fh))
        seen = {r["id"] for r in rows} | {r["title"] for r in rows}
        try:
            items = fetch_news(tk)
        except Exception as exc:
            print(f"{tk}: headlines unavailable ({exc})", file=sys.stderr)
            continue
        new = 0
        for it in items:
            if it["id"] in seen or it["title"] in seen:
                continue
            it["score"] = round(vader.polarity_scores(it.pop("text"))["compound"], 4)
            rows.append(it)
            seen |= {it["id"], it["title"]}
            new += 1
        rows.sort(key=lambda r: r["published_utc"])
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"{tk}: {new} new headline(s), {len(rows)} total since {rows[0]['published_utc'][:10] if rows else '-'}")


if __name__ == "__main__":
    main()
