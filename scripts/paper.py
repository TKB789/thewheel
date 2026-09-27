#!/usr/bin/env python3
"""
Paper trading.

1. Automatic track record: on Mondays and Thursdays, logs what the tool
   recommended (the delta-based "Call to sell" pick and the schedule-card pick
   for each window that starts that day) to data/paper_auto.csv. Weeks where it
   said to wait are logged too, as a hypothetical trade, so you can see whether
   waiting actually helped.
2. Your own paper trades: rows you add from the page (saved to paper_trades.csv).
3. Settlement: once a trade expires, the stock's closing price that day decides
   whether it was assigned. Result = premium kept, minus the per-contract fee,
   minus (if assigned) what you gave up versus just holding the shares.

Writes data/paper.json for the page.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
AUTO_LOG = DATA / "paper_auto.csv"
MANUAL_LOG = ROOT / "paper_trades.csv"
OUT = DATA / "paper.json"

FIELDS = ["id", "source", "logged_at", "ticker", "type", "strategy", "expiry", "strike",
          "premium", "contracts", "stock_price", "decision", "verdict", "note"]
ET = ZoneInfo("America/New_York")
FEE_PER_CONTRACT = 0.65          # Fidelity's options commission per contract
SETTLE_AFTER = time(16, 30)      # an expiring trade settles once the close is in

STRATEGY_NAMES = {
    "delta": "Call to sell (0.20 delta)",
    "mon_wed": "Schedule: Monday → Wednesday",
    "thu_fri": "Schedule: Thursday → Friday",
    "thu_mon": "Schedule: Thursday → next Monday",
    "put": "Wheel put",
    "csp": "Cash-secured put (0.25 delta)",
    "csp_mon_wed": "Put schedule: Monday → Wednesday",
    "csp_thu_fri": "Put schedule: Thursday → Friday",
    "csp_thu_mon": "Put schedule: Thursday → next Monday",
    "manual": "Your pick",
}


# ------------------------------------------------------------------ files
def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")  # drops helper fields like _per_day
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------ auto logging
def fill_price(opt_row):
    """What you'd realistically get selling: the bid when the market is open, else the last trade."""
    if opt_row.get("live_quote") and opt_row.get("bid", 0) > 0:
        return opt_row["bid"], "bid"
    return opt_row.get("mid", 0), "last trade"


def earnings_hits(opt, entry: date, expiry: date):
    e = opt.get("earnings_date")
    if not e:
        return False
    d = date.fromisoformat(e)
    return entry - timedelta(days=1) <= d <= expiry   # includes a report the evening before you sell


def schedule_pick(opt, win, entry: date, side="call"):
    """The schedule card's strike: for calls, the first listed strike at or above the study's
    upside target; for puts, the first listed strike at or below its downside target."""
    expiry = entry + timedelta(days=win["exit_offset"])
    exp = next((e for e in opt.get("expiries", []) if e["date"] == expiry.isoformat()), None)
    sug = win.get("suggestion") or {}
    if not exp:
        return expiry, None
    if side == "call":
        target = opt["price"] * (1 + sug["move_pct"] / 100)
        cands = sorted((c for c in exp["calls"] if c["strike"] >= target), key=lambda c: c["strike"])
    else:
        if sug.get("put_move_pct") is None:
            return expiry, None
        target = opt["price"] * (1 - sug["put_move_pct"] / 100)
        cands = sorted((c for c in exp["puts"] if c["strike"] <= target), key=lambda c: -c["strike"])
    return expiry, (cands[0] if cands else None)


SIDES = {
    # side: (verdict key, recommended key, chain key, delta strategy key, schedule prefix, target delta, band)
    "call": ("verdict", "recommended_call", "calls", "delta", "", 0.20, (0.10, 0.30)),
    "put": ("put_verdict", "recommended_put", "puts", "csp", "csp_", 0.25, (0.15, 0.35)),
}


def auto_log(now_et: datetime):
    today = now_et.date()
    rows = read_csv(AUTO_LOG)
    if today.weekday() not in (0, 3):
        return rows, 0
    have = {(r["ticker"], r["strategy"], r["logged_at"][:10]) for r in rows}
    index = load_json(DATA / "index.json") or {}
    added = 0
    for tk in index.get("tickers", []):
        opt = load_json(DATA / f"{tk}.json")
        if not opt:
            continue
        updated = datetime.fromisoformat(opt["updated"]).astimezone(ET).date()
        if updated != today:          # never log from stale quotes
            continue
        if opt.get("session_date", today.isoformat()) != today.isoformat():
            continue                  # market holiday: quotes are from the last session
        res = load_json(DATA / "research" / f"{tk}.json")
        stamp = now_et.isoformat(timespec="minutes")

        for side, (vkey, rkey, chain, dkey, prefix, tdelta, band) in SIDES.items():
            verdict = opt.get(vkey)
            if not verdict:
                continue
            level = verdict["level"]

            # 1) the delta-based card ("Call to sell" / the recommended cash-secured put)
            if (tk, dkey, today.isoformat()) not in have:
                rec = opt.get(rkey)
                decision = "sell" if rec and level != "wait" else "wait"
                pick = rec
                if pick is None and opt.get("expiries"):      # hypothetical trade for "hold off" days
                    near = opt["expiries"][0]
                    cands = [c for c in near[chain] if band[0] <= abs(c["delta"]) <= band[1]]
                    if cands:
                        pick = dict(min(cands, key=lambda c: abs(abs(c["delta"]) - tdelta)), expiry=near["date"])
                if pick and date.fromisoformat(pick["expiry"]) > today:
                    px, src = fill_price(pick)
                    rows.append({"id": f"A-{tk}-{dkey}-{today:%Y%m%d}", "source": "auto", "logged_at": stamp,
                                 "ticker": tk, "type": side, "strategy": dkey, "expiry": pick["expiry"],
                                 "strike": pick["strike"], "premium": round(px, 2), "contracts": 1,
                                 "stock_price": opt["price"], "decision": decision,
                                 "verdict": verdict["label"], "note": f"premium = {src}"})
                    added += 1

            # 2) schedule cards for windows that start today. On Thursday the Friday and Monday
            #    expirations are alternatives: the one paying more premium per day is the pick,
            #    the other is logged as "alternative" for comparison.
            picks = []
            for win in (res or {}).get("windows", []):
                key = prefix + win["key"]
                if win["entry_dow"] != today.weekday() or (tk, key, today.isoformat()) in have:
                    continue
                if not win.get("suggestion"):     # not enough history to study this window yet
                    continue
                expiry, o = schedule_pick(opt, win, today, side)
                if not o:
                    continue
                skip = earnings_hits(opt, today, expiry)
                px, src = fill_price(o)
                picks.append({"id": f"A-{tk}-{key}-{today:%Y%m%d}", "source": "auto", "logged_at": stamp,
                              "ticker": tk, "type": side, "strategy": key, "expiry": expiry.isoformat(),
                              "strike": o["strike"], "premium": round(px, 2), "contracts": 1,
                              "stock_price": opt["price"],
                              "decision": "wait" if skip or level == "wait" else "sell",
                              "verdict": "earnings in window" if skip else verdict["label"],
                              "note": f"premium = {src}",
                              "_per_day": px / max(1, (expiry - today).days)})
            live = [p for p in picks if p["decision"] == "sell"]
            if len(live) > 1:
                best = max(live, key=lambda p: p["_per_day"])
                for p in live:
                    if p is not best:
                        p["decision"] = "alternative"
            rows.extend(picks)
            added += len(picks)
    write_csv(AUTO_LOG, rows)
    return rows, added


# -------------------------------------------------------------- settlement
def close_on(closes: dict, d: date):
    """Closing price on d, or the last close before it (holiday expirations)."""
    for back in range(0, 5):
        c = closes.get(d - timedelta(days=back))
        if c is not None:
            return c
    return None


def settle(trade, closes, now_et: datetime):
    t = dict(trade)
    try:
        strike = float(t["strike"])
        prem = float(t["premium"])
        n = int(float(t.get("contracts") or 1))
        exp = date.fromisoformat(t["expiry"])
    except (KeyError, ValueError):
        t["status"] = "invalid"
        return t
    t.update({"strike": strike, "premium": prem, "contracts": n})
    done = exp < now_et.date() or (exp == now_et.date() and now_et.time() >= SETTLE_AFTER)
    if not done:
        t["status"] = "open"
        return t
    close = close_on(closes, exp)
    if close is None:
        t["status"] = "awaiting price"
        return t
    income = prem * 100 * n
    fee = FEE_PER_CONTRACT * n
    if t["type"] == "put":
        assigned = close < strike
        cost = max(0.0, strike - close) * 100 * n     # bought shares above market
    else:
        assigned = close > strike
        cost = max(0.0, close - strike) * 100 * n     # upside given up vs holding
    t.update({"status": "assigned" if assigned else "expired", "close": round(close, 2),
              "income": round(income, 2), "fee": round(fee, 2), "given_up": round(cost, 2),
              "net": round(income - fee - cost, 2)})
    return t


def summarize(trades):
    done = [t for t in trades if t.get("status") in ("expired", "assigned")]
    if not done:
        return {"n": 0, "open": sum(1 for t in trades if t.get("status") == "open")}
    nets = [t["net"] for t in done]
    return {
        "n": len(done),
        "open": sum(1 for t in trades if t.get("status") == "open"),
        "assigned": sum(1 for t in done if t["status"] == "assigned"),
        "income": round(sum(t["income"] for t in done), 2),
        "fees": round(sum(t["fee"] for t in done), 2),
        "given_up": round(sum(t["given_up"] for t in done), 2),
        "net": round(sum(nets), 2),
        "avg_net": round(sum(nets) / len(nets), 2),
        "wins": sum(1 for x in nets if x > 0),
    }


def load_closes(tickers, since: date):
    import yfinance as yf
    out = {}
    for tk in tickers:
        try:
            h = yf.Ticker(tk).history(start=since - timedelta(days=7), auto_adjust=False)
            out[tk] = {i.date(): float(c) for i, c in h["Close"].dropna().items()}
        except Exception as exc:
            print(f"{tk}: price history failed ({exc})", file=sys.stderr)
            out[tk] = {}
    return out


def build(auto_rows, manual_rows, closes, now_et):
    trades = []
    for r in auto_rows:
        trades.append(settle(r, closes.get(r["ticker"], {}), now_et))
    for r in manual_rows:
        r = dict(r, source="manual")
        r.setdefault("strategy", "manual")
        trades.append(settle(r, closes.get((r.get("ticker") or "").upper(), {}), now_et))
    trades = [{k: v for k, v in t.items() if not k.startswith("_")} for t in trades if t.get("status") != "invalid"]
    trades.sort(key=lambda t: t.get("logged_at", ""), reverse=True)

    schedule = ("mon_wed", "thu_fri", "thu_mon")
    auto = [t for t in trades if t["source"] == "auto"]
    # "followed" = the one call your routine would have sold each Monday / Thursday
    followed = [t for t in auto if t["strategy"] in schedule and t["decision"] == "sell"]
    skipped = [t for t in auto if t["strategy"] in schedule and t["decision"] == "wait"]
    put_schedule = tuple("csp_" + k for k in schedule)
    puts_followed = [t for t in auto if t["strategy"] in put_schedule and t["decision"] == "sell"]
    puts_skipped = [t for t in auto if t["strategy"] in put_schedule and t["decision"] == "wait"]
    mine = [t for t in trades if t["source"] == "manual"]
    by_strategy = {}
    for t in auto:
        if t["decision"] != "wait":
            by_strategy.setdefault(t["strategy"], []).append(t)
    return {
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fee_per_contract": FEE_PER_CONTRACT,
        "strategy_names": STRATEGY_NAMES,
        "summary": {
            "followed": summarize(followed),
            "skipped": summarize(skipped),
            "puts_followed": summarize(puts_followed),
            "puts_skipped": summarize(puts_skipped),
            "mine": summarize(mine),
            "by_strategy": {k: summarize(v) for k, v in by_strategy.items()},
        },
        "trades": trades,
    }


def main():
    now_et = datetime.now(ET)
    auto_rows, added = auto_log(now_et)
    manual_rows = read_csv(MANUAL_LOG)
    all_rows = auto_rows + manual_rows
    tickers = sorted({(r.get("ticker") or "").upper() for r in all_rows if r.get("ticker")})
    expiries = [date.fromisoformat(r["expiry"]) for r in all_rows if r.get("expiry")]
    closes = load_closes(tickers, min(expiries)) if expiries else {}
    report = build(auto_rows, manual_rows, closes, now_et)
    OUT.write_text(json.dumps(report, indent=1))
    s = report["summary"]
    print(f"Logged {added} automatic trade(s). Settled: followed {s['followed'].get('n', 0)}, "
          f"skipped {s['skipped'].get('n', 0)}, yours {s['mine'].get('n', 0)}.")


if __name__ == "__main__":
    main()
