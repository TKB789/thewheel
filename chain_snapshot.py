#!/usr/bin/env python3
"""
Daily option-chain snapshots, saved so the site can later test whether option readings (the
Greeks, implied volatility, skew) say more about assignment risk than the market's odds alone.

Free historical option prices don't exist, so the only way to study them is to start keeping
them. Once a trading day, on the first update after 9:45 AM New York time (normally the
10:15 AM run), this saves for every watchlist ticker:

  data/chains/<YYYY-MM>/<TICKER>.csv   one row per contract: out-of-the-money calls and puts
                                       expiring within the next 10 days, 0.03-0.50 delta
  data/chains/summary/<TICKER>.csv     one row per day: price, recent and implied volatility,
                                       skew, momentum, put/call open interest

Whether each contract finished in the money is worked out later from the closing price on its
expiration day (scripts/greeks_study.py), so nothing about outcomes is stored here.
Called from fetch_options.py; a failure here never stops the option update.
"""
from __future__ import annotations

import csv
import math
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
CHAINS_DIR = ROOT / "data" / "chains"
NY = ZoneInfo("America/New_York")
RISK_FREE = 0.04
MAX_DAYS = 10                    # expirations within this many calendar days (covers Mon->Wed ... Thu->Mon)
MAX_EXPIRIES = 4                 # at most the 4 nearest (ETFs list one every day)
DELTA_RANGE = (0.05, 0.50)
WINDOW = (time(9, 45), time(16, 0))
EXPIRY_CLOSE = time(16, 0)

# price and time of day are in the summary file (same date), so they aren't repeated per contract
ROW_FIELDS = ["date", "expiry", "hours_left", "type", "strike", "bid", "ask", "mid",
              "iv", "delta", "gamma", "theta", "vega", "prob_itm", "open_interest", "volume"]
SUMMARY_FIELDS = ["date", "time_et", "ticker", "spot", "prev_close", "hv20", "atm_iv", "iv_hv", "skew_25d",
                  "rsi14", "pct_ma20", "put_call_oi", "contracts"]


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _npdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _price(S, K, T, v, kind, r=RISK_FREE):
    d1 = (math.log(S / K) + (r + 0.5 * v * v) * T) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)
    if kind == "call":
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def _iv(price, S, K, T, kind):
    lo, hi = 0.01, 5.0
    intrinsic = max(0.0, S - K) if kind == "call" else max(0.0, K - S)
    if T <= 0 or price <= intrinsic + 1e-4 or _price(S, K, T, hi, kind) < price:
        return None
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if _price(S, K, T, mid, kind) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def greeks(S, K, T, v, kind, r=RISK_FREE):
    """Per share: delta, gamma, theta (per calendar day), vega (per 1 point of volatility),
    and the market's chance of finishing in the money."""
    sq = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * v * v) * T) / (v * sq)
    d2 = d1 - v * sq
    pdf = _npdf(d1)
    gamma = pdf / (S * v * sq)
    vega = S * pdf * sq / 100
    if kind == "call":
        delta = _ncdf(d1)
        theta = (-S * pdf * v / (2 * sq) - r * K * math.exp(-r * T) * _ncdf(d2)) / 365
        p_itm = _ncdf(d2)
    else:
        delta = _ncdf(d1) - 1
        theta = (-S * pdf * v / (2 * sq) + r * K * math.exp(-r * T) * _ncdf(-d2)) / 365
        p_itm = _ncdf(-d2)
    return delta, gamma, theta, vega, p_itm


def _num(x):
    try:
        v = float(x)
        return v if v == v else 0.0
    except (TypeError, ValueError):
        return 0.0


def in_window(now_et: datetime):
    return now_et.weekday() < 5 and WINDOW[0] <= now_et.time() <= WINDOW[1]


def build(ticker, S, prev_close, raw_expiries, now_et: datetime, hv20=None, rsi14=None, pct_ma20=None,
          put_call_oi=None):
    """Rows for the snapshot plus the day's summary. raw_expiries: [{"date", "calls_raw", "puts_raw"}]."""
    rows, atm_iv, skew, used = [], None, None, 0
    for e in raw_expiries:
        if used >= MAX_EXPIRIES:
            break
        exp = date.fromisoformat(e["date"])
        close_at = datetime.combine(exp, EXPIRY_CLOSE, NY)
        hours = (close_at - now_et).total_seconds() / 3600
        if hours <= 0 or hours > MAX_DAYS * 24:
            continue
        used += 1
        T = hours / (365 * 24)
        calls_iv, puts_iv = [], []          # (delta, iv) for the skew
        for kind, raw in (("call", e.get("calls_raw") or []), ("put", e.get("puts_raw") or [])):
            for rw in raw:
                K = _num(rw.get("strike"))
                if K <= 0:
                    continue
                bid, ask, last = _num(rw.get("bid")), _num(rw.get("ask")), _num(rw.get("lastPrice"))
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
                if mid <= 0.01:
                    continue
                v = _iv(mid, S, K, T, kind)
                if v is None:
                    continue
                d, g, th, vg, p = greeks(S, K, T, v, kind)
                (calls_iv if kind == "call" else puts_iv).append((abs(d), v, K))
                otm = K > S if kind == "call" else K < S
                if not otm or not (DELTA_RANGE[0] <= abs(d) <= DELTA_RANGE[1]):
                    continue
                rows.append([now_et.date().isoformat(), e["date"],
                             round(hours, 1), "C" if kind == "call" else "P", round(K, 2), round(bid, 2), round(ask, 2),
                             round(mid, 3), round(v, 4), round(d, 4), round(g, 5), round(th, 4), round(vg, 4),
                             round(p, 4), int(_num(rw.get("openInterest"))), int(_num(rw.get("volume")))])
        # at-the-money volatility and 25-delta skew from the nearest expiration at least a day out
        if atm_iv is None and hours >= 20 and calls_iv:
            atm = min(calls_iv + puts_iv, key=lambda x: abs(x[2] - S))
            atm_iv = atm[1]
            c25 = [x for x in calls_iv if x[2] > S]
            p25 = [x for x in puts_iv if x[2] < S]
            if c25 and p25:
                c = min(c25, key=lambda x: abs(x[0] - 0.25))
                p = min(p25, key=lambda x: abs(x[0] - 0.25))
                if abs(c[0] - 0.25) < 0.1 and abs(p[0] - 0.25) < 0.1:
                    skew = p[1] - c[1]
    summary = [now_et.date().isoformat(), now_et.strftime("%H:%M"), ticker, round(S, 2),
               round(prev_close, 2) if prev_close else "", _r(hv20, 4), _r(atm_iv, 4),
               _r(atm_iv / hv20 if atm_iv and hv20 else None, 3), _r(skew, 4), _r(rsi14, 1), _r(pct_ma20, 2),
               _r(put_call_oi, 3), len(rows)]
    return rows, summary


def _r(v, n):
    return "" if v is None else round(v, n)


def already_saved(ticker, day: str):
    p = CHAINS_DIR / "summary" / f"{ticker}.csv"
    if not p.exists():
        return False
    with p.open() as fh:
        last = None
        for line in fh:
            last = line
    return bool(last) and last.startswith(day + ",")


def save(ticker, rows, summary):
    day = summary[0]
    month_dir = CHAINS_DIR / day[:7]
    month_dir.mkdir(parents=True, exist_ok=True)
    (CHAINS_DIR / "summary").mkdir(parents=True, exist_ok=True)
    for path, fields, data in ((month_dir / f"{ticker}.csv", ROW_FIELDS, rows),
                               (CHAINS_DIR / "summary" / f"{ticker}.csv", SUMMARY_FIELDS, [summary])):
        new = not path.exists()
        with path.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(fields)
            w.writerows(data)


def maybe_snapshot(ticker, S, prev_close, raw_expiries, session_date: str, hv20=None, rsi14=None,
                   pct_ma20=None, put_call_oi=None, now_et: datetime | None = None):
    """Save today's snapshot if it's a trading day in the window and not saved yet. Returns rows saved."""
    now_et = now_et or datetime.now(NY)
    if not in_window(now_et) or session_date != now_et.date().isoformat():
        return 0                               # outside market hours, or a holiday
    if already_saved(ticker, session_date):
        return 0
    rows, summary = build(ticker, S, prev_close, raw_expiries, now_et, hv20, rsi14, pct_ma20, put_call_oi)
    if rows:
        save(ticker, rows, summary)
    return len(rows)
