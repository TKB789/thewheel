#!/usr/bin/env python3
"""
Backtest: sell one covered call at Monday's open, expiring that Friday.

For every week in the last ~10 years this works out which call the site would have
recommended, then checks Friday's close to see whether it would have been assigned.

Two ways of choosing the strike are tested:
  * delta:  the site's standard ~0.20-delta call (0.15 when the stock was running hot).
            Free historical option prices don't exist, so delta is estimated from how much
            the stock had actually been moving (20-day realized volatility).
  * study:  the history-based strike: where the stock rose past it by Friday in ~15% of
            earlier weeks. It is recomputed each week from earlier weeks only, so the test
            never uses information from the future.

The site's rules are applied (earnings weeks skipped). "All weeks" results are included
for comparison. Strikes are rounded up to typical listed-strike spacing. Premiums are
estimates (Black-Scholes at realized volatility); real premiums are usually a bit higher
because options tend to price in more movement than actually happens. Assignment is
decided by the real Friday close.

Output: data/backtest/<TICKER>.json, read by index.html.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research as rs  # reuses the price download, earnings history and feature code

OUT_DIR = rs.ROOT / "data" / "backtest"
RISK_FREE = 0.04
FEE = 0.65                 # per contract
DELTA, DELTA_HOT = 0.20, 0.15
HIST_RATE = 0.15           # study method: strike rose past in ~15% of earlier weeks
MIN_HISTORY_WEEKS = 52     # study method needs a year of earlier weeks first
N = NormalDist()


def strike_step(price):
    return 0.5 if price < 25 else 1.0 if price < 200 else 2.5 if price < 500 else 5.0


def round_up(k, step):
    return math.ceil(k / step - 1e-9) * step


def bs_call(S, K, T, vol, r=RISK_FREE):
    if T <= 0 or vol <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + vol * vol / 2) * T) / (vol * math.sqrt(T))
    d2 = d1 - vol * math.sqrt(T)
    return S * N.cdf(d1) - K * math.exp(-r * T) * N.cdf(d2)


def delta_strike(S, T, vol, target, r=RISK_FREE):
    """Strike whose call delta equals target: N(d1) = target."""
    d1 = N.inv_cdf(target)
    return S * math.exp(-d1 * vol * math.sqrt(T) + (r + vol * vol / 2) * T)


def run_weeks(F: pd.DataFrame, earn_dates: list):
    """One row per week that starts on a Monday trading day. Expiry is the last trading day
    of that week (Friday, or Thursday when Friday is a market holiday)."""
    idx = F.index
    earn = [pd.Timestamp(x) for x in earn_dates]
    iso = idx.isocalendar()
    groups = pd.Series(range(len(idx)), index=idx).groupby([iso.year.values, iso.week.values])
    rows, history = [], []          # history: z-scores of earlier non-earnings weeks
    for _, pos in groups:
        pos = list(pos)
        i, j = pos[0], pos[-1]
        e, x = idx[i], idx[j]
        if e.weekday() != 0 or x.weekday() < 3 or i == 0:
            continue
        prev = F.iloc[i - 1]
        S, close = float(F["open"].iloc[i]), float(F["close"].iloc[j])
        vol = float(prev["hv20"])
        if not (vol > 0 and S > 0):
            continue
        sessions = j - i + 1
        T = (x - e).days + 6.5 / 24
        T /= 365.0
        hot = bool((prev["rsi14"] > 70) or (prev["pct_ma20"] > 0.06))
        earnings = any(idx[i - 1] <= d <= x for d in earn)
        step = strike_step(S)
        sig_w = vol * math.sqrt(sessions / 252)

        k_delta = round_up(delta_strike(S, T, vol, DELTA_HOT if hot else DELTA), step)
        k_study = None
        if len(history) >= MIN_HISTORY_WEEKS:
            zq = float(np.quantile(history, 1 - HIST_RATE))
            k_study = round_up(S * (1 + max(zq, 0.0) * sig_w) + 1e-9, step)
            if k_study <= S:
                k_study = round_up(S + step / 2, step)

        row = {"monday": e.date().isoformat(), "expiry": x.date().isoformat(), "open": round(S, 2),
               "close": round(close, 2), "earnings": earnings, "hot": hot}
        for name, K in (("delta", k_delta), ("study", k_study)):
            if K is None:
                continue
            prem = bs_call(S, K, T, vol)
            given = max(0.0, close - K)
            row[name] = {"strike": round(K, 2), "otm_pct": round((K / S - 1) * 100, 2),
                         "assigned": close > K, "est_premium": round(prem, 2),
                         "given_up": round(given, 2), "est_net": round((prem - given) * 100 - FEE, 2)}
        rows.append(row)
        if not earnings:               # only after the week is over does it join the history
            history.append((close / S - 1) / sig_w)
    return rows


def summarize(rows, method, rules=True):
    use = [r for r in rows if method in r and (not rules or not r["earnings"])]
    if not use:
        return {"n": 0}
    a = [r for r in use if r[method]["assigned"]]
    years = {}
    for r in use:
        y = years.setdefault(r["monday"][:4], {"year": r["monday"][:4], "n": 0, "assigned": 0})
        y["n"] += 1
        y["assigned"] += int(r[method]["assigned"])
    return {
        "n": len(use), "assigned": len(a), "rate": round(len(a) / len(use), 4),
        "avg_otm_pct": round(float(np.mean([r[method]["otm_pct"] for r in use])), 2),
        "est_premium": round(sum(r[method]["est_premium"] for r in use) * 100, 2),
        "given_up": round(sum(r[method]["given_up"] for r in use) * 100, 2),
        "fees": round(FEE * len(use), 2),
        "est_net": round(sum(r[method]["est_net"] for r in use), 2),
        "worst_week": min((r[method]["est_net"] for r in use), default=None),
        "first": use[0]["monday"], "last": use[-1]["monday"],
        "by_year": sorted(years.values(), key=lambda y: y["year"]),
    }


def backtest(ticker, px, earn):
    F = rs.build_features(px, ticker, [], None, earn)
    rows = run_weeks(F, earn)
    return {
        "ticker": ticker,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": "Monday open → Friday close",
        "settings": {"delta": DELTA, "delta_hot": DELTA_HOT, "study_rate": HIST_RATE,
                     "study_min_weeks": MIN_HISTORY_WEEKS, "fee": FEE},
        "summary": {m: {"rules": summarize(rows, m, True), "all_weeks": summarize(rows, m, False)}
                    for m in ("delta", "study")},
        "earnings_weeks": sum(1 for r in rows if r["earnings"]),
        # keeps the file small: the last year of weeks, plus every week that got assigned
        "weeks": [r for i, r in enumerate(rows)
                  if i >= len(rows) - 52 or any(r.get(m, {}).get("assigned") for m in ("delta", "study"))],
    }


def main():
    tickers, _ = rs.tickers_to_run(rs.WATCHLIST)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = 0
    for tk in tickers:
        try:
            px = rs.load_prices([tk])
            if tk not in px:
                raise RuntimeError("no price history")
            rep = backtest(tk, px, rs.load_earnings(tk))
            (OUT_DIR / f"{tk}.json").write_text(json.dumps(rep, separators=(",", ":")))
            s = rep["summary"]["delta"]["rules"]
            print(f"{tk}: {s.get('assigned', 0)} of {s.get('n', 0)} Monday calls assigned (0.20-delta estimate)")
        except Exception as exc:
            failed += 1
            print(f"{tk}: FAILED ({exc})", file=sys.stderr)
    if failed == len(tickers):
        sys.exit(1)


if __name__ == "__main__":
    main()
