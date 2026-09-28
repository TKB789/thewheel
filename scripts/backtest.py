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

It also simulates the full wheel on the same weeks: covered calls while holding shares,
cash-secured puts after the shares are called away (paused in weeks when the cash doesn't
cover the put), and compares the account's value with simply holding the shares.

Output: data/backtest/<TICKER>.json, read by index.html.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research as rs  # reuses the price download, earnings history and feature code

OUT_DIR = rs.ROOT / "data" / "backtest"
RISK_FREE = 0.04
FEE = 0.65                 # per contract
DELTA, DELTA_HOT = 0.20, 0.15
PUT_DELTA, PUT_DELTA_SLIDING = 0.25, 0.15
PUT_HIST_RATE = 0.25       # study method for puts: stock fell below the strike in ~25% of earlier weeks
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


def bs_put(S, K, T, vol, r=RISK_FREE):
    return bs_call(S, K, T, vol, r) - S + K * math.exp(-r * T)


def put_delta_strike(S, T, vol, target, r=RISK_FREE):
    """Strike whose put delta equals -target: N(d1) = 1 - target."""
    d1 = N.inv_cdf(1 - target)
    return S * math.exp(-d1 * vol * math.sqrt(T) + (r + vol * vol / 2) * T)


def round_down(k, step):
    return math.floor(k / step + 1e-9) * step


def week_info(F: pd.DataFrame, earn_dates: list, now_et=None):
    """One entry per week that starts on a Monday trading day. Expiry is the last trading day
    of that week (Friday, or Thursday when Friday is a market holiday). Every value is known
    at Monday's open except `close`, which is only used to settle the week."""
    idx = F.index
    earn = [pd.Timestamp(x) for x in earn_dates]
    iso = idx.isocalendar()
    groups = pd.Series(range(len(idx)), index=idx).groupby([iso.year.values, iso.week.values])
    now_et = now_et or datetime.now(ZoneInfo("America/New_York"))
    today = pd.Timestamp(now_et.date())
    out = []
    for _, pos in groups:
        pos = list(pos)
        i, j = pos[0], pos[-1]
        e, x = idx[i], idx[j]
        if e.weekday() != 0 or x.weekday() < 3 or i == 0:
            continue
        friday = e + pd.Timedelta(days=4)
        if j == len(idx) - 1 and (today < friday or (today == friday and now_et.time() < rs.CLOSE_SETTLED)):
            continue                  # this week hasn't finished yet; it settles after Friday's close
        prev = F.iloc[i - 1]
        S, close = float(F["open"].iloc[i]), float(F["close"].iloc[j])
        vol = float(prev["hv20"])
        if not (vol > 0 and S > 0):
            continue
        sessions = j - i + 1
        out.append({
            "e": e, "x": x, "S": S, "close": close, "vol": vol,
            "T": ((x - e).days + 6.5 / 24) / 365.0,
            "sig_w": vol * math.sqrt(sessions / 252),
            "hot": bool((prev["rsi14"] > 70) or (prev["pct_ma20"] > 0.06)),
            "sliding": bool((prev["rsi14"] < 30) or (prev["pct_ma20"] < -0.06)),
            "earnings": any(idx[i - 1] <= d <= x for d in earn),
            "step": strike_step(S),
        })
    return out


def call_strike(w, method, history):
    if method == "delta":
        return round_up(delta_strike(w["S"], w["T"], w["vol"], DELTA_HOT if w["hot"] else DELTA), w["step"])
    if len(history) < MIN_HISTORY_WEEKS:
        return None
    zq = float(np.quantile(history, 1 - HIST_RATE))
    k = round_up(w["S"] * (1 + max(zq, 0.0) * w["sig_w"]) + 1e-9, w["step"])
    return k if k > w["S"] else round_up(w["S"] + w["step"] / 2, w["step"])


def put_strike(w, method, history):
    if method == "delta":
        return round_down(put_delta_strike(w["S"], w["T"], w["vol"], PUT_DELTA_SLIDING if w["sliding"] else PUT_DELTA), w["step"])
    if len(history) < MIN_HISTORY_WEEKS:
        return None
    zq = float(np.quantile(history, PUT_HIST_RATE))
    k = round_down(w["S"] * (1 + min(zq, 0.0) * w["sig_w"]) - 1e-9, w["step"])
    return k if k < w["S"] else round_down(w["S"] - w["step"] / 2, w["step"])


def run_weeks(F: pd.DataFrame, earn_dates: list, now_et=None):
    """Covered calls only: a call every Monday, assigned or not, on the same 100 shares."""
    rows, history = [], []          # history: z-scores of earlier non-earnings weeks
    for w in week_info(F, earn_dates, now_et):
        S, close = w["S"], w["close"]
        row = {"monday": w["e"].date().isoformat(), "expiry": w["x"].date().isoformat(), "open": round(S, 2),
               "close": round(close, 2), "earnings": w["earnings"], "hot": w["hot"]}
        for name in ("delta", "study"):
            K = call_strike(w, name, history)
            if K is None:
                continue
            prem = bs_call(S, K, w["T"], w["vol"])
            given = max(0.0, close - K)
            row[name] = {"strike": round(K, 2), "otm_pct": round((K / S - 1) * 100, 2),
                         "assigned": close > K, "est_premium": round(prem, 2),
                         "given_up": round(given, 2), "est_net": round((prem - given) * 100 - FEE, 2)}
        rows.append(row)
        if not w["earnings"]:           # only after the week is over does it join the history
            history.append((close / S - 1) / w["sig_w"])
    return rows


def wheel_sim(weeks, method, tbill=None):
    """The wheel: start owning 100 shares with no cash. Holding shares -> sell a covered call each
    Monday. Called away -> hold the cash and sell a cash-secured put each Monday, but only if the
    cash covers strike x 100; otherwise that week is paused. Put assigned -> buy 100 shares at the
    strike and go back to selling calls. Earnings weeks are skipped in both phases.
    Cash earns the 3-month Treasury bill rate each week, roughly what a money market fund
    (like Fidelity's core position) pays, when that rate series is available."""
    shares, cash = 100, 0.0
    history, timeline, events, log = [], [], [], []
    c = {"call_weeks": 0, "put_weeks": 0, "paused": 0, "earnings_skipped": 0, "no_history": 0,
         "calls_assigned": 0, "puts_assigned": 0, "call_premium": 0.0, "put_premium": 0.0, "fees": 0.0,
         "longest_pause": 0, "interest": 0.0}
    run = 0
    for n_w, w in enumerate(weeks):
        S, close, T, vol = w["S"], w["close"], w["T"], w["vol"]
        day = w["e"].date().isoformat()
        if tbill is not None and cash > 0 and n_w > 0:
            rate = tbill.asof(w["e"])
            if rate == rate:
                days = (w["e"] - weeks[n_w - 1]["e"]).days
                earned = cash * (float(rate) / 100) * days / 365
                cash += earned
                c["interest"] += earned
        entry = {"date": day, "holding": "shares" if shares else "cash", "open": round(S, 2), "close": round(close, 2)}
        if w["earnings"]:
            c["earnings_skipped"] += 1
            entry["action"] = "skip_earnings"
        elif shares:
            K = call_strike(w, method, history)
            if K is None:
                c["no_history"] += 1
                entry["action"] = "no_history"
            else:
                prem = bs_call(S, K, T, vol) * 100
                cash += prem - FEE
                c["call_weeks"] += 1; c["call_premium"] += prem; c["fees"] += FEE
                entry.update({"action": "call", "strike": round(K, 2), "premium": round(prem, 2), "outcome": "kept"})
                if close > K:
                    shares, cash = 0, cash + K * 100
                    c["calls_assigned"] += 1
                    entry["outcome"] = "called_away"
                    events.append({"date": day, "type": "called away", "strike": round(K, 2), "close": round(close, 2)})
            run = 0
        else:
            K = put_strike(w, method, history)
            if K is None:
                c["no_history"] += 1
                entry["action"] = "no_history"
            elif cash < K * 100:
                c["paused"] += 1
                entry.update({"action": "paused", "strike": round(K, 2), "cash": round(cash, 2)})
                run += 1
                c["longest_pause"] = max(c["longest_pause"], run)
            else:
                run = 0
                prem = bs_put(S, K, T, vol) * 100
                cash += prem - FEE
                c["put_weeks"] += 1; c["put_premium"] += prem; c["fees"] += FEE
                entry.update({"action": "put", "strike": round(K, 2), "premium": round(prem, 2), "outcome": "expired"})
                if close < K:
                    shares, cash = 100, cash - K * 100
                    c["puts_assigned"] += 1
                    entry["outcome"] = "bought"
                    events.append({"date": day, "type": "bought back", "strike": round(K, 2), "close": round(close, 2)})
        if not w["earnings"]:
            history.append((close / S - 1) / w["sig_w"])
        timeline.append([day, round(cash + shares * close, 2), round(100 * close, 2), 1 if shares else 0])
        log.append(entry)
    if not weeks:
        return None
    start = 100 * weeks[0]["S"]
    end_wheel, end_hold = timeline[-1][1], timeline[-1][2]
    c = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()}
    c.update({"start_value": round(start, 2), "end_wheel": end_wheel, "end_hold": end_hold,
              "wheel_return_pct": round((end_wheel / start - 1) * 100, 1),
              "hold_return_pct": round((end_hold / start - 1) * 100, 1),
              "difference": round(end_wheel - end_hold, 2),
              "premium_total": round(c["call_premium"] + c["put_premium"], 2),
              "ends_holding": bool(shares), "cash_now": round(cash, 2),
              "first": timeline[0][0], "last": timeline[-1][0]})
    return {"summary": c, "timeline": timeline, "events": events, "log": log[-52:]}


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


def backtest(ticker, px, earn, now_et=None):
    F = rs.build_features(px, ticker, [], None, earn)
    rows = run_weeks(F, earn, now_et)
    weeks = week_info(F, earn, now_et)
    tbill = None
    try:
        t = rs.load_prices(["^IRX"]).get("^IRX")
        if t is not None and not t.empty:
            tbill = rs._clean(t)["Close"].dropna()
    except Exception as exc:
        print(f"  T-bill rate unavailable, cash earns nothing: {exc}", file=sys.stderr)
    return {
        "ticker": ticker,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": "Monday open → Friday close",
        "settings": {"delta": DELTA, "delta_hot": DELTA_HOT, "study_rate": HIST_RATE,
                     "study_min_weeks": MIN_HISTORY_WEEKS, "fee": FEE},
        "summary": {m: {"rules": summarize(rows, m, True), "all_weeks": summarize(rows, m, False)}
                    for m in ("delta", "study")},
        "earnings_weeks": sum(1 for r in rows if r["earnings"]),
        "wheel": {m: wheel_sim(weeks, m, tbill) for m in ("delta", "study")},
        "cash_interest": tbill is not None,
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
            wh = (rep["wheel"]["delta"] or {}).get("summary", {})
            print(f"{tk}: {s.get('assigned', 0)} of {s.get('n', 0)} Monday calls assigned (0.20-delta estimate); "
                  f"wheel {wh.get('wheel_return_pct')}% vs holding {wh.get('hold_return_pct')}%")
        except Exception as exc:
            failed += 1
            print(f"{tk}: FAILED ({exc})", file=sys.stderr)
    if failed == len(tickers):
        sys.exit(1)


if __name__ == "__main__":
    main()
