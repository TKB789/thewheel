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

It also simulates the full wheel, on two schedules (once a week Mon->Fri, and twice a week
Mon->Wed then Thu->Fri): covered calls while holding shares,
cash-secured puts after the shares are called away (paused in weeks when the cash doesn't
cover the put), and compares the account's value with simply holding the shares.

"Before each assignment": for every Monday -> Friday week it records the signal readings known
at Monday's open, compares assigned weeks with weeks that expired worthless, and lists each
assigned week's 10 most similar earlier weeks ("lookalikes") with how those turned out. A
shuffle test shows whether lookalikes predict assignment better than chance.

Output: data/backtest/<TICKER>.json (read by index.html when a ticker loads) and
data/backtest/<TICKER>_history.json (every trade since the start, loaded when you ask for it) and
data/backtest/<TICKER>_policy.json (the wheel following the site's recommendation, loaded when picked).
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
import fetch_options as fo  # the live recommendation's limits, so the backtest follows the same rules

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


SCHEDULES = {
    # leg key: (entry weekday, exit weekday or "last" = last trading day of the week, Thu/Fri)
    "weekly": [("mon_fri", 0, "last")],
    "twice": [("mon_wed", 0, 2), ("thu_fri", 3, 4)],
}


def periods(F: pd.DataFrame, earn_dates: list, now_et=None, schedule="weekly"):
    """Every trade period in the schedule, in date order. Each is entered at the entry day's open
    and settles at the exit day's close. Every value is known at the entry except `close`.
    A period whose exit day is a market holiday is skipped (no expiration that day)."""
    idx = F.index
    earn = [pd.Timestamp(x) for x in earn_dates]
    iso = idx.isocalendar()
    groups = pd.Series(range(len(idx)), index=idx).groupby([iso.year.values, iso.week.values])
    now_et = now_et or datetime.now(ZoneInfo("America/New_York"))
    today = pd.Timestamp(now_et.date())
    out = []
    for _, pos in groups:
        pos = list(pos)
        days = {idx[k].weekday(): k for k in pos}
        monday = idx[pos[0]] - pd.Timedelta(days=idx[pos[0]].weekday())
        for leg, entry_dow, exit_dow in SCHEDULES[schedule]:
            if entry_dow not in days:
                continue
            i = days[entry_dow]
            if exit_dow == "last":
                j = pos[-1]
                if idx[j].weekday() < 3:
                    continue
                exit_cal = monday + pd.Timedelta(days=4)
            else:
                if exit_dow not in days:
                    continue
                j = days[exit_dow]
                exit_cal = monday + pd.Timedelta(days=exit_dow)
            if i == 0 or j < i:
                continue
            if j == len(idx) - 1 and (today < exit_cal or (today == exit_cal and now_et.time() < rs.CLOSE_SETTLED)):
                continue                  # not finished yet; it settles after the exit day's close
            e, x = idx[i], idx[j]
            prev = F.iloc[i - 1]
            S, close = float(F["open"].iloc[i]), float(F["close"].iloc[j])
            vol = float(prev["hv20"])
            if not (vol > 0 and S > 0):
                continue
            sessions = j - i + 1
            out.append({
                "leg": leg, "e": e, "x": x, "S": S, "close": close, "vol": vol,
                "T": ((x - e).days + 6.5 / 24) / 365.0,
                "sig_w": vol * math.sqrt(sessions / 252),
                "hot": bool((prev["rsi14"] > 70) or (prev["pct_ma20"] > 0.06)),
                "sliding": bool((prev["rsi14"] < 30) or (prev["pct_ma20"] < -0.06)),
                "earnings": any(idx[i - 1] <= d <= x for d in earn),
                "step": strike_step(S),
            })
    return out


def week_info(F: pd.DataFrame, earn_dates: list, now_et=None):
    """Monday open -> last trading day of the week (Friday, or Thursday before a Friday holiday)."""
    return periods(F, earn_dates, now_et, "weekly")


# limits for the similar-weeks adjustment of the history-based target (same as the daily study)
STUDY_CALL_RATE_LIMITS = (0.07, 0.25)
STUDY_PUT_RATE_LIMITS = (0.12, 0.35)


def call_strike(w, method, history, ratio=1.0):
    """ratio: how much likelier assignment looks from similar past weeks (1 = no adjustment)."""
    if method == "delta":
        d = DELTA_HOT if w["hot"] else DELTA
        if ratio != 1.0:
            d = min(max(d / ratio, fo.LK_CALL_DELTA_LIMITS[0]), fo.LK_CALL_DELTA_LIMITS[1])
        return round_up(delta_strike(w["S"], w["T"], w["vol"], d), w["step"])
    if len(history) < MIN_HISTORY_WEEKS:
        return None
    rate = min(max(HIST_RATE / ratio, STUDY_CALL_RATE_LIMITS[0]), STUDY_CALL_RATE_LIMITS[1]) if ratio != 1.0 else HIST_RATE
    zq = float(np.quantile(history, 1 - rate))
    k = round_up(w["S"] * (1 + max(zq, 0.0) * w["sig_w"]) + 1e-9, w["step"])
    return k if k > w["S"] else round_up(w["S"] + w["step"] / 2, w["step"])


def put_strike(w, method, history, ratio=1.0):
    if method == "delta":
        d = PUT_DELTA_SLIDING if w["sliding"] else PUT_DELTA
        if ratio != 1.0:
            d = min(max(d / ratio, fo.LK_PUT_DELTA_LIMITS[0]), fo.LK_PUT_DELTA_LIMITS[1])
        return round_down(put_delta_strike(w["S"], w["T"], w["vol"], d), w["step"])
    if len(history) < MIN_HISTORY_WEEKS:
        return None
    rate = min(max(PUT_HIST_RATE / ratio, STUDY_PUT_RATE_LIMITS[0]), STUDY_PUT_RATE_LIMITS[1]) if ratio != 1.0 else PUT_HIST_RATE
    zq = float(np.quantile(history, rate))
    k = round_down(w["S"] * (1 + min(zq, 0.0) * w["sig_w"]) - 1e-9, w["step"])
    return k if k < w["S"] else round_down(w["S"] - w["step"] / 2, w["step"])


def run_trades(P: list):
    """Covered calls only: a call every period of the schedule, assigned or not, on the same
    100 shares. Each window (e.g. Mon -> Wed, Thu -> Fri) keeps its own history for the
    history-based strike, since a 2-day and a 5-day trade move by different amounts."""
    rows, histories = [], {}        # history: z-scores of earlier non-earnings periods, per window
    for w in P:
        history = histories.setdefault(w["leg"], [])
        S, close = w["S"], w["close"]
        row = {"monday": w["e"].date().isoformat(), "expiry": w["x"].date().isoformat(), "leg": w["leg"],
               "open": round(S, 2), "close": round(close, 2), "earnings": w["earnings"], "hot": w["hot"]}
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
        if not w["earnings"]:           # only after the period is over does it join the history
            history.append((close / S - 1) / w["sig_w"])
    return rows


def run_weeks(F: pd.DataFrame, earn_dates: list, now_et=None):
    return run_trades(week_info(F, earn_dates, now_et))


ROW_FIELDS = ["date", "expiry", "window", "open", "close", "earnings", "hot",
              "delta_strike", "delta_assigned", "delta_premium", "delta_net",
              "study_strike", "study_assigned", "study_premium", "study_net"]


def pack_row(r):
    """Calls-only rows as short arrays (see ROW_FIELDS) so ten years fit in a small file."""
    out = [r["monday"], r["expiry"], r["leg"], r["open"], r["close"], int(r["earnings"]), int(r["hot"])]
    for m in ("delta", "study"):
        x = r.get(m)
        out += [x["strike"], int(x["assigned"]), x["est_premium"], x["est_net"]] if x else [None, None, None, None]
    return out


LOG_FIELDS = ["date", "expiry", "window", "holding", "open", "close", "action", "strike", "premium",
              "outcome", "added", "cash"]


def pack_log(e):
    return [e["date"], e["expiry"], e["leg"], 1 if e["holding"] == "shares" else 0, e["open"], e["close"],
            e["action"], e.get("strike"), e.get("premium"), e.get("outcome"), e.get("added"), e.get("cash")]


POLICIES = {"site": "Follow the site", "favorable": "Favorable weeks only"}


def _decision(policy, w, kind):
    """The site's call for this trade: (similar-weeks ratio, number of flags)."""
    d = ((policy or {}).get(w["leg"]) or {}).get(kind, {}).get(w["e"].date().isoformat())
    trend = w["hot"] if kind == "call" else w["sliding"]
    if d is None:                  # before the weighting's first fit: only the trend flag
        return 1.0, int(trend)
    return d


def wheel_sim(weeks, method, tbill=None, topup=False, legs_per_week=1, policy=None, mode=None):
    """The wheel: start owning 100 shares with no cash. Holding shares -> sell a covered call each
    Monday. Called away -> hold the cash and sell a cash-secured put each Monday, but only if the
    cash covers strike x 100; otherwise that week is paused. Put assigned -> buy 100 shares at the
    strike and go back to selling calls. Earnings weeks are skipped in both phases.
    Cash earns the 3-month Treasury bill rate each week, roughly what a money market fund
    (like Fidelity's core position) pays, when that rate series is available.

    topup=True: instead of pausing, add just enough money to cover the put and keep going.
    The benchmark then also buys the stock with each deposit on the same day, so the
    comparison isn't flattered by the extra money.

    policy / mode: follow the site's recommendation instead of trading every period. Strikes use
    the similar-weeks adjustment (walk-forward, see recommendation_backtest), and a trade is skipped
    when the verdict says so: mode "site" skips "hold off" (two flags), mode "favorable" skips any
    flag. A skipped call week keeps the shares; a skipped put week keeps the cash earning interest."""
    shares, cash = 100, 0.0
    histories, timeline, events, log = {}, [], [], []
    c = {"call_weeks": 0, "put_weeks": 0, "paused": 0, "earnings_skipped": 0, "no_history": 0,
         "calls_assigned": 0, "puts_assigned": 0, "call_premium": 0.0, "put_premium": 0.0, "fees": 0.0,
         "longest_pause": 0, "interest": 0.0, "added": 0.0, "topups": 0, "largest_topup": 0.0,
         "skipped_calls": 0, "skipped_puts": 0}
    bench_extra_shares = 0.0          # shares the benchmark buys with the same deposits
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
        history = histories.setdefault(w["leg"], [])
        entry = {"date": day, "expiry": w["x"].date().isoformat(), "leg": w["leg"],
                 "holding": "shares" if shares else "cash", "open": round(S, 2), "close": round(close, 2)}
        if w["earnings"]:
            c["earnings_skipped"] += 1
            entry["action"] = "skip_earnings"
        elif shares and mode and _decision(policy, w, "call")[1] >= (2 if mode == "site" else 1):
            c["skipped_calls"] += 1
            entry.update({"action": "skip_flag", "outcome": "hold_off" if _decision(policy, w, "call")[1] >= 2 else "not_favorable"})
            run = 0
        elif not shares and mode and _decision(policy, w, "put")[1] >= (2 if mode == "site" else 1):
            c["skipped_puts"] += 1
            entry.update({"action": "skip_flag", "outcome": "hold_off" if _decision(policy, w, "put")[1] >= 2 else "not_favorable"})
        elif shares:
            K = call_strike(w, method, history, _decision(policy, w, "call")[0] if mode else 1.0)
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
            K = put_strike(w, method, history, _decision(policy, w, "put")[0] if mode else 1.0)
            if K is None:
                c["no_history"] += 1
                entry["action"] = "no_history"
            elif cash < K * 100 and not topup:
                c["paused"] += 1
                entry.update({"action": "paused", "strike": round(K, 2), "cash": round(cash, 2)})
                run += 1
                c["longest_pause"] = max(c["longest_pause"], run)
            else:
                run = 0
                if cash < K * 100:        # add money to cover the put
                    add = K * 100 - cash
                    cash += add
                    c["added"] += add; c["topups"] += 1; c["largest_topup"] = max(c["largest_topup"], add)
                    bench_extra_shares += add / S
                    entry["added"] = round(add, 2)
                    events.append({"date": day, "type": "added money", "amount": round(add, 2), "strike": round(K, 2), "close": round(close, 2)})
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
        timeline.append([day, round(cash + shares * close, 2), round((100 + bench_extra_shares) * close, 2), 1 if shares else 0])
        log.append(entry)
    if not weeks:
        return None
    # one chart point per week keeps the file small (the last period of each week)
    weekly_tl = {}
    for t in timeline:
        d = pd.Timestamp(t[0])
        weekly_tl[(d.isocalendar().year, d.isocalendar().week)] = t
    timeline_all = timeline
    timeline = list(weekly_tl.values())
    start = 100 * weeks[0]["S"]
    end_wheel, end_hold = timeline_all[-1][1], timeline_all[-1][2]
    c = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()}
    c.update({"topup_mode": topup, "bench_shares": round(100 + bench_extra_shares, 4),
              "net_of_added": round(end_wheel - c["added"], 2),
              "start_value": round(start, 2), "end_wheel": end_wheel, "end_hold": end_hold,
              "wheel_return_pct": round((end_wheel / (start + c["added"]) - 1) * 100, 1),
              "hold_return_pct": round((end_hold / (start + c["added"]) - 1) * 100, 1),
              "difference": round(end_wheel - end_hold, 2),
              "premium_total": round(c["call_premium"] + c["put_premium"], 2),
              "ends_holding": bool(shares), "cash_now": round(cash, 2),
              "first": timeline_all[0][0], "last": timeline_all[-1][0],
              "last_leg": weeks[-1]["leg"], "last_expiry": weeks[-1]["x"].date().isoformat(),
              "trades_per_week": legs_per_week, "policy": mode})
    timeline = [[t[0], round(t[1]), round(t[2]), t[3]] for t in timeline]     # whole dollars for the chart
    return {"summary": c, "timeline": timeline, "events": events[-200:], "events_total": len(events),
            "log": log[-52 * legs_per_week:], "log_all": log}


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
        "by_window": [{"window": leg, "n": len(g), "assigned": sum(1 for r in g if r[method]["assigned"]),
                       "rate": round(sum(1 for r in g if r[method]["assigned"]) / len(g), 4),
                       "est_premium": round(sum(r[method]["est_premium"] for r in g) * 100, 2),
                       "est_net": round(sum(r[method]["est_net"] for r in g), 2)}
                      for leg in dict.fromkeys(r.get("leg", "mon_fri") for r in use)
                      for g in [[r for r in use if r.get("leg", "mon_fri") == leg]]],
    }


# ------------------------------------------------------------ before each assignment
# "Lookalike" weeks: for every Monday -> Friday week, the signal readings known at Monday's open,
# and which earlier weeks had the most similar readings. Similarity is measured on this fixed
# set of core signals, chosen in advance (not picked after looking at the results).
CORE = ["vix", "ret_5d", "rsi14", "hv20", "pct_ma20", "analyst_net_30d", "wiki_spike", "gap_open", "news_tone"]
SIGNAL_LABELS = {**rs.LABELS, "news_tone": "Yahoo headline tone, past 3 days"}
K_NEAR = 10            # lookalikes shown per week
K_TEST = 50            # lookalikes used for the steadier rate and the chance test (10 is too noisy)
MIN_POOL = 52          # a week needs a year of earlier weeks before it gets lookalikes
MIN_SHARED = 4         # signals two weeks must both have to be compared
SHUFFLES = 200
DETAIL_WEEKS = 20      # assigned trades listed in full, per side
PCT_SIGNALS = {"ret_1d", "ret_5d", "ret_20d", "pct_ma20", "pct_ma50", "hv20", "rel_qqq_5d", "qqq_5d", "smh_5d",
               "peers_1d", "peers_5d", "dxy_5d", "oil_5d", "gap_open", "pt_gap"}


def shown(f, v):
    if v is None or v != v:
        return "—"
    if f in PCT_SIGNALS:
        return f"{v * 100:+.1f}%" if f != "hv20" else f"{v * 100:.0f}%"
    if f in ("wiki_spike", "wiki_trend", "vix_term", "vol_expansion", "volume_ratio", "range_ratio"):
        return f"{v:.2f}×"
    if f in ("analyst_net_7d", "analyst_net_30d", "pt_net_30d", "analyst_activity_7d", "days_to_earn",
             "days_since_rating"):
        return f"{v:+.0f}" if f.startswith(("analyst_net", "pt_net")) else f"{v:.0f}"
    if f in ("tnx_chg_5d", "vix_chg_5d"):
        return f"{v:+.2f}"
    if f == "news_tone":
        return f"{v:+.2f}"
    return f"{v:.1f}"


def week_readings(F: pd.DataFrame, weeks: list, news: pd.Series | None):
    """One row per non-earnings trade in one window: the readings known at the entry day's open
    (previous close for price-based signals; the entry day for page views, analysts, headlines),
    whether the site's standard call / put would have been assigned, and the estimated result
    per contract (premium, minus anything given up past the strike, minus the fee)."""
    pos = {d: i for i, d in enumerate(F.index)}
    rows = []
    for w in weeks:
        if w["earnings"] or w["e"] not in pos:
            continue
        i = pos[w["e"]]
        prev, cur = F.iloc[i - 1], F.iloc[i]
        r = {f: prev[f] for f in rs.FEATURES}
        for f in rs.ENTRY_DAY_FEATURES:
            r[f] = cur[f]
        r["gap_open"] = w["S"] / prev["close"] - 1 if prev["close"] > 0 else np.nan
        r["news_tone"] = float(news.get(w["e"], np.nan)) if news is not None else np.nan
        kc, kp = call_strike(w, "delta", []), put_strike(w, "delta", [])
        S, close = w["S"], w["close"]
        net_c = (bs_call(S, kc, w["T"], w["vol"]) - max(0.0, close - kc)) * 100 - FEE
        net_p = (bs_put(S, kp, w["T"], w["vol"]) - max(0.0, kp - close)) * 100 - FEE
        r.update({"entry": w["e"], "expiry": w["x"], "S": S, "close": close, "T": w["T"], "vol": w["vol"],
                  "step": w["step"], "hot": w["hot"], "sliding": w["sliding"],
                  "ret": close / S - 1, "kc": kc, "kp": kp, "net_call": net_c, "net_put": net_p,
                  "call": close > kc, "put": close < kp})
        rows.append(r)
    return pd.DataFrame(rows)


def _pct_rank(values: pd.Series, v):
    """Where v sits among the weeks' readings, 0-100 (50 = typical)."""
    x = np.sort(values.dropna().values)
    if v is None or v != v or len(x) == 0:
        return np.nan
    return 100.0 * (np.searchsorted(x, v, "left") + np.searchsorted(x, v, "right")) / (2 * len(x))


def _distances(P: np.ndarray, q: np.ndarray):
    """Mean absolute percentile gap between q and every row of P, over signals both have."""
    d = np.abs(P - q)
    shared = np.sum(~np.isnan(d), axis=1)
    with np.errstate(invalid="ignore"):
        dist = np.nanmean(np.where(np.isnan(d), np.nan, d), axis=1) if d.size else d
    dist = np.where(shared >= MIN_SHARED, dist, np.nan)
    return dist


RECENT_FRAC = 0.30     # "held up recently": same direction in the most recent 30% of trades


def two_prop_p(a: pd.Series, b: pd.Series):
    """Two-sided test that two assignment rates differ (pooled two-proportion z-test)."""
    n1, n2 = len(a), len(b)
    p1, p2 = float(a.mean()), float(b.mean())
    pool = (a.sum() + b.sum()) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    return float(2 * (1 - rs.norm_cdf(abs(p1 - p2) / se)))


def _range_text(f, lo, hi, first, last):
    if first:
        return f"below {shown(f, hi)}"
    if last:
        return f"above {shown(f, lo)}"
    return f"{shown(f, lo)} to {shown(f, hi)}"


def favorable(R: pd.DataFrame, side: str):
    """When selling worked best: each signal's readings split into thirds (low / middle / high),
    with the assignment rate and average estimated result per contract in each third.
    Tested on the assignment rate (real closing prices), top third vs bottom third, with
    false-discovery control across signals, plus a check that the difference points the same
    way in the most recent 30% of trades. The estimated result is shown but not tested, since
    the premium estimate itself comes from the volatility reading."""
    net = R["net_call" if side == "call" else "net_put"].astype(float)
    y = R[side].astype(bool)
    cut_recent = int(len(R) * (1 - RECENT_FRAC))
    rows = []
    for f in SIGNAL_LABELS:
        if f not in R:
            continue
        x = R[f].astype(float)
        ok = x.notna()
        if ok.sum() < 90 or x[ok].nunique() < 3:
            continue
        try:
            codes, edges = pd.qcut(x[ok], 3, labels=False, retbins=True, duplicates="drop")
        except ValueError:
            continue
        nb = int(codes.max()) + 1 if len(codes) else 0
        if nb < 2:
            continue
        bins = []
        for b in range(nb):
            m = codes == b
            idx = codes.index[m]
            bins.append({"range": _range_text(f, edges[b], edges[b + 1], b == 0, b == nb - 1),
                         "n": int(m.sum()), "assigned": int(y[idx].sum()),
                         "rate": round(float(y[idx].mean()) * 100, 1),
                         "avg_net": round(float(net[idx].mean()), 2)})
        lo_i, hi_i = codes.index[codes == 0], codes.index[codes == nb - 1]
        if len(lo_i) < 20 or len(hi_i) < 20:
            continue
        p = two_prop_p(y[hi_i], y[lo_i])
        diff = float(y[hi_i].mean() - y[lo_i].mean())
        pos = pd.Series(range(len(R)), index=R.index)
        rh, rl = hi_i[pos[hi_i] >= cut_recent], lo_i[pos[lo_i] >= cut_recent]
        recent = float(y[rh].mean() - y[rl].mean()) if len(rh) >= 10 and len(rl) >= 10 else None
        # favorable = least often assigned; ties go to the better estimated result
        best = min(range(nb), key=lambda b: (bins[b]["rate"], -bins[b]["avg_net"]))
        worst = max(range(nb), key=lambda b: (bins[b]["rate"], -bins[b]["avg_net"]))
        rows.append({"key": f, "label": SIGNAL_LABELS[f], "bins": bins, "p": round(p, 4),
                     "high_minus_low": round(diff * 100, 1),
                     "recent_high_minus_low": None if recent is None else round(recent * 100, 1),
                     "recent_same": None if recent is None or diff == 0 else bool(np.sign(recent) == np.sign(diff)),
                     "best": best, "worst": worst})
    passed = rs.bh_pass([r["p"] for r in rows])
    for r, ok in zip(rows, passed):
        r["found"] = bool(ok)
    rows.sort(key=lambda r: (not r["found"], r["p"]))
    return {"signals": rows, "avg_net": round(float(net.mean()), 2), "rate": round(float(y.mean()) * 100, 1),
            "n": int(len(R)), "recent_frac": RECENT_FRAC}


# ------------------------------------------------------------ does following the recommendation help?
WF_MIN_FIT = 104       # trades with lookalikes needed before the first walk-forward fit (~2 years)
WF_SHUFFLES = 100


def _wf_fit(y, NB, J, rate, known, rng):
    """Fit the lookalike weighting on the trades in `known` (a boolean mask over J) only:
    does it beat chance there, and how often were trades in each lookalike-rate third assigned."""
    sub = np.where(known)[0]
    if len(sub) < WF_MIN_FIT:
        return None
    t = y[J[sub]]
    if t.sum() < 10 or (~t).sum() < 10:
        return None
    def gap(labels):
        rr = labels[NB[sub]].mean(axis=1)
        tt = labels[J[sub]]
        return rr[tt].mean() - rr[~tt].mean()
    g = gap(y)
    idx_known = np.unique(np.concatenate([J[sub], NB[sub].ravel()]))
    sh = []
    for _ in range(WF_SHUFFLES):
        perm = y.copy()
        perm[idx_known] = rng.permutation(y[idx_known])
        sh.append(gap(perm))
    p = (np.sum(np.array(sh) >= g) + 1) / (len(sh) + 1)
    rr = rate[sub]
    cuts = np.quantile(rr, [1 / 3, 2 / 3])
    grp = np.searchsorted(cuts, rr, side="right")
    B = []
    for k in range(3):
        m = grp == k
        if m.sum():
            B.append({"label": str(k), "weeks": int(m.sum()), "rate": float(t[m].mean()) * 100,
                      "lo": float(rr[m].min()) * 100, "hi": float(rr[m].max()) * 100})
    return {"beats": bool(p < 0.05 and g > 0), "base": float(t.mean()) * 100, "B": B}


def recommendation_backtest(R: pd.DataFrame, y: np.ndarray, NB, J, rate, side: str, seed=11):
    """Walk-forward test of the live recommendation on one window and option type.
    Each January the lookalike weighting is refit using only trades that had already expired,
    then applied to that year's trades exactly as the live site would:
      * strike: the standard delta (0.20 / 0.15 when running hot for calls; 0.25 / 0.15 when
        sliding for puts) divided by how much likelier assignment looks, within the same limits;
      * flags: running hot (sliding, for puts), and lookalikes at least 1.25x likelier.
        One flag = sell further out; two = hold off (the site's verdict).
    The premium-level flag (implied vs recent volatility) can't be rebuilt, since old option prices
    aren't available, so it isn't part of this test. Earnings weeks are already left out."""
    if len(J) < WF_MIN_FIT + 52:
        return None
    rng = np.random.default_rng(seed)
    call = side == "call"
    entries = R["entry"].values
    exp = R["expiry"].values
    years = pd.DatetimeIndex(entries[J]).year
    ratio = np.ones(len(J))
    fitted = np.zeros(len(J), bool)
    refits = 0
    for yr in sorted(set(years)):
        start = np.datetime64(f"{yr}-01-01")
        known = exp[J] < start
        fit = _wf_fit(y, NB, J, rate, known, rng)
        m = years == yr
        if fit is None:
            continue
        refits += 1
        fitted[m] = True
        if fit["beats"]:
            for k in np.where(m)[0]:
                ratio[k] = _calibrated_ratio(fit["B"], fit["base"], rate[k] * 100)[0]
    if fitted.sum() < 52:
        return None
    applied = np.abs(ratio - 1) >= 0.1
    ratio = np.where(applied, ratio, 1.0)

    rows = R.iloc[J]
    flag_trend = rows["hot" if call else "sliding"].values.astype(bool)
    flag_lk = ratio >= fo.LK_FLAG_RATIO
    flags = flag_trend.astype(int) + flag_lk.astype(int)
    base = np.where(flag_trend, DELTA_HOT if call else PUT_DELTA_SLIDING, DELTA if call else PUT_DELTA)
    lim = fo.LK_CALL_DELTA_LIMITS if call else fo.LK_PUT_DELTA_LIMITS
    adj_delta = np.clip(base / ratio, lim[0], lim[1])
    adj_delta = np.where(applied, adj_delta, base)

    S, close = rows["S"].values, rows["close"].values
    T, vol, step = rows["T"].values, rows["vol"].values, rows["step"].values
    k_std = rows["kc" if call else "kp"].values
    k_adj = np.array([round_up(delta_strike(S[i], T[i], vol[i], adj_delta[i]), step[i]) if call else
                      round_down(put_delta_strike(S[i], T[i], vol[i], adj_delta[i]), step[i])
                      for i in range(len(J))])

    def outcome(K):
        prem = np.array([(bs_call if call else bs_put)(S[i], K[i], T[i], vol[i]) for i in range(len(K))]) * 100
        give = (np.maximum(0, close - K) if call else np.maximum(0, K - close)) * 100
        asg = close > K if call else close < K
        return prem, give, asg

    std, adj = outcome(k_std), outcome(k_adj)
    use = fitted                                  # compare from the first fit onward
    n_all = int(use.sum())

    def summary(key, label, sell, res):
        prem, give, asg = res
        m = use & sell
        net = prem[m] - give[m] - FEE
        return {"key": key, "label": label, "sold": int(m.sum()), "skipped": int((use & ~sell).sum()),
                "assigned": int(asg[m].sum()), "rate": round(float(asg[m].mean()) * 100, 1) if m.sum() else None,
                "premium": round(float(prem[m].sum()), 2), "given_up": round(float(give[m].sum()), 2),
                "fees": round(FEE * int(m.sum()), 2), "net": round(float(net.sum()), 2),
                "net_per_week": round(float(net.sum()) / n_all, 2),
                "net_per_sold": round(float(net.mean()), 2) if m.sum() else None,
                "worst": round(float(net.min()), 2) if m.sum() else None}

    every = np.ones(len(J), bool)
    site_sell = flags < 2
    fav_sell = flags == 0
    out_rows = [
        summary("every", "Every week (standard strike)", every, std),
        summary("every_adj", "Every week, strike adjusted", every, adj),
        summary("site", "Follow the site (skip hold-off weeks, strike adjusted)", site_sell, adj),
        summary("favorable", "Favorable weeks only, strike adjusted", fav_sell, adj),
        summary("favorable_std", "Favorable weeks only, standard strike", fav_sell, std),
    ]

    def skipped(sell):
        m = use & ~sell
        prem, give, asg = std
        net = prem[m] - give[m] - FEE
        return {"n": int(m.sum()), "assigned_if_sold": int(asg[m].sum()),
                "rate_if_sold": round(float(asg[m].mean()) * 100, 1) if m.sum() else None,
                "net_if_sold": round(float(net.sum()), 2), "net_per_if_sold": round(float(net.mean()), 2) if m.sum() else None}

    first = pd.Timestamp(entries[J][use][0]).date().isoformat()
    # per-trade decisions for the wheel simulation (not saved): entry date -> (ratio, flags)
    decisions = {pd.Timestamp(entries[J[i]]).date().isoformat(): (float(ratio[i]), int(flags[i])) for i in range(len(J))}
    return {"_decisions": decisions, "since": first, "weeks": n_all, "refits": refits,
            "weighting_on": int((use & applied).sum()),
            "flag_trend": int((use & flag_trend).sum()), "flag_lookalike": int((use & flag_lk).sum()),
            "base_rate": round(float(std[2][use].mean()) * 100, 1),
            "rows": out_rows,
            "skipped_site": skipped(site_sell), "skipped_favorable": skipped(fav_sell)}


def lookalike_side(R: pd.DataFrame, P: pd.DataFrame, core: list, side: str, current: dict | None, rng):
    y = R[side].astype(bool).values
    n = len(R)
    # ---- signal averages: assigned weeks vs weeks that expired worthless
    sig_rows = []
    for f in SIGNAL_LABELS:
        if f not in R or R[f].notna().sum() < 40 or R[f].nunique() < 3:
            continue
        eff, p, n1 = rs.group_test(R[f].astype(float), R[side], min_group=10)
        if p != p:
            continue
        a, b = R.loc[R[side], f].dropna(), R.loc[~R[side], f].dropna()
        sig_rows.append({"key": f, "label": SIGNAL_LABELS[f],
                         "pct_assigned": round(float(P.loc[R[side], f].mean()), 1),
                         "pct_expired": round(float(P.loc[~R[side], f].mean()), 1),
                         "median_assigned": shown(f, float(a.median())), "median_expired": shown(f, float(b.median())),
                         "n_assigned": int(len(a)), "n_expired": int(len(b)),
                         "effect": round(eff, 3), "p": round(p, 4)})
    passed = rs.bh_pass([r["p"] for r in sig_rows])
    for r, ok in zip(sig_rows, passed):
        r["found"] = bool(ok)
    sig_rows.sort(key=lambda r: r["p"])

    # ---- lookalikes: each week's nearest earlier weeks
    M = P[core].values.astype(float)
    neigh, dists = {}, {}
    for j in range(MIN_POOL, n):
        dj = _distances(M[:j], M[j])
        ok = np.where(~np.isnan(dj))[0]
        if len(ok) < MIN_POOL // 2:
            continue
        if len(ok) < K_TEST:
            continue
        near = ok[np.argsort(dj[ok], kind="stable")[:K_TEST]]
        neigh[j], dists[j] = near, dj[near]
    J = np.array(sorted(neigh))
    NB = np.array([neigh[j] for j in J]) if len(J) else np.empty((0, K_TEST), int)

    def gap(labels):
        rate = labels[NB].mean(axis=1)
        t = labels[J]
        if t.sum() == 0 or (~t).sum() == 0:
            return np.nan, rate
        return rate[t].mean() - rate[~t].mean(), rate

    g, rate = gap(y)
    sh = []
    for _ in range(SHUFFLES):
        s_, _r = gap(rng.permutation(y))
        if s_ == s_:
            sh.append(s_)
    sh = np.array(sh)
    t = y[J] if len(J) else np.array([], bool)
    buckets = []
    if len(J) >= 30:
        # weeks split into thirds by how often their lookalikes were assigned
        cuts = np.quantile(rate, [1 / 3, 2 / 3])
        grp = np.searchsorted(cuts, rate, side="right")
        for k, lab in enumerate(("Lookalikes least often assigned", "Middle third", "Lookalikes most often assigned")):
            m = grp == k
            buckets.append({"label": lab, "weeks": int(m.sum()), "assigned": int(t[m].sum()),
                            "lookalike_rate": round(float(rate[m].mean()) * 100, 1) if m.sum() else None,
                            "lo": round(float(rate[m].min()) * 100, 1) if m.sum() else None,
                            "hi": round(float(rate[m].max()) * 100, 1) if m.sum() else None,
                            "rate": round(float(t[m].mean()) * 100, 1) if m.sum() else None})
    test = {
        "weeks": int(len(J)), "assigned": int(t.sum()),
        "base_rate": round(float(t.mean()) * 100, 1) if len(J) else None,
        "rate_when_assigned": round(float(rate[t].mean()) * 100, 1) if t.sum() else None,
        "rate_when_expired": round(float(rate[~t].mean()) * 100, 1) if (~t).sum() else None,
        "gap": round(float(g) * 100, 1) if g == g else None,
        "shuffle_mean": round(float(sh.mean()) * 100, 1) if len(sh) else None,
        "shuffle_lo": round(float(np.quantile(sh, 0.025)) * 100, 1) if len(sh) else None,
        "shuffle_hi": round(float(np.quantile(sh, 0.975)) * 100, 1) if len(sh) else None,
        "p": round(float((np.sum(sh >= g) + 1) / (len(sh) + 1)), 4) if len(sh) and g == g else None,
        "buckets": buckets,
    }
    test["beats_chance"] = bool(test["p"] is not None and test["p"] < 0.05 and (test["gap"] or 0) > 0)

    def reading_list(get_raw, get_pct):
        # [shown value, rank 0-100] in the order of "core" (labels live there)
        return [[shown(f, get_raw(f)), None if get_pct(f) != get_pct(f) else round(float(get_pct(f)))] for f in core]

    def neighbor_list(idx, dd):
        out = []
        for i, d in zip(idx, dd):
            row = R.iloc[int(i)]
            # [entry date, assigned 1/0, move open -> expiry close %, similarity %]
            out.append([row["entry"].date().isoformat(), int(bool(row[side])),
                        round(float(row["ret"]) * 100, 1), round(100 - float(d))])
        return out

    detail = []
    for j in [j for j in range(n) if y[j] and j in neigh][-DETAIL_WEEKS:][::-1]:
        row = R.iloc[j]
        detail.append({"monday": row["entry"].date().isoformat(), "expiry": row["expiry"].date().isoformat(),
                       "open": round(float(row["S"]), 2), "close": round(float(row["close"]), 2),
                       "strike": round(float(row["kc" if side == "call" else "kp"]), 2),
                       "ret_pct": round(float(row["ret"]) * 100, 2),
                       "readings": reading_list(lambda f: row[f], lambda f: P.iloc[j][f]),
                       "lookalikes": neighbor_list(neigh[j][:K_NEAR], dists[j][:K_NEAR]),
                       "lookalikes_assigned": int(y[neigh[j][:K_NEAR]].sum()),
                       "rate_wide": round(float(y[neigh[j]].mean()) * 100, 1)})
    now = None
    if current:
        q = np.array([current["pct"].get(f, np.nan) for f in core], float)
        dq = _distances(M, q)
        ok = np.where(~np.isnan(dq))[0]
        if len(ok) >= K_TEST:
            near = ok[np.argsort(dq[ok], kind="stable")[:K_TEST]]
            now = {"as_of": current["as_of"],
                   "readings": reading_list(lambda f: current["raw"].get(f, np.nan), lambda f: current["pct"].get(f, np.nan)),
                   "lookalikes": neighbor_list(near[:K_NEAR], dq[near[:K_NEAR]]),
                   "lookalikes_assigned": int(y[near[:K_NEAR]].sum()),
                   "rate_wide": round(float(y[near].mean()) * 100, 1)}
    try:
        strategies = recommendation_backtest(R, y, NB, J, rate, side) if len(J) else None
    except Exception as exc:
        print(f"  recommendation backtest skipped: {exc}", file=sys.stderr)
        strategies = None
    return {"signals": sig_rows, "test": test, "detail": detail, "now": now, "strategies": strategies,
            "favorable": favorable(R, side), "assigned_weeks": int(y.sum()), "weeks": n}


def lookalikes(F: pd.DataFrame, weeks: list, news: pd.Series | None, news_now=None, seed=7):
    """weeks: the periods of ONE window (e.g. every Monday -> Friday)."""
    R = week_readings(F, weeks, news)
    if len(R) < MIN_POOL + 20:
        return None
    sig = [f for f in SIGNAL_LABELS if f in R]
    P = pd.DataFrame({f: R[f].astype(float).rank(pct=True) * 100 for f in sig})
    core = [f for f in CORE if f in R and R[f].notna().mean() >= 0.3 and R[f].nunique() >= 3]
    if len(core) < MIN_SHARED:
        return None
    # this week's readings as of the latest close (gap unknown until Monday's open)
    last = F.iloc[-1]
    raw = {f: last[f] for f in rs.FEATURES}
    for f in rs.ENTRY_DAY_FEATURES:
        raw[f] = np.nan
    raw.update(F.attrs.get("wiki_now") or {})
    raw.update({k: v for k, v in (F.attrs.get("analyst_now") or {}).items() if k in rs.ANALYST_FEATURES})
    raw["gap_open"] = np.nan
    raw["news_tone"] = news_now if news_now is not None else np.nan
    raw = {k: (np.nan if v is None else v) for k, v in raw.items()}
    current = {"as_of": F.index[-1].date().isoformat(), "raw": raw,
               "pct": {f: _pct_rank(R[f], raw.get(f)) for f in core}}
    rng = np.random.default_rng(seed)
    return {"core": [{"key": f, "label": SIGNAL_LABELS[f]} for f in core],
            "k": K_NEAR, "k_test": K_TEST, "min_pool": MIN_POOL, "shuffles": SHUFFLES,
            "call_delta": DELTA, "call_delta_hot": DELTA_HOT, "put_delta": PUT_DELTA, "put_delta_sliding": PUT_DELTA_SLIDING,
            "calls": lookalike_side(R, P, core, "call", current, rng),
            "puts": lookalike_side(R, P, core, "put", current, rng)}


# ------------------------------------------------------------ weighting the live recommendation
SHRINK_K = 50          # pulls a group's assignment rate toward the overall rate; 50 trades = half weight
RATIO_LIMITS = (0.5, 2.0)


def lookalike_adjustment(X: dict | None):
    """How much more (or less) likely assignment looks for the next trade, from its lookalikes.
    Used only when lookalikes have beaten chance for this ticker, window and option type.
    Calibrated on history rather than taken at face value: find the group of past trades whose
    lookalikes were assigned about as often as the next trade's are, and use how often THOSE trades
    actually got assigned, relative to all trades (shrunk toward 'no different' for small groups)."""
    if not X:
        return None
    T, now = X["test"], X.get("now")
    out = {"beats_chance": bool(T.get("beats_chance")), "p": T.get("p"), "base_rate": T.get("base_rate"),
           "rate_now": now and now.get("rate_wide"), "lookalikes_assigned": now and now.get("lookalikes_assigned"),
           "as_of": now and now.get("as_of"), "ratio": 1.0, "applied": False}
    B = [b for b in T.get("buckets") or [] if b.get("rate") is not None and b.get("lo") is not None]
    if not (out["beats_chance"] and now and B and T.get("base_rate")):
        return out
    ratio, b = _calibrated_ratio(B, T["base_rate"], now["rate_wide"])
    out.update({"group": b["label"], "group_rate": b["rate"], "group_trades": b["weeks"],
                "ratio": round(ratio, 2), "applied": abs(ratio - 1) >= 0.1})
    return out


def _calibrated_ratio(B, base_rate, r):
    """B: groups of past trades (by how often their lookalikes were assigned) with lo/hi/rate/weeks.
    Returns (ratio, group): the group's actual assignment rate over the base rate, shrunk toward 1."""
    inside = [b for b in B if b["lo"] <= r <= b["hi"]]
    b = inside[0] if inside else min(B, key=lambda b: min(abs(r - b["lo"]), abs(r - b["hi"])))
    raw = b["rate"] / base_rate if base_rate else 1.0
    w = b["weeks"] / (b["weeks"] + SHRINK_K)
    return min(max(1 + (raw - 1) * w, RATIO_LIMITS[0]), RATIO_LIMITS[1]), b


def lookalike_now(F: pd.DataFrame, earn, news=None, news_now=None, now_et=None):
    """For the daily study: lookalike adjustments for each window, as of the latest close."""
    out = {}
    for sk in ("weekly", "twice"):
        P = periods(F, earn, now_et, sk)
        for leg, _, _ in SCHEDULES[sk]:
            L = lookalikes(F, [w for w in P if w["leg"] == leg], news, news_now)
            if L:
                out[leg] = {"calls": lookalike_adjustment(L["calls"]), "puts": lookalike_adjustment(L["puts"])}
    return out


SCHEDULE_LABELS = {"weekly": "Once a week: Monday → Friday", "twice": "Twice a week: Monday → Wednesday, Thursday → Friday"}
WINDOW_LABELS = {"mon_fri": "Monday → Friday", "mon_wed": "Monday → Wednesday", "thu_fri": "Thursday → Friday"}


def backtest(ticker, px, earn, now_et=None, peers=(), wiki=None, analyst=None, news=None, news_now=None):
    """Returns (report, history). The report is what the page loads first; the history file holds
    every trade and every wheel step since the start, loaded only when you ask to see it all."""
    F = rs.build_features(px, ticker, list(peers), wiki, earn, analyst)
    per = {"weekly": week_info(F, earn, now_et), "twice": periods(F, earn, now_et, "twice")}
    tbill = None
    try:
        t = rs.load_prices(["^IRX"]).get("^IRX")
        if t is not None and not t.empty:
            tbill = rs._clean(t)["Close"].dropna()
    except Exception as exc:
        print(f"  T-bill rate unavailable, cash earns nothing: {exc}", file=sys.stderr)

    schedules, hist_rows, policy = {}, {}, {}
    for sk, P in per.items():
        rows = run_trades(P)
        if not rows:
            continue
        legs = [leg for leg, _, _ in SCHEDULES[sk] if any(w["leg"] == leg for w in P)]
        last_day = pd.Timestamp(rows[-1]["monday"]) - pd.Timedelta(days=364)
        recent = [pack_row(r) for r in rows
                  if pd.Timestamp(r["monday"]) >= last_day or any(r.get(m, {}).get("assigned") for m in ("delta", "study"))]
        schedules[sk] = {
            "label": SCHEDULE_LABELS[sk],
            "windows": [{"key": k, "label": WINDOW_LABELS[k]} for k in legs],
            "calls_only": {m: {"rules": summarize(rows, m, True), "all_weeks": summarize(rows, m, False)}
                           for m in ("delta", "study")},
            "earnings_trades": sum(1 for r in rows if r["earnings"]),
            # the last year of trades, plus every assigned one; the rest is in the history file
            "rows": recent, "rows_total": len(rows),
            "lookalikes": {k: _safe_lookalikes(F, [w for w in P if w["leg"] == k], news, news_now) for k in legs},
        }
        hist_rows[sk] = [pack_row(r) for r in rows]
        for leg, L in schedules[sk]["lookalikes"].items():
            for side, kind in (("calls", "call"), ("puts", "put")):
                st = (L or {}).get(side, {}).get("strategies") if L else None
                if st and "_decisions" in st:
                    policy.setdefault(leg, {})[kind] = st.pop("_decisions")

    wheels = {
        "wheel": {m: wheel_sim(per["weekly"], m, tbill) for m in ("delta", "study")},
        "wheel_topup": {m: wheel_sim(per["weekly"], m, tbill, topup=True) for m in ("delta", "study")},
        "wheel2": {m: wheel_sim(per["twice"], m, tbill, legs_per_week=2) for m in ("delta", "study")},
        "wheel2_topup": {m: wheel_sim(per["twice"], m, tbill, topup=True, legs_per_week=2) for m in ("delta", "study")},
    }
    # the same wheels, following the site's recommendation (see POLICIES). They go in a separate
    # file the page loads only when you pick one; the main file keeps just their summaries.
    policy_wheels = {}
    for mode in POLICIES:
        for base, sk, legs in (("wheel", "weekly", 1), ("wheel2", "twice", 2)):
            for topup in (False, True):
                key = base + ("_topup" if topup else "") + "_" + mode
                policy_wheels[key] = {m: wheel_sim(per[sk], m, tbill, topup=topup, legs_per_week=legs, policy=policy, mode=mode)
                                      for m in ("delta", "study")}
                for W in policy_wheels[key].values():
                    if W:              # every step, packed like the history file (see LOG_FIELDS)
                        W["log_full"] = [pack_log(e) for e in W.pop("log_all")]
    hist_logs = {}
    for k, bym in wheels.items():
        for m, W in bym.items():
            if W:
                hist_logs.setdefault(k, {})[m] = [pack_log(e) for e in W.pop("log_all")]
    updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    weekly = schedules.get("weekly", {})
    report = {
        "ticker": ticker, "updated": updated, "version": 2,
        "settings": {"delta": DELTA, "delta_hot": DELTA_HOT, "put_delta": PUT_DELTA, "put_delta_sliding": PUT_DELTA_SLIDING,
                     "study_rate": HIST_RATE, "put_study_rate": PUT_HIST_RATE,
                     "study_min_weeks": MIN_HISTORY_WEEKS, "fee": FEE},
        "row_fields": ROW_FIELDS, "log_fields": LOG_FIELDS,
        "schedules": schedules,
        # kept for pages from before the schedule switch (Monday -> Friday only)
        "summary": weekly.get("calls_only"),
        "cash_interest": tbill is not None,
        **wheels,
        "policies": POLICIES,
        "policy_summary": {k: {m: (W or {}).get("summary") for m, W in v.items()} for k, v in policy_wheels.items()},
    }
    history = {"ticker": ticker, "updated": updated, "row_fields": ROW_FIELDS, "log_fields": LOG_FIELDS,
               "calls_only": hist_rows, "wheel_logs": hist_logs}
    report["_policy_wheels"] = {"ticker": ticker, "updated": updated, **policy_wheels}
    return report, history


def _safe_lookalikes(F, weeks, news, news_now):
    try:
        return lookalikes(F, weeks, news, news_now)
    except Exception as exc:          # the rest of the backtest is still worth saving
        print(f"  lookalike analysis skipped: {exc}", file=sys.stderr)
        return None


def main():
    tickers, _ = rs.tickers_to_run(rs.WATCHLIST)
    cfg = json.loads(rs.CONFIG.read_text()) if rs.CONFIG.exists() else {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        px_market = rs.load_prices(list(rs.MARKET.values()))
    except Exception as exc:
        print(f"Market series unavailable: {exc}", file=sys.stderr)
        px_market = {}
    failed = 0
    for tk in tickers:
        try:
            d = rs.load_inputs(tk, cfg, px_market)
            news, news_now = rs.headline_tone(tk)
            rep, hist = backtest(tk, d["px"], d["earn"], peers=d["peers"], wiki=d["wiki"], analyst=d["analyst"],
                                 news=news, news_now=news_now)
            pol = rep.pop("_policy_wheels")
            (OUT_DIR / f"{tk}.json").write_text(json.dumps(rep, separators=(",", ":")))
            (OUT_DIR / f"{tk}_history.json").write_text(json.dumps(hist, separators=(",", ":")))
            (OUT_DIR / f"{tk}_policy.json").write_text(json.dumps(pol, separators=(",", ":")))
            s = rep["schedules"]["weekly"]["calls_only"]["delta"]["rules"]
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
