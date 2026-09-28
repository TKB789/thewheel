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


def wheel_sim(weeks, method, tbill=None, topup=False, legs_per_week=1):
    """The wheel: start owning 100 shares with no cash. Holding shares -> sell a covered call each
    Monday. Called away -> hold the cash and sell a cash-secured put each Monday, but only if the
    cash covers strike x 100; otherwise that week is paused. Put assigned -> buy 100 shares at the
    strike and go back to selling calls. Earnings weeks are skipped in both phases.
    Cash earns the 3-month Treasury bill rate each week, roughly what a money market fund
    (like Fidelity's core position) pays, when that rate series is available.

    topup=True: instead of pausing, add just enough money to cover the put and keep going.
    The benchmark then also buys the stock with each deposit on the same day, so the
    comparison isn't flattered by the extra money."""
    shares, cash = 100, 0.0
    histories, timeline, events, log = {}, [], [], []
    c = {"call_weeks": 0, "put_weeks": 0, "paused": 0, "earnings_skipped": 0, "no_history": 0,
         "calls_assigned": 0, "puts_assigned": 0, "call_premium": 0.0, "put_premium": 0.0, "fees": 0.0,
         "longest_pause": 0, "interest": 0.0, "added": 0.0, "topups": 0, "largest_topup": 0.0}
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
              "trades_per_week": legs_per_week})
    timeline = [[t[0], round(t[1]), round(t[2]), t[3]] for t in timeline]     # whole dollars for the chart
    return {"summary": c, "timeline": timeline, "events": events[-200:], "events_total": len(events),
            "log": log[-52 * legs_per_week:]}


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
DETAIL_WEEKS = 30      # assigned weeks listed in full, per side
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
    """One row per non-earnings Monday -> Friday week: the readings known at Monday's open
    (previous close for price-based signals; the entry day for page views, analysts, headlines)
    and whether the site's standard call / put would have been assigned."""
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
        r.update({"entry": w["e"], "expiry": w["x"], "S": w["S"], "close": w["close"],
                  "ret": w["close"] / w["S"] - 1, "kc": kc, "kp": kp,
                  "call": w["close"] > kc, "put": w["close"] < kp})
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
        return [{"key": f, "value": shown(f, get_raw(f)),       # labels are in "core"
                 "pct": None if get_pct(f) != get_pct(f) else round(float(get_pct(f)))} for f in core]

    def neighbor_list(idx, dd):
        out = []
        for i, d in zip(idx, dd):
            row = R.iloc[int(i)]
            out.append({"monday": row["entry"].date().isoformat(), "assigned": bool(row[side]),
                        "ret_pct": round(float(row["ret"]) * 100, 2), "similarity": round(100 - float(d), 1)})
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
    return {"signals": sig_rows, "test": test, "detail": detail, "now": now,
            "assigned_weeks": int(y.sum()), "weeks": n}


def lookalikes(F: pd.DataFrame, weeks: list, news: pd.Series | None, news_now=None, seed=7):
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


def backtest(ticker, px, earn, now_et=None, peers=(), wiki=None, analyst=None, news=None, news_now=None):
    F = rs.build_features(px, ticker, list(peers), wiki, earn, analyst)
    rows = run_weeks(F, earn, now_et)
    weeks = week_info(F, earn, now_et)
    twice = periods(F, earn, now_et, "twice")
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
        "wheel_topup": {m: wheel_sim(weeks, m, tbill, topup=True) for m in ("delta", "study")},
        "wheel2": {m: wheel_sim(twice, m, tbill, legs_per_week=2) for m in ("delta", "study")},
        "wheel2_topup": {m: wheel_sim(twice, m, tbill, topup=True, legs_per_week=2) for m in ("delta", "study")},
        "cash_interest": tbill is not None,
        "lookalikes": _safe_lookalikes(F, weeks, news, news_now),
        # keeps the file small: the last year of weeks, plus every week that got assigned
        "weeks": [r for i, r in enumerate(rows)
                  if i >= len(rows) - 52 or any(r.get(m, {}).get("assigned") for m in ("delta", "study"))],
    }


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
            rep = backtest(tk, d["px"], d["earn"], peers=d["peers"], wiki=d["wiki"], analyst=d["analyst"],
                           news=news, news_now=news_now)
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
