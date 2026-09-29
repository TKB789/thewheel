"""Data for the "Other strategies" tabs (iron condor, short strangle, bull put spread, long straddle, collar).

The page builds and prices each strategy itself, so you can change strikes and the pricing assumption
and see the backtest update. This file only exports what the page needs, per trade period:
the entry open, the exit close, time to expiration, the stock's recent volatility, earnings, trend,
and (for index funds) the market's implied volatility from a volatility index.

Why a volatility index: old option prices aren't free. For SPY, QQQ and IWM, Cboe publishes the
market's implied volatility every day (VIX9D / VIX, VXN, RVX), which is what the options were
actually priced at, so their backtests use it. Single stocks have no free equivalent, so the page
prices them from recent volatility times a ratio (measured from the chain snapshots once there are
enough, see chain_snapshot.py), and says so.

Called from backtest.py; writes data/backtest/<T>_strategies.json.
"""
from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from statistics import median

import pandas as pd

import backtest as bt
import research as rs

FIELDS = ["entry", "expiry", "open", "close", "days", "sessions", "hv20", "iv", "earnings",
          "pct_ma20", "rsi14", "hv_rank"]

# the volatility index that prices each index fund's options: short trades / 4-week trades
INDEX_IV = {
    "SPY": ("^VIX9D", "^VIX", "Cboe 9-day volatility index (VIX9D)", "Cboe volatility index (VIX)"),
    "QQQ": ("^VXN", "^VXN", "Cboe Nasdaq-100 volatility index (VXN)", "Cboe Nasdaq-100 volatility index (VXN)"),
    "IWM": ("^RVX", "^RVX", "Cboe Russell 2000 volatility index (RVX)", "Cboe Russell 2000 volatility index (RVX)"),
    "DIA": ("^VXD", "^VXD", "Cboe Dow volatility index (VXD)", "Cboe Dow volatility index (VXD)"),
}

WINDOW_LABELS = {"mon_wed": "Monday → Wednesday", "mon_fri": "Monday → Friday", "tue_wed": "Tuesday → Wednesday",
                 "thu_fri": "Thursday → Friday", "four_week": "Four weeks: Monday → Friday, 4 weeks later"}
WINDOW_ORDER = ["mon_fri", "mon_wed", "tue_wed", "thu_fri", "four_week"]
MIN_RATIO_DAYS = 10          # days of chain snapshots before the measured ratio is offered as the default


def four_week_periods(F: pd.DataFrame, earn_dates, now_et=None):
    """Monday open -> the last trading day four weeks later (usually Friday), one after another."""
    idx = F.index
    earn = [pd.Timestamp(x) for x in earn_dates]
    iso = idx.isocalendar()
    groups = [list(p) for _, p in pd.Series(range(len(idx)), index=idx).groupby([iso.year.values, iso.week.values])]
    today = pd.Timestamp((now_et or datetime.now(timezone.utc)).date())
    out, g = [], 0
    while g + 3 < len(groups):
        first, last = groups[g], groups[g + 3]
        i, j = first[0], last[-1]
        if idx[i].weekday() != 0 or i == 0 or idx[j].weekday() < 3:
            g += 1
            continue
        if j == len(idx) - 1 and idx[j] >= today:
            break                  # not settled yet
        prev = F.iloc[i - 1]
        S, close, vol = float(F["open"].iloc[i]), float(F["close"].iloc[j]), float(prev["hv20"])
        if vol > 0 and S > 0:
            out.append({"leg": "four_week", "e": idx[i], "x": idx[j], "S": S, "close": close, "vol": vol,
                        "T": ((idx[j] - idx[i]).days + 6.5 / 24) / 365.0, "sessions": j - i + 1,
                        "earnings": any(idx[i - 1] <= d <= idx[j] for d in earn)})
        g += 4
    return out


def _index_series(symbol):
    try:
        s = rs.load_prices([symbol]).get(symbol)
        if s is None or s.empty:
            return None
        return rs._clean(s)["Close"].dropna()
    except Exception:
        return None


def measured_ratio(ticker):
    """Median of implied / recent volatility from the daily chain snapshots (one per day)."""
    path = rs.ROOT / "data" / "chains" / "summary" / f"{ticker}.csv"
    if not path.exists():
        return None
    by_day = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                v = float(row["iv_hv"])
            except (KeyError, TypeError, ValueError):
                continue
            if 0.3 < v < 5:
                by_day[row["date"]] = v
    if not by_day:
        return None
    return {"median": round(median(by_day.values()), 3), "days": len(by_day),
            "first": min(by_day), "last": max(by_day)}


def build(ticker, F: pd.DataFrame, per: dict, earn_dates, now_et=None):
    F = F.copy()
    F["hv_rank"] = F["hv20"].rolling(252, min_periods=120).rank(pct=True)
    pos = {d: k for k, d in enumerate(F.index)}

    idx_cfg = INDEX_IV.get(ticker.upper())
    series = {}
    if idx_cfg:
        for sym in set(idx_cfg[:2]):
            s = _index_series(sym)
            if s is not None:
                series[sym] = s

    windows = {}
    taken, sources = {}, []            # thu_fri is in two schedules: take it once
    for sk in ("weekly", "twice", "tuewed"):
        for w in per.get(sk) or []:
            if taken.setdefault(w["leg"], sk) == sk:
                sources.append(w)
    sources += four_week_periods(F, earn_dates, now_et)

    for w in sources:
        leg = w["leg"]
        i = pos.get(w["e"])
        if i is None or i == 0:
            continue
        prev = F.iloc[i - 1]
        sessions = w.get("sessions") or max(1, round((w["sig_w"] / w["vol"]) ** 2 * 252))
        iv = None
        if idx_cfg:
            sym = idx_cfg[1] if leg == "four_week" else idx_cfg[0]
            s = series.get(sym)
            if s is not None:
                v = s.asof(F.index[i - 1])
                if v == v and v > 0:
                    iv = round(float(v) / 100, 4)
        rank = prev["hv_rank"]
        windows.setdefault(leg, []).append([
            w["e"].date().isoformat(), w["x"].date().isoformat(), round(w["S"], 2), round(w["close"], 2),
            round(w["T"] * 365, 3), int(sessions), round(float(w["vol"]), 4), iv, 1 if w["earnings"] else 0,
            round(float(prev["pct_ma20"]), 4) if prev["pct_ma20"] == prev["pct_ma20"] else None,
            round(float(prev["rsi14"]), 1) if prev["rsi14"] == prev["rsi14"] else None,
            round(float(rank), 3) if rank == rank else None,
        ])
    out_windows = {k: {"label": WINDOW_LABELS.get(k, k), "rows": sorted(windows[k])}
                   for k in WINDOW_ORDER if windows.get(k)}
    has_iv = idx_cfg and any(r[7] is not None for v in out_windows.values() for r in v["rows"])
    return {
        "ticker": ticker,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fields": FIELDS,
        "fee": bt.FEE,
        "risk_free": bt.RISK_FREE,
        "iv_source": ({"kind": "index", "short": idx_cfg[2], "long": idx_cfg[3]} if has_iv else {"kind": "hv"}),
        "measured_ratio": measured_ratio(ticker),
        "windows": out_windows,
    }
