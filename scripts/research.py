#!/usr/bin/env python3
"""
Correlation study for short-dated covered calls.

For each ticker in watchlist.txt this looks back ~10 years and asks: which
publicly available numbers, known at the time you sell, lined up with how far
the stock rose by expiry? It studies the three windows you trade:

    Monday open    -> Wednesday close
    Thursday open  -> Friday close
    Thursday open  -> next Monday close

Guarding against flukes:
  * Every signal uses only data available before you sell (prior close, plus
    the opening gap, which you can see when you sell after the open).
  * Weeks that contain an earnings report are left out (you skip those anyway).
  * Patterns are found on the first 70% of history, with a false-discovery
    correction for testing many signals, then must repeat on the last 30%.
  * Only signals that hold up on both are used to adjust strike suggestions.

Output: data/research/<TICKER>.json, read by index.html.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "research"
WATCHLIST = ROOT / "watchlist.txt"
CONFIG = ROOT / "research_config.json"
FIDELITY_LOG = ROOT / "fidelity_log.csv"

MARKET = {"QQQ": "QQQ", "SMH": "SMH", "VIX": "^VIX", "VIX9D": "^VIX9D",
          "TNX": "^TNX", "DXY": "DX-Y.NYB", "OIL": "CL=F"}

WINDOWS = [
    {"key": "mon_wed", "label": "Monday → Wednesday", "entry_dow": 0, "exit_offset": 2, "sessions": 3},
    {"key": "thu_fri", "label": "Thursday → Friday", "entry_dow": 3, "exit_offset": 1, "sessions": 2},
    {"key": "thu_mon", "label": "Thursday → next Monday", "entry_dow": 3, "exit_offset": 4, "sessions": 3},
]

LABELS = {
    "gap_open": "Opening gap on the sell day",
    "ret_1d": "Previous day's move",
    "ret_5d": "Past week's move",
    "ret_20d": "Past month's move",
    "rsi14": "RSI (14-day)",
    "pct_ma20": "Distance above 20-day average",
    "pct_ma50": "Distance above 50-day average",
    "hv20": "Recent volatility (20-day)",
    "vol_expansion": "Volatility picking up (5-day vs 20-day)",
    "volume_ratio": "Previous day's volume vs normal",
    "range_ratio": "Previous day's trading range vs normal",
    "rel_qqq_5d": "Past week vs Nasdaq-100",
    "qqq_5d": "Nasdaq-100 (QQQ) past week",
    "smh_5d": "Chip stocks (SMH) past week",
    "peers_1d": "Peer stocks' previous day",
    "peers_5d": "Peer stocks' past week",
    "vix": "VIX level",
    "vix_chg_5d": "VIX change over the past week",
    "vix_term": "Short-term fear vs 30-day (VIX9D ÷ VIX)",
    "tnx_chg_5d": "10-year Treasury yield change, past week",
    "dxy_5d": "US dollar index, past week",
    "oil_5d": "Oil price, past week",
    "wiki_spike": "Wikipedia page views vs normal",
    "wiki_trend": "Wikipedia page views, week over week",
    "days_to_earn": "Days until next earnings",
}
FEATURES = list(LABELS)
ENTRY_DAY_FEATURES = ("wiki_spike", "wiki_trend")

TARGETS = {
    "direction": "how far it rose (or fell)",
    "size": "how big the move was versus normal",
    "high_end": "rises past one expected move when the reading is in its top fifth",
    "low_end": "rises past one expected move when the reading is in its bottom fifth",
}

TARGET_ASSIGN = 0.15   # aim strikes at ~15% historical assignment (≈ 0.15 delta)
TRAIN_FRAC = 0.70
FDR_Q = 0.10
TEST_P = 0.10
MIN_COND = 40
FIDELITY_MIN = 20
STRIKE_STEPS = [0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6]
PAIR_MIN_GROUP = 15    # smallest "both high" style group in the earlier years
PAIR_MIN_GROUP_TEST = 10  # and in the recent years, where combinations are rarer


# ------------------------------------------------------------------- stats
def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def spearman(a: pd.Series, b: pd.Series):
    m = a.notna() & b.notna() & np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    n = len(a)
    if n < 25 or a.nunique() < 5:
        return float("nan"), float("nan"), n
    r = a.rank().corr(b.rank())
    if not np.isfinite(r):
        return float("nan"), float("nan"), n
    r_c = max(min(r, 0.999999), -0.999999)
    z = math.atanh(r_c) * math.sqrt(max(n - 3, 1))
    return float(r), float(2 * (1 - norm_cdf(abs(z)))), n


def bh_pass(pvals, q=FDR_Q):
    """Benjamini-Hochberg: which p-values survive testing many signals at once."""
    idx = [i for i, p in enumerate(pvals) if p == p]
    ranked = sorted(idx, key=lambda i: pvals[i])
    m = len(ranked)
    cutoff = -1
    for k, i in enumerate(ranked, 1):
        if pvals[i] <= q * k / m:
            cutoff = k
    keep = set(ranked[:cutoff]) if cutoff > 0 else set()
    return [i in keep for i in range(len(pvals))]


def group_test(y: pd.Series, g: pd.Series, min_group=15):
    """Mann-Whitney: are y values different inside group g than outside it?
    Returns (rank-biserial effect, p, n_group). Positive = higher inside the group."""
    m = y.notna() & np.isfinite(y) & g.notna()
    y, g = y[m], g[m].astype(bool)
    n1, n2 = int(g.sum()), int((~g).sum())
    if n1 < min_group or n2 < min_group:
        return float("nan"), float("nan"), n1
    ranks = y.rank()
    u = ranks[g].sum() - n1 * (n1 + 1) / 2
    mu, sd = n1 * n2 / 2, math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u - mu) / sd
    return float(2 * u / (n1 * n2) - 1), float(2 * (1 - norm_cdf(abs(z)))), n1


def tail_test(x: pd.Series, y: pd.Series, lo: float, hi: float):
    """Do moves differ when the reading is inside [lo, hi] vs outside?"""
    g = x.between(lo, hi).where(x.notna())
    return group_test(y, g)


# ---------------------------------------------------------------- features
def _clean(df):
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def build_features(px: dict, ticker: str, peers: list, wiki: pd.Series | None, earn_dates: list):
    s = _clean(px[ticker]).dropna(subset=["Open", "Close"])
    c = s["Close"]
    F = pd.DataFrame(index=s.index)
    F["open"], F["close"] = s["Open"], c
    lr = np.log(c).diff()
    F["ret_1d"] = c.pct_change()
    F["ret_5d"] = c.pct_change(5)
    F["ret_20d"] = c.pct_change(20)
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    F["rsi14"] = 100 - 100 / (1 + gain / loss)
    F["pct_ma20"] = c / c.rolling(20).mean() - 1
    F["pct_ma50"] = c / c.rolling(50).mean() - 1
    F["hv20"] = lr.rolling(20).std() * math.sqrt(252)
    F["vol_expansion"] = (lr.rolling(5).std() * math.sqrt(252)) / F["hv20"]
    F["volume_ratio"] = s["Volume"] / s["Volume"].rolling(20).mean()
    rng = (s["High"] - s["Low"]) / c
    F["range_ratio"] = rng / rng.rolling(20).mean()

    def close_of(sym):
        if sym not in px or px[sym] is None or px[sym].empty:
            return None
        return _clean(px[sym])["Close"].reindex(F.index).ffill()

    q = close_of(MARKET["QQQ"])
    if q is not None:
        F["qqq_5d"] = q.pct_change(5)
        F["rel_qqq_5d"] = F["ret_5d"] - F["qqq_5d"]
    smh = close_of(MARKET["SMH"])
    if smh is not None:
        F["smh_5d"] = smh.pct_change(5)
    pc = [close_of(p) for p in peers]
    pc = [x for x in pc if x is not None]
    if pc:
        P = pd.concat(pc, axis=1)
        F["peers_1d"] = P.pct_change().mean(axis=1)
        F["peers_5d"] = P.pct_change(5).mean(axis=1)
    vix = close_of(MARKET["VIX"])
    if vix is not None:
        F["vix"] = vix
        F["vix_chg_5d"] = vix.diff(5)
        v9 = close_of(MARKET["VIX9D"])
        if v9 is not None:
            F["vix_term"] = v9 / vix
    tnx = close_of(MARKET["TNX"])
    if tnx is not None:
        F["tnx_chg_5d"] = tnx.diff(5)
    dxy = close_of(MARKET["DXY"])
    if dxy is not None:
        F["dxy_5d"] = dxy.pct_change(5)
    oil = close_of(MARKET["OIL"])
    if oil is not None:
        F["oil_5d"] = oil.where(oil > 1).pct_change(5)
    if wiki is not None and len(wiki) > 60:
        w = wiki.astype(float)
        w = w.asfreq("D")
        spike = (w / w.shift(1).rolling(28, min_periods=20).median()).rolling(3, min_periods=1).max()
        trend = w.rolling(7).sum() / w.shift(7).rolling(7).sum()
        # Stored on each trading day as "known before that day's open": the 3 calendar days
        # ending the day before (so a Monday sale sees Friday, Saturday and Sunday).
        F["wiki_spike"] = spike.shift(1).reindex(F.index)
        F["wiki_trend"] = trend.shift(1).reindex(F.index)
        F.attrs["wiki_now"] = {"wiki_spike": float(spike.dropna().iloc[-1]) if spike.notna().any() else None,
                               "wiki_trend": float(trend.dropna().iloc[-1]) if trend.notna().any() else None}
    if earn_dates:
        ed = np.array(sorted(pd.Timestamp(x) for x in earn_dates), dtype="datetime64[ns]")
        pos = np.searchsorted(ed, F.index.values, side="left")
        days = [((ed[p] - t) / np.timedelta64(1, "D")) if p < len(ed) else np.nan
                for p, t in zip(pos, F.index.values)]
        F["days_to_earn"] = days
    for f in FEATURES:
        if f not in F:
            F[f] = np.nan
    return F.replace([np.inf, -np.inf], np.nan)


def make_events(F: pd.DataFrame, win: dict, earn_dates: list):
    idx = F.index
    pos = {d: i for i, d in enumerate(idx)}
    earn = [pd.Timestamp(x) for x in earn_dates]
    rows = []
    for i in range(1, len(idx)):
        e = idx[i]
        if e.weekday() != win["entry_dow"]:
            continue
        x = e + timedelta(days=win["exit_offset"])
        if x not in pos:
            continue
        p = idx[i - 1]
        prev = F.iloc[i - 1]
        sig = prev["hv20"] * math.sqrt(win["sessions"] / 252)
        if not (sig > 0):
            continue
        ret = F.at[x, "close"] / F.at[e, "open"] - 1
        row = {f: prev[f] for f in FEATURES}
        for f in ENTRY_DAY_FEATURES:          # already defined as "known before the open"
            row[f] = F.iloc[i][f]
        row["gap_open"] = F.at[e, "open"] / prev["close"] - 1
        row.update({"entry": e, "exit": x, "ret": ret, "sigma": sig, "z": ret / sig,
                    "earnings": any(p <= d <= x for d in earn)})
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- study
PROPER = ("Nasdaq", "VIX", "US ", "Wikipedia", "RSI", "Chip", "Oil", "Peer", "Fidelity")


def in_sentence(label):
    """Lowercase a label's first letter unless it starts with a name or abbreviation."""
    return label if label.startswith(PROPER[:5]) else label[0].lower() + label[1:]


def effect_text(t, ticker, exit_day):
    label, a, b = in_sentence(t["label"]), t["top_rate"], t["bottom_rate"]
    if t["kind"] == "tail":
        where = "in its top fifth" if t["target"] == "high_end" else "in its bottom fifth"
        return (f"When {label} was {where}, {ticker} rose past one expected move by {exit_day} "
                f"{a:.0%} of the time, vs {b:.0%} otherwise (average move {t['top_mean']:+.2%} vs {t['bottom_mean']:+.2%}).")
    if t["target"] == "size":
        if t["r_test"] > 0:
            hi_lo, first, second, x, y = "high", "top", "bottom", a, b
        else:
            hi_lo, first, second, x, y = "low", "bottom", "top", b, a
        return (f"When {label} was {hi_lo}, {ticker}'s moves to {exit_day} ran bigger than usual, in both directions. "
                f"It rose past one expected move {x:.0%} of the time in the {first} fifth of readings "
                f"vs {y:.0%} in the {second} fifth.")
    lead = "rose further" if t["r_test"] > 0 else "rose less"
    return (f"When {label} was high, {ticker} {lead} by {exit_day}: averaging "
            f"{t['top_mean']:+.2%} in the top fifth of readings vs {t['bottom_mean']:+.2%} in the bottom fifth. "
            f"It rose past one expected move {a:.0%} vs {b:.0%} of the time.")


def study_window(ev: pd.DataFrame, win: dict, ticker: str):
    ev = ev[~ev["earnings"]].sort_values("entry").reset_index(drop=True)
    n = len(ev)
    split = int(n * TRAIN_FRAC)
    train, test = ev.iloc[:split], ev.iloc[split:]
    ev = ev.assign(size=ev["z"].abs())
    train = train.assign(size=train["z"].abs())
    test = test.assign(size=test["z"].abs())
    exit_day = win["label"].split("→")[1].strip().replace("next ", "")

    tests = []
    for f in FEATURES:
        for tgt, col in (("direction", "ret"), ("size", "size")):
            r1, p1, n1 = spearman(train[f], train[col])
            r2, p2, n2 = spearman(test[f], test[col])
            tests.append({"feature": f, "label": LABELS[f], "target": tgt, "kind": "rank",
                          "r_train": r1, "p_train": p1, "n_train": n1,
                          "r_test": r2, "p_test": p2, "n_test": n2})
        # threshold tests: cut points come from the training years only, then applied to the test years
        tv = train[f].dropna()
        if len(tv) >= 50 and tv.nunique() >= 5:
            q20, q80 = float(tv.quantile(0.2)), float(tv.quantile(0.8))
            for tgt, lo, hi in (("high_end", q80, float("inf")), ("low_end", float("-inf"), q20)):
                r1, p1, n1 = tail_test(train[f], train["z"], lo, hi)
                r2, p2, n2 = tail_test(test[f], test["z"], lo, hi)
                tests.append({"feature": f, "label": LABELS[f], "target": tgt, "kind": "tail",
                              "lo": lo, "hi": hi,
                              "r_train": r1, "p_train": p1, "n_train": n1,
                              "r_test": r2, "p_test": p2, "n_test": n2})
    passes = bh_pass([t["p_train"] for t in tests])
    for t, ok in zip(tests, passes):
        same_sign = t["r_train"] == t["r_train"] and t["r_test"] == t["r_test"] and \
            np.sign(t["r_train"]) == np.sign(t["r_test"])
        t["found"] = bool(ok)
        t["holds"] = bool(ok and same_sign and t["p_test"] < TEST_P)
        if t["holds"]:
            v = ev[t["feature"]]
            if t["kind"] == "tail":
                inside = (v >= t["lo"]) & (v <= t["hi"])
                top, bot = ev[inside], ev[~inside & v.notna()]
            else:
                top, bot = ev[v >= v.quantile(0.8)], ev[v <= v.quantile(0.2)]
            t.update({
                "top_rate": float((top["z"] > 1).mean()), "bottom_rate": float((bot["z"] > 1).mean()),
                "top_mean": float(top["ret"].mean()), "bottom_mean": float(bot["ret"].mean()),
            })
            t["text"] = effect_text(t, ticker, exit_day)
        for k in ("r_train", "p_train", "r_test", "p_test"):
            t[k] = None if t[k] != t[k] else round(t[k], 4)
        for k in ("lo", "hi"):
            if k in t and not np.isfinite(t[k]):
                t[k] = None

    pairs, n_pair_tests, n_usable = study_pairs(ev, train, test, ticker, exit_day, tests)

    baseline = {
        "n": n, "n_train": len(train), "n_test": len(test),
        "pair_tests": n_pair_tests, "pairs_of": n_usable * (n_usable - 1) // 2,
        "first": ev["entry"].min().date().isoformat() if n else None,
        "last": ev["entry"].max().date().isoformat() if n else None,
        "mean_ret": float(ev["ret"].mean()) if n else None,
        "up_rate": float((ev["ret"] > 0).mean()) if n else None,
        "beyond_1_move": float((ev["z"] > 1).mean()) if n else None,
    }
    return ev, tests, pairs, baseline


CORNERS = {
    "both_high": ("high", "high"), "both_low": ("low", "low"),
    "high_low": ("high", "low"), "low_high": ("low", "high"),
}


def _third(df, f, side, cuts):
    lo, hi = cuts[f]
    v = df[f]
    g = (v >= hi) if side == "high" else (v <= lo)
    return g.where(v.notna())


def pair_text(t, ticker, exit_day):
    a, b = in_sentence(t["label_a"]), in_sentence(t["label_b"])
    if t["kind"] == "pair_corner":
        ta = "top" if t["side_a"] == "high" else "bottom"
        tb = "top" if t["side_b"] == "high" else "bottom"
        return (f"When {a} was in its {ta} third and {b} was in its {tb} third at the same time, "
                f"{ticker} rose past one expected move by {exit_day} {t['top_rate']:.0%} of the time, "
                f"vs {t['bottom_rate']:.0%} otherwise (average move {t['top_mean']:+.2%} vs {t['bottom_mean']:+.2%}). "
                f"This beat either signal on its own.")
    together = "pointed the same way (both high or both low)" if t["r_test"] > 0 else "pointed opposite ways"
    if t["target"] == "size":
        return (f"When {a} and {b} {together}, {ticker}'s moves to {exit_day} ran bigger than usual: "
                f"past one expected move {t['top_rate']:.0%} of the time vs {t['bottom_rate']:.0%}.")
    return (f"When {a} and {b} {together}, {ticker} rose further by {exit_day}: averaging "
            f"{t['top_mean']:+.2%} vs {t['bottom_mean']:+.2%}.")


def study_pairs(ev, train, test, ticker, exit_day, single_tests):
    """Test every pair of signals, two ways:
    - product of percentile ranks (does it matter when both move together?)
    - corners: both high, both low, or one high and one low (top/bottom thirds)
    A pair only counts if it beats each of its two signals alone on the recent years."""
    usable = [f for f in FEATURES if train[f].notna().sum() >= 100 and train[f].nunique() >= 5]
    cuts = {f: (float(train[f].quantile(1 / 3)), float(train[f].quantile(2 / 3))) for f in usable}
    pr_train = {f: train[f].rank(pct=True) - 0.5 for f in usable}
    pr_test = {f: test[f].rank(pct=True) - 0.5 for f in usable}
    single_r = {}
    for f in usable:
        for tgt, col in (("direction", "ret"), ("size", "size")):
            single_r[(f, tgt)] = spearman(test[f], test[col])[0]
    single_third = {}
    for f in usable:
        for side in ("high", "low"):
            single_third[(f, side)] = group_test(test["z"], _third(test, f, side, cuts))[0]

    tests = []
    for a, b in itertools.combinations(usable, 2):
        for tgt, col in (("direction", "ret"), ("size", "size")):
            r1, p1, n1 = spearman(pr_train[a] * pr_train[b], train[col])
            r2, p2, n2 = spearman(pr_test[a] * pr_test[b], test[col])
            best_single = max(abs(single_r[(a, tgt)] or 0), abs(single_r[(b, tgt)] or 0))
            tests.append({"kind": "pair_product", "a": a, "b": b, "target": tgt,
                          "r_train": r1, "p_train": p1, "n_train": n1, "r_test": r2, "p_test": p2, "n_test": n2,
                          "beats_parts": bool(r2 == r2 and abs(r2) > best_single)})
        for key, (sa, sb) in CORNERS.items():
            g1 = _third(train, a, sa, cuts) * _third(train, b, sb, cuts)
            g2 = _third(test, a, sa, cuts) * _third(test, b, sb, cuts)
            r1, p1, n1 = group_test(train["z"], g1, PAIR_MIN_GROUP)
            r2, p2, n2 = group_test(test["z"], g2, PAIR_MIN_GROUP_TEST)
            parts = [single_third[(a, sa)], single_third[(b, sb)]]
            parts = [x for x in parts if x == x]
            beats = bool(r2 == r2 and all(np.sign(r2) != np.sign(x) or abs(r2) > abs(x) for x in parts))
            tests.append({"kind": "pair_corner", "a": a, "b": b, "corner": key, "side_a": sa, "side_b": sb,
                          "target": "corner", "cuts_a": cuts[a], "cuts_b": cuts[b],
                          "r_train": r1, "p_train": p1, "n_train": n1, "r_test": r2, "p_test": p2, "n_test": n2,
                          "beats_parts": beats})

    passes = bh_pass([t["p_train"] for t in tests])
    kept = []
    for t, ok in zip(tests, passes):
        if not ok:
            continue
        same_sign = t["r_test"] == t["r_test"] and np.sign(t["r_train"]) == np.sign(t["r_test"])
        t["found"] = True
        t["holds"] = bool(same_sign and t["p_test"] < TEST_P and t["beats_parts"])
        t["label_a"], t["label_b"] = LABELS[t["a"]], LABELS[t["b"]]
        t["feature"] = f"{t['a']}|{t['b']}"
        t["label"] = f"{LABELS[t['a']]} + {LABELS[t['b']]}"
        if t["holds"]:
            if t["kind"] == "pair_corner":
                g = (_third(ev, t["a"], t["side_a"], cuts) * _third(ev, t["b"], t["side_b"], cuts))
                top, bot = ev[g == 1], ev[g == 0]
            else:
                prod = (ev[t["a"]].rank(pct=True) - 0.5) * (ev[t["b"]].rank(pct=True) - 0.5)
                top, bot = ev[prod >= prod.quantile(0.8)], ev[prod <= prod.quantile(0.2)]
                if t["r_test"] < 0:
                    top, bot = bot, top
            t.update({"top_rate": float((top["z"] > 1).mean()), "bottom_rate": float((bot["z"] > 1).mean()),
                      "top_mean": float(top["ret"].mean()), "bottom_mean": float(bot["ret"].mean()),
                      "n_group": int(len(top))})
            t["text"] = pair_text(t, ticker, exit_day)
        for k in ("r_train", "p_train", "r_test", "p_test"):
            t[k] = None if t[k] != t[k] else round(t[k], 4)
        kept.append(t)
    return kept, len(tests), len(usable)


def suggest(ev: pd.DataFrame, holding: list, current: dict, sigma_now: float):
    """Pick the strike distance that historically finished in the money ~15% of the time,
    using only past weeks that looked like today on the signals that held up."""
    sample = ev
    used = []
    for t in sorted(holding, key=lambda t: -abs(t["r_test"] or 0)):
        if len(used) == 2:
            break
        f = t["feature"]
        if t["kind"] == "pair_product":
            continue
        if t["kind"] == "pair_corner":
            ca, cb = current.get(t["a"]), current.get(t["b"])
            if ca is None or cb is None or ca != ca or cb != cb:
                continue
            def side_ok(v, side, cuts):
                return v >= cuts[1] if side == "high" else v <= cuts[0]
            def side_mask(col, side, cuts):
                return (sample[col] >= cuts[1]) if side == "high" else (sample[col] <= cuts[0])
            # a combination only says something when today is inside it
            if not (side_ok(ca, t["side_a"], t["cuts_a"]) and side_ok(cb, t["side_b"], t["cuts_b"])):
                continue
            both = side_mask(t["a"], t["side_a"], t["cuts_a"]) & side_mask(t["b"], t["side_b"], t["cuts_b"])
            sub = sample[both]
            if len(sub) >= MIN_COND // 2:
                sample = sub
                used.append(f"{t['label_a']} {t['side_a']} and {in_sentence(t['label_b'])} {t['side_b']}, together")
            continue
        cur = current.get(f)
        if cur is None or cur != cur:
            continue
        if t["kind"] == "tail":
            lo = -np.inf if t["lo"] is None else t["lo"]
            hi = np.inf if t["hi"] is None else t["hi"]
            inside = lo <= cur <= hi
            mask = sample[f].between(lo, hi) if inside else ~sample[f].between(lo, hi) & sample[f].notna()
            fifth = "top" if t["target"] == "high_end" else "bottom"
            name = f"in its {fifth} fifth" if inside else f"outside its {fifth} fifth"
        else:
            lo, hi = sample[f].quantile(1 / 3), sample[f].quantile(2 / 3)
            if cur <= lo:
                mask, name = sample[f] <= lo, "low"
            elif cur >= hi:
                mask, name = sample[f] >= hi, "high"
            else:
                mask, name = (sample[f] > lo) & (sample[f] < hi), "in the middle"
        sub = sample[mask]
        if len(sub) >= MIN_COND and f"{LABELS[f]}" not in " ".join(used):
            sample = sub
            used.append(f"{LABELS[f]} is {name} now")
    zq = float(sample["z"].quantile(1 - TARGET_ASSIGN))
    move = max(zq * sigma_now, 0.0025)
    table = [{"move_pct": k, "rate": float((sample["z"] * sigma_now > k / 100).mean())} for k in STRIKE_STEPS]
    return {
        "sigma_pct": round(sigma_now * 100, 3),
        "move_pct": round(move * 100, 3),
        "target_rate": TARGET_ASSIGN,
        "n_sample": int(len(sample)),
        "conditioned_on": used,
        "exceed_table": table,
    }


def fidelity_study(F: pd.DataFrame, ticker: str):
    if not FIDELITY_LOG.exists():
        return {"n": 0, "status": "collecting", "need": FIDELITY_MIN}
    rows = []
    with FIDELITY_LOG.open() as fh:
        for r in csv.DictReader(fh):
            if (r.get("ticker") or "").strip().upper() != ticker:
                continue
            try:
                d = pd.Timestamp(r["date"].strip())
                rows.append({"date": d, "starmine": float(r["starmine"]), "sscore": float(r["sscore"])})
            except (KeyError, ValueError):
                continue
    idx = list(F.index)
    pos = {d: i for i, d in enumerate(idx)}
    data = []
    for r in rows:
        i = pos.get(r["date"])
        if i is None or i + 2 >= len(idx) or i < 1:
            continue
        sig = F["hv20"].iloc[i - 1] * math.sqrt(3 / 252)
        ret = F["close"].iloc[i + 2] / F["open"].iloc[i] - 1
        data.append({**r, "ret": ret, "size": abs(ret / sig) if sig > 0 else np.nan})
    n = len(data)
    out = {"n": n, "logged": len(rows)}
    if n < FIDELITY_MIN:
        out.update({"status": "collecting", "need": FIDELITY_MIN - n})
        return out
    D = pd.DataFrame(data)
    res = []
    for f, label in (("starmine", "Fidelity analyst score (StarMine)"), ("sscore", "Fidelity social sentiment (S-score)")):
        for tgt, col in (("direction", "ret"), ("size", "size")):
            r, p, nn = spearman(D[f], D[col])
            res.append({"feature": f, "label": label, "target": tgt, "r": None if r != r else round(r, 3),
                        "p": None if p != p else round(p, 4), "n": nn})
    out.update({"status": "preliminary", "results": res})
    return out


# ----------------------------------------------------------------- fetching
def load_prices(symbols):
    import yfinance as yf
    out = {}
    for sym in symbols:
        try:
            df = yf.Ticker(sym).history(period="10y", auto_adjust=True)
            if df is not None and not df.empty:
                out[sym] = df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as exc:
            print(f"  price download failed for {sym}: {exc}", file=sys.stderr)
    return out


def load_wiki(article):
    if not article:
        return None
    start = (date.today() - timedelta(days=3650)).strftime("%Y%m%d")
    end = date.today().strftime("%Y%m%d")
    url = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/"
           f"{urllib.parse.quote(article, safe='')}/daily/{start}/{end}")
    req = urllib.request.Request(url, headers={"User-Agent": "covered-call-analyzer/1.0 (GitHub Actions)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            items = json.load(r)["items"]
        return pd.Series({pd.Timestamp(i["timestamp"][:8]): i["views"] for i in items}).sort_index()
    except Exception as exc:
        print(f"  Wikipedia views unavailable for {article}: {exc}", file=sys.stderr)
        return None


def load_earnings(ticker):
    import yfinance as yf
    try:
        ed = yf.Ticker(ticker).get_earnings_dates(limit=60)
        return sorted({pd.Timestamp(i).tz_localize(None).normalize() for i in ed.index})
    except Exception as exc:
        print(f"  earnings history unavailable for {ticker}: {exc}", file=sys.stderr)
        return []


def run_ticker(ticker, cfg, px_market):
    tcfg = {**cfg.get("_default", {}), **cfg.get(ticker, {})}
    peers = [p for p in tcfg.get("peers", []) if p != ticker]
    px = dict(px_market)
    px.update(load_prices([ticker] + [p for p in peers if p not in px]))
    if ticker not in px:
        raise RuntimeError("no price history")
    wiki = load_wiki(tcfg.get("wiki"))
    earn = load_earnings(ticker)
    return analyze(ticker, px, peers, wiki, earn, tcfg.get("wiki"))


def analyze(ticker, px, peers, wiki, earn, wiki_article):
    F = build_features(px, ticker, peers, wiki, earn)
    last = F.iloc[-1]
    current = {f: (None if pd.isna(last[f]) else float(last[f])) for f in FEATURES}
    current["gap_open"] = None  # unknown until the sell day opens
    current.update(F.attrs.get("wiki_now", {}))

    windows = []
    for win in WINDOWS:
        ev = make_events(F, win, earn)
        ev_clean, tests, pairs, base = study_window(ev, win, ticker)
        holding = [t for t in tests if t["holds"]] + [t for t in pairs if t["holds"]]
        sigma_now = float(last["hv20"]) * math.sqrt(win["sessions"] / 252)
        windows.append({
            "key": win["key"], "label": win["label"], "entry_dow": win["entry_dow"],
            "exit_offset": win["exit_offset"], "baseline": base,
            "tests": tests, "pairs": pairs,
            "holding": [t["feature"] + ":" + t["target"] + (":" + t["corner"] if "corner" in t else "") for t in holding],
            "suggestion": suggest(ev_clean, holding, current, sigma_now),
            "earnings_weeks_excluded": int(ev["earnings"].sum()) if len(ev) else 0,
        })

    return {
        "ticker": ticker,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "price": round(float(last["close"]), 2),
        "history_from": F.index[0].date().isoformat(),
        "signals_tested": sum(1 for f in FEATURES if F[f].notna().sum() > 100),
        "signals_missing": [LABELS[f] for f in FEATURES if F[f].notna().sum() <= 100 and f != "gap_open"],
        "wiki_article": wiki_article if wiki is not None else None,
        "peers": peers,
        "current": {k: (None if v is None else round(v, 5)) for k, v in current.items()},
        "windows": windows,
        "fidelity": fidelity_study(F, ticker),
        "method": {"train_frac": TRAIN_FRAC, "fdr_q": FDR_Q, "test_p": TEST_P,
                   "target_assign": TARGET_ASSIGN, "min_conditioned_sample": MIN_COND},
    }


def main():
    tickers = []
    for line in WATCHLIST.read_text().splitlines():
        s = line.split("#")[0].strip().upper()
        if s and s not in tickers:
            tickers.append(s)
    cfg = json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading market series...")
    px_market = load_prices(list(MARKET.values()))
    failed = 0
    for tk in tickers:
        print(f"Studying {tk}...")
        try:
            rep = run_ticker(tk, cfg, px_market)
            (OUT_DIR / f"{tk}.json").write_text(json.dumps(rep, indent=1, default=str))
            for w in rep["windows"]:
                print(f"  {w['label']}: {w['baseline']['n']} weeks, "
                      f"{len(w['holding'])} signals held up, strike +{w['suggestion']['move_pct']:.2f}%")
        except Exception as exc:
            failed += 1
            print(f"  FAILED: {exc}", file=sys.stderr)
    if failed == len(tickers):
        sys.exit(1)


if __name__ == "__main__":
    main()
