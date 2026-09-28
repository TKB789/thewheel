#!/usr/bin/env python3
"""
Greeks study: do option readings say more about assignment risk than the market's odds alone?

Uses the daily chain snapshots (scripts/chain_snapshot.py). Each saved contract is matched with
the closing price on its expiration day: a call "finished assigned" if that close was above the
strike, a put if it was below. Then, separately for calls and puts:

  1. Market odds vs what happened. Contracts are grouped by the market's chance of finishing in
     the money (from the option's own price), and each group's actual rate is shown. If the
     market says 20% and it happens 28% of the time, the market has been underpricing that risk.
  2. Readings. For each reading (implied vs recent volatility, skew, gamma, theta, strike distance
     measured with recent volatility, days to expiry, momentum...), contracts in the top third are
     compared with the bottom third on "actual minus market odds". A reading matters only if the
     gap is bigger than chance (resampling whole expirations, since contracts expiring together
     share one outcome), after controlling for testing many readings at once, and, once there's
     enough history, if it points the same way in the most recent 30% of days.
  3. Model check (with a year of data): does adding the readings to the market's odds predict
     assignment better on later days it wasn't fit on?

A first peek after 5 trading days (shown, nothing tested), an early look with tests after 60,
and the held-out checks after 250.
Output: data/greeks_study.json, read by index.html. Run from research.py after the close.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CHAINS_DIR = ROOT / "data" / "chains"
OUT = ROOT / "data" / "greeks_study.json"
NY = ZoneInfo("America/New_York")

PREVIEW_DAYS = 5          # a first peek after a week: shown, but nothing is tested yet
EARLY_DAYS = 60
FULL_DAYS = 250
RECENT_FRAC = 0.30
ODDS_RANGE = (0.03, 0.50)
BUCKETS = [0.03, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50]
BOOT = 400
FDR_Q = 0.10

READINGS = {
    "dist_hv": "Strike distance in recent-volatility moves",
    "iv_hv": "Implied vs recent volatility (IV ÷ HV)",
    "iv_rank": "Implied volatility rank (past year)",
    "skew_25d": "Put skew (25-delta put IV minus call IV)",
    "gamma_pct": "Gamma (delta change per 1% move)",
    "theta_share": "Theta as a share of the premium, per day",
    "vega_share": "Vega as a share of the premium",
    "hours_left": "Time to expiry",
    "iv": "Implied volatility of the contract",
    "rsi14": "RSI (14-day)",
    "pct_ma20": "Distance above 20-day average",
    "put_call_oi": "Put/call open interest",
    "open_interest": "Open interest at the strike",
}


def load_snapshots():
    files = sorted(CHAINS_DIR.glob("20??-??/*.csv"))
    if not files:
        return None, None
    parts = []
    for f in files:
        try:
            d = pd.read_csv(f)
            d["ticker"] = f.stem
            parts.append(d)
        except Exception as exc:
            print(f"  skipped {f}: {exc}", file=sys.stderr)
    rows = pd.concat(parts, ignore_index=True) if parts else None
    sums = []
    for f in sorted((CHAINS_DIR / "summary").glob("*.csv")):
        try:
            sums.append(pd.read_csv(f))
        except Exception:
            pass
    summary = pd.concat(sums, ignore_index=True) if sums else None
    return rows, summary


def add_outcomes(rows: pd.DataFrame, closes: dict, now_et: datetime):
    """closes: ticker -> daily close Series. Keeps contracts whose expiration has settled."""
    out = []
    today = now_et.date().isoformat()
    settled_today = now_et.time() >= time(16, 15)
    for tk, g in rows.groupby("ticker"):
        c = closes.get(tk)
        if c is None or c.empty:
            continue
        c = c.copy()
        c.index = pd.to_datetime(c.index).tz_localize(None).normalize()
        exp = pd.to_datetime(g["expiry"])
        close = exp.map(lambda d: c.get(d, np.nan))
        ok = close.notna() & ((g["expiry"] < today) | ((g["expiry"] == today) & settled_today))
        g = g[ok.values].copy()
        g["close_at_expiry"] = close[ok.values].values
        out.append(g)
    if not out:
        return None
    R = pd.concat(out, ignore_index=True)
    R["assigned"] = np.where(R["type"] == "C", R["close_at_expiry"] > R["strike"], R["close_at_expiry"] < R["strike"])
    return R


def add_readings(R: pd.DataFrame, summary: pd.DataFrame | None):
    if summary is not None and len(summary):
        s = summary.drop_duplicates(["ticker", "date"], keep="first").copy()
        s = s.sort_values(["ticker", "date"])
        # implied-volatility rank: where today's at-the-money IV sits in the ticker's past year (needs 120 days)
        def rank(x):
            v = x.values
            out = np.full(len(v), np.nan)
            for i in range(len(v)):
                past = v[max(0, i - 251):i + 1]
                past = past[~np.isnan(past)]
                if len(past) >= 120 and not np.isnan(v[i]):
                    out[i] = (past < v[i]).mean() * 100
            return pd.Series(out, index=x.index)
        s["iv_rank"] = s.groupby("ticker")["atm_iv"].transform(rank)
        R = R.merge(s[["ticker", "date", "spot", "hv20", "iv_hv", "iv_rank", "skew_25d", "rsi14", "pct_ma20",
                       "put_call_oi"]], on=["ticker", "date"], how="left")
    for col in ("spot", "hv20", "iv_hv", "iv_rank", "skew_25d", "rsi14", "pct_ma20", "put_call_oi"):
        if col not in R:
            R[col] = np.nan
    T = R["hours_left"] / (365 * 24)
    move = np.log(R["strike"] / R["spot"]).abs()
    R["dist_hv"] = move / (R["hv20"] * np.sqrt(T))                  # strike distance in recent-volatility moves
    R["gamma_pct"] = R["gamma"] * R["spot"] * 0.01                   # delta change for a 1% move
    R["theta_share"] = (R["theta"].abs() / R["mid"]).where(R["mid"] > 0)
    R["vega_share"] = (R["vega"] / R["mid"]).where(R["mid"] > 0)
    R["cluster"] = R["ticker"] + "|" + R["expiry"].astype(str)
    return R.replace([np.inf, -np.inf], np.nan)


def calibration(D: pd.DataFrame, rng):
    out = []
    for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
        m = (D["prob_itm"] >= lo) & (D["prob_itm"] < hi) if hi < BUCKETS[-1] else (D["prob_itm"] >= lo) & (D["prob_itm"] <= hi)
        g = D[m]
        if len(g) < 20:
            out.append({"lo": lo, "hi": hi, "n": int(len(g))})
            continue
        lo_ci, hi_ci = boot_ci(g, rng)
        out.append({"lo": lo, "hi": hi, "n": int(len(g)), "expirations": int(g["cluster"].nunique()),
                    "market": round(float(g["prob_itm"].mean()) * 100, 1),
                    "actual": round(float(g["assigned"].mean()) * 100, 1),
                    "ci": [round(lo_ci * 100, 1), round(hi_ci * 100, 1)]})
    return out


def boot_ci(g: pd.DataFrame, rng, n=BOOT):
    """90% range of the assignment rate, resampling whole expirations (contracts expiring
    together share one outcome, so they aren't independent)."""
    agg = g.groupby("cluster")["assigned"].agg(["sum", "count"]).values
    idx = rng.integers(0, len(agg), size=(n, len(agg)))
    rates = agg[idx, 0].sum(axis=1) / agg[idx, 1].sum(axis=1)
    return float(np.quantile(rates, 0.05)), float(np.quantile(rates, 0.95))


def bh(p, q=FDR_Q):
    p = np.asarray(p, float)
    order = np.argsort(p)
    ok = np.zeros(len(p), bool)
    k = 0
    for i, j in enumerate(order, 1):
        if p[j] <= q * i / len(p):
            k = i
    ok[order[:k]] = True
    return ok


def reading_tests(D: pd.DataFrame, full: bool, rng):
    """Top third vs bottom third of each reading, on 'actual minus market odds'."""
    D = D.copy()
    D["excess"] = D["assigned"].astype(float) - D["prob_itm"]
    days = np.sort(D["date"].unique())
    recent_from = days[int(len(days) * (1 - RECENT_FRAC))] if len(days) > 10 else days[-1]
    res = []
    for f, label in READINGS.items():
        x = D[f]
        ok = x.notna()
        if ok.sum() < 150 or x[ok].nunique() < 3:
            continue
        lo_cut, hi_cut = x[ok].quantile(1 / 3), x[ok].quantile(2 / 3)
        low, high = D[ok & (x <= lo_cut)], D[ok & (x >= hi_cut)]
        if len(low) < 50 or len(high) < 50 or low["cluster"].nunique() < 15 or high["cluster"].nunique() < 15:
            continue
        diff = float(high["excess"].mean() - low["excess"].mean())
        # resample expirations within each group; p = share of resamples on the other side of zero
        gh, gl = {k: v["excess"].values for k, v in high.groupby("cluster")}, {k: v["excess"].values for k, v in low.groupby("cluster")}
        kh, kl = list(gh), list(gl)
        boots = []
        for _ in range(BOOT):
            a = np.concatenate([gh[kh[i]] for i in rng.choice(len(kh), len(kh))])
            b = np.concatenate([gl[kl[i]] for i in rng.choice(len(kl), len(kl))])
            boots.append(a.mean() - b.mean())
        boots = np.array(boots)
        p = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()) + 1 / BOOT))
        rh, rl = high[high["date"] >= recent_from], low[low["date"] >= recent_from]
        recent = float(rh["excess"].mean() - rl["excess"].mean()) if len(rh) >= 30 and len(rl) >= 30 else None
        res.append({"key": f, "label": label, "low_cut": round(float(lo_cut), 4), "high_cut": round(float(hi_cut), 4),
                    "low": {"n": int(len(low)), "market": round(float(low["prob_itm"].mean()) * 100, 1),
                            "actual": round(float(low["assigned"].mean()) * 100, 1)},
                    "high": {"n": int(len(high)), "market": round(float(high["prob_itm"].mean()) * 100, 1),
                             "actual": round(float(high["assigned"].mean()) * 100, 1)},
                    "gap_pts": round(diff * 100, 1), "p": round(p, 4),
                    "recent_gap_pts": None if recent is None else round(recent * 100, 1),
                    "recent_same": None if (recent is None or not full) else bool(np.sign(recent) == np.sign(diff))})
    ok = bh([r["p"] for r in res]) if res else []
    for r, o in zip(res, ok):
        r["passed"] = bool(o)
        r["found"] = bool(o) and (r["recent_same"] is not False)
    res.sort(key=lambda r: (not r["found"], r["p"]))
    return res


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _fit_logistic(X, y, iters=50, l2=1.0):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = X @ w
        p = 1 / (1 + np.exp(-z))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + l2 * np.eye(X.shape[1])
        g = X.T @ (p - y) + l2 * w
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def _logloss(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def model_check(D: pd.DataFrame, found: list):
    """Held-out: market odds alone (recalibrated) vs market odds plus the readings that held up."""
    feats = [r["key"] for r in found]
    if not feats:
        return None
    days = np.sort(D["date"].unique())
    cut = days[int(len(days) * 0.7)]
    D = D.dropna(subset=feats).copy()
    tr, te = D[D["date"] < cut], D[D["date"] >= cut]
    if len(tr) < 500 or len(te) < 200:
        return None
    mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1)
    def X(d, extra):
        cols = [np.ones(len(d)), _logit(d["prob_itm"].values)]
        if extra:
            cols += [((d[f] - mu[f]) / sd[f]).values for f in feats]
        return np.column_stack(cols)
    y_tr, y_te = tr["assigned"].values.astype(float), te["assigned"].values.astype(float)
    w0, w1 = _fit_logistic(X(tr, False), y_tr), _fit_logistic(X(tr, True), y_tr)
    raw = _logloss(y_te, te["prob_itm"].values)
    base = _logloss(y_te, 1 / (1 + np.exp(-X(te, False) @ w0)))
    plus = _logloss(y_te, 1 / (1 + np.exp(-X(te, True) @ w1)))
    return {"readings": feats, "test_from": str(cut), "n_test": int(len(te)),
            "logloss_market": round(raw, 4), "logloss_market_recalibrated": round(base, 4),
            "logloss_with_readings": round(plus, 4),
            "improvement_pct": round((base - plus) / base * 100, 2),
            "helps": bool(plus < base * 0.99)}


def study(R: pd.DataFrame, rng):
    days = int(R["date"].nunique())
    status = ("collecting" if days < PREVIEW_DAYS else "preview" if days < EARLY_DAYS
              else "early" if days < FULL_DAYS else "full")
    out = {"days": days, "status": status, "need_preview": PREVIEW_DAYS, "need_early": EARLY_DAYS, "need_full": FULL_DAYS}
    D = R[(R["prob_itm"] >= ODDS_RANGE[0]) & (R["prob_itm"] <= ODDS_RANGE[1])]
    for kind, name in (("C", "calls"), ("P", "puts")):
        g = D[D["type"] == kind]
        side = {"contracts": int(len(g)), "expirations": int(g["cluster"].nunique()) if len(g) else 0,
                "market": round(float(g["prob_itm"].mean()) * 100, 1) if len(g) else None,
                "actual": round(float(g["assigned"].mean()) * 100, 1) if len(g) else None}
        if status != "collecting" and len(g) >= 100:
            side["calibration"] = calibration(g, rng)
            side["readings"] = reading_tests(g, status == "full", rng)
            if status == "preview":            # shown as a peek only: too few independent outcomes to test
                for r in side["readings"]:
                    r["found"] = r["passed"] = False
            if status == "full":
                side["model"] = model_check(g, [r for r in side["readings"] if r["found"]])
        out[name] = side
    return out


def run(closes: dict, now_et: datetime | None = None, seed=5):
    """closes: ticker -> daily close Series (research.py passes the prices it already downloaded)."""
    now_et = now_et or datetime.now(NY)
    rows, summary = load_snapshots()
    rep = {"updated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "labels": READINGS,
           "first_day": None, "snapshot_days": 0, "contracts_saved": 0}
    if rows is not None and len(rows):
        rep.update({"first_day": str(rows["date"].min()), "snapshot_days": int(rows["date"].nunique()),
                    "contracts_saved": int(len(rows)), "tickers": sorted(rows["ticker"].unique().tolist())})
        R = add_outcomes(rows, closes, now_et)
        if R is not None and len(R):
            R = add_readings(R, summary)
            rng = np.random.default_rng(seed)
            rep["pooled"] = study(R, rng)
            rep["by_ticker"] = {tk: study(g, rng) for tk, g in R.groupby("ticker") if g["date"].nunique() >= PREVIEW_DAYS}
    OUT.write_text(json.dumps(rep, separators=(",", ":"), default=str))
    return rep
