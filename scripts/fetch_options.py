#!/usr/bin/env python3
"""
Covered call data builder.

Reads tickers from watchlist.txt, pulls price history and option chains from
Yahoo Finance (via yfinance), and writes data/<TICKER>.json plus data/index.json
for index.html to read. It runs on a schedule in GitHub Actions because Yahoo
blocks requests made directly from a web page.

The rules follow common covered-call practice for someone who wants to keep
their shares and run the wheel if assigned:
  * sell calls around 0.15-0.30 delta (about a 15-30% chance of assignment)
  * don't hold a short call through an earnings report
  * prefer selling when implied volatility is rich versus realized volatility
  * if assigned, sell cash-secured puts around 0.25 delta to buy back in
"""
from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WATCHLIST = ROOT / "watchlist.txt"

RISK_FREE = 0.04            # annual risk-free rate used in the option math
MAX_DTE = 45                # look at expirations up to this many days out
MAX_EXPIRIES = 6
CALL_TARGET_DELTA = 0.20    # "keep my shares" target
CALL_TARGET_DELTA_HOT = 0.15  # tighter when the stock is running hot
CALL_DELTA_BAND = (0.10, 0.30)
PUT_TARGET_DELTA = 0.25     # standard for cash-secured puts you'd be glad to have assigned
PUT_TARGET_DELTA_FALLING = 0.15  # further out when the stock is sliding
PUT_DELTA_BAND = (0.15, 0.35)
SHOW_DELTA_RANGE = (0.03, 0.50)
# Similar past trades ("lookalikes", from the daily study): when they've predicted assignment
# better than chance for this ticker, the target delta is divided by how much likelier assignment
# looks now, and a big difference counts as a flag in the verdict.
LOOKALIKE_WINDOW = {0: "mon_wed", 3: "thu_fri"}    # your schedule: Monday -> Wednesday, Thursday -> Friday
WINDOW_NAMES = {"mon_wed": "Monday → Wednesday", "thu_fri": "Thursday → Friday", "mon_fri": "Monday → Friday"}
LK_CALL_DELTA_LIMITS = (0.10, 0.25)
LK_PUT_DELTA_LIMITS = (0.15, 0.30)
LK_FLAG_RATIO = 1.25      # at least this much likelier than usual -> counts as a flag
LK_GOOD_RATIO = 0.80      # this much less likely -> a point in favor


# ---------------------------------------------------------------- option math
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1_d2(S, K, T, vol, r):
    d1 = (math.log(S / K) + (r + 0.5 * vol * vol) * T) / (vol * math.sqrt(T))
    return d1, d1 - vol * math.sqrt(T)


def bs_price(S, K, T, vol, r, kind):
    if T <= 0 or vol <= 0:
        return max(0.0, S - K) if kind == "call" else max(0.0, K - S)
    d1, d2 = _d1_d2(S, K, T, vol, r)
    if kind == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def implied_vol(price, S, K, T, r, kind):
    """Solve for volatility by bisection. Returns None if the price is unusable."""
    intrinsic = max(0.0, S - K) if kind == "call" else max(0.0, K - S)
    if T <= 0 or price <= intrinsic + 1e-4:
        return None
    lo, hi = 0.01, 5.0
    if bs_price(S, K, T, hi, r, kind) < price:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if bs_price(S, K, T, mid, r, kind) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def greeks(S, K, T, vol, r, kind):
    """Delta and the risk-neutral chance of finishing in the money (assignment)."""
    d1, d2 = _d1_d2(S, K, T, vol, r)
    if kind == "call":
        return norm_cdf(d1), norm_cdf(d2)
    return norm_cdf(d1) - 1.0, norm_cdf(-d2)


# ----------------------------------------------------------------- indicators
def realized_vol(closes, n=20):
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(len(closes) - n, len(closes))]
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252)


def rsi(closes, n=14):
    gains, losses = [], []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    avg_g = sum(gains[:n]) / n
    avg_l = sum(losses[:n]) / n
    for g, l in zip(gains[n:], losses[n:]):
        avg_g = (avg_g * (n - 1) + g) / n
        avg_l = (avg_l * (n - 1) + l) / n
    if avg_l == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_g / avg_l)


# ------------------------------------------------------------------- building
def num(x, default=0.0):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) else f


def build_side(raw_rows, S, dte, kind, fallback_vol):
    """Turn raw chain rows into out-of-the-money rows with delta and yield."""
    T = max(dte, 0.5) / 365.0
    rows = []
    for rw in raw_rows:
        K = num(rw.get("strike"))
        if K <= 0 or (kind == "call" and K <= S) or (kind == "put" and K >= S):
            continue
        bid, ask, last = num(rw.get("bid")), num(rw.get("ask")), num(rw.get("lastPrice"))
        live_quote = bid > 0 and ask > 0
        mid = (bid + ask) / 2 if live_quote else last
        if mid <= 0.01:
            continue
        vol = implied_vol(mid, S, K, T, RISK_FREE, kind)
        if vol is None:
            yv = num(rw.get("impliedVolatility"))
            vol = yv if 0.03 < yv < 3 else fallback_vol
        delta, p_itm = greeks(S, K, T, vol, RISK_FREE, kind)
        if not (SHOW_DELTA_RANGE[0] <= abs(delta) <= SHOW_DELTA_RANGE[1]):
            continue
        base = S if kind == "call" else K   # calls: yield on shares; puts: yield on cash set aside
        rows.append({
            "strike": round(K, 2),
            "bid": round(bid, 2),
            "ask": round(ask, 2),
            "mid": round(mid, 2),
            "live_quote": live_quote,
            "iv": round(vol, 4),
            "delta": round(delta, 3),
            "prob_assign": round(p_itm, 3),
            "premium": round(mid * 100, 2),
            "yield_pct": round(mid / base * 100, 3),
            "ann_yield_pct": round(mid / base * 365 / max(dte, 1) * 100, 1),
            "cash_needed": round(K * 100, 2) if kind == "put" else None,
            "open_interest": int(num(rw.get("openInterest"))),
            "volume": int(num(rw.get("volume"))),
        })
    rows.sort(key=lambda r: r["strike"])
    return rows


def pick(expiries, side, target, band, before=None):
    """Closest-to-target delta per allowed expiry, then best annualized yield."""
    best = None
    for e in expiries:
        if before and e["date"] >= before.isoformat():
            continue
        cands = [r for r in e[side] if band[0] <= abs(r["delta"]) <= band[1]]
        if not cands:
            continue
        r = min(cands, key=lambda c: abs(abs(c["delta"]) - target))
        choice = dict(r, expiry=e["date"], dte=e["dte"])
        if best is None or choice["ann_yield_pct"] > best["ann_yield_pct"]:
            best = choice
    return best


def lookalike_factor(adj, window, kind):
    """A verdict factor from the lookalike adjustment, or None when it doesn't apply."""
    if not adj or not adj.get("beats_chance") or adj.get("rate_now") is None:
        return None
    what = "call" if kind == "call" else "put"
    r, base, ratio = adj["rate_now"], adj["base_rate"], adj["ratio"]
    group = (f" Past trades in the same spot were actually assigned {adj['group_rate']:.0f}% of the time"
             f" ({adj['group_trades']} trades).") if adj.get("group_rate") is not None else ""
    head = (f"{adj['lookalikes_assigned']} of the 10 past {WINDOW_NAMES.get(window, window)} trades most like today ended with the {what} assigned "
            f"({r:.0f}% of the closest 50, vs {base:.0f}% of all trades).{group}")
    if not adj.get("applied"):
        return {"name": "Similar past weeks", "status": "neutral", "value": "About usual",
                "note": head + " That's close to normal, so the strike isn't changed."}
    if ratio >= LK_FLAG_RATIO:
        return {"name": "Similar past weeks", "status": "bad", "value": f"{ratio:.1f}× likelier",
                "note": head + f" Assignment looks {ratio:.1f}× as likely as usual, so the recommended {what} is moved further out."}
    if ratio <= LK_GOOD_RATIO:
        return {"name": "Similar past weeks", "status": "good", "value": f"{ratio:.1f}× as likely",
                "note": head + f" Assignment looks less likely than usual, so the recommended {what} sits a little closer for more premium."}
    return {"name": "Similar past weeks", "status": "neutral", "value": f"{ratio:.1f}× as likely",
            "note": head + " The recommended strike is adjusted slightly."}


def load_lookalike(ticker, today):
    """Today's lookalike adjustment for your schedule window (Mondays and Thursdays only)."""
    window = LOOKALIKE_WINDOW.get(today.weekday())
    if not window:
        return None, None
    try:
        R = json.loads((DATA_DIR / "research" / f"{ticker}.json").read_text())
        return window, (R.get("lookalike_now") or {}).get(window)
    except Exception:
        return window, None


def adjusted_target(base, adj, limits):
    if not adj or not adj.get("applied"):
        return base
    return round(min(max(base / adj["ratio"], limits[0]), limits[1]), 3)


def assess(ctx):
    """Score the setup for someone who wants to keep their shares."""
    f = []
    today, near = ctx["today"], ctx["nearest_expiry"]
    earn = ctx["earnings_date"]

    earnings_blocks = False
    if earn and near and today <= earn <= near:
        earnings_blocks = True
        f.append({"name": "Earnings", "status": "bad", "value": earn.isoformat(),
                  "note": f"Earnings on {earn:%b %-d} land before the {near:%b %-d} expiry. "
                          "A big earnings move can jump past any strike."})
    elif earn and (earn - today).days <= MAX_DTE:
        f.append({"name": "Earnings", "status": "neutral", "value": earn.isoformat(),
                  "note": f"Earnings on {earn:%b %-d}. Only expirations before that date are recommended."})
    elif earn:
        f.append({"name": "Earnings", "status": "good", "value": earn.isoformat(),
                  "note": f"Next earnings {earn:%b %-d}, outside the next {MAX_DTE} days."})
    else:
        f.append({"name": "Earnings", "status": "neutral", "value": "Unknown",
                  "note": "No earnings date found. Check your broker before selling."})

    ratio = ctx["atm_iv"] / ctx["hv20"] if ctx["hv20"] else None
    if ratio is None:
        f.append({"name": "Premium level", "status": "neutral", "value": "n/a", "note": "Not enough data."})
    elif ratio >= 1.15:
        f.append({"name": "Premium level", "status": "good", "value": f"IV/HV {ratio:.2f}",
                  "note": "Premium is rich. Options are priced for more movement than the stock has shown lately."})
    elif ratio >= 0.9:
        f.append({"name": "Premium level", "status": "neutral", "value": f"IV/HV {ratio:.2f}",
                  "note": "Premium is fairly priced against recent movement."})
    else:
        f.append({"name": "Premium level", "status": "bad", "value": f"IV/HV {ratio:.2f}",
                  "note": "Premium is thin. The stock has been moving more than options are paying for."})

    r, gap = ctx["rsi14"], ctx["pct_vs_ma20"]
    hot = r > 70 or gap > 6
    if hot:
        f.append({"name": "Momentum", "status": "bad", "value": f"RSI {r:.0f}",
                  "note": f"Stock is running hot ({gap:+.1f}% vs its 20-day average). "
                          "Higher chance it climbs through your strike."})
    elif r < 30:
        f.append({"name": "Momentum", "status": "neutral", "value": f"RSI {r:.0f}",
                  "note": "Stock is oversold. Rebounds can be sharp, and call premium is usually lower here."})
    else:
        f.append({"name": "Momentum", "status": "good", "value": f"RSI {r:.0f}",
                  "note": f"Momentum is normal ({gap:+.1f}% vs its 20-day average)."})

    exd = ctx["ex_div_date"]
    if exd and near and today <= exd <= near:
        f.append({"name": "Ex-dividend", "status": "neutral", "value": exd.isoformat(),
                  "note": f"Ex-dividend {exd:%b %-d}. If your call goes in the money, it can be assigned "
                          "early the day before so the buyer collects the dividend."})

    pc = ctx["put_call_oi"]
    if pc is not None:
        desc = ("Heavy put positioning: traders are hedging against a drop." if pc > 1.2 else
                "Call-heavy positioning: traders are leaning bullish." if pc < 0.7 else
                "Balanced positioning between puts and calls.")
        f.append({"name": "Put/call (open interest)", "status": "info", "value": f"{pc:.2f}", "note": desc})
    lk = lookalike_factor(ctx.get("lookalike_calls"), ctx.get("lookalike_window"), "call")
    if lk:
        f.append(lk)

    bads = sum(1 for x in f if x["status"] == "bad" and x["name"] != "Earnings")
    if earnings_blocks:
        verdict = {"level": "wait", "reason": "earnings", "label": "Hold off until after earnings",
                   "summary": "Every near expiration spans the earnings report. Sell covered calls once earnings are out."}
    elif bads == 0:
        verdict = {"level": "good", "label": "Favorable to sell covered calls",
                   "summary": "No red flags. The recommended call below balances premium against keeping your shares."}
    elif bads == 1:
        verdict = {"level": "caution", "label": "Sell covered calls further out of the money",
                   "summary": "One flag is up. The recommended call already uses a lower delta; consider selling fewer contracts."}
    else:
        verdict = {"level": "wait", "label": "Hold off on selling calls",
                   "summary": "Several flags are up. The premium isn't worth the risk to your shares right now."}
    return f, verdict, hot


def assess_puts(ctx):
    """Score the setup for selling cash-secured puts. The risk is the reverse of a covered call:
    a stock in a steep slide can blow through the strike, so you buy shares that keep falling."""
    f = []
    today, near, earn = ctx["today"], ctx["nearest_expiry"], ctx["earnings_date"]
    earnings_blocks = bool(earn and near and today <= earn <= near)
    if earnings_blocks:
        f.append({"name": "Earnings", "status": "bad", "value": earn.isoformat(),
                  "note": f"Earnings on {earn:%b %-d} land before the {near:%b %-d} expiry. "
                          "A bad report can gap the stock far below any strike."})
    elif earn and (earn - today).days <= MAX_DTE:
        f.append({"name": "Earnings", "status": "neutral", "value": earn.isoformat(),
                  "note": f"Earnings on {earn:%b %-d}. Only expirations before that date are recommended."})
    elif earn:
        f.append({"name": "Earnings", "status": "good", "value": earn.isoformat(),
                  "note": f"Next earnings {earn:%b %-d}, outside the next {MAX_DTE} days."})
    else:
        f.append({"name": "Earnings", "status": "neutral", "value": "Unknown",
                  "note": "No earnings date found. Check your broker before selling."})

    ratio = ctx["atm_iv"] / ctx["hv20"] if ctx["hv20"] else None
    if ratio is None:
        f.append({"name": "Premium level", "status": "neutral", "value": "n/a", "note": "Not enough data."})
    elif ratio >= 1.15:
        f.append({"name": "Premium level", "status": "good", "value": f"IV/HV {ratio:.2f}",
                  "note": "Premium is rich. Options are priced for more movement than the stock has shown lately."})
    elif ratio >= 0.9:
        f.append({"name": "Premium level", "status": "neutral", "value": f"IV/HV {ratio:.2f}",
                  "note": "Premium is fairly priced against recent movement."})
    else:
        f.append({"name": "Premium level", "status": "bad", "value": f"IV/HV {ratio:.2f}",
                  "note": "Premium is thin. The stock has been moving more than options are paying for."})

    r, gap = ctx["rsi14"], ctx["pct_vs_ma20"]
    falling = r < 30 or gap < -6
    if falling:
        f.append({"name": "Trend", "status": "bad", "value": f"RSI {r:.0f}",
                  "note": f"Stock is sliding ({gap:+.1f}% vs its 20-day average). Puts sold into a slide "
                          "often get assigned, and the shares can keep falling after."})
    elif r > 70:
        f.append({"name": "Trend", "status": "neutral", "value": f"RSI {r:.0f}",
                  "note": "Stock is running hot. Put premium is thinner here, and pullbacks can come fast."})
    else:
        f.append({"name": "Trend", "status": "good", "value": f"RSI {r:.0f}",
                  "note": f"Trend is steady ({gap:+.1f}% vs its 20-day average)."})

    pc = ctx["put_call_oi"]
    if pc is not None:
        desc = ("Heavy put positioning: traders are hedging against a drop." if pc > 1.2 else
                "Call-heavy positioning: traders are leaning bullish." if pc < 0.7 else
                "Balanced positioning between puts and calls.")
        f.append({"name": "Put/call (open interest)", "status": "info", "value": f"{pc:.2f}", "note": desc})
    lk = lookalike_factor(ctx.get("lookalike_puts"), ctx.get("lookalike_window"), "put")
    if lk:
        f.append(lk)

    bads = sum(1 for x in f if x["status"] == "bad" and x["name"] != "Earnings")
    if earnings_blocks:
        v = {"level": "wait", "reason": "earnings", "label": "Hold off until after earnings",
             "summary": "Every near expiration spans the earnings report. Sell puts once earnings are out."}
    elif bads == 0:
        v = {"level": "good", "label": "Favorable to sell cash-secured puts",
             "summary": "No red flags. The recommended put below pays you to wait for a lower buy price."}
    elif bads == 1:
        v = {"level": "caution", "label": "Sell puts further out of the money",
             "summary": "One flag is up. The recommended put already uses a lower delta; consider selling fewer contracts."}
    else:
        v = {"level": "wait", "label": "Hold off on selling puts",
             "summary": "Several flags are up. The premium isn't worth the risk of buying into weakness right now."}
    return f, v, falling


def build_report(ticker, closes, price, prev_close, raw_expiries, earnings_date,
                 ex_div_date, dividend_rate, today, updated, lookalike=None, lookalike_window=None):
    hv20 = realized_vol(closes, 20)
    rsi14 = rsi(closes[-120:], 14)
    ma20 = sum(closes[-20:]) / 20
    pct_vs_ma20 = (price / ma20 - 1) * 100

    expiries, call_oi, put_oi, call_vol, put_vol = [], 0.0, 0.0, 0.0, 0.0
    atm_iv = None
    for e in raw_expiries:
        dte = e["dte"]
        call_oi += sum(num(r.get("openInterest")) for r in e["calls_raw"])
        put_oi += sum(num(r.get("openInterest")) for r in e["puts_raw"])
        call_vol += sum(num(r.get("volume")) for r in e["calls_raw"])
        put_vol += sum(num(r.get("volume")) for r in e["puts_raw"])
        if atm_iv is None and e["calls_raw"]:
            atm = min(e["calls_raw"], key=lambda r: abs(num(r.get("strike")) - price))
            b, a, l = num(atm.get("bid")), num(atm.get("ask")), num(atm.get("lastPrice"))
            m = (b + a) / 2 if b > 0 and a > 0 else l
            atm_iv = implied_vol(m, price, num(atm.get("strike")), max(dte, 0.5) / 365, RISK_FREE, "call")
        expiries.append({
            "date": e["date"], "dte": dte,
            "calls": build_side(e["calls_raw"], price, dte, "call", hv20),
            "puts": build_side(e["puts_raw"], price, dte, "put", hv20),
        })
    atm_iv = atm_iv or hv20
    nearest = date.fromisoformat(expiries[0]["date"]) if expiries else None

    ctx = {
        "today": today, "nearest_expiry": nearest, "earnings_date": earnings_date,
        "atm_iv": atm_iv, "hv20": hv20, "rsi14": rsi14, "pct_vs_ma20": pct_vs_ma20,
        "ex_div_date": ex_div_date,
        "put_call_oi": (put_oi / call_oi) if call_oi else None,
        "lookalike_window": lookalike_window,
        "lookalike_calls": (lookalike or {}).get("calls"),
        "lookalike_puts": (lookalike or {}).get("puts"),
    }
    factors, verdict, hot = assess(ctx)
    put_factors, put_verdict, falling = assess_puts(ctx)

    before = earnings_date if earnings_date and earnings_date >= today else None
    call_target = adjusted_target(CALL_TARGET_DELTA_HOT if hot else CALL_TARGET_DELTA,
                                  ctx["lookalike_calls"], LK_CALL_DELTA_LIMITS)
    rec_call = None if verdict.get("reason") == "earnings" else \
        pick(expiries, "calls", call_target, CALL_DELTA_BAND, before)
    put_target = adjusted_target(PUT_TARGET_DELTA_FALLING if falling else PUT_TARGET_DELTA,
                                 ctx["lookalike_puts"], LK_PUT_DELTA_LIMITS)
    rec_put = None if put_verdict.get("reason") == "earnings" else \
        pick(expiries, "puts", put_target, PUT_DELTA_BAND, before)

    return {
        "ticker": ticker,
        "updated": updated,
        "price": round(price, 2),
        "prev_close": round(prev_close, 2),
        "change_pct": round((price / prev_close - 1) * 100, 2),
        "hv20": round(hv20, 4),
        "atm_iv": round(atm_iv, 4),
        "rsi14": round(rsi14, 1),
        "ma20": round(ma20, 2),
        "pct_vs_ma20": round(pct_vs_ma20, 2),
        "earnings_date": earnings_date.isoformat() if earnings_date else None,
        "ex_div_date": ex_div_date.isoformat() if ex_div_date else None,
        "dividend_rate": dividend_rate,
        "put_call_oi": round(put_oi / call_oi, 2) if call_oi else None,
        "put_call_volume": round(put_vol / call_vol, 2) if call_vol else None,
        "call_target_delta": call_target,
        "put_target_delta": put_target,
        "call_base_delta": CALL_TARGET_DELTA_HOT if hot else CALL_TARGET_DELTA,
        "put_base_delta": PUT_TARGET_DELTA_FALLING if falling else PUT_TARGET_DELTA,
        "lookalike": {"window": lookalike_window, **(lookalike or {})} if lookalike else None,
        "verdict": verdict,
        "factors": factors,
        "put_verdict": put_verdict,
        "put_factors": put_factors,
        "recommended_call": rec_call,
        "recommended_put": rec_put,
        "expiries": expiries,
    }


# ------------------------------------------------------------------ fetching
def _to_date(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
        if v is None:
            return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, (int, float)):
        return datetime.fromtimestamp(v, tz=timezone.utc).date()
    try:
        return date.fromisoformat(str(v)[:10])
    except ValueError:
        return None


def fetch(ticker):
    import yfinance as yf

    t = yf.Ticker(ticker)
    hist = t.history(period="6mo", auto_adjust=False)
    closes = [float(x) for x in hist["Close"].dropna().tolist()]
    if len(closes) < 40:
        raise RuntimeError("not enough price history")
    price, prev_close = closes[-1], closes[-2]
    try:
        live = float(t.fast_info["last_price"])
        if live > 0:
            price = live
    except Exception:
        pass

    today = date.today()
    raw = []
    for exp in t.options:
        d = date.fromisoformat(exp)
        dte = (d - today).days
        if dte < 1:
            continue
        if dte > MAX_DTE or len(raw) >= MAX_EXPIRIES:
            break
        ch = t.option_chain(exp)
        raw.append({"date": exp, "dte": dte,
                    "calls_raw": ch.calls.to_dict("records"),
                    "puts_raw": ch.puts.to_dict("records")})

    earnings = ex_div = None
    try:
        cal = t.calendar
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date") or []
            future = [d for d in (_to_date(x) for x in dates) if d and d >= today]
            earnings = min(future) if future else None
            ex_div = _to_date(cal.get("Ex-Dividend Date"))
    except Exception:
        pass
    if earnings is None:
        try:
            ed = t.get_earnings_dates(limit=8)
            future = [i.date() for i in ed.index if i.date() >= today]
            earnings = min(future) if future else None
        except Exception:
            pass

    div_rate = None
    try:
        info = t.info
        div_rate = info.get("dividendRate")
        if ex_div is None:
            ex_div = _to_date(info.get("exDividendDate"))
    except Exception:
        pass
    if ex_div and ex_div < today:
        ex_div = None

    lk_window, lk = load_lookalike(ticker, today)
    report = build_report(ticker, closes, price, prev_close, raw, earnings, ex_div, div_rate,
                          today, datetime.now(timezone.utc).isoformat(timespec="seconds"), lk, lk_window)
    # the trading day these prices come from (differs from today on weekends and market holidays)
    report["session_date"] = hist.index[-1].date().isoformat()
    # recent daily closes, so the page can score paper trades saved in your browser
    report["closes"] = [[d.date().isoformat(), round(float(c), 2)] for d, c in hist["Close"].dropna().tail(90).items()]
    try:                        # daily chain snapshot for the Greeks study (see chain_snapshot.py)
        import chain_snapshot
        n = chain_snapshot.maybe_snapshot(ticker, price, prev_close, raw, report["session_date"],
                                          hv20=report["hv20"], rsi14=report["rsi14"],
                                          pct_ma20=report["pct_vs_ma20"], put_call_oi=report["put_call_oi"])
        if n:
            print(f"  saved {n} contracts to the chain snapshot")
    except Exception as exc:
        print(f"  chain snapshot skipped: {exc}", file=sys.stderr)
    return report


def tickers_to_run(watchlist_path):
    """One ticker from the command line (on-demand lookup), otherwise the whole watchlist."""
    import re as _re
    if len(sys.argv) > 1 and sys.argv[1].strip():
        tk = sys.argv[1].strip().upper()
        if not _re.fullmatch(r"[A-Z0-9.^=-]{1,12}", tk):
            sys.exit(f"'{tk}' doesn't look like a ticker symbol")
        return [tk], True
    out = []
    for line in watchlist_path.read_text().splitlines():
        s = line.split("#")[0].strip().upper()
        if s and s not in out:
            out.append(s)
    return out, False


def main():
    tickers, single = tickers_to_run(WATCHLIST)
    DATA_DIR.mkdir(exist_ok=True)
    ok, failed = [], []
    for tk in tickers:
        try:
            report = fetch(tk)
            (DATA_DIR / f"{tk}.json").write_text(json.dumps(report, indent=1))
            ok.append(tk)
            print(f"{tk}: {report['verdict']['label']}")
        except Exception as exc:  # keep going; one bad ticker shouldn't stop the rest
            failed.append(tk)
            print(f"{tk}: FAILED ({exc})", file=sys.stderr)
    if single:  # on-demand lookup: keep every ticker already listed
        try:
            prev = json.loads((DATA_DIR / "index.json").read_text()).get("tickers", [])
        except (OSError, ValueError):
            prev = []
        ok = sorted(set(prev) | set(ok))
    (DATA_DIR / "index.json").write_text(json.dumps({
        "tickers": ok, "failed": failed,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=1))
    if not ok or (single and failed):   # a failed lookup must show as failed so the page can say so
        sys.exit(1)


if __name__ == "__main__":
    main()
