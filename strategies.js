/* Other strategies: a guide tab plus one tab per strategy (iron condor, short strangle,
   bull put spread, long straddle, collar). Each tab has:
     - Build it today: the position from the latest option chain, its payoff, and a table of
       when you'd make or lose money, with how often this stock has moved that much before.
     - Backtest: every past trade period, priced in the browser (so changing strikes or the
       pricing assumption re-runs it), with when it worked and when it didn't.
     - Your paper trades: saved in this browser, with export / import.
   Loaded after the main script and uses its helpers ($, esc, money, signed, fmtDate, getJSON,
   store, recall, state, T for glossary terms). Data: data/<T>.json (chain, closes) and
   data/backtest/<T>_strategies.json (written by scripts/strategies.py). */
(function () {
"use strict";
const term = (k, text) => T(k, text || k);
const RATE = 0.04, FEE = 0.65;

// ------------------------------------------------------------------ math
function ncdf(x) {
  const t = 1 / (1 + 0.2316419 * Math.abs(x)), d = 0.3989422804014327 * Math.exp(-x * x / 2);
  const p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return x > 0 ? 1 - p : p;
}
function ninv(p) {                          // Acklam's inverse normal
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425;
  if (p < lo) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  if (p > 1 - lo) { const q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}
function bs(kind, S, K, t, v) {
  if (t <= 0 || v <= 0) return Math.max(0, kind === "C" ? S - K : K - S);
  const sq = v * Math.sqrt(t), d1 = (Math.log(S / K) + (RATE + v * v / 2) * t) / sq, d2 = d1 - sq;
  return kind === "C" ? S * ncdf(d1) - K * Math.exp(-RATE * t) * ncdf(d2) : K * Math.exp(-RATE * t) * ncdf(-d2) - S * ncdf(-d1);
}
function deltaStrike(kind, S, t, v, d) {    // the strike whose delta is d (calls) or -d (puts)
  const d1 = ninv(kind === "C" ? d : 1 - d);
  return S * Math.exp(-d1 * v * Math.sqrt(t) + (RATE + v * v / 2) * t);
}
const stepOf = (S) => S < 25 ? 0.5 : S < 200 ? 1 : S < 500 ? 2.5 : 5;
const rUp = (k, s) => Math.ceil(k / s - 1e-9) * s, rDn = (k, s) => Math.floor(k / s + 1e-9) * s, rNear = (k, s) => Math.round(k / s) * s;
const optLegs = (legs) => legs.filter((l) => l.t !== "S");
function payoff(l, X) { return l.t === "C" ? Math.max(X - l.K, 0) : l.t === "P" ? Math.max(l.K - X, 0) : X; }
function plAt(legs, X, fee = FEE) {          // dollars for one position (100 shares per contract), fees at the open
  let v = 0;
  for (const l of legs) v += l.side * (payoff(l, X) - l.px) * 100;
  return v - fee * optLegs(legs).length;
}
const netOf = (legs) => -optLegs(legs).reduce((a, l) => a + l.side * l.px * 100, 0);   // > 0: credit

// ------------------------------------------------------------------ strategies
// Leg specs: t = C / P / S (shares), side +1 buy / -1 sell, d = target delta, rnd = how to round the
// model strike, beyond = index of the leg this one must be further out than (the wings).
const STRATS = {
  condor: {
    name: "Iron condor", tab: "Iron condor", icon: "▭",
    tagline: "Paid up front to bet the stock stays inside a range.",
    view: "Neutral", pay: "Collect a credit", risk: "Defined",
    params: [{ key: "short", label: "Short strikes", opts: [0.10, 0.16, 0.20, 0.25, 0.30], def: 0.16 },
             { key: "wing", label: "Wings", opts: [0.03, 0.05, 0.08, 0.10], def: 0.05 }],
    specs: (p) => [{ t: "P", side: 1, d: Math.min(p.wing, p.short - 0.01), rnd: "dn", beyond: 1, role: "wing" },
                   { t: "P", side: -1, d: p.short, rnd: "dn", role: "short" },
                   { t: "C", side: -1, d: p.short, rnd: "up", role: "short" },
                   { t: "C", side: 1, d: Math.min(p.wing, p.short - 0.01), rnd: "up", beyond: 2, role: "wing" }],
    capital(legs, net) { const w = Math.max(legs[3].K - legs[2].K, legs[1].K - legs[0].K); return Math.max(w * 100 - net, 1); },
    capitalNote: "Margin for an iron condor is the wider wing's width × 100, minus the credit: the most you can lose.",
    seller: true, window: "mon_fri",
    how: `Sell a put below today's price and a call above it, then buy a cheaper put and call further out as insurance (the ${"wings"}). You collect a credit up front. If the stock finishes between the two sold strikes, all four options expire worthless and the credit is yours. If it breaks out either way, the loss grows until it reaches the wing, where the bought option stops it.`,
    when: "Used when you expect the stock to stay put and options look expensive: after a big move has calmed down, in quiet markets, or when the market is pricing in more movement than the stock has been making. Traders usually avoid holding one through earnings or major news, and often close it early (at half the credit, say) rather than wait for expiration.",
    feel: "Many small wins and an occasional loss several times larger. Whether it pays over time depends on whether the wins outnumber the losses by enough, which is what the backtest below checks.",
    greeks: [["Time passing", "helps", "good"], ["Implied volatility rising", "hurts", "bad"], ["Stock moving either way", "hurts once it nears a sold strike", "bad"]],
  },
  strangle: {
    name: "Short strangle", tab: "Short strangle", icon: "∩",
    tagline: "The iron condor without the insurance: more credit, no limit on the loss.",
    view: "Neutral", pay: "Collect a credit", risk: "Undefined",
    params: [{ key: "short", label: "Sold strikes", opts: [0.10, 0.16, 0.20, 0.25, 0.30], def: 0.16 }],
    specs: (p) => [{ t: "P", side: -1, d: p.short, rnd: "dn", role: "short" }, { t: "C", side: -1, d: p.short, rnd: "up", role: "short" }],
    capital(legs, net, S) {                // Reg T margin for a short strangle: the larger side's requirement plus the other side's premium
      const [p, c] = legs;
      const callReq = c.px * 100 + Math.max(0.2 * S - Math.max(c.K - S, 0), 0.1 * S) * 100;
      const putReq = p.px * 100 + Math.max(0.2 * S - Math.max(S - p.K, 0), 0.1 * p.K) * 100;
      return Math.max(callReq + p.px * 100, putReq + c.px * 100);
    },
    capitalNote: "Brokers' standard margin for a short strangle is roughly 20% of the stock's value (less how far out of the money the option is) on the riskier side, plus the other side's premium. It can rise quickly if the stock moves against you.",
    seller: true, window: "mon_fri",
    how: "Sell a put below the price and a call above it, with nothing bought as protection. The credit is bigger than an iron condor's because you keep the whole premium. Between the two strikes at expiration, you keep all of it. Past either strike, you lose a dollar for every dollar the stock moves, with no floor on the call side.",
    when: "Used by experienced traders with margin accounts, on liquid stocks and ETFs, when options are expensive. It's the wheel's two legs without the stock or the cash: the put side is your cash-secured put, the call side is a covered call with no shares behind it. A gap through a strike (earnings, news overnight) can cost many times the credit, which is why brokers require their highest options approval level for it.",
    feel: "Wins even more often than an iron condor (the strikes are the same, and there's no wing cost), but the rare loss has no limit.",
    greeks: [["Time passing", "helps", "good"], ["Implied volatility rising", "hurts", "bad"], ["A big move either way", "hurts, without limit", "bad"]],
  },
  bullput: {
    name: "Bull put spread", tab: "Bull put spread", icon: "⌟",
    tagline: "A cash-secured put with a floor under the loss, using far less cash.",
    view: "Neutral to bullish", pay: "Collect a credit", risk: "Defined",
    params: [{ key: "short", label: "Sold put", opts: [0.15, 0.20, 0.25, 0.30, 0.35], def: 0.25 },
             { key: "long", label: "Bought put", opts: [0.05, 0.10, 0.15], def: 0.10 }],
    specs: (p) => [{ t: "P", side: 1, d: Math.min(p.long, p.short - 0.02), rnd: "dn", beyond: 1, role: "wing" }, { t: "P", side: -1, d: p.short, rnd: "dn", role: "short" }],
    capital(legs, net) { return Math.max((legs[1].K - legs[0].K) * 100 - net, 1); },
    capitalNote: "Margin is the width between the strikes × 100, minus the credit. A cash-secured put at the same strike would tie up the full strike × 100.",
    seller: true, window: "mon_fri",
    how: "Sell a put, just as you do for the wheel, and buy a put at a lower strike. You collect the difference as a credit. If the stock stays above the sold strike, both expire and you keep it. Below the bought strike the loss stops growing: it can't exceed the width between the strikes, minus the credit.",
    when: "Used for a neutral-to-bullish view when you don't want the shares or can't set aside the cash: a $300 stock's cash-secured put ties up $30,000, while a $5-wide spread ties up under $500. The trade-off: a cash-secured put that goes wrong leaves you owning a stock you like, which you can wheel out of; a spread that goes wrong is just a loss.",
    feel: "Like your puts, it wins most weeks. The losing weeks are capped, but on a small amount of capital each one wipes out many weeks of credits.",
    greeks: [["Time passing", "helps", "good"], ["Stock rising", "helps", "good"], ["Stock falling", "hurts, down to the bought put", "bad"]],
  },
  straddle: {
    name: "Long straddle", tab: "Long straddle", icon: "V",
    tagline: "Paying for a big move in either direction.",
    view: "Big move, direction unknown", pay: "Pay a debit", risk: "Defined (the debit)",
    params: [{ key: "legs", label: "Strikes", opts: [0.5, 0.3, 0.2], def: 0.5, fmt: (v) => v === 0.5 ? "At the money (straddle)" : `${v.toFixed(2)} delta each (strangle)` }],
    specs: (p) => p.legs === 0.5 ? [{ t: "P", side: 1, atm: true }, { t: "C", side: 1, atm: true }]
      : [{ t: "P", side: 1, d: p.legs, rnd: "dn" }, { t: "C", side: 1, d: p.legs, rnd: "up" }],
    capital(legs, net) { return Math.max(-net, 1); },
    capitalNote: "You pay the debit up front; that's also the most you can lose.",
    seller: false, window: "mon_fri",
    how: "Buy a call and a put at the same strike, near today's price. You pay for both. At expiration one of them is worth whatever the stock moved past the strike, the other nothing. You profit only if the move is bigger than what you paid. Buying both a little out of the money (a long strangle) costs less but needs an even bigger move.",
    when: "Used when you expect a big move but not its direction: before earnings or a court or FDA decision, or when options look cheap next to how much the stock has been moving. The catch: before a known event, the options are already priced for a big move, and that price collapses right after the news (the volatility crush). The move has to beat what the market expected, not just be big.",
    feel: "Loses a little most weeks as time eats the premium, then occasionally wins big. It's the other side of the selling strategies: whoever sells you this straddle is running the short strangle tab.",
    greeks: [["Time passing", "hurts every day", "bad"], ["Implied volatility rising", "helps", "good"], ["A big move either way", "helps", "good"]],
  },
  collar: {
    name: "Collar", tab: "Collar", icon: "⊓",
    tagline: "Your covered call, with the premium spent on insurance for the shares.",
    view: "Own the shares, want protection", pay: "Low or no cost", risk: "Defined",
    params: [{ key: "call", label: "Sold call", opts: [0.15, 0.20, 0.25, 0.30], def: 0.20 },
             { key: "put", label: "Bought put", opts: [0.10, 0.15, 0.20, 0.25, 0.30], def: 0.20 }],
    specs: (p) => [{ t: "S", side: 1 }, { t: "P", side: 1, d: p.put, rnd: "dn" }, { t: "C", side: -1, d: p.call, rnd: "up", role: "short" }],
    capital(legs, net, S) { return S * 100 - net; },
    capitalNote: "Capital is the 100 shares, adjusted by what the options paid or cost.",
    seller: false, collar: true, window: "mon_fri",
    how: "Own 100 shares, sell a covered call above the price, and use that premium to buy a put below it. The put is a floor: below its strike, every dollar the stock loses the put gains. The call is a ceiling: above its strike, your gains stop. Set the two so the call pays for the put and it's a zero-cost collar.",
    when: "Used by long-term holders to protect a big gain or ride out a risky stretch (earnings, a shaky market) without selling the shares and triggering taxes. For a wheel trader, it's the covered call with a safety net: less income, since the premium buys the put, but a bad week has a floor.",
    feel: "Smoother than holding: the worst weeks are capped by the put, the best weeks by the call. Over long periods it usually trails just holding, since the puts cost money and rallies get capped. You're buying calm, and the backtest shows the price of that calm.",
    greeks: [["Stock falling", "cushioned below the put", "good"], ["Stock rising", "capped above the call", "bad"], ["Time passing", "mostly neutral (one option bought, one sold)", "info"]],
  },
};
const ORDER = ["condor", "strangle", "bullput", "straddle", "collar"];
const fmtD = (v) => v.toFixed(2) + " delta";

// ------------------------------------------------------------------ building a position
// From the model (backtest): strikes from deltas at volatility v, prices from Black-Scholes.
function modelLegs(key, p, S, t, v) {
  const st = stepOf(S), legs = [];
  for (const sp of STRATS[key].specs(p)) {
    if (sp.t === "S") { legs.push({ t: "S", side: 1, K: null, px: S }); continue; }
    let K = sp.atm ? rNear(S, st) : (sp.rnd === "up" ? rUp : rDn)(deltaStrike(sp.t, S, t, v, sp.d), st);
    if (sp.beyond != null) {
      const ref = legs[sp.beyond] || null;
      if (ref && sp.t === "P" && K >= ref.K) K = ref.K - st;
      if (ref && sp.t === "C" && K <= ref.K) K = ref.K + st;
    }
    legs.push({ t: sp.t, side: sp.side, K, px: 0 });
  }
  // wings refer to legs listed after them (the condor's long put): fix in a second pass
  STRATS[key].specs(p).forEach((sp, i) => {
    if (sp.beyond == null) return;
    const ref = legs[sp.beyond], l = legs[i];
    if (sp.t === "P" && l.K >= ref.K) l.K = ref.K - st;
    if (sp.t === "C" && l.K <= ref.K) l.K = ref.K + st;
  });
  for (const l of legs) if (l.t !== "S") l.px = bs(l.t, S, l.K, t, v);
  return legs;
}

// From the live chain: the listed option nearest each target delta, at its mid price.
function chainLegs(key, p, e, S) {
  const legs = [], specs = STRATS[key].specs(p), missing = [];
  const rows = (t) => (t === "C" ? e.calls : e.puts).filter((o) => o.mid > 0);
  const toLeg = (sp, o, extra = {}) => ({ t: sp.t, side: sp.side, K: o.strike, px: o.mid, delta: o.delta, iv: o.iv, bid: o.bid, ask: o.ask, ...extra });
  let approx = false;
  specs.forEach((sp, i) => {
    if (sp.t === "S") { legs[i] = { t: "S", side: 1, K: null, px: S }; return; }
    if (sp.atm) {
      if (e.atm && e.atm.call && e.atm.put) { const o = sp.t === "C" ? e.atm.call : e.atm.put; legs[i] = toLeg(sp, { ...o, strike: e.atm.strike }); }
      else {                                 // older data files: the nearest out-of-the-money call and put
        const r = rows(sp.t); const o = sp.t === "C" ? r[0] : r[r.length - 1];
        if (o) { legs[i] = toLeg(sp, o); approx = true; } else missing.push(i);
      }
      return;
    }
  });
  specs.forEach((sp, i) => {                 // plain legs first, then wings (which need their reference leg)
    if (sp.t === "S" || sp.atm || sp.beyond != null) return;
    const r = rows(sp.t);
    const o = r.slice().sort((a, b) => Math.abs(Math.abs(a.delta) - sp.d) - Math.abs(Math.abs(b.delta) - sp.d))[0];
    if (o) legs[i] = toLeg(sp, o); else missing.push(i);
  });
  specs.forEach((sp, i) => {
    if (sp.beyond == null) return;
    const ref = legs[sp.beyond];
    if (!ref) { missing.push(i); return; }
    const r = rows(sp.t).filter((o) => sp.t === "P" ? o.strike < ref.K : o.strike > ref.K);
    const o = r.slice().sort((a, b) => Math.abs(Math.abs(a.delta) - sp.d) - Math.abs(Math.abs(b.delta) - sp.d))[0];
    if (o) legs[i] = toLeg(sp, o); else missing.push(i);
  });
  return { legs, missing, approx };
}

function metrics(key, legs, S) {
  const Ks = [...new Set(optLegs(legs).map((l) => l.K))].sort((a, b) => a - b);
  const hi = Math.max(S, ...Ks) * 3;
  const pts = [0, ...Ks, hi].map((X) => [X, plAt(legs, X)]);
  const slopeHi = legs.reduce((a, l) => a + (l.t === "C" || l.t === "S" ? l.side : 0), 0);
  let maxG = Math.max(...pts.map((p) => p[1])), maxL = Math.min(...pts.map((p) => p[1]));
  if (slopeHi > 0) maxG = Infinity;
  if (slopeHi < 0) maxL = -Infinity;
  const be = [];
  for (let i = 1; i < pts.length; i++) {
    const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
    if ((y0 < 0 && y1 >= 0) || (y0 >= 0 && y1 < 0)) be.push(x0 + (x1 - x0) * (0 - y0) / (y1 - y0));
  }
  const net = netOf(legs);
  return { net, maxG, maxL, be, capital: STRATS[key].capital(legs, net, S) };
}

// Chance the position ends in profit if the price at expiration is lognormal with volatility v.
function marketOdds(legs, S, t, v, lo, hi) {
  const sd = v * Math.sqrt(t), mu = Math.log(S) + (RATE - v * v / 2) * t;
  const cdf = (X) => X <= 0 ? 0 : ncdf((Math.log(X) - mu) / sd);
  const prob = (a, b) => cdf(b) - cdf(a);
  let win = 0; const n = 600, a = S * Math.exp(-6 * sd), b = S * Math.exp(6 * sd);
  for (let i = 0; i < n; i++) {
    const x0 = a + (b - a) * i / n, x1 = a + (b - a) * (i + 1) / n;
    if (plAt(legs, (x0 + x1) / 2) > 0) win += prob(x0, x1);
  }
  return { win, prob };
}

window.STRAT_CORE = { ncdf, ninv, bs, deltaStrike, modelLegs, chainLegs, metrics, plAt, netOf, marketOdds, STRATS, stepOf };

// ------------------------------------------------------------------ formatting
const sg = (v) => v == null || Number.isNaN(v) ? "–" : !Number.isFinite(v) ? (v > 0 ? `<span class="pos">Unlimited</span>` : `<span class="neg">Unlimited</span>`)
  : `<span class="${v >= 0 ? "pos" : "neg"}" style="white-space:nowrap">${v >= 0 ? "+" : "−"}$${Math.abs(Math.round(v)).toLocaleString("en-US")}</span>`;
const d0 = (v) => v == null ? "–" : "$" + Math.round(v).toLocaleString("en-US");
const px2 = (v) => "$" + (Math.round(v * 100) / 100).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const kfmt = (k) => "$" + (Number.isInteger(k) ? k.toLocaleString("en-US") : k.toLocaleString("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 2 }));
const pc = (v, d = 0) => v == null || Number.isNaN(v) ? "–" : (v * 100).toFixed(d) + "%";
const legName = (l) => l.t === "S" ? "100 shares" : `${kfmt(l.K)} ${l.t === "C" ? "call" : "put"}`;
const legShort = (l) => l.t === "S" ? "+100 sh" : `${l.side > 0 ? "+" : "−"}${kfmt(l.K).slice(1)}${l.t}`;
const yearOf = (d) => d.slice(0, 4);
function etNow() {
  const parts = new Intl.DateTimeFormat("en-US", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false })
    .formatToParts(new Date()).reduce((a, p) => (a[p.type] = p.value, a), {});
  return { date: `${parts.year}-${parts.month}-${parts.day}`, hour: (+parts.hour % 24) + (+parts.minute) / 60 };
}
function sessionsUntil(expiry) {           // trading sessions left, counting what's left of today
  const now = etNow();
  let n = 0;
  const d = new Date(now.date + "T12:00:00"), end = new Date(expiry + "T12:00:00");
  const wd = d.getDay();
  if (wd > 0 && wd < 6 && now.hour < 16) n += Math.min(1, (16 - Math.max(now.hour, 9.5)) / 6.5);
  for (d.setDate(d.getDate() + 1); d <= end; d.setDate(d.getDate() + 1)) if (d.getDay() > 0 && d.getDay() < 6) n++;
  return Math.max(n, 0.25);
}
function yearsUntil(expiry) {
  const ms = new Date(expiry + "T16:00:00-04:00") - Date.now();
  return Math.max(ms / (365 * 864e5), 0.5 / 365);
}

// ------------------------------------------------------------------ state
const ST = { tab: "guide", params: {}, win: {}, basis: {}, earn: {}, all: {}, expiry: {}, data: {}, loading: {}, reports: {}, cache: {} };
for (const k of ORDER) ST.params[k] = Object.fromEntries(STRATS[k].params.map((p) => [p.key, p.def]));
try { const saved = JSON.parse(recall("cca-strat-params") || "{}"); for (const k of ORDER) Object.assign(ST.params[k], saved[k] || {}); } catch (e) {}
const saveParams = () => store("cca-strat-params", JSON.stringify(ST.params));
const tk = () => state.report && state.report.ticker;

async function loadData(t) {
  if (!t || ST.data[t] !== undefined || ST.loading[t]) return;
  ST.loading[t] = true;
  try { ST.data[t] = await getJSON(`data/backtest/${encodeURIComponent(t)}_strategies.json`); }
  catch (e) { ST.data[t] = null; }
  ST.loading[t] = false;
  if (t === tk()) render();
}
function rowsOf(D, win) {
  const W = D && D.windows && D.windows[win];
  if (!W) return [];
  const f = D.fields;
  return W.rows.map((a) => Object.fromEntries(f.map((k, i) => [k, a[i]])));
}

// ------------------------------------------------------------------ backtest
function basisOptions(D, win) {
  const out = [];
  const idx = D && D.iv_source && D.iv_source.kind === "index";
  if (idx) out.push(["index", `Market's implied volatility: ${win === "four_week" ? D.iv_source.long : D.iv_source.short}`]);
  const m = D && D.measured_ratio;
  if (m && m.days >= 1) out.push(["m", `Recent movement × ${m.median.toFixed(2)}: measured on this stock's option chains (${m.days} day${m.days === 1 ? "" : "s"})`]);
  for (const r of [1.0, 0.9, 1.1, 1.2, 1.3]) out.push(["r" + r.toFixed(1), `Recent movement × ${r.toFixed(1)}${r === 1 ? ": options priced at exactly what the stock had been doing" : ""}`]);
  return out;
}
function defaultBasis(D) {
  if (D && D.iv_source && D.iv_source.kind === "index") return "index";
  if (D && D.measured_ratio && D.measured_ratio.days >= 10) return "m";
  return "r1.0";
}
function ivFor(row, basis, D) {
  if (basis === "index") return row.iv;
  const ratio = basis === "m" ? (D.measured_ratio && D.measured_ratio.median) || 1 : parseFloat(basis.slice(1));
  return row.hv20 > 0 ? row.hv20 * ratio : null;
}
function runBacktest(key, D, win, basis, skipEarn) {
  const p = ST.params[key];
  const ck = [D.ticker, D.updated, key, JSON.stringify(p), win, basis, skipEarn].join("|");
  if (ST.cache[ck]) return ST.cache[ck];
  const out = [];
  for (const r of rowsOf(D, win)) {
    if (skipEarn && r.earnings) continue;
    const v = ivFor(r, basis, D);
    if (!(v > 0) || !(r.open > 0)) continue;
    const t = r.days / 365;
    const legs = modelLegs(key, p, r.open, t, v);
    const m = metrics(key, legs, r.open);
    const pl = plAt(legs, r.close, D.fee ?? FEE);
    const hold = (r.close - r.open) * 100;
    out.push({ ...r, v, legs, net: m.net, cap: m.capital, pl, hold, edge: STRATS[key].collar ? pl - hold : pl,
      z: Math.log(r.close / r.open) / (v * Math.sqrt(t)) });
  }
  ST.cache[ck] = out;
  return out;
}
function statsOf(tr, f = "pl") {
  const n = tr.length;
  if (!n) return { n: 0 };
  const v = tr.map((x) => x[f]), wins = v.filter((x) => x > 0), losses = v.filter((x) => x <= 0);
  const total = v.reduce((a, b) => a + b, 0);
  let peak = 0, cum = 0, dd = 0;
  for (const x of v) { cum += x; peak = Math.max(peak, cum); dd = Math.min(dd, cum - peak); }
  return { n, total, avg: total / n, win: wins.length / n, avgWin: wins.length ? wins.reduce((a, b) => a + b, 0) / wins.length : null,
    avgLoss: losses.length ? losses.reduce((a, b) => a + b, 0) / losses.length : null, worst: Math.min(...v), best: Math.max(...v), dd };
}

const MOVE_BUCKETS = [[-Infinity, -1.5, "Fell more than 1.5 expected moves"], [-1.5, -0.75, "Fell 0.75 to 1.5 expected moves"],
  [-0.75, -0.25, "Fell 0.25 to 0.75"], [-0.25, 0.25, "Barely moved (within 0.25)"], [0.25, 0.75, "Rose 0.25 to 0.75"],
  [0.75, 1.5, "Rose 0.75 to 1.5 expected moves"], [1.5, Infinity, "Rose more than 1.5 expected moves"]];
function conditions(key, tr, basis, skipEarn) {
  const f = STRATS[key].collar ? "edge" : "pl";
  const group = (title, sub, defs) => ({ title, sub, rows: defs.map(([label, test]) => ({ label, ...statsOf(tr.filter(test), f) })).filter((r) => r.n) });
  const out = [
    group("How far the stock moved", `Measured in expected moves: how much the options were priced for. A condor or strangle wants the middle rows; a straddle wants the ends.`,
      MOVE_BUCKETS.map(([a, b, l]) => [l, (x) => x.z >= a && x.z < b])),
    group("How volatile the stock had been", "Its recent volatility compared with the year before the trade: the calmest third, the middle, the most volatile third.",
      [["Calm", (x) => x.hv_rank != null && x.hv_rank < 1 / 3], ["Normal", (x) => x.hv_rank != null && x.hv_rank >= 1 / 3 && x.hv_rank < 2 / 3], ["Volatile", (x) => x.hv_rank != null && x.hv_rank >= 2 / 3]]),
    group("Trend going in", `The same trend flags the covered-call tabs use: ${term("running hot")} (RSI over 70 or 6% above the 20-day average), sliding (RSI under 30 or 6% below), or neither.`,
      [["Running hot", (x) => (x.rsi14 > 70) || (x.pct_ma20 > 0.06)], ["Neither", (x) => !((x.rsi14 > 70) || (x.pct_ma20 > 0.06)) && !((x.rsi14 < 30) || (x.pct_ma20 < -0.06))], ["Sliding", (x) => (x.rsi14 < 30) || (x.pct_ma20 < -0.06)]]),
  ];
  if (!skipEarn) out.push(group("Earnings inside the trade", "Earnings weeks move more. Their premiums are estimated from ordinary weeks here, so the real ones were higher (see the note above).",
    [["Earnings week", (x) => x.earnings], ["No earnings", (x) => !x.earnings]]));
  if (basis === "index") out.push(group("How options were priced", "Implied volatility (what the options charged) divided by the recent actual movement, when the trade was opened.",
    [["Priced below recent movement", (x) => x.v / x.hv20 < 1], ["Up to 1.3× recent movement", (x) => x.v / x.hv20 >= 1 && x.v / x.hv20 < 1.3], ["More than 1.3× recent movement", (x) => x.v / x.hv20 >= 1.3]]));
  const years = [...new Set(tr.map((x) => yearOf(x.entry)))];
  out.push(group("By year", "Each calendar year on its own.", years.map((y) => [y, (x) => yearOf(x.entry) === y])));
  return out.filter((g) => g.rows.length > 1);
}

// ------------------------------------------------------------------ charts
const W_ = 640, H_ = 250, PAD = { l: 58, r: 12, t: 12, b: 28 };
function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v * 1e6) / 1e6);
  return out;
}
const yLabel = (v) => (v < 0 ? "−" : "") + "$" + Math.abs(v).toLocaleString("en-US", { maximumFractionDigits: 0 });

function payoffChart(el, legs, S, em, be) {
  const Ks = optLegs(legs).map((l) => l.K);
  let lo = Math.min(S - 2.6 * em, ...Ks.map((k) => k - 0.4 * em)), hi = Math.max(S + 2.6 * em, ...Ks.map((k) => k + 0.4 * em));
  lo = Math.max(lo, 0.01);
  const n = 160, xs = Array.from({ length: n + 1 }, (_, i) => lo + (hi - lo) * i / n), ys = xs.map((x) => plAt(legs, x));
  let yLo = Math.min(0, ...ys), yHi = Math.max(0, ...ys);
  const padY = (yHi - yLo) * 0.08 || 10; yLo -= padY; yHi += padY;
  const X = (x) => PAD.l + (x - lo) / (hi - lo) * (W_ - PAD.l - PAD.r), Y = (y) => PAD.t + (yHi - y) / (yHi - yLo) * (H_ - PAD.t - PAD.b);
  const line = xs.map((x, i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(ys[i]).toFixed(1)}`).join("");
  const area = line + `L${X(hi).toFixed(1)},${Y(0).toFixed(1)}L${X(lo).toFixed(1)},${Y(0).toFixed(1)}Z`;
  const id = "pc" + Math.random().toString(36).slice(2, 7);
  const yt = niceTicks(yLo, yHi, 5), xt = niceTicks(lo, hi, 6);
  el.innerHTML = `<svg viewBox="0 0 ${W_} ${H_}" role="img" aria-label="Profit or loss at expiration by stock price">
    <defs><clipPath id="${id}u"><rect x="0" y="0" width="${W_}" height="${Y(0)}"/></clipPath><clipPath id="${id}d"><rect x="0" y="${Y(0)}" width="${W_}" height="${H_}"/></clipPath></defs>
    <rect x="${X(S - em)}" y="${PAD.t}" width="${X(S + em) - X(S - em)}" height="${H_ - PAD.t - PAD.b}" fill="var(--cash-band)"/>
    ${yt.map((v) => `<line x1="${PAD.l}" x2="${W_ - PAD.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${PAD.l - 6}" y="${Y(v) + 4}" text-anchor="end">${yLabel(v)}</text>`).join("")}
    ${xt.map((v) => `<text x="${X(v)}" y="${H_ - 8}" text-anchor="middle">${yLabel(v)}</text>`).join("")}
    <path d="${area}" fill="var(--good)" opacity=".18" clip-path="url(#${id}u)"/><path d="${area}" fill="var(--bad)" opacity=".18" clip-path="url(#${id}d)"/>
    <line x1="${PAD.l}" x2="${W_ - PAD.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--muted)"/>
    <line x1="${X(S)}" x2="${X(S)}" y1="${PAD.t}" y2="${H_ - PAD.b}" stroke="var(--accent)" stroke-dasharray="4 3"/>
    <text x="${X(S)}" y="${PAD.t + 10}" text-anchor="middle" style="fill:var(--accent)">now</text>
    ${be.filter((b) => b > lo && b < hi).map((b) => `<circle cx="${X(b)}" cy="${Y(0)}" r="3.5" fill="var(--ink)"/>`).join("")}
    <path d="${line}" fill="none" stroke="var(--series-1)" stroke-width="2.2" stroke-linejoin="round"/>
    <line class="hov" x1="0" x2="0" y1="${PAD.t}" y2="${H_ - PAD.b}" stroke="var(--muted)" visibility="hidden"/>
  </svg><div class="chart-tip" hidden></div>`;
  hover(el, (fx) => { const x = lo + (fx * W_ - PAD.l) / (W_ - PAD.l - PAD.r) * (hi - lo); return x < lo || x > hi ? null : { x: X(x), html: `<b>Stock at ${px2(x)}</b><div class="row"><span>At expiration</span><span>${sg(plAt(legs, x))}</span></div>` }; });
}

function lineChart(el, dates, series) {
  const n = dates.length;
  if (n < 2) { el.innerHTML = ""; return; }
  const all = series.flatMap((s) => s.ys);
  let yLo = Math.min(0, ...all), yHi = Math.max(0, ...all);
  const padY = (yHi - yLo) * 0.06 || 10; yLo -= padY; yHi += padY;
  const X = (i) => PAD.l + i / (n - 1) * (W_ - PAD.l - PAD.r), Y = (y) => PAD.t + (yHi - y) / (yHi - yLo) * (H_ - PAD.t - PAD.b);
  const years = []; dates.forEach((d, i) => { if (!i || yearOf(d) !== yearOf(dates[i - 1])) years.push([i, yearOf(d)]); });
  const every = Math.ceil(years.length / 8);
  el.innerHTML = `<svg viewBox="0 0 ${W_} ${H_}" role="img" aria-label="Cumulative result">
    ${niceTicks(yLo, yHi, 5).map((v) => `<line x1="${PAD.l}" x2="${W_ - PAD.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="var(--line)"/><text x="${PAD.l - 6}" y="${Y(v) + 4}" text-anchor="end">${yLabel(v)}</text>`).join("")}
    ${years.filter((_, k) => k % every === 0).map(([i, y]) => `<text x="${X(i)}" y="${H_ - 8}" text-anchor="middle">${y}</text>`).join("")}
    <line x1="${PAD.l}" x2="${W_ - PAD.r}" y1="${Y(0)}" y2="${Y(0)}" stroke="var(--muted)"/>
    ${series.map((s) => `<path d="${s.ys.map((y, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(y).toFixed(1)}`).join("")}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round"/>`).join("")}
    <line class="hov" x1="0" x2="0" y1="${PAD.t}" y2="${H_ - PAD.b}" stroke="var(--muted)" visibility="hidden"/>
  </svg><div class="chart-tip" hidden></div>`;
  hover(el, (fx) => {
    const i = Math.round((fx * W_ - PAD.l) / (W_ - PAD.l - PAD.r) * (n - 1));
    if (i < 0 || i >= n) return null;
    return { x: X(i), html: `<b>${fmtDate(dates[i])} ${yearOf(dates[i])}</b>` + series.map((s) => `<div class="row"><span>${s.label}</span><span>${sg(s.ys[i])}</span></div>`).join("") };
  });
}
function hover(el, fn) {
  const svg = el.querySelector("svg"), tip = el.querySelector(".chart-tip"), ln = svg.querySelector(".hov");
  const move = (ev) => {
    const b = svg.getBoundingClientRect(), fx = (ev.clientX - b.left) / b.width, h = fn(fx);
    if (!h) { tip.hidden = true; ln.setAttribute("visibility", "hidden"); return; }
    ln.setAttribute("x1", h.x); ln.setAttribute("x2", h.x); ln.setAttribute("visibility", "visible");
    tip.innerHTML = h.html; tip.hidden = false;
    const px = h.x / W_ * b.width, w = tip.offsetWidth;
    tip.style.left = Math.min(Math.max(px - w / 2, 0), b.width - w) + "px"; tip.style.top = "0px";
  };
  svg.addEventListener("pointermove", move);
  svg.addEventListener("pointerdown", move);
  svg.addEventListener("pointerleave", () => { tip.hidden = true; ln.setAttribute("visibility", "hidden"); });
}

// ------------------------------------------------------------------ tab: one strategy
function paramSelects(key, prefix) {
  return STRATS[key].params.map((p) => `<label><span>${esc(p.label)}</span><select data-param="${p.key}" data-strat="${key}">
    ${p.opts.map((o) => `<option value="${o}"${ST.params[key][p.key] === o ? " selected" : ""}>${p.fmt ? p.fmt(o) : fmtD(o)}</option>`).join("")}</select></label>`).join("");
}
function introCard(key) {
  const s = STRATS[key];
  return `<section class="card">
    <div class="stitle"><span class="sicon" aria-hidden="true">${s.icon}</span><div><h3>${esc(s.name)}</h3><p class="sub" style="margin:0">${esc(s.tagline)}</p></div></div>
    <div class="schips"><span class="chip info">View: ${esc(s.view)}</span><span class="chip info">${esc(s.pay)}</span><span class="chip ${s.risk === "Undefined" ? "bad" : "good"}">Risk: ${esc(s.risk)}</span></div>
    <div class="scols">
      <div><h4>How it works</h4><p>${s.how}</p></div>
      <div><h4>When traders use it</h4><p>${s.when}</p></div>
      <div><h4>What it's like to run</h4><p>${s.feel}</p>
        <ul class="sgreeks">${s.greeks.map(([a, b, c]) => `<li><span>${esc(a)}</span><span class="chip ${c}">${esc(b)}</span></li>`).join("")}</ul></div>
    </div></section>`;
}

function buildCard(key) {
  const r = state.report, s = STRATS[key];
  if (!r || !r.expiries || !r.expiries.length) return `<section class="card"><h3>Build it today</h3><p class="sub">Pick a ticker above to load today's option prices.</p></section>`;
  const exps = r.expiries;
  const ex = exps.find((e) => e.date === ST.expiry[key]) || exps[0];
  ST.expiry[key] = ex.date;
  const S = r.price;
  const { legs, missing, approx } = chainLegs(key, ST.params[key], ex, S);
  const head = `<section class="card" id="stBuild"><div class="chain-head" style="margin-bottom:4px"><h3>Build it today · ${esc(r.ticker)} at ${px2(S)}</h3></div>
    <p class="sub tabq">The ${esc(s.name.toLowerCase())} from the latest option prices, what it pays and risks, and when it would make or lose money.</p>
    <div class="controls btctl"><label><span>Expiration</span><select data-expiry="${key}">${exps.map((e) => `<option value="${e.date}"${e.date === ex.date ? " selected" : ""}>${fmtDate(e.date)} · ${e.dte} day${e.dte === 1 ? "" : "s"}</option>`).join("")}</select></label>${paramSelects(key)}</div>`;
  if (missing.length || legs.some((l) => !l)) return head + `<p class="notice" style="margin-top:14px">The ${fmtDate(ex.date)} chain doesn't list every strike this needs (the site saves options between 0.03 and 0.50 delta). Try a different expiration or wider strikes.</p></section>`;
  const t = yearsUntil(ex.date), m = metrics(key, legs, S);
  const atmIv = ex.atm && ex.atm.call && ex.atm.put ? (ex.atm.call.iv + ex.atm.put.iv) / 2 : (r.atm_iv || r.hv20 || 0.3);
  const em = S * atmIv * Math.sqrt(t);
  const odds = marketOdds(legs, S, t, atmIv);
  // history: this stock's past trades of about the same length, rescaled to today's volatility
  const D = ST.data[r.ticker];
  const sess = sessionsUntil(ex.date);
  let hist = null;
  if (D && D.windows) {
    const wins = Object.keys(D.windows).map((w) => { const rows = rowsOf(D, w); return [w, rows, rows.length ? rows[rows.length - 1].sessions : 99]; });
    const [wkey, rows] = wins.sort((a, b) => Math.abs(a[2] - sess) - Math.abs(b[2] - sess))[0] || [];
    const earnIn = r.earnings_date && r.earnings_date <= ex.date && r.earnings_date >= etNow().date;
    let use = rows ? rows.filter((x) => (earnIn ? x.earnings : !x.earnings) && x.hv20 > 0) : [];
    let note = earnIn ? "earnings weeks only, since earnings fall inside this trade" : "weeks without earnings";
    if (earnIn && use.length < 6) { use = rows.filter((x) => x.hv20 > 0); note = "all weeks (too few past earnings weeks)"; }
    const scale = (r.hv20 || atmIv) * Math.sqrt(sess / 252);
    const Xs = use.map((x) => S * Math.exp(Math.log(x.close / x.open) / (x.hv20 * Math.sqrt(x.sessions / 252)) * scale));
    if (Xs.length >= 20) {
      const pls = Xs.map((X) => plAt(legs, X));
      hist = { Xs, win: pls.filter((v) => v > 0).length / pls.length, avg: pls.reduce((a, b) => a + b, 0) / pls.length, n: Xs.length, label: D.windows[wkey].label, note };
    }
  } else if (D === undefined) loadData(r.ticker);
  let evMarket = 0; { const sd = atmIv * Math.sqrt(t), a = S * Math.exp(-6 * sd), b = S * Math.exp(6 * sd), n = 600;
    for (let i = 0; i < n; i++) { const x0 = a + (b - a) * i / n, x1 = a + (b - a) * (i + 1) / n; evMarket += plAt(legs, (x0 + x1) / 2) * odds.prob(x0, x1); } }

  const legRows = legs.map((l) => l.t === "S" ? `<tr><td>Own</td><td>100 shares</td><td>${px2(l.px)}</td><td>1.00</td><td class="muted hs">–</td></tr>`
    : `<tr><td>${l.side > 0 ? "Buy" : "Sell"}</td><td>${legName(l)}</td><td>${px2(l.px)}</td><td>${(l.delta ?? 0).toFixed(2)}</td><td class="muted hs">${l.bid != null ? `${px2(l.bid)} / ${px2(l.ask)}` : "–"}</td></tr>`).join("");
  const credit = m.net >= 0;
  const breakevens = m.be.length ? m.be.map((b) => px2(b)).join(" and ") : "none";
  const buckets = [[-Infinity, -2, "Falls more than 2 expected moves"], [-2, -1, "Falls 1 to 2 expected moves"], [-1, -0.5, "Falls ½ to 1"], [-0.5, 0.5, "Stays within ½ an expected move"],
    [0.5, 1, "Rises ½ to 1"], [1, 2, "Rises 1 to 2 expected moves"], [2, Infinity, "Rises more than 2 expected moves"]];
  const scen = buckets.map(([a, b, label]) => {
    const lo = Math.max(S + Math.max(a, -3.5) * em, 0.01), hi = S + Math.min(b, 3.5) * em;
    const vals = Array.from({ length: 41 }, (_, i) => plAt(legs, lo + (hi - lo) * i / 40));
    const mn = Math.min(...vals), mx = Math.max(...vals);
    const range = Math.abs(mx - mn) < 1 ? sg(mn) : `${sg(mn)} to ${sg(mx)}`;
    const cls = mn > 0 ? "win" : mx <= 0 ? "loss" : "mixed";
    const pa = Number.isFinite(a) ? S + a * em : 0, pb = Number.isFinite(b) ? S + b * em : Infinity;
    const mkt = odds.prob(Math.max(pa, 1e-6), pb === Infinity ? 1e12 : pb);
    const h = hist ? hist.Xs.filter((X) => X >= pa && X < pb).length / hist.n : null;
    const prices = !Number.isFinite(a) ? `below ${px2(pb)}` : !Number.isFinite(b) ? `above ${px2(pa)}` : `${px2(pa)} to ${px2(pb)}`;
    return `<tr class="${cls}"><td>${label}<span class="r">${prices}</span></td><td>${range}</td><td class="odds"><span><b>${pc(mkt)}</b> market</span><span><b>${h == null ? "–" : pc(h)}</b> before</span></td></tr>`;
  }).join("");
  const cap = m.capital;
  return head + `
    <div class="scroll"><table class="slegs"><thead><tr><th>Action</th><th>Contract</th><th>Price (mid)</th><th>${term("delta", "Delta")}</th><th class="hs">Bid / ask</th></tr></thead><tbody>${legRows}</tbody></table></div>
    ${approx ? `<p class="muted">This data file doesn't have a same-strike call and put yet, so this uses the nearest call and put on each side (a narrow strangle). The next update adds them.</p>` : ""}
    <div class="tiles" style="margin-top:14px">
      <div class="tile"><div class="t-label">${credit ? `Net ${term("credit", "credit")}` : `Net ${term("debit", "debit")}`}${s.collar ? " (options)" : ""}</div><div class="t-big">${d0(Math.abs(m.net))}</div>
        <div class="t-line">${credit ? "Paid to you now, per position" : "You pay this now, per position"} · fees ${d0(FEE * optLegs(legs).length)}</div></div>
      <div class="tile"><div class="t-label">Most you can make / lose</div><div class="t-big">${sg(m.maxG)} <span class="muted">/</span> ${sg(m.maxL)}</div>
        <div class="t-line">${term("breakeven", "Breakeven")} at expiration: ${breakevens}</div></div>
      <div class="tile"><div class="t-label">Chance of a profit</div><div class="t-big">${pc(odds.win)}</div>
        <div class="t-line">Market's odds, from the ${pc(atmIv)} ${term("implied volatility")}${hist ? ` · ${pc(hist.win)} in this stock's past ${hist.n} trades` : ""}</div></div>
      <div class="tile"><div class="t-label">Capital tied up</div><div class="t-big">${d0(cap)}</div>
        <div class="t-line">${esc(s.capitalNote)}</div></div>
    </div>
    <h4 class="subhead">Profit or loss at expiration</h4>
    <div class="chart-legend"><span><span class="key" style="background:var(--series-1)"></span>Result at each stock price</span><span><span class="keybox"></span>±1 ${term("expected move")} (${px2(S - em)} to ${px2(S + em)})</span><span>● breakeven</span></div>
    <div class="chart-wrap" id="stPayoff"></div>
    <h4 class="subhead">When you'd make money and when you'd lose</h4>
    <p class="sub">Where the stock could be at expiration, in ${term("expected move", "expected moves")} of ${px2(em)} (what today's option prices imply). In "How likely", <b>market</b> is what the option prices assume and <b>before</b> is how often ${esc(r.ticker)} actually moved that far in its past ${hist ? `${hist.n} ${esc(hist.label)} trades (${esc(hist.note)}), rescaled to today's recent volatility` : "trades (loading, or not available yet)"}.</p>
    <div class="scroll"><table class="sscen"><thead><tr><th>If the stock at expiration…</th><th>Your result</th><th>How likely</th></tr></thead><tbody>${scen}</tbody></table></div>
    <p class="sub" style="margin-top:10px">${hist ? `<strong>If ${esc(r.ticker)}'s past ${hist.n} moves repeated from today, this position would have averaged ${sg(hist.avg)}.</strong> At the market's own odds it averages ${sg(evMarket)}: about the fees, since the market prices options to break even on its own assumptions. The gap between the two is the edge history suggests, ${hist.avg > evMarket ? "in your favor" : "against you"}. It's one stock's history, not a promise.` : "History for this stock loads with the backtest data."}</p>
    <div class="tp-actions"><input id="stNote" class="stnote" placeholder="Note (optional)" maxlength="80"><button type="button" class="btn2" data-paper="${key}">Paper trade this</button></div>
    <p class="muted" style="margin-top:8px">Prices are mids from the last update (${new Date(r.updated).toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}). Real fills are usually a bit worse: sell nearer the bid, buy nearer the ask.</p>
  </section>`;
}

function btCard(key) {
  const r = state.report, t = tk(), D = ST.data[t], s = STRATS[key];
  const head = `<section class="card" id="stBt"><h3>Backtest${t ? ` · ${esc(t)}` : ""}</h3>`;
  if (D === undefined) { loadData(t); return head + `<p class="sub">Loading…</p></section>`; }
  if (!D) return head + `<p class="sub">No backtest data for ${esc(t || "this ticker")} yet. It's created by the backtest step of the next data update.</p></section>`;
  const wins = Object.keys(D.windows);
  const win = wins.includes(ST.win[key]) ? ST.win[key] : wins.includes(s.window) ? s.window : wins[0];
  ST.win[key] = win;
  const bopts = basisOptions(D, win);
  const basis = bopts.some((o) => o[0] === ST.basis[key]) ? ST.basis[key] : defaultBasis(D);
  ST.basis[key] = basis;
  const skip = ST.earn[key] != null ? ST.earn[key] : s.seller || s.collar;
  const tr = runBacktest(key, D, win, basis, skip);
  const f = s.collar ? "edge" : "pl";
  const st = statsOf(tr, f), stAll = statsOf(tr, "pl");
  const ctl = `<p class="sub tabq">Every ${esc(D.windows[win].label.replace(/^Four weeks: /, "four-week "))} trade since ${tr.length ? yearOf(tr[0].entry) : "–"}, opened at the first day's open and held to expiration. Stock prices are real. Past option prices aren't free, so each trade is priced from ${basis === "index" ? "the market's implied volatility that day (a real, published number) using the standard option formula" : "the stock's recent volatility times the ratio you pick, using the standard option formula"}. The strike settings are shared with "Build it today" above.</p>
    <div class="controls btctl">
      <label><span>Trade length</span><select data-win="${key}">${wins.map((w) => `<option value="${w}"${w === win ? " selected" : ""}>${esc(D.windows[w].label)}</option>`).join("")}</select></label>
      <label><span>Option prices from</span><select data-basis="${key}">${bopts.map(([v, l]) => `<option value="${v}"${v === basis ? " selected" : ""}>${esc(l)}</option>`).join("")}</select></label>
      <label><span>Earnings weeks</span><select data-earn="${key}"><option value="1"${skip ? " selected" : ""}>Skip them</option><option value="0"${!skip ? " selected" : ""}>Include them</option></select></label>
      ${paramSelects(key)}
    </div>
    <p class="snote">${basisNote(D, basis, win, key)}</p>`;
  if (!st.n) return head + ctl + `<p class="sub">No trades to show with these settings.</p></section>`;
  const cum = []; let c = 0; for (const x of tr) { c += x[f]; cum.push(c); }
  const series = s.collar
    ? (() => { let a = 0, b = 0; const A = [], B = []; for (const x of tr) { a += x.pl; b += x.hold; A.push(a); B.push(b); } return [{ ys: A, color: "var(--series-1)", label: "Collar (shares + options)" }, { ys: B, color: "var(--series-2)", label: "Just holding 100 shares" }]; })()
    : [{ ys: cum, color: "var(--series-1)", label: "Total so far" }];
  const avgCap = tr.reduce((a, x) => a + x.cap, 0) / tr.length;
  const years = Math.max((new Date(tr[tr.length - 1].expiry) - new Date(tr[0].entry)) / (365.25 * 864e5), 0.25);
  const lossRatio = st.avgWin && st.avgLoss ? Math.abs(st.avgLoss) / st.avgWin : null;
  const tiles = `<div class="tiles" style="margin-top:14px">
    <div class="tile"><div class="t-label">${s.collar ? "Collar vs just holding" : "Total result, one position"}</div><div class="t-big">${sg(st.total)}</div>
      <div class="t-line">${st.n} trades since ${yearOf(tr[0].entry)} · ${pc(st.win)} ${s.collar ? "beat holding" : "ended in profit"}</div>
      ${s.collar ? `<div class="t-line">Collar ${sg(stAll.total)} · holding ${sg(tr.reduce((a, x) => a + x.hold, 0))}</div>` : ""}</div>
    <div class="tile"><div class="t-label">Typical trade</div><div class="t-big">${sg(st.avg)}</div>
      <div class="t-line">Average win ${sg(st.avgWin)} · average loss ${sg(st.avgLoss)}</div>
      <div class="t-line">${lossRatio ? `The average loss is ${lossRatio.toFixed(1)}× the average win` : ""}</div></div>
    <div class="tile"><div class="t-label">Worst and best trade</div><div class="t-big">${sg(st.worst)} <span class="muted">/</span> ${sg(st.best)}</div>
      <div class="t-line">Deepest drop from a high: ${sg(st.dd)}</div></div>
    <div class="tile"><div class="t-label">On the capital tied up</div><div class="t-big">${pc(st.avg / avgCap, 1)} <span class="muted">per trade</span></div>
      <div class="t-line">Average capital ${d0(avgCap)} per position. The ${yearsTxt(years)} total is ${(st.total / avgCap).toFixed(1)}× that capital${st.total < 0 ? ", lost" : ""}.</div></div>
  </div>`;
  const conds = conditions(key, tr, basis, skip).map((g) => {
    const avgs = g.rows.map((x) => x.avg), best = Math.max(...avgs), worst = Math.min(...avgs);
    const maxAbs = Math.max(...avgs.map(Math.abs)) || 1;
    return `<div class="fvc"><div class="fvh"><span class="fvn">${esc(g.title)}</span></div><p class="fvr" style="margin:0 0 8px">${g.sub}</p>
      <div class="srows">${g.rows.map((x) => `<div class="srow ${x.avg === best && g.rows.length > 1 && best > 0 ? "best" : x.avg === worst && g.rows.length > 1 && worst < 0 ? "worst" : ""}">
        <span class="sl">${esc(x.label)} <span class="muted">${x.n}</span></span>
        <span class="sw">${pc(x.win)}<span class="muted"> won</span></span>
        <span class="sa">${sg(x.avg)}<span class="bar2"><i class="${x.avg >= 0 ? "p" : "n"}" style="width:${Math.round(Math.abs(x.avg) / maxAbs * 100)}%"></i></span></span></div>`).join("")}</div></div>`;
  }).join("");
  const showAll = ST.all[key];
  const list = tr.slice().reverse().slice(0, showAll ? undefined : 15);
  const trades = `<div class="scroll"><table class="strades"><thead><tr><th>Opened · stock</th><th>Position</th><th class="hs">${s.seller ? "Credit" : s.collar ? "Options net" : "Debit"}</th><th>${s.collar ? "vs holding" : "Result"}</th></tr></thead><tbody>${list.map((x) => `<tr>
    <td>${fmtDate(x.entry)} ${yearOf(x.entry)}${x.earnings ? ` <span class="chip neutral">earnings</span>` : ""}<span class="r">${px2(x.open)} → ${px2(x.close)} (${x.close >= x.open ? "+" : ""}${((x.close / x.open - 1) * 100).toFixed(1)}%)</span></td>
    <td>${x.legs.filter((l) => l.t !== "S").map(legShort).join(" ")}</td><td class="hs">${d0(Math.abs(x.net))}</td><td>${sg(x[f])}</td></tr>`).join("")}</tbody></table></div>
    ${tr.length > 15 ? `<button type="button" class="linkish" data-all="${key}">${showAll ? "Show the latest 15" : `Show all ${tr.length} trades`}</button>` : ""}`;
  return head + ctl + tiles + `
    <div class="chart-legend">${series.map((x) => `<span><span class="key" style="background:${x.color}"></span>${x.label}</span>`).join("")}</div>
    <div class="chart-wrap" id="stCum"></div>
    <h4 class="subhead">When it worked and when it didn't</h4>
    <p class="sub">The same trades grouped by what the stock did and the conditions going in. Each row shows how many trades, how many ${s.collar ? "beat just holding" : "made money"}, and the average result.</p>
    <div class="fvgrid">${conds}</div>
    <h4 class="subhead">Trades</h4>${trades}</section>`;
}
const yearsTxt = (y) => y >= 1.5 ? `${Math.round(y)}-year` : `${Math.round(y * 12)}-month`;
function basisNote(D, basis, win, key) {
  const rows = rowsOf(D, win).filter((x) => x.iv > 0 && x.hv20 > 0);
  const ratios = rows.map((x) => x.iv / x.hv20).sort((a, b) => a - b);
  const med = ratios.length ? ratios[Math.floor(ratios.length / 2)] : null;
  const idxLine = med ? ` Over this history, the options were priced at a median of ${med.toFixed(2)}× the recent actual movement.` : "";
  if (basis === "index") return `<strong>Real option-price levels.</strong> The volatility index is what the market was charging for options that day.${idxLine} Still estimated: every strike is priced at that one level (out-of-the-money puts usually cost a bit more) and fills are at fair value, with the fee of $${(D.fee ?? FEE).toFixed(2)} per contract.`;
  const m = D.measured_ratio;
  return `<strong>Estimated option prices.</strong> There's no free history of ${esc(D.ticker)}'s option prices, so each is priced from its recent actual movement × the ratio above. Real options usually charge more than recent movement, and much more before earnings, which is why earnings weeks are skipped by default.${m ? ` This stock's chain snapshots so far show ${m.median.toFixed(2)}× (${m.days} day${m.days === 1 ? "" : "s"}, from ${fmtDate(m.first)}); it becomes the default after 10 days.` : ""} Try a higher ratio: ${STRATS[key].seller ? "selling strategies look better when options are priced richer" : "buying strategies look worse when options are priced richer"}. The answer that holds up across ratios is the one to trust.`;
}

// ------------------------------------------------------------------ paper trades
const PKEY = "cca-strat-trades";
function ptrades() { try { return JSON.parse(recall(PKEY) || "[]"); } catch (e) { return []; } }
function psave(list) { store(PKEY, JSON.stringify(list)); }
function paperAdd(key) {
  const r = state.report, ex = r && r.expiries.find((e) => e.date === ST.expiry[key]);
  if (!ex) return;
  const { legs, missing } = chainLegs(key, ST.params[key], ex, r.price);
  if (missing.length || legs.some((l) => !l)) return;
  const note = ($("stNote") || {}).value || "";
  const p = ST.params[key];
  psave([...ptrades(), { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), strategy: key, ticker: r.ticker,
    opened: new Date().toISOString(), expiry: ex.date, price: r.price,
    legs: legs.map((l) => ({ t: l.t, side: l.side, K: l.K, px: Math.round(l.px * 100) / 100 })),
    setup: STRATS[key].params.map((x) => `${x.label} ${x.fmt ? x.fmt(p[x.key]) : fmtD(p[x.key])}`).join(", "), note }]);
  ST.msg = `Saved: ${STRATS[key].name} on ${r.ticker}, expiring ${fmtDate(ex.date)}.`;
  render();
  const el = document.getElementById("stPaper"); if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
}
async function reportFor(t) {
  if (state.report && state.report.ticker === t) return state.report;
  if (ST.reports[t] !== undefined) return ST.reports[t];
  try { ST.reports[t] = await getJSON(`data/${encodeURIComponent(t)}.json`); } catch (e) { ST.reports[t] = null; }
  return ST.reports[t];
}
function scoreTrade(x, rep) {
  const legs = x.legs;
  if (!rep) return { status: "nodata" };
  const now = etNow();
  const closes = rep.closes || [];
  const settled = x.expiry < now.date || (x.expiry === now.date && now.hour >= 16.25);
  if (settled) {
    const c = closes.filter((d) => d[0] <= x.expiry).pop();
    if (!c || (closes.length && closes[0][0] > x.expiry)) return { status: "old" };
    if (c[0] !== x.expiry && x.expiry >= (closes[closes.length - 1] || [""])[0]) return { status: "waiting" };
    return { status: "settled", close: c[1], pl: plAt(legs, c[1]), hold: legs.some((l) => l.t === "S") ? (c[1] - x.price) * 100 : null };
  }
  const ex = (rep.expiries || []).find((e) => e.date === x.expiry);
  let mark = null;
  if (ex) {
    let v = 0, ok = true;
    for (const l of legs) {
      if (l.t === "S") { v += (rep.price - l.px) * 100; continue; }
      const pool = [...(l.t === "C" ? ex.calls : ex.puts)];
      if (ex.atm && ex.atm.strike === l.K) pool.push({ strike: l.K, mid: (l.t === "C" ? ex.atm.call : ex.atm.put).mid });
      const o = pool.find((q) => Math.abs(q.strike - l.K) < 1e-6);
      if (o && o.mid > 0) v += l.side * (o.mid - l.px) * 100;
      else if ((l.t === "C" && l.K > rep.price * 1.25) || (l.t === "P" && l.K < rep.price * 0.75)) v += l.side * (0 - l.px) * 100;  // far out: worth about nothing
      else ok = false;
    }
    if (ok) mark = v - FEE * optLegs(legs).length;
  }
  return { status: x.expiry === now.date ? "today" : "open", mark, price: rep.price, updated: rep.updated, intrinsic: plAt(legs, rep.price) };
}
async function paperCard(key) {
  const all = ptrades(), mine = key ? all.filter((x) => x.strategy === key) : all;
  const tks = [...new Set(mine.map((x) => x.ticker))];
  const reps = Object.fromEntries(await Promise.all(tks.map(async (t) => [t, await reportFor(t)])));
  const scored = mine.map((x) => ({ ...x, sc: scoreTrade(x, reps[x.ticker]) })).sort((a, b) => b.opened.localeCompare(a.opened));
  const done = scored.filter((x) => x.sc.status === "settled"), open = scored.filter((x) => ["open", "today"].includes(x.sc.status));
  const f = (x) => STRATS[x.strategy] && STRATS[x.strategy].collar ? x.sc.pl - x.sc.hold : x.sc.pl;
  const tot = done.reduce((a, x) => a + f(x), 0), won = done.filter((x) => f(x) > 0).length;
  const status = (x) => {
    const s = x.sc;
    if (s.status === "settled") return `<span class="chip ${s.pl > 0 ? "good" : "bad"}">expired at ${px2(s.close)}</span>`;
    if (s.status === "today") return `<span class="chip info">expires today · ${px2(s.price)} now</span>`;
    if (s.status === "open") return `<span class="chip info">open</span>${s.mark != null ? ` <span class="muted">worth ${sg(s.mark)} at the last update</span>` : ""}`;
    if (s.status === "waiting") return `<span class="chip neutral">waiting for the closing price</span>`;
    if (s.status === "old") return `<span class="chip neutral">too old to score (closes kept 90 days)</span>`;
    return `<span class="chip neutral">can't score: not on the watchlist</span>`;
  };
  const rows = scored.map((x) => { const n = netOf(x.legs); return `<tr>
    <td><b>${esc(x.ticker)}</b> ${key ? "" : `<span class="chip info">${esc(STRATS[x.strategy] ? STRATS[x.strategy].tab : x.strategy)}</span> `}${x.legs.map(legShort).join(" ")}
      <span class="r">Opened ${fmtDate(x.opened.slice(0, 10))} · expires ${fmtDate(x.expiry)} · ${n >= 0 ? "credit" : "debit"} ${d0(Math.abs(n))}${x.note ? ` · ${esc(x.note)}` : ""}</span>
      <button type="button" class="linkish" data-pdel="${esc(x.id)}">delete</button></td>
    <td>${status(x)}</td><td>${x.sc.status === "settled" ? sg(f(x)) : "–"}</td></tr>`; }).join("");
  const msg = ST.msg; ST.msg = "";
  return `<section class="card" id="stPaper"><h3>Your paper trades${key ? ` · ${esc(STRATS[key].name)}` : ""}</h3>
    <p class="sub tabq">Try it without real money: each saved trade is scored at expiration from the stock's closing price.${msg ? ` <strong>${esc(msg)}</strong>` : ""}</p>
    <div class="tiles"><div class="tile"><div class="t-label">Settled</div><div class="t-big">${done.length ? sg(tot) : "–"}</div><div class="t-line">${done.length} trade${done.length === 1 ? "" : "s"}${done.length ? ` · ${won} ${key && STRATS[key].collar ? "beat holding" : "made money"}` : ""}</div></div>
      <div class="tile"><div class="t-label">Open</div><div class="t-big">${open.length}</div><div class="t-line">${open.filter((x) => x.sc.mark != null).length ? `Worth ${sg(open.reduce((a, x) => a + (x.sc.mark || 0), 0))} at the last update` : "Scored after expiration"}</div></div></div>
    <div class="scroll" style="margin-top:12px"><table class="spaper"><thead><tr><th>Position</th><th>Status</th><th>${key && STRATS[key].collar ? "vs holding" : "Result"}</th></tr></thead><tbody>${rows || `<tr><td colspan="3">Nothing yet. Use "Paper trade this" above.</td></tr>`}</tbody></table></div>
    <div class="tp-actions"><button type="button" class="ghost" data-pexport>Export all (CSV)</button><button type="button" class="ghost" data-pimport>Import</button><input type="file" accept=".csv,text/csv" id="stImport" hidden></div>
    <p class="muted" style="margin-top:8px">Saved only in this browser (${all.length} across all strategies). Safari clears a site's data after about a week without a visit, so export now and then; import brings them back or onto another device.</p></section>`;
}
const PHEAD = ["id", "strategy", "ticker", "opened", "expiry", "stock_price", "legs", "setup", "note"];
const csvq = (v) => /[",\n]/.test(String(v ?? "")) ? `"${String(v).replace(/"/g, '""')}"` : String(v ?? "");
function exportCsv() {
  const lines = [PHEAD.join(",")].concat(ptrades().map((x) => [x.id, x.strategy, x.ticker, x.opened, x.expiry, x.price,
    x.legs.map((l) => `${l.t}${l.side > 0 ? "+" : "-"}${l.K ?? ""}@${l.px}`).join(" "), x.setup, x.note].map(csvq).join(",")));
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
  a.download = `strategy-paper-trades-${etNow().date}.csv`;
  document.body.appendChild(a); a.click(); a.remove();
}
function parseCsv(text) {
  const rows = []; let row = [], cur = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) { if (ch === '"' && text[i + 1] === '"') { cur += '"'; i++; } else if (ch === '"') q = false; else cur += ch; }
    else if (ch === '"') q = true; else if (ch === ",") { row.push(cur); cur = ""; }
    else if (ch === "\n" || ch === "\r") { if (ch === "\r" && text[i + 1] === "\n") i++; row.push(cur); rows.push(row); row = []; cur = ""; }
    else cur += ch;
  }
  if (cur || row.length) { row.push(cur); rows.push(row); }
  return rows.filter((r) => r.some((c) => c.trim()));
}
function importCsv(text) {
  const rows = parseCsv(text), head = rows.shift() || [];
  const ix = Object.fromEntries(head.map((h, i) => [h.trim(), i]));
  if (!("legs" in ix) || !("strategy" in ix)) { ST.msg = "That file isn't a strategy paper-trade export."; return; }
  const have = ptrades(), ids = new Set(have.map((x) => x.id));
  let added = 0;
  for (const r of rows) {
    const g = (k) => (r[ix[k]] ?? "").trim();
    const legs = g("legs").split(/\s+/).filter(Boolean).map((s) => { const m = s.match(/^([CPS])([+-])([\d.]*)@([\d.]+)$/); return m && { t: m[1], side: m[2] === "+" ? 1 : -1, K: m[3] ? +m[3] : null, px: +m[4] }; });
    if (!legs.length || legs.some((l) => !l) || !g("expiry") || !g("ticker")) continue;
    const id = g("id") || Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    if (ids.has(id)) continue;
    ids.add(id); added++;
    have.push({ id, strategy: g("strategy"), ticker: g("ticker").toUpperCase(), opened: g("opened") || new Date().toISOString(), expiry: g("expiry"), price: +g("stock_price") || null, legs, setup: g("setup"), note: g("note") });
  }
  psave(have);
  ST.msg = `Imported ${added} trade${added === 1 ? "" : "s"}${rows.length - added ? ` (${rows.length - added} already here or unreadable)` : ""}.`;
}

// ------------------------------------------------------------------ guide tab
const FACTS = {
  cc: { name: "Covered call", tab: null, view: "Neutral to mildly bullish, own the shares", build: "Own 100 shares, sell a call above", gain: "Premium, plus the rise up to the strike", loss: "The shares can fall (the premium cushions it a little)", wins: "Stock stays below the strike, or rises to it", link: "analyzer" },
  csp: { name: "Cash-secured put", tab: null, view: "Neutral to bullish, happy to buy", build: "Hold the cash, sell a put below", gain: "The premium", loss: "Strike × 100 minus the premium, if the stock goes to zero", wins: "Stock stays above the strike", link: "puts" },
  condor: { view: "Neutral: stays in a range", build: "Sell a put and a call, buy both further out", gain: "The credit", loss: "Wing width minus the credit", wins: "Stock ends between the sold strikes" },
  strangle: { view: "Neutral: stays in a range", build: "Sell a put below and a call above", gain: "The credit", loss: "Unlimited above; large below", wins: "Stock ends between the strikes" },
  bullput: { view: "Neutral to bullish", build: "Sell a put, buy a lower put", gain: "The credit", loss: "Strike width minus the credit", wins: "Stock stays above the sold put" },
  straddle: { view: "A big move, either direction", build: "Buy a call and a put at the same strike", gain: "Unlimited above; large below", loss: "What you paid", wins: "The move beats what you paid" },
  collar: { view: "Own the shares, want a floor", build: "Own 100 shares, buy a put below, sell a call above", gain: "Up to the call strike", loss: "Down to the put strike", wins: "Protects in a drop, gives up a rally" },
};
async function guide() {
  const card = (k) => {
    const f = FACTS[k], s = STRATS[k] || {};
    const name = s.name || f.name;
    const btn = s.name ? `<button type="button" class="jumpbtn" data-go="${k}">Open its tab →</button>` : `<button type="button" class="jumpbtn" data-view="${f.link}">Open its tab →</button>`;
    return `<div class="gcard${s.name ? "" : " wheel"}"><div class="gh"><b>${esc(name)}</b>${s.name ? `<span class="chip ${s.risk === "Undefined" ? "bad" : "info"}">${s.risk === "Undefined" ? "undefined risk" : s.pay.toLowerCase()}</span>` : `<span class="chip good">the wheel</span>`}</div>
      <dl><dt>Your view</dt><dd>${esc(f.view)}</dd><dt>Built from</dt><dd>${esc(f.build)}</dd><dt>Most you make</dt><dd>${esc(f.gain)}</dd><dt>Most you lose</dt><dd>${esc(f.loss)}</dd><dt>Wins when</dt><dd>${esc(f.wins)}</dd></dl>${btn}</div>`;
  };
  const all = ptrades();
  const byS = ORDER.map((k) => [k, all.filter((x) => x.strategy === k).length]).filter((x) => x[1]);
  return `<section class="card"><h3>Other options strategies</h3>
    <p class="sub tabq">The wheel is two strategies taking turns: covered calls and cash-secured puts. Here are five more that traders use for different views of a stock. Each has its own tab with the same tools as the wheel: build it from today's option prices, see when it makes or loses money, backtest it on this stock, and paper trade it.</p>
    <div class="gconcepts">
      <div><b>Every strategy is a bet on two things</b><p>Where the stock goes (up, down, or nowhere), and how far it moves compared with what the options are priced for. A stock can do exactly what you expected and the trade still lose, because the options already charged for that.</p></div>
      <div><b>Selling options vs buying them</b><p>Sellers are paid up front, and time works for them: they win most of the time and lose occasionally, sometimes big. Buyers pay up front and lose a little most of the time, winning big occasionally. Options have historically been priced for a bit more movement than actually happens, which has favored sellers on average, but the sellers' losses bunch up in crashes.</p></div>
      <div><b>The ${term("expected move")}</b><p>The move that today's option prices imply by expiration, about one standard deviation. In the market's own math the stock ends inside it about two times in three. Each tab compares that with how often this stock actually stayed inside it.</p></div>
      <div><b>${term("defined risk", "Defined vs undefined risk")}</b><p>A defined-risk trade has a bought option capping the loss, so you know the worst case before you start. Undefined risk (the short strangle) can lose far more than it collects, and brokers require margin and a higher options approval level for it.</p></div>
      <div><b>Earnings are different</b><p>Before earnings, options charge for a big move; right after, that charge vanishes (the ${term("volatility crush")}). Sellers collect more but risk a gap; buyers need the move to beat an already-high price. The backtests skip earnings weeks for selling strategies by default.</p></div>
      <div><b>What's real on these tabs</b><p>Real: today's option prices, 10 years of stock prices, and how often the stock moved each amount. Estimated: past option prices. For SPY, QQQ and IWM they come from the market's published volatility index; for single stocks, from recent movement × a ratio you can change. Every trade is held to expiration, with no early exits.</p></div>
    </div></section>
    <section class="card"><h3>Which strategy for which view</h3>
      <div class="ggrid">${["cc", "csp", ...ORDER].map(card).join("")}</div></section>
    ${byS.length ? `<section class="card"><h3>Your paper trades so far</h3><p class="sub">${byS.map(([k, n]) => `${esc(STRATS[k].name)}: ${n}`).join(" · ")}. Each tab scores its own.</p></section>` : ""}`;
}

// ------------------------------------------------------------------ page
async function render() {
  const root = $("stratView");
  if (!root || root.hidden) return;
  if (!root.dataset.built) {
    root.dataset.built = "1";
    root.innerHTML = `<nav class="subviews" id="stNav" aria-label="Strategies">${["guide", ...ORDER].map((k) => `<button type="button" data-tab="${k}">${k === "guide" ? "Guide" : esc(STRATS[k].tab)}</button>`).join("")}</nav><div id="stBody" class="wrap"></div>`;
  }
  root.querySelectorAll("#stNav button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.tab === ST.tab));
  const body = $("stBody"), key = ST.tab;
  if (key === "guide") { body.innerHTML = await guide(); return; }
  if (tk()) loadData(tk());
  const y = window.scrollY;
  body.innerHTML = introCard(key) + buildCard(key) + btCard(key) + `<div id="stPaperSlot"></div>`;
  window.scrollTo(0, y);
  const r = state.report;
  if ($("stPayoff") && r) {
    const ex = r.expiries.find((e) => e.date === ST.expiry[key]);
    const { legs } = chainLegs(key, ST.params[key], ex, r.price);
    const t = yearsUntil(ex.date), iv = ex.atm && ex.atm.call && ex.atm.put ? (ex.atm.call.iv + ex.atm.put.iv) / 2 : (r.atm_iv || r.hv20 || 0.3);
    payoffChart($("stPayoff"), legs, r.price, r.price * iv * Math.sqrt(t), metrics(key, legs, r.price).be);
  }
  const D = ST.data[tk()];
  if ($("stCum") && D) {
    const s = STRATS[key], tr = runBacktest(key, D, ST.win[key], ST.basis[key], ST.earn[key] != null ? ST.earn[key] : s.seller || s.collar);
    let a = 0, b = 0; const A = [], B = [];
    for (const x of tr) { a += s.collar ? x.pl : x.pl; b += x.hold; A.push(a); B.push(b); }
    lineChart($("stCum"), tr.map((x) => x.expiry), s.collar ? [{ ys: A, color: "var(--series-1)", label: "Collar" }, { ys: B, color: "var(--series-2)", label: "Holding" }] : [{ ys: A, color: "var(--series-1)", label: "Total so far" }]);
  }
  const slot = $("stPaperSlot"), html = await paperCard(key);
  if (ST.tab === key && slot.isConnected) slot.outerHTML = html;
}
function setTab(k) {
  ST.tab = ORDER.includes(k) ? k : "guide";
  store("cca-strat-tab", ST.tab);
  try { history.replaceState(null, "", "#strategies" + (ST.tab === "guide" ? "" : "-" + ST.tab)); } catch (e) {}
  render();
}

document.addEventListener("click", (ev) => {
  const root = $("stratView"); if (!root || !root.contains(ev.target)) return;
  const b = ev.target.closest("button"); if (!b) return;
  if (b.dataset.tab) { setTab(b.dataset.tab); window.scrollTo({ top: root.offsetTop - 10 }); }
  else if (b.dataset.go) { setTab(b.dataset.go); window.scrollTo({ top: root.offsetTop - 10 }); }
  else if (b.dataset.view) setView(b.dataset.view);
  else if (b.dataset.paper) paperAdd(b.dataset.paper);
  else if (b.dataset.all) { ST.all[b.dataset.all] = !ST.all[b.dataset.all]; render(); }
  else if (b.dataset.pdel != null) { if (confirm("Delete this paper trade from this browser?")) { psave(ptrades().filter((x) => x.id !== b.dataset.pdel)); render(); } }
  else if (b.hasAttribute("data-pexport")) exportCsv();
  else if (b.hasAttribute("data-pimport")) $("stImport").click();
});
document.addEventListener("change", (ev) => {
  const root = $("stratView"); if (!root || !root.contains(ev.target)) return;
  const el = ev.target, d = el.dataset;
  if (d.param) { ST.params[d.strat][d.param] = parseFloat(el.value); saveParams(); }
  else if (d.expiry) ST.expiry[d.expiry] = el.value;
  else if (d.win) ST.win[d.win] = el.value;
  else if (d.basis) ST.basis[d.basis] = el.value;
  else if (d.earn) ST.earn[d.earn] = el.value === "1";
  else if (el.id === "stImport" && el.files[0]) { el.files[0].text().then((t) => { importCsv(t); render(); }); return; }
  else return;
  render();
});

const css = document.createElement("style");
css.textContent = `
  nav.subviews { display: flex; gap: 6px; overflow-x: auto; padding-bottom: 2px; scrollbar-width: thin; }
  nav.subviews button { flex: none; padding: 7px 13px; font: 600 13px var(--sans); border-radius: 999px; border: 1px solid var(--line); background: var(--surface); color: var(--muted); cursor: pointer; }
  nav.subviews button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); }
  #stratView { gap: 16px; }
  .stitle { display: flex; gap: 12px; align-items: center; margin-bottom: 10px; }
  .stitle h3 { margin: 0 0 2px !important; font-size: 20px !important; }
  .sicon { flex: none; width: 40px; height: 40px; border-radius: 8px; background: var(--info-bg); color: var(--info); display: grid; place-items: center; font: 600 20px var(--mono); }
  .schips { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
  .scols { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px 22px; }
  .scols h4 { margin: 0 0 4px; font-size: 14px; } .scols p { margin: 0; font-size: 14px; color: var(--muted); line-height: 1.55; }
  .sgreeks { list-style: none; margin: 10px 0 0; padding: 0; display: grid; gap: 6px; font-size: 13px; }
  .sgreeks li { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; flex-wrap: wrap; }
  .sgreeks .chip { white-space: normal; text-align: right; font-family: var(--sans); }
  @media (max-width: 800px) { .scols { grid-template-columns: minmax(0, 1fr); } }
  table.slegs, table.sscen, table.strades, table.spaper { min-width: 0; }
  table.sscen td:first-child { white-space: normal; font-family: var(--sans); }
  table.sscen td .r { display: block; font-size: 12px; color: var(--muted); font-family: var(--mono); }
  table.sscen tr.win td:first-child { box-shadow: inset 4px 0 0 var(--good); }
  table.sscen tr.loss td:first-child { box-shadow: inset 4px 0 0 var(--bad); }
  table.sscen tr.mixed td:first-child { box-shadow: inset 4px 0 0 var(--caution); }
  table.sscen td { vertical-align: top; } table.sscen td:nth-child(2) { white-space: normal; }
  table.strades td, table.spaper td { white-space: normal; vertical-align: top; }
  table.strades td:last-child, table.spaper td:last-child { white-space: nowrap; }
  table.strades td .r, table.spaper td .r { display: block; font-size: 12px; color: var(--muted); margin-top: 2px; }
  table.spaper td:first-child { font-family: var(--mono); }
  table.spaper .linkish { font-size: 12px; }
  td.odds span { display: block; white-space: nowrap; } td.odds span b { font-weight: 600; } td.odds span:last-child { color: var(--muted); }
  @media (max-width: 600px) { .hs { display: none; } table.sscen td, table.sscen th, table.strades td, table.strades th, table.spaper td, table.spaper th, table.slegs td, table.slegs th { padding-left: 6px; padding-right: 6px; } }
  .snote { font-size: 13px; color: var(--muted); background: var(--surface-2); border-radius: 6px; padding: 10px 12px; margin: 12px 0 0; line-height: 1.5; }
  .srows { display: grid; gap: 4px; }
  .srow { display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(0, .7fr) minmax(0, 1fr); gap: 8px; align-items: center; padding: 6px 8px; border-radius: 6px; background: var(--surface-2); font-size: 13px; }
  .srow.best { background: var(--good-bg); } .srow.worst { background: var(--bad-bg); }
  .srow .sw, .srow .sa { font-family: var(--mono); text-align: right; white-space: nowrap; }
  .srow .sa { display: grid; gap: 3px; justify-items: end; }
  .bar2 { display: block; width: 100%; height: 4px; border-radius: 2px; background: var(--line); overflow: hidden; }
  .bar2 i { display: block; height: 100%; } .bar2 i.p { background: var(--good); } .bar2 i.n { background: var(--bad); margin-left: auto; }
  .stnote { padding: 8px 10px; font: 14px var(--sans); border: 1px solid var(--line); border-radius: 6px; background: var(--surface); color: var(--ink); min-width: 0; flex: 1 1 200px; max-width: 320px; }
  .gconcepts { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
  .gconcepts > div { border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; background: var(--surface-2); }
  .gconcepts b { display: block; margin-bottom: 4px; font-size: 14px; } .gconcepts p { margin: 0; font-size: 13px; color: var(--muted); line-height: 1.55; }
  @media (max-width: 800px) { .gconcepts { grid-template-columns: minmax(0, 1fr); } }
  .ggrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 300px), 1fr)); gap: 10px; }
  .gcard { border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; display: grid; gap: 10px; align-content: start; }
  .gcard.wheel { background: var(--surface-2); }
  .gcard .gh { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; }
  .gcard dl { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; font-size: 13px; }
  .gcard dt { color: var(--muted); } .gcard dd { margin: 0; }
  .jumpbtn { justify-self: start; font: 600 13px var(--sans); padding: 6px 12px; border-radius: 999px; border: 1px solid var(--accent); background: transparent; color: var(--accent); cursor: pointer; }
  .jumpbtn:hover { background: var(--surface-2); }
`;
document.head.appendChild(css);

Object.assign(GLOSSARY, {
  "iron condor": ["Iron condor", "Selling a put and a call around the price and buying a further-out put and call as insurance. Paid a credit; profits if the stock stays between the sold strikes."],
  "credit": ["Credit", "Money you receive when opening a trade, because what you sold is worth more than what you bought. It's the most a credit trade can make."],
  "debit": ["Debit", "Money you pay to open a trade, because what you bought costs more than what you sold. It's the most a debit trade can lose."],
  "breakeven": ["Breakeven", "The stock price at expiration where the trade neither makes nor loses money, after the credit or debit (fees aside)."],
  "expected move": ["Expected move", "How far the stock could move by expiration according to today's option prices: price × implied volatility × √(time in years). About one standard deviation: in the market's math, the stock ends inside it about two times in three."],
  "defined risk": ["Defined risk", "A trade whose worst case is fixed at the start, because a bought option caps the loss (iron condor, spreads, collar). Undefined-risk trades (a short strangle) can lose much more than they collect."],
  "volatility crush": ["Volatility crush", "The drop in option prices right after an event like earnings. Before it, options charge for a big move; once the news is out, that part of the price disappears, even if the stock moved."],
  "margin": ["Margin", "Money a broker requires you to set aside for a trade that could lose more than you were paid. For spreads it's the width between strikes minus the credit; for naked options it's a formula based on the stock's price."],
});

window.Strat = {
  show(sub) { if (sub) ST.tab = ORDER.includes(sub) ? sub : "guide"; else { const s = recall("cca-strat-tab"); if (s) ST.tab = s; } render(); },
  onTicker() { ST.cache = {}; render(); },
  _core: window.STRAT_CORE, _st: ST, _run: runBacktest,
};
})();
