# EdgeLab — design of a research-grade intraday ML trading system

**Goal:** decide, with statistics, whether a tradable edge exists after all real costs. Making money is
a possible *outcome*, not the objective. The default assumption is **no edge**. The burden of proof is on the strategy.

**First result (Oct 2026, free 60-day data):** no edge. 57 out-of-sample trades, +2.9 bp gross,
−5.6 bp costs, **−2.7 bp net per trade**, p = 0.73. The model's AUC of ~0.75 turned out to measure
*volatility*, not direction (direction AUC 0.51). See §12 and the dashboard.

---

## 0. Skeptical framing: what is unlikely to work, and what replaces it

| Common idea | Why it usually fails | Replacement used here |
|---|---|---|
| "Predict +0.4% in 30 min" with a fixed threshold | 0.4% means very different things for PEP and TSLA, and on calm vs. volatile days. The base rate drifts with regime, so the model learns *volatility*, not direction. | **Volatility-scaled triple barrier** (b = 1·σ₃₀ₘ, clipped to 25–100 bp). The fixed 0.4% version is still run as a robustness variant. |
| Judging the model by AUC/accuracy | A barrier-hit label has high AUC just from forecasting volatility. That number says nothing about P&L. | Report **direction AUC** (among rows where a barrier was hit), gross bp by decile, and **net bp per trade after costs**. |
| Retail 5-minute momentum/mean reversion on megacaps | These are the most competed-over signals in the market. Realistic gross edges are 0–3 bp against 4–8 bp round-trip cost. | Treat this as the *null hypothesis to beat*. Lower turnover (longer horizons, fewer trades) is the most reliable lever. |
| LLM/news sentiment as the main signal | By the time headlines reach a retail feed, the price move is usually done. Timestamps are often wrong, which causes look-ahead. | Optional *filter* (e.g. skip names with breaking news), added only if it improves OOS by more than its trial-count penalty. |
| Deep learning on bars | Too many parameters for the signal-to-noise of intraday returns; overfits silently. | Shallow GBDT (depth 4, ≥200 rows per leaf) with a logistic-regression baseline. Escalate only if the GBDT beats the baseline OOS. |
| "Optimize the thresholds" | Each look at the test set is a trial; the best of N noisy trials looks great by chance. | Thresholds are set **inside each training window** (top-5% quantile of the training-validation predictions + cost break-even). Every variant is counted in `n_configs_tried` and feeds the Deflated Sharpe. |

If the honest answer after Stage 2 (2+ years of data) is still "no edge", the right move is **not** to keep tuning.
Change the problem: a longer horizon (1–3 days), a different universe (mid-caps, where spreads are wider but inefficiencies larger), or a structural edge (event-driven, rebalance flows). Or accept index investing.

---

## 1. System architecture

```
                ┌──────────────────────── RESEARCH (offline) ────────────────────────┐
 Vendor data ─► │ data.py  ─► features.py ─► labels.py ─► validation.py (walk-fwd)  │
 (bars, quotes, │  cache +     point-in-time   triple        purge/embargo           │
  corp actions, │  QC          features        barrier       │                       │
  news)         │                              ▼             ▼                       │
                │                    model.py (GBDT probs) ─► OOS probabilities      │
                │                              │                                     │
                │   backtest.py ◄── risk.py ◄──┘ (deterministic rules)               │
                │   costs.py         │                                               │
                │   metrics.py ─► report.json ─► dashboard (virtual-money replay)    │
                └───────────────────────────────┬────────────────────────────────────┘
                                                │ frozen model + config hash
                ┌────────────── LIVE / PAPER ───▼────────────────────────────────────┐
 Stream (SIP) ─►│ bar builder ─► features.py (same code) ─► model ─► rules           │
                │                                             │                      │
                │                             risk.py (SAME class) ─► broker.py      │
                │  reconcile ◄── broker fills/positions ◄─────┘  (bracket orders)    │
                │  decisions.jsonl ─► drift monitor (live vs OOS hit rate/slippage)  │
                │  kill switch: daily loss limit, 8% DD, data staleness, API errors  │
                └────────────────────────────────────────────────────────────────────┘
```

Design principles:
1. **One code path.** Features, rules and `RiskManager` are the same Python objects in backtest, paper and live. A rule that exists only in the backtest is a bug.
2. **ML proposes, rules decide.** The model outputs `p_long`, `p_short`. Deterministic rules turn those into orders: quantile gate, cost break-even, direction, sizing, limits.
3. **Pessimistic by default.** Every ambiguous fill convention resolves against the strategy.
4. **Everything versioned.** Config, data snapshot hash and code commit are written into each report.

---

## 2. Data requirements and sources

| Data | Minimum | Ideal | Source (cost, approx.) |
|---|---|---|---|
| 1-min or 5-min OHLCV bars, 26 stocks + 9 ETFs | **2 years** | 5+ years, including 2020 and 2022 | Polygon.io Stocks Starter/Developer (~$30–80/mo), Databento (pay-per-use, best quality), Alpaca (free IEX feed; SIP with paid plan), FirstRate Data (one-off historical files) |
| NBBO quotes (bid/ask) | Spread samples per symbol and time of day | Full TAQ-style quotes for the slippage model | Databento, Polygon (quotes add-on) |
| Corporate actions | Splits/dividends | — | Polygon reference API; bars must be split-adjusted *as of each date* |
| Daily context | VIX, sector ETFs, earnings calendar | Macro calendar (FOMC, CPI) | yfinance for prototyping; Polygon/FMP earnings calendar |
| News (optional) | Timestamped headlines with *first-seen* time | — | Benzinga via Polygon/Alpaca news API. Never use timestamps that a vendor revised after the fact. |
| Free prototype | 60 days of 5-min bars | — | **yfinance** (used in this repo; delayed, rate-limited, unofficial; research only) |

**Quality rules (in `data.py`):** regular session only; drop half-days and sessions with fewer than 70/78 bars rather than forward-filling; reject OHLC inconsistencies and zero volume; drop the still-forming current session; log max |5-min return| for spike checks.
**Survivorship:** a fixed list of today's megacaps is survivorship-biased. With multi-year data, rebuild the universe each month from *then-current* liquidity (top-N by dollar volume).

---

## 3. Features and target

All features at bar *t* use only bars whose close is ≤ *t*. `tests/test_no_lookahead.py` corrupts every bar after a cutoff and asserts no earlier feature changes.

| Group | Features |
|---|---|
| Momentum | log returns over 3/6/12/24 bars; vol-normalized z-returns; return since open; overnight gap; prior-day return; 5-day return |
| Volatility | realized vol over 12/48 bars, their ratio, bar range / vol, prior-day range |
| VWAP / intraday location | (close − session VWAP)/σ; position in day's range; opening-range (first 30 min) breakout distance |
| Volume | relative volume vs. the same time-of-day over the prior 10 sessions (bar and cumulative) |
| Oscillator | RSI(14) |
| Market | QQQ and SPY 3/6/12-bar returns, QQQ vol, QQQ return since open, QQQ VWAP distance |
| Sector | sector ETF 6-bar return and return since open; stock minus sector, stock minus QQQ |
| Cross-section | percentile rank across the universe at time *t* of 6-bar return, relative volume, VWAP distance |
| Calendar | minutes since open (and sin/cos), day of week |
| Optional | news-count and sentiment over the last 30/120 min (first-seen timestamps only); earnings-day flag |

**Target (triple barrier, `labels.py`):**
- Signal at close of bar *t*; market entry at **open of bar t+1**.
- Barrier b = clip(1.0·σ₃₀ₘ, 25 bp, 100 bp), with σ₃₀ₘ = 48-bar realized vol × √6.
- `y_long = 1` if the high trades **through** entry·(1+b) by 1 bp before the low touches entry·(1−b), within 6 bars (30 min) and before the session ends. `y_short` is the mirror image.
- Same-bar TP and SL counts as SL. A gapped stop fills at the bar open. Time exit is the close of bar t+6.
- The gross return of each outcome is stored with the label, so the backtest never re-derives outcomes.

---

## 4. ML methodology

- **Two binary classifiers** (long, short): LightGBM, falling back to sklearn `HistGradientBoostingClassifier` (also a histogram GBDT) when LightGBM's DLL fails to load. Settings: depth 4, 15 leaves, `min_child_samples=200`, subsample/colsample 0.7, L2 = 5, learning rate 0.03.
- **Early stopping on a time-ordered validation tail** (last 20% of training sessions), never a random split.
- **Baseline:** regularized logistic regression on the same features. If the GBDT doesn't beat it OOS, use the simpler model.
- **Evaluation per fold:** AUC, log-loss and Brier vs. the base-rate forecast, **direction AUC**, calibration table.
- **Interpretability:** permutation importance on held-out data. Split-gain importance is biased.
- **Calibration:** check the reliability table. If it is off, fit isotonic calibration on the validation tail only.
- **Decision rule** (`backtest.select_signals`): trade long if `pq_long ≥ 0.95` (percentile within the *training-period* validation predictions) **and** `p_long > (b + cost)/(2b)` (break-even with symmetric barriers) **and** `p_long ≥ p_short`. Short is the mirror image.

---

## 5. Backtesting and walk-forward methodology

- **Walk-forward by session:** expanding window, ≥20 training sessions, **1-session embargo**, 5-session test blocks. Labels never cross a session, so purging reduces to the embargo.
- **Stage 2 (real data):** retrain monthly. Test on 1-month blocks across 2+ years. Hold out the **final 6–12 months untouched** and run them once, at the end, after everything is frozen.
- **Event-driven portfolio simulation:** entries at bar open; exits at end of bar, or intrabar for barriers. Exits that complete before an entry are processed first; same-bar ties go against the strategy.
- **Costs per round trip** (`costs.py`): half-spread 1 bp + slippage 1 bp + latency 0.5 bp on entry; exits pay 0 bp at the TP limit, 2 bp at market and 4 bp at the stop. Plus $0.0035/share commission (min $0.35), SEC fee $27.80 per $1M sold and FINRA TAF. About 4.5–6 bp total. **Stress-tested at 0×, 2× and 3×.**
- **Latency:** decision at bar close, fill at the next open, plus a 0.5 bp adverse-drift charge. In live trading, measure decision-to-fill time and slippage, and recalibrate.
- **Metrics** (`metrics.py`): Sharpe, Sortino, extrapolated CAGR (flagged as meaningless on short samples), max drawdown, profit factor, win rate, trade count, daily turnover, gross/cost/net bp per trade, t-test on trade returns, block-bootstrap 95% CI on Sharpe, Probabilistic Sharpe Ratio, **Deflated Sharpe** (adjusted for the number of configurations tried), and a **random-entry baseline** (2,000 samples on the same OOS rows).
- **Robustness:** by symbol, test block, hour, side, QQQ up/down day, high/low volatility day; selectivity quantiles 80–99%; cost multipliers; fixed vs. vol-scaled barrier.

---

## 6. Risk-management rules (`risk.py`, enforced identically everywhere)

| Rule | Value |
|---|---|
| Risk per trade (loss if the stop is hit) | **0.35%** of equity (range 0.25–0.5%) |
| Shares | ⌊risk$ / (price × stop distance)⌋, capped by 25% of equity notional per position |
| Gross exposure | ≤ 1.0× equity. **No leverage, no margin.** |
| Concurrent positions | ≤ 4, one per symbol, no pyramiding |
| Daily loss limit | −1.5%. New entries are blocked once *realized P&L − open risk − new trade risk* would breach it. Trading halts for the day once realized loss hits it. |
| Max trades per day | 12 |
| Anti-revenge cooldown | No re-entry in a symbol for 30 min after a loss in it |
| Time windows | No entries in the first 15 min or after 15:20; flat by 15:55 |
| Kill switch | −8% from the equity peak stops trading. **A human must reset it**; it never restarts on its own. |
| Forbidden | Martingale, averaging down, size-up after losses, overriding the limits intraday, overnight holds |
| Operational kill | Stale data (> 2 bars), broker API errors, or position mismatch: cancel all orders, flatten, alert |
| Scaling | Raise capital only per §9 stage gates; never raise risk % to "make back" losses |

---

## 7. Execution architecture and broker/API

- **Recommended:** **Alpaca** for Stage 4–5. It has a free paper account mirroring the live API, bracket orders (entry + TP limit + SL stop, OCO), a websocket market-data stream and Python SDK `alpaca-py`. Upgrade to the SIP data feed; the free IEX feed covers only a few percent of volume.
- **Alternative:** **Interactive Brokers** (`ib_async`). Better fills and routing, IBKR-Pro tiered commissions, institutional-grade. Its API is more complex (TWS/Gateway must be running).
- **Order flow:** completed bar → features → probabilities → rules → `RiskManager.size_and_approve` → **bracket order** with an idempotent `client_order_id` (date-symbol-bar), so retries never duplicate orders.
- **Reconciliation each cycle:** the broker is the source of truth for positions and equity. The daily-loss check uses real fills.
- **Logging:** every decision is appended to `logs/decisions.jsonl` (features hash, probabilities, verdict, order id, fill price, slippage vs. the signal bar's close) for live-vs-backtest drift analysis.
- **Deployment:** one small always-on VM in us-east (near NY4/NJ data centers), process supervisor, NTP time sync, alerting to phone (Pushover/Telegram), daily automated report.
- **Safety:** live mode requires `EDGELAB_LIVE_CONFIRM=I_ACCEPT_REAL_MONEY_RISK`; the default is paper. API keys live in env vars or a secret manager, never in the repo.

---

## 8. Project structure

```
edgelab/
├── DESIGN.md                  this document
├── README.md                  how to run
├── requirements.txt
├── edgelab/
│   ├── config.py              all parameters (dataclasses), saved into every report
│   ├── data.py                download, parquet cache (grows over time), quality filters
│   ├── features.py            point-in-time features + FEATURES list
│   ├── labels.py              triple-barrier labels with pessimistic fills
│   ├── model.py               GBDT probability model + permutation importance
│   ├── validation.py          walk-forward splits, embargo, causal percentile ranks
│   ├── costs.py               spread/slippage/latency/commission/fees
│   ├── risk.py                RiskManager (shared by backtest/paper/live)
│   ├── backtest.py            signal rules, event-driven portfolio sim, random baseline
│   ├── metrics.py             Sharpe/Sortino/DD/PF, PSR, Deflated Sharpe, bootstrap
│   ├── research.py            end-to-end research run + robustness + exports
│   └── execution/broker.py    Broker ABC, AlpacaBroker (bracket orders), SimBroker
├── scripts/
│   ├── run_research.py        full run → results/ + dashboard
│   ├── build_dashboard.py     injects results into the HTML template
│   └── paper_trade.py         live paper loop (Alpaca paper or --dry-run)
├── tests/test_no_lookahead.py leakage + time-ordering tests
├── dashboard/template.html    UI (virtual-money replay, diagnostics, checklist)
├── data/raw/                  parquet cache          (git-ignored)
├── results/                   report.json, trades.csv, sim_data.json
└── logs/decisions.jsonl       paper/live decision log
```
Planned additions for Stage 2+: `edgelab/data_polygon.py`, `edgelab/universe.py` (point-in-time universe), `edgelab/monitor.py` (live drift vs. OOS), `tests/test_costs.py`, `tests/test_risk.py`.

---

## 9. Staged MVP plan, with gates

| Stage | Work | Exit gate (all must pass) |
|---|---|---|
| **0. Plumbing** ✅ | Data cache, QC, features, labels, leakage tests | `test_no_lookahead` passes; data QC clean |
| **1. Free-data prototype** ✅ | Full walk-forward + backtest + dashboard on 60 days | Pipeline runs end to end. *(Result: no edge, as expected for this sample size.)* |
| **2. Real research** | Buy 2–5 years of 1-min data; point-in-time universe; monthly retrain; pre-register the hypothesis and the number of variants (≤ 20) before looking | All checklist items in §11 except paper trading pass on the walk-forward period |
| **3. Final holdout** | Freeze code + config (git tag). Run the untouched last 6–12 months **once** | Net bp > 0, p < 0.05, within the walk-forward CI. If it fails, go back to Stage 2 with a *new* hypothesis, not a tweak |
| **4. Paper trading** (3–6 months) | `paper_trade.py` on Alpaca paper with the SIP feed; weekly drift report | ≥ 150 trades; live hit rate and net bp inside the backtest 95% CI; measured slippage ≤ modeled; zero risk-rule violations; zero unhandled errors |
| **5. Tiny real capital** | $2–5k, 0.25% risk, identical code | 3 months with live results consistent with paper/OOS |
| **6. Scale** | Increase capital ≤ 2× per quarter | Each step repeats the Stage 5 consistency test. Stop scaling if slippage grows with size |

**Stop conditions at any stage:** live Sharpe below the backtest CI lower bound for 2 months, a broken risk rule, or a −8% drawdown. Then return to research.

---

## 10. Core pseudocode

```python
# features (point-in-time): everything uses bars with close <= t
vwap      = cumsum(tp*vol by day) / cumsum(vol by day)
vwap_dist = log(close/vwap) / sigma_30m
rel_vol   = log(vol / mean(vol at same minute, previous 10 sessions))

# label: decision at close(t), fill at open(t+1)
for k in 1..6:                          # bars t+1..t+6, same session only
    if low[k] <= entry*(1-b):   y=0; ret=-b (or worse if gapped); break   # stop first on ties
    if high[k] >= entry*(1+b)*(1+1bp): y=1; ret=+b; break
else: ret = close[t+6]/entry - 1        # time stop

# walk-forward
for fold in walk_forward(days, min_train=20, test=5, embargo=1):
    m_long  = GBDT.fit(train, y_long,  early_stop_on=val_tail)
    m_short = GBDT.fit(train, y_short, early_stop_on=val_tail)
    test.p_long, test.pq_long = m_long.predict(test), rank_within(m_long.predict(val_tail))

# deterministic rule + risk
if pq_long >= 0.95 and p_long > (b + cost)/(2b) and p_long >= p_short:
    pos = risk.size_and_approve(sym, +1, price, stop_dist=b, ts)   # may return None
    if pos: broker.submit_bracket(sym, "buy", pos.shares, tp=price*(1+b), sl=price*(1-b))
```
The real implementations are in `edgelab/` (≈900 lines).

---

## 11. Edge checklist (all required before real money)

1. **≥ 300 OOS trades** (prefer 1,000+), drawn from ≥ 2 years and multiple regimes (2020 crash, 2022 bear, 2023–24 rally).
2. **Net expectancy > 0 after realistic costs**, and still > 0 at **2× costs**.
3. **Mean trade return significant**: p < 0.05 (t-test and block bootstrap agree).
4. **Beats random entries** with the same timing and rules: p < 0.05.
5. **Deflated Sharpe > 0.95** given the honest count of configurations tried.
6. **Direction AUC ≥ 0.53** and a monotonic gross-return-by-decile curve. Volatility AUC doesn't count.
7. **Beats the logistic baseline** OOS, or use the baseline.
8. **Stable across time:** ≥ 70% of walk-forward blocks profitable; no single month > 30% of P&L.
9. **Stable across names:** profitable in > 60% of symbols; top symbol < 30% of profit.
10. **Works in both regimes:** QQQ up and down days, high- and low-vol days, long and short side (or explain why not).
11. **Parameter plateau:** neighbors of the chosen threshold/barrier/horizon are also profitable. A lone spike means overfitting.
12. **Untouched holdout** passes once (Stage 3).
13. **Capacity:** the strategy's size is < 0.1% of 5-min volume in each traded bar.
14. **Paper trading** for 3–6 months matches OOS within the CI, including measured slippage.
15. **Explainable:** you can state *why* the edge exists and who is on the other side.

Current status: **1 of 12** of the dashboard's automated checks pass.

---

## 12. Sources of overfitting and false profitability to avoid

- **Look-ahead:** using bar *t*'s close to fill at bar *t*; daily features computed with the full day; VWAP or day high/low including future bars; split-adjusted prices applied retroactively; news with revised timestamps; normalizing with full-sample mean/std.
- **Leakage across folds:** random K-fold on time series; overlapping labels between train and test; early stopping on a random split (sklearn's default). The last one is why `model.py` disables it.
- **Optimistic fills:** filling limits on a touch; ignoring that stop and target can both hit in one bar; zero spread; mid-price fills; no latency; assuming fills at the signal bar's close.
- **The AUC trap** (found in this very run): a barrier label yields AUC 0.76 from volatility alone, while direction AUC is 0.51.
- **Multiple testing:** trying many features, thresholds, horizons and universes, then reporting the best. Count every variant; use Deflated Sharpe; pre-register.
- **Threshold snooping:** picking the probability cutoff on the test period. Here it is set from training-period predictions only. Moving the dashboard's sliders counts as trials too.
- **Survivorship bias:** today's megacap list didn't look like that 5 years ago. Use a point-in-time universe.
- **Regime luck:** a short bull window makes long-biased strategies look skilled. Always compare against buy-and-hold and against random entries.
- **Concentration:** one stock, one week or one news event producing the P&L.
- **Short samples:** 2 months of daily returns gives a Sharpe CI of roughly ±4 (here −4.5 to +2.8). Nothing can be concluded.
- **Compounding illusions:** extrapolated CAGR from weeks of data; ignoring that costs scale with turnover.
- **Live/backtest divergence:** different code paths for features or risk; data-feed differences (IEX vs. SIP); clock skew; partial fills; halted symbols; short-sale restrictions (SSR) and borrow availability.
- **Human overfitting:** "just one more tweak" after a bad paper month; overriding the kill switch; increasing size after losses.
