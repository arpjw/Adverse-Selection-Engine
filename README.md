# Adverse Selection Spread Decomposition Engine

A real-time market microstructure analysis engine that decomposes bid-ask spreads into their economic components using live Binance WebSocket trade data. Implements three canonical microstructure models simultaneously on a rolling window of trades, with a live terminal dashboard and optional CSV export.

---

## What this does

Every time you trade, the market maker on the other side charges you a spread. That spread isn't arbitrary; rather, it's compensation for three distinct economic costs:

```
Effective Spread = Adverse Selection + Inventory Cost + Order Processing
```

This engine estimates each component in real time:

- **Adverse selection** — the cost of potentially trading against someone who knows more than you. If informed traders are active, prices move against the market maker after the trade. This is the most important component for execution timing.
- **Inventory cost** — the risk premium for holding an undesired position until it can be offloaded.
- **Order processing** — fixed operational costs (technology, clearing, etc.). The residual after the other two are estimated.

When the adverse selection share is elevated, you are likely trading in a toxic flow environment, informed participants are dominant, spreads are wide, and price impact will be high. When the adverse selection share is low and spreads are tight, the market is deep and liquid.

---

## Models implemented

### Roll (1984)
Estimates the effective spread from the negative serial covariance of transaction price changes. Requires no quote data since it works purely from trade prices.

```
S = 2 * sqrt(-Cov(ΔP_t, ΔP_{t-1}))
```

The intuition: in a market with a spread, consecutive trades that bounce between bid and ask create negative autocorrelation in price changes. The magnitude of that autocorrelation reveals the spread.

**Reference**: Roll, R. (1984). A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market. *Journal of Finance*, 39(4), 1127–1139.

---

### Huang & Stoll (1997)
Decomposes the spread into adverse selection (α), inventory (β), and order processing (1 - α - β) components using a GMM-style regression of price changes on current and lagged trade direction indicators.

```
ΔP_t = (α + β) * S/2 * d_t  -  β * S/2 * d_{t-1}  +  ε_t
```

Where `d_t ∈ {-1, +1}` is the trade initiator direction (buyer or seller). The coefficient on the current indicator recovers the total variable cost share; the coefficient on the lagged indicator recovers the inventory share. Adverse selection is the difference.

**Reference**: Huang, R. D., & Stoll, H. R. (1997). The Components of the Bid-Ask Spread: A General Approach. *Review of Financial Studies*, 10(4), 995–1034.

---

### Kyle (1985) — Lambda
Estimates price impact via OLS regression of price changes on signed order flow (volume × direction). The slope coefficient λ measures how much the price moves per unit of signed volume, which is a direct measure of market depth.

```
ΔP_t = λ * (Q_t * d_t) + ε_t
```

Higher λ means the market is thinner; each unit of aggressive flow moves the price more. Lower λ means deep, liquid conditions.

**Reference**: Kyle, A. S. (1985). Continuous Auctions and Insider Trading. *Econometrica*, 53(6), 1315–1335.

---

## Liquidity regime classifier

After accumulating enough history, the engine classifies each rolling window into one of three regimes based on z-scores of the adverse selection share and effective spread relative to recent history:

| Regime | Terminal color | Interpretation |
|---|---|---|
| Deep / stable | Green | Low adverse selection, tight spreads, high depth — good execution conditions |
| Thin / one-sided | Red | Elevated informed flow or one-sided pressure — widen your spread, reduce size |
| Fragmented | Yellow | Mixed signals, transitional state — proceed with caution |

---

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/yourusername/adverse-selection-engine
cd adverse-selection-engine
pip install -r requirements.txt
```

---

## Usage

```bash
# BTC/USDT live dashboard (default)
python main.py

# Different trading pair
python main.py --symbol ETHUSDT

# Smaller trade window (faster regime transitions, noisier estimates)
python main.py --symbol SOLUSDT --window 300

# Export estimates to CSV (written every 10th estimate)
python main.py --export

# Full options
python main.py --help
```

### CLI reference

| Flag | Default | Description |
|---|---|---|
| `--symbol` | `BTCUSDT` | Any Binance spot pair with a `@trade` stream |
| `--window` | `500` | Number of trades in the rolling estimation window |
| `--history` | `120` | Length of the regime history buffer |
| `--refresh` | `1.0` | Dashboard refresh interval in seconds |
| `--export` | off | Write decomposition estimates to CSV |
| `--export-dir` | `exports/` | Output directory for CSV files |

---

## Dashboard

The terminal dashboard updates every second and displays:

- **Price + trade count + window duration + uptime** — top header
- **Stacked bar** — visual breakdown of the spread into adverse selection (red), inventory (yellow), and order processing (blue)
- **Component table** — half-spread in dollars, basis points, percentage share, and z-score for each component
- **Kyle's λ + R²** — price impact coefficient and regression fit
- **Regime label** — current liquidity regime with color coding
- **Sparkline** — rolling history of the adverse selection share (α) over the last 60 estimates

Press `Ctrl+C` to exit cleanly.

---

## CSV export

When `--export` is passed, estimates are written to `exports/{SYMBOL}_{timestamp}.csv` every 10th update. Columns:

| Column | Description |
|---|---|
| `timestamp` | ISO 8601 timestamp |
| `mid_price` | Median price over the window |
| `effective_spread` | Roll (1984) spread estimate in dollars |
| `adverse_selection` | Adverse selection component in dollars |
| `inventory` | Inventory cost component in dollars |
| `order_processing` | Order processing component in dollars |
| `adverse_selection_share` | α ∈ [0, 1] |
| `inventory_share` | β ∈ [0, 1] |
| `order_processing_share` | 1 - α - β ∈ [0, 1] |
| `kyle_lambda` | Price impact coefficient |
| `r_squared` | OLS R² for Kyle regression |
| `regime` | Liquidity regime label |
| `n_trades` | Number of trades in window |

---

## Project structure

```
adverse-selection-engine/
├── main.py                      # Orchestrator, CLI argument parsing, CSV export
├── requirements.txt
├── data/
│   ├── __init__.py
│   └── stream.py                # Binance WebSocket trade stream ingester
├── models/
│   ├── __init__.py
│   ├── decomposition.py         # Roll, Huang-Stoll, Kyle estimators
│   └── rolling.py               # Rolling window manager + regime classifier
├── viz/
│   ├── __init__.py
│   └── dashboard.py             # Rich terminal dashboard
└── tests/
    ├── __init__.py
    └── test_decomposition.py    # Unit tests for all estimators
```

---

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```

15 tests covering Roll spread edge cases, Kyle lambda finite output, decomposition shares summing to 1, non-negativity, validity, directional correctness under informed flow, and rolling decomposer regime coverage.

---

## Extending

**Different exchange**: Replace `data/stream.py` with any WebSocket source that emits price, quantity, and aggressor side. The `Trade` dataclass is the only contract the rest of the system depends on.

**Gate execution on regime**: Import `RollingDecomposer` and check `estimate.regime == Regime.DEEP_STABLE` before routing an order. Combine with Kyle's λ for a two-factor execution quality filter.

**Use as a signal**: `adverse_selection_share` is a normalized [0, 1] flow toxicity proxy. Wire it into a position sizing multiplier — scale down when α is elevated, scale up when the market is deep and stable.

**Persist to a database**: Replace the CSV writer in `main.py` with any async DB insert. `SpreadComponents` is a plain dataclass and serializes trivially to dict or JSON.

---

## Dependencies

| Package | Purpose |
|---|---|
| `websockets` | Async Binance WebSocket connection |
| `numpy` | Array math for all estimators |
| `pandas` | Data handling utilities |
| `scipy` | Statistical utilities |
| `rich` | Terminal dashboard rendering |

---

## References

- Roll, R. (1984). A Simple Implicit Measure of the Effective Bid-Ask Spread in an Efficient Market. *Journal of Finance*, 39(4), 1127–1139.
- Huang, R. D., & Stoll, H. R. (1997). The Components of the Bid-Ask Spread: A General Approach. *Review of Financial Studies*, 10(4), 995–1034.
- Kyle, A. S. (1985). Continuous Auctions and Insider Trading. *Econometrica*, 53(6), 1315–1335.
- Glosten, L. R., & Milgrom, P. R. (1985). Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders. *Journal of Financial Economics*, 14(1), 71–100.
