# Backtest & Optimization

Standalone module for historical strategy validation and parameter optimization. Runs in its **own isolated virtual environment** — do not share the root venv with this module, as the dependency versions are intentionally different (e.g. `pandas 1.5.3`, `numpy 1.26.4`).

---

## Prerequisites

- Python 3.11
- TA-Lib system library

**macOS**
```bash
brew install ta-lib
```

**Linux (Debian/Ubuntu)**
```bash
sudo apt-get install libta-lib-dev
```

**Windows** — install the prebuilt wheel manually before the rest of the dependencies (see Setup below).

---

## Setup

Create and activate a dedicated virtual environment from inside this directory.

**macOS / Linux**
```bash
cd backtest
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Windows**
```powershell
cd backtest
C:\Python311\python.exe -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install https://github.com/cgohlke/talib-build/releases/download/v0.6.8/ta_lib-0.6.8-cp311-cp311-win_amd64.whl
pip install -r requirements.txt
```

> All commands below assume the backtest venv is active and you are inside the `backtest/` directory.

---

## Running a Backtest

```bash
python backtest.py --strategy RSIBollingerStrategy
```

Results are written to an HTML plot file in the current directory. Execution log is written to `{strategy}-backtest-last-execution.log`.

**Available strategies**

| Strategy | Style | Key Indicators |
|---|---|---|
| `RSIBollingerStrategy` | Mean-reversion | RSI, Bollinger Bands, ATR |

---

## Configuration

Backtest parameters are stored in `strategies/RSIBollingerStrategy.json` and committed to the repository. Edit this file to change any parameter before running the backtest.

The file is organized into four categories:

**Backtest Date Range**
| Key | Default | Description |
|---|---|---|
| `start_date` | `"2026-01-01"` | Inclusive simulation start date (YYYY-MM-DD) |
| `end_date` | `"2026-04-10"` | Inclusive simulation end date (YYYY-MM-DD) |
| `dataset` | `"nq_intraday-15min.csv"` | OHLC dataset file from `datasets/` |

**Engine Configuration**
| Key | Default | Description |
|---|---|---|
| `initial_cash_balance` | `400000` | Starting capital in USD |
| `leverage` | `20.0` | Simulated leverage (1:20 — mirrors IG retail account) |
| `commission` | `0.00012` | Round-trip commission per trade |
| `silent_mode` | `false` | `false` = emit detailed buy/sell logs to console |
| `objective_type` | `"single"` | Optimisation mode: `single` · `multiple` · `weighted` |

**Risk Controls**
| Key | Default | Description |
|---|---|---|
| `max_positions` | `5` | Maximum open grid levels |
| `min_dist_between_entries_ticks` | `100.0` | Minimum tick gap between consecutive entries |
| `martingale_multiplier` | `1.5` | Grid size multiplier (contracts scale as ×1, ×1.5, ×2.25 …) |
| `take_profit_ticks` | `240.0` | Profit target in ticks from basket average price |
| `atr_period` | `12` | ATR lookback period for dynamic stop-loss |
| `atr_sl_multiplier` | `11.0` | ATR multiplier for stop-loss distance |

**Indicator Settings**
| Key | Default | Description |
|---|---|---|
| `bb_dev` | `1.9` | Bollinger Band standard deviation width |
| `bb_period` | `20` | Bollinger Band and SMA lookback period |
| `rsi_period` | `11` | RSI oscillator lookback period |
| `rsi_overbought` | `76` | RSI level above which short entries are considered |
| `rsi_oversold` | `25` | RSI level below which long entries are considered |
| `use_trend_filter` | `false` | `false` = pure mean-reversion, ignores trend direction |

> **Note**: These are independent from the live trading parameters in `../strategies/RSIBollingerStrategy.json`. Tuning results from Optuna can be applied here to improve backtest fidelity, but they do not automatically propagate to the live config.

---

## Parameter Optimization

Uses [Optuna](https://optuna.org/) to search the parameter space defined in `tuning.py`.

```bash
python tuning.py \
  --strategy RSIBollingerStrategy \
  --start_date 2024-01-01 \
  --end_date 2024-12-31 \
  --objective_type multiple \
  --trials 100
```

**Arguments**

| Argument | Required | Description |
|---|---|---|
| `--strategy` | no | `RSIBollingerStrategy` (default) |
| `--start_date` | yes | Backtest start date (`YYYY-MM-DD`) |
| `--end_date` | yes | Backtest end date (`YYYY-MM-DD`) |
| `--objective_type` | yes | `single` · `multiple` · `weighted` (see below) |
| `--trials` | yes | Number of Optuna trials to run |

**Objective types**

| Type | Description |
|---|---|
| `single` | Maximize final equity. Fast, suitable for grid search. |
| `multiple` | Pareto front across 6 objectives: portfolio value (↑), max drawdown (↓), win rate (↑), avg loss (↓), Sortino ratio (↑), Sharpe ratio (↑). |
| `weighted` | Composite score: 0.7 × equity − 0.3 × max_drawdown. |

Outputs are written to `tuning_output/` — one JSON file with trial results and one log file per run, both timestamped as `tuning_{strategy}_{objective}_{YYYYMMDD_HHMM}`.

> **Note**: `tuning.py` hardcodes `nq_intraday-15min.csv` as the dataset regardless of the strategy selected. To change the dataset, update `FILE_CONFIG['dataset_name']` in `tuning.py` before running.

---

## Datasets

Historical OHLC data located in `datasets/`:

| File | Resolution | Instrument | Used by |
|---|---|---|---|
| `nq_intraday-15min.csv` | 15-min | NASDAQ 100 futures | `RSIBollingerStrategy` |
| `es_intraday-15min.csv` | 15-min | S&P 500 futures | — |
| `es_intraday-5min.csv` | 5-min | S&P 500 futures | — |

---

## Further Reading

- [`../docs/architecture.md`](../docs/architecture.md) — backtest engine internals and optimization pipeline design
- [`../docs/development.md`](../docs/development.md) — pre-commit setup, conventions, and detailed instructions for adding a strategy
