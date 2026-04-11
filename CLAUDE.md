# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Kuroko** is an algorithmic trading bot that trades financial futures (NASDAQ/S&P 500) via IG Markets. It has two concerns: **live trading** (executes against the IG Markets API) and **backtesting/optimization** (offline strategy validation and parameter tuning).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Note on TA-Lib**: On Windows, the wheel must be installed from a prebuilt GitHub release before `pip install -r requirements.txt`. On macOS/Linux, `brew install ta-lib` / `apt install libta-lib-dev` first.

Environment variables are loaded from `credentials.env` (not committed):
```
ig_username, ig_password, ig_api_key, ig_acc_number, ig_acc_type
table_storage_connection  # Azure Table Storage (strategy config)
```

## Running

**Live trading bot:**
```bash
python ig_main.py [partition_key]
# e.g. python ig_main.py DEV_US500
```

**Backtest a strategy:**
```bash
python backtest/backtest.py --strategy RSIBollingerStrategy
```

**Optuna parameter optimization:**
```bash
cd backtest
python tuning.py --strategy RSIBollingerStrategy \
  --start_date 2026-01-01 --end_date 2026-04-10 \
  --objective_type single --trials 100
# objective_type: single | multiple | weighted
```

There is no automated test suite, linting config, or CI pipeline.

## Architecture

### Live Trading (`ig_main.py` → `ig_client.py` + `ig_strategy.py`)

```
ig_main.py
  ├── load_params()         → Azure Table Storage (ConfigParameters table, keyed by PartitionKey)
  ├── Setup logging         → AzureBlobHandler (daily append-blob rotation)
  ├── IGClient              → IG Markets REST API session (handles auth + retry)
  └── Strategy.run()        → main loop
```

- **`IGClient`**: Thin wrapper around IG Markets API. Handles session auth, token refresh, exponential backoff on failures, and candle caching (in-memory + parquet).
- **`Strategy`**: All trading logic lives here. Manages positions, computes signals via TA-Lib, and enforces risk rules.
- **`AzureBlobHandler`**: Custom `logging.Handler` that ships logs to Azure Blob Storage.

### Strategy Logic (RSI + Bollinger Bands, mean-reversion)

- **Entry**: Buy when price < BB_lower AND RSI < oversold threshold
- **Sizing**: Martingale grid — each new position uses a 1.5× multiplier (up to 5 open positions)
- **Exit**: Basket take-profit (close all when avg entry + take_profit_ticks is reached)
- **Protection**: ATR-based dynamic stop-loss; max drawdown freeze (75% threshold); margin check before entry
- Incomplete (current) candle is always stripped before signal calculation

### Backtesting (`backtest/`)

Uses the `backtesting.py` framework. Strategy classes follow this pattern:
- **Class attributes** = tunable parameters (read by Optuna)
- `prepare_data(df, start_date, end_date)` = classmethod for preprocessing
- `init()` = indicator setup
- `next()` = per-bar logic

Optuna supports three objective modes:
- `single`: maximize equity
- `multiple`: Pareto front across 6 objectives
- `weighted`: equity − 0.3 × drawdown

Historical datasets live in `backtest/datasets/` (15-min OHLC CSVs for ES and NQ futures).

### Configuration

Strategy parameters for live trading are stored in **Azure Table Storage** (`ConfigParameters` table), not in code. `PartitionKey` selects the config set (e.g., `DEV_US500`, `BASE_CONF`). This means changing a live parameter requires updating Azure, not the repo.

### Key Design Decisions

- `is_live_account = False` is currently hardcoded — virtual margin calculation is always used (simulates 1:20 leverage strictly)
- Candle caching uses parquet files to minimize IG API calls across restarts
- Demo account balance is `initial_cash_balance = 4000` mapped to the 20,000 IG demo account via leverage
