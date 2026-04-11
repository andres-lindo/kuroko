# Kuroko

Algorithmic trading bot for financial futures (NASDAQ 100, S&P 500) via [IG Markets](https://www.ig.com/). Includes a live trading engine, a historical backtesting framework, and a parameter optimization pipeline.

---

## Prerequisites

- Python 3.11.x (Python 3.12+ is not supported)
- TA-Lib system library (required before `pip install`)

**macOS**
```bash
brew install ta-lib
```

**Linux (Debian/Ubuntu)**
```bash
sudo apt-get install libta-lib-dev
```

**Windows** — install the prebuilt wheel directly from the [TA-Lib releases](https://github.com/cgohlke/talib-build/releases). Pick the file matching your Python version (`cp311` = Python 3.11):
```powershell
pip install https://github.com/cgohlke/talib-build/releases/download/v0.6.8/ta_lib-0.6.8-cp311-cp311-win_amd64.whl
```

---

## Setup

All development and execution MUST be done inside the virtual environment.

**macOS / Linux**
```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Windows**
```powershell
C:\Python311\python.exe -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install https://github.com/cgohlke/talib-build/releases/download/v0.6.8/ta_lib-0.6.8-cp311-cp311-win_amd64.whl
pip install -r requirements.txt
```

> Deactivate the environment at any time with `deactivate`.

---

## Environment Configuration

Create a `credentials.env` file in the project root (never commit this file):

```env
ig_username=your_username
ig_password=your_password
ig_api_key=your_api_key
ig_acc_number=your_account_number
ig_acc_type=DEMO

table_storage_connection=DefaultEndpointsProtocol=https;AccountName=...
```

> `ig_acc_type` accepts `DEMO` or `LIVE`. This must match `is_live_account` in `ig_strategy.py` — see [`docs/architecture.md`](docs/architecture.md#risk-controls) before switching to a live account.

Strategy parameters are **not stored in code** — they live in Azure Table Storage (`ConfigParameters` table), keyed by `PartitionKey`. The `partition_key` argument selects which configuration set to load at runtime.

> **Before running the bot**, the Azure Table Storage must be provisioned with at least a `BASE_CONF` partition and one strategy-specific partition (e.g. `DEV_US500`). Without these rows the bot will start and immediately fail to load parameters. See [`docs/architecture.md`](docs/architecture.md#configuration-flow) for the expected table structure.

---

## Project Structure

```
kuroko/
├── credentials.env              # Secrets — never committed
├── requirements.txt             # Live trading dependencies
├── .pre-commit-config.yaml      # pre-commit hooks
│
├── ig_main.py                   # Entry point for the live trading bot
├── ig_client.py                 # IG Markets API wrapper (auth, retry, caching)
├── ig_strategy.py               # Trading logic, signal generation, risk management
├── azure_log_handler.py         # Custom logging handler → Azure Blob Storage
│
├── docs/
│   ├── architecture.md          # Component breakdown, trading logic, config flow
│   └── development.md           # Branching model, pre-commit, conventions, adding strategies
│
└── backtest/                    # Offline backtesting and optimization (isolated venv)
    ├── requirements.txt         # Backtest-specific dependencies
    ├── backtest.py              # Backtest engine and runner
    ├── tuning.py                # Optuna-based parameter optimization
    ├── datasets/
    │   ├── es_intraday-15min.csv   # S&P 500 futures (15-min OHLC)
    │   └── nq_intraday-15min.csv  # NASDAQ 100 futures (15-min OHLC)
    └── strategies/
        ├── __init__.py
        ├── rsi_bollinger.py        # RSI + Bollinger Bands (mean-reversion)
        └── ema_crossover.py        # EMA crossover (trend-following)
```

---

## Running the Live Bot

**Always activate the virtual environment first.**

```bash
source venv/bin/activate          # macOS/Linux
.\venv\Scripts\activate           # Windows

python ig_main.py [partition_key]
```

`partition_key` selects the configuration set from Azure Table Storage. Defaults to `DEV_US500` if omitted.

```bash
python ig_main.py DEV_US500       # Development config for S&P 500
python ig_main.py PROD_NQ100      # Production config for NASDAQ 100
```

Stop the bot with `CTRL+C`.

> **Troubleshooting**: if the bot crashes immediately at startup with an Azure error, verify that `table_storage_connection` in `credentials.env` is valid and that the `ConfigParameters` table contains both the `BASE_CONF` partition and your strategy partition.

---

## Backtesting & Optimization

The backtest module runs in its **own isolated environment** with its own dependencies. See [`backtest/README.md`](backtest/README.md) for setup, execution, and optimization instructions.

---

## Further Reading

- [`docs/architecture.md`](docs/architecture.md) — component breakdown, trading logic, configuration flow, backtest engine internals
- [`docs/development.md`](docs/development.md) — branching model, pre-commit hooks, conventions, adding strategies
