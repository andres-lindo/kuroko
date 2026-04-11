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

Strategy parameters are stored in `strategy_parameters.json` at the project root and committed to the repository. Edit that file to change any live-trading parameter before running the bot.

---

## Project Structure

```
kuroko/
├── credentials.env              # Secrets — never committed
├── strategy_parameters.json         # Live trading strategy parameters
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
        └── rsi_bollinger.py        # RSI + Bollinger Bands (mean-reversion)
```

---

## Running the Live Bot

**Always activate the virtual environment first.**

```bash
source venv/bin/activate          # macOS/Linux
.\venv\Scripts\activate           # Windows

python ig_main.py [partition_key]
```

`partition_key` is the label used to identify this deployment's log blob in Azure Blob Storage. Defaults to `DEV_NQ100` if omitted.

```bash
python ig_main.py DEV_NQ100       # Development config for NASDAQ 100
python ig_main.py PROD_NQ100      # Production config for NASDAQ 100
```

Stop the bot with `CTRL+C`.

> **Startup failure**: if the bot exits immediately with a `CRITICAL` log entry, verify that `strategy_parameters.json` exists at the project root, contains valid JSON, has all 23 required keys with the correct types, and that `candle_frecuency` matches the pattern `\d+min` (e.g. `"15min"`). The error log will list every missing key and type mismatch in one report. Startup parameter load is the only fatal failure — everything else is recovered automatically.

> **Runtime failures**: the bot does not crash on IG API errors. If the IG API is unavailable (maintenance window, timeout, empty response), the bot skips the affected cycle, logs a WARNING or ERROR, and retries on the next tick (~1 minute). It recovers automatically when the API comes back. See [`docs/architecture.md`](docs/architecture.md#fault-tolerance-and-self-healing) for the full recovery model.

---

## Backtesting & Optimization

The backtest module runs in its **own isolated environment** with its own dependencies. See [`backtest/README.md`](backtest/README.md) for setup, execution, and optimization instructions.

---

## Further Reading

- [`docs/architecture.md`](docs/architecture.md) — component breakdown, trading logic, configuration flow, backtest engine internals
- [`docs/development.md`](docs/development.md) — branching model, pre-commit hooks, conventions, adding strategies
