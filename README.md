```
  ██╗  ██╗██╗   ██╗██████╗  ██████╗ ██╗  ██╗ ██████╗
  ██║ ██╔╝██║   ██║██╔══██╗██╔═══██╗██║ ██╔╝██╔═══██╗
  █████╔╝ ██║   ██║██████╔╝██║   ██║█████╔╝ ██║   ██║
  ██╔═██╗ ██║   ██║██╔══██╗██║   ██║██╔═██╗ ██║   ██║
  ██║  ██╗╚██████╔╝██║  ██║╚██████╔╝██║  ██╗╚██████╔╝
  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              OPERATE IN THE SHADOWS
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

# Kuroko

Algorithmic trading bot for financial futures (NASDAQ 100, S&P 500) via [IG Markets](https://www.ig.com/). Includes a live trading engine, a historical backtesting framework, and a parameter optimization pipeline.

---

## Prerequisites

- Python 3.11.x
- TA-Lib system library

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
pip install ta-lib
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

> `ig_acc_type` controls live vs demo mode (`DEMO` or `LIVE`). See [`docs/architecture.md`](docs/architecture.md#account-mode) before switching to a live account.

Strategy parameters are stored in `strategies/RSIBollingerStrategy.json` (live trading) and `backtest/strategies/RSIBollingerStrategy.json` (backtesting) — both committed to the repository. Edit the corresponding file before running. See [RSIBollingerStrategy documentation](docs/strategies/RSIBollingerStrategy.md) for the full parameter reference.

---

## Project Structure

```
kuroko/
├── credentials.env              # Secrets — never committed
├── requirements.txt             # Live trading dependencies
├── .pre-commit-config.yaml      # pre-commit hooks
│
├── ig_main.py                   # Entry point — loads strategy dynamically
├── ig_client.py                 # IG Markets API wrapper (auth, retry, caching)
├── azure_log_handler.py         # Custom logging handler → Azure Blob Storage
│
├── strategies/                  # Python package — live strategy modules (.py) and config files (.json)
│   ├── __init__.py
│   ├── RSIBollingerStrategy.py  # RSIBollingerStrategy: trading logic, signal generation, risk management
│   └── RSIBollingerStrategy.json
│
├── docs/
│   ├── architecture.md          # Component breakdown, trading logic, config flow
│   ├── development.md           # Branching model, pre-commit, conventions, adding strategies
│   └── strategies/
│       └── RSIBollingerStrategy.md  # Strategy parameters, entry/exit logic, protection mechanisms
│
└── backtest/                    # Offline backtesting and optimization (isolated venv)
    ├── requirements.txt         # Backtest-specific dependencies
    ├── backtest.py              # Backtest engine and runner
    ├── tuning.py                # Optuna-based parameter optimization
    ├── tuning_params.json       # Tuning engine config and Optuna search spaces
    ├── datasets/
    │   ├── es_intraday-15min.csv   # S&P 500 futures (15-min OHLC)
    │   └── nq_intraday-15min.csv  # NASDAQ 100 futures (15-min OHLC)
    └── strategies/
        ├── __init__.py
        ├── RSIBollingerStrategy.py # RSI + Bollinger Bands (mean-reversion)
        └── RSIBollingerStrategy.json  # Backtest default parameters (date range, engine, strategy)
```

---

## Running the Live Bot

**Always activate the virtual environment first.**

```bash
source venv/bin/activate          # macOS/Linux
.\venv\Scripts\activate           # Windows

python ig_main.py --strategy <StrategyName>
```

`--strategy` is required and must name the strategy class to run. The Azure Blob log partition key is read from `log_partition_key` in the strategy JSON config.

```bash
python ig_main.py --strategy RSIBollingerStrategy
```

See [RSIBollingerStrategy documentation](docs/strategies/RSIBollingerStrategy.md) for parameter reference.

Stop the bot with `CTRL+C`.

> **Startup failure**: if the bot exits immediately with a `CRITICAL` log entry, first check the strategy module name (e.g. `strategies/RSIBollingerStrategy.py` must exist). Then verify that `strategies/RSIBollingerStrategy.json` contains valid JSON, has all 24 required keys with the correct types, and that `candle_frecuency` matches the pattern `\d+min` (e.g. `"15min"`). The error log will list every missing key and type mismatch in one report. Startup failure is the only fatal failure — everything else is recovered automatically.

> **Runtime failures**: the bot does not crash on IG API errors. If the IG API is unavailable (maintenance window, timeout, empty response), the bot skips the affected cycle, logs a WARNING or ERROR, and retries on the next tick (~1 minute). It recovers automatically when the API comes back. See [`docs/architecture.md`](docs/architecture.md#fault-tolerance-and-self-healing) for the full recovery model.

---

## Backtesting & Optimization

The backtest module runs in its **own isolated environment** with its own dependencies. See [`backtest/README.md`](backtest/README.md) for setup, execution, and optimization instructions.

---

## Further Reading

- [`docs/architecture.md`](docs/architecture.md) — component breakdown, trading logic, configuration flow, backtest engine internals
- [`docs/development.md`](docs/development.md) — branching model, pre-commit hooks, conventions, adding strategies
