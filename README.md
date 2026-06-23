```
  ██╗  ██╗██╗   ██╗██████╗  ██████╗ ██╗  ██╗ ██████╗
  ██║ ██╔╝██║   ██║██╔══██╗██╔═══██╗██║ ██╔╝██╔═══██╗
  █████╔╝ ██║   ██║██████╔╝██║   ██║█████╔╝ ██║   ██║
  ██╔═██╗ ██║   ██║██╔══██╗██║   ██║██╔═██╗ ██║   ██║
  ██║  ██╗╚██████╔╝██║  ██║╚██████╔╝██║  ██╗╚██████╔╝
  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              OPERATES IN THE SHADOWS
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

**Run the test suite** (pytest is configured via `pyproject.toml`):
```bash
pytest
```

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

> `table_storage_connection` is an Azure **Blob** Storage connection string (despite the variable name). Used exclusively by `AzureBlobHandler` for log shipping.

### config.json

`config.json` is committed to the repository and holds non-secret infrastructure configuration. Edit it before running in production.

| Group | Keys |
|-------|------|
| `logging` | `log_type`, `log_dir`, `log_file_name`, `retention_days`, `console_logging`, `azure_log_partition_key` |
| `trading` | `leverage`, `demo_starting_balance`, `initial_cash_balance`, `security_buffer` |

> **Before production deployment**, review:
> - `azure_log_partition_key` — change from `"DEV_NQ100"` to your production partition key
> - `log_type` — add `"azure_table"` to enable Azure Blob log shipping (default: `["file"]`)
> - `initial_cash_balance` and `leverage` — must match your actual account setup

> `epic` (the traded instrument) is NOT in `config.json` — it is set per strategy in `strategies/<StrategyName>.json`.

Strategy parameters are stored in the corresponding `strategies/<StrategyName>.json` file and committed to the repository. Edit the file before running. See the strategy documentation for the full parameter reference:

- [RSIBollingerStrategy](docs/strategies/RSIBollingerStrategy.md) — REST-polling, martingale grid, 15-min candles
- [RSIBollingerStrategyV2](docs/strategies/RSIBollingerStrategyV2.md) — Lightstreamer streaming, bidirectional grids, 5-min candles

---

## Project Structure

```
kuroko/
├── credentials.env              # Secrets — never committed
├── config.json                  # Logging and trading infrastructure config (leverage, balance, log settings)
├── requirements.txt             # Live trading dependencies
├── pyproject.toml               # pytest configuration
├── .pre-commit-config.yaml      # pre-commit hooks
│
├── kuroko.py                    # Entry point — loads strategy dynamically
├── ig_client.py                 # IG Markets REST API wrapper (auth, retry, caching)
├── ig_streaming_client.py       # Lightstreamer streaming client — delivers OHLC candles (and raw ticks in tick mode) via callbacks
├── logging_setup.py             # Loads config.json, sets up file/console/Azure log handlers
├── azure_log_handler.py         # Custom logging handler → Azure Blob Storage
│
├── strategies/                  # Python package — live strategy modules (.py) and config files (.json)
│   ├── __init__.py
│   ├── RSIBollingerStrategy.py      # V1: REST-polling, martingale grid, 15-min candles
│   ├── RSIBollingerStrategy.json
│   ├── RSIBollingerStrategyV2.py    # V2: Lightstreamer streaming, bidirectional grids, 5-min candles
│   └── RSIBollingerStrategyV2.json
│
├── docs/
│   ├── architecture.md          # Component breakdown, trading logic, config flow
│   ├── development.md           # Branching model, pre-commit, conventions, adding strategies
│   └── strategies/
│       ├── RSIBollingerStrategy.md   # V1 strategy parameters, entry/exit logic, protection mechanisms
│       └── RSIBollingerStrategyV2.md # V2 strategy parameters, entry/exit logic (streaming, bidirectional)
│
├── tests/                       # Test suite (pytest)
│
└── backtest/                    # Offline backtesting and optimization (isolated venv)
    ├── README.md                # Backtest setup, execution, and optimization instructions
    ├── requirements.txt         # Backtest-specific dependencies
    ├── backtest.py              # Backtest engine and runner
    ├── tuning.py                # Optuna-based parameter optimization
    ├── tuning_params.json       # Tuning engine config and Optuna search spaces
    ├── datasets/
    │   ├── es_intraday-15min.csv   # S&P 500 futures (15-min OHLC)
    │   └── nq_intraday-15min.csv  # NASDAQ 100 futures (15-min OHLC)
    └── strategies/
        ├── __init__.py
        ├── RSIBollingerStrategy.py # RSI + Bollinger Bands + ATR stop-loss + EMA trend filter (mean-reversion, mirrors live V1)
        └── RSIBollingerStrategy.json  # Backtest default parameters (date range, engine, strategy)
```

---

## Running the Live Bot

**Always activate the virtual environment first.**

```bash
source venv/bin/activate          # macOS/Linux
.\venv\Scripts\activate           # Windows

python kuroko.py --strategy <StrategyName> --log-level DEBUG
```

`--strategy` is required and must name the strategy class to run. `--log-level` accepts `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` (default: `INFO`).

```bash
# REST-polling strategy (V1)
python kuroko.py --strategy RSIBollingerStrategy

# Streaming strategy (V2 — Lightstreamer, bidirectional grids)
python kuroko.py --strategy RSIBollingerStrategyV2
```

Available strategies:

- **RSIBollingerStrategy** — REST-polling, martingale grid, 15-min candles. See [docs](docs/strategies/RSIBollingerStrategy.md).
- **RSIBollingerStrategyV2** — Lightstreamer streaming, bidirectional grids, 5-min candles. Supports two operating modes: **candle mode** (signals fire on each 5-min candle close) and **tick mode** (signals fire on every live bid/ask tick using cached indicators). Current default: tick mode (`operation_mode: "tick"` in the strategy JSON). See [RSIBollingerStrategyV2 docs](docs/strategies/RSIBollingerStrategyV2.md).

Stop the bot with `CTRL+C`.

> **Startup failure**: if the bot exits immediately with a `CRITICAL` log entry, first check the strategy module name (e.g. `strategies/RSIBollingerStrategy.py` must exist). Then verify that `strategies/<StrategyName>.json` contains valid JSON, has all required keys with the correct types (see the strategy docs for the full schema), and that `api_mode` is either `"rest"` or `"streaming"`. For V2, also verify `operation_mode` is `"candle"` or `"tick"` — it is required and its absence causes `SystemExit(1)`. The error log will list every missing key and type mismatch in one report. Startup failure is the only fatal failure — everything else is recovered automatically.

> **Runtime failures**: the bot does not crash on IG API errors. If the IG API is unavailable (maintenance window, timeout, empty response), the bot skips the affected cycle, logs a WARNING or ERROR, and retries on the next event: ~1 minute for V1 (REST poll interval); next candle close for V2 candle mode; next live tick for V2 tick mode. It recovers automatically when the API comes back. See [`docs/architecture.md`](docs/architecture.md#fault-tolerance-and-self-healing) for the full recovery model.

> **Candle cache**: at startup, `IGClient` caches candle data to `./cache/` as Parquet files to minimize API calls across restarts. If you change the traded `epic` or encounter stale data issues, delete the `cache/` directory before restarting.

---

## Backtesting & Optimization

The backtest module runs in its **own isolated environment** with its own dependencies. See [`backtest/README.md`](backtest/README.md) for setup, execution, and optimization instructions.

---

## Further Reading

- [`docs/architecture.md`](docs/architecture.md) — component breakdown, trading logic, configuration flow, backtest engine internals
- [`docs/development.md`](docs/development.md) — branching model, pre-commit hooks, conventions, adding strategies
- [`docs/strategies/RSIBollingerStrategy.md`](docs/strategies/RSIBollingerStrategy.md) — V1 strategy parameters and logic
- [`docs/strategies/RSIBollingerStrategyV2.md`](docs/strategies/RSIBollingerStrategyV2.md) — V2 strategy parameters and logic (streaming)
