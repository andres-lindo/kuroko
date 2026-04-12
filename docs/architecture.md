# Architecture

## System Overview

Kuroko is split into two independent execution contexts that share no runtime state and are never run simultaneously:

| Context | Entry point | Environment |
|---|---|---|
| Live trading | `ig_main.py` | Root venv |
| Backtesting / Optimization | `backtest/backtest.py`, `backtest/tuning.py` | `backtest/` venv |

---

## Live Trading

### Startup Sequence

```
ig_main.py  ← load_dotenv("credentials.env") runs at module scope, before main()
├── load_strategy(args.strategy) → (RSIBollingerStrategy, load_params)
├── load_params("strategies/RSIBollingerStrategy.json") → types.SimpleNamespace
│   └── keys: candle_frecuency, epic, max_positions, rsi_period, bb_period, ...
│
├── AzureBlobHandler
│   └── Azure Blob Storage → container: logs
│       └── append-blob per partition_key, rotates at midnight UTC
│
├── IGClient()
│   ├── Reads env vars set by load_dotenv (username, password, api_key, acc_number)
│   ├── Creates IG Markets REST session
│   └── Creates ./cache/ directory for parquet persistence
│
└── RSIBollingerStrategy(params, ig_client).run()
    └── main loop (15-min cadence, aligned to candle close)
```

### Components

#### `IGClient` (`ig_client.py`)

Wrapper around the `trading-ig` library. All API calls go through `_safe_api_call`, which provides:

- **Retry with backoff**: up to 3 attempts, waits 1 → 2 → 4 seconds between attempts
- **Token refresh**: detects expired session errors and calls `create_session()` before retrying. Session refresh is NOT triggered for `json.JSONDecodeError` — empty-body responses are content errors, not auth errors
- **Maintenance window handling**: `json.JSONDecodeError` (IG returning an empty HTTP body) is included in the retry clause so transient outages are retried automatically
- **Candle caching**: two-layer cache
  - In-memory (`dict`): avoids redundant API calls within the same process lifetime
  - Parquet files (`./cache/`): survives restarts; cache key is `{epic}_{resolution}`
- **Incomplete candle removal**: the current (open) candle is always stripped before returning data to the strategy

Key methods: `get_candles()`, `get_open_positions()`, `open_position()`, `close_position()`, `update_position()`.

#### `RSIBollingerStrategy` (`strategies/rsi_bollinger.py`)

Contains all trading logic. Initialized with the config object from `strategies/RSIBollingerStrategy.json` and an `IGClient` instance. See [RSIBollingerStrategy documentation](strategies/RSIBollingerStrategy.md) for entry logic, position sizing, exit logic, risk controls, and parameter reference.

#### `AzureBlobHandler` (`azure_log_handler.py`)

Custom `logging.Handler` that ships all log records to Azure Blob Storage. Uses append-blob mode so multiple writes don't overwrite existing content. Rotates to a new blob daily at midnight UTC. Blob name format: `{partition_key}_{YYYY-MM-DD}.log`.

### Fault Tolerance and Self-Healing

The live trading engine is designed to survive transient IG API failures (maintenance windows, timeouts, empty responses) without operator intervention. Recovery is layered — each layer handles failures at its own level and passes only unrecoverable conditions upward.

#### Recovery layers (inner → outer)

**Layer 1 — API call (`_safe_api_call`)**

Every API call is retried up to 3 times with exponential backoff (1s, 2s, 4s). Handles: `ConnectionError`, `RequestException`, `IGException`, `json.JSONDecodeError`. Token-expired errors also trigger a session refresh before retrying. If all 3 attempts fail, the exception propagates to the caller.

**Layer 2 — Candle fetch (`Strategy.get_candles`)**

If `IGClient.get_candles()` raises after exhausting retries, `Strategy.get_candles()` catches the exception, logs an ERROR, and returns the last successfully fetched DataFrame from its internal cache. The strategy operates on stale data for that tick rather than crashing. If the cache is empty (first startup, first call ever failed), an empty DataFrame is returned and the cycle is skipped.

**Layer 3 — Account data guard (`manage_positions`, `log_account_status`)**

Both methods call `get_account_summary()` at the start of each cycle. If the API is down, `get_account_summary()` returns an empty dict `{}`. Both methods check `if not account_info` immediately and return with a WARNING log. The strategy does not trade on missing data — it skips the cycle entirely. All remaining key accesses use `.get(key, default)` so no `KeyError` can occur.

**Layer 4 — Per-cycle recovery (`Strategy.run` loop)**

The entire tick body — `get_candles()` + `manage_positions()` + `log_account_status()` — is wrapped in a `try/except Exception` block inside the `while True` loop. Any exception that reaches this layer is logged as ERROR with a full traceback, and the loop continues to the next tick. `KeyboardInterrupt` is explicitly re-raised before the generic handler to preserve clean Ctrl+C shutdown.

```
while True:
    try:
        get_candles() → manage_positions() → log_account_status()
    except KeyboardInterrupt:
        raise                       # clean shutdown
    except Exception:
        log ERROR with traceback    # cycle failed, do NOT exit
    next_tick += 1 min              # always runs — correct clock advance
    sleep until next_tick
```

**Layer 5 — Position close retry (`close_all_positions`)**

Each individual `close_position()` call is wrapped in its own retry loop: 3 attempts with 1s/2s backoff. A failure on one position does not block the remaining closes. Failed deal IDs are accumulated and reported in a single WARNING after all positions are processed.

**Layer 6 — Startup config (`strategies/RSIBollingerStrategy.json` in `rsi_bollinger.py`)**

A failure here is fatal by design — the bot cannot trade without its configuration. `load_params()` handles three failure modes, each logging CRITICAL and calling `sys.exit(1)`:

1. **File errors** — missing file (`FileNotFoundError`), invalid JSON (`JSONDecodeError`), or unreadable file (`OSError`).
2. **Schema errors** — `_validate_params()` checks that all 22 required keys are present and correctly typed (e.g. `int` fields reject `bool`, `float` fields accept `int`). All errors are collected and reported at once before exiting.
3. **Format error** — `candle_frecuency` must match the regex `^\d+min$` (e.g. `"15min"`).

No retry is attempted; the process manager (systemd, supervisor, etc.) handles restart scheduling.

#### Behaviour during an IG maintenance window

```
IG API unavailable
│
├── Tick N:   _safe_api_call retries 3× in ~3s → raises
│             get_candles()  → returns stale cache (or empty DataFrame)
│             manage_positions() → account_info is {} → WARNING → return
│             run() except → logs "Trading cycle failed — will retry next tick"
│             sleeps ~1 min
│
├── Tick N+1: same — retries, skips, sleeps
│   ...
│
└── Tick N+K: IG API recovers → _safe_api_call succeeds → normal cycle resumes
```

The process **never exits** on API failures. It retries every tick indefinitely until the API recovers, then resumes normal operation automatically. The only observable impact is WARNING/ERROR log entries and skipped trading cycles during the outage.

#### What this does NOT cover

| Scenario | Behaviour |
|---|---|
| Extended outage (hours) | Bot keeps retrying every tick; produces one log entry per cycle; no trades during the outage |
| `get_open_positions()` returning `[]` on error | Strategy sees zero open positions for that cycle; state reconciles on the next successful call |
| Stale candle cache | Signals computed on data up to N minutes old; negligible on a 15-min candle strategy |
| No circuit breaker | No threshold for consecutive failures — the bot retries indefinitely; process manager handles restarts if needed |

### Configuration Flow

Strategy parameters are stored in `strategies/RSIBollingerStrategy.json` and loaded at startup via `load_params()`. Changing a parameter requires editing the file and redeploying the bot.

```
strategies/RSIBollingerStrategy.json
└── json.load() → dict
    └── types.SimpleNamespace(**data) → params
        └── RSIBollingerStrategy(params=params, ig_client=ig)
            └── self.params.candle_frecuency / .epic / .max_positions / ...
```

See [RSIBollingerStrategy documentation](strategies/RSIBollingerStrategy.md) for the full parameter reference.

`table_storage_connection` is NOT in this file — it is read from the environment (`credentials.env`) exclusively for `AzureBlobHandler` log shipping.

### Account Mode

`is_live_account` is derived from the `ig_acc_type` environment variable in `credentials.env`.

| `ig_acc_type` | `is_live_account` | Behaviour |
|---------------|-------------------|-----------|
| `LIVE` | `True` | Real-money trades via IG Markets |
| `DEMO` | `False` | Virtual mode — simulates 1:20 leverage, no real money |
| missing / other | `False` | Treated as demo |

> **Warning**: `ig_acc_type=LIVE` places real-money orders. The match is case-sensitive — `live` or `Live` will NOT activate live mode.

For how this flag affects margin and equity calculations inside the strategy, see [RSIBollingerStrategy — Account Mode](strategies/RSIBollingerStrategy.md#account-mode).

---

## Backtesting

See [`backtest/README.md`](../backtest/README.md) for setup and execution. The architecture below describes how the backtest engine works internally.

### Backtest Engine (`backtest/backtest.py`)

Built on top of `backtesting.py` 0.3.3. The engine loads a CSV dataset, instantiates the strategy class, and runs a bar-by-bar simulation.

```
load_raw_data(csv)
    └── pandas DataFrame (no date parsing yet — delegated to strategy)

load_strategy_class(name)
    └── STRATEGY_REGISTRY → dynamic import → class reference

run(strategy_class, data, params)
    └── backtesting.Backtest(data, strategy_class, **engine_config).run()
```

Strategy classes follow the `backtesting.py` pattern:
- **Class attributes** = tunable parameters (Optuna reads these directly)
- `prepare_data(df, start_date, end_date)` = classmethod for date filtering and preprocessing
- `init()` = indicator setup (called once)
- `next()` = per-bar logic (called on every candle)

### Optimization Pipeline (`backtest/tuning.py`)

```
STRATEGY_SEARCH_SPACES[strategy_name]
    └── declarative param ranges → get_trial_params(trial)
        └── Optuna suggest_int / suggest_float / suggest_categorical

Optuna Study
├── single   → maximize final equity
├── multiple → Pareto front across 6 objectives:
│              portfolio value (↑), max drawdown (↓), win rate (↑),
│              avg loss (↓), Sortino ratio (↑), Sharpe ratio (↑)
└── weighted → 0.7 × equity − 0.3 × max_drawdown

Output → tuning_output/{strategy}_{objective}_{timestamp}.json
         tuning_output/{strategy}_{objective}_log_{timestamp}.txt
```

---

## Contributing

For branching model, pre-commit setup, coding conventions, and how to add a strategy, see [`docs/development.md`](development.md).
