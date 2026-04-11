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
├── load_params(partition_key)
│   └── Azure Table Storage → ConfigParameters table
│       ├── PartitionKey == partition_key  (strategy-specific config)
│       └── PartitionKey == 'BASE_CONF'   (shared defaults)
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
└── Strategy(params, ig_client).run()
    └── main loop (15-min cadence, aligned to candle close)
```

### Components

#### `IGClient` (`ig_client.py`)

Wrapper around the `trading-ig` library. All API calls go through `_safe_api_call`, which provides:

- **Retry with backoff**: up to 3 attempts, waits 1 → 2 seconds between attempts (no wait before the final attempt)
- **Token refresh**: detects expired session errors and calls `create_session()` before retrying
- **Candle caching**: two-layer cache
  - In-memory (`dict`): avoids redundant API calls within the same process lifetime
  - Parquet files (`./cache/`): survives restarts; cache key is `{epic}_{resolution}`
- **Incomplete candle removal**: the current (open) candle is always stripped before returning data to the strategy

Key methods: `get_candles()`, `get_open_positions()`, `open_position()`, `close_position()`, `update_position()`.

#### `Strategy` (`ig_strategy.py`)

Contains all trading logic. Initialized with the config object from Azure and an `IGClient` instance.

**Indicators computed per cycle** (via TA-Lib on 15-min NASDAQ futures — epic `IX.D.NASDAQ.IFMM.IP`, hardcoded; the `cfd_symbol` config key is loaded but not used by `Strategy`):

| Indicator | Parameter source |
|---|---|
| RSI | `rsi_period` |
| EMA (trend filter) | `ema_period = 200` (fixed) |
| ATR | `atr_period` |
| Bollinger Bands | `bb_period`, `bb_dev` |

**Entry logic (RSI + Bollinger Bands, mean-reversion)**

- Buy when: `close < bb_lower` AND `rsi < rsi_oversold`
- Sell when: `close > bb_upper` AND `rsi > rsi_overbought`
- Minimum distance between entries: `min_dist_between_entries_ticks` (prevents stacking on fast moves)
- Optional trend filter: only buy when `close > ema_200`

**Position sizing (martingale grid)**

- Up to `max_positions` (default: 5) open simultaneously
- Each new position uses `position_size × martingale_multiplier^n` where `n` is the current open count
- Default multiplier: `1.5` (1×, 1.5×, 2.25×, 3.375×, 5.06×)

**Exit logic**

- **Basket take-profit**: closes all positions when the average entry price + `take_profit_ticks` is reached
- **Per-position limit order**: a broker-level TP is set at the time of `open_position` as a secondary safety net
- **Dynamic stop-loss**: per-position stop at `entry_price - (atr × atr_sl_multiplier)`

**Risk controls**

- **Max drawdown freeze**: if drawdown exceeds `max_drawdown_pct` (75.75%), no new entries are opened for the rest of the session
- **Margin check**: verifies sufficient free margin before opening any position
- **Virtual margin** (`is_live_account = False`): simulates 1:20 leverage against `initial_cash_balance = 4000` (the virtual capital base) regardless of the actual IG demo balance. `demo_starting_balance = 20000` is the IG demo account reference used only for realized P&L calculation. Set `is_live_account = True` only when switching to a real account — this changes margin and equity calculations to use raw broker figures instead of the virtual simulation

#### `AzureBlobHandler` (`azure_log_handler.py`)

Custom `logging.Handler` that ships all log records to Azure Blob Storage. Uses append-blob mode so multiple writes don't overwrite existing content. Rotates to a new blob daily at midnight UTC. Blob name format: `{partition_key}_{YYYY-MM-DD}.log`.

### Configuration Flow

Strategy parameters are **not in code** — they are loaded at startup from Azure Table Storage and injected into `Strategy.__init__`. This allows changing parameters without redeployment.

```
Azure Table Storage
└── ConfigParameters table
    ├── PartitionKey: BASE_CONF    → shared baseline values
    └── PartitionKey: DEV_US500   → environment/instrument overrides
        └── merged at runtime → Config object → Strategy
```

Both partitions are queried together and merged at runtime. If the same `RowKey` exists in both, the strategy-specific partition (`DEV_US500`) is intended to take precedence — ensure `RowKey` values are unique across partitions to avoid relying on undefined iteration order from the Azure SDK.

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
