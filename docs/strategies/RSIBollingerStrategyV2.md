# RSIBollingerStrategyV2

Event-driven bidirectional mean-reversion strategy for IG Markets live trading.
Trades NASDAQ 100 (or S&P 500) futures on 5-minute candles delivered via
Lightstreamer streaming, using RSI and Bollinger Bands signals with independent
long and short position grids.

Module: `strategies/RSIBollingerStrategyV2.py` | Logger name: `strategies.RSIBollingerStrategyV2`

---

## Overview

V2 is a ground-up redesign of the mean-reversion approach with three key
differences from V1:

1. **Streaming data source** — candles arrive via Lightstreamer push (native
   5-minute OHLC subscription), not REST polling. No per-cycle API calls for
   price data.
2. **Bidirectional grids** — long and short grids are independent. Both can
   hold open positions at the same time.
3. **Simplified risk model** — no martingale sizing, no drawdown freeze.
   Broker-level take-profit and stop-loss are set at open; exits are
   handled exclusively by broker TP/SL orders set at position open.

V2 also supports an optional **tick mode** (`operation_mode: "tick"`) where
entry signals are evaluated on every live tick using indicators cached
from the most recently closed candle.

---

## Operation Modes

`operation_mode` controls whether signals are evaluated on candle close or on
every live tick.

### Candle mode (`operation_mode: "candle"`)

Default behavior. `_on_candle` computes indicators from each closed candle and
immediately evaluates entry signals via `_manage_longs()` and
`_manage_shorts()`. This is identical to the original V2 behavior.

### Tick mode (`operation_mode: "tick"`)

Indicators are computed and cached (`_cached_indicators`) on every candle close.
Entry signals are evaluated on each live tick by `_on_tick()`, using
the cached indicators together with the live `bid` and `ofr` prices from the tick.

**Warmup gate**: `_on_tick` silently discards ticks until `_cached_indicators` is set — which happens on the first streaming candle that produces valid indicators (requires `max(bb_period, rsi_period, atr_period) + 1` entries in the candle window). After a successful warmup this is the first streaming candle. After a cold-start (warmup failure), this requires `max(bb_period, rsi_period, atr_period)` additional streaming candles. No WARNING is logged during this period.

**In-flight guards**: Two flags prevent duplicate REST calls in tick mode — `_tick_long_in_flight` and `_tick_short_in_flight`, managed inside `_tick_try_open`. Each flag is set immediately before the REST call and reset in a `finally` block so it is always `False` after the handler returns, even if the call raises.

**Spread and `entry_spread`**: LONG positions opened in tick mode store
`entry_spread` — the bid/ask spread at the moment the position was opened.
This preserves the actual ask cost at entry time for internal reference.
Exits are handled exclusively by broker TP/SL orders set at position open.

**Native subscription fallback**: If the native candle subscription fails (e.g., account tier restriction), `IGStreamingClient` falls back to tick aggregation. In that case the direct tick subscription (`CHART:{epic}:TICK`) is not opened, and tick-mode entry signals are disabled for the session. A WARNING is logged. The strategy continues receiving synthetic candles from the tick aggregator and operates as if in candle mode.

**Switching modes**: Set `"operation_mode": "tick"` in the strategy JSON and
redeploy. To revert, set it back to `"candle"`.

**Invalid value handling**: If `operation_mode` is set to an unrecognised value
(e.g., `"turbo"`), the strategy logs a WARNING and falls back to `"candle"` mode.
No crash, no schema break.

### Data flow in tick mode

```
Lightstreamer
  ├── CHART:{epic}:{res} → _CandleSubscriptionListener → queue (type=candle)
  │                                                              ↓
  │                                                     _on_candle → cache indicators
  │
  └── CHART:{epic}:TICK  → _DirectTickListener         → queue (type=tick)
                                                               ↓
                                                       _on_tick → warmup gate
                                                               → in-flight guard
                                                               → entry signals
```

---

## Startup Lifecycle and Warm-Up

`run()` executes the following sequence on startup:

1. **`_warmup()`** — fetches `max(bb_period, rsi_period, atr_period) + 1` historical
   candles from the REST API via `IGClient.get_candles()` and appends each row's
   `Close`, `High`, and `Low` prices directly to `_candle_window`, `_high_window`,
   and `_low_window`. This pre-fills the windows so that all indicators (BB, RSI,
   ATR) are valid on the very first live streaming candle.
2. **`_seed_positions_from_broker()`** — fetches all open positions from the broker
   via `IGClient.get_open_positions()`, filters by `self.epic`, and populates
   `_long_positions` / `_short_positions` so that the strategy correctly tracks
   any positions that were open when the bot restarted.
3. **`streaming_client.start(_on_candle, on_tick=...)`** — opens the Lightstreamer
   connection and begins delivering live candles. In tick mode, also opens a second
   subscription to `CHART:{epic}:TICK` via `_DirectTickListener`.
4. **`_stop_event.wait()`** — blocks until `stop()` is called.

### Warm-up details

`_warmup()` calls `ig.clear_cache()` before anything else, discarding both the
in-memory and parquet caches so every restart triggers a full historical fetch
rather than the incremental 3-candle update path. It then calls
`IGClient.get_candles(epic, candle_frequency, num_candles)`,
where `num_candles = max(bb_period, rsi_period, atr_period) + 1`. The returned
DataFrame has capitalized OHLC columns (`Open`, `High`, `Low`, `Close`) and a
`DatetimeIndex`.

Each row's `Close`, `High`, and `Low` prices are appended directly to
`_candle_window`, `_high_window`, and `_low_window` as floats. No conversion
to a candle dict is performed and `_on_candle()` is not called during warm-up,
so trading logic cannot fire on REST data regardless of window fill level.

### `_last_warmup_ts` — deduplication guard

After warm-up, `_last_warmup_ts` holds the `datetime` of the last REST candle
processed. `_on_candle()` silently discards any incoming streaming candle whose
`timestamp <= _last_warmup_ts`, preventing double-counting at a candle boundary.

`_last_warmup_ts` is stored as a **UTC-aware** `datetime`. The IG REST API
returns `snapshotTime` in London local time (naive). `_warmup()` localizes the
last candle's naive timestamp to `Europe/London` and converts it to UTC before
storing, so the value is timezone-correct regardless of the machine's local
timezone. At comparison time in `_on_candle()`, both the candle timestamp and
`_last_warmup_ts` are normalized to naive UTC before the `<=` check.

`_last_warmup_ts` defaults to `None`. When `None`, the guard is a no-op and all
streaming candles are processed normally (cold-start behavior).

### Graceful degradation

If `get_candles()` returns `None` or raises an exception, `_warmup()` logs a
WARNING and returns without filling the window. The strategy proceeds to streaming
with an empty candle window — identical to the previous cold-start behavior. There
is no abort, no retry (the underlying `get_candles` already retries internally).

### Position reconciliation at restart

`_seed_positions_from_broker()` runs immediately after `_warmup()` and before
streaming starts. It calls `IGClient.get_open_positions()`, filters the results
by `self.epic`, sorts them by `createdDate` ascending (matching the runtime
invariant relied on by the minimum-distance guard), and appends each position
dict — `deal_id`, `entry_price`, `size` — to the appropriate grid.

**Direction mapping**: `BUY` → `_long_positions`; `SELL` → `_short_positions`.
Positions with an unexpected direction value are skipped with a WARNING.

**No `entry_spread` on seeded positions**: positions restored from broker state
have no recorded spread.

**Capacity guard**: if the seeded count exceeds `max_long_positions` or
`max_short_positions`, a WARNING is logged recommending operator review. The
excess positions are still seeded — no positions are silently dropped.

**Graceful degradation**: any exception from `get_open_positions()` or from
parsing a single record is caught. The method logs a WARNING and continues —
grids remain empty (or partially seeded) rather than aborting startup.

---

## Data Source

Candles come from `IGStreamingClient`, which subscribes to
`CHART:{epic}:{resolution}` on the IG Markets Lightstreamer adapter. The
resolution is derived from `candle_frequency` in the strategy JSON (e.g.
`"5min"` maps to `"5MINUTE"`). When a completed candle arrives (`CONS_END=1`),
the streaming client enqueues it and delivers it to the strategy from a
dedicated worker thread.

If the native subscription fails (e.g., account tier restriction), the client
falls back automatically to `CHART:{epic}:TICK` and aggregates ticks into
candles of the configured resolution in-process using `TickAggregator`. The
strategy receives the same candle dict regardless of which path produced it.

**Price basis**: `open`, `high`, `low`, and `close` in the candle dict are
derived from **BID prices** (`BID_OPEN`, `BID_HIGH`, `BID_LOW`, `BID_CLOSE`)
from the Lightstreamer adapter — not mid-market prices. `bid_close` and
`ofr_close` are also included for spread calculation. Signal logic operates
on bid prices throughout.

The strategy discards candle data until the rolling window holds at least
`max(bb_period, rsi_period, atr_period) + 1` completed candles, ensuring
indicators are computed from a full dataset.

---

## Entry Logic

### Long entry

A long position is opened when ALL of the following are true on the most
recently closed candle:

- `close < bb_lower` — price has closed below the lower Bollinger Band
- `rsi < rsi_oversold` — RSI is in oversold territory
- Current number of open long positions < `max_long_positions`
- Distance from the last long entry price >= `min_dist_between_entries_ticks`
  (or no long positions are open)

**In tick mode**: entry uses the live `bid` price instead of candle `close`.
The same BB and RSI thresholds apply. `entry_price` is recorded as `bid` at
the moment the tick fires (not the candle close). The distance check compares
`bid` against `_long_positions[-1]["entry_price"]`.

### Short entry

A short position is opened when ALL of the following are true:

- `close > bb_upper` — price has closed above the upper Bollinger Band
- `rsi > rsi_overbought` — RSI is in overbought territory
- Current number of open short positions < `max_short_positions`
- Distance from the last short entry price >= `min_dist_between_entries_ticks`
  (or no short positions are open)

**In tick mode**: entry uses the live `bid` price instead of candle `close`.
The same BB and RSI thresholds apply. `entry_price` is recorded as `bid` at
the moment the tick fires (not the candle close). The distance check compares
`bid` against `_short_positions[-1]["entry_price"]`.

The long and short grids are evaluated independently on every candle. An
active short signal does not suppress long entry evaluation and vice versa.

---

## Exit Logic

Positions are closed exclusively by the broker's take-profit (TP) and stop-loss (SL) orders set at open. The strategy does not close positions itself — BB-cross exits have been removed.

TP/SL distances are resolved by `_get_close_params()` at position open:

- **Fixed mode** (`close_mode: "fixed"`): uses `take_profit_ticks` and `stop_loss_ticks` directly.
- **Dynamic mode** (`close_mode: "dynamic"`): derives TP and SL distances independently from ATR — `limit_dist = round(atr_multiplier_tp * ATR[-1])`, `stop_dist = round(atr_multiplier_sl * ATR[-1])`. See Dynamic Close Mode below.

### Spread

Spread is updated dynamically — `_on_candle` sets `_current_spread` from each candle's `OFR_CLOSE - BID_CLOSE` field, and in tick mode the live spread is available as `ofr - bid`. These values are stored and logged in debug output but are not used for broker-level exit decisions (exits are handled exclusively by broker TP/SL orders).

---

## Position Management

| Behaviour | Detail |
|-----------|--------|
| Position grids | Long and short grids are independent; both may hold positions simultaneously |
| Startup reconciliation | `_seed_positions_from_broker()` pre-populates both grids from broker state before streaming starts — open positions survive restarts |
| Runtime reconciliation | `_reconcile_positions()` runs unconditionally on every candle close. It compares local grids against broker state, removes positions closed by broker TP/SL (no longer present at the broker), and seeds any broker positions absent from local grids (bidirectional sync). |
| Max positions per grid | `max_long_positions` for longs; `max_short_positions` for shorts |
| Minimum entry distance | New entry rejected if `abs(close - last_entry) < min_dist_between_entries_ticks` |
| Position sizing | Flat `contract_size` for every entry — no martingale, no scaling |
| Broker take-profit | Distance set by `_get_close_params()`: `take_profit_ticks` in fixed mode, `round(atr_multiplier_tp * ATR)` in dynamic mode |
| Broker stop-loss | Distance set by `_get_close_params()`: `stop_loss_ticks` in fixed mode, `round(atr_multiplier_sl * ATR)` in dynamic mode |

---

## Risk Management

Risk is bounded by:

- **Position caps** (`max_long_positions`, `max_short_positions`) — hard upper
  limit on exposure in each direction.
- **Broker-level take-profit** — set as a limit order at open; the broker
  closes the position automatically if the target is reached.
- **Broker-level stop-loss** — set at open via `stop_loss_ticks` (fixed mode)
  or ATR-derived distance (dynamic mode); the broker closes the position
  automatically if the loss threshold is hit.
- **Entry distance guard** — prevents adding to a losing grid faster than
  `min_dist_between_entries_ticks` ticks.

There is no drawdown freeze and no margin check in V2.

---

## Dynamic Close Mode

`close_mode` controls how TP/SL distances are resolved at position open. It is set in the strategy JSON and can be changed at runtime via hot-reload.

### Fixed mode (`close_mode: "fixed"`)

TP and SL distances are taken directly from `take_profit_ticks` and
`stop_loss_ticks`. Behavior is identical to the original V2 implementation —
no change to existing setups.

### Dynamic mode (`close_mode: "dynamic"`, default)

Both TP and SL distances are derived from the current ATR value:

```
limit_dist = round(atr_multiplier_tp * ATR[-1])
stop_dist  = round(atr_multiplier_sl * ATR[-1])
open_position(limit=limit_dist, stop=stop_dist)
```

ATR is computed on every candle (regardless of `close_mode`) using:
- `ta.ATR(highs, lows, closes, timeperiod=atr_period)` from TA-Lib
- High/low data comes from `_high_window` / `_low_window` — parallel deques
  maintained alongside `_candle_window`

**In tick mode**, `_tick_try_open` reads ATR from `_cached_indicators` (set by
the last `_on_candle` call). No extra REST call is made.

### Guard behavior

| Condition | Behavior |
|-----------|----------|
| `close_mode = "dynamic"` and ATR > 0 | `limit_dist = round(atr_multiplier_tp * ATR)`, `stop_dist = round(atr_multiplier_sl * ATR)` |
| `close_mode = "dynamic"` and ATR ≤ 0 or NaN | Log ERROR `"ATR unavailable in dynamic mode"` — **skip open** (`open_position` not called) |
| `close_mode = "dynamic"` and `dist < 5` | Log WARNING `"ATR-derived distance N is below min spread guard of 5 points"` — **still opens** the position |
| `close_mode = "fixed"` | Always uses `take_profit_ticks` / `stop_loss_ticks`; ATR is computed but not used |

### Hot-reload behavior

`close_mode`, `atr_multiplier_tp`, and `atr_multiplier_sl` are **hot-safe** — they can be changed at
runtime without restarting the bot. The new values take effect on the next
closed candle.

`atr_period` is **restart-required** — changing it requires a bot restart
because the ATR indicator window must be re-computed from scratch.

### Switching from fixed to dynamic

1. Edit `strategies/RSIBollingerStrategyV2.json`: set `"close_mode": "dynamic"`.
2. Optionally tune `atr_multiplier_tp` (default `1.0`) and `atr_multiplier_sl` (default `1.5`).
3. Save the file — the change takes effect on the next candle close (no restart needed).

To revert: set `"close_mode"` back to `"fixed"`.

---

---

## Guardrails

Guardrails are time-based restrictions that block specific actions regardless
of signal state. They are checked before entries and never block exits.

### Friday 14:00 NY — long entry block

`_is_long_entry_allowed(ts=None) -> bool` returns `False` on any Friday at or
after 14:00 New York time, preventing new long positions from being opened over
the weekend. The rationale: positions left open Friday afternoon roll into
Monday and accrue intraday fees with no active session to manage them.

**What is blocked**: long entries only — in both candle mode (`_manage_longs`)
and tick mode (`_on_tick`).

**What is NOT blocked**: any short entry. The guard is
injected at the top of the entry block in each path.

**DST handling**: the check converts the evaluation timestamp to
`America/New_York` using `ZoneInfo("America/New_York")`. DST transitions are
handled automatically — no hardcoded UTC offset.

**Log message** (when blocked): `[GUARD] Long entry skipped — Friday after 14:00 NY`
— logged at INFO level in candle mode, DEBUG in tick mode.

**`ts` parameter**: when called with `ts=None` (runtime), wall-clock time is
used. Passing an explicit `datetime` (e.g. in tests) converts it to NY timezone
before the day/hour check.

---

## Account Status Logging

`log_account_status()` is called on every candle close — after trade decisions in candle mode, after caching indicators in tick mode. It is **not** called from `_on_tick`. It makes two broker REST calls per candle (`get_account_summary` and `get_open_positions`) and emits one `STATUS |` INFO log line.

### Log format

```
STATUS | Mode: {mode} | Equity: ${equity} | Used Margin: ${used_margin} | Margin Level: {level} {health} | Free: ${free_margin} | Longs: {n} (avg: {price}) | Shorts: {n} (avg: {price}) | Total: {n}
```

`mode` is `LIVE` or `VIRTUAL (1:{leverage})` depending on `ig_acc_type`.

### Equity and margin calculation

**LIVE mode** (`ig_acc_type=LIVE`):

| Field | Source |
|-------|--------|
| `current_equity` | `balance + profitLoss` from `get_account_summary()` |
| `used_margin` | `sum(size × level / leverage)` over broker positions filtered to `self.epic` |
| `free_margin` | `current_equity − used_margin` |

**DEMO/VIRTUAL mode**:

| Field | Source |
|-------|--------|
| `current_equity` | `initial_cash_balance + (balance − demo_starting_balance) + profitLoss` |
| `used_margin` | `sum(size × level / leverage)` over broker positions filtered to `self.epic` |
| `free_margin` | `current_equity − used_margin` |

Only positions matching `self.epic` are included in the margin calculation.

### Health labels

| Label | Condition |
|-------|-----------|
| `[IDLE]` | No open positions (`used_margin == 0`) |
| `[DANGER]` | Margin level < 120% |
| `[ALERT]` | Margin level 120–199% |
| `[HEALTHY]` | Margin level ≥ 200% |

### Average entry prices

`Longs` and `Shorts` avg entry prices are computed from the **local position grids** (`_long_positions`, `_short_positions`) using a size-weighted mean of `entry_price`. Returns `N/A` when the grid is empty.

### Error handling

Any exception from either broker call is caught by a broad `try/except`. The method logs an ERROR and returns without emitting the STATUS line — it never propagates to the caller.

---

## Threading Model

The strategy runs on two threads:

```
┌────────────────────────────────┐      ┌──────────────────────────────────────────┐
│  Lightstreamer (LS) thread     │      │  IGStreamingClient worker thread          │
│                                │      │                                          │
│  _CandleSubscriptionListener   ──────► queue.put({type:"candle",...})            │
│  _DirectTickListener (tick mode)──────► queue.put({type:"tick",...})             │
│                                │      │  ↓                                       │
│                                │      │  queue.get() → dispatch by type          │
└────────────────────────────────┘      │  type=candle → _on_candle()              │
                                        │    → cache indicators (tick mode)        │
                                        │    → _manage_longs/_manage_shorts (candle)│
                                        │  type=tick  → _on_tick()                 │
                                        │    → warmup gate / in-flight guard       │
                                        │    → entry via ig_client                 │
                                        └──────────────────────────────────────────┘
```

All data flows through the **single shared queue**. The worker thread dispatches
by `item.get("type", "candle")` — items without a `"type"` key default to candle
for backward compatibility. There is no concurrent REST call risk because `_on_candle`
and `_on_tick` run sequentially on the same worker thread.

`run()` blocks on `_stop_event.wait()` while the streaming client's worker thread
handles all delivery and trading logic.

---

## Parameters Reference

Strategy parameters are stored in `strategies/RSIBollingerStrategyV2.json` and
loaded at startup into a `types.SimpleNamespace` via `load_params()`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `api_mode` | string | `"streaming"` | Must be `"streaming"` — tells `kuroko.py` to instantiate `IGStreamingClient` |
| `operation_mode` | string | `"candle"` | Signal evaluation mode. `"candle"`: signals fire on each closed candle (default). `"tick"`: indicators cached on candle close; signals fire on each live tick. Any other value logs a WARNING and falls back to `"candle"`. Required field — removing it causes startup to fail with a CRITICAL log and `SystemExit(1)`. |
| `candle_frequency` | string | `"5min"` | Candle resolution in `"Nmin"` format (e.g. `"1min"`, `"5min"`, `"15min"`, `"60min"`). Mapped to IG Lightstreamer resolution strings (`"1MINUTE"`, `"5MINUTE"`, `"15MINUTE"`, `"1HOUR"`). Applied to both native candle subscription and tick-aggregation fallback. |
| `bb_period` | int | `20` | Bollinger Bands lookback period |
| `bb_std` | float | `1.5` | Bollinger Bands standard deviation multiplier |
| `rsi_period` | int | `7` | RSI lookback period |
| `rsi_oversold` | float | `30.0` | RSI level below which long entries are considered |
| `rsi_overbought` | float | `70.0` | RSI level above which short entries are considered |
| `max_long_positions` | int | `10` | Maximum number of simultaneously open long positions |
| `max_short_positions` | int | `10` | Maximum number of simultaneously open short positions |
| `contract_size` | float | `3.0` | Position size in contracts — uniform for every entry |
| `min_dist_between_entries_ticks` | float | `10` | Minimum price distance between consecutive entries in the same grid (ticks) |
| `take_profit_ticks` | float | `8` | Broker take-profit distance from entry price (ticks). Used only when `close_mode` is `"fixed"`. |
| `stop_loss_ticks` | float | `16` | Broker stop-loss distance from entry price (ticks). Used only when `close_mode` is `"fixed"`. |
| `close_mode` | string | `"dynamic"` | Controls how TP/SL distances are determined. `"fixed"`: use `take_profit_ticks` and `stop_loss_ticks` directly. `"dynamic"`: derive both distances from ATR (see Dynamic Close Mode below). Only `"fixed"` and `"dynamic"` are valid — any other value fails schema validation at startup. Can be changed at runtime via hot-reload. |
| `atr_period` | int | `14` | ATR lookback period. Used in both `"fixed"` and `"dynamic"` modes (computed unconditionally for hot-switch readiness). Requires restart to change. |
| `atr_multiplier_tp` | float | `1.0` | ATR multiplier for take-profit distance in dynamic mode. `limit_dist = round(atr_multiplier_tp * ATR)`. Can be changed at runtime via hot-reload. |
| `atr_multiplier_sl` | float | `1.5` | ATR multiplier for stop-loss distance in dynamic mode. `stop_dist = round(atr_multiplier_sl * ATR)`. Can be changed at runtime via hot-reload. |

Infrastructure parameters (`leverage`, `initial_cash_balance`,
`demo_starting_balance`) come from `config.json["trading"]`. `epic` is stored
in the strategy JSON (`strategies/RSIBollingerStrategyV2.json`).

> **Note on types**: `float` fields accept integer values — `"take_profit_ticks": 8` and `"take_profit_ticks": 8.0` are both valid.

> **Note**: `config.json["trading"]` also contains `security_buffer` (carried over from V1). V2 does not use it.

> **Note**: `spread` is no longer a static config parameter. The strategy reads
> spread dynamically from each candle's `OFR_CLOSE - BID_CLOSE` value delivered
> by `IGStreamingClient`. No spread key is needed in `config.json`.

> **Note on defaults**: The "Default" column shows reference values from the original implementation. The committed `RSIBollingerStrategyV2.json` may differ — it reflects current live-tuned values.

---

## Configuration Example

`strategies/RSIBollingerStrategyV2.json`:

```json
{
  "epic": "IX.D.SPTRD.IFMM.IP",

  "api_mode": "streaming",
  "candle_frequency": "5min",
  "operation_mode": "tick",

  "bb_period": 20,
  "bb_std": 1.5,
  "rsi_period": 7,
  "rsi_oversold": 30.0,
  "rsi_overbought": 70.0,

  "max_long_positions": 10,
  "max_short_positions": 10,

  "contract_size": 3.0,
  "min_dist_between_entries_ticks": 10,
  "take_profit_ticks": 8,
  "stop_loss_ticks": 16,

  "close_mode": "dynamic",
  "atr_period": 14,
  "atr_multiplier_tp": 1.0,
  "atr_multiplier_sl": 1.5
}
```

---

## How to Run

```bash
source venv/bin/activate          # macOS/Linux
.\venv\Scripts\activate           # Windows

python kuroko.py --strategy RSIBollingerStrategyV2
```

`kuroko.py` reads `api_mode` from the strategy JSON and instantiates
`IGStreamingClient` automatically. The streaming connection is established
inside `strategy.run()` and torn down when the process exits (via `CTRL+C` or
any exception).

Stop the bot with `CTRL+C` — `kuroko.py` calls `streaming_client.stop()` in
its `finally` block, which disconnects the Lightstreamer session cleanly.

> Note: `strategy.stop()` is not called by `kuroko.py` on `CTRL+C`. The teardown sequence is: `KeyboardInterrupt` → `_run_strategy` catches it → `finally` calls `streaming_client.stop()` → Lightstreamer disconnects → worker thread drains the queue and exits.

---

## Hot-Reload Config

The strategy can apply changes to `strategies/RSIBollingerStrategyV2.json` at runtime — without restarting the bot. The file is checked on every candle close via `os.path.getmtime`. When the mtime changes, the file is re-parsed and validated; safe params are applied immediately to `self.params`.

### How to trigger a reload

1. Edit `strategies/RSIBollingerStrategyV2.json` and save.
2. The change takes effect on the next closed candle (within 5 minutes with default frequency).
3. Check the logs for `[HOT-RELOAD]` lines confirming what was applied.

### Hot-safe parameters (apply without restart)

| Parameter | Description |
|-----------|-------------|
| `rsi_oversold` | RSI threshold for long entry |
| `rsi_overbought` | RSI threshold for short entry |
| `max_long_positions` | Maximum simultaneous long positions |
| `max_short_positions` | Maximum simultaneous short positions |
| `min_dist_between_entries_ticks` | Minimum price distance between grid entries |
| `take_profit_ticks` | Broker take-profit distance (used in fixed mode) |
| `stop_loss_ticks` | Broker stop-loss distance (used in fixed mode) |
| `contract_size` | Position size for new entries |
| `bb_std` | Bollinger Band standard deviation multiplier |
| `close_mode` | Switch between `"fixed"` and `"dynamic"` TP/SL without restart |
| `atr_multiplier_tp` | ATR multiplier for take-profit distance in dynamic mode |
| `atr_multiplier_sl` | ATR multiplier for stop-loss distance in dynamic mode |

### Restart-required parameters (change is logged but NOT applied)

| Parameter | Reason |
|-----------|--------|
| `bb_period` | Changes the indicator calculation window — existing candle buffer would produce inconsistent results |
| `rsi_period` | Same reason as `bb_period` |
| `atr_period` | Same reason as `bb_period` — ATR window would be inconsistent |
| `epic` | The streaming subscription is bound to the epic at startup |
| `candle_frequency` | The streaming resolution is set when `IGStreamingClient` is created |
| `api_mode` | Determines which execution path is used; wired at startup |
| `operation_mode` | Determines whether `_on_candle` routes to tick or candle mode; wired at init |

When a restart-required param changes, a `WARNING` is logged and the current value is kept.

### Log messages

| Level | Format |
|-------|--------|
| INFO | `[HOT-RELOAD] Strategy params file changed — reloading` |
| INFO | `[HOT-RELOAD] <param>: <old> → <new>` |
| WARNING | `[HOT-RELOAD] <param> changed but requires restart — keeping <old>` |
| INFO | `[HOT-RELOAD] Applied <N> param(s), discarded <M> (restart required)` |
| ERROR | `[HOT-RELOAD] Failed to reload params — keeping current: <exc>` |

### Disabling hot-reload

Hot-reload is enabled by default in `kuroko.py` (via `params_path=strategy_path`). To disable it, pass `params_path=None` when constructing the strategy. Existing tests that do not pass `params_path` are unaffected — reload is silently disabled.

---

## Differences from V1

| Feature | RSIBollingerStrategy (V1) | RSIBollingerStrategyV2 |
|---------|---------------------------|------------------------|
| Data source | REST polling (15-min candles, per-cycle API call) | Lightstreamer streaming (5-min candles by default, push) |
| Candle resolution | Configurable via `candle_frequency` (default 15 min) | Configurable via `candle_frequency` |
| Directions | Long only (short entry is defined but rarely triggered in V1) | Bidirectional — independent long and short grids |
| Position sizing | Martingale: each grid level multiplies base size by `martingale_multiplier` | Flat: every entry uses `contract_size` |
| Stop-loss | ATR-based dynamic stop (`atr_sl_multiplier * ATR`) | Broker stop-loss set at open: `stop_loss_ticks` (fixed mode) or ATR-derived distance (dynamic mode) |
| Take-profit | Basket TP (close all positions when avg entry + TP ticks is reached) | Per-position broker TP (set as limit order at open) |
| Exit signal | Weighted average basket crossover | Broker TP/SL orders set at open (no programmatic exit) |
| Drawdown protection | Max drawdown freeze at 75% threshold | None |
| Margin check | Pre-entry margin check with `security_buffer` | None |
| Trend filter | Optional EMA-200 filter (`use_trend_filter`) | None |
| Entry distance | `min_dist_between_entries_ticks` | `min_dist_between_entries_ticks` (same concept) |
