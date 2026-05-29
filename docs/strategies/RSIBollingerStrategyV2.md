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
3. **Simplified risk model** — no martingale sizing, no ATR stop-loss, no
   drawdown freeze. Broker-level take-profit is the sole exit mechanism for
   trend continuations; per-position spread-aware profit is the guard for
   mean-reversion exits.

V2 also supports an optional **tick mode** (`operation_mode: "tick"`) where
entry and exit signals are evaluated on every live tick using indicators cached
from the most recently closed candle.

---

## Operation Modes

`operation_mode` controls whether signals are evaluated on candle close or on
every live tick.

### Candle mode (`operation_mode: "candle"`)

Default behavior. `_on_candle` computes indicators from each closed candle and
immediately evaluates entry and exit signals via `_manage_longs()` and
`_manage_shorts()`. This is identical to the original V2 behavior.

### Tick mode (`operation_mode: "tick"`)

Indicators are computed and cached (`_cached_indicators`) on every candle close.
Entry and exit signals are then evaluated on each live tick by `_on_tick()`, using
the cached indicators together with the live `bid` and `ofr` prices from the tick.

**Warmup gate**: `_on_tick` silently discards ticks until at least one candle has
been processed. No WARNING is logged during this warmup period.

**In-flight guard**: `_tick_long_in_flight` and `_tick_short_in_flight` flags
prevent duplicate REST calls if multiple ticks qualify before the first REST
response returns. Each flag is set immediately before the REST call and reset in
a `finally` block — so it is always `False` after `_on_tick` returns, even if the
REST call raises an exception.

**Spread**: In tick mode, exit profit is calculated using the live `ofr - bid`
spread from the tick itself (sub-second accuracy), not the candle-close spread
stored in `_current_spread`.

**Switching modes**: Set `"operation_mode": "tick"` in the strategy JSON and
redeploy. To revert, set it back to `"candle"` (or remove the key — missing key
defaults to `"candle"`).

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
                                                               → entry/exit signals
```

---

## Startup Lifecycle and Warm-Up

`run()` executes the following sequence on startup:

1. **`_warmup()`** — fetches `max(bb_period, rsi_period) + 1` historical candles
   from the REST API via `IGClient.get_candles()` and appends each row's `Close`
   price directly to `_candle_window`. This pre-fills the window so that indicators
   are valid on the very first live streaming candle.
2. **`_seed_positions_from_broker()`** — fetches all open positions from the broker
   via `IGClient.get_open_positions()`, filters by `self.epic`, and populates
   `_long_positions` / `_short_positions` so that the strategy correctly tracks
   any positions that were open when the bot restarted.
3. **`streaming_client.start(_on_candle, on_tick=...)`** — opens the Lightstreamer
   connection and begins delivering live candles. In tick mode, also opens a second
   subscription to `CHART:{epic}:TICK` via `_DirectTickListener`.
4. **`_stop_event.wait()`** — blocks until `stop()` is called.

### Warm-up details

`_warmup()` calls `IGClient.get_candles(epic, candle_frequency, num_candles)`,
where `num_candles = max(bb_period, rsi_period) + 1`. The returned DataFrame has
capitalized OHLC columns (`Open`, `High`, `Low`, `Close`) and a `DatetimeIndex`.

Each row's `Close` price is appended directly to `_candle_window` as a float.
No conversion to a candle dict is performed and `_on_candle()` is not called
during warm-up, so trading logic cannot fire on REST data regardless of window
fill level.

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
have no recorded spread. In tick mode, `_tick_close_positions` uses the live
tick spread as a fallback for profit checks — a WARNING is logged for every
seeded LONG position noting this.

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
`max(bb_period, rsi_period) + 1` completed candles, ensuring indicators are
computed from a full dataset.

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

### Short entry

A short position is opened when ALL of the following are true:

- `close > bb_upper` — price has closed above the upper Bollinger Band
- `rsi > rsi_overbought` — RSI is in overbought territory
- Current number of open short positions < `max_short_positions`
- Distance from the last short entry price >= `min_dist_between_entries_ticks`
  (or no short positions are open)

The long and short grids are evaluated independently on every candle. An
active short signal does not suppress long entry evaluation and vice versa.

---

## Exit Logic

### Long exit

A long position is closed when BOTH of the following are true:

- `close > bb_upper` — price has closed above the upper Bollinger Band
- `(close - entry_price - spread) * size > 0` — the position is profitable
  after the bid/ask spread

Positions that are still underwater after spread remain open. Only profitable
positions exit.

### Short exit

A short position is closed when BOTH of the following are true:

- `close < bb_lower` — price has closed below the lower Bollinger Band
- `(entry_price - close - spread) * size > 0` — the position is profitable
  after the bid/ask spread

### Spread calculation

The spread used in profit checks is **dynamic** — calculated from each incoming
candle as `OFR_CLOSE - BID_CLOSE`. The streaming client includes a `spread` key
in every candle dict (both native 5-minute and tick-aggregated). `_on_candle`
updates `_current_spread` from this field before evaluating exits, ensuring the
live market spread is always used.

If no candle has been processed yet (strategy just started), the spread defaults
to `0.0` — a conservative fallback that never suppresses a profitable exit.

---

## Position Management

| Behaviour | Detail |
|-----------|--------|
| Position grids | Long and short grids are independent; both may hold positions simultaneously |
| Startup reconciliation | `_seed_positions_from_broker()` pre-populates both grids from broker state before streaming starts — open positions survive restarts |
| Max positions per grid | `max_long_positions` for longs; `max_short_positions` for shorts |
| Minimum entry distance | New entry rejected if `abs(close - last_entry) < min_dist_between_entries_ticks` |
| Position sizing | Flat `contract_size` for every entry — no martingale, no scaling |
| Broker take-profit | Set at `entry_price + take_profit_ticks` (long) or `entry_price - take_profit_ticks` (short) at open |

---

## Risk Management

There is no programmatic stop-loss in V2. Risk is bounded by:

- **Position caps** (`max_long_positions`, `max_short_positions`) — hard upper
  limit on exposure in each direction.
- **Broker-level take-profit** — set as a limit order at open; the broker
  closes the position automatically if the target is reached.
- **Entry distance guard** — prevents adding to a losing grid faster than
  `min_dist_between_entries_ticks` ticks.

There is no drawdown freeze, no ATR rule, and no margin check in V2.

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
                                        │    → entry/exit via ig_client            │
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
| `operation_mode` | string | `"candle"` | Signal evaluation mode. `"candle"`: signals fire on each closed candle (default). `"tick"`: indicators cached on candle close; signals fire on each live tick. Any other value logs a WARNING and falls back to `"candle"`. Missing key defaults to `"candle"`. |
| `candle_frequency` | string | `"1min"` | Candle resolution in `"Nmin"` format (e.g. `"1min"`, `"5min"`, `"15min"`, `"60min"`). Mapped to IG Lightstreamer resolution strings (`"1MINUTE"`, `"5MINUTE"`, `"15MINUTE"`, `"1HOUR"`). Applied to both native candle subscription and tick-aggregation fallback. |
| `bb_period` | int | `20` | Bollinger Bands lookback period |
| `bb_std` | float | `1.5` | Bollinger Bands standard deviation multiplier |
| `rsi_period` | int | `14` | RSI lookback period |
| `rsi_oversold` | int | `40` | RSI level below which long entries are considered |
| `rsi_overbought` | int | `60` | RSI level above which short entries are considered |
| `max_long_positions` | int | `5` | Maximum number of simultaneously open long positions |
| `max_short_positions` | int | `5` | Maximum number of simultaneously open short positions |
| `contract_size` | float | `0.1` | Position size in contracts — uniform for every entry |
| `min_dist_between_entries_ticks` | float | `30` | Minimum price distance between consecutive entries in the same grid (ticks) |
| `take_profit_ticks` | float | `30` | Broker take-profit distance from entry price (ticks) |

Infrastructure parameters (`epic`, `leverage`, etc.) come from
`config.json["trading"]` and are NOT stored in the strategy JSON.

> **Note**: `spread` is no longer a static config parameter. The strategy reads
> spread dynamically from each candle's `OFR_CLOSE - BID_CLOSE` value delivered
> by `IGStreamingClient`. No spread key is needed in `config.json`.

---

## Configuration Example

`strategies/RSIBollingerStrategyV2.json`:

```json
{
  "api_mode": "streaming",
  "candle_frequency": "1min",
  "operation_mode": "candle",
  "bb_period": 20,
  "bb_std": 1.5,
  "rsi_period": 14,
  "rsi_oversold": 40,
  "rsi_overbought": 60,
  "max_long_positions": 5,
  "max_short_positions": 5,
  "contract_size": 0.1,
  "min_dist_between_entries_ticks": 30,
  "take_profit_ticks": 30
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

---

## Differences from V1

| Feature | RSIBollingerStrategy (V1) | RSIBollingerStrategyV2 |
|---------|---------------------------|------------------------|
| Data source | REST polling (15-min candles, per-cycle API call) | Lightstreamer streaming (5-min candles by default, push) |
| Candle resolution | Configurable via `candle_frequency` (default 15 min) | Configurable via `candle_frequency` (default 1min) |
| Directions | Long only (short entry is defined but rarely triggered in V1) | Bidirectional — independent long and short grids |
| Position sizing | Martingale: each grid level multiplies base size by `martingale_multiplier` | Flat: every entry uses `contract_size` |
| Stop-loss | ATR-based dynamic stop (`atr_sl_multiplier * ATR`) | None |
| Take-profit | Basket TP (close all positions when avg entry + TP ticks is reached) | Per-position broker TP (set as limit order at open) |
| Exit signal | Weighted average basket crossover | Per-position spread-aware profit check at opposite BB band |
| Drawdown protection | Max drawdown freeze at 75% threshold | None |
| Margin check | Pre-entry margin check with `security_buffer` | None |
| Trend filter | Optional EMA-200 filter (`use_trend_filter`) | None |
| Entry distance | `min_dist_between_entries_ticks` | `min_dist_between_entries_ticks` (same concept) |
