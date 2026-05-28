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
┌────────────────────────────────┐      ┌──────────────────────────────────┐
│  Lightstreamer (LS) thread     │      │  IGStreamingClient worker thread  │
│                                │      │                                  │
│  LS adapter fires onItemUpdate ──────► queue.put(candle)                 │
│  on _CandleSubscriptionListener│      │  ↓                               │
│                                │      │  queue.get() → on_candle(candle) │
└────────────────────────────────┘      │  _on_candle → _manage_longs()   │
                                        │            → _manage_shorts()   │
                                        │  ig_client.open/close_position()│
                                        └──────────────────────────────────┘
```

Candle data flows: **LS thread** → `_CandleSubscriptionListener.onItemUpdate()`
enqueues to `IGStreamingClient._candle_queue` → **worker thread** dequeues and
calls `_on_candle()` directly (registered as the callback via `streaming_client.start(_on_candle)`).

There is no intermediate re-queue inside the strategy. `run()` blocks on
`_stop_event.wait()` while the streaming client's worker thread handles all candle
delivery and trading logic. All IG REST calls (`open_position`, `close_position`)
happen on the worker thread — no concurrent REST calls.

---

## Parameters Reference

Strategy parameters are stored in `strategies/RSIBollingerStrategyV2.json` and
loaded at startup into a `types.SimpleNamespace` via `load_params()`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `api_mode` | string | `"streaming"` | Must be `"streaming"` — tells `kuroko.py` to instantiate `IGStreamingClient` |
| `candle_frequency` | string | `"5min"` | Candle resolution in `"Nmin"` format (e.g. `"1min"`, `"5min"`, `"15min"`, `"60min"`). Mapped to IG Lightstreamer resolution strings (`"1MINUTE"`, `"5MINUTE"`, `"15MINUTE"`, `"1HOUR"`). Applied to both native candle subscription and tick-aggregation fallback. |
| `bb_period` | int | `20` | Bollinger Bands lookback period |
| `bb_std` | float | `2.0` | Bollinger Bands standard deviation multiplier |
| `rsi_period` | int | `14` | RSI lookback period |
| `rsi_oversold` | int | `30` | RSI level below which long entries are considered |
| `rsi_overbought` | int | `70` | RSI level above which short entries are considered |
| `max_long_positions` | int | `5` | Maximum number of simultaneously open long positions |
| `max_short_positions` | int | `5` | Maximum number of simultaneously open short positions |
| `contract_size` | float | `0.1` | Position size in contracts — uniform for every entry |
| `min_dist_between_entries_ticks` | float | `20.0` | Minimum price distance between consecutive entries in the same grid (ticks) |
| `take_profit_ticks` | float | `240.0` | Broker take-profit distance from entry price (ticks) |

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
  "candle_frequency": "5min",
  "bb_period": 20,
  "bb_std": 2.0,
  "rsi_period": 14,
  "rsi_oversold": 30,
  "rsi_overbought": 70,
  "max_long_positions": 5,
  "max_short_positions": 5,
  "contract_size": 0.1,
  "min_dist_between_entries_ticks": 20,
  "take_profit_ticks": 240.0
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
| Candle resolution | Configurable via `candle_frequency` (default 15 min) | Configurable via `candle_frequency` (default 5 min) |
| Directions | Long only (short entry is defined but rarely triggered in V1) | Bidirectional — independent long and short grids |
| Position sizing | Martingale: each grid level multiplies base size by `martingale_multiplier` | Flat: every entry uses `contract_size` |
| Stop-loss | ATR-based dynamic stop (`atr_sl_multiplier * ATR`) | None |
| Take-profit | Basket TP (close all positions when avg entry + TP ticks is reached) | Per-position broker TP (set as limit order at open) |
| Exit signal | Weighted average basket crossover | Per-position spread-aware profit check at opposite BB band |
| Drawdown protection | Max drawdown freeze at 75% threshold | None |
| Margin check | Pre-entry margin check with `security_buffer` | None |
| Trend filter | Optional EMA-200 filter (`use_trend_filter`) | None |
| Entry distance | `min_dist_between_entries_ticks` | `min_dist_between_entries_ticks` (same concept) |
