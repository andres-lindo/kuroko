# RSIBollingerStrategy

Mean-reversion grid strategy for IG Markets live trading. Trades NASDAQ 100
(or S&P 500) futures on a 15-minute candle cadence using RSI and Bollinger
Bands signals, martingale position sizing, and ATR-based dynamic stop-losses.

Module: `strategies/RSIBollingerStrategy.py` | Logger name: `strategies.RSIBollingerStrategy`

---

## Entry Logic

A long position is opened when both conditions are true simultaneously on the
most recently completed candle (the current incomplete candle is always stripped
before signal calculation):

- `close < bb_lower` — price has broken below the lower Bollinger Band
- `rsi < rsi_oversold` — RSI is in oversold territory

A short position is opened when:

- `close > bb_upper` — price has broken above the upper Bollinger Band
- `rsi > rsi_overbought` — RSI is in overbought territory

**Optional trend filter** (`use_trend_filter = true`): long entries are only
allowed when `close > ema_200`; short entries only when `close < ema_200`.

**Minimum distance guard**: a new grid entry is rejected if the current price
is within `min_dist_between_entries_ticks` of the last open position. This
prevents stacking on fast moves.

**Margin check**: before every entry, the required margin is calculated as
`(current_price / leverage) * current_size`. If this exceeds
`free_margin - security_buffer`, the entry is skipped with a WARNING log.

---

## Position Sizing (Martingale Grid)

Up to `max_positions` (default: 5) positions may be open simultaneously in the
same direction. Each new grid level scales the base size by the martingale
multiplier:

```
size_n = max(position_size, round(position_size * martingale_multiplier ^ n, 2))
```

where `n` is the current number of open positions before the new entry.

| Grid level | Multiplier (default 1.5x) | Effective size (base 0.13) |
|------------|--------------------------|---------------------------|
| 1st entry  | 1.00x | 0.13 |
| 2nd entry  | 1.50x | 0.195 |
| 3rd entry  | 2.25x | 0.2925 |
| 4th entry  | 3.375x | 0.439 |
| 5th entry  | 5.0625x | 0.658 |

---

## Exit Logic

**Basket take-profit**: all positions are closed together when the weighted
average entry price satisfies the take-profit condition:

- Long basket: `current_price >= avg_entry + take_profit_ticks`
- Short basket: `current_price <= avg_entry - take_profit_ticks`

Profit is calculated as `(current_price - avg_entry) * total_size` for longs
(reversed for shorts) and logged before the close.

**Broker-level TP**: at the time each position is opened, a limit order is
placed at `avg_entry + take_profit_ticks` as a secondary safety net (calculated
from the projected post-fill average, then recalculated from the actual fill).

**Dynamic stop-loss**: each position is assigned a stop at
`entry_price - (atr * atr_sl_multiplier)` for longs (reversed for shorts).
Stops are recalculated and updated on all existing positions after every new
grid entry.

---

## Protection Mechanisms

### ATR-based dynamic stop-loss

The stop distance is computed as `atr * atr_sl_multiplier` using the ATR of
the most recent completed candle. After each new grid entry, all existing
positions have their stop-loss updated to `current_price - sl_dist` (longs)
or `current_price + sl_dist` (shorts) via `IGClient.update_position`.

### Max drawdown freeze

When `current_equity < initial_cash_balance * (1 - max_drawdown_pct / 100)`,
no new positions are opened. The freeze state is stored in
`self.max_drawdown_reached`. The freeze lifts automatically when equity
recovers above the floor.

In DEMO mode, `current_equity = virtual_balance + open_pnl` where
`virtual_balance = initial_cash_balance + realized_profit`.

### Margin check (DEMO and LIVE)

Before every entry: `cost_to_open = (current_price / leverage) * current_size`.
If `cost_to_open > free_margin - security_buffer`, the entry is skipped.

In DEMO mode, `free_margin` is computed as
`current_equity - used_margin` where
`used_margin = sum(size * level / leverage)` for all open positions.

In LIVE mode, `free_margin` comes directly from the broker's `available` field.

---

## Account Mode

`is_live_account` is determined inside `RSIBollingerStrategy.__init__` by reading `os.getenv("ig_acc_type")`. It is `True` only when the value is exactly `"LIVE"` (case-sensitive); any other value — including `"DEMO"` or missing — results in `False`. See [Architecture — Account Mode](../architecture.md#account-mode) for full details.

**In DEMO mode** (`is_live_account=False`):
- Equity = `initial_cash_balance` + realized P&L (virtual simulation)
- Free margin = `current_equity - used_margin` (computed locally)
- `demo_starting_balance` is used only as a reference for realized P&L calculation

**In LIVE mode** (`is_live_account=True`):
- Equity and free margin come directly from the broker's account summary
- `initial_cash_balance` and `demo_starting_balance` are ignored

---

## Parameters Reference

### Signal / Risk Parameters

Stored in `strategies/RSIBollingerStrategy.json` and loaded at startup via
`load_params()` into a `types.SimpleNamespace`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `candle_frequency` | string | `"15min"` | Candle resolution; must match `[1-9]\d*min` (positive integer followed by 'min') |
| `lookback` | int | `300` | Number of candles to fetch per cycle |
| `max_positions` | int | `5` | Maximum number of simultaneous open positions in the grid |
| `position_size` | float | `0.13` | Base position size in contracts |
| `min_dist_between_entries_ticks` | float | `100.0` | Minimum price distance between consecutive grid entries (ticks) |
| `martingale_multiplier` | float | `1.5` | Size multiplier applied at each grid level |
| `take_profit_ticks` | float | `240.0` | Basket take-profit distance from weighted average entry (ticks) |
| `max_drawdown_pct` | float | `75.75` | Maximum drawdown percentage before new entries are frozen |
| `bb_period` | int | `20` | Bollinger Bands lookback period |
| `bb_dev` | float | `1.9` | Bollinger Bands standard deviation multiplier |
| `rsi_period` | int | `11` | RSI lookback period |
| `rsi_overbought` | int | `76` | RSI level above which shorts are allowed |
| `rsi_oversold` | int | `25` | RSI level below which longs are allowed |
| `use_trend_filter` | bool | `false` | Enable EMA trend filter for directional entry control |
| `atr_period` | int | `12` | ATR lookback period for dynamic stop-loss calculation |
| `atr_sl_multiplier` | float | `11.0` | ATR multiplier applied to compute the stop-loss distance |
| `ema_period` | int | `200` | EMA period used by the optional trend filter |

### Infrastructure / Deployment Parameters

Trading parameters are stored in `config.json["trading"]` and loaded via `load_app_config()`. These are passed to the strategy constructor as a `types.SimpleNamespace` (`trading_config`). They are not strategy logic parameters and must not appear in `strategies/RSIBollingerStrategy.json`.

| Key | Type | Default | Source | Description |
|-----|------|---------|--------|-------------|
| `azure_log_partition_key` | string | `"DEV_NQ100"` | `config.json["logging"]` | Azure Blob Storage log blob label; passed to `setup_logging()` as `partition_key` |
| `epic` | string | `"IX.D.NASDAQ.IFMM.IP"` | `config.json["trading"]` | IG Markets instrument identifier |
| `leverage` | int | `20` | `config.json["trading"]` | Leverage ratio used for virtual margin calculation in DEMO mode |
| `demo_starting_balance` | float | `20000.0` | `config.json["trading"]` | IG demo account reference balance used only for realized P&L calculation |
| `initial_cash_balance` | float | `4000.0` | `config.json["trading"]` | Simulated capital base for virtual margin and drawdown floor |
| `security_buffer` | float | `1000.0` | `config.json["trading"]` | Minimum free margin buffer required before any entry (USD) |