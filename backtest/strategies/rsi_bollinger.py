"""RSI + Bollinger Bands mean-reversion strategy for backtesting.

Enters long when price falls below the lower Bollinger Band and RSI is
oversold; enters short when price rises above the upper band and RSI is
overbought. Uses a martingale grid for position sizing and a basket
take-profit to exit all legs simultaneously.
"""

from backtesting import Strategy
import talib
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class RSIBollingerStrategy(Strategy):
    """Mean-reversion strategy combining RSI and Bollinger Bands signals.

    Entry signals require both a band breach (price outside the BB envelope)
    and an RSI extreme (oversold for longs, overbought for shorts). Position
    sizing follows a martingale grid: each additional entry at a worse price
    uses a larger contract count to lower the basket average. All positions
    share a single take-profit level anchored to the basket average price.

    Attributes:
        initial_cash_balance: Starting capital in USD (mirrors engine config).
        leverage: Margin leverage ratio used for cost calculations.
        commission: Simulated round-trip commission rate per trade.
        silent_mode: When True, suppresses per-bar log output for faster
            Optuna trial execution.
        max_positions: Maximum number of simultaneous open legs in the grid.
        min_dist_between_entries_ticks: Minimum price distance (in ticks)
            between consecutive grid entries to prevent clustering.
        martingale_multiplier: Contract-size growth factor per grid level
            (e.g. 1.5 → sizes: 1×, 1.5×, 2.25×, …).
        bb_dev: Bollinger Band standard deviation width.
        bb_period: Bollinger Band and SMA lookback period.
        rsi_period: RSI oscillator lookback period.
        rsi_overbought: RSI level above which short entries are triggered.
        rsi_oversold: RSI level below which long entries are triggered.
        use_trend_filter: When True, only take trades in the EMA-200 trend
            direction (disables pure mean-reversion mode).
        take_profit_ticks: Profit target distance in ticks from the basket
            average entry price.
        atr_period: ATR lookback period for dynamic stop-loss calculation.
        atr_sl_multiplier: ATR multiplier applied to determine stop distance.
        contract_multiplier: Point-value multiplier (set to 1 for NQ/ES
            futures in the backtest environment).
        position_size: Base contract count for the first grid entry.
        max_drawdown_pct: Equity drawdown percentage that triggers the
            trading freeze.
        ema_period: EMA lookback period used by the optional trend filter.
        security_buffer: Minimum free-margin buffer (in USD) required before
            opening a new position, mirroring IG's margin model.
        log_all_candles: When True, logs every bar even without a trade
            event (useful for debugging).
    """

    # Parameters that also configure the engine
    initial_cash_balance = 300000
    leverage = 20
    commission = 0.00012
    silent_mode = False

    # Tunable strategy parameters
    max_positions = 5
    min_dist_between_entries_ticks = 16.0
    martingale_multiplier = 1.5
    bb_dev = 1.5
    bb_period = 23
    rsi_period = 14
    rsi_overbought = 77
    rsi_oversold = 40
    use_trend_filter = False
    take_profit_ticks = 60.0
    atr_period = 11
    atr_sl_multiplier = 12.0
    
    # Internal state variables — not included in Optuna search spaces.
    contract_multiplier = 1
    position_size = 13           # 20 integer units representing 0.2 contracts in live
    max_drawdown_pct = 80
    ema_period = 200
    security_buffer = 100000.0    # 50000 representing $500 buffer in live scale

    # Debugging flag
    log_all_candles = False

    @classmethod
    def prepare_data(cls, df, start_date, end_date):
        """Filter and normalise raw OHLC CSV data for this strategy.

        Handles the column naming conventions used by common NQ/ES data
        vendors: renames ``Last`` → ``Close``, lower-case variants, and
        ``vol`` → ``Volume``. Sets the ``Time`` column as a datetime index
        if present, then slices to the requested date range.

        Args:
            df: Raw :class:`pandas.DataFrame` loaded from a CSV file.
            start_date: Inclusive start of the simulation window
                (string ``'YYYY-MM-DD'`` or datetime-compatible).
            end_date: Inclusive end of the simulation window.

        Returns:
            Filtered :class:`pandas.DataFrame` with a datetime index,
            standard OHLCV column names, and only rows within
            ``[start_date, end_date]``.
        """
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df.set_index('Time', inplace=True)

        rename_map = {
            'Last': 'Close', 'last': 'Close', 'close': 'Close', 
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'volume': 'Volume', 'vol': 'Volume'
        }
        df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)
        return df[(df.index >= start_date) & (df.index <= end_date)].copy()

    def init(self):
        """Initialise technical indicators. Called once before the first bar.

        All indicators are wrapped with ``self.I()`` so ``backtesting.py``
        can track and plot them correctly. The ``max_drawdown_reached`` flag
        and ``last_closed_count`` sentinel are also reset here so the strategy
        is stateless across repeated ``bt.run()`` calls during optimisation.
        """
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=self.bb_period, nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0
        )
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.max_drawdown_reached = False 

        # Sentinel for the closed-trade detector in next(); tracks how many
        # trades were closed as of the previous bar.
        self.last_closed_count = 0

    def log(self, txt):
        """Emit a structured log line with timestamp, price, and margin data.

        Silenced when ``silent_mode`` is True (e.g. during Optuna trials) to
        avoid flooding output with per-bar diagnostics. Margin metrics are
        recomputed on every call so the log always reflects the current bar's
        state rather than cached values.

        Args:
            txt: Free-form message to append after the standard prefix
                (timestamp, price, equity, margin stats).
        """
        if not self.silent_mode:
            dt = self.data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            price = self.data.Close[-1]
            used_margin = sum((price / self.leverage) * abs(t.size) for t in self.trades)
            margin_level_pct = (self.equity / used_margin) * 100 if used_margin > 0 else 1000
            free_margin = self.equity - used_margin
            logger.info(f"{dt}, {txt}, Price: {price:.2f}, Equity: {self.equity:.2f}. Used Margin: {abs(used_margin):.2f}. Margin Level: {margin_level_pct:.2f}%. Free Margin: {free_margin:.2f}")

    def next(self):
        """Execute strategy logic for the current bar.

        Called by ``backtesting.py`` on every bar after the indicator warmup
        period. Evaluates in the following order:

        1. Simulated-broker closed-trade detector (TP/SL hit detection).
        2. Manual basket take-profit fallback (exits all legs at once).
        3. Margin call and liquidation checks.
        4. Max-drawdown freeze guard.
        5. Grid entry logic (RSI + Bollinger Band signal with margin check).
        """
        if self.max_drawdown_reached:
            return

        # --- SIMULATED BROKER TP/SL DETECTOR ---
        # backtesting.py closes positions silently when a broker-level TP or
        # SL is hit. Comparing the current closed_trades count to the previous
        # bar's count lets us detect and log those automatic closures.
        current_closed_count = len(self.closed_trades)
        if current_closed_count > self.last_closed_count:
            # Slice only the trades that were closed on this bar.
            newly_closed = self.closed_trades[self.last_closed_count:]

            # Aggregate P&L and size across the entire basket that was closed.
            total_profit = sum(t.pl for t in newly_closed)
            total_size = sum(abs(t.size) for t in newly_closed)

            # Positive PnL → TP triggered; negative → SL triggered.
            if total_profit > 0:
                self.log(f"TAKE PROFIT AUTO | Closed: {len(newly_closed)} pos | Size: {total_size} | Net Profit: ${total_profit:.2f}")
            else:
                self.log(f"STOP LOSS AUTO | Closed: {len(newly_closed)} pos | Size: {total_size} | Net Loss: ${total_profit:.2f}")

            # Advance the sentinel so the next bar starts from the right count.
            self.last_closed_count = current_closed_count
        # ----------------------------------------

        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]
        trades = self.trades
        n_trades = len(trades)

        if self.log_all_candles:
            self.log("OK") 

        # --- 0. EXITS (BASKET TP MANUAL FALLBACK) ---
        if n_trades > 0:
            avg_price = sum(abs(t.size) * t.entry_price for t in trades) / sum(abs(t.size) for t in trades)
            total_size_abs = sum(abs(t.size) for t in trades)

            if trades[0].is_long:
                if price >= avg_price + self.take_profit_ticks:
                    self.log(f"WIN (LONG) | Size: {total_size_abs} | Profit: ${(price - avg_price) * total_size_abs:.2f}")
                    for t in trades: t.close()
                    return
            else:
                if price <= avg_price - self.take_profit_ticks:
                    self.log(f"WIN (SHORT) | Size: {total_size_abs} | Profit: ${(avg_price - price) * total_size_abs:.2f}")
                    for t in trades: t.close()
                    return

        # MARGIN CALL AND LIQUIDATION
        used_margin = sum((price / self.leverage) * abs(t.size) for t in trades)
        margin_level_pct = (self.equity / used_margin) * 100 if used_margin > 0 else 1000

        if margin_level_pct <= 100 and margin_level_pct > 50:
            self.log(f"MARGIN CALL: Level {margin_level_pct:.2f}% <= 100%. Equity: ${self.equity:.2f}, Used Margin: ${used_margin:.2f}")

        # Stop Out: broker force-closes all positions at the liquidation level.
        if margin_level_pct <= 50:
            self.log(f"LIQUIDATION: Level {margin_level_pct:.2f}% <= 50%. Equity: ${self.equity:.2f}, Used Margin: ${used_margin:.2f}")
            for t in trades:
                t.close()  # Force-close each leg.
            self.max_drawdown_reached = True 
            return

        # --- 1. SAFETY: MAX DRAWDOWN (FREEZE) ---
        floor_value = self.initial_cash_balance * (1 - (self.max_drawdown_pct / 100))
        if self.equity < floor_value:
            if not self.max_drawdown_reached:
                self.log(f"MAX DD HIT! Equity ${self.equity:.2f} < ${floor_value:.2f}. Freezing.")
                self.max_drawdown_reached = True
        elif self.max_drawdown_reached and self.equity > floor_value:
            self.log(f"RECOVERED. Resuming entries.")
            self.max_drawdown_reached = False

        # --- 2. ENTRIES (GRID) ---
        # Block new entries if the drawdown freeze is active or the grid is full.
        if self.max_drawdown_reached or n_trades >= self.max_positions:
            return

        if n_trades > 0:
            if abs(price - trades[-1].entry_price) < self.min_dist_between_entries_ticks: return

        # Trend and signal filters
        sl_dist = atr * self.atr_sl_multiplier
        can_buy = price > self.ema[-1] if self.use_trend_filter else True
        can_sell = price < self.ema[-1] if self.use_trend_filter else True

        # Martingale sizing: multiply the base size by the factor raised to the
        # current grid level, then round to the nearest integer so that
        # backtesting.py receives a whole contract count.
        current_size = max(self.position_size, int(round(self.position_size * (self.martingale_multiplier ** n_trades))))

        # --- 3. MARGIN CHECK (REALITY CHECK) ---
        cost_to_open = (price / self.leverage) * current_size
        used_margin = sum((price / self.leverage) * abs(t.size) for t in trades)
        free_margin = self.equity - used_margin

        # Enforce a security buffer before each entry, mirroring IG's margin model.
        if cost_to_open > (free_margin - self.security_buffer):
            self.log(f"NO MARGIN | Req: ${cost_to_open:.2f} | Free: ${free_margin:.2f} | Buffer: ${self.security_buffer}")
            return

        # --- 4. DYNAMIC TARGET PRICE CALCULATION ---
        if n_trades > 0:
            total_size = sum(abs(t.size) for t in trades)
            total_value = sum(abs(t.size) * t.entry_price for t in trades)
        else:
            total_size = 0
            total_value = 0

        futuro_total_size = total_size + current_size
        futuro_total_value = total_value + (current_size * price)
        futuro_avg_price = futuro_total_value / futuro_total_size

        # --- 5. EXECUTION (WITH TP/SL UPDATE ON ALL OPEN LEGS) ---
        if can_buy and price < self.bb_lower[-1] and rsi < self.rsi_oversold:
            if n_trades > 0 and (trades[0].is_short or price >= trades[-1].entry_price): return 
            
            # Compute exact TP/SL levels as they would be submitted to the broker.
            target_price = round(futuro_avg_price + self.take_profit_ticks, 2)
            stop_price = round(price - sl_dist, 2)

            self.buy(size=current_size, sl=stop_price, tp=target_price)
            self.log(f"BUY #{n_trades + 1} (x{current_size}) @ {price:.2f} | SL: {stop_price} | TP: {target_price}")
            
            for t in self.trades:
                t.sl = stop_price
                t.tp = target_price

        elif can_sell and price > self.bb_upper[-1] and rsi > self.rsi_overbought:
            if n_trades > 0 and (trades[0].is_long or price <= trades[-1].entry_price): return 
            
            target_price = round(futuro_avg_price - self.take_profit_ticks, 2)
            stop_price = round(price + sl_dist, 2)

            self.sell(size=current_size, sl=stop_price, tp=target_price)
            self.log(f"SELL #{n_trades + 1} (x{current_size}) @ {price:.2f} | SL: {stop_price} | TP: {target_price}")
            
            for t in self.trades:
                t.sl = stop_price
                t.tp = target_price