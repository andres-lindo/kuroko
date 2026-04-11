"""Live trading strategy for IG Markets.

Implements a mean-reversion grid strategy using RSI and Bollinger Bands
signals, martingale position sizing, and ATR-based dynamic stop-losses.
"""
import time
import logging
import talib as ta
import pandas as pd

from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class Strategy:
    """Mean-reversion grid strategy for IG Markets live trading.

    Enters long when price is below the lower Bollinger Band and RSI is
    oversold; enters short on the opposite condition. Each new position
    in the grid uses a martingale multiplier. The entire basket is closed
    when the weighted average entry price reaches the take-profit target.

    Attributes:
        params: Configuration object loaded from Azure Table Storage.
        ig: IGClient instance used for all broker interactions.
        candles: Most-recent OHLC DataFrame with computed indicators.
        max_drawdown_reached: Flag set when equity breaches the drawdown floor.
    """

    def __init__(self, params, ig_client):
        """Initialise strategy state and load parameters.

        Args:
            params: Config object with attributes loaded from Azure Table
                Storage (e.g. candle_frecuency).
            ig_client: Authenticated IGClient instance.
        """
        self.params = params
        self.ig = ig_client
        self.candles = pd.DataFrame()

        # Trading parameters — sourced from Azure Table Storage at startup
        self.max_positions = 5
        self.min_dist_between_entries_ticks = 100.0
        self.martingale_multiplier = 1.5
        self.bb_dev = 1.9
        self.bb_period = 20
        self.rsi_period = 11
        self.rsi_overbought = 76
        self.rsi_oversold = 25
        self.use_trend_filter = False
        self.take_profit_ticks = 240.0
        self.atr_period = 12
        self.atr_sl_multiplier = 11.0

        # Internal state — managed at runtime
        self.position_size = 0.13
        self.max_drawdown_pct = 75.75
        self.ema_period = 200
        self.max_drawdown_reached = False

        # System configuration — change is_live_account to True ONLY for real-money trading
        self.is_live_account = False          # False = DEMO (virtual equity mirror); True = LIVE (broker equity)
        self.demo_starting_balance = 20000.0  # IG demo account starting balance
        self.initial_cash_balance = 4000.0    # Simulated capital to track (maps to demo via leverage)

        self.leverage = 20
        self.epic = "IX.D.NASDAQ.IFMM.IP"
        self.lookback = 300
        self.security_buffer = 1000.0

    def get_candles(self):
        """Fetch the latest candles from IG and compute all indicators.

        Retrieves OHLC data for the configured epic and resolution, then
        calculates RSI, EMA, ATR, and Bollinger Bands using TA-Lib.
        The result is stored in self.candles.
        """
        df = self.ig.get_candles(self.epic, '15min', self.lookback).copy()

        df['rsi'] = ta.RSI(df["Close"], timeperiod=self.rsi_period)
        logger.info(f"RSI calculated: period={self.rsi_period}")

        df['ema'] = ta.EMA(df["Close"], timeperiod=self.ema_period)
        logger.info(f"EMA calculated: period={self.ema_period}")

        df['atr'] = ta.ATR(df["High"], df["Low"], df["Close"], timeperiod=self.atr_period)
        logger.info(f"ATR calculated: period={self.atr_period}")

        df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.BBANDS(
            df["Close"], timeperiod=self.bb_period,
            nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0
        )
        logger.info(f"BBANDS calculated: period={self.bb_period}")

        self.candles = df.tail(self.lookback)

    def manage_positions(self):
        """Evaluate signals and manage the open position grid.

        Checks exit conditions (basket take-profit) first, then enforces
        risk controls (drawdown freeze, margin check), and finally evaluates
        entry conditions (RSI + Bollinger Band breakout). Updates stop-loss
        and take-profit levels on all existing positions after every new entry.
        """
        if self.candles.empty:
            logger.warning("No candle data available to manage positions.")
            return

        current_candle = self.candles.iloc[-1]
        current_price = current_candle['Close']

        positions = self.ig.get_open_positions()
        n_trades = len(positions)

        # --- 0. EXITS (TIME STOP & BASKET TP) ---
        if n_trades > 0:
            positions.sort(key=lambda x: x['createdDate'])
            first_trade = positions[0]

            total_size = sum(p['size'] for p in positions)
            avg_price = sum(p['size'] * p['level'] for p in positions) / total_size
            is_long = first_trade['direction'] == 'BUY'

            if is_long:
                if current_price >= avg_price + self.take_profit_ticks:
                    profit = (current_price - avg_price) * total_size
                    logger.info(f"💰 WIN (LONG) | Size: {total_size} | Avg: {avg_price:.2f} | Curr: {current_price:.2f} | Profit: {profit:.2f}")
                    self.close_all_positions(positions, reason="BasketTP")
                    return
            else:
                if current_price <= avg_price - self.take_profit_ticks:
                    profit = (avg_price - current_price) * total_size
                    logger.info(f"💰 WIN (SHORT) | Size: {total_size} | Avg: {avg_price:.2f} | Curr: {current_price:.2f} | Profit: {profit:.2f}")
                    self.close_all_positions(positions, reason="BasketTP")
                    return

        # Fetch live account state from the broker
        account_info = self.ig.get_account_summary()
        open_pnl = account_info['profitLoss']

        # --- 1. RISK & EQUITY (DEMO vs LIVE) ---
        if self.is_live_account:
            # LIVE mode: trust broker numbers directly
            current_equity = account_info['balance'] + open_pnl
            free_margin = account_info['available']
        else:
            # DEMO mode: mirror realized P&L onto the simulated capital to
            # enforce strict 1:20 leverage against initial_cash_balance
            realized_profit = account_info['balance'] - self.demo_starting_balance
            virtual_balance = self.initial_cash_balance + realized_profit

            current_equity = virtual_balance + open_pnl
            used_margin = sum((p['size'] * p['level'] / self.leverage) for p in positions) if n_trades > 0 else 0
            free_margin = current_equity - used_margin

        # Drawdown floor: equity must not fall below this fraction of starting capital
        floor_value = self.initial_cash_balance * (1 - (self.max_drawdown_pct / 100))

        if current_equity < floor_value:
            if not self.max_drawdown_reached:
                logger.error(f"⚠️ MAX DRAWDOWN | Equity ${current_equity:.2f} < Floor ${floor_value:.2f}. Freezing.")
                self.max_drawdown_reached = True
        elif self.max_drawdown_reached and current_equity > floor_value:
            logger.info(f"✅ RECOVERED | Equity ${current_equity:.2f} > Floor. Reactivating.")
            self.max_drawdown_reached = False

        # --- 2. ENTRIES (GRID) ---
        if self.max_drawdown_reached or n_trades >= self.max_positions:
            return

        if n_trades > 0:
            dist_to_last = abs(current_price - positions[-1]['level'])
            if dist_to_last < self.min_dist_between_entries_ticks:
                return

        # Indicators and filters
        current_atr = current_candle['atr']
        sl_dist = current_atr * self.atr_sl_multiplier

        if self.use_trend_filter:
            current_ema = current_candle['ema']
            can_buy = current_price > current_ema
            can_sell = current_price < current_ema
        else:
            can_buy = True
            can_sell = True

        # Fractional martingale sizing: each grid level scales by multiplier^n,
        # floored at the base position_size to avoid sub-minimum orders
        current_size = max(self.position_size, round(self.position_size * (self.martingale_multiplier ** n_trades), 2))

        # --- 3. MARGIN CHECK (unified for DEMO and LIVE) ---
        cost_to_open = (current_price / self.leverage) * current_size

        if cost_to_open > (free_margin - self.security_buffer):
            modo = "LIVE" if self.is_live_account else "DEMO/VIRTUAL"
            logger.warning(
                f"🚫 INSUFFICIENT MARGIN {modo} | Req: ${cost_to_open:.2f} | "
                f"Free: ${free_margin:.2f} | Buffer: ${self.security_buffer} | "
                f"Attempt: x{current_size} @ {current_price:.2f}"
            )
            return

        # --- 4. PROJECTED AVG PRICE BEFORE ENTRY ---
        # Pre-compute what the weighted average entry price would become if
        # this new order fills, so TP levels can be set correctly at open time
        if n_trades > 0:
            total_size = sum(p['size'] for p in positions)
            total_value = sum(p['size'] * p['level'] for p in positions)
        else:
            total_size = 0
            total_value = 0

        futuro_total_size = total_size + current_size
        futuro_total_value = total_value + (current_size * current_price)
        futuro_avg_price = futuro_total_value / futuro_total_size

        # --- 5. EXECUTION ---
        bb_lower = current_candle['bb_lower']
        bb_upper = current_candle['bb_upper']
        current_rsi = current_candle['rsi']

        # LONG
        if can_buy and current_price < bb_lower and current_rsi < self.rsi_oversold:
            if n_trades > 0:
                if positions[0]['direction'] == 'SELL':
                    return
                if current_price >= positions[-1]['level']:
                    return

            # 1. Target and stop as exact price levels (used to update existing positions)
            target_price = round(futuro_avg_price + self.take_profit_ticks, 2)
            stop_price = round(current_price - sl_dist, 2)

            # 2. Target and stop as distances (used when opening the new position in IG)
            limit_dist = round(abs(target_price - current_price), 2)
            stop_dist = round(sl_dist, 2)

            try:
                # Open the new position
                self.ig.open_position(epic=self.epic, size=current_size, side='BUY', stop=stop_dist, limit=limit_dist)
                logger.info(f"⬆️ BUY #{n_trades + 1} | x{current_size} @ {current_price:.2f} | SL Level: {stop_price} | TP Level: {target_price}")

                # Brief pause to avoid race conditions when updating positions immediately after open
                time.sleep(2)

                # Refresh positions to include the newly opened one
                upd_df = self.ig.get_open_positions()

                if len(upd_df) > 0:
                    # Recalculate the actual weighted average after the fill
                    real_size = sum(p['size'] for p in upd_df)
                    real_avg = sum(p['size'] * p['level'] for p in upd_df) / real_size

                    # Derive TP and SL from the confirmed post-fill average
                    real_tp = round(real_avg + self.take_profit_ticks, 2)
                    real_sl = round(current_price - sl_dist, 2)

                    for p in upd_df:
                        deal_id = p['dealId']
                        # Normalise floats to 2 decimal places to avoid IG API precision errors
                        safe_limit = round(float(real_tp), 2)
                        safe_stop = round(float(real_sl), 2)

                        self.ig.update_position(dealid=deal_id, limit=safe_limit, stop=safe_stop)
                        logger.info(f"🔄 Position {deal_id} updated -> New TP: {safe_limit} | New SL: {safe_stop}")
            except Exception as e:
                logger.error(f"Error opening/updating LONG positions: {e}")

        # SHORT
        elif can_sell and current_price > bb_upper and current_rsi > self.rsi_overbought:
            if n_trades > 0:
                if positions[0]['direction'] == 'BUY':
                    return
                if current_price <= positions[-1]['level']:
                    return

            # 1. Target and stop as exact price levels (used to update existing positions)
            target_price = round(futuro_avg_price - self.take_profit_ticks, 2)
            stop_price = round(current_price + sl_dist, 2)

            # 2. Target and stop as distances (used when opening the new position in IG)
            limit_dist = round(abs(current_price - target_price), 2)
            stop_dist = round(sl_dist, 2)

            try:
                # Open the new position
                self.ig.open_position(epic=self.epic, size=current_size, side='SELL', stop=stop_dist, limit=limit_dist)
                logger.info(f"⬇️ SELL #{n_trades + 1} | x{current_size} @ {current_price:.2f} | SL Level: {stop_price} | TP Level: {target_price}")

                # Brief pause to avoid race conditions when updating positions immediately after open
                time.sleep(2)

                # Refresh positions to include the newly opened one
                upd_df = self.ig.get_open_positions()

                # Update TP and SL on all existing positions
                if len(upd_df) > 0:
                    # Recalculate the actual weighted average after the fill
                    real_size = sum(p['size'] for p in upd_df)
                    real_avg = sum(p['size'] * p['level'] for p in upd_df) / real_size

                    # Derive TP and SL from the confirmed post-fill average
                    real_tp = round(real_avg - self.take_profit_ticks, 2)
                    real_sl = round(current_price + sl_dist, 2)

                    for p in upd_df:
                        deal_id = p['dealId']
                        # Normalise floats to 2 decimal places to avoid IG API precision errors
                        safe_limit = round(float(real_tp), 2)
                        safe_stop = round(float(real_sl), 2)

                        self.ig.update_position(dealid=deal_id, limit=safe_limit, stop=safe_stop)
                        logger.info(f"🔄 Position {deal_id} updated -> New TP: {safe_limit} | New SL: {safe_stop}")
            except Exception as e:
                logger.error(f"Error opening/updating SHORT positions: {e}")

    def close_all_positions(self, positions, reason):
        """Close every open position in the basket.

        Args:
            positions: List of open position dicts as returned by
                IGClient.get_open_positions().
            reason: Label string logged with each close (e.g. 'BasketTP').
        """
        for p in positions:
            close_direction = 'SELL' if p['direction'] == 'BUY' else 'BUY'
            self.ig.close_position(p['dealId'], close_direction, p['size'])
            logger.info(f"Closing {p['dealId']} ({p['direction']}) -> {close_direction} reason={reason}")

    def log_account_status(self):
        """Log a structured account snapshot to the configured logger.

        Reads live account data from the broker and computes equity,
        used margin, free margin, and margin level percentage. In DEMO
        mode the same virtual equity mirror used in manage_positions is
        applied so the logged numbers are consistent with trading decisions.
        """
        try:
            account_info = self.ig.get_account_summary()
            open_pnl = account_info['profitLoss']
            positions = self.ig.get_open_positions()
            n_trades = len(positions) if positions else 0

            if self.is_live_account:
                # LIVE mode: use raw broker figures
                current_equity = account_info['balance'] + open_pnl
                used_margin = account_info['margin']
                free_margin = account_info['available']
                modo = "LIVE"
            else:
                # DEMO mode: strict 1:20 virtual simulation against initial_cash_balance
                realized_profit = account_info['balance'] - self.demo_starting_balance
                virtual_balance = self.initial_cash_balance + realized_profit

                current_equity = virtual_balance + open_pnl
                used_margin = sum((p['size'] * p['level'] / self.leverage) for p in positions) if n_trades > 0 else 0
                free_margin = current_equity - used_margin
                modo = f"VIRTUAL (1:{self.leverage})"

            # --- MARGIN LEVEL (%) ---
            if used_margin > 0:
                margin_level_pct = (current_equity / used_margin) * 100
                margin_str = f"{margin_level_pct:.2f}%"

                if margin_level_pct < 120:
                    health_icon = "🚨 DANGER"
                elif margin_level_pct < 200:
                    health_icon = "⚠️ ALERT"
                else:
                    health_icon = "✅ HEALTHY"
            else:
                margin_str = "N/A"
                health_icon = "💤 IDLE"

            # Print summary block
            logger.info(
                f"📊 STATUS {modo} | "
                f"Equity: ${current_equity:.2f} | "
                f"Used Margin: ${used_margin:.2f} | "
                f"Margin Level: {margin_str} {health_icon} | "
                f"Free: ${free_margin:.2f} | "
                f"Positions: {n_trades}"
            )
        except Exception as e:
            logger.error(f"Error generating account status report: {e}")

    def run(self):
        """Start the main trading loop.

        Wakes up every minute, checks whether the current minute aligns
        with the configured candle frequency, and runs the full cycle:
        fetch candles → manage positions → log account status.
        """
        freq = int(self.params.candle_frecuency.replace("min", ""))
        logger.info("Strategy running. Execution every %d minutes.", freq)
        next_tick = (datetime.now() + timedelta(minutes=1)).replace(second=0, microsecond=0)

        while True:
            sleep_seconds = (next_tick - datetime.now()).total_seconds()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            now = datetime.now()

            if now.minute % freq == 0:
                logger.info(f"Strategy execution at {now.strftime('%H:%M:%S')}")
                self.get_candles()

                if self.candles.empty:
                    logger.info("No candles available to trade.")
                else:
                    logger.info("Last 5 candles.\n%s", self.candles.tail(5).to_string())
                    self.manage_positions()

                    # Log account snapshot after every cycle
                    self.log_account_status()

            next_tick += timedelta(minutes=1)
