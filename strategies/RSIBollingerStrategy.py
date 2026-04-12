"""RSI + Bollinger Bands mean-reversion live trading strategy.

Implements a mean-reversion grid strategy using RSI and Bollinger Bands
signals, martingale position sizing, and ATR-based dynamic stop-losses.
"""
import os
import re
import sys
import json
import time
import types
import logging
import talib as ta
import pandas as pd

from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


# Expected type for each parameter key.
# float fields accept int values (e.g. 240 is valid for take_profit_ticks).
# bool fields are checked before int because bool is a subclass of int in Python.
_PARAMS_SCHEMA: dict[str, type] = {
    "epic":                           str,
    "candle_frecuency":               str,
    "leverage":                       int,
    "lookback":                       int,
    "demo_starting_balance":          float,
    "initial_cash_balance":           float,
    "security_buffer":                float,
    "max_positions":                  int,
    "position_size":                  float,
    "min_dist_between_entries_ticks": float,
    "martingale_multiplier":          float,
    "take_profit_ticks":              float,
    "max_drawdown_pct":               float,
    "bb_period":                      int,
    "bb_dev":                         float,
    "rsi_period":                     int,
    "rsi_overbought":                 int,
    "rsi_oversold":                   int,
    "use_trend_filter":               bool,
    "atr_period":                     int,
    "atr_sl_multiplier":              float,
    "ema_period":                     int,
}


def _validate_params(data: dict, path: str) -> None:
    """Validate that all required keys are present and correctly typed.

    Collects every missing key and every type mismatch before logging them
    all at once, so a single bad file produces a complete error report.

    Args:
        data: Parsed JSON dict to validate.
        path: File path used in error messages.

    Raises:
        SystemExit: If any key is missing or has the wrong type.
    """
    errors: list[str] = []

    for key, expected in _PARAMS_SCHEMA.items():
        if key not in data:
            errors.append(f"  missing key: '{key}'")
            continue

        value = data[key]

        if expected is bool:
            if not isinstance(value, bool):
                errors.append(
                    f"  '{key}': expected bool, got {type(value).__name__} ({value!r})"
                )
        elif expected is int:
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(
                    f"  '{key}': expected int, got {type(value).__name__} ({value!r})"
                )
        elif expected is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(
                    f"  '{key}': expected float, got {type(value).__name__} ({value!r})"
                )
        elif not isinstance(value, expected):
            errors.append(
                f"  '{key}': expected {expected.__name__}, got {type(value).__name__} ({value!r})"
            )

    if errors:
        logging.critical(
            f"Parameter validation failed for {path} — {len(errors)} error(s):\n" + "\n".join(errors)
        )
        sys.exit(1)

    if not re.match(r"^\d+min$", data["candle_frecuency"]):
        logging.critical(
            f"Invalid candle_frecuency in {path} — must match '<N>min' (e.g. '15min'), got: {data['candle_frecuency']!r}"
        )
        sys.exit(1)


def load_params(path: str = "strategies/RSIBollingerStrategy.json") -> types.SimpleNamespace:
    """Load and validate strategy parameters from a JSON file.

    Reads the JSON file at ``path``, validates all required keys and their
    types, and returns the parameters as a SimpleNamespace for attribute-style
    access.

    Args:
        path: Path to the JSON parameters file. Defaults to
            ``strategies/RSIBollingerStrategy.json`` in the working directory.

    Returns:
        SimpleNamespace with one attribute per JSON key.

    Raises:
        SystemExit: If the file is missing, unreadable, contains invalid JSON,
            has missing keys, type mismatches, or an invalid ``candle_frecuency``.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        logging.critical(f"Parameters file not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.critical(f"Invalid JSON in {path}: {e}")
        sys.exit(1)
    except OSError as e:
        logging.critical(f"Could not read {path}: {e}")
        sys.exit(1)

    _validate_params(data, path)

    logging.info(f"Parameters loaded from {path}.")
    return types.SimpleNamespace(**data)


class RSIBollingerStrategy:
    """Mean-reversion grid strategy for IG Markets live trading.

    Enters long when price is below the lower Bollinger Band and RSI is
    oversold; enters short on the opposite condition. Each new position
    in the grid uses a martingale multiplier. The entire basket is closed
    when the weighted average entry price reaches the take-profit target.

    Attributes:
        params: Configuration object loaded from strategies/RSIBollingerStrategy.json.
        ig: IGClient instance used for all broker interactions.
        candles: Most-recent OHLC DataFrame with computed indicators.
        max_drawdown_reached: Flag set when equity breaches the drawdown floor.
    """

    def __init__(self, params, ig_client):
        """Initialise strategy state and load parameters.

        Args:
            params: Config object loaded from strategies/RSIBollingerStrategy.json
                (e.g. candle_frecuency, max_positions, epic).
            ig_client: Authenticated IGClient instance.
        """
        self.params = params
        self.ig = ig_client
        self.candles = pd.DataFrame()

        # Trading parameters — loaded from strategies/RSIBollingerStrategy.json at startup
        self.max_positions = params.max_positions
        self.min_dist_between_entries_ticks = params.min_dist_between_entries_ticks
        self.martingale_multiplier = params.martingale_multiplier
        self.bb_dev = params.bb_dev
        self.bb_period = params.bb_period
        self.rsi_period = params.rsi_period
        self.rsi_overbought = params.rsi_overbought
        self.rsi_oversold = params.rsi_oversold
        self.use_trend_filter = params.use_trend_filter
        self.take_profit_ticks = params.take_profit_ticks
        self.atr_period = params.atr_period
        self.atr_sl_multiplier = params.atr_sl_multiplier

        # Internal state — managed at runtime
        self.position_size = params.position_size
        self.max_drawdown_pct = params.max_drawdown_pct
        self.ema_period = params.ema_period
        self.max_drawdown_reached = False  # runtime state, not a config param

        # Safety flag — True when ig_acc_type="LIVE", False otherwise (including "DEMO")
        self.is_live_account = os.getenv("ig_acc_type") == "LIVE"
        self.demo_starting_balance = params.demo_starting_balance
        self.initial_cash_balance = params.initial_cash_balance

        self.leverage = params.leverage
        self.epic = params.epic
        self.lookback = params.lookback
        self.security_buffer = params.security_buffer

    def get_candles(self):
        """Fetch the latest candles from IG and compute all indicators.

        Retrieves OHLC data for the configured epic and resolution, then
        calculates RSI, EMA, ATR, and Bollinger Bands using TA-Lib.
        The result is stored in self.candles and also returned.

        On a successful fetch the internal cache (self.candles) is updated
        and the new DataFrame is returned. If self.ig.get_candles() returns
        None explicitly the cache is NOT updated — the existing cached value
        is returned unchanged.

        On any exception the error is logged with a full traceback and the
        existing cache is returned. If the cache is still an empty DataFrame
        (no prior successful fetch) the run() loop will skip the cycle via
        the self.candles.empty check.

        Returns:
            The current (possibly cached) candle DataFrame. May be an empty
            DataFrame when no cache exists and the fetch fails.
        """
        try:
            df = self.ig.get_candles(self.epic, "15min", self.lookback)

            if df is not None:
                df = df.copy()

                df["rsi"] = ta.RSI(df["Close"], timeperiod=self.rsi_period)
                logger.info(f"RSI calculated: period={self.rsi_period}")

                df["ema"] = ta.EMA(df["Close"], timeperiod=self.ema_period)
                logger.info(f"EMA calculated: period={self.ema_period}")

                df["atr"] = ta.ATR(
                    df["High"], df["Low"], df["Close"], timeperiod=self.atr_period
                )
                logger.info(f"ATR calculated: period={self.atr_period}")

                df["bb_upper"], df["bb_middle"], df["bb_lower"] = ta.BBANDS(
                    df["Close"],
                    timeperiod=self.bb_period,
                    nbdevup=self.bb_dev,
                    nbdevdn=self.bb_dev,
                    matype=0,
                )
                logger.info(f"BBANDS calculated: period={self.bb_period}")

                self.candles = df.tail(self.lookback)

            return self.candles
        except Exception as e:
            logger.error(f"Candle fetch failed — using cached data: {e}", exc_info=True)
            return self.candles

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
        if not account_info:
            logger.warning("Account data unavailable — skipping cycle")
            return

        open_pnl = account_info.get("profitLoss", 0.0)

        # --- 1. RISK & EQUITY (DEMO vs LIVE) ---
        if self.is_live_account:
            # LIVE mode: trust broker numbers directly
            current_equity = account_info.get("balance", 0.0) + open_pnl
            free_margin = account_info.get("available", 0.0)
        else:
            # DEMO mode: mirror realized P&L onto the simulated capital to
            # enforce strict 1:20 leverage against initial_cash_balance
            realized_profit = (
                account_info.get("balance", self.demo_starting_balance)
                - self.demo_starting_balance
            )
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
        failed = []
        for p in positions:
            deal_id = p.get("dealId", "unknown")
            close_direction = "SELL" if p["direction"] == "BUY" else "BUY"
            last_exc = None
            for attempt in range(1, 4):
                try:
                    self.ig.close_position(deal_id, close_direction, p["size"])
                    logger.info(
                        f"Closing {deal_id} ({p['direction']}) -> {close_direction}"
                        f" reason={reason}"
                    )
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
                    logger.warning(f"Close attempt {attempt}/3 failed for position {deal_id}: {e}")
                    if attempt < 3:
                        time.sleep(2 ** (attempt - 1))  # 1s, 2s

            if last_exc is not None:
                logger.error(
                    f"All 3 close attempts failed for position {deal_id}: {last_exc}",
                    exc_info=True,
                )
                failed.append(deal_id)

        if failed:
            logger.warning(f"Could not close {len(failed)} position(s) after retries: {failed}")

    def log_account_status(self):
        """Log a structured account snapshot to the configured logger.

        Reads live account data from the broker and computes equity,
        used margin, free margin, and margin level percentage. In DEMO
        mode the same virtual equity mirror used in manage_positions is
        applied so the logged numbers are consistent with trading decisions.
        """
        try:
            account_info = self.ig.get_account_summary()
            if not account_info:
                logger.warning("Account data unavailable — skipping status log")
                return

            open_pnl = account_info.get("profitLoss", 0.0)
            positions = self.ig.get_open_positions()
            n_trades = len(positions) if positions else 0

            if self.is_live_account:
                # LIVE mode: use raw broker figures
                current_equity = account_info.get("balance", 0.0) + open_pnl
                used_margin = account_info.get("margin", 0.0)
                free_margin = account_info.get("available", 0.0)
                modo = "LIVE"
            else:
                # DEMO mode: strict 1:20 virtual simulation against initial_cash_balance
                realized_profit = (
                    account_info.get("balance", self.demo_starting_balance)
                    - self.demo_starting_balance
                )
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
        logger.info(f"Strategy running. Execution every {freq} minutes.")
        next_tick = (datetime.now() + timedelta(minutes=1)).replace(second=0, microsecond=0)

        while True:
            sleep_seconds = (next_tick - datetime.now()).total_seconds()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            now = datetime.now()

            if now.minute % freq == 0:
                logger.info(f"Strategy execution at {now.strftime('%H:%M:%S')}")
                try:
                    self.get_candles()

                    if not self.candles.empty:
                        logger.info(f"Last 5 candles.\n{self.candles.tail(5).to_string()}")
                        self.manage_positions()
                        self.log_account_status()
                    else:
                        logger.info("No candles available — skipping cycle.")
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(
                        f"Trading cycle failed — will retry next tick: {e}",
                        exc_info=True,
                    )

            next_tick += timedelta(minutes=1)
