"""Event-driven bidirectional mean-reversion strategy for IG Markets live trading.

Implements independent long/short position grids using Bollinger Bands (BB) and
RSI signals. Candles are delivered via a queue from IGStreamingClient. All trading
logic runs on a single worker thread; REST calls are serialized through IGClient.

No martingale. All positions use flat contract_size. TP/SL distances are either
fixed (configured ticks) or dynamic (ATR-derived), controlled by close_mode.
"""

import json
import os
import sys
import time
import types
import threading
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import talib as ta
import numpy as np

from collections import deque

logger = logging.getLogger(__name__)


# Expected type for each V2 parameter key.
# float fields accept int values (e.g. 240 is valid for take_profit_ticks).
_PARAMS_SCHEMA: dict[str, type | tuple[type, ...]] = {
    "epic": str,
    "api_mode": str,
    "operation_mode": str,
    "candle_frequency": str,
    "bb_period": int,
    "bb_std": float,
    "rsi_period": int,
    "rsi_oversold": (
        int,
        float,
    ),  # RSI values from TA-Lib are float; allow float thresholds
    "rsi_overbought": (int, float),  # e.g. 29.5 or 70.0 are both valid
    "max_long_positions": int,
    "max_short_positions": int,
    "contract_size": float,
    "min_dist_between_entries_ticks": float,
    "take_profit_ticks": float,
    "stop_loss_ticks": (int, float),
    "close_mode": str,
    "atr_period": int,
    "atr_multiplier_tp": float,
    "atr_multiplier_sl": float,
    "enable_adx_filter": bool,
    "adx_period": int,
    "adx_threshold": (int, float),
}


def _validate_params(data: dict, path: str, *, fatal: bool = True) -> None:
    """Validate that all required V2 keys are present and correctly typed.

    Collects every missing key and every type mismatch before logging them
    all at once, so a single bad file produces a complete error report.

    Args:
        data: Parsed JSON dict to validate.
        path: File path used in error messages.
        fatal: When True (default), logs at CRITICAL and calls sys.exit(1).
            When False, logs at ERROR and raises ValueError instead.

    Raises:
        SystemExit: If fatal is True and any key is missing or has the wrong type.
        ValueError: If fatal is False and any key is missing or has the wrong type.
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
        elif isinstance(expected, tuple):
            # Multi-type schema entry (e.g. (int, float)) — value must match any allowed type
            if isinstance(value, bool) or not isinstance(value, expected):
                type_names = " | ".join(t.__name__ for t in expected)
                errors.append(
                    f"  '{key}': expected {type_names}, got {type(value).__name__} ({value!r})"
                )
        elif not isinstance(value, expected):
            errors.append(
                f"  '{key}': expected {expected.__name__}, got {type(value).__name__} ({value!r})"
            )

    # Enum validation: close_mode must be one of the accepted values
    close_mode_val = data.get("close_mode")
    if close_mode_val is not None and close_mode_val not in ("fixed", "dynamic"):
        errors.append(
            f"  'close_mode': must be 'fixed' or 'dynamic', got {close_mode_val!r}"
        )

    if errors:
        msg = (
            f"Parameter validation failed for {path} — {len(errors)} error(s):\n"
            + "\n".join(errors)
        )
        if fatal:
            logger.critical(msg)
            sys.exit(1)
        else:
            logger.error(msg)
            raise ValueError(msg)


def load_params(
    path: str = "strategies/RSIBollingerStrategyV2.json",
) -> types.SimpleNamespace:
    """Load and validate V2 strategy parameters from a JSON file.

    Reads the JSON file at ``path``, validates all required keys and their
    types, and returns the parameters as a SimpleNamespace for attribute-style
    access.

    Args:
        path: Path to the JSON parameters file. Defaults to
            ``strategies/RSIBollingerStrategyV2.json`` in the working directory.

    Returns:
        SimpleNamespace with one attribute per JSON key.

    Raises:
        SystemExit: If the file is missing, unreadable, contains invalid JSON,
            has missing keys, or type mismatches.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        logger.critical(f"Parameters file not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.critical(f"Invalid JSON in {path}: {e}")
        sys.exit(1)
    except OSError as e:
        logger.critical(f"Could not read {path}: {e}")
        sys.exit(1)

    _validate_params(data, path)

    logger.info(f"Parameters loaded from {path}.")
    return types.SimpleNamespace(**data)


class RSIBollingerStrategyV2:
    """Event-driven bidirectional mean-reversion strategy.

    Consumes closed 5-minute candles from a queue (delivered by IGStreamingClient).
    Maintains independent long and short position grids. Entry signals are
    BB + RSI crossovers; exits are handled exclusively by broker TP/SL orders
    set at position open. All positions use flat contract_size.

    Attributes:
        params: Configuration object loaded from strategies/RSIBollingerStrategyV2.json.
        ig: IGClient instance used for all broker interactions.
        streaming_client: IGStreamingClient instance (owns the Lightstreamer connection).
        trading_config: SimpleNamespace with infrastructure params (epic).
    """

    # Parameters that can be applied live without restarting the bot.
    _HOT_SAFE_PARAMS: frozenset = frozenset(
        {
            "rsi_oversold",
            "rsi_overbought",
            "max_long_positions",
            "max_short_positions",
            "min_dist_between_entries_ticks",
            "take_profit_ticks",
            "stop_loss_ticks",
            "contract_size",
            "bb_std",
            "close_mode",
            "atr_multiplier_tp",
            "atr_multiplier_sl",
            "enable_adx_filter",
            "adx_threshold",
        }
    )

    # Parameters that require a full restart to take effect.
    _RESTART_REQUIRED_PARAMS: frozenset = frozenset(
        {
            "bb_period",
            "rsi_period",
            "epic",
            "candle_frequency",
            "api_mode",
            "operation_mode",
            "atr_period",
            "adx_period",
        }
    )

    def __init__(
        self,
        params,
        ig_client,
        streaming_client,
        trading_config,
        *,
        params_path: str | None = None,
    ):
        """Initialise strategy state.

        Args:
            params: SimpleNamespace loaded from RSIBollingerStrategyV2.json.
                Required keys: epic, bb_period, bb_std, rsi_period, rsi_oversold,
                rsi_overbought, max_long_positions, max_short_positions,
                contract_size, min_dist_between_entries_ticks, take_profit_ticks.
            ig_client: Authenticated IGClient instance for REST position management.
            streaming_client: IGStreamingClient instance for candle delivery.
            trading_config: SimpleNamespace with infrastructure params sourced from
                config.json["trading"].
            params_path: Optional path to the strategy JSON file. When provided,
                the strategy checks the file's mtime at each candle and hot-applies
                safe parameter changes without restarting. When None (default),
                hot-reload is disabled.
        """
        self.params = params
        self.ig = ig_client
        self.streaming_client = streaming_client
        self.trading_config = trading_config
        self.epic = params.epic

        # Account / mode attributes (mirrors V1)
        self.is_live_account = os.getenv("ig_acc_type") == "LIVE"
        self.leverage = self.trading_config.leverage
        self.initial_cash_balance = self.trading_config.initial_cash_balance
        self.demo_starting_balance = self.trading_config.demo_starting_balance

        # operation_mode: 'candle' or 'tick'. Invalid value falls back to 'candle'.
        _raw_mode = getattr(params, "operation_mode", "candle")
        if _raw_mode not in ("candle", "tick"):
            logger.warning(
                f"Unknown operation_mode '{_raw_mode}' — defaulting to 'candle'"
            )
            _raw_mode = "candle"
        self._operation_mode: str = _raw_mode

        logger.debug(
            f"RSIBollingerStrategyV2 initialised: epic={self.epic} "
            f"operation_mode={self._operation_mode} "
            f"bb_period={params.bb_period} bb_std={params.bb_std} "
            f"rsi_period={params.rsi_period} rsi_oversold={params.rsi_oversold} "
            f"rsi_overbought={params.rsi_overbought} "
            f"max_long={params.max_long_positions} max_short={params.max_short_positions} "
            f"contract_size={params.contract_size} "
            f"min_dist={params.min_dist_between_entries_ticks} "
            f"take_profit_ticks={params.take_profit_ticks} "
            f"stop_loss_ticks={params.stop_loss_ticks}"
        )

        # Spread is calculated dynamically from each candle's OFR_CLOSE - BID_CLOSE.
        # Stored for logging/debugging; None until the first candle is processed.
        self._current_spread: float | None = None

        # Rolling window of closed candles — minimum length for indicator calculation
        # When ADX filter is enabled, ADX(period=N) needs 2*N bars to stabilize,
        # so use adx_period * 2 as the ADX term when the filter is active.
        _adx_window_term = (
            params.adx_period * 2
            if getattr(params, "enable_adx_filter", False)
            else params.adx_period
        )
        min_window = (
            max(
                params.bb_period, params.rsi_period, params.atr_period, _adx_window_term
            )
            + 1
        )
        self._candle_window: deque = deque(maxlen=min_window + 50)
        # Parallel high/low windows for ATR computation — same maxlen as _candle_window
        self._high_window: deque = deque(maxlen=min_window + 50)
        self._low_window: deque = deque(maxlen=min_window + 50)
        logger.debug(
            f"Candle window initialised: min_required={min_window} maxlen={min_window + 50}"
        )

        # Independent position grids
        # Each entry: {"deal_id": str, "entry_price": float, "size": float}
        self._long_positions: list[dict] = []
        self._short_positions: list[dict] = []

        # Hot-reload: path to the strategy JSON and its last observed mtime.
        # When _params_path is None, hot-reload is disabled.
        self._params_path: str | None = params_path
        self._params_mtime: float = (
            os.path.getmtime(params_path) if params_path else 0.0
        )

        # Stop event — set by stop() to signal run() to exit
        self._stop_event: threading.Event = threading.Event()

        # Tick mode: cached indicators (set by _on_candle, read by _on_tick).
        # None until the first candle is processed — serves as the warmup gate.
        self._cached_indicators: dict | None = None

        # Tick mode: in-flight flags prevent duplicate REST calls on back-to-back ticks.
        # Each flag is set to True immediately before a REST call and reset in a
        # finally block — so it is always False after _on_tick returns.
        self._tick_long_in_flight: bool = False
        self._tick_short_in_flight: bool = False

        # Timestamp of the last REST candle loaded during warm-up.
        # Used by _on_candle to discard overlapping streaming candles.
        # None when no warm-up has run (cold-start behavior — no dedup filtering).
        self._last_warmup_ts: datetime | None = None

        # Reconnect guard — True while a reconnect attempt is in progress.
        # Set and cleared exclusively on the worker thread; no lock required.
        self._reconnecting: bool = False

    # ---------------------------------------------------------------------- #
    # Indicator computation                                                    #
    # ---------------------------------------------------------------------- #

    def _compute_indicators(self, candle: dict) -> dict | None:
        """Add candle to the rolling windows and compute BB, RSI, and ATR.

        Args:
            candle: OHLC candle dict with keys 'close', 'high', and 'low'.

        Returns:
            Dict with keys bb_upper, bb_middle, bb_lower, rsi, close, atr when
            enough history is available; None if the window is too short.
        """
        close = candle["close"]
        self._candle_window.append(close)
        # Fall back to close when high/low are absent (e.g. minimal test candles)
        self._high_window.append(candle.get("high", close))
        self._low_window.append(candle.get("low", close))

        window_size = len(self._candle_window)
        _adx_window_term = (
            self.params.adx_period * 2
            if getattr(self.params, "enable_adx_filter", False)
            else self.params.adx_period
        )
        min_required = (
            max(
                self.params.bb_period,
                self.params.rsi_period,
                self.params.atr_period,
                _adx_window_term,
            )
            + 1
        )

        if window_size < min_required:
            logger.debug(
                f"Candle window too small ({window_size}/{min_required}) — skipping."
            )
            return None

        closes = np.array(list(self._candle_window), dtype=float)
        logger.debug(
            f"Computing indicators: window_size={window_size} "
            f"close[-1]={closes[-1]:.2f} close[-5:]={[round(c, 2) for c in closes[-5:]]}"
        )

        bb_upper, bb_middle, bb_lower = ta.BBANDS(
            closes,
            timeperiod=self.params.bb_period,
            nbdevup=self.params.bb_std,
            nbdevdn=self.params.bb_std,
            matype=0,
        )
        rsi = ta.RSI(closes, timeperiod=self.params.rsi_period)

        # ATR requires aligned high/low/close arrays. Use the minimum length so that
        # manually pre-filled _candle_window (in tests) does not cause shape mismatches.
        n = min(len(self._candle_window), len(self._high_window), len(self._low_window))
        if n > 0:
            highs = np.array(list(self._high_window)[-n:], dtype=float)
            lows = np.array(list(self._low_window)[-n:], dtype=float)
            closes_for_atr = closes[-n:]
            atr_arr = ta.ATR(
                highs, lows, closes_for_atr, timeperiod=self.params.atr_period
            )
        else:
            atr_arr = np.array([np.nan])

        if n > 0:
            adx_arr = ta.ADX(
                highs, lows, closes_for_atr, timeperiod=self.params.adx_period
            )
            current_adx = float(adx_arr[-1])
        else:
            current_adx = float("nan")

        current_close = closes[-1]
        current_bb_upper = float(bb_upper[-1])
        current_bb_middle = float(bb_middle[-1])
        current_bb_lower = float(bb_lower[-1])
        current_rsi = float(rsi[-1])
        current_atr = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0

        nan_check = [current_bb_upper, current_bb_lower, current_rsi]
        if any(np.isnan(v) for v in nan_check):
            logger.debug("NaN indicators — skipping candle.")
            return None

        logger.debug(
            f"Indicators computed: close={current_close:.2f} "
            f"BB=[{current_bb_lower:.2f}, {current_bb_middle:.2f}, {current_bb_upper:.2f}] "
            f"RSI={current_rsi:.2f} ATR={current_atr:.4f} ADX={current_adx:.2f}"
        )

        return {
            "bb_upper": current_bb_upper,
            "bb_middle": current_bb_middle,
            "bb_lower": current_bb_lower,
            "rsi": current_rsi,
            "close": current_close,
            "atr": current_atr,
            "adx": current_adx,
        }

    # ---------------------------------------------------------------------- #
    # Close parameter resolution                                              #
    # ---------------------------------------------------------------------- #

    def _get_close_params(
        self, indicators: dict
    ) -> tuple[int | float, int | float] | None:
        """Resolve TP/SL distances based on close_mode.

        In fixed mode, returns the statically configured tick distances.
        In dynamic mode, derives both distances from the current ATR value
        stored in the indicators dict.

        Args:
            indicators: Dict from _compute_indicators (or _cached_indicators in
                tick mode). Must contain an 'atr' key when close_mode is
                'dynamic'. Missing or zero/NaN ATR causes an early return of None.

        Returns:
            (limit_distance, stop_distance) as ints (rounded points), or None
            when dynamic mode cannot produce a valid distance (ATR <= 0 or NaN).
            Callers MUST skip open_position() when None is returned.
        """
        if self.params.close_mode == "dynamic":
            atr = indicators.get("atr", 0.0)
            if not atr or np.isnan(atr) or atr <= 0:
                logger.error(
                    f"ATR unavailable in dynamic mode (atr={atr!r}) — skipping open"
                )
                return None
            limit_dist = round(self.params.atr_multiplier_tp * atr)
            stop_dist = round(self.params.atr_multiplier_sl * atr)
            if stop_dist < 5:
                logger.warning(
                    f"ATR-derived stop distance {stop_dist} is below min spread guard of 5 points"
                )
            return (limit_dist, stop_dist)
        # Fixed mode — pass through configured ticks
        return (self.params.take_profit_ticks, self.params.stop_loss_ticks)

    # ---------------------------------------------------------------------- #
    # Spread update                                                            #
    # ---------------------------------------------------------------------- #

    def _update_spread_from_candle(self, candle: dict) -> None:
        """Update the current spread from a candle's spread field.

        Records the bid/ask spread (OFR_CLOSE - BID_CLOSE) from the candle
        for debug logging. Not used for broker-level exit decisions — exits
        are handled exclusively by broker TP/SL orders.

        Args:
            candle: OHLC candle dict delivered by IGStreamingClient. Must
                contain a ``spread`` key with a non-negative float value.
        """
        spread = candle.get("spread")
        if spread is not None:
            self._current_spread = float(spread)
            logger.debug(f"Spread updated from candle: {self._current_spread:.4f}")

    # ---------------------------------------------------------------------- #
    # REST warm-up                                                             #
    # ---------------------------------------------------------------------- #

    def _warmup(self) -> None:
        """Pre-fill the candle windows from REST historical data before streaming starts.

        Fetches num_candles via IGClient.get_candles() and appends each row's
        Close/High/Low directly to _candle_window/_high_window/_low_window.
        Sets _last_warmup_ts to the last REST candle's timestamp so that
        _on_candle can discard overlapping streaming candles.

        num_candles formula:
          enable_adx_filter=True:  max(bb_period, rsi_period, atr_period, adx_period * 2) + 1
          enable_adx_filter=False: max(bb_period, rsi_period, atr_period, adx_period) + 1

        The adx_period * 2 term ensures ADX is non-NaN on the first returned
        indicators dict when the filter is active.

        On failure (None response or any exception), logs a WARNING and returns early.
        The strategy then starts in cold-start mode with an empty candle window.
        """
        self.ig.clear_cache()
        logger.info("Candle cache cleared — forcing fresh historical load.")
        _adx_window_term = (
            self.params.adx_period * 2
            if getattr(self.params, "enable_adx_filter", False)
            else self.params.adx_period
        )
        num_candles = (
            max(
                self.params.bb_period,
                self.params.rsi_period,
                self.params.atr_period,
                _adx_window_term,
            )
            + 1
        )
        logger.debug(
            f"Warm-up: requesting {num_candles} candles "
            f"(bb_period={self.params.bb_period} rsi_period={self.params.rsi_period} "
            f"atr_period={self.params.atr_period} adx_period={self.params.adx_period})"
        )
        logger.info(
            f"Warm-up starting: fetching {num_candles} historical candles "
            f"({self.params.candle_frequency}) for {self.epic}."
        )
        try:
            df = self.ig.get_candles(
                self.epic, self.params.candle_frequency, num_candles, price_type="mid"
            )

            if df is None:
                logger.warning(
                    "Warm-up skipped — get_candles returned None. "
                    "Falling back to cold-start."
                )
                return

            if df.empty:
                logger.warning("Warm-up skipped — get_candles returned 0 rows.")
                return

            logger.debug(
                f"Warm-up: get_candles returned {len(df)} rows "
                f"(index range: {df.index[0]} → {df.index[-1]})"
            )

            loaded = 0
            for _, row in df.iterrows():
                self._candle_window.append(float(row["Close"]))
                self._high_window.append(float(row["High"]))
                self._low_window.append(float(row["Low"]))
                loaded += 1

            # IG REST snapshotTime is London local time (naive). Localise to
            # Europe/London then convert to UTC so dedup comparisons in
            # _on_candle work correctly against the UTC-aware streaming UTM.
            _LONDON = ZoneInfo("Europe/London")
            last_ts_naive = df.index[-1].to_pydatetime().replace(tzinfo=None)
            self._last_warmup_ts = last_ts_naive.replace(tzinfo=_LONDON).astimezone(
                timezone.utc
            )
            logger.debug(
                f"Warm-up: last REST candle ts={self._last_warmup_ts} (UTC) "
                f"window_size={len(self._candle_window)}"
            )

            if loaded < num_candles:
                logger.warning(
                    f"Warm-up partial: {loaded}/{num_candles} candles loaded."
                )
            else:
                logger.info(
                    f"Warm-up complete: {loaded} candles loaded into candle window."
                )

        except Exception as e:
            logger.warning(
                f"Warm-up skipped — exception during warm-up: {e}. "
                f"Falling back to cold-start."
            )

    # ---------------------------------------------------------------------- #
    # Reconnect handler                                                        #
    # ---------------------------------------------------------------------- #

    def _on_reconnect(self) -> None:
        """Handle a terminal Lightstreamer disconnect by re-establishing streaming.

        Called by the worker thread when a ``{"type": "reconnect"}`` sentinel is
        dequeued. Resets candle state, re-warms from REST, and calls
        ``streaming_client._restart_streaming()`` with bounded exponential backoff.

        Position grids (``_long_positions``, ``_short_positions``) are intentionally
        preserved — positions held at the broker during the disconnect window are
        still open and will be reconciled by ``_reconcile_positions()`` on the first
        post-reconnect candle.

        The ``_reconnecting`` flag prevents a second concurrent invocation if
        duplicate sentinels are dispatched before the first attempt completes.

        Backoff schedule: ``[5, 10, 20, 60]`` seconds, capped at 60s.
        Max attempts: 10 (~7-minute window). On exhaustion, logs CRITICAL and
        sets ``_stop_event`` for a clean strategy exit.
        """
        if self._reconnecting:
            logger.warning(
                "Reconnect already in progress — ignoring duplicate sentinel."
            )
            return

        if self._stop_event.is_set():
            return

        self._reconnecting = True
        try:
            delays = [5, 10, 20, 60]
            max_attempts = 10

            for attempt in range(1, max_attempts + 1):
                logger.warning(f"Reconnect attempt {attempt}/{max_attempts}...")
                try:
                    # Reset candle state — order matters: clear first so that a
                    # failed warmup does not leave stale partial data from a
                    # previous loop iteration.
                    self._last_warmup_ts = None
                    self._candle_window.clear()
                    self._high_window.clear()
                    self._low_window.clear()
                    self._cached_indicators = None

                    # Re-warm from REST — proceed even on failure (cold-start mode).
                    # Wrap separately so a warmup exception does not abort the attempt.
                    try:
                        self._warmup()
                    except Exception as warmup_err:
                        logger.warning(
                            f"Warmup failed during reconnect attempt {attempt}: {warmup_err}. "
                            "Proceeding in cold-start mode."
                        )

                    # Restart the Lightstreamer service without stopping the worker.
                    on_tick = self._on_tick if self._operation_mode == "tick" else None
                    self.streaming_client._restart_streaming(
                        self._on_candle, on_tick=on_tick
                    )

                    logger.info(f"Reconnected successfully on attempt {attempt}.")
                    return

                except (Exception, SystemExit) as e:
                    delay = delays[min(attempt - 1, len(delays) - 1)]
                    logger.warning(
                        f"Reconnect attempt {attempt} failed: {e}. "
                        f"Retrying in {delay}s."
                    )
                    if self._stop_event.wait(delay):
                        logger.info(
                            "Stop event set during reconnect backoff — aborting reconnect."
                        )
                        return

            logger.critical(
                f"Streaming reconnect failed after {max_attempts} attempts. "
                "Stopping strategy."
            )
            self._stop_event.set()

        finally:
            self._reconnecting = False

    # ---------------------------------------------------------------------- #
    # Guardrails                                                               #
    # ---------------------------------------------------------------------- #

    def _is_long_entry_allowed(self, ts: datetime | None = None) -> bool:
        """Return False on Fridays from 14:00 New York time (DST-aware).

        Args:
            ts: Timestamp to evaluate. If None, uses the current wall-clock time.
        """
        if ts is None:
            ts = datetime.now(ZoneInfo("America/New_York"))
        else:
            ts = ts.astimezone(ZoneInfo("America/New_York"))
        return not (ts.weekday() == 4 and ts.hour >= 14)

    # ---------------------------------------------------------------------- #
    # Long grid management                                                     #
    # ---------------------------------------------------------------------- #

    def _manage_longs(self, indicators: dict) -> None:
        """Evaluate long entry conditions using the latest indicators.

        Entry condition: price STRICTLY < BB_lower AND RSI STRICTLY < rsi_oversold
                         AND longs < max_long_positions
                         AND distance from last entry >= min_dist_between_entries_ticks.
            (price == BB_lower does NOT trigger entry)

        Exits are handled exclusively by broker TP/SL orders set at position open.

        Args:
            indicators: Dict with bb_upper, bb_lower, rsi, close from
                _compute_indicators.
        """
        close = indicators["close"]
        bb_upper = indicators["bb_upper"]
        bb_lower = indicators["bb_lower"]
        rsi = indicators["rsi"]

        spread = self._current_spread if self._current_spread is not None else 0.0
        logger.debug(
            f"_manage_longs: close={close:.2f} bb_lower={bb_lower:.2f} "
            f"bb_upper={bb_upper:.2f} rsi={rsi:.2f} "
            f"spread={spread:.4f} open_longs={len(self._long_positions)}"
        )

        if not self._is_long_entry_allowed():
            logger.info("[GUARD] Long entry skipped — Friday after 14:00 NY")
            return
        if close >= bb_lower or rsi >= self.params.rsi_oversold:
            logger.debug(
                f"Long entry skipped — signal not met: "
                f"close={close:.2f} bb_lower={bb_lower:.2f} "
                f"rsi={rsi:.2f} rsi_oversold={self.params.rsi_oversold}"
            )
            return
        if self.params.enable_adx_filter:
            adx = indicators.get("adx", float("nan"))
            if adx > self.params.adx_threshold:
                logger.info(
                    f"Long entry skipped — ADX filter: adx={adx:.2f} > threshold={self.params.adx_threshold:.1f}"
                )
                return
        if len(self._long_positions) >= self.params.max_long_positions:
            logger.debug(
                f"Long entry skipped — max positions reached: "
                f"{len(self._long_positions)}/{self.params.max_long_positions}"
            )
            return
        if self._long_positions:
            last_entry = self._long_positions[-1]["entry_price"]
            if not _distance_ok(
                close, last_entry, self.params.min_dist_between_entries_ticks
            ):
                logger.debug(
                    f"Long entry skipped — distance too small: "
                    f"close={close:.2f} last_entry={last_entry:.2f} "
                    f"distance={abs(close - last_entry):.2f} "
                    f"min_dist={self.params.min_dist_between_entries_ticks}"
                )
                return

        close_params = self._get_close_params(indicators)
        if close_params is None:
            return
        limit_distance, stop_distance = close_params
        size = self.params.contract_size
        logger.debug(
            f"Long entry signal: close={close:.2f} bb_lower={bb_lower:.2f} "
            f"rsi={rsi:.2f} ATR={indicators['atr']:.2f} "
            f"limit={limit_distance} stop={stop_distance} size={size} "
            f"grid_level={len(self._long_positions) + 1}/{self.params.max_long_positions}"
        )

        try:
            response = self.ig.open_position(
                epic=self.epic,
                size=size,
                side="BUY",
                limit=limit_distance,
                stop=stop_distance,
            )
            deal_id = _extract_deal_id(response)
            logger.debug(f"open_position (LONG) response: deal_id={deal_id!r}")
            if deal_id == "unknown":
                logger.critical(
                    f"LONG position opened but deal_id could not be extracted "
                    f"(response={response!r}). Skipping grid entry to prevent phantom position."
                )
            else:
                self._long_positions.append(
                    {"deal_id": deal_id, "entry_price": close, "size": size}
                )
                logger.debug(
                    f"Long grid updated: {len(self._long_positions)} position(s) open "
                    f"entries={[round(p['entry_price'], 2) for p in self._long_positions]}"
                )
                logger.info(
                    f"Opened LONG {deal_id} @ {close:.2f} | "
                    f"size={size} | ATR={indicators['atr']:.2f} | TP dist={limit_distance} | SL dist={stop_distance} | "
                    f"ADX={indicators.get('adx', float('nan')):.2f} (filter={'ON' if self.params.enable_adx_filter else 'OFF'})"
                )
        except Exception as e:
            logger.error(f"Failed to open long position: {e}")

    # ---------------------------------------------------------------------- #
    # Short grid management                                                    #
    # ---------------------------------------------------------------------- #

    def _manage_shorts(self, indicators: dict) -> None:
        """Evaluate short entry conditions using the latest indicators.

        Entry condition: price STRICTLY > BB_upper AND RSI STRICTLY > rsi_overbought
                         AND shorts < max_short_positions
                         AND distance from last entry >= min_dist_between_entries_ticks.
            (price == BB_upper does NOT trigger entry)

        Exits are handled exclusively by broker TP/SL orders set at position open.

        Args:
            indicators: Dict with bb_upper, bb_lower, rsi, close from
                _compute_indicators.
        """
        close = indicators["close"]
        bb_upper = indicators["bb_upper"]
        bb_lower = indicators["bb_lower"]
        rsi = indicators["rsi"]

        spread = self._current_spread if self._current_spread is not None else 0.0
        logger.debug(
            f"_manage_shorts: close={close:.2f} bb_lower={bb_lower:.2f} "
            f"bb_upper={bb_upper:.2f} rsi={rsi:.2f} "
            f"spread={spread:.4f} open_shorts={len(self._short_positions)}"
        )

        if close <= bb_upper or rsi <= self.params.rsi_overbought:
            logger.debug(
                f"Short entry skipped — signal not met: "
                f"close={close:.2f} bb_upper={bb_upper:.2f} "
                f"rsi={rsi:.2f} rsi_overbought={self.params.rsi_overbought}"
            )
            return
        if self.params.enable_adx_filter:
            adx = indicators.get("adx", float("nan"))
            if adx > self.params.adx_threshold:
                logger.info(
                    f"Short entry skipped — ADX filter: adx={adx:.2f} > threshold={self.params.adx_threshold:.1f}"
                )
                return
        if len(self._short_positions) >= self.params.max_short_positions:
            logger.debug(
                f"Short entry skipped — max positions reached: "
                f"{len(self._short_positions)}/{self.params.max_short_positions}"
            )
            return
        if self._short_positions:
            last_entry = self._short_positions[-1]["entry_price"]
            if not _distance_ok(
                close, last_entry, self.params.min_dist_between_entries_ticks
            ):
                logger.debug(
                    f"Short entry skipped — distance too small: "
                    f"close={close:.2f} last_entry={last_entry:.2f} "
                    f"distance={abs(close - last_entry):.2f} "
                    f"min_dist={self.params.min_dist_between_entries_ticks}"
                )
                return

        close_params = self._get_close_params(indicators)
        if close_params is None:
            return
        limit_distance, stop_distance = close_params
        size = self.params.contract_size
        logger.debug(
            f"Short entry signal: close={close:.2f} bb_upper={bb_upper:.2f} "
            f"rsi={rsi:.2f} ATR={indicators['atr']:.2f} "
            f"limit={limit_distance} stop={stop_distance} size={size} "
            f"grid_level={len(self._short_positions) + 1}/{self.params.max_short_positions}"
        )

        try:
            response = self.ig.open_position(
                epic=self.epic,
                size=size,
                side="SELL",
                limit=limit_distance,
                stop=stop_distance,
            )
            deal_id = _extract_deal_id(response)
            logger.debug(f"open_position (SHORT) response: deal_id={deal_id!r}")
            if deal_id == "unknown":
                logger.critical(
                    f"SHORT position opened but deal_id could not be extracted "
                    f"(response={response!r}). Skipping grid entry to prevent phantom position."
                )
            else:
                self._short_positions.append(
                    {"deal_id": deal_id, "entry_price": close, "size": size}
                )
                logger.debug(
                    f"Short grid updated: {len(self._short_positions)} position(s) open "
                    f"entries={[round(p['entry_price'], 2) for p in self._short_positions]}"
                )
                logger.info(
                    f"Opened SHORT {deal_id} @ {close:.2f} | "
                    f"size={size} | ATR={indicators['atr']:.2f} | TP dist={limit_distance} | SL dist={stop_distance} | "
                    f"ADX={indicators.get('adx', float('nan')):.2f} (filter={'ON' if self.params.enable_adx_filter else 'OFF'})"
                )
        except Exception as e:
            logger.error(f"Failed to open short position: {e}")

    # ---------------------------------------------------------------------- #
    # Reconciliation                                                           #
    # ---------------------------------------------------------------------- #

    def _reconcile_positions(self) -> None:
        """Reconcile local position grids against the broker's open positions.

        Runs unconditionally on every candle close. Detects positions closed by
        the broker (via TP/SL) and removes them from local grids. Also seeds any
        broker position (filtered by ``self.epic``) whose ``dealId`` is absent
        from both local grids — bidirectional sync.

        Performs two passes:

        1. **Removal pass** — removes local positions no longer present at the
           broker (phantom positions closed by TP/SL or manually). Clears the
           ``needs_reconciliation`` flag on positions that are still open.
        2. **Seed pass** — appends any broker position (filtered by
           ``self.epic``) whose ``dealId`` is absent from both local grids.
           This ensures positions opened after startup are not invisible to the
           strategy. Uses the same dict format as ``_seed_positions_from_broker``.

        Designed to be simple — one broker API call per candle, no retries, no
        partial state. If the broker call itself fails, reconciliation is skipped
        for that candle and retried on the next.
        """
        logger.debug(
            f"Reconciliation triggered: "
            f"longs={len(self._long_positions)} shorts={len(self._short_positions)}"
        )
        try:
            broker_data = self.ig.get_open_positions()
        except Exception as e:
            logger.error(f"Failed to fetch open positions for reconciliation: {e}")
            return

        # get_open_positions() returns a flat list of dicts. Each dict has a
        # top-level 'dealId' key (not nested under 'position').
        broker_deal_ids = {pos["dealId"] for pos in broker_data if "dealId" in pos}
        logger.debug(
            f"Broker reports {len(broker_deal_ids)} open position(s): {broker_deal_ids}"
        )

        def _filter(positions: list[dict]) -> list[dict]:
            kept = []
            for pos in positions:
                if pos["deal_id"] in broker_deal_ids:
                    pos.pop("needs_reconciliation", None)
                    kept.append(pos)
                else:
                    logger.warning(
                        f"Reconciliation: removing phantom position {pos['deal_id']} "
                        f"(not found at broker)"
                    )
            return kept

        self._long_positions = _filter(self._long_positions)
        self._short_positions = _filter(self._short_positions)

        # Seed pass: add broker positions that are absent from local grids.
        local_deal_ids = {
            pos["deal_id"] for pos in self._long_positions + self._short_positions
        }
        for record in broker_data:
            deal_id = record.get("dealId")
            if not deal_id:
                continue
            if record.get("epic") != self.epic:
                continue
            if deal_id in local_deal_ids:
                continue
            try:
                direction = record.get("direction")
                pos = {
                    "deal_id": deal_id,
                    "entry_price": float(record["level"]),
                    "size": float(record["size"]),
                }
                if direction == "BUY":
                    self._long_positions.append(pos)
                    local_deal_ids.add(deal_id)
                    logger.info(
                        f"Reconciliation: seeded missing LONG {deal_id} "
                        f"(entry={pos['entry_price']}, size={pos['size']})"
                    )
                elif direction == "SELL":
                    self._short_positions.append(pos)
                    local_deal_ids.add(deal_id)
                    logger.info(
                        f"Reconciliation: seeded missing SHORT {deal_id} "
                        f"(entry={pos['entry_price']}, size={pos['size']})"
                    )
                else:
                    logger.warning(
                        f"Reconciliation: unexpected direction '{direction}' for "
                        f"deal {deal_id} — position not seeded."
                    )
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(
                    f"Reconciliation: skipping malformed broker record for "
                    f"{deal_id}: {e} (record={record!r})"
                )

        logger.info("Position reconciliation complete.")

    def _needs_reconciliation(self) -> bool:
        """Return True if any local position is flagged for reconciliation."""
        return any(
            pos.get("needs_reconciliation")
            for pos in self._long_positions + self._short_positions
        )

    # ---------------------------------------------------------------------- #
    # Hot-reload                                                               #
    # ---------------------------------------------------------------------- #

    def _load_params_safe(self, path: str) -> types.SimpleNamespace | None:
        """Load and validate strategy params without exiting on failure.

        Wraps the module-level ``_validate_params`` call and catches all
        expected failure modes (bad JSON, missing file, validation errors).

        Args:
            path: Absolute path to the strategy JSON file.

        Returns:
            SimpleNamespace with all param attributes on success, None on any
            error. Logs an ERROR message before returning None.
        """
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
            logger.error(
                f"[HOT-RELOAD] Failed to reload params — keeping current: {exc}"
            )
            return None

        try:
            _validate_params(data, path, fatal=False)
        except ValueError as exc:
            logger.error(
                f"[HOT-RELOAD] Failed to reload params — keeping current: {exc}"
            )
            return None

        return types.SimpleNamespace(**data)

    def _reload_params_if_changed(self) -> None:
        """Check params file mtime and hot-apply safe param changes.

        Called as the first logic in ``_on_candle`` after the malformed-candle
        guard. No-ops immediately when ``_params_path`` is None (reload disabled)
        or when the file mtime has not changed since the last successful reload.

        Hot-safe params are applied via individual ``setattr`` on ``self.params``.
        Restart-required params are discarded with a WARNING log. On bad JSON or
        validation failure, ``self.params`` is unchanged.
        """
        if self._params_path is None:
            return

        try:
            current_mtime = os.path.getmtime(self._params_path)
        except OSError:
            return  # file temporarily unavailable — skip silently

        if current_mtime == self._params_mtime:
            return

        logger.info("[HOT-RELOAD] Strategy params file changed — reloading")

        new_params = self._load_params_safe(self._params_path)
        if new_params is None:
            # mtime already changed; don't update _params_mtime so next candle retries
            return

        applied = 0
        discarded = 0

        for key in self._HOT_SAFE_PARAMS:
            old_val = getattr(self.params, key, None)
            new_val = getattr(new_params, key, None)
            if old_val != new_val:
                logger.info("[HOT-RELOAD] %s: %s → %s", key, old_val, new_val)
                setattr(self.params, key, new_val)
                applied += 1

        for key in self._RESTART_REQUIRED_PARAMS:
            old_val = getattr(self.params, key, None)
            new_val = getattr(new_params, key, None)
            if old_val != new_val:
                logger.warning(
                    "[HOT-RELOAD] %s changed but requires restart — keeping %s",
                    key,
                    old_val,
                )
                discarded += 1

        if applied > 0 or discarded > 0:
            logger.info(
                "[HOT-RELOAD] Applied %d param(s), discarded %d (restart required)",
                applied,
                discarded,
            )

        self._params_mtime = current_mtime

    # ---------------------------------------------------------------------- #
    # Candle handler                                                           #
    # ---------------------------------------------------------------------- #

    _REQUIRED_CANDLE_KEYS = ("close", "bid_close", "ofr_close", "spread")

    def _on_candle(self, candle: dict) -> None:
        """Process a single closed candle: update spread, compute indicators, manage grids.

        Called from the worker thread after dequeuing a candle from the
        streaming client's delivery mechanism. The candle's spread field
        (OFR_CLOSE - BID_CLOSE) is recorded for debug logging.

        Malformed candles (missing required keys) are logged and skipped
        explicitly rather than propagating a KeyError.

        Args:
            candle: OHLC dict delivered by IGStreamingClient callback.
                Required keys: close, bid_close, ofr_close, spread.
        """
        logger.debug(
            f"_on_candle received: ts={candle.get('timestamp')} "
            f"close={candle.get('close')} spread={candle.get('spread')}"
        )
        missing = [k for k in self._REQUIRED_CANDLE_KEYS if k not in candle]
        if missing:
            logger.warning(
                f"Skipping malformed candle — missing required keys: {missing}"
            )
            return

        if self._last_warmup_ts is not None and candle.get("timestamp") is not None:
            candle_ts = candle["timestamp"]
            warmup_ts = self._last_warmup_ts
            # Normalize both to naive UTC for safe comparison — both sources use UTC
            if candle_ts.tzinfo is not None:
                candle_ts = candle_ts.astimezone(timezone.utc).replace(tzinfo=None)
            if warmup_ts.tzinfo is not None:
                warmup_ts = warmup_ts.astimezone(timezone.utc).replace(tzinfo=None)
            if candle_ts <= warmup_ts:
                logger.debug(
                    f"Skipping duplicate candle (warmup overlap): ts={candle['timestamp']}"
                )
                return

        self._reload_params_if_changed()

        self._reconcile_positions()

        self._update_spread_from_candle(candle)
        indicators = self._compute_indicators(candle)
        if indicators is None:
            logger.debug(
                f"Indicators not yet available — window_size={len(self._candle_window)}"
            )
            return

        logger.info(
            f"Candle processed: close={indicators['close']:.2f} "
            f"BB=[{indicators['bb_lower']:.2f}, {indicators['bb_upper']:.2f}] "
            f"RSI={indicators['rsi']:.2f} "
            f"ATR={indicators['atr']:.2f} "
            f"ADX={indicators.get('adx', float('nan')):.2f} (filter={'ON' if self.params.enable_adx_filter else 'OFF'}) "
            f"longs={len(self._long_positions)} shorts={len(self._short_positions)}"
        )

        # In tick mode, cache the indicators so _on_tick can evaluate signals
        # from live bid/ofr prices, then return early — tick handler owns entries.
        if self._operation_mode == "tick":
            self._cached_indicators = indicators
            self.log_account_status()
            return

        self._manage_longs(indicators)
        self._manage_shorts(indicators)

        # Log account status AFTER trade decisions so the log reflects post-decision state.
        self.log_account_status()

    # ---------------------------------------------------------------------- #
    # Account status logging                                                   #
    # ---------------------------------------------------------------------- #

    def log_account_status(self) -> None:
        """Log account health, equity, margin, and per-grid position breakdown.

        Reads ``balance`` and ``profitLoss`` from get_account_summary().
        Calls get_open_positions() and filters to positions matching ``self.epic``
        for margin calculations.

        Computes margin level % and classifies health:
          - used_margin == 0 → [IDLE]
          - margin_level < 120% → [DANGER]
          - margin_level < 200% → [ALERT]
          - margin_level >= 200% → [HEALTHY]

        ``used_margin`` is derived from broker positions filtered to ``self.epic``
        and divided by ``self.leverage``.

        LIVE mode: equity = balance + profitLoss.
        DEMO (virtual) mode: equity is computed from the virtual starting balance
        plus realized profit and open P&L; used_margin and free_margin are
        derived from epic-filtered broker positions and the configured leverage.

        Per-grid avg entry: size-weighted average of entry_price from the local
        long and short position grids. Reports 'N/A' when a grid is empty.

        Emits a single INFO log line starting with STATUS. Never raises — full
        try/except with logger.error on failure.
        """
        try:
            account_info = self.ig.get_account_summary()
            if not account_info:
                logger.warning(
                    "Account data unavailable — skipping account status log."
                )
                return

            raw_positions = self.ig.get_open_positions()
            positions = (
                [p for p in raw_positions if p.get("epic") == self.epic]
                if raw_positions
                else []
            )

            if self.is_live_account:
                open_pnl = account_info.get("profitLoss", 0.0)
                current_equity = account_info.get("balance", 0.0) + open_pnl
                used_margin = (
                    sum(
                        p.get("size", 0.0) * p.get("level", 0.0) / self.leverage
                        for p in positions
                    )
                    if positions
                    else 0.0
                )
                free_margin = current_equity - used_margin
                mode = "LIVE"
            else:
                open_pnl = account_info.get("profitLoss", 0.0)
                realized_profit = (
                    account_info.get("balance", self.demo_starting_balance)
                    - self.demo_starting_balance
                )
                virtual_balance = self.initial_cash_balance + realized_profit
                current_equity = virtual_balance + open_pnl
                used_margin = (
                    sum(
                        p.get("size", 0.0) * p.get("level", 0.0) / self.leverage
                        for p in positions
                    )
                    if positions
                    else 0.0
                )
                free_margin = current_equity - used_margin
                mode = f"VIRTUAL (1:{self.leverage})"

            if used_margin == 0:
                health_label = "[IDLE]"
                margin_level_str = "N/A"
            else:
                margin_level_pct = (current_equity / used_margin) * 100
                margin_level_str = f"{margin_level_pct:.1f}%"
                if margin_level_pct < 120:
                    health_label = "[DANGER]"
                elif margin_level_pct < 200:
                    health_label = "[ALERT]"
                else:
                    health_label = "[HEALTHY]"

            n_longs = len(self._long_positions)
            n_shorts = len(self._short_positions)
            n_trades = n_longs + n_shorts
            long_avg = _avg_entry(self._long_positions)
            short_avg = _avg_entry(self._short_positions)

            logger.info(
                f"STATUS | Mode: {mode} | Equity: ${current_equity:.2f} | "
                f"Used Margin: ${used_margin:.2f} | "
                f"Margin Level: {margin_level_str} {health_label} | "
                f"Free: ${free_margin:.2f} | "
                f"Longs: {n_longs} (avg: {long_avg}) | "
                f"Shorts: {n_shorts} (avg: {short_avg}) | "
                f"Total: {n_trades}"
            )
        except Exception as e:
            logger.error(f"log_account_status failed: {e}")

    # ---------------------------------------------------------------------- #
    # Tick handler                                                             #
    # ---------------------------------------------------------------------- #

    def _tick_try_open(self, side: str, bid: float, spread: float) -> None:
        """Attempt to open a position via REST in tick mode.

        Sets the in-flight flag before the REST call and resets it in a
        finally block — guaranteeing the flag is always False after this
        method returns, even when the REST call raises.

        Opens a position via the broker and records ``entry_spread`` (for
        LONG/BUY positions) for informational purposes only — it is not
        used for exit decisions (exits are handled by broker TP/SL orders).

        Args:
            side: Trade direction — 'BUY' (long) or 'SELL' (short).
            bid: Current bid price used as the entry reference price.
            spread: Live spread (ofr - bid) at the open tick. Recorded for
                informational logging.
        """
        if side == "BUY":
            self._tick_long_in_flight = True
            positions = self._long_positions
        else:
            self._tick_short_in_flight = True
            positions = self._short_positions

        # Resolve TP/SL from cached indicators (set by last _on_candle)
        close_params = self._get_close_params(self._cached_indicators or {})
        if close_params is None:
            # ATR unavailable in dynamic mode — skip open
            if side == "BUY":
                self._tick_long_in_flight = False
            else:
                self._tick_short_in_flight = False
            return
        limit, stop = close_params

        try:
            response = self.ig.open_position(
                epic=self.epic,
                size=self.params.contract_size,
                side=side,
                limit=limit,
                stop=stop,
            )
            deal_id = _extract_deal_id(response)
            if deal_id != "unknown":
                pos = {
                    "deal_id": deal_id,
                    "entry_price": bid,
                    "size": self.params.contract_size,
                }
                if side == "BUY":
                    # Record spread at open for informational logging.
                    pos["entry_spread"] = spread
                positions.append(pos)
                logger.info(
                    f"Tick: opened {side} {deal_id} @ bid={bid:.2f} "
                    f"ATR={self._cached_indicators.get('atr', 0.0):.2f} "
                    f"limit={limit} stop={stop}"
                )
        except Exception as e:
            logger.error(f"Tick: failed to open {side}: {e}")
        finally:
            if side == "BUY":
                self._tick_long_in_flight = False
            else:
                self._tick_short_in_flight = False

    def _on_tick(self, tick: dict) -> None:
        """Evaluate entry signals from a live tick in tick mode.

        Called from the worker thread via the dispatcher. Reads cached indicators
        (set by the last _on_candle call) and compares live bid/ofr prices to the
        Bollinger Band levels. Delegates REST open calls to _tick_try_open().

        Warmup gate: silently returns when _cached_indicators is None (no candle
        has been processed yet).

        Args:
            tick: Dict with keys 'bid' (float), 'ofr' (float), 'utm' (int).
        """
        indicators = self._cached_indicators
        if indicators is None:
            return  # warmup gate — no candle processed yet

        if self._needs_reconciliation():
            logger.debug("tick: reconciliation triggered")
            self._reconcile_positions()

        bid = tick["bid"]
        if bid <= 0:
            return  # stale/heartbeat tick with no real price — skip all signal evaluation
        spread = tick["ofr"] - bid  # live spread from the tick itself
        bb_upper = indicators["bb_upper"]
        bb_lower = indicators["bb_lower"]
        rsi = indicators["rsi"]

        logger.debug(
            "tick bid=%.5f ask=%.5f spread=%.5f | bb_lower=%.5f bb_upper=%.5f rsi=%.2f"
            " | longs=%d shorts=%d",
            bid,
            tick["ofr"],
            spread,
            bb_lower,
            bb_upper,
            rsi,
            len(self._long_positions),
            len(self._short_positions),
        )

        # --- Long entry ---
        if not self._is_long_entry_allowed():
            logger.debug("tick long_entry: skipped (guardrail) Friday after 14:00 NY")
        elif (
            bid < bb_lower
            and rsi < self.params.rsi_oversold
            and not self._tick_long_in_flight
            and len(self._long_positions) < self.params.max_long_positions
        ):
            _long_dist_ok = True
            if self._long_positions:
                last_entry = self._long_positions[-1]["entry_price"]
                if not _distance_ok(
                    bid, last_entry, self.params.min_dist_between_entries_ticks
                ):
                    _long_dist_ok = False
                    logger.debug(
                        "tick long_entry: skipped (distance guard) bid=%.5f last_entry=%.5f",
                        bid,
                        last_entry,
                    )
            if _long_dist_ok and self.params.enable_adx_filter:
                _adx = self._cached_indicators.get("adx", float("nan"))
                if _adx > self.params.adx_threshold:
                    logger.info("tick long_entry: skipped (ADX filter) adx=%.2f", _adx)
                    _long_dist_ok = False
            if _long_dist_ok:
                logger.debug(
                    "tick long_entry: triggered bid=%.5f < bb_lower=%.5f rsi=%.2f",
                    bid,
                    bb_lower,
                    rsi,
                )
                self._tick_try_open("BUY", bid, spread)

        # --- Short entry ---
        if (
            bid > bb_upper
            and rsi > self.params.rsi_overbought
            and not self._tick_short_in_flight
            and len(self._short_positions) < self.params.max_short_positions
        ):
            _short_dist_ok = True
            if self._short_positions:
                last_entry = self._short_positions[-1]["entry_price"]
                if not _distance_ok(
                    bid, last_entry, self.params.min_dist_between_entries_ticks
                ):
                    _short_dist_ok = False
                    logger.debug(
                        "tick short_entry: skipped (distance guard) bid=%.5f last_entry=%.5f",
                        bid,
                        last_entry,
                    )
            if _short_dist_ok and self.params.enable_adx_filter:
                _adx = self._cached_indicators.get("adx", float("nan"))
                if _adx > self.params.adx_threshold:
                    logger.info("tick short_entry: skipped (ADX filter) adx=%.2f", _adx)
                    _short_dist_ok = False
            if _short_dist_ok:
                logger.debug(
                    "tick short_entry: triggered bid=%.5f > bb_upper=%.5f rsi=%.2f",
                    bid,
                    bb_upper,
                    rsi,
                )
                self._tick_try_open("SELL", bid, spread)

    # ---------------------------------------------------------------------- #
    # Main loop                                                                #
    # ---------------------------------------------------------------------- #

    def _seed_positions_from_broker(self) -> None:
        """Seed local position grids from broker open positions at startup.

        Fetches open positions via ig.get_open_positions(), filters by
        self.epic, classifies by direction (BUY -> long, SELL -> short),
        and populates _long_positions / _short_positions.

        On failure, logs WARNING and returns — grids remain empty (same
        as current behavior). No entry_spread is set on seeded positions.
        """
        try:
            positions = self.ig.get_open_positions()

            # Filter and sort by createdDate ascending so that the last element
            # in each grid is the most recently opened position — matching the
            # runtime invariant relied on by the min_dist guard.
            matching = [r for r in positions if r.get("epic") == self.epic]

            def _parse_created_date(record):
                raw = record.get("createdDate", "")
                if not raw:
                    return datetime.min
                try:
                    # IG Markets format: "2026/05/29 10:00:00:000"
                    return datetime.strptime(raw, "%Y/%m/%d %H:%M:%S:%f")
                except (ValueError, TypeError):
                    pass
                try:
                    # ISO 8601 fallback: "2026-05-29T10:00:00"
                    return datetime.fromisoformat(raw)
                except (ValueError, TypeError):
                    logger.warning(
                        f"Could not parse createdDate '{raw}' for deal "
                        f"{record.get('dealId', 'unknown')} — using datetime.min"
                    )
                    return datetime.min

            matching.sort(key=_parse_created_date)

            longs_seeded = 0
            shorts_seeded = 0
            for record in matching:
                try:
                    direction = record.get("direction")
                    pos = {
                        "deal_id": record["dealId"],
                        "entry_price": float(record["level"]),
                        "size": float(record["size"]),
                    }
                    if direction == "BUY":
                        self._long_positions.append(pos)
                        longs_seeded += 1
                    elif direction == "SELL":
                        self._short_positions.append(pos)
                        shorts_seeded += 1
                    else:
                        logger.warning(
                            f"Unexpected direction '{direction}' for deal "
                            f"{record.get('dealId', 'unknown')} — position not "
                            f"added to any grid."
                        )
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(
                        f"Skipping malformed position record during seeding: {e} "
                        f"(record={record!r})"
                    )

            if longs_seeded > self.params.max_long_positions:
                logger.warning(
                    f"Seeded {longs_seeded} long(s) exceeds max_long_positions "
                    f"({self.params.max_long_positions}) — operator review recommended."
                )
            if shorts_seeded > self.params.max_short_positions:
                logger.warning(
                    f"Seeded {shorts_seeded} short(s) exceeds max_short_positions "
                    f"({self.params.max_short_positions}) — operator review recommended."
                )

            logger.info(
                f"Seeded {longs_seeded} long(s) and {shorts_seeded} short(s) "
                f"from broker state for {self.epic}."
            )
        except Exception as e:
            logger.warning(
                f"_seed_positions_from_broker failed — starting with empty grids: {e}"
            )

    def run(self) -> None:
        """Start streaming and block until stop() is called.

        Registers _on_candle as the callback directly with the streaming client
        and calls start(). The IGStreamingClient's own worker thread delivers
        candles to _on_candle — no intermediate re-queuing in this class.
        run() then blocks on _stop_event until stop() is called.
        """
        logger.info("RSIBollingerStrategyV2 starting.")
        self._stop_event.clear()
        logger.debug("Stop event cleared — entering warm-up phase.")
        self._warmup()
        logger.debug("Warm-up complete — seeding positions from broker state.")
        self._seed_positions_from_broker()
        logger.debug(
            f"Seeding complete — starting streaming client (window_size={len(self._candle_window)})."
        )
        on_tick = self._on_tick if self._operation_mode == "tick" else None
        self.streaming_client.start(
            self._on_candle, on_tick=on_tick, on_reconnect=self._on_reconnect
        )
        logger.debug("Streaming client started — blocking on stop event.")
        self._stop_event.wait()
        logger.info("RSIBollingerStrategyV2 stopped.")

    def stop(self) -> None:
        """Signal the run() loop to exit and halt candle delivery immediately.

        Calls streaming_client.stop() to disconnect the Lightstreamer session
        so no new candles arrive after stop() returns. Then sets _stop_event
        to unblock run().
        """
        logger.debug("stop() called — halting streaming client and setting stop event.")
        self.streaming_client.stop()
        self._stop_event.set()
        logger.debug("Stop event set — run() will unblock.")


# --------------------------------------------------------------------------- #
# Utilities                                                                    #
# --------------------------------------------------------------------------- #


def _avg_entry(positions: list[dict]) -> str:
    """Compute the size-weighted average entry price for a position grid.

    Args:
        positions: List of position dicts, each with 'entry_price' and 'size'.
            Reads the local grid list (not broker positions) so that the
            avg reflects the actual entry prices recorded at order time.

    Returns:
        Size-weighted average formatted to two decimal places, or 'N/A'
        when the list is empty or total size sums to zero.
    """
    if not positions:
        return "N/A"
    total_size = sum(p["size"] for p in positions)
    if total_size == 0:
        return "N/A"
    weighted_sum = sum(p["size"] * p["entry_price"] for p in positions)
    return f"{weighted_sum / total_size:.2f}"


def _distance_ok(close: float, last_entry: float, min_ticks: float) -> bool:
    """Return True when the distance between close and last entry meets the minimum.

    Args:
        close: Current close price.
        last_entry: Entry price of the most recent grid position.
        min_ticks: Minimum required distance in ticks (points).

    Returns:
        True when abs(close - last_entry) >= min_ticks.
    """
    return abs(close - last_entry) >= min_ticks


def _extract_deal_id(response) -> str:
    """Extract deal_id from an IG open_position confirms response.

    The IG confirms endpoint returns a JSON body that always contains a
    ``dealReference`` (the key used to query confirms) and a ``dealStatus``
    that indicates whether the deal was actually executed. Only a response
    with ``dealStatus == "ACCEPTED"`` represents a live broker position.

    The stable broker position identifier is ``dealId`` — this is what
    close_position and reconciliation use. ``dealReference`` is ephemeral
    and must NOT be stored as the grid position ID.

    Args:
        response: Return value of IGClient.open_position — the confirms dict
            returned by trading_ig's fetch_deal_by_deal_reference(), or any
            object with a .get() method (e.g. MagicMock in tests).

    Returns:
        The ``dealId`` string when ``dealStatus == "ACCEPTED"`` and ``dealId``
        is non-empty. Returns ``'unknown'`` in all other cases:
        - ``dealStatus`` is absent or not ``"ACCEPTED"`` (rejected deal)
        - ``dealId`` is absent or empty
        - response has no ``.get()`` method
        - any unexpected exception
    """
    try:
        if not hasattr(response, "get"):
            return "unknown"
        if response.get("dealStatus") != "ACCEPTED":
            return "unknown"
        return response.get("dealId", "unknown") or "unknown"
    except Exception:
        return "unknown"
