"""Event-driven bidirectional mean-reversion strategy for IG Markets live trading.

Implements independent long/short position grids using Bollinger Bands (BB) and
RSI signals. Candles are delivered via a queue from IGStreamingClient. All trading
logic runs on a single worker thread; REST calls are serialized through IGClient.

No martingale, no ATR, no stop-loss. All positions use flat contract_size.
"""

import json
import os
import sys
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
    BB + RSI crossovers; exits are opposite-band crossovers with per-position
    spread-aware profit check. All positions use flat contract_size.

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
            "contract_size",
            "bb_std",
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
            f"take_profit_ticks={params.take_profit_ticks}"
        )

        # Spread is calculated dynamically from each candle's OFR_CLOSE - BID_CLOSE.
        # None until the first candle is processed; profit checks use 0.0 as fallback.
        self._current_spread: float | None = None

        # Rolling window of closed candles — minimum length for indicator calculation
        # Needs bb_period + rsi_period candles for both indicators to be valid
        min_window = max(params.bb_period, params.rsi_period) + 1
        self._candle_window: deque = deque(maxlen=min_window + 50)
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
        self._tick_long_close_in_flight: bool = False
        self._tick_short_close_in_flight: bool = False

        # Timestamp of the last REST candle loaded during warm-up.
        # Used by _on_candle to discard overlapping streaming candles.
        # None when no warm-up has run (cold-start behavior — no dedup filtering).
        self._last_warmup_ts: datetime | None = None

    # ---------------------------------------------------------------------- #
    # Indicator computation                                                    #
    # ---------------------------------------------------------------------- #

    def _compute_indicators(self, candle: dict) -> dict | None:
        """Add candle to the rolling window and compute BB and RSI.

        Args:
            candle: OHLC candle dict with at least a 'close' key.

        Returns:
            Dict with keys bb_upper, bb_middle, bb_lower, rsi, close when
            enough history is available; None if the window is too short.
        """
        self._candle_window.append(candle["close"])

        window_size = len(self._candle_window)
        min_required = max(self.params.bb_period, self.params.rsi_period) + 1

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

        current_close = closes[-1]
        current_bb_upper = float(bb_upper[-1])
        current_bb_middle = float(bb_middle[-1])
        current_bb_lower = float(bb_lower[-1])
        current_rsi = float(rsi[-1])

        if any(np.isnan(v) for v in [current_bb_upper, current_bb_lower, current_rsi]):
            logger.debug("NaN indicators — skipping candle.")
            return None

        logger.debug(
            f"Indicators computed: close={current_close:.2f} "
            f"BB=[{current_bb_lower:.2f}, {current_bb_middle:.2f}, {current_bb_upper:.2f}] "
            f"RSI={current_rsi:.2f}"
        )

        return {
            "bb_upper": current_bb_upper,
            "bb_middle": current_bb_middle,
            "bb_lower": current_bb_lower,
            "rsi": current_rsi,
            "close": current_close,
        }

    # ---------------------------------------------------------------------- #
    # Spread update                                                            #
    # ---------------------------------------------------------------------- #

    def _update_spread_from_candle(self, candle: dict) -> None:
        """Update the current spread from a candle's spread field.

        The candle's ``spread`` key (OFR_CLOSE - BID_CLOSE) represents the
        live market spread at candle close. Calling this before evaluating
        exit conditions ensures profit calculations use the most recent spread.

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
        """Pre-fill the candle window from REST historical data before streaming starts.

        Fetches max(bb_period, rsi_period) + 1 candles via IGClient.get_candles()
        and appends each row's Close price directly to _candle_window. Sets
        _last_warmup_ts to the last REST candle's timestamp so that _on_candle
        can discard overlapping streaming candles.

        On failure (None response or any exception), logs a WARNING and returns early.
        The strategy then starts in cold-start mode with an empty candle window.
        """
        num_candles = max(self.params.bb_period, self.params.rsi_period) + 1
        logger.debug(
            f"Warm-up: requesting {num_candles} candles "
            f"(bb_period={self.params.bb_period} rsi_period={self.params.rsi_period})"
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
        """Evaluate long exits then long entries using the latest indicators.

        Exit condition: price STRICTLY > BB_upper AND profit after spread > 0.
            (price == BB_upper does NOT trigger exit)
        Entry condition: price STRICTLY < BB_lower AND RSI STRICTLY < rsi_oversold
                         AND longs < max_long_positions
                         AND distance from last entry >= min_dist_between_entries_ticks.
            (price == BB_lower does NOT trigger entry)

        Args:
            indicators: Dict with bb_upper, bb_lower, rsi, close from
                _compute_indicators.
        """
        close = indicators["close"]
        bb_upper = indicators["bb_upper"]
        bb_lower = indicators["bb_lower"]
        rsi = indicators["rsi"]

        # --- EXITS ---
        # Exit condition: price STRICTLY > bb_upper AND profit after spread > 0.
        spread = self._current_spread if self._current_spread is not None else 0.0
        logger.debug(
            f"_manage_longs: close={close:.2f} bb_lower={bb_lower:.2f} "
            f"bb_upper={bb_upper:.2f} rsi={rsi:.2f} "
            f"spread={spread:.4f} open_longs={len(self._long_positions)}"
        )
        if close > bb_upper and self._long_positions:
            logger.debug(
                f"Long exit condition met: close={close:.2f} > bb_upper={bb_upper:.2f} "
                f"evaluating {len(self._long_positions)} position(s)"
            )
            to_close = []  # list of (pos, profit) tuples — profit computed once
            to_keep = []
            for pos in self._long_positions:
                profit = _long_profit(close, pos["entry_price"], spread, pos["size"])
                logger.debug(
                    f"Long exit eval: deal_id={pos['deal_id']} "
                    f"entry={pos['entry_price']:.2f} close={close:.2f} "
                    f"spread={spread:.4f} size={pos['size']} profit={profit:.2f}"
                )
                if profit > 0:
                    to_close.append((pos, profit))
                else:
                    logger.debug(
                        f"Long {pos['deal_id']} not profitable after spread — keeping"
                    )
                    to_keep.append(pos)

            for pos, profit in to_close:
                closed_ok = False
                try:
                    self.ig.close_position(pos["deal_id"], "SELL", pos["size"])
                    closed_ok = True
                except Exception as e:
                    logger.error(f"Failed to close long {pos['deal_id']}: {e}")
                    pos["needs_reconciliation"] = True
                    to_keep.append(pos)

                if closed_ok:
                    logger.info(
                        f"Closed LONG {pos['deal_id']} @ {close:.2f} "
                        f"(entry={pos['entry_price']:.2f}, profit={profit:.2f})"
                    )

            self._long_positions = to_keep

        # --- ENTRIES ---
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

        limit_distance = self.params.take_profit_ticks
        size = self.params.contract_size
        logger.debug(
            f"Long entry signal: close={close:.2f} bb_lower={bb_lower:.2f} "
            f"rsi={rsi:.2f} size={size} tp_dist={limit_distance} "
            f"grid_level={len(self._long_positions) + 1}/{self.params.max_long_positions}"
        )

        try:
            response = self.ig.open_position(
                epic=self.epic,
                size=size,
                side="BUY",
                limit=limit_distance,
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
                    f"size={size} | TP dist={limit_distance}"
                )
        except Exception as e:
            logger.error(f"Failed to open long position: {e}")

    # ---------------------------------------------------------------------- #
    # Short grid management                                                    #
    # ---------------------------------------------------------------------- #

    def _manage_shorts(self, indicators: dict) -> None:
        """Evaluate short exits then short entries using the latest indicators.

        Exit condition: price STRICTLY < BB_lower AND profit after spread > 0.
            (price == BB_lower does NOT trigger exit)
        Entry condition: price STRICTLY > BB_upper AND RSI STRICTLY > rsi_overbought
                         AND shorts < max_short_positions
                         AND distance from last entry >= min_dist_between_entries_ticks.
            (price == BB_upper does NOT trigger entry)

        Args:
            indicators: Dict with bb_upper, bb_lower, rsi, close from
                _compute_indicators.
        """
        close = indicators["close"]
        bb_upper = indicators["bb_upper"]
        bb_lower = indicators["bb_lower"]
        rsi = indicators["rsi"]

        # --- EXITS ---
        # Exit condition: price STRICTLY < bb_lower AND profit after spread > 0.
        spread = self._current_spread if self._current_spread is not None else 0.0
        logger.debug(
            f"_manage_shorts: close={close:.2f} bb_lower={bb_lower:.2f} "
            f"bb_upper={bb_upper:.2f} rsi={rsi:.2f} "
            f"spread={spread:.4f} open_shorts={len(self._short_positions)}"
        )
        if close < bb_lower and self._short_positions:
            logger.debug(
                f"Short exit condition met: close={close:.2f} < bb_lower={bb_lower:.2f} "
                f"evaluating {len(self._short_positions)} position(s)"
            )
            to_close = []  # list of (pos, profit) tuples — profit computed once
            to_keep = []
            for pos in self._short_positions:
                profit = _short_profit(close, pos["entry_price"], spread, pos["size"])
                logger.debug(
                    f"Short exit eval: deal_id={pos['deal_id']} "
                    f"entry={pos['entry_price']:.2f} close={close:.2f} "
                    f"spread={spread:.4f} size={pos['size']} profit={profit:.2f}"
                )
                if profit > 0:
                    to_close.append((pos, profit))
                else:
                    logger.debug(
                        f"Short {pos['deal_id']} not profitable after spread — keeping"
                    )
                    to_keep.append(pos)

            for pos, profit in to_close:
                closed_ok = False
                try:
                    self.ig.close_position(pos["deal_id"], "BUY", pos["size"])
                    closed_ok = True
                except Exception as e:
                    logger.error(f"Failed to close short {pos['deal_id']}: {e}")
                    pos["needs_reconciliation"] = True
                    to_keep.append(pos)

                if closed_ok:
                    logger.info(
                        f"Closed SHORT {pos['deal_id']} @ {close:.2f} "
                        f"(entry={pos['entry_price']:.2f}, profit={profit:.2f})"
                    )

            self._short_positions = to_keep

        # --- ENTRIES ---
        if close <= bb_upper or rsi <= self.params.rsi_overbought:
            logger.debug(
                f"Short entry skipped — signal not met: "
                f"close={close:.2f} bb_upper={bb_upper:.2f} "
                f"rsi={rsi:.2f} rsi_overbought={self.params.rsi_overbought}"
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

        limit_distance = self.params.take_profit_ticks
        size = self.params.contract_size
        logger.debug(
            f"Short entry signal: close={close:.2f} bb_upper={bb_upper:.2f} "
            f"rsi={rsi:.2f} size={size} tp_dist={limit_distance} "
            f"grid_level={len(self._short_positions) + 1}/{self.params.max_short_positions}"
        )

        try:
            response = self.ig.open_position(
                epic=self.epic,
                size=size,
                side="SELL",
                limit=limit_distance,
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
                    f"size={size} | TP dist={limit_distance}"
                )
        except Exception as e:
            logger.error(f"Failed to open short position: {e}")

    # ---------------------------------------------------------------------- #
    # Reconciliation                                                           #
    # ---------------------------------------------------------------------- #

    def _reconcile_positions(self) -> None:
        """Reconcile local position grids against the broker's open positions.

        Called before entry evaluation whenever any local position is flagged
        ``needs_reconciliation=True`` (set after a failed close_position call).
        Fetches the current open positions from the broker and removes any local
        position that is no longer present at the broker. Positions that are
        still open at the broker have their flag cleared.

        Designed to be simple — one broker API call per reconciliation trigger,
        no retries, no partial state. If the broker call itself fails, the
        positions remain flagged and reconciliation is retried on the next candle.
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
        (OFR_CLOSE - BID_CLOSE) is recorded first so that exit profit checks
        always use the most recent live spread.

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

        logger.debug(
            f"Candle processed: close={indicators['close']:.2f} "
            f"BB=[{indicators['bb_lower']:.2f}, {indicators['bb_upper']:.2f}] "
            f"RSI={indicators['rsi']:.2f} "
            f"longs={len(self._long_positions)} shorts={len(self._short_positions)}"
        )

        # In tick mode, cache the indicators so _on_tick can evaluate signals
        # from live bid/ofr prices, then return early — tick handler owns entries/exits.
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

        For LONG (BUY) positions, ``spread`` is stored as ``entry_spread`` in
        the position dict so that profit checks at exit time use the spread that
        was active at open (i.e. the actual ask cost), not the potentially
        narrower spread at the exit tick.

        Args:
            side: Trade direction — 'BUY' (long) or 'SELL' (short).
            bid: Current bid price used as the entry reference price.
            spread: Live spread (ofr - bid) at the open tick. Stored for LONG
                profit checks to avoid premature exits when spread narrows.
        """
        if side == "BUY":
            self._tick_long_in_flight = True
            positions = self._long_positions
        else:
            self._tick_short_in_flight = True
            positions = self._short_positions

        try:
            response = self.ig.open_position(
                epic=self.epic,
                size=self.params.contract_size,
                side=side,
                limit=self.params.take_profit_ticks,
            )
            deal_id = _extract_deal_id(response)
            if deal_id != "unknown":
                pos = {
                    "deal_id": deal_id,
                    "entry_price": bid,
                    "size": self.params.contract_size,
                }
                if side == "BUY":
                    # Store spread at open so LONG profit check uses the actual
                    # ask cost (bid + spread_at_open), not the exit tick spread.
                    pos["entry_spread"] = spread
                positions.append(pos)
                logger.info(f"Tick: opened {side} {deal_id} @ bid={bid:.2f}")
        except Exception as e:
            logger.error(f"Tick: failed to open {side}: {e}")
        finally:
            if side == "BUY":
                self._tick_long_in_flight = False
            else:
                self._tick_short_in_flight = False

    def _tick_close_positions(
        self,
        positions: list,
        close_side: str,
        bid: float,
        spread: float,
        direction: str,
    ) -> list:
        """Close profitable positions and return those that should remain open.

        For LONG positions, profit is calculated using the spread stored at
        entry time (``pos["entry_spread"]``) rather than the current exit-tick
        spread. This prevents premature closes when the spread narrows between
        open and close — a common pattern in mean-reversion where entries fire
        during volatile (wide-spread) spikes and exits fire during calmer
        (tight-spread) recovery periods.

        For SHORT positions, the exit-tick spread is correct because we pay
        the ask price when buying back to close a short.

        Args:
            positions: Current position list (longs or shorts).
            close_side: REST close direction — 'SELL' for longs, 'BUY' for shorts.
            bid: Current bid price.
            spread: Live spread (ofr - bid) from the tick. Used for SHORT profit
                checks; LONGs use their stored ``entry_spread`` instead.
            direction: 'long' or 'short' — used for profit calculation and logging.

        Returns:
            List of positions that were not closed (kept open).
        """
        profit_fn = _long_profit if direction == "long" else _short_profit
        to_close = []
        to_keep = []
        for pos in positions:
            if direction == "long":
                # Use the spread captured at entry so the profit check reflects
                # the actual ask cost paid at open, not the current exit spread.
                effective_spread = pos.get("entry_spread", spread)
            else:
                effective_spread = spread
            profit = profit_fn(bid, pos["entry_price"], effective_spread, pos["size"])
            if profit > 0:
                to_close.append(pos)
            else:
                to_keep.append(pos)
        for pos in to_close:
            try:
                self.ig.close_position(pos["deal_id"], close_side, pos["size"])
                logger.info(
                    f"Tick: closed {direction.upper()} {pos['deal_id']} @ bid={bid:.2f}"
                )
            except Exception as e:
                logger.error(f"Tick: failed to close {direction} {pos['deal_id']}: {e}")
                pos["needs_reconciliation"] = True
                to_keep.append(pos)
        return to_keep

    def _on_tick(self, tick: dict) -> None:
        """Evaluate entry and exit signals from a live tick in tick mode.

        Called from the worker thread via the dispatcher. Reads cached indicators
        (set by the last _on_candle call) and compares live bid/ofr prices to the
        Bollinger Band levels. Delegates REST calls to _tick_try_open() and exit
        logic to _tick_close_positions() so this method stays under 50 lines.

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

        # --- Long exit ---
        if (
            not self._tick_long_close_in_flight
            and bid > bb_upper
            and self._long_positions
        ):
            logger.debug(
                "tick long_exit: triggered bid=%.5f > bb_upper=%.5f positions=%d",
                bid,
                bb_upper,
                len(self._long_positions),
            )
            self._tick_long_close_in_flight = True
            try:
                self._long_positions = self._tick_close_positions(
                    self._long_positions, "SELL", bid, spread, "long"
                )
            finally:
                self._tick_long_close_in_flight = False
        elif (
            self._tick_long_close_in_flight and bid > bb_upper and self._long_positions
        ):
            logger.debug("tick long_exit: skipped (in_flight) bid=%.5f", bid)

        # --- Short exit ---
        if (
            not self._tick_short_close_in_flight
            and bid < bb_lower
            and self._short_positions
        ):
            logger.debug(
                "tick short_exit: triggered bid=%.5f < bb_lower=%.5f positions=%d",
                bid,
                bb_lower,
                len(self._short_positions),
            )
            self._tick_short_close_in_flight = True
            try:
                self._short_positions = self._tick_close_positions(
                    self._short_positions, "BUY", bid, spread, "short"
                )
            finally:
                self._tick_short_close_in_flight = False
        elif (
            self._tick_short_close_in_flight
            and bid < bb_lower
            and self._short_positions
        ):
            logger.debug("tick short_exit: skipped (in_flight) bid=%.5f", bid)

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
        as current behavior). No entry_spread is set on seeded positions;
        _tick_close_positions uses its existing fallback.
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
                        logger.warning(
                            f"Seeded LONG {pos['deal_id']} has no entry_spread "
                            f"(position restored from broker state after restart). "
                            f"_tick_close_positions will use live tick spread as "
                            f"fallback for profit checks."
                        )
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
        self.streaming_client.start(self._on_candle, on_tick=on_tick)
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


def _long_profit(close: float, entry_price: float, spread: float, size: float) -> float:
    """Compute unrealised profit for a long position after spread.

    Args:
        close: Current close price.
        entry_price: Price at which the long was opened.
        spread: Instrument bid/ask spread in points.
        size: Position size in contracts.

    Returns:
        Unrealised profit. Positive means profitable.
    """
    return (close - entry_price - spread) * size


def _short_profit(
    close: float, entry_price: float, spread: float, size: float
) -> float:
    """Compute unrealised profit for a short position after spread.

    Args:
        close: Current close price.
        entry_price: Price at which the short was opened.
        spread: Instrument bid/ask spread in points.
        size: Position size in contracts.

    Returns:
        Unrealised profit. Positive means profitable.
    """
    return (entry_price - close - spread) * size


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
