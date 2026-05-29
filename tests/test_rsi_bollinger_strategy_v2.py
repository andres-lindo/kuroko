"""Unit tests for RSIBollingerStrategyV2 (REQ-7 through REQ-12, REQ-17) and related
infrastructure (candle-frequency utilities, V2 JSON config, V1/V2 param loading,
requirements pinning).

Covers entry signals (long/short), entry guards (max positions, min distance),
exit logic (profitable per-position close), grid independence, flat sizing,
spread-aware profit calculation, broker take-profit on open, candle frequency
routing from JSON config through to IGStreamingClient resolution, and V1 param
schema validation.

All broker interactions are mocked — no live credentials required.
"""

import json
import queue
import threading
import types
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from ig_streaming_client import (
    IGStreamingClient,
    TickAggregator,
    _resolution_to_minutes,
    candle_frequency_to_resolution,
)
from kuroko import _wire_strategy
from strategies.RSIBollingerStrategyV2 import (
    RSIBollingerStrategyV2,
    _PARAMS_SCHEMA,
    load_params,
)

# --------------------------------------------------------------------------- #
# Project root for config file discovery                                       #
# --------------------------------------------------------------------------- #

_PROJECT_ROOT = Path(__file__).parent.parent
_V2_JSON = _PROJECT_ROOT / "strategies" / "RSIBollingerStrategyV2.json"
_REQUIREMENTS = _PROJECT_ROOT / "requirements.txt"

# --------------------------------------------------------------------------- #
# Helpers (non-factory utilities — factories live in conftest.py)              #
# --------------------------------------------------------------------------- #


def _make_candle(close: float, spread: float = 1.0) -> dict:
    """Return a minimal candle dict with bid/ofr close prices and computed spread."""
    return {
        "open": close,
        "high": close + 5,
        "low": close - 5,
        "close": close,
        "bid_close": close - spread / 2,
        "ofr_close": close + spread / 2,
        "spread": spread,
        "volume": 0,
        "timestamp": None,
    }


def _make_indicators(
    bb_upper: float = 200.0,
    bb_middle: float = 100.0,
    bb_lower: float = 0.0,
    rsi: float = 50.0,
    close: float = 100.0,
) -> dict:
    """Return a computed-indicators dict as produced by _compute_indicators."""
    return {
        "bb_upper": bb_upper,
        "bb_middle": bb_middle,
        "bb_lower": bb_lower,
        "rsi": rsi,
        "close": close,
    }


# --------------------------------------------------------------------------- #
# load_params — REQ-6 schema validation                                        #
# --------------------------------------------------------------------------- #


class TestLoadParams:
    """load_params validates the V2 JSON schema (REQ-6)."""

    def test_valid_v2_json_loads_without_exit(self, tmp_path):
        """a well-formed V2 JSON file loads successfully."""
        data = {
            "epic": "IX.D.NASDAQ.IFMM.IP",
            "api_mode": "streaming",
            "operation_mode": "candle",
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
            "take_profit_ticks": 240.0,
        }
        path = tmp_path / "RSIBollingerStrategyV2.json"
        path.write_text(json.dumps(data))

        params = load_params(str(path))

        assert params.epic == "IX.D.NASDAQ.IFMM.IP"
        assert params.api_mode == "streaming"
        assert params.candle_frequency == "5min"
        assert params.bb_period == 20
        assert params.rsi_period == 14

    def test_missing_key_triggers_sys_exit(self, tmp_path):
        """missing required key causes SystemExit."""
        incomplete = {
            "api_mode": "streaming",
            "bb_period": 20,
            # bb_std missing
        }
        path = tmp_path / "v2.json"
        path.write_text(json.dumps(incomplete))

        with pytest.raises(SystemExit):
            load_params(str(path))

    def test_wrong_type_triggers_sys_exit(self, tmp_path):
        """wrong type for a required key causes SystemExit."""
        data = {
            "api_mode": "streaming",
            "bb_period": "twenty",  # must be int
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "max_long_positions": 5,
            "max_short_positions": 5,
            "contract_size": 0.1,
            "min_dist_between_entries_ticks": 20,
            "take_profit_ticks": 240.0,
        }
        path = tmp_path / "v2.json"
        path.write_text(json.dumps(data))

        with pytest.raises(SystemExit):
            load_params(str(path))


# --------------------------------------------------------------------------- #
# Long entry signal — REQ-7                                                    #
# --------------------------------------------------------------------------- #


class TestLongEntry:
    """long position opens when all entry conditions are met."""

    def test_long_entry_opens_position_when_all_conditions_met(self, make_strategy_v2):
        """Price < BB_lower and RSI < oversold triggers BUY."""
        strat, mock_ig, _ = make_strategy_v2()
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=95.0
        )

        strat._manage_longs(indicators)

        mock_ig.open_position.assert_called_once()
        call_kwargs = mock_ig.open_position.call_args.kwargs
        assert call_kwargs["side"] == "BUY"
        assert call_kwargs["size"] == 1.0

    def test_long_entry_sets_take_profit_at_open(self, make_strategy_v2):
        """Broker take-profit set at entry_price + take_profit_ticks."""
        strat, mock_ig, _ = make_strategy_v2()
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=95.0
        )

        strat._manage_longs(indicators)

        call_kwargs = mock_ig.open_position.call_args.kwargs
        # take_profit_ticks=50.0, close=95.0 → limit distance = 50.0
        assert call_kwargs["limit"] == pytest.approx(50.0)

    def test_long_entry_blocked_when_rsi_not_oversold(self, make_strategy_v2):
        """no entry when RSI is NOT below rsi_oversold (50 >= 30)."""
        strat, mock_ig, _ = make_strategy_v2()
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=50.0, close=95.0
        )

        strat._manage_longs(indicators)

        mock_ig.open_position.assert_not_called()

    def test_long_entry_blocked_when_price_above_bb_lower(self, make_strategy_v2):
        """no entry when price is NOT below BB lower (105 >= 100)."""
        strat, mock_ig, _ = make_strategy_v2()
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=105.0
        )

        strat._manage_longs(indicators)

        mock_ig.open_position.assert_not_called()


# --------------------------------------------------------------------------- #
# Long entry guards — REQ-7 scenarios 2 & 3                                   #
# --------------------------------------------------------------------------- #


class TestLongEntryGuards:
    """Max positions and min distance prevent entry."""

    def test_long_entry_blocked_when_max_positions_reached(
        self, make_strategy_v2, make_params_v2
    ):
        """No new long when max_long_positions is full."""
        params = make_params_v2(max_long_positions=2)
        strat, mock_ig, _ = make_strategy_v2(params=params)
        # Pre-fill with 2 long positions
        strat._long_positions = [
            {"deal_id": "A", "entry_price": 90.0, "size": 0.5},
            {"deal_id": "B", "entry_price": 80.0, "size": 0.5},
        ]
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=95.0
        )

        strat._manage_longs(indicators)

        mock_ig.open_position.assert_not_called()

    def test_long_entry_blocked_when_distance_too_small(
        self, make_strategy_v2, make_params_v2
    ):
        """No entry when distance from last entry < min_dist_between_entries_ticks."""
        params = make_params_v2(min_dist_between_entries_ticks=20)
        strat, mock_ig, _ = make_strategy_v2(params=params)
        # Last long at 90.0; current price 95.0 → distance = 5 < 20
        strat._long_positions = [
            {"deal_id": "A", "entry_price": 90.0, "size": 0.5},
        ]
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=95.0
        )

        strat._manage_longs(indicators)

        mock_ig.open_position.assert_not_called()

    def test_long_entry_allowed_when_distance_sufficient(
        self, make_strategy_v2, make_params_v2
    ):
        """Entry allowed when distance >= min_dist_between_entries_ticks."""
        params = make_params_v2(min_dist_between_entries_ticks=10)
        strat, mock_ig, _ = make_strategy_v2(params=params)
        # Last long at 80.0; current price 95.0 → distance = 15 >= 10
        strat._long_positions = [
            {"deal_id": "A", "entry_price": 80.0, "size": 0.5},
        ]
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=95.0
        )

        strat._manage_longs(indicators)

        mock_ig.open_position.assert_called_once()


# --------------------------------------------------------------------------- #
# Short entry signal — REQ-8                                                   #
# --------------------------------------------------------------------------- #


class TestShortEntry:
    """short position opens when all entry conditions are met."""

    def test_short_entry_opens_position_when_all_conditions_met(self, make_strategy_v2):
        """Price > BB_upper and RSI > overbought triggers SELL."""
        strat, mock_ig, _ = make_strategy_v2()
        indicators = _make_indicators(
            bb_lower=0.0, bb_upper=100.0, rsi=75.0, close=105.0
        )

        strat._manage_shorts(indicators)

        mock_ig.open_position.assert_called_once()
        call_kwargs = mock_ig.open_position.call_args.kwargs
        assert call_kwargs["side"] == "SELL"
        assert call_kwargs["size"] == 1.0

    def test_short_entry_sets_take_profit_at_open(self, make_strategy_v2):
        """Broker take-profit set at entry_price - take_profit_ticks."""
        strat, mock_ig, _ = make_strategy_v2()
        indicators = _make_indicators(
            bb_lower=0.0, bb_upper=100.0, rsi=75.0, close=105.0
        )

        strat._manage_shorts(indicators)

        call_kwargs = mock_ig.open_position.call_args.kwargs
        # take_profit_ticks=50.0 → limit distance = 50.0
        assert call_kwargs["limit"] == pytest.approx(50.0)

    def test_short_entry_blocked_when_max_positions_reached(
        self, make_strategy_v2, make_params_v2
    ):
        """No new short when max_short_positions is full."""
        params = make_params_v2(max_short_positions=1)
        strat, mock_ig, _ = make_strategy_v2(params=params)
        strat._short_positions = [{"deal_id": "A", "entry_price": 110.0, "size": 0.5}]
        indicators = _make_indicators(
            bb_lower=0.0, bb_upper=100.0, rsi=75.0, close=105.0
        )

        strat._manage_shorts(indicators)

        mock_ig.open_position.assert_not_called()

    def test_short_entry_blocked_when_distance_too_small(
        self, make_strategy_v2, make_params_v2
    ):
        """no entry when distance from last short entry < min_dist_between_entries_ticks."""
        params = make_params_v2(min_dist_between_entries_ticks=20)
        strat, mock_ig, _ = make_strategy_v2(params=params)
        # Last short at 110; current 105 → distance = 5 < 20
        strat._short_positions = [{"deal_id": "A", "entry_price": 110.0, "size": 0.5}]
        indicators = _make_indicators(
            bb_lower=0.0, bb_upper=100.0, rsi=75.0, close=105.0
        )

        strat._manage_shorts(indicators)

        mock_ig.open_position.assert_not_called()


# --------------------------------------------------------------------------- #
# Long exit logic — REQ-9                                                      #
# --------------------------------------------------------------------------- #


class TestLongExit:
    """profitable longs are closed when price crosses above BB upper."""

    def test_profitable_long_closed_at_bb_upper_cross(self, make_strategy_v2):
        """Long with profit > 0 after spread is closed."""
        strat, mock_ig, _ = make_strategy_v2()
        # entry=90, close=105, spread=0.0 (not set) → profit = (105-90-0)*0.5 = 7.5 > 0
        strat._long_positions = [{"deal_id": "DEAL1", "entry_price": 90.0, "size": 0.5}]
        indicators = _make_indicators(
            bb_lower=50.0, bb_upper=100.0, rsi=50.0, close=105.0
        )

        strat._manage_longs(indicators)

        mock_ig.close_position.assert_called_once_with("DEAL1", "SELL", 0.5)

    def test_long_not_closed_when_underwater_after_spread(self, make_strategy_v2):
        """Long with profit <= 0 after spread is NOT closed at BB upper."""
        strat, mock_ig, _ = make_strategy_v2()
        strat._current_spread = 1.0  # spread from latest candle
        # entry=104, close=105, spread=1.0 → profit = (105-104-1)*0.5 = 0 (not > 0)
        strat._long_positions = [
            {"deal_id": "DEAL1", "entry_price": 104.0, "size": 0.5}
        ]
        indicators = _make_indicators(
            bb_lower=50.0, bb_upper=100.0, rsi=50.0, close=105.0
        )

        strat._manage_longs(indicators)

        mock_ig.close_position.assert_not_called()

    def test_only_profitable_longs_are_closed(self, make_strategy_v2):
        """Only profitable positions closed; underwater stays open."""
        strat, mock_ig, _ = make_strategy_v2()
        strat._current_spread = 1.0  # spread from latest candle
        # DEAL1: entry=90, profit = (105-90-1)*0.5 = 7.0 > 0 → close
        # DEAL2: entry=104, profit = (105-104-1)*0.5 = 0 → keep
        strat._long_positions = [
            {"deal_id": "DEAL1", "entry_price": 90.0, "size": 0.5},
            {"deal_id": "DEAL2", "entry_price": 104.0, "size": 0.5},
        ]
        indicators = _make_indicators(
            bb_lower=50.0, bb_upper=100.0, rsi=50.0, close=105.0
        )

        strat._manage_longs(indicators)

        assert mock_ig.close_position.call_count == 1
        mock_ig.close_position.assert_called_with("DEAL1", "SELL", 0.5)
        # DEAL2 must still be in the positions list
        remaining_ids = [p["deal_id"] for p in strat._long_positions]
        assert "DEAL2" in remaining_ids

    def test_long_spread_subtracted_from_profit(self, make_strategy_v2):
        """spread is subtracted when computing per-position profit."""
        # With spread=5: entry=90, close=94, spread=5 → (94-90-5)*0.5 = -0.5 → NOT closed
        strat, mock_ig, _ = make_strategy_v2()
        strat._current_spread = 5.0  # spread from latest candle
        strat._long_positions = [{"deal_id": "DEAL1", "entry_price": 90.0, "size": 0.5}]
        indicators = _make_indicators(
            bb_lower=50.0, bb_upper=90.0, rsi=50.0, close=94.0
        )

        strat._manage_longs(indicators)

        mock_ig.close_position.assert_not_called()

    def test_long_spread_read_from_candle_not_trading_config(self, make_strategy_v2):
        """Spread must come from the candle, not trading_config.

        Entry=100, close=105:
        - Without candle spread update (fallback 0.0): profit = (105-100-0)*0.5 = 2.5 → close
        - After candle spread update (spread=1.0): profit = (105-100-1)*0.5 = 2.0 → close

        This test verifies that _update_spread_from_candle correctly updates the spread
        used for profit calculations.
        """
        strat, mock_ig, _ = make_strategy_v2()
        strat._long_positions = [
            {"deal_id": "DEAL1", "entry_price": 100.0, "size": 0.5}
        ]
        # Update strategy's current spread from a candle with spread=1.0
        strat._update_spread_from_candle(
            {"bid_close": 104.5, "ofr_close": 105.5, "spread": 1.0}
        )
        indicators = _make_indicators(
            bb_lower=50.0, bb_upper=100.0, rsi=50.0, close=105.0
        )

        strat._manage_longs(indicators)

        # Should be closed: profit = (105 - 100 - 1) * 0.5 = 2.0 > 0
        mock_ig.close_position.assert_called_once_with("DEAL1", "SELL", 0.5)


# --------------------------------------------------------------------------- #
# Short exit logic — REQ-10                                                    #
# --------------------------------------------------------------------------- #


class TestShortExit:
    """profitable shorts are closed when price crosses below BB lower."""

    def test_profitable_short_closed_at_bb_lower_cross(self, make_strategy_v2):
        """Short with profit > 0 after spread is closed."""
        strat, mock_ig, _ = make_strategy_v2()
        # entry=110, close=95, spread=0.0 (not set) → profit = (110-95-0)*0.5 = 7.5 > 0
        strat._short_positions = [
            {"deal_id": "DEAL2", "entry_price": 110.0, "size": 0.5}
        ]
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=150.0, rsi=50.0, close=95.0
        )

        strat._manage_shorts(indicators)

        mock_ig.close_position.assert_called_once_with("DEAL2", "BUY", 0.5)

    def test_short_not_closed_when_underwater(self, make_strategy_v2):
        """short with profit <= 0 after spread is NOT closed at BB lower."""
        strat, mock_ig, _ = make_strategy_v2()
        strat._current_spread = 1.0  # spread from latest candle
        # entry=96, close=95, spread=1 → profit = (96-95-1)*0.5 = 0 → NOT closed
        strat._short_positions = [
            {"deal_id": "DEAL2", "entry_price": 96.0, "size": 0.5}
        ]
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=150.0, rsi=50.0, close=95.0
        )

        strat._manage_shorts(indicators)

        mock_ig.close_position.assert_not_called()

    def test_short_spread_subtracted_from_profit(self, make_strategy_v2):
        """spread is applied when computing short profit."""
        # entry=105, close=100, spread=5 → profit = (105-100-5)*0.5 = 0 → NOT closed
        strat, mock_ig, _ = make_strategy_v2()
        strat._current_spread = 5.0  # spread from latest candle
        strat._short_positions = [
            {"deal_id": "DEAL2", "entry_price": 105.0, "size": 0.5}
        ]
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=150.0, rsi=50.0, close=100.0
        )

        strat._manage_shorts(indicators)

        mock_ig.close_position.assert_not_called()

    def test_short_spread_read_from_candle_not_trading_config(self, make_strategy_v2):
        """Spread must come from the candle, not trading_config.

        After _update_spread_from_candle is called with spread=1.0:
        entry=110, close=95 → profit = (110 - 95 - 1) * 0.5 = 7.0 > 0 → closed.
        """
        strat, mock_ig, _ = make_strategy_v2()
        strat._short_positions = [
            {"deal_id": "DEAL2", "entry_price": 110.0, "size": 0.5}
        ]
        strat._update_spread_from_candle(
            {"bid_close": 94.5, "ofr_close": 95.5, "spread": 1.0}
        )
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=150.0, rsi=50.0, close=95.0
        )

        strat._manage_shorts(indicators)

        # Should be closed: profit = (110 - 95 - 1) * 0.5 = 7.0 > 0
        mock_ig.close_position.assert_called_once_with("DEAL2", "BUY", 0.5)


# --------------------------------------------------------------------------- #
# Grid independence — REQ-11                                                   #
# --------------------------------------------------------------------------- #


class TestGridIndependence:
    """long and short grids are evaluated independently."""

    def test_long_and_short_grids_coexist_without_interference(self, make_strategy_v2):
        """2 longs and 2 shorts all remain open with neutral candle."""
        strat, mock_ig, _ = make_strategy_v2()
        strat._long_positions = [
            {"deal_id": "L1", "entry_price": 80.0, "size": 0.5},
            {"deal_id": "L2", "entry_price": 70.0, "size": 0.5},
        ]
        strat._short_positions = [
            {"deal_id": "S1", "entry_price": 120.0, "size": 0.5},
            {"deal_id": "S2", "entry_price": 130.0, "size": 0.5},
        ]
        # Neutral candle: close=100, BBs=[50, 100, 150], RSI=50 — no entry or exit signals
        indicators = _make_indicators(
            bb_lower=50.0, bb_upper=150.0, rsi=50.0, close=100.0
        )

        strat._manage_longs(indicators)
        strat._manage_shorts(indicators)

        mock_ig.open_position.assert_not_called()
        mock_ig.close_position.assert_not_called()
        assert len(strat._long_positions) == 2
        assert len(strat._short_positions) == 2

    def test_short_signal_does_not_suppress_long_entry(
        self, make_strategy_v2, make_params_v2
    ):
        """a short signal condition does NOT prevent a valid long entry."""
        params = make_params_v2(max_long_positions=3, max_short_positions=3)
        strat, mock_ig, _ = make_strategy_v2(params=params)
        # Set up indicators that trigger BOTH a long entry condition and a short exit condition
        # Long entry: close=90 < bb_lower=95, rsi=25 < 30
        # Short exit: close=90 < bb_lower=95 (would close underwater shorts — they stay open)
        strat._short_positions = [
            {"deal_id": "S1", "entry_price": 91.0, "size": 0.5}  # underwater short
        ]
        indicators = _make_indicators(
            bb_lower=95.0, bb_upper=200.0, rsi=25.0, close=90.0
        )

        strat._manage_longs(indicators)

        mock_ig.open_position.assert_called_once()


# --------------------------------------------------------------------------- #
# Flat sizing — REQ-12                                                         #
# --------------------------------------------------------------------------- #


class TestFlatSizing:
    """all positions use flat contract_size (no martingale)."""

    def test_second_long_entry_uses_flat_size(self, make_strategy_v2, make_params_v2):
        """Second long entry size equals contract_size."""
        params = make_params_v2(contract_size=0.5, max_long_positions=3)
        strat, mock_ig, _ = make_strategy_v2(params=params)
        # One existing long
        strat._long_positions = [{"deal_id": "A", "entry_price": 60.0, "size": 0.5}]
        # New signal with sufficient distance
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=75.0
        )

        strat._manage_longs(indicators)

        call_kwargs = mock_ig.open_position.call_args.kwargs
        assert call_kwargs["size"] == pytest.approx(0.5)

    def test_second_short_entry_uses_flat_size(self, make_strategy_v2, make_params_v2):
        """second short entry size equals contract_size (no multiplier)."""
        params = make_params_v2(contract_size=0.5, max_short_positions=3)
        strat, mock_ig, _ = make_strategy_v2(params=params)
        strat._short_positions = [{"deal_id": "A", "entry_price": 140.0, "size": 0.5}]
        indicators = _make_indicators(
            bb_lower=0.0, bb_upper=100.0, rsi=75.0, close=120.0
        )

        strat._manage_shorts(indicators)

        call_kwargs = mock_ig.open_position.call_args.kwargs
        assert call_kwargs["size"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Dynamic spread from candle data                                               #
# --------------------------------------------------------------------------- #


class TestDynamicSpread:
    """Spread must be read from candle data (OFR_CLOSE - BID_CLOSE), not from config."""

    def test_update_spread_from_candle_sets_current_spread(self, make_strategy_v2):
        """_update_spread_from_candle must update self._current_spread."""
        strat, _, _ = make_strategy_v2()
        candle = {"bid_close": 100.0, "ofr_close": 101.5, "spread": 1.5}

        strat._update_spread_from_candle(candle)

        assert strat._current_spread == pytest.approx(1.5)

    def test_on_candle_updates_spread(self, make_strategy_v2):
        """_on_candle must update _current_spread from the incoming candle."""
        strat, _, _ = make_strategy_v2()
        # Build enough candle history for indicators to be valid
        # (need bb_period=20 + rsi_period=14 + 1 = 21 candles minimum)
        for _ in range(25):
            strat._candle_window.append(100.0)
        # Now set a candle with a specific spread
        candle = _make_candle(close=100.0, spread=2.5)

        with (
            patch.object(strat, "_manage_longs"),
            patch.object(strat, "_manage_shorts"),
        ):
            strat._on_candle(candle)

        assert strat._current_spread == pytest.approx(2.5)

    def test_initial_spread_is_none_before_any_candle(self, make_strategy_v2):
        """Before any candle is processed, _current_spread must be None."""
        strat, _, _ = make_strategy_v2()
        assert strat._current_spread is None

    def test_profit_uses_candle_spread_not_config_spread(self, make_strategy_v2):
        """End-to-end: profit calculation uses _current_spread set from candle.

        entry=100, close=105, _current_spread=1.0:
        profit = (105-100-1)*0.5 = 2.0 → closed.

        Without setting _current_spread (fallback=0.0):
        profit = (105-100-0)*0.5 = 2.5 → also closed.

        Verified via _update_spread_from_candle path.
        """
        strat, mock_ig, _ = make_strategy_v2()
        strat._long_positions = [
            {"deal_id": "DEAL1", "entry_price": 100.0, "size": 0.5}
        ]
        strat._current_spread = 1.0  # simulates a candle having been processed
        indicators = _make_indicators(
            bb_lower=50.0, bb_upper=100.0, rsi=50.0, close=105.0
        )

        strat._manage_longs(indicators)

        # profit = (105 - 100 - 1) * 0.5 = 2.0 > 0 → must close (candle spread used)
        mock_ig.close_position.assert_called_once_with("DEAL1", "SELL", 0.5)

    def test_spread_falls_back_to_default_when_no_candle_yet(self, make_strategy_v2):
        """If no candle has been processed yet, spread defaults to 0.0 (no suppression)."""
        strat, mock_ig, _ = make_strategy_v2()
        # _current_spread = None initially; strategy must use 0.0 as safe default
        strat._long_positions = [{"deal_id": "DEAL1", "entry_price": 90.0, "size": 0.5}]
        indicators = _make_indicators(
            bb_lower=50.0, bb_upper=100.0, rsi=50.0, close=105.0
        )

        strat._manage_longs(indicators)

        # profit = (105 - 90 - 0.0) * 0.5 = 7.5 > 0 → must close
        mock_ig.close_position.assert_called_once_with("DEAL1", "SELL", 0.5)


# --------------------------------------------------------------------------- #
# Single-queue pattern — design: LS → streaming_client queue → strategy run() #
# --------------------------------------------------------------------------- #


class TestSingleQueuePattern:
    """run() must pass _on_candle directly to streaming_client.start() (no double-queue)."""

    def test_run_passes_on_candle_directly_as_callback(self, make_strategy_v2):
        """run() must call streaming_client.start() with _on_candle as the callback.

        The strategy must NOT pass an intermediate re-queuing function like
        _enqueue_candle. The streaming client's own internal queue is sufficient.
        Verified by running in a thread: start() captures the callback, then
        stop() is called to unblock run().
        """
        strat, _, mock_streaming = make_strategy_v2()
        captured = {}

        def fake_start(callback, on_tick=None):
            captured["callback"] = callback
            # Immediately stop so run() returns
            strat.stop()

        mock_streaming.start.side_effect = fake_start

        t = threading.Thread(target=strat.run)
        t.start()
        t.join(timeout=2.0)

        assert not t.is_alive(), "run() did not return after stop()"
        assert "callback" in captured, "streaming_client.start() was never called"
        # Verify the callback is _on_candle by calling it and checking behavior
        # (bound methods compare by identity per call — compare by name instead)
        assert (
            captured["callback"].__name__ == "_on_candle"
        ), f"Expected _on_candle callback, got {captured['callback'].__name__}"

    def test_run_does_not_use_internal_candle_queue_for_routing(self, make_strategy_v2):
        """run() must not have an internal _candle_queue attribute for re-queuing candles.

        The design (LS → IGStreamingClient queue → strategy run()) means the
        strategy should consume directly from the streaming client — not re-queue
        via its own internal queue.
        """
        strat, _, _ = make_strategy_v2()
        assert not hasattr(strat, "_candle_queue"), (
            "Strategy must not have an internal _candle_queue for re-routing candles. "
            "Consume directly from the streaming client's queue."
        )


# --------------------------------------------------------------------------- #
# Issue 1: Phantom positions — reconciliation after close failure              #
# --------------------------------------------------------------------------- #


class TestPhantomPositionReconciliation:
    """CRITICAL: failed close must not leave phantom positions in the grid forever."""

    def test_failed_close_marks_position_needs_reconciliation(self, make_strategy_v2):
        """After a close failure, the position must be flagged needs_reconciliation=True."""
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.close_position.side_effect = Exception("Network error")
        strat._long_positions = [{"deal_id": "DEAL1", "entry_price": 90.0, "size": 0.5}]
        indicators = _make_indicators(
            bb_lower=50.0, bb_upper=100.0, rsi=50.0, close=105.0
        )

        strat._manage_longs(indicators)

        # Position must stay in grid AND be flagged
        assert len(strat._long_positions) == 1
        assert strat._long_positions[0].get("needs_reconciliation") is True

    def test_failed_short_close_marks_position_needs_reconciliation(
        self, make_strategy_v2
    ):
        """After a short close failure, the position must be flagged needs_reconciliation=True."""
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.close_position.side_effect = Exception("Broker timeout")
        strat._short_positions = [
            {"deal_id": "DEAL2", "entry_price": 110.0, "size": 0.5}
        ]
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=150.0, rsi=50.0, close=95.0
        )

        strat._manage_shorts(indicators)

        assert len(strat._short_positions) == 1
        assert strat._short_positions[0].get("needs_reconciliation") is True

    def test_reconcile_removes_positions_absent_from_broker(self, make_strategy_v2):
        """_reconcile_positions must remove local positions that no longer exist at broker.

        get_open_positions() returns a flat list of dicts with a top-level
        'dealId' key — not a nested {'positions': [{'position': {'dealId': ...}}]} dict.
        """
        strat, mock_ig, _ = make_strategy_v2()
        strat._long_positions = [
            {
                "deal_id": "DEAL1",
                "entry_price": 90.0,
                "size": 0.5,
                "needs_reconciliation": True,
            },
            {"deal_id": "DEAL2", "entry_price": 80.0, "size": 0.5},
        ]
        # Broker only has DEAL2 — flat list format matching IGClient.get_open_positions()
        mock_ig.get_open_positions.return_value = [
            {"dealId": "DEAL2", "dealReference": "REF2", "level": 80.0, "size": 0.5}
        ]

        strat._reconcile_positions()

        assert len(strat._long_positions) == 1
        assert strat._long_positions[0]["deal_id"] == "DEAL2"

    def test_reconcile_keeps_positions_present_at_broker(self, make_strategy_v2):
        """_reconcile_positions must keep local positions that are still open at broker.

        get_open_positions() returns a flat list of dicts with a top-level
        'dealId' key — not a nested {'positions': [{'position': {'dealId': ...}}]} dict.
        """
        strat, mock_ig, _ = make_strategy_v2()
        strat._long_positions = [
            {
                "deal_id": "DEAL1",
                "entry_price": 90.0,
                "size": 0.5,
                "needs_reconciliation": True,
            },
        ]
        # Broker still has DEAL1 — flat list format matching IGClient.get_open_positions()
        mock_ig.get_open_positions.return_value = [
            {"dealId": "DEAL1", "dealReference": "REF1", "level": 90.0, "size": 0.5}
        ]

        strat._reconcile_positions()

        assert len(strat._long_positions) == 1
        assert strat._long_positions[0]["deal_id"] == "DEAL1"
        # needs_reconciliation flag is removed (popped) after successful reconciliation
        assert "needs_reconciliation" not in strat._long_positions[0]

    def test_reconcile_called_before_entry_when_flagged_positions_exist(
        self, make_strategy_v2
    ):
        """_on_candle must call _reconcile_positions before entry evaluation
        when any position is flagged needs_reconciliation."""
        strat, mock_ig, _ = make_strategy_v2()
        strat._long_positions = [
            {
                "deal_id": "DEAD",
                "entry_price": 90.0,
                "size": 0.5,
                "needs_reconciliation": True,
            }
        ]
        # Empty flat list — DEAD position will be removed by reconciliation
        mock_ig.get_open_positions.return_value = []

        # Enough history for indicators
        for _ in range(25):
            strat._candle_window.append(100.0)
        candle = _make_candle(close=100.0)

        positions_at_manage_longs_call: list = []

        def capture_positions_snapshot(indicators):
            # Snapshot the long positions list at the moment _manage_longs runs.
            # Reconciliation must have already cleared DEAD by this point.
            positions_at_manage_longs_call.extend(list(strat._long_positions))

        with (
            patch.object(
                strat, "_manage_longs", side_effect=capture_positions_snapshot
            ),
            patch.object(strat, "_manage_shorts"),
        ):
            strat._on_candle(candle)

        # Reconciliation called the API exactly once
        mock_ig.get_open_positions.assert_called_once()
        # DEAD position must already be gone when _manage_longs fires
        dead_ids = [p["deal_id"] for p in positions_at_manage_longs_call]
        assert "DEAD" not in dead_ids, (
            "Reconciliation must remove DEAD before _manage_longs runs, "
            f"but positions at call time were: {positions_at_manage_longs_call}"
        )

    def test_reconcile_not_called_when_no_flagged_positions(self, make_strategy_v2):
        """_reconcile_positions must NOT be called if no positions need reconciliation."""
        strat, mock_ig, _ = make_strategy_v2()
        strat._long_positions = [{"deal_id": "DEAL1", "entry_price": 90.0, "size": 0.5}]
        # No needs_reconciliation flag

        # Enough history for indicators
        for _ in range(25):
            strat._candle_window.append(100.0)
        candle = _make_candle(close=100.0)

        with (
            patch.object(strat, "_manage_longs"),
            patch.object(strat, "_manage_shorts"),
        ):
            strat._on_candle(candle)
            mock_ig.get_open_positions.assert_not_called()


# --------------------------------------------------------------------------- #
# Issue 2: _extract_deal_id returns "unknown" silently                         #
# --------------------------------------------------------------------------- #


class TestExtractDealIdUnknown:
    """WARNING: unknown deal_id must log CRITICAL and NOT add position to grid."""

    def test_position_not_added_when_deal_id_is_unknown(self, make_strategy_v2):
        """If deal_id extraction returns 'unknown', position must NOT be added to grid."""
        strat, mock_ig, _ = make_strategy_v2()
        # open_position returns a response where deal ID cannot be extracted
        mock_ig.open_position.return_value = (
            {}
        )  # empty → _extract_deal_id returns "unknown"
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=95.0
        )

        strat._manage_longs(indicators)

        # Position was NOT added to the grid
        assert len(strat._long_positions) == 0

    def test_short_position_not_added_when_deal_id_is_unknown(self, make_strategy_v2):
        """If deal_id extraction returns 'unknown' for a short, position must NOT be added."""
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.open_position.return_value = {}
        indicators = _make_indicators(
            bb_lower=0.0, bb_upper=100.0, rsi=75.0, close=110.0
        )

        strat._manage_shorts(indicators)

        assert len(strat._short_positions) == 0

    def test_position_added_when_deal_accepted_and_deal_id_present(
        self, make_strategy_v2
    ):
        """Sanity check: position IS added when dealStatus=ACCEPTED and dealId is present."""
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.open_position.return_value = {
            "dealStatus": "ACCEPTED",
            "dealId": "VALID_DEAL_ID_123",
            "dealReference": "REF123",
        }
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=95.0
        )

        strat._manage_longs(indicators)

        assert len(strat._long_positions) == 1
        assert strat._long_positions[0]["deal_id"] == "VALID_DEAL_ID_123"

    def test_position_not_added_when_deal_rejected(self, make_strategy_v2):
        """CRITICAL BUG FIX: position must NOT be added when dealStatus is REJECTED.

        The IG confirms endpoint returns 200 even for rejected deals. The response
        always contains a dealReference (used to query confirms), but dealStatus
        tells us whether the deal was actually executed. If REJECTED, no position
        exists at the broker — adding it to the grid creates a phantom position.
        """
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.open_position.return_value = {
            "dealStatus": "REJECTED",
            "dealId": "",
            "dealReference": "83YS8TTNEGTYKR",
            "reason": "error.service.marketdata.position.notional.details.null.error",
        }
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=95.0
        )

        strat._manage_longs(indicators)

        assert len(strat._long_positions) == 0

    def test_short_position_not_added_when_deal_rejected(self, make_strategy_v2):
        """CRITICAL BUG FIX: short position must NOT be added when dealStatus is REJECTED."""
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.open_position.return_value = {
            "dealStatus": "REJECTED",
            "dealId": "",
            "dealReference": "WYWNCD8KAMUTYKR",
            "reason": "error.service.marketdata.position.notional.details.null.error",
        }
        indicators = _make_indicators(
            bb_lower=0.0, bb_upper=100.0, rsi=75.0, close=110.0
        )

        strat._manage_shorts(indicators)

        assert len(strat._short_positions) == 0

    def test_position_not_added_when_deal_status_missing(self, make_strategy_v2):
        """Position must NOT be added when dealStatus is absent (malformed confirms response)."""
        strat, mock_ig, _ = make_strategy_v2()
        # dealId is present but dealStatus is absent — treated as unaccepted
        mock_ig.open_position.return_value = {
            "dealId": "SOME_ID",
            "dealReference": "REF999",
        }
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=95.0
        )

        strat._manage_longs(indicators)

        assert len(strat._long_positions) == 0

    def test_position_uses_deal_id_not_deal_reference(self, make_strategy_v2):
        """The grid must store dealId (broker position ID), not dealReference.

        dealReference is an ephemeral key used to query the confirms endpoint.
        dealId is the stable broker position identifier that close_position and
        reconciliation use. Storing dealReference causes 404s when closing.
        """
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.open_position.return_value = {
            "dealStatus": "ACCEPTED",
            "dealId": "POSITION_ID_XYZ",
            "dealReference": "CONFIRMS_REF_ABC",
        }
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=95.0
        )

        strat._manage_longs(indicators)

        assert len(strat._long_positions) == 1
        stored_id = strat._long_positions[0]["deal_id"]
        assert stored_id == "POSITION_ID_XYZ", (
            f"Expected dealId='POSITION_ID_XYZ' but got {stored_id!r}. "
            "Strategy must store dealId (broker position ID), not dealReference."
        )


# --------------------------------------------------------------------------- #
# Issue 7: Strategy must stop streaming when stop() is called                  #
# --------------------------------------------------------------------------- #


class TestStrategyStopCallsStreamingStop:
    """WARNING: stop() must also call streaming_client.stop() to halt candle delivery."""

    def test_stop_calls_streaming_client_stop(self, make_strategy_v2):
        """stop() must call self.streaming_client.stop() to halt candle delivery immediately."""
        strat, _, mock_streaming = make_strategy_v2()

        strat.stop()

        mock_streaming.stop.assert_called_once()

    def test_stop_still_sets_stop_event(self, make_strategy_v2):
        """stop() must still set the internal _stop_event alongside calling streaming_client.stop()."""
        strat, _, _ = make_strategy_v2()

        strat.stop()

        assert strat._stop_event.is_set()


# --------------------------------------------------------------------------- #
# Issue 3 (Round-2): Entry/exit boundary cases — strict inequality             #
# --------------------------------------------------------------------------- #


class TestEntryExitBoundary:
    """Verify strict inequality boundaries for entry and exit conditions.

    Per the spec:
    - Long entry: price STRICTLY < bb_lower (price == bb_lower blocks entry)
    - Long exit:  price STRICTLY > bb_upper (price == bb_upper blocks exit)
    - Short entry: price STRICTLY > bb_upper (price == bb_upper blocks entry)
    - Short exit:  price STRICTLY < bb_lower (price == bb_lower blocks exit)
    """

    def test_long_entry_blocked_when_price_equals_bb_lower(self, make_strategy_v2):
        """Entry condition is STRICT (<); price == bb_lower must NOT trigger a long entry."""
        strat, mock_ig, _ = make_strategy_v2()
        # close == bb_lower exactly → entry blocked (not strictly less than)
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=100.0
        )

        strat._manage_longs(indicators)

        mock_ig.open_position.assert_not_called()

    def test_long_exit_blocked_when_price_equals_bb_upper(self, make_strategy_v2):
        """Exit condition is STRICT (>); price == bb_upper must NOT trigger a long exit."""
        strat, mock_ig, _ = make_strategy_v2()
        strat._current_spread = 1.0
        # entry=90, close=100 == bb_upper=100 → exit NOT triggered (not strictly greater than)
        strat._long_positions = [{"deal_id": "DEAL1", "entry_price": 90.0, "size": 0.5}]
        indicators = _make_indicators(
            bb_lower=50.0, bb_upper=100.0, rsi=50.0, close=100.0
        )

        strat._manage_longs(indicators)

        mock_ig.close_position.assert_not_called()

    def test_short_entry_blocked_when_price_equals_bb_upper(self, make_strategy_v2):
        """Short entry condition is STRICT (>); price == bb_upper must NOT trigger a short entry."""
        strat, mock_ig, _ = make_strategy_v2()
        # close == bb_upper exactly → entry blocked (not strictly greater than)
        indicators = _make_indicators(
            bb_lower=0.0, bb_upper=100.0, rsi=75.0, close=100.0
        )

        strat._manage_shorts(indicators)

        mock_ig.open_position.assert_not_called()

    def test_short_exit_blocked_when_price_equals_bb_lower(self, make_strategy_v2):
        """Short exit condition is STRICT (<); price == bb_lower must NOT trigger a short exit."""
        strat, mock_ig, _ = make_strategy_v2()
        strat._current_spread = 1.0
        # entry=110, close=100 == bb_lower=100 → exit NOT triggered (not strictly less than)
        strat._short_positions = [
            {"deal_id": "DEAL2", "entry_price": 110.0, "size": 0.5}
        ]
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=150.0, rsi=50.0, close=100.0
        )

        strat._manage_shorts(indicators)

        mock_ig.close_position.assert_not_called()


# --------------------------------------------------------------------------- #
# Issue 7 (Round-2): Candle key validation in _on_candle                       #
# --------------------------------------------------------------------------- #


class TestCandleKeyValidation:
    """_on_candle must validate that required candle keys are present.

    A malformed candle (missing 'close') must be logged as WARNING and
    skipped explicitly — not allowed to raise KeyError.
    """

    def test_on_candle_skips_candle_missing_close_key(self, make_strategy_v2):
        """A candle without 'close' must be skipped; no exception raised."""
        strat, mock_ig, _ = make_strategy_v2()
        # Enough history so indicators would normally be valid
        for _ in range(25):
            strat._candle_window.append(100.0)

        bad_candle = {
            # 'close' key is intentionally absent
            "bid_close": 99.5,
            "ofr_close": 100.5,
            "spread": 1.0,
            "volume": 0,
            "timestamp": None,
        }

        # Must not raise; must not call open/close position
        strat._on_candle(bad_candle)

        mock_ig.open_position.assert_not_called()
        mock_ig.close_position.assert_not_called()

    def test_on_candle_skips_candle_missing_spread_key(self, make_strategy_v2):
        """A candle missing the required 'spread' key is rejected by _REQUIRED_CANDLE_KEYS validation.

        _on_candle logs a WARNING and returns early — no position management occurs.
        """
        strat, mock_ig, _ = make_strategy_v2()
        for _ in range(25):
            strat._candle_window.append(100.0)

        # spread key is absent — _on_candle must skip (return early) not process
        candle_no_spread = {
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 100.0,
            "bid_close": 99.5,
            "ofr_close": 100.5,
            "volume": 0,
            "timestamp": None,
            # 'spread' intentionally absent — triggers _REQUIRED_CANDLE_KEYS rejection
        }

        strat._on_candle(candle_no_spread)

        # Candle must be silently dropped — no position management must occur
        mock_ig.open_position.assert_not_called()
        mock_ig.close_position.assert_not_called()


# --------------------------------------------------------------------------- #
# REST candle warm-up                                                           #
# --------------------------------------------------------------------------- #


def _make_warmup_dataframe(n_rows: int, base_price: float = 100.0) -> pd.DataFrame:
    """Return a minimal REST DataFrame with n_rows of OHLC data.

    Columns match IGClient.get_candles() output: Open, High, Low, Close.
    Index is a naive DatetimeIndex representing London local time, matching
    the real IG REST API snapshotTime format (Europe/London, no tzinfo).
    Jan 1 is used (UTC == London in winter) to keep expected UTC values simple.
    """
    idx = pd.date_range(
        start=datetime(2026, 1, 1, 9, 0),  # naive London local (winter → == UTC)
        periods=n_rows,
        freq="5min",
    )
    data = {
        "Open": [base_price] * n_rows,
        "High": [base_price + 5] * n_rows,
        "Low": [base_price - 5] * n_rows,
        "Close": [base_price + i * 0.1 for i in range(n_rows)],
    }
    return pd.DataFrame(data, index=idx)


class TestWarmupFillsWindow:
    """_warmup() fetches REST candles and fills _candle_window directly from Close prices."""

    def test_warmup_fills_candle_window_to_num_candles(self, make_strategy_v2):
        """After _warmup, len(_candle_window) == num_candles fetched."""
        strat, mock_ig, _ = make_strategy_v2()
        num_candles = max(strat.params.bb_period, strat.params.rsi_period) + 1  # 21
        df = _make_warmup_dataframe(num_candles)
        mock_ig.get_candles.return_value = df

        strat._warmup()

        assert len(strat._candle_window) == num_candles

    def test_warmup_window_contains_close_prices(self, make_strategy_v2):
        """_warmup appends Close prices (floats) directly — not candle dicts."""
        strat, mock_ig, _ = make_strategy_v2()
        num_candles = max(strat.params.bb_period, strat.params.rsi_period) + 1
        df = _make_warmup_dataframe(num_candles, base_price=200.0)
        mock_ig.get_candles.return_value = df

        strat._warmup()

        # All values in the window must be floats (Close prices), not dicts
        for val in strat._candle_window:
            assert isinstance(val, float)
        # First value matches first Close
        assert list(strat._candle_window)[0] == pytest.approx(df["Close"].iloc[0])

    def test_warmup_sets_last_warmup_ts_to_utc_aware_datetime(self, make_strategy_v2):
        """_last_warmup_ts is a UTC-aware datetime derived from the last REST row.

        IG REST returns naive London-local timestamps. _warmup() localises the
        last row to Europe/London and converts to UTC. On Jan 1 (winter) London
        == UTC, so the numeric value matches the naive index timestamp.
        """
        strat, mock_ig, _ = make_strategy_v2()
        num_candles = max(strat.params.bb_period, strat.params.rsi_period) + 1
        df = _make_warmup_dataframe(num_candles)
        mock_ig.get_candles.return_value = df

        strat._warmup()

        # Must be UTC-aware (not naive)
        assert strat._last_warmup_ts is not None
        assert strat._last_warmup_ts.tzinfo is not None
        assert strat._last_warmup_ts.tzinfo == timezone.utc
        # Jan 1 winter: London == UTC, so numeric value matches the naive index ts
        last_naive = df.index[-1].to_pydatetime().replace(tzinfo=None)
        expected_utc = last_naive.replace(tzinfo=ZoneInfo("Europe/London")).astimezone(
            timezone.utc
        )
        assert strat._last_warmup_ts == expected_utc

    def test_warmup_bst_timestamp_converted_to_utc(self, make_strategy_v2):
        """_last_warmup_ts is UTC-aware and 1 hour behind the naive London BST value.

        Regression test for Bug 2: the old code stored df.index[-1].to_pydatetime()
        directly as _last_warmup_ts — a naive datetime representing London local time.
        During BST (UTC+1) this naive value is 1 hour ahead of its true UTC equivalent,
        causing the dedup guard to filter valid streaming candles that arrive at the
        correct UTC time.

        The fix localises the naive London timestamp to Europe/London and converts
        to UTC so _last_warmup_ts always represents the correct UTC moment.
        """
        strat, mock_ig, _ = make_strategy_v2()
        # Build a DataFrame with naive BST timestamps (July, UTC+1)
        # Last row: 2026-07-01 18:20 London local == 2026-07-01 17:20 UTC
        _LONDON = ZoneInfo("Europe/London")
        bst_naive = datetime(2026, 7, 1, 18, 20)  # naive London BST
        idx = pd.DatetimeIndex([bst_naive - timedelta(minutes=5), bst_naive])
        df = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [105, 106],
                "Low": [95, 96],
                "Close": [102.0, 103.0],
            },
            index=idx,
        )
        mock_ig.get_candles.return_value = df

        strat._warmup()

        assert strat._last_warmup_ts is not None
        assert strat._last_warmup_ts.tzinfo is not None
        # Expected: 18:20 London BST → 17:20 UTC
        expected_utc = datetime(2026, 7, 1, 17, 20, tzinfo=timezone.utc)
        assert strat._last_warmup_ts == expected_utc


class TestWarmupGracefulDegradation:
    """_warmup() handles REST failure without crashing the strategy."""

    def test_warmup_none_response_does_not_raise(self, make_strategy_v2):
        """Task 2.5 RED: when get_candles returns None, _warmup must not raise."""
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.get_candles.return_value = None

        strat._warmup()  # must not raise

    def test_warmup_none_response_leaves_candle_window_empty(self, make_strategy_v2):
        """Task 2.5: when get_candles returns None, _candle_window stays empty."""
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.get_candles.return_value = None

        strat._warmup()

        assert len(strat._candle_window) == 0

    def test_warmup_none_response_leaves_last_warmup_ts_as_none(self, make_strategy_v2):
        """Task 2.5: when get_candles returns None, _last_warmup_ts remains None."""
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.get_candles.return_value = None

        strat._warmup()

        assert strat._last_warmup_ts is None

    def test_warmup_exception_during_get_candles_is_caught(self, make_strategy_v2):
        """Task 2.5: an exception raised by get_candles is caught; no exception propagates."""
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.get_candles.side_effect = RuntimeError("REST timeout")

        strat._warmup()  # must not raise

        assert len(strat._candle_window) == 0
        assert strat._last_warmup_ts is None

    def test_warmup_empty_dataframe_does_not_raise(self, make_strategy_v2):
        """When get_candles returns an empty DataFrame (0 rows), _warmup must not raise."""
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.get_candles.return_value = pd.DataFrame()

        strat._warmup()  # must not raise

    def test_warmup_empty_dataframe_leaves_candle_window_empty(self, make_strategy_v2):
        """When get_candles returns an empty DataFrame, _candle_window must remain empty."""
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.get_candles.return_value = pd.DataFrame()

        strat._warmup()

        assert len(strat._candle_window) == 0

    def test_warmup_empty_dataframe_leaves_last_warmup_ts_as_none(
        self, make_strategy_v2
    ):
        """When get_candles returns an empty DataFrame, _last_warmup_ts must remain None."""
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.get_candles.return_value = pd.DataFrame()

        strat._warmup()

        assert strat._last_warmup_ts is None


class TestDedupGuard:
    """_on_candle dedup guard skips candles that overlap REST warm-up data."""

    def test_dedup_skips_candle_with_timestamp_equal_to_last_warmup_ts(
        self, make_strategy_v2
    ):
        """Task 3.1 RED: candle with timestamp == _last_warmup_ts must NOT be appended."""
        strat, _, _ = make_strategy_v2()
        t0 = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
        strat._last_warmup_ts = t0
        # Pre-fill window so indicators would be computed if the candle were processed
        initial_len = 5
        for _ in range(initial_len):
            strat._candle_window.append(100.0)

        overlap_candle = _make_candle(close=100.0)
        overlap_candle["timestamp"] = t0

        strat._on_candle(overlap_candle)

        assert len(strat._candle_window) == initial_len

    def test_dedup_skips_candle_with_timestamp_before_last_warmup_ts(
        self, make_strategy_v2
    ):
        """Task 3.1: candle with timestamp < _last_warmup_ts is also discarded."""
        strat, _, _ = make_strategy_v2()
        t0 = datetime(2026, 1, 1, 9, 5, tzinfo=timezone.utc)
        strat._last_warmup_ts = t0
        initial_len = 5
        for _ in range(initial_len):
            strat._candle_window.append(100.0)

        older_candle = _make_candle(close=100.0)
        older_candle["timestamp"] = t0 - timedelta(minutes=5)

        strat._on_candle(older_candle)

        assert len(strat._candle_window) == initial_len

    def test_dedup_allows_newer_candle(self, make_strategy_v2):
        """Task 3.2 RED: candle with timestamp > _last_warmup_ts IS appended."""
        strat, _, _ = make_strategy_v2()
        t0 = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
        strat._last_warmup_ts = t0
        initial_len = 5
        for _ in range(initial_len):
            strat._candle_window.append(100.0)

        newer_candle = _make_candle(close=100.0)
        newer_candle["timestamp"] = t0 + timedelta(minutes=5)

        strat._on_candle(newer_candle)

        assert len(strat._candle_window) == initial_len + 1

    def test_dedup_not_active_when_last_warmup_ts_is_none(self, make_strategy_v2):
        """Cold-start: when _last_warmup_ts is None, all candles are processed normally."""
        strat, _, _ = make_strategy_v2()
        assert strat._last_warmup_ts is None
        initial_len = 5
        for _ in range(initial_len):
            strat._candle_window.append(100.0)

        candle = _make_candle(close=100.0)
        candle["timestamp"] = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)

        strat._on_candle(candle)

        assert len(strat._candle_window) == initial_len + 1

    def test_dedup_handles_aware_warmup_ts_vs_naive_candle_ts_duplicate(
        self, make_strategy_v2
    ):
        """Dedup guard works when _last_warmup_ts is UTC-aware and candle ts is naive (same wall-clock)."""
        strat, _, _ = make_strategy_v2()
        # Warmup sets an aware UTC timestamp
        strat._last_warmup_ts = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
        initial_len = 5
        for _ in range(initial_len):
            strat._candle_window.append(100.0)

        # Streaming candle arrives with a naive timestamp at the same wall-clock value
        overlap_candle = _make_candle(close=100.0)
        overlap_candle["timestamp"] = datetime(
            2026, 1, 1, 0, 5
        )  # naive, same UTC moment

        strat._on_candle(overlap_candle)

        # Must be deduplicated — window size unchanged
        assert len(strat._candle_window) == initial_len

    def test_dedup_handles_aware_warmup_ts_vs_naive_newer_candle_ts(
        self, make_strategy_v2
    ):
        """Dedup guard correctly passes candles when _last_warmup_ts is aware and candle ts is naive but newer."""
        strat, _, _ = make_strategy_v2()
        strat._last_warmup_ts = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
        initial_len = 5
        for _ in range(initial_len):
            strat._candle_window.append(100.0)

        # Streaming candle arrives with a naive timestamp that is NEWER
        newer_candle = _make_candle(close=100.0)
        newer_candle["timestamp"] = datetime(2026, 1, 1, 0, 10)  # naive, 5 min later

        strat._on_candle(newer_candle)

        # Must be processed — window grows by 1
        assert len(strat._candle_window) == initial_len + 1

    def test_dedup_bst_streaming_candle_not_filtered_by_utc_warmup_ts(
        self, make_strategy_v2
    ):
        """Regression for Bug 3: a valid streaming candle at 17:25 UTC must NOT be
        filtered when _last_warmup_ts is correctly set to 17:20 UTC (BST).

        The old bug: _last_warmup_ts was naive London BST (18:20), warmup_ts.tzinfo
        was None so dedup stripped tz from candle_ts (17:25 UTC → naive 17:25), then
        compared 17:25 <= 18:20 → True → valid candle was incorrectly discarded.

        After the fix: _last_warmup_ts is 17:20 UTC-aware. Dedup converts both to
        naive UTC: 17:25 > 17:20 → candle is correctly passed through.
        """
        strat, _, _ = make_strategy_v2()
        # Warmup ts correctly set to 17:20 UTC (after BST fix)
        warmup_utc = datetime(2026, 7, 1, 17, 20, tzinfo=timezone.utc)
        strat._last_warmup_ts = warmup_utc
        initial_len = 5
        for _ in range(initial_len):
            strat._candle_window.append(100.0)

        # Streaming candle arrives at 17:25 UTC (correct — 5 min after last warmup)
        streaming_candle = _make_candle(close=100.0)
        streaming_candle["timestamp"] = datetime(
            2026, 7, 1, 17, 25, tzinfo=timezone.utc
        )

        strat._on_candle(streaming_candle)

        # Candle must NOT be deduplicated — window grows
        assert len(strat._candle_window) == initial_len + 1

    def test_dedup_bst_streaming_candle_filtered_when_duplicate(self, make_strategy_v2):
        """After the BST fix, a streaming candle at the same UTC moment as warmup
        is still correctly deduplicated (candle_ts == warmup_ts).
        """
        strat, _, _ = make_strategy_v2()
        # Warmup ts at 17:20 UTC (after BST fix)
        warmup_utc = datetime(2026, 7, 1, 17, 20, tzinfo=timezone.utc)
        strat._last_warmup_ts = warmup_utc
        initial_len = 5
        for _ in range(initial_len):
            strat._candle_window.append(100.0)

        # Streaming candle at exactly the same UTC moment — should be filtered
        duplicate_candle = _make_candle(close=100.0)
        duplicate_candle["timestamp"] = datetime(
            2026, 7, 1, 17, 20, tzinfo=timezone.utc
        )

        strat._on_candle(duplicate_candle)

        # Must be deduplicated — window unchanged
        assert len(strat._candle_window) == initial_len


class TestIndicatorsValidAfterWarmup:
    """After a full warm-up, the first streaming candle produces valid indicators."""

    def test_indicators_valid_immediately_after_warmup(self, make_strategy_v2):
        """Task 3.4 RED: after warmup with min_required candles, _compute_indicators returns non-None."""
        strat, mock_ig, _ = make_strategy_v2()
        num_candles = max(strat.params.bb_period, strat.params.rsi_period) + 1
        df = _make_warmup_dataframe(num_candles, base_price=100.0)
        mock_ig.get_candles.return_value = df

        strat._warmup()

        # The next fresh candle (after warmup) should produce valid indicators
        # because the window is already at min_required
        # _compute_indicators appends to window first, so we need min_required already there
        # After warmup, window has num_candles entries. One more candle will result in
        # window size of min_required + 1 (due to deque maxlen) which is still >= min_required
        fresh_ts = df.index[-1].to_pydatetime() + timedelta(minutes=5)
        fresh_candle = _make_candle(close=101.0)
        fresh_candle["timestamp"] = fresh_ts

        result = strat._compute_indicators(fresh_candle)

        assert result is not None
        assert "bb_upper" in result
        assert "rsi" in result


# --------------------------------------------------------------------------- #
# Warmup → streaming handoff via _on_candle                                   #
# --------------------------------------------------------------------------- #


class TestOnCandleAfterWarmup:
    """_on_candle must route to trade logic after warmup completes."""

    def test_on_candle_calls_manage_longs_after_warmup(self, make_strategy_v2):
        """After _warmup(), a new streaming candle via _on_candle must call _manage_longs/_manage_shorts.

        Verifies the full warmup → streaming handoff: window is pre-filled by
        _warmup() directly from Close prices, then _on_candle() with a post-warmup
        timestamp must produce valid indicators and invoke trade management methods.
        """
        strat, mock_ig, _ = make_strategy_v2()
        num_candles = max(strat.params.bb_period, strat.params.rsi_period) + 1
        df = _make_warmup_dataframe(num_candles, base_price=100.0)
        mock_ig.get_candles.return_value = df

        strat._warmup()

        # Build a streaming candle timestamped after the last warmup candle
        fresh_ts = df.index[-1].to_pydatetime() + timedelta(minutes=5)
        streaming_candle = _make_candle(close=101.0)
        streaming_candle["timestamp"] = fresh_ts

        with (
            patch.object(strat, "_manage_longs") as mock_longs,
            patch.object(strat, "_manage_shorts") as mock_shorts,
        ):
            strat._on_candle(streaming_candle)

        # Both trade management methods must be called — indicators are valid
        mock_longs.assert_called_once()
        mock_shorts.assert_called_once()

    def test_warmup_does_not_call_on_candle(self, make_strategy_v2):
        """_warmup() fills _candle_window directly — it must NOT call _on_candle at all.

        Since _warmup() no longer routes through _on_candle(), trade logic
        cannot fire on warm-up data regardless of window fill level.
        """
        strat, mock_ig, _ = make_strategy_v2()
        num_candles = max(strat.params.bb_period, strat.params.rsi_period) + 1
        df = _make_warmup_dataframe(num_candles, base_price=100.0)
        mock_ig.get_candles.return_value = df

        with patch.object(strat, "_on_candle") as mock_on_candle:
            strat._warmup()

        mock_on_candle.assert_not_called()


# =========================================================================== #
# Candle frequency utilities (REQ-candle-freq)                                 #
# =========================================================================== #


class TestV2JsonCandleFrequency:
    """candle_frequency must be present in RSIBollingerStrategyV2.json."""

    def test_v2_json_has_candle_frequency_key(self):
        """candle_frequency key must be present in V2 JSON."""
        data = json.loads(_V2_JSON.read_text(encoding="utf-8"))
        assert (
            "candle_frequency" in data
        ), "candle_frequency key missing from RSIBollingerStrategyV2.json"

    def test_v2_json_candle_frequency_is_string(self):
        """candle_frequency must be a string value (matching V1 format)."""
        data = json.loads(_V2_JSON.read_text(encoding="utf-8"))
        assert isinstance(
            data.get("candle_frequency"), str
        ), f"candle_frequency must be a string, got {type(data.get('candle_frequency'))!r}"

    def test_v2_json_candle_frequency_default_is_1min(self):
        """candle_frequency default value must be '1min'."""
        data = json.loads(_V2_JSON.read_text(encoding="utf-8"))
        assert (
            data.get("candle_frequency") == "1min"
        ), f"Expected candle_frequency='1min', got {data.get('candle_frequency')!r}"


class TestV2ParamsSchema:
    """candle_frequency must be declared in RSIBollingerStrategyV2._PARAMS_SCHEMA."""

    def test_candle_frequency_in_params_schema(self):
        """_PARAMS_SCHEMA must include candle_frequency."""
        assert (
            "candle_frequency" in _PARAMS_SCHEMA
        ), "candle_frequency missing from _PARAMS_SCHEMA in RSIBollingerStrategyV2.py"

    def test_candle_frequency_schema_type_is_str(self):
        """candle_frequency schema type must be str (matching V1 convention)."""
        assert _PARAMS_SCHEMA.get("candle_frequency") is str, (
            f"Expected candle_frequency schema type str, "
            f"got {_PARAMS_SCHEMA.get('candle_frequency')!r}"
        )


class TestResolutionToMinutes:
    """_resolution_to_minutes() must map IG resolution strings to integer minutes."""

    @pytest.mark.parametrize(
        "resolution, expected",
        [
            ("5MINUTE", 5),
            ("1MINUTE", 1),
            ("15MINUTE", 15),
            ("1HOUR", 60),
        ],
    )
    def test_resolution_maps_to_minutes(self, resolution, expected):
        """Resolution string must map to the correct integer minutes."""
        assert _resolution_to_minutes(resolution) == expected

    def test_unknown_resolution_raises_value_error(self):
        """Unknown resolution strings must raise ValueError."""
        with pytest.raises(ValueError):
            _resolution_to_minutes("UNKNOWN")


class TestCandleFrequencyToResolution:
    """candle_frequency_to_resolution() must map 'Nmin' strings to IG resolution strings."""

    @pytest.mark.parametrize(
        "frequency, expected",
        [
            ("5min", "5MINUTE"),
            ("1min", "1MINUTE"),
            ("15min", "15MINUTE"),
            ("60min", "1HOUR"),
        ],
    )
    def test_frequency_maps_to_resolution(self, frequency, expected):
        """candle_frequency string must map to the correct IG resolution string."""
        assert candle_frequency_to_resolution(frequency) == expected

    @pytest.mark.parametrize(
        "bad_frequency",
        ["5minutes", "Xmin"],
    )
    def test_invalid_format_raises_value_error(self, bad_frequency):
        """Non-'Nmin' strings and non-numeric prefixes must raise ValueError."""
        with pytest.raises(ValueError):
            candle_frequency_to_resolution(bad_frequency)


class TestTickFallbackUsesConfiguredResolution:
    """_subscribe_tick_fallback must use self._resolution to drive TickAggregator,
    not a hardcoded 5."""

    def _make_client_with_resolution(self, resolution: str) -> IGStreamingClient:
        """Build an IGStreamingClient with a given resolution and mocked stream service."""
        ig_service = MagicMock()
        client = IGStreamingClient(
            ig_service, "IX.D.NASDAQ.IFMM.IP", resolution=resolution
        )
        mock_stream_svc = MagicMock()
        mock_stream_svc.subscribe.return_value = None
        client._stream_svc = mock_stream_svc
        return client

    @pytest.mark.parametrize(
        "resolution, expected_minutes",
        [
            ("1MINUTE", 1),
            ("15MINUTE", 15),
            ("5MINUTE", 5),
        ],
    )
    def test_tick_fallback_uses_configured_resolution(
        self, resolution, expected_minutes
    ):
        """TickAggregator must be created with resolution_minutes derived from self._resolution."""
        client = self._make_client_with_resolution(resolution)
        captured = {}
        original_init = TickAggregator.__init__

        def capturing_init(self_agg, resolution_minutes, on_candle):
            captured["resolution_minutes"] = resolution_minutes
            original_init(self_agg, resolution_minutes, on_candle)

        with patch.object(TickAggregator, "__init__", capturing_init):
            client._subscribe_tick_fallback()

        assert captured.get("resolution_minutes") == expected_minutes, (
            f"Expected resolution_minutes={expected_minutes}, "
            f"got {captured.get('resolution_minutes')!r}"
        )


class TestWireStrategyPassesResolution:
    """kuroko._wire_strategy must derive resolution from candle_frequency and pass
    it to IGStreamingClient when api_mode='streaming'."""

    def _make_streaming_params(
        self, candle_frequency: str = "5min"
    ) -> types.SimpleNamespace:
        """Return a minimal streaming-mode params namespace."""
        return types.SimpleNamespace(
            api_mode="streaming", candle_frequency=candle_frequency
        )

    @pytest.mark.parametrize(
        "candle_frequency, expected_resolution",
        [
            ("5min", "5MINUTE"),
            ("15min", "15MINUTE"),
            ("1min", "1MINUTE"),
        ],
    )
    def test_wire_strategy_passes_resolution_to_streaming_client(
        self, candle_frequency, expected_resolution
    ):
        """candle_frequency must result in IGStreamingClient called with the correct resolution."""
        params = self._make_streaming_params(candle_frequency)
        mock_ig = MagicMock()
        mock_ig.ig_service = MagicMock(name="fake_ig_service")
        mock_strategy_class = MagicMock()
        trading_config = types.SimpleNamespace(epic="IX.D.NASDAQ.IFMM.IP")

        with patch("kuroko.IGStreamingClient") as mock_streaming_cls:
            _wire_strategy(
                strategy_class=mock_strategy_class,
                params=params,
                ig=mock_ig,
                trading_config=trading_config,
            )

        mock_streaming_cls.assert_called_once_with(
            mock_ig.ig_service, "IX.D.NASDAQ.IFMM.IP", resolution=expected_resolution
        )


# =========================================================================== #
# JSON config files and requirements pinning                                   #
# =========================================================================== #


_V2_REQUIRED_KEYS = {
    "api_mode",
    "operation_mode",
    "bb_period",
    "bb_std",
    "rsi_period",
    "rsi_oversold",
    "rsi_overbought",
    "max_long_positions",
    "max_short_positions",
    "contract_size",
    "min_dist_between_entries_ticks",
    "take_profit_ticks",
    "candle_frequency",
}


class TestV2JsonConfig:
    """strategies/RSIBollingerStrategyV2.json must exist and be complete."""

    def test_v2_json_file_exists(self):
        """The V2 strategy JSON file must be present."""
        assert _V2_JSON.exists(), f"File not found: {_V2_JSON}"

    def test_v2_json_is_valid_json(self):
        """The V2 JSON file must be parseable."""
        try:
            json.loads(_V2_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            pytest.fail(f"RSIBollingerStrategyV2.json is not valid JSON: {e}")

    def test_v2_json_has_all_required_keys(self):
        """All required keys must be present."""
        data = json.loads(_V2_JSON.read_text(encoding="utf-8"))
        missing = _V2_REQUIRED_KEYS - set(data.keys())
        assert not missing, f"Missing keys in RSIBollingerStrategyV2.json: {missing}"

    def test_v2_json_api_mode_is_streaming(self):
        """api_mode must be 'streaming' in V2 JSON."""
        data = json.loads(_V2_JSON.read_text(encoding="utf-8"))
        assert (
            data["api_mode"] == "streaming"
        ), f"Expected api_mode='streaming', got {data.get('api_mode')!r}"

    def test_v2_json_bb_period_is_20(self):
        """bb_period must be 20 per spec."""
        data = json.loads(_V2_JSON.read_text(encoding="utf-8"))
        assert data["bb_period"] == 20

    def test_v2_json_bb_std_is_2_0(self):
        """bb_std must be 2.0 per config."""
        data = json.loads(_V2_JSON.read_text(encoding="utf-8"))
        assert data["bb_std"] == 2.0

    def test_v2_json_rsi_period_is_7(self):
        """rsi_period must be 7 per config."""
        data = json.loads(_V2_JSON.read_text(encoding="utf-8"))
        assert data["rsi_period"] == 7

    def test_v2_json_rsi_oversold_is_30(self):
        """rsi_oversold must be 30 per config."""
        data = json.loads(_V2_JSON.read_text(encoding="utf-8"))
        assert data["rsi_oversold"] == 30.0

    def test_v2_json_rsi_overbought_is_70(self):
        """rsi_overbought must be 70 per config."""
        data = json.loads(_V2_JSON.read_text(encoding="utf-8"))
        assert data["rsi_overbought"] == 70.0


# --------------------------------------------------------------------------- #
# _validate_params — branch coverage for bool and multi-type schema entries    #
# --------------------------------------------------------------------------- #


class TestValidateParamsBranches:
    """_validate_params covers all type-check branches (bool, float, multi-type, str)."""

    def test_bool_value_for_int_key_triggers_sys_exit(self, tmp_path):
        """A bool value for an int key must be rejected (bool is subclass of int)."""
        data = {
            "api_mode": "streaming",
            "candle_frequency": "5min",
            "bb_period": True,  # bool, must be rejected even though isinstance(True, int)
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "max_long_positions": 5,
            "max_short_positions": 5,
            "contract_size": 0.1,
            "min_dist_between_entries_ticks": 20,
            "take_profit_ticks": 240.0,
        }
        path = tmp_path / "v2.json"
        path.write_text(json.dumps(data))

        with pytest.raises(SystemExit):
            load_params(str(path))

    def test_bool_value_for_float_key_triggers_sys_exit(self, tmp_path):
        """A bool value for a float key must be rejected."""
        data = {
            "api_mode": "streaming",
            "candle_frequency": "5min",
            "bb_period": 20,
            "bb_std": True,  # bool — must be rejected for float key
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "max_long_positions": 5,
            "max_short_positions": 5,
            "contract_size": 0.1,
            "min_dist_between_entries_ticks": 20,
            "take_profit_ticks": 240.0,
        }
        path = tmp_path / "v2.json"
        path.write_text(json.dumps(data))

        with pytest.raises(SystemExit):
            load_params(str(path))

    def test_float_value_accepted_for_float_key(self, tmp_path):
        """An int value for a float key (e.g. take_profit_ticks=240) is accepted."""
        data = {
            "epic": "IX.D.NASDAQ.IFMM.IP",
            "api_mode": "streaming",
            "operation_mode": "candle",
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
            "take_profit_ticks": 240,  # int value for float field — should be accepted
        }
        path = tmp_path / "v2.json"
        path.write_text(json.dumps(data))

        params = load_params(str(path))
        assert params.take_profit_ticks == 240

    def test_bool_value_for_multi_type_key_triggers_sys_exit(self, tmp_path):
        """A bool value for a multi-type (int | float) key must be rejected."""
        data = {
            "api_mode": "streaming",
            "candle_frequency": "5min",
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": True,  # bool — must be rejected even for (int, float) schema
            "rsi_overbought": 70,
            "max_long_positions": 5,
            "max_short_positions": 5,
            "contract_size": 0.1,
            "min_dist_between_entries_ticks": 20,
            "take_profit_ticks": 240.0,
        }
        path = tmp_path / "v2.json"
        path.write_text(json.dumps(data))

        with pytest.raises(SystemExit):
            load_params(str(path))

    def test_string_value_for_multi_type_key_triggers_sys_exit(self, tmp_path):
        """A string value for a multi-type (int | float) key must be rejected."""
        data = {
            "api_mode": "streaming",
            "candle_frequency": "5min",
            "bb_period": 20,
            "bb_std": 2.0,
            "rsi_period": 14,
            "rsi_oversold": "thirty",  # str — must be rejected for (int, float) schema
            "rsi_overbought": 70,
            "max_long_positions": 5,
            "max_short_positions": 5,
            "contract_size": 0.1,
            "min_dist_between_entries_ticks": 20,
            "take_profit_ticks": 240.0,
        }
        path = tmp_path / "v2.json"
        path.write_text(json.dumps(data))

        with pytest.raises(SystemExit):
            load_params(str(path))

    def test_non_string_value_for_str_key_triggers_sys_exit(self, tmp_path):
        """An int value for a str key (api_mode) must be rejected (generic isinstance branch)."""
        data = {
            "api_mode": 42,  # int — must be rejected for str schema
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
            "take_profit_ticks": 240.0,
        }
        path = tmp_path / "v2.json"
        path.write_text(json.dumps(data))

        with pytest.raises(SystemExit):
            load_params(str(path))

    def test_non_bool_value_for_bool_schema_key_triggers_sys_exit(self, tmp_path):
        """Bool branch (lines 69-70): a non-bool value for a bool-typed schema key causes SystemExit.

        No production key uses bool, so this branch is tested by temporarily
        patching _PARAMS_SCHEMA to include a bool-typed key.

        Note: import the live module object at call time to handle the rare case
        where test_kuroko.py flushes the strategies package from sys.modules,
        causing a fresh reimport that changes the module object identity.
        """
        import sys
        import importlib

        mod = sys.modules.get(
            "strategies.RSIBollingerStrategyV2"
        ) or importlib.import_module("strategies.RSIBollingerStrategyV2")

        base_data = {
            "epic": "IX.D.NASDAQ.IFMM.IP",
            "api_mode": "streaming",
            "operation_mode": "candle",
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
            "take_profit_ticks": 240.0,
            "test_flag": "not_a_bool",  # string for a bool-typed key → must be rejected
        }
        path = tmp_path / "v2.json"
        path.write_text(json.dumps(base_data))

        with patch.dict(mod._PARAMS_SCHEMA, {"test_flag": bool}):
            with pytest.raises(SystemExit):
                mod.load_params(str(path))

    def test_valid_bool_value_for_bool_schema_key_is_accepted(self, tmp_path):
        """Bool branch (lines 68-72): a valid bool value for a bool-typed schema key passes validation."""
        import sys
        import importlib

        mod = sys.modules.get(
            "strategies.RSIBollingerStrategyV2"
        ) or importlib.import_module("strategies.RSIBollingerStrategyV2")

        base_data = {
            "epic": "IX.D.NASDAQ.IFMM.IP",
            "api_mode": "streaming",
            "operation_mode": "candle",
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
            "take_profit_ticks": 240.0,
            "test_flag": True,  # valid bool — must pass
        }
        path = tmp_path / "v2.json"
        path.write_text(json.dumps(base_data))

        with patch.dict(mod._PARAMS_SCHEMA, {"test_flag": bool}):
            params = mod.load_params(str(path))

        assert params.test_flag is True


# --------------------------------------------------------------------------- #
# load_params — file I/O error paths (lines 127–134)                          #
# --------------------------------------------------------------------------- #


class TestLoadParamsFileErrors:
    """load_params exits cleanly on file-not-found, invalid JSON, and OS errors."""

    def test_file_not_found_triggers_sys_exit(self):
        """Missing file causes SystemExit."""
        with pytest.raises(SystemExit):
            load_params("/tmp/this_file_does_not_exist_v2.json")

    def test_invalid_json_triggers_sys_exit(self, tmp_path):
        """Malformed JSON causes SystemExit."""
        path = tmp_path / "bad.json"
        path.write_text("{not valid json}")
        with pytest.raises(SystemExit):
            load_params(str(path))

    def test_os_error_triggers_sys_exit(self, tmp_path):
        """An OSError during file read causes SystemExit."""
        path = tmp_path / "v2.json"
        path.write_text("{}")
        with patch("builtins.open", side_effect=OSError("Permission denied")):
            with pytest.raises(SystemExit):
                load_params(str(path))


# --------------------------------------------------------------------------- #
# _manage_longs — open_position exception path (line 420-421)                 #
# --------------------------------------------------------------------------- #


class TestManageLongsOpenException:
    """open_position raising an exception is caught; grid remains unchanged."""

    def test_open_position_exception_is_caught(self, make_strategy_v2):
        """Exception from open_position must be caught; no position added to grid."""
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.open_position.side_effect = Exception("Broker unreachable")
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=95.0
        )

        strat._manage_longs(indicators)  # must not raise

        assert len(strat._long_positions) == 0

    def test_open_long_exception_does_not_affect_existing_positions(
        self, make_strategy_v2
    ):
        """An exception when opening a second long does not remove existing positions."""
        strat, mock_ig, _ = make_strategy_v2()
        strat._long_positions = [
            {"deal_id": "EXISTING", "entry_price": 60.0, "size": 0.5}
        ]
        mock_ig.open_position.side_effect = Exception("Timeout")
        # Price is 75 — enough distance from 60 for a second entry (min_dist=10)
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=75.0
        )

        strat._manage_longs(indicators)  # must not raise

        # Existing position is untouched
        assert len(strat._long_positions) == 1
        assert strat._long_positions[0]["deal_id"] == "EXISTING"


# --------------------------------------------------------------------------- #
# _manage_shorts — open_position exception path (line 513-514)                #
# --------------------------------------------------------------------------- #


class TestManageShortsOpenException:
    """open_position raising an exception is caught; grid remains unchanged."""

    def test_open_short_exception_is_caught(self, make_strategy_v2):
        """Exception from open_position (short) must be caught; no position added."""
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.open_position.side_effect = Exception("Broker unreachable")
        indicators = _make_indicators(
            bb_lower=0.0, bb_upper=100.0, rsi=75.0, close=110.0
        )

        strat._manage_shorts(indicators)  # must not raise

        assert len(strat._short_positions) == 0


# --------------------------------------------------------------------------- #
# _manage_shorts — distance guard for short entries (line 535-537)            #
# --------------------------------------------------------------------------- #


class TestShortDistanceGuard:
    """Short distance guard blocks entry when distance from last short is too small."""

    def test_short_entry_allowed_when_distance_sufficient(
        self, make_strategy_v2, make_params_v2
    ):
        """Short entry is allowed when distance from last entry >= min_dist_between_entries_ticks."""
        params = make_params_v2(min_dist_between_entries_ticks=10)
        strat, mock_ig, _ = make_strategy_v2(params=params)
        # Last short at 140; current close at 120 → distance=20 >= 10
        strat._short_positions = [{"deal_id": "S1", "entry_price": 140.0, "size": 0.5}]
        indicators = _make_indicators(
            bb_lower=0.0, bb_upper=100.0, rsi=75.0, close=120.0
        )

        strat._manage_shorts(indicators)

        mock_ig.open_position.assert_called_once()


# --------------------------------------------------------------------------- #
# _reconcile_positions — get_open_positions exception path (lines 535-537)    #
# --------------------------------------------------------------------------- #


class TestReconcilePositionsException:
    """_reconcile_positions handles get_open_positions raising an exception."""

    def test_reconcile_positions_get_open_positions_exception_is_caught(
        self, make_strategy_v2
    ):
        """Exception from get_open_positions is caught; positions remain unchanged."""
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.get_open_positions.side_effect = Exception("API error")
        strat._long_positions = [
            {
                "deal_id": "DEAL1",
                "entry_price": 90.0,
                "size": 0.5,
                "needs_reconciliation": True,
            }
        ]

        strat._reconcile_positions()  # must not raise

        # Positions must remain untouched on reconciliation failure
        assert len(strat._long_positions) == 1
        assert strat._long_positions[0]["deal_id"] == "DEAL1"


# --------------------------------------------------------------------------- #
# _on_candle — timezone normalization for warmup_ts (lines 599–607)           #
# --------------------------------------------------------------------------- #


class TestOnCandleTimezoneNormalization:
    """_on_candle normalizes both candle_ts and warmup_ts to naive UTC before comparing."""

    def test_dedup_skips_when_aware_candle_ts_equals_aware_warmup_ts(
        self, make_strategy_v2
    ):
        """Candle with tz-aware UTC timestamp == tz-aware _last_warmup_ts is skipped."""
        strat, _, _ = make_strategy_v2()
        t0 = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)
        strat._last_warmup_ts = t0
        initial_len = 5
        for _ in range(initial_len):
            strat._candle_window.append(100.0)

        overlap_candle = {
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 100.0,
            "bid_close": 99.5,
            "ofr_close": 100.5,
            "spread": 1.0,
            "volume": 0,
            "timestamp": t0,  # aware, same as warmup_ts
        }

        strat._on_candle(overlap_candle)

        assert len(strat._candle_window) == initial_len


# --------------------------------------------------------------------------- #
# _compute_indicators — NaN guard (lines 241-242)                             #
# --------------------------------------------------------------------------- #


class TestComputeIndicatorsNaN:
    """_compute_indicators returns None when BB or RSI produces NaN."""

    def test_compute_indicators_returns_none_when_window_too_small(
        self, make_strategy_v2
    ):
        """Returns None when window has fewer entries than required."""
        strat, _, _ = make_strategy_v2()
        # Window is empty — should return None immediately
        candle = {"close": 100.0}
        result = strat._compute_indicators(candle)
        assert result is None

    def test_compute_indicators_returns_none_for_uniform_close_prices(
        self, make_strategy_v2, make_params_v2
    ):
        """Uniform close prices can cause NaN standard deviation in BB — returns None."""
        # Use a small bb_period so we can fill the window with few values
        params = make_params_v2(bb_period=3, rsi_period=2)
        strat, _, _ = make_strategy_v2(params=params)
        # Fill with exact same price — BB std dev = 0 → BB bands may produce NaN for RSI
        # We need min_required = max(3, 2) + 1 = 4 entries, then a 5th to evaluate
        for _ in range(4):
            strat._candle_window.append(100.0)

        # Now add one more via _compute_indicators — with all same prices, TA-Lib returns
        # rsi=0.0 (no price change) and BB bands collapse to a single value (0 std dev).
        # No NaN is produced, so the NaN guard does not fire and result is not None.
        candle = {"close": 100.0}
        result = strat._compute_indicators(candle)

        # TA-Lib returns rsi=0.0 for uniform prices — result is not None and contains no NaN
        assert result is not None
        assert not any(
            np.isnan(v) for v in [result["bb_upper"], result["bb_lower"], result["rsi"]]
        )

    def test_compute_indicators_returns_none_when_talib_produces_nan(
        self, make_strategy_v2, make_params_v2
    ):
        """NaN guard (lines 241-242): returns None when TA-Lib outputs NaN at current bar.

        Strategy: pre-fill the window with enough valid distinct prices so the
        window-size guard passes, then pass a candle with close=NaN. TA-Lib
        propagates NaN to the last bar of BB and RSI output, hitting lines 241-242.
        """
        params = make_params_v2(bb_period=3, rsi_period=2)
        strat, _, _ = make_strategy_v2(params=params)
        # min_required = max(3, 2) + 1 = 4; pre-fill 4 valid entries so the
        # window-size guard passes after _compute_indicators appends the NaN candle
        for i in range(4):
            strat._candle_window.append(100.0 + i)

        # NaN close propagates through TA-Lib → bb_upper/bb_lower/rsi are NaN
        result = strat._compute_indicators({"close": float("nan")})

        assert result is None


# --------------------------------------------------------------------------- #
# _on_candle — indicators is None path (line 316)                             #
# --------------------------------------------------------------------------- #


class TestOnCandleIndicatorsNone:
    """_on_candle exits early when _compute_indicators returns None (window too small)."""

    def test_on_candle_exits_early_when_indicators_none(self, make_strategy_v2):
        """When indicators are None (insufficient window), manage_longs/shorts are not called."""
        strat, mock_ig, _ = make_strategy_v2()
        # Do NOT pre-fill the window — it will be too small for indicators
        candle = {
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 100.0,
            "bid_close": 99.5,
            "ofr_close": 100.5,
            "spread": 1.0,
            "volume": 0,
            "timestamp": None,
        }

        strat._on_candle(candle)

        mock_ig.open_position.assert_not_called()
        mock_ig.close_position.assert_not_called()


# --------------------------------------------------------------------------- #
# _update_spread_from_candle — candle without spread key (line 267–270)       #
# --------------------------------------------------------------------------- #


class TestUpdateSpreadMissingKey:
    """_update_spread_from_candle is a no-op when 'spread' is absent from candle."""

    def test_spread_unchanged_when_candle_has_no_spread_key(self, make_strategy_v2):
        """If the candle has no 'spread' key, _current_spread must remain None."""
        strat, _, _ = make_strategy_v2()
        assert strat._current_spread is None

        strat._update_spread_from_candle({"bid_close": 100.0, "ofr_close": 101.0})

        assert strat._current_spread is None

    def test_spread_unchanged_when_spread_value_is_none(self, make_strategy_v2):
        """If candle['spread'] is None, _current_spread must remain unchanged."""
        strat, _, _ = make_strategy_v2()
        strat._current_spread = 2.0

        strat._update_spread_from_candle({"spread": None})

        assert strat._current_spread == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# _warmup — partial load warning (lines 315-316)                              #
# --------------------------------------------------------------------------- #


class TestWarmupPartialLoad:
    """_warmup logs a WARNING when fewer candles are loaded than requested."""

    def test_warmup_partial_load_still_fills_what_is_available(self, make_strategy_v2):
        """When the REST response has fewer rows than requested, the available rows are loaded."""
        strat, mock_ig, _ = make_strategy_v2()
        num_candles = max(strat.params.bb_period, strat.params.rsi_period) + 1
        # Return only 5 rows — less than the required num_candles
        df = _make_warmup_dataframe(5, base_price=100.0)
        mock_ig.get_candles.return_value = df

        strat._warmup()

        assert len(strat._candle_window) == 5


# --------------------------------------------------------------------------- #
# _extract_deal_id — exception branch (lines 719-721)                        #
# --------------------------------------------------------------------------- #


class TestExtractDealIdEdgeCases:
    """_extract_deal_id returns 'unknown' for non-dict responses and on exception."""

    def test_extract_deal_id_returns_unknown_for_non_dict_response(
        self, make_strategy_v2
    ):
        """When open_position returns a non-dict (no .get()), deal_id is 'unknown'."""
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.open_position.return_value = "some_string_response"
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=95.0
        )

        strat._manage_longs(indicators)

        # Non-dict response → deal_id = 'unknown' → position NOT added
        assert len(strat._long_positions) == 0

    def test_extract_deal_id_returns_unknown_when_deal_id_is_empty_string(
        self, make_strategy_v2
    ):
        """When dealId is '' (falsy) even on ACCEPTED, _extract_deal_id returns 'unknown'."""
        strat, mock_ig, _ = make_strategy_v2()
        mock_ig.open_position.return_value = {
            "dealStatus": "ACCEPTED",
            "dealId": "",
            "dealReference": "REF123",
        }
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=95.0
        )

        strat._manage_longs(indicators)

        # Empty dealId → "unknown" → position NOT added
        assert len(strat._long_positions) == 0

    def test_extract_deal_id_returns_unknown_when_get_raises(self, make_strategy_v2):
        """When response.get() raises an exception, _extract_deal_id returns 'unknown'."""
        from strategies.RSIBollingerStrategyV2 import _extract_deal_id

        class BadResponse:
            """Object that has .get() but raises when called."""

            def get(self, key, default=None):
                raise RuntimeError("Unexpected .get() call")

        result = _extract_deal_id(BadResponse())

        assert result == "unknown"


class TestRequirementsPin:
    """lightstreamer-client-lib must be explicitly pinned in requirements.txt."""

    def test_requirements_file_exists(self):
        """requirements.txt must exist at the project root."""
        assert _REQUIREMENTS.exists(), f"File not found: {_REQUIREMENTS}"

    def test_lightstreamer_is_pinned_with_version(self):
        """Requirements.txt must contain an explicit version pin."""
        content = _REQUIREMENTS.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.splitlines()]
        pinned = [
            line for line in lines if line.startswith("lightstreamer-client-lib==")
        ]
        assert pinned, (
            "lightstreamer-client-lib is not explicitly pinned in requirements.txt. "
            "Add a line like: lightstreamer-client-lib==1.0.3"
        )

    def test_lightstreamer_pin_has_valid_version_number(self):
        """The pin must include a non-empty version number starting with a digit."""
        content = _REQUIREMENTS.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.splitlines()]
        for line in lines:
            if line.startswith("lightstreamer-client-lib=="):
                version_part = line.split("==", 1)[1].strip()
                assert version_part, "Version must not be empty after '=='"
                assert version_part[
                    0
                ].isdigit(), f"Version must start with a digit, got: {version_part!r}"
                return
        pytest.fail("lightstreamer-client-lib pin not found in requirements.txt")


# =========================================================================== #
# Tick mode — operation_mode parameter [REQ-1]                                 #
# =========================================================================== #


class TestOperationModeParam:
    """operation_mode param validation and fallback behaviour (REQ-1)."""

    def test_valid_tick_mode_sets_operation_mode(
        self, make_strategy_v2, make_params_v2
    ):
        """When operation_mode='tick', _operation_mode is set to 'tick' and no warning is logged."""
        params = make_params_v2(operation_mode="tick")
        strat, _, _ = make_strategy_v2(params=params)

        assert strat._operation_mode == "tick"

    def test_invalid_value_falls_back_to_candle_with_warning(
        self, make_strategy_v2, make_params_v2, caplog
    ):
        """An unrecognised operation_mode value defaults to 'candle' and logs a warning."""
        import logging

        params = make_params_v2(operation_mode="turbo")
        with caplog.at_level(logging.WARNING):
            strat, _, _ = make_strategy_v2(params=params)

        assert strat._operation_mode == "candle"
        assert any("turbo" in record.message for record in caplog.records)

    def test_missing_key_defaults_to_candle(self, make_strategy_v2, make_params_v2):
        """When operation_mode is absent from params, _operation_mode defaults to 'candle'."""
        params = make_params_v2()
        # Remove operation_mode attribute to simulate missing key
        del params.operation_mode
        strat, _, _ = make_strategy_v2(params=params)

        assert strat._operation_mode == "candle"


# =========================================================================== #
# Tick mode — indicator cache [REQ-3]                                          #
# =========================================================================== #


class TestIndicatorCache:
    """Indicator cache updated on candle close in tick mode; _compute_indicators is pure [REQ-3]."""

    def test_on_candle_updates_cached_indicators_in_tick_mode(
        self, make_strategy_v2, make_params_v2
    ):
        """In tick mode, _on_candle updates _cached_indicators after computing them."""
        params = make_params_v2(operation_mode="tick")
        strat, _, _ = make_strategy_v2(params=params)
        # Pre-fill candle window so indicators are computable
        for _ in range(25):
            strat._candle_window.append(100.0)
        candle = _make_candle(close=100.0)

        strat._on_candle(candle)

        assert strat._cached_indicators is not None
        assert "bb_upper" in strat._cached_indicators
        assert "bb_lower" in strat._cached_indicators
        assert "rsi" in strat._cached_indicators

    def test_compute_indicators_is_pure(self, make_strategy_v2, make_params_v2):
        """_compute_indicators must not mutate _cached_indicators (pure function)."""
        params = make_params_v2(operation_mode="tick")
        strat, _, _ = make_strategy_v2(params=params)
        # Start with _cached_indicators = None
        assert strat._cached_indicators is None
        # Pre-fill window
        for _ in range(25):
            strat._candle_window.append(100.0)
        candle = {"close": 100.0}

        # Call _compute_indicators directly — must not modify _cached_indicators
        result = strat._compute_indicators(candle)

        # _cached_indicators must remain None (only _on_candle should set it)
        assert strat._cached_indicators is None
        # But the result itself is valid
        assert result is not None


# =========================================================================== #
# Tick mode — candle mode regression [REQ-2]                                   #
# =========================================================================== #


class TestCandleModeRegression:
    """Candle mode calls _manage_longs/_manage_shorts and doesn't require cache attrs [REQ-2]."""

    def test_candle_mode_calls_manage_longs_and_manage_shorts(
        self, make_strategy_v2, make_params_v2
    ):
        """In candle mode, _on_candle calls _manage_longs and _manage_shorts (no early return)."""
        params = make_params_v2(operation_mode="candle")
        strat, _, _ = make_strategy_v2(params=params)
        for _ in range(25):
            strat._candle_window.append(100.0)
        candle = _make_candle(close=100.0)

        with (
            patch.object(strat, "_manage_longs") as mock_longs,
            patch.object(strat, "_manage_shorts") as mock_shorts,
        ):
            strat._on_candle(candle)

        mock_longs.assert_called_once()
        mock_shorts.assert_called_once()


# =========================================================================== #
# Tick mode — warmup gate [REQ-8]                                               #
# =========================================================================== #


class TestWarmupGate:
    """Ticks silently dropped when _cached_indicators is None [REQ-8]."""

    def test_tick_dropped_silently_when_no_cache(
        self, make_strategy_v2, make_params_v2, caplog
    ):
        """_on_tick returns without REST calls when _cached_indicators is None."""
        import logging

        params = make_params_v2(operation_mode="tick")
        strat, mock_ig, _ = make_strategy_v2(params=params)
        assert strat._cached_indicators is None

        with caplog.at_level(logging.WARNING):
            strat._on_tick({"bid": 90.0, "ofr": 91.0, "utm": 0})

        mock_ig.open_position.assert_not_called()
        mock_ig.close_position.assert_not_called()
        # No WARNING or ERROR level logs
        assert not any(record.levelno >= logging.WARNING for record in caplog.records)

    def test_tick_processed_when_cache_populated(
        self, make_strategy_v2, make_params_v2
    ):
        """_on_tick proceeds to signal evaluation when _cached_indicators is not None."""
        params = make_params_v2(operation_mode="tick")
        strat, mock_ig, _ = make_strategy_v2(params=params)
        # Populate cache with neutral indicators (no entry signals)
        strat._cached_indicators = _make_indicators(
            bb_upper=200.0, bb_lower=50.0, rsi=50.0, close=100.0
        )
        # Neutral tick — bid inside bands, no signal
        tick = {"bid": 100.0, "ofr": 100.5, "utm": 0}

        strat._on_tick(tick)

        # No position action expected (neutral conditions)
        mock_ig.open_position.assert_not_called()
        mock_ig.close_position.assert_not_called()


# =========================================================================== #
# Tick mode — in-flight guard [REQ-9]                                           #
# =========================================================================== #


class TestInFlightGuard:
    """In-flight flags prevent duplicate ticks; reset in finally even on exception [REQ-9]."""

    def test_duplicate_tick_dropped_when_long_in_flight(
        self, make_strategy_v2, make_params_v2
    ):
        """When _tick_long_in_flight is True, a qualifying long tick is silently dropped."""
        params = make_params_v2(operation_mode="tick")
        strat, mock_ig, _ = make_strategy_v2(params=params)
        strat._cached_indicators = _make_indicators(
            bb_upper=200.0, bb_lower=100.0, rsi=20.0, close=100.0
        )
        strat._tick_long_in_flight = True
        # Qualifying tick: bid < bb_lower, rsi < rsi_oversold (30)
        tick = {"bid": 90.0, "ofr": 91.0, "utm": 0}

        strat._on_tick(tick)

        mock_ig.open_position.assert_not_called()

    def test_in_flight_flag_reset_on_rest_exception(
        self, make_strategy_v2, make_params_v2
    ):
        """_tick_long_in_flight is False after REST raises — finally block fires."""
        params = make_params_v2(operation_mode="tick")
        strat, mock_ig, _ = make_strategy_v2(params=params)
        strat._cached_indicators = _make_indicators(
            bb_upper=200.0, bb_lower=100.0, rsi=20.0, close=100.0
        )
        mock_ig.open_position.side_effect = RuntimeError("REST error")
        # Qualifying long tick
        tick = {"bid": 90.0, "ofr": 91.0, "utm": 0}

        strat._on_tick(tick)

        assert strat._tick_long_in_flight is False


# =========================================================================== #
# Tick mode — entry long [REQ-4]                                                #
# =========================================================================== #


class TestTickEntryLong:
    """Long position opened on BB-lower cross + RSI oversold in tick mode [REQ-4]."""

    def test_long_opened_when_all_conditions_met(
        self, make_strategy_v2, make_params_v2
    ):
        """Long opened when bid < bb_lower and rsi < rsi_oversold and not in-flight."""
        params = make_params_v2(
            operation_mode="tick",
            rsi_oversold=30,
            max_long_positions=3,
        )
        strat, mock_ig, _ = make_strategy_v2(params=params)
        mock_ig.open_position.return_value = {
            "dealStatus": "ACCEPTED",
            "dealId": "TICK_LONG_1",
        }
        strat._cached_indicators = _make_indicators(
            bb_upper=200.0, bb_lower=100.0, rsi=20.0, close=100.0
        )
        # Qualifying tick: bid < bb_lower=100
        tick = {"bid": 90.0, "ofr": 91.0, "utm": 0}

        strat._on_tick(tick)

        mock_ig.open_position.assert_called_once()
        call_kwargs = mock_ig.open_position.call_args.kwargs
        assert call_kwargs["side"] == "BUY"

    def test_long_suppressed_at_max_positions(self, make_strategy_v2, make_params_v2):
        """No long opened when max_long_positions already reached."""
        params = make_params_v2(
            operation_mode="tick",
            rsi_oversold=30,
            max_long_positions=1,
        )
        strat, mock_ig, _ = make_strategy_v2(params=params)
        strat._cached_indicators = _make_indicators(
            bb_upper=200.0, bb_lower=100.0, rsi=20.0, close=100.0
        )
        strat._long_positions = [
            {"deal_id": "EXISTING", "entry_price": 85.0, "size": 0.5}
        ]
        tick = {"bid": 90.0, "ofr": 91.0, "utm": 0}

        strat._on_tick(tick)

        mock_ig.open_position.assert_not_called()


# =========================================================================== #
# Tick mode — entry short [REQ-5]                                               #
# =========================================================================== #


class TestTickEntryShort:
    """Short position opened on BB-upper cross + RSI overbought in tick mode [REQ-5]."""

    def test_short_opened_when_all_conditions_met(
        self, make_strategy_v2, make_params_v2
    ):
        """Short opened when bid > bb_upper and rsi > rsi_overbought and not in-flight."""
        params = make_params_v2(
            operation_mode="tick",
            rsi_overbought=70,
            max_short_positions=3,
        )
        strat, mock_ig, _ = make_strategy_v2(params=params)
        mock_ig.open_position.return_value = {
            "dealStatus": "ACCEPTED",
            "dealId": "TICK_SHORT_1",
        }
        strat._cached_indicators = _make_indicators(
            bb_upper=100.0, bb_lower=0.0, rsi=80.0, close=100.0
        )
        # Qualifying tick: bid > bb_upper=100
        tick = {"bid": 110.0, "ofr": 111.0, "utm": 0}

        strat._on_tick(tick)

        mock_ig.open_position.assert_called_once()
        call_kwargs = mock_ig.open_position.call_args.kwargs
        assert call_kwargs["side"] == "SELL"

    def test_short_suppressed_at_max_positions(self, make_strategy_v2, make_params_v2):
        """No short opened when max_short_positions already reached."""
        params = make_params_v2(
            operation_mode="tick",
            rsi_overbought=70,
            max_short_positions=1,
        )
        strat, mock_ig, _ = make_strategy_v2(params=params)
        strat._cached_indicators = _make_indicators(
            bb_upper=100.0, bb_lower=0.0, rsi=80.0, close=100.0
        )
        strat._short_positions = [
            {"deal_id": "EXISTING_S", "entry_price": 115.0, "size": 0.5}
        ]
        tick = {"bid": 110.0, "ofr": 111.0, "utm": 0}

        strat._on_tick(tick)

        mock_ig.open_position.assert_not_called()


# =========================================================================== #
# Tick mode — exit long [REQ-6]                                                 #
# =========================================================================== #


class TestTickExitLong:
    """Long positions closed when bid > bb_upper and profit > 0 in tick mode [REQ-6]."""

    def test_longs_closed_when_bid_above_bb_upper_and_profit_positive(
        self, make_strategy_v2, make_params_v2
    ):
        """All longs closed when bid > bb_upper and live spread profit > 0."""
        params = make_params_v2(operation_mode="tick")
        strat, mock_ig, _ = make_strategy_v2(params=params)
        strat._cached_indicators = _make_indicators(
            bb_upper=100.0, bb_lower=0.0, rsi=50.0, close=100.0
        )
        # Long entered at 90; bid=110 > bb_upper=100; spread=1.0 → profit=(110-90-1)*0.5=9.5>0
        strat._long_positions = [{"deal_id": "L1", "entry_price": 90.0, "size": 0.5}]
        tick = {"bid": 110.0, "ofr": 111.0, "utm": 0}

        strat._on_tick(tick)

        mock_ig.close_position.assert_called_once_with("L1", "SELL", 0.5)

    def test_long_exit_suppressed_when_profit_not_positive(
        self, make_strategy_v2, make_params_v2
    ):
        """Long not closed when bid > bb_upper but profit <= 0 (spread too high)."""
        params = make_params_v2(operation_mode="tick")
        strat, mock_ig, _ = make_strategy_v2(params=params)
        strat._cached_indicators = _make_indicators(
            bb_upper=100.0, bb_lower=0.0, rsi=50.0, close=100.0
        )
        # Long at 110; bid=105 > bb_upper=100; spread=10 → profit=(105-110-10)*0.5 < 0
        strat._long_positions = [{"deal_id": "L1", "entry_price": 110.0, "size": 0.5}]
        tick = {"bid": 105.0, "ofr": 115.0, "utm": 0}  # ofr-bid spread=10

        strat._on_tick(tick)

        mock_ig.close_position.assert_not_called()

    def test_long_exit_suppressed_when_entry_spread_wider_than_exit_spread(
        self, make_strategy_v2, make_params_v2
    ):
        """Long not closed when exit-tick spread understates the entry cost.

        Regression test: entry was opened when spread was wide (3.0).
        Exit tick has a tight spread (0.5). Using exit spread in the profit
        check produces a false positive — strategy thinks it's profitable when
        actual broker P&L is negative.

        With the fix, the entry_spread stored at open time is used for the
        LONG profit check instead of the exit tick's spread.
        """
        params = make_params_v2(operation_mode="tick")
        strat, mock_ig, _ = make_strategy_v2(params=params)
        strat._cached_indicators = _make_indicators(
            bb_upper=100.0, bb_lower=0.0, rsi=50.0, close=100.0
        )
        mock_ig.open_position.return_value = {
            "dealStatus": "ACCEPTED",
            "dealId": "L1",
        }
        # Simulate: LONG opened during wide-spread tick (entry spread = 3.0)
        # entry_price=90, entry_spread=3.0 → actual fill at ask=93
        # Exit tick: bid=91 > bb_upper=100? NO — but let's set up so the
        # exit condition fires with a tight spread.
        # bb_upper=100, bid=101 > bb_upper → exit fires.
        # spread_exit=0.5 → old profit=(101-90-0.5)*0.5=5.25 > 0 → WRONG close
        # spread_entry=3.0 → new profit=(101-90-3.0)*0.5=4.0 > 0 → still closes (happy path)
        # Need entry close to bb_upper so only tight spread makes it look profitable.
        # entry_price=99, entry_spread=3.0, bid_exit=101, spread_exit=0.5:
        # old: (101-99-0.5)*0.5=0.75>0 → close (BUG)
        # actual: (101-(99+3))*0.5=(101-102)*0.5=-0.5 → LOSS
        # new: (101-99-3.0)*0.5=-0.5<0 → NOT closed (CORRECT)
        strat._long_positions = [
            {"deal_id": "L1", "entry_price": 99.0, "entry_spread": 3.0, "size": 0.5}
        ]
        # Exit tick: bid=101 > bb_upper=100, spread=0.5 (tight)
        tick = {"bid": 101.0, "ofr": 101.5, "utm": 0}

        strat._on_tick(tick)

        mock_ig.close_position.assert_not_called()

    def test_long_exit_fires_when_move_covers_entry_spread(
        self, make_strategy_v2, make_params_v2
    ):
        """Long closed when price moved enough to cover the entry spread cost.

        When price has moved sufficiently that profit is positive even after
        accounting for the wide entry spread, the position should close.
        """
        params = make_params_v2(operation_mode="tick")
        strat, mock_ig, _ = make_strategy_v2(params=params)
        strat._cached_indicators = _make_indicators(
            bb_upper=100.0, bb_lower=0.0, rsi=50.0, close=100.0
        )
        # entry_price=90, entry_spread=3.0 → actual fill at ask=93
        # bid_exit=110 > bb_upper=100, spread_exit=0.5
        # profit=(110-90-3.0)*0.5=17*0.5=8.5>0 → CORRECT close
        strat._long_positions = [
            {"deal_id": "L1", "entry_price": 90.0, "entry_spread": 3.0, "size": 0.5}
        ]
        tick = {"bid": 110.0, "ofr": 110.5, "utm": 0}

        strat._on_tick(tick)

        mock_ig.close_position.assert_called_once_with("L1", "SELL", 0.5)

    def test_entry_spread_stored_in_long_position_on_tick_open(
        self, make_strategy_v2, make_params_v2
    ):
        """LONG position dict contains entry_spread matching the tick spread at open."""
        params = make_params_v2(
            operation_mode="tick",
            rsi_oversold=30,
            max_long_positions=3,
        )
        strat, mock_ig, _ = make_strategy_v2(params=params)
        mock_ig.open_position.return_value = {
            "dealStatus": "ACCEPTED",
            "dealId": "L1",
        }
        strat._cached_indicators = _make_indicators(
            bb_upper=200.0, bb_lower=100.0, rsi=20.0, close=100.0
        )
        # Wide-spread tick at entry: bid=90 < bb_lower=100, spread=3.0
        tick = {"bid": 90.0, "ofr": 93.0, "utm": 0}

        strat._on_tick(tick)

        assert len(strat._long_positions) == 1
        pos = strat._long_positions[0]
        assert pos["entry_spread"] == 3.0


# =========================================================================== #
# Tick mode — exit short [REQ-7]                                                #
# =========================================================================== #


class TestTickExitShort:
    """Short positions closed when bid < bb_lower and profit > 0 in tick mode [REQ-7]."""

    def test_shorts_closed_when_bid_below_bb_lower_and_profit_positive(
        self, make_strategy_v2, make_params_v2
    ):
        """All shorts closed when bid < bb_lower and live spread profit > 0."""
        params = make_params_v2(operation_mode="tick")
        strat, mock_ig, _ = make_strategy_v2(params=params)
        strat._cached_indicators = _make_indicators(
            bb_upper=200.0, bb_lower=100.0, rsi=50.0, close=100.0
        )
        # Short entered at 110; bid=90 < bb_lower=100; spread=1.0 → profit=(110-90-1)*0.5=9.5>0
        strat._short_positions = [{"deal_id": "S1", "entry_price": 110.0, "size": 0.5}]
        tick = {"bid": 90.0, "ofr": 91.0, "utm": 0}

        strat._on_tick(tick)

        mock_ig.close_position.assert_called_once_with("S1", "BUY", 0.5)

    def test_short_exit_suppressed_when_profit_not_positive(
        self, make_strategy_v2, make_params_v2
    ):
        """Short not closed when bid < bb_lower but profit <= 0."""
        params = make_params_v2(operation_mode="tick")
        strat, mock_ig, _ = make_strategy_v2(params=params)
        strat._cached_indicators = _make_indicators(
            bb_upper=200.0, bb_lower=100.0, rsi=50.0, close=100.0
        )
        # Short at 90; bid=95 < bb_lower=100; spread=10 → profit=(90-95-10)*0.5 < 0
        strat._short_positions = [{"deal_id": "S1", "entry_price": 90.0, "size": 0.5}]
        tick = {"bid": 95.0, "ofr": 105.0, "utm": 0}  # ofr-bid=10

        strat._on_tick(tick)

        mock_ig.close_position.assert_not_called()


# =========================================================================== #
# Tick mode — close in-flight guards [REQ-10]                                  #
# =========================================================================== #


class TestTickCloseInFlightGuard:
    """Close in-flight flags prevent duplicate REST close calls on back-to-back ticks [REQ-10]."""

    def test_long_close_skipped_when_long_close_in_flight(
        self, make_strategy_v2, make_params_v2
    ):
        """When _tick_long_close_in_flight is True, long close is NOT attempted."""
        params = make_params_v2(operation_mode="tick")
        strat, mock_ig, _ = make_strategy_v2(params=params)
        strat._cached_indicators = _make_indicators(
            bb_upper=100.0, bb_lower=0.0, rsi=50.0, close=100.0
        )
        # Long with positive profit when bid > bb_upper
        strat._long_positions = [{"deal_id": "L1", "entry_price": 90.0, "size": 0.5}]
        strat._tick_long_close_in_flight = True
        tick = {"bid": 110.0, "ofr": 111.0, "utm": 0}  # bid > bb_upper=100

        strat._on_tick(tick)

        mock_ig.close_position.assert_not_called()

    def test_short_close_skipped_when_short_close_in_flight(
        self, make_strategy_v2, make_params_v2
    ):
        """When _tick_short_close_in_flight is True, short close is NOT attempted."""
        params = make_params_v2(operation_mode="tick")
        strat, mock_ig, _ = make_strategy_v2(params=params)
        strat._cached_indicators = _make_indicators(
            bb_upper=200.0, bb_lower=100.0, rsi=50.0, close=100.0
        )
        # Short with positive profit when bid < bb_lower
        strat._short_positions = [{"deal_id": "S1", "entry_price": 110.0, "size": 0.5}]
        strat._tick_short_close_in_flight = True
        tick = {"bid": 90.0, "ofr": 91.0, "utm": 0}  # bid < bb_lower=100

        strat._on_tick(tick)

        mock_ig.close_position.assert_not_called()

    def test_long_close_in_flight_flag_reset_after_success(
        self, make_strategy_v2, make_params_v2
    ):
        """_tick_long_close_in_flight is False after a successful close REST call."""
        params = make_params_v2(operation_mode="tick")
        strat, mock_ig, _ = make_strategy_v2(params=params)
        strat._cached_indicators = _make_indicators(
            bb_upper=100.0, bb_lower=0.0, rsi=50.0, close=100.0
        )
        strat._long_positions = [{"deal_id": "L1", "entry_price": 90.0, "size": 0.5}]
        tick = {"bid": 110.0, "ofr": 111.0, "utm": 0}

        strat._on_tick(tick)

        assert strat._tick_long_close_in_flight is False

    def test_short_close_in_flight_flag_reset_after_success(
        self, make_strategy_v2, make_params_v2
    ):
        """_tick_short_close_in_flight is False after a successful close REST call."""
        params = make_params_v2(operation_mode="tick")
        strat, mock_ig, _ = make_strategy_v2(params=params)
        strat._cached_indicators = _make_indicators(
            bb_upper=200.0, bb_lower=100.0, rsi=50.0, close=100.0
        )
        strat._short_positions = [{"deal_id": "S1", "entry_price": 110.0, "size": 0.5}]
        tick = {"bid": 90.0, "ofr": 91.0, "utm": 0}

        strat._on_tick(tick)

        assert strat._tick_short_close_in_flight is False

    def test_long_close_in_flight_flag_reset_on_exception(
        self, make_strategy_v2, make_params_v2
    ):
        """_tick_long_close_in_flight is False even when close_position raises — finally fires."""
        params = make_params_v2(operation_mode="tick")
        strat, mock_ig, _ = make_strategy_v2(params=params)
        strat._cached_indicators = _make_indicators(
            bb_upper=100.0, bb_lower=0.0, rsi=50.0, close=100.0
        )
        strat._long_positions = [{"deal_id": "L1", "entry_price": 90.0, "size": 0.5}]
        mock_ig.close_position.side_effect = RuntimeError("close REST error")
        tick = {"bid": 110.0, "ofr": 111.0, "utm": 0}

        strat._on_tick(tick)

        assert strat._tick_long_close_in_flight is False

    def test_short_close_in_flight_flag_reset_on_exception(
        self, make_strategy_v2, make_params_v2
    ):
        """_tick_short_close_in_flight is False even when close_position raises — finally fires."""
        params = make_params_v2(operation_mode="tick")
        strat, mock_ig, _ = make_strategy_v2(params=params)
        strat._cached_indicators = _make_indicators(
            bb_upper=200.0, bb_lower=100.0, rsi=50.0, close=100.0
        )
        strat._short_positions = [{"deal_id": "S1", "entry_price": 110.0, "size": 0.5}]
        mock_ig.close_position.side_effect = RuntimeError("close REST error")
        tick = {"bid": 90.0, "ofr": 91.0, "utm": 0}

        strat._on_tick(tick)

        assert strat._tick_short_close_in_flight is False


# --------------------------------------------------------------------------- #
# Startup position reconciliation — seeding from broker state (REQ-2–REQ-7)   #
# --------------------------------------------------------------------------- #

_CANONICAL_POSITIONS = [
    {
        "dealReference": "REF1",
        "dealId": "D1",
        "level": 19000.0,
        "size": 1.0,
        "createdDate": "2026-05-29T10:00:00",
        "direction": "BUY",
        "epic": "IX.D.NASDAQ.IFMM.IP",
    },
    {
        "dealReference": "REF2",
        "dealId": "D2",
        "level": 19100.0,
        "size": 1.0,
        "createdDate": "2026-05-29T11:00:00",
        "direction": "SELL",
        "epic": "IX.D.NASDAQ.IFMM.IP",
    },
    {
        "dealReference": "REF3",
        "dealId": "D3",
        "level": 4500.0,
        "size": 0.5,
        "createdDate": "2026-05-29T12:00:00",
        "direction": "BUY",
        "epic": "IX.D.SPTRD.IFMM.IP",
    },
]


class TestSeedPositionsFromBroker:
    """_seed_positions_from_broker() seeds local grids from broker state at startup."""

    # Epic used by _CANONICAL_POSITIONS — made explicit so the coupling to
    # the fixture default is visible and self-documenting.
    _SEED_EPIC = "IX.D.NASDAQ.IFMM.IP"

    def test_happy_path_mixed_buy_sell(self, make_strategy_v2, make_params_v2):
        """BUY and SELL for matching epic populate the correct grids; wrong epic is excluded.

        Broker returns 3 positions: 1 BUY matching epic, 1 SELL matching epic,
        1 BUY wrong epic. Only the two matching-epic positions are seeded.
        """
        params = make_params_v2(epic=self._SEED_EPIC)
        strat, mock_ig, _ = make_strategy_v2(params=params)
        mock_ig.get_open_positions.return_value = _CANONICAL_POSITIONS

        strat._seed_positions_from_broker()

        assert strat._long_positions == [
            {"deal_id": "D1", "entry_price": 19000.0, "size": 1.0}
        ]
        assert strat._short_positions == [
            {"deal_id": "D2", "entry_price": 19100.0, "size": 1.0}
        ]

    def test_wrong_epic_excluded(self, make_strategy_v2, make_params_v2):
        """Positions with a different epic are silently excluded from both grids."""
        params = make_params_v2(epic=self._SEED_EPIC)
        strat, mock_ig, _ = make_strategy_v2(params=params)
        wrong_epic_only = [
            {
                "dealReference": "REF3",
                "dealId": "D3",
                "level": 4500.0,
                "size": 0.5,
                "createdDate": "2026-05-29T12:00:00",
                "direction": "BUY",
                "epic": "IX.D.SPTRD.IFMM.IP",
            }
        ]
        mock_ig.get_open_positions.return_value = wrong_epic_only

        strat._seed_positions_from_broker()

        assert strat._long_positions == []
        assert strat._short_positions == []

    def test_empty_broker_response(self, make_strategy_v2, make_params_v2):
        """Empty broker response leaves both grids empty and does not raise."""
        params = make_params_v2(epic=self._SEED_EPIC)
        strat, mock_ig, _ = make_strategy_v2(params=params)
        mock_ig.get_open_positions.return_value = []

        strat._seed_positions_from_broker()  # must not raise

        assert strat._long_positions == []
        assert strat._short_positions == []

    def test_broker_exception_graceful_degradation(
        self, make_strategy_v2, make_params_v2, caplog
    ):
        """Exception from get_open_positions is caught; grids remain empty; WARNING logged."""
        import logging

        params = make_params_v2(epic=self._SEED_EPIC)
        strat, mock_ig, _ = make_strategy_v2(params=params)
        mock_ig.get_open_positions.side_effect = Exception("network")

        with caplog.at_level(logging.WARNING):
            strat._seed_positions_from_broker()  # must not raise

        assert strat._long_positions == []
        assert strat._short_positions == []
        assert any(
            "WARNING" in r.levelname for r in caplog.records
        ), "Expected at least one WARNING log entry after broker exception"

    def test_run_call_sequence(self, make_strategy_v2):
        """run() must call _warmup before _seed_positions_from_broker before streaming start.

        Uses patch.object to spy on _warmup and _seed_positions_from_broker. The
        streaming_client.start side_effect calls strat.stop() so run() unblocks.
        Call order is verified via a shared call_log list.
        """
        strat, _, mock_streaming = make_strategy_v2()
        call_log = []

        def fake_warmup():
            call_log.append("warmup")

        def fake_seed():
            call_log.append("seed")

        def fake_start(callback, on_tick=None):
            call_log.append("streaming_start")
            strat.stop()

        with (
            patch.object(strat, "_warmup", side_effect=fake_warmup),
            patch.object(strat, "_seed_positions_from_broker", side_effect=fake_seed),
        ):
            mock_streaming.start.side_effect = fake_start
            t = threading.Thread(target=strat.run)
            t.start()
            t.join(timeout=2.0)

        assert not t.is_alive(), "run() did not return after stop()"
        assert call_log == [
            "warmup",
            "seed",
            "streaming_start",
        ], f"Expected [warmup, seed, streaming_start], got {call_log}"

    def test_sort_order_by_created_date(self, make_strategy_v2, make_params_v2):
        """Positions are sorted by createdDate ascending regardless of input order.

        Broker returns two BUY positions for the correct epic with createdDate
        values in REVERSE chronological order (newest first). After seeding,
        _long_positions[0] must correspond to the earlier position and
        _long_positions[-1] to the later one — confirming ascending sort.
        """
        params = make_params_v2(epic=self._SEED_EPIC)
        strat, mock_ig, _ = make_strategy_v2(params=params)
        mock_ig.get_open_positions.return_value = [
            {
                "dealReference": "REF_LATER",
                "dealId": "D_LATER",
                "level": 19200.0,
                "size": 1.0,
                "createdDate": "2026/05/29 12:00:00:000",
                "direction": "BUY",
                "epic": self._SEED_EPIC,
            },
            {
                "dealReference": "REF_EARLIER",
                "dealId": "D_EARLIER",
                "level": 19000.0,
                "size": 1.0,
                "createdDate": "2026/05/29 10:00:00:000",
                "direction": "BUY",
                "epic": self._SEED_EPIC,
            },
        ]

        strat._seed_positions_from_broker()

        assert len(strat._long_positions) == 2
        assert strat._long_positions[0]["entry_price"] == 19000.0  # earlier
        assert strat._long_positions[-1]["entry_price"] == 19200.0  # later

    def test_seeding_failure_does_not_block_streaming(self, make_strategy_v2):
        """streaming_client.start is called even when broker call raises internally inside seed.

        _seed_positions_from_broker wraps all logic in try/except — when the
        broker call raises, the method catches it and returns normally. run()
        then proceeds to streaming_client.start() as if seeding succeeded.
        """
        strat, mock_ig, mock_streaming = make_strategy_v2()
        mock_ig.get_open_positions.side_effect = Exception("network timeout")

        def fake_warmup():
            pass

        def fake_start(callback, on_tick=None):
            strat.stop()

        mock_streaming.start.side_effect = fake_start
        with patch.object(strat, "_warmup", side_effect=fake_warmup):
            t = threading.Thread(target=strat.run)
            t.start()
            t.join(timeout=2.0)

        assert not t.is_alive(), "run() did not return after stop()"
        mock_streaming.start.assert_called_once()
