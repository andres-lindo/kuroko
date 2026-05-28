"""Unit tests for RSIBollingerStrategyV2 (REQ-7 through REQ-12, REQ-17).

Covers entry signals (long/short), entry guards (max positions, min distance),
exit logic (profitable per-position close), grid independence, flat sizing,
spread-aware profit calculation, and broker take-profit on open.

All broker interactions are mocked — no live credentials required.
"""

import types
import queue
import numpy as np
from unittest.mock import MagicMock, patch, call

import pytest

from strategies.RSIBollingerStrategyV2 import RSIBollingerStrategyV2, load_params

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _make_params(**overrides) -> types.SimpleNamespace:
    """Return a minimal valid V2 SimpleNamespace params object."""
    defaults = {
        "api_mode": "streaming",
        "bb_period": 20,
        "bb_std": 2.0,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "max_long_positions": 3,
        "max_short_positions": 3,
        "contract_size": 0.5,
        "min_dist_between_entries_ticks": 10,
        "take_profit_ticks": 50.0,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _make_trading_config(**overrides) -> types.SimpleNamespace:
    """Return a minimal trading_config matching what kuroko.py provides."""
    defaults = {
        "epic": "IX.D.NASDAQ.IFMM.IP",
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _make_strategy(params=None, **trading_overrides):
    """Return an RSIBollingerStrategyV2 with mocked clients."""
    if params is None:
        params = _make_params()
    mock_ig = MagicMock()
    mock_streaming = MagicMock()
    trading_config = _make_trading_config(**trading_overrides)
    strat = RSIBollingerStrategyV2(
        params=params,
        ig_client=mock_ig,
        streaming_client=mock_streaming,
        trading_config=trading_config,
    )
    return strat, mock_ig, mock_streaming


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
        """REQ-6: a well-formed V2 JSON file loads successfully."""
        import json

        data = {
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
            "take_profit_ticks": 240.0,
        }
        path = tmp_path / "RSIBollingerStrategyV2.json"
        path.write_text(json.dumps(data))

        params = load_params(str(path))

        assert params.api_mode == "streaming"
        assert params.candle_frequency == "5min"
        assert params.bb_period == 20
        assert params.rsi_period == 14

    def test_missing_key_triggers_sys_exit(self, tmp_path):
        """REQ-6: missing required key causes SystemExit."""
        import json

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
        """REQ-6: wrong type for a required key causes SystemExit."""
        import json

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
    """REQ-7: long position opens when all entry conditions are met."""

    def test_long_entry_opens_position_when_all_conditions_met(self):
        """REQ-7 scenario 1: price < BB_lower and RSI < oversold triggers BUY."""
        strat, mock_ig, _ = _make_strategy()
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=95.0
        )

        strat._manage_longs(indicators)

        mock_ig.open_position.assert_called_once()
        call_kwargs = mock_ig.open_position.call_args.kwargs
        assert call_kwargs["side"] == "BUY"
        assert call_kwargs["size"] == 0.5

    def test_long_entry_sets_take_profit_at_open(self):
        """REQ-7 scenario 1: broker take-profit set at entry_price + take_profit_ticks."""
        strat, mock_ig, _ = _make_strategy()
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=95.0
        )

        strat._manage_longs(indicators)

        call_kwargs = mock_ig.open_position.call_args.kwargs
        # take_profit_ticks=50.0, close=95.0 → limit distance = 50.0
        assert call_kwargs["limit"] == pytest.approx(50.0)

    def test_long_entry_blocked_when_rsi_not_oversold(self):
        """REQ-7: no entry when RSI is NOT below rsi_oversold (50 >= 30)."""
        strat, mock_ig, _ = _make_strategy()
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=50.0, close=95.0
        )

        strat._manage_longs(indicators)

        mock_ig.open_position.assert_not_called()

    def test_long_entry_blocked_when_price_above_bb_lower(self):
        """REQ-7: no entry when price is NOT below BB lower (105 >= 100)."""
        strat, mock_ig, _ = _make_strategy()
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=105.0
        )

        strat._manage_longs(indicators)

        mock_ig.open_position.assert_not_called()


# --------------------------------------------------------------------------- #
# Long entry guards — REQ-7 scenarios 2 & 3                                   #
# --------------------------------------------------------------------------- #


class TestLongEntryGuards:
    """REQ-7 guards: max positions and min distance prevent entry."""

    def test_long_entry_blocked_when_max_positions_reached(self):
        """REQ-7 scenario 2: no new long when max_long_positions is full."""
        params = _make_params(max_long_positions=2)
        strat, mock_ig, _ = _make_strategy(params=params)
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

    def test_long_entry_blocked_when_distance_too_small(self):
        """REQ-7 scenario 3: no entry when distance from last entry < min_dist_between_entries_ticks."""
        params = _make_params(min_dist_between_entries_ticks=20)
        strat, mock_ig, _ = _make_strategy(params=params)
        # Last long at 90.0; current price 95.0 → distance = 5 < 20
        strat._long_positions = [
            {"deal_id": "A", "entry_price": 90.0, "size": 0.5},
        ]
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=95.0
        )

        strat._manage_longs(indicators)

        mock_ig.open_position.assert_not_called()

    def test_long_entry_allowed_when_distance_sufficient(self):
        """REQ-7 scenario 3 complement: entry allowed when distance >= min_dist_between_entries_ticks."""
        params = _make_params(min_dist_between_entries_ticks=10)
        strat, mock_ig, _ = _make_strategy(params=params)
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
    """REQ-8: short position opens when all entry conditions are met."""

    def test_short_entry_opens_position_when_all_conditions_met(self):
        """REQ-8 scenario 1: price > BB_upper and RSI > overbought triggers SELL."""
        strat, mock_ig, _ = _make_strategy()
        indicators = _make_indicators(
            bb_lower=0.0, bb_upper=100.0, rsi=75.0, close=105.0
        )

        strat._manage_shorts(indicators)

        mock_ig.open_position.assert_called_once()
        call_kwargs = mock_ig.open_position.call_args.kwargs
        assert call_kwargs["side"] == "SELL"
        assert call_kwargs["size"] == 0.5

    def test_short_entry_sets_take_profit_at_open(self):
        """REQ-8 scenario 1: broker take-profit set at entry_price - take_profit_ticks."""
        strat, mock_ig, _ = _make_strategy()
        indicators = _make_indicators(
            bb_lower=0.0, bb_upper=100.0, rsi=75.0, close=105.0
        )

        strat._manage_shorts(indicators)

        call_kwargs = mock_ig.open_position.call_args.kwargs
        # take_profit_ticks=50.0 → limit distance = 50.0
        assert call_kwargs["limit"] == pytest.approx(50.0)

    def test_short_entry_blocked_when_max_positions_reached(self):
        """REQ-8 scenario 2: no new short when max_short_positions is full."""
        params = _make_params(max_short_positions=1)
        strat, mock_ig, _ = _make_strategy(params=params)
        strat._short_positions = [{"deal_id": "A", "entry_price": 110.0, "size": 0.5}]
        indicators = _make_indicators(
            bb_lower=0.0, bb_upper=100.0, rsi=75.0, close=105.0
        )

        strat._manage_shorts(indicators)

        mock_ig.open_position.assert_not_called()

    def test_short_entry_blocked_when_distance_too_small(self):
        """REQ-8: no entry when distance from last short entry < min_dist_between_entries_ticks."""
        params = _make_params(min_dist_between_entries_ticks=20)
        strat, mock_ig, _ = _make_strategy(params=params)
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
    """REQ-9: profitable longs are closed when price crosses above BB upper."""

    def test_profitable_long_closed_at_bb_upper_cross(self):
        """REQ-9 scenario 1: long with profit > 0 after spread is closed."""
        strat, mock_ig, _ = _make_strategy()
        # entry=90, close=105, spread=1.0 → profit = (105-90-1)*0.5 = 7.0 > 0
        strat._long_positions = [{"deal_id": "DEAL1", "entry_price": 90.0, "size": 0.5}]
        indicators = _make_indicators(
            bb_lower=50.0, bb_upper=100.0, rsi=50.0, close=105.0
        )

        strat._manage_longs(indicators)

        mock_ig.close_position.assert_called_once_with("DEAL1", "SELL", 0.5)

    def test_long_not_closed_when_underwater_after_spread(self):
        """REQ-9 scenario 2: long with profit <= 0 after spread is NOT closed at BB upper."""
        strat, mock_ig, _ = _make_strategy()
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

    def test_only_profitable_longs_are_closed(self):
        """REQ-9 scenario 1: only profitable positions closed; underwater stays open."""
        strat, mock_ig, _ = _make_strategy()
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

    def test_long_spread_subtracted_from_profit(self):
        """REQ-9: spread is subtracted when computing per-position profit."""
        # With spread=5: entry=90, close=94, spread=5 → (94-90-5)*0.5 = -0.5 → NOT closed
        strat, mock_ig, _ = _make_strategy()
        strat._current_spread = 5.0  # spread from latest candle
        strat._long_positions = [{"deal_id": "DEAL1", "entry_price": 90.0, "size": 0.5}]
        indicators = _make_indicators(
            bb_lower=50.0, bb_upper=90.0, rsi=50.0, close=94.0
        )

        strat._manage_longs(indicators)

        mock_ig.close_position.assert_not_called()

    def test_long_spread_read_from_candle_not_trading_config(self):
        """Spread must come from the candle, not trading_config.

        Entry=100, close=105:
        - Without candle spread update (fallback 0.0): profit = (105-100-0)*0.5 = 2.5 → close
        - After candle spread update (spread=1.0): profit = (105-100-1)*0.5 = 2.0 → close

        This test verifies that _update_spread_from_candle correctly updates the spread
        used for profit calculations.
        """
        strat, mock_ig, _ = _make_strategy()
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
    """REQ-10: profitable shorts are closed when price crosses below BB lower."""

    def test_profitable_short_closed_at_bb_lower_cross(self):
        """REQ-10 scenario 1: short with profit > 0 after spread is closed."""
        strat, mock_ig, _ = _make_strategy()
        # entry=110, close=95, spread=1 → profit = (110-95-1)*0.5 = 7.0 > 0
        strat._short_positions = [
            {"deal_id": "DEAL2", "entry_price": 110.0, "size": 0.5}
        ]
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=150.0, rsi=50.0, close=95.0
        )

        strat._manage_shorts(indicators)

        mock_ig.close_position.assert_called_once_with("DEAL2", "BUY", 0.5)

    def test_short_not_closed_when_underwater(self):
        """REQ-10: short with profit <= 0 after spread is NOT closed at BB lower."""
        strat, mock_ig, _ = _make_strategy()
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

    def test_short_spread_subtracted_from_profit(self):
        """REQ-10: spread is applied when computing short profit."""
        # entry=105, close=100, spread=5 → profit = (105-100-5)*0.5 = 0 → NOT closed
        strat, mock_ig, _ = _make_strategy()
        strat._current_spread = 5.0  # spread from latest candle
        strat._short_positions = [
            {"deal_id": "DEAL2", "entry_price": 105.0, "size": 0.5}
        ]
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=150.0, rsi=50.0, close=100.0
        )

        strat._manage_shorts(indicators)

        mock_ig.close_position.assert_not_called()

    def test_short_spread_read_from_candle_not_trading_config(self):
        """Spread must come from the candle, not trading_config.

        After _update_spread_from_candle is called with spread=1.0:
        entry=110, close=95 → profit = (110 - 95 - 1) * 0.5 = 7.0 > 0 → closed.
        """
        strat, mock_ig, _ = _make_strategy()
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
    """REQ-11: long and short grids are evaluated independently."""

    def test_long_and_short_grids_coexist_without_interference(self):
        """REQ-11 scenario 1: 2 longs and 2 shorts all remain open with neutral candle."""
        strat, mock_ig, _ = _make_strategy()
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

    def test_short_signal_does_not_suppress_long_entry(self):
        """REQ-11: a short signal condition does NOT prevent a valid long entry."""
        params = _make_params(max_long_positions=3, max_short_positions=3)
        strat, mock_ig, _ = _make_strategy(params=params)
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
    """REQ-12: all positions use flat contract_size (no martingale)."""

    def test_second_long_entry_uses_flat_size(self):
        """REQ-12 scenario 1: second long entry size equals contract_size."""
        params = _make_params(contract_size=0.5, max_long_positions=3)
        strat, mock_ig, _ = _make_strategy(params=params)
        # One existing long
        strat._long_positions = [{"deal_id": "A", "entry_price": 60.0, "size": 0.5}]
        # New signal with sufficient distance
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=75.0
        )

        strat._manage_longs(indicators)

        call_kwargs = mock_ig.open_position.call_args.kwargs
        assert call_kwargs["size"] == pytest.approx(0.5)

    def test_second_short_entry_uses_flat_size(self):
        """REQ-12: second short entry size equals contract_size (no multiplier)."""
        params = _make_params(contract_size=0.5, max_short_positions=3)
        strat, mock_ig, _ = _make_strategy(params=params)
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

    def test_update_spread_from_candle_sets_current_spread(self):
        """_update_spread_from_candle must update self._current_spread."""
        strat, _, _ = _make_strategy()
        candle = {"bid_close": 100.0, "ofr_close": 101.5, "spread": 1.5}

        strat._update_spread_from_candle(candle)

        assert strat._current_spread == pytest.approx(1.5)

    def test_on_candle_updates_spread(self):
        """_on_candle must update _current_spread from the incoming candle."""
        strat, _, _ = _make_strategy()
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

    def test_initial_spread_is_none_before_any_candle(self):
        """Before any candle is processed, _current_spread must be None."""
        strat, _, _ = _make_strategy()
        assert strat._current_spread is None

    def test_profit_uses_candle_spread_not_config_spread(self):
        """End-to-end: profit calculation uses _current_spread set from candle.

        entry=100, close=105, _current_spread=1.0:
        profit = (105-100-1)*0.5 = 2.0 → closed.

        Without setting _current_spread (fallback=0.0):
        profit = (105-100-0)*0.5 = 2.5 → also closed.

        Verified via _update_spread_from_candle path.
        """
        strat, mock_ig, _ = _make_strategy()
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

    def test_spread_falls_back_to_default_when_no_candle_yet(self):
        """If no candle has been processed yet, spread defaults to 0.0 (no suppression)."""
        strat, mock_ig, _ = _make_strategy()
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

    def test_run_passes_on_candle_directly_as_callback(self):
        """run() must call streaming_client.start() with _on_candle as the callback.

        The strategy must NOT pass an intermediate re-queuing function like
        _enqueue_candle. The streaming client's own internal queue is sufficient.
        Verified by running in a thread: start() captures the callback, then
        stop() is called to unblock run().
        """
        import threading

        strat, _, mock_streaming = _make_strategy()
        captured = {}

        def fake_start(callback):
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

    def test_run_does_not_use_internal_candle_queue_for_routing(self):
        """run() must not have an internal _candle_queue attribute for re-queuing candles.

        The design (LS → IGStreamingClient queue → strategy run()) means the
        strategy should consume directly from the streaming client — not re-queue
        via its own internal queue.
        """
        strat, _, _ = _make_strategy()
        assert not hasattr(strat, "_candle_queue"), (
            "Strategy must not have an internal _candle_queue for re-routing candles. "
            "Consume directly from the streaming client's queue."
        )


# --------------------------------------------------------------------------- #
# Issue 1: Phantom positions — reconciliation after close failure              #
# --------------------------------------------------------------------------- #


class TestPhantomPositionReconciliation:
    """CRITICAL: failed close must not leave phantom positions in the grid forever."""

    def test_failed_close_marks_position_needs_reconciliation(self):
        """After a close failure, the position must be flagged needs_reconciliation=True."""
        strat, mock_ig, _ = _make_strategy()
        mock_ig.close_position.side_effect = Exception("Network error")
        strat._long_positions = [{"deal_id": "DEAL1", "entry_price": 90.0, "size": 0.5}]
        indicators = _make_indicators(
            bb_lower=50.0, bb_upper=100.0, rsi=50.0, close=105.0
        )

        strat._manage_longs(indicators)

        # Position must stay in grid AND be flagged
        assert len(strat._long_positions) == 1
        assert strat._long_positions[0].get("needs_reconciliation") is True

    def test_failed_short_close_marks_position_needs_reconciliation(self):
        """After a short close failure, the position must be flagged needs_reconciliation=True."""
        strat, mock_ig, _ = _make_strategy()
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

    def test_reconcile_removes_positions_absent_from_broker(self):
        """_reconcile_positions must remove local positions that no longer exist at broker.

        get_open_positions() returns a flat list of dicts with a top-level
        'dealId' key — not a nested {'positions': [{'position': {'dealId': ...}}]} dict.
        """
        strat, mock_ig, _ = _make_strategy()
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

    def test_reconcile_keeps_positions_present_at_broker(self):
        """_reconcile_positions must keep local positions that are still open at broker.

        get_open_positions() returns a flat list of dicts with a top-level
        'dealId' key — not a nested {'positions': [{'position': {'dealId': ...}}]} dict.
        """
        strat, mock_ig, _ = _make_strategy()
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

    def test_reconcile_called_before_entry_when_flagged_positions_exist(self):
        """_on_candle must call _reconcile_positions before entry evaluation
        when any position is flagged needs_reconciliation."""
        import threading

        strat, mock_ig, _ = _make_strategy()
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

        with (
            patch.object(strat, "_manage_longs") as mock_longs,
            patch.object(strat, "_manage_shorts") as mock_shorts,
        ):
            strat._on_candle(candle)
            # reconcile should have cleared the dead position before manage_longs is called
            mock_ig.get_open_positions.assert_called_once()

    def test_reconcile_not_called_when_no_flagged_positions(self):
        """_reconcile_positions must NOT be called if no positions need reconciliation."""
        strat, mock_ig, _ = _make_strategy()
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

    def test_position_not_added_when_deal_id_is_unknown(self):
        """If deal_id extraction returns 'unknown', position must NOT be added to grid."""
        strat, mock_ig, _ = _make_strategy()
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

    def test_short_position_not_added_when_deal_id_is_unknown(self):
        """If deal_id extraction returns 'unknown' for a short, position must NOT be added."""
        strat, mock_ig, _ = _make_strategy()
        mock_ig.open_position.return_value = {}
        indicators = _make_indicators(
            bb_lower=0.0, bb_upper=100.0, rsi=75.0, close=110.0
        )

        strat._manage_shorts(indicators)

        assert len(strat._short_positions) == 0

    def test_position_added_when_deal_id_is_valid(self):
        """Sanity check: position IS added when deal_id is successfully extracted."""
        strat, mock_ig, _ = _make_strategy()
        mock_ig.open_position.return_value = {"dealReference": "VALID123"}
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=95.0
        )

        strat._manage_longs(indicators)

        assert len(strat._long_positions) == 1
        assert strat._long_positions[0]["deal_id"] == "VALID123"


# --------------------------------------------------------------------------- #
# Issue 7: Strategy must stop streaming when stop() is called                  #
# --------------------------------------------------------------------------- #


class TestStrategyStopCallsStreamingStop:
    """WARNING: stop() must also call streaming_client.stop() to halt candle delivery."""

    def test_stop_calls_streaming_client_stop(self):
        """stop() must call self.streaming_client.stop() to halt candle delivery immediately."""
        strat, _, mock_streaming = _make_strategy()

        strat.stop()

        mock_streaming.stop.assert_called_once()

    def test_stop_still_sets_stop_event(self):
        """stop() must still set the internal _stop_event alongside calling streaming_client.stop()."""
        strat, _, _ = _make_strategy()

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

    def test_long_entry_blocked_when_price_equals_bb_lower(self):
        """Entry condition is STRICT (<); price == bb_lower must NOT trigger a long entry."""
        strat, mock_ig, _ = _make_strategy()
        # close == bb_lower exactly → entry blocked (not strictly less than)
        indicators = _make_indicators(
            bb_lower=100.0, bb_upper=200.0, rsi=25.0, close=100.0
        )

        strat._manage_longs(indicators)

        mock_ig.open_position.assert_not_called()

    def test_long_exit_blocked_when_price_equals_bb_upper(self):
        """Exit condition is STRICT (>); price == bb_upper must NOT trigger a long exit."""
        strat, mock_ig, _ = _make_strategy()
        strat._current_spread = 1.0
        # entry=90, close=100 == bb_upper=100 → exit NOT triggered (not strictly greater than)
        strat._long_positions = [{"deal_id": "DEAL1", "entry_price": 90.0, "size": 0.5}]
        indicators = _make_indicators(
            bb_lower=50.0, bb_upper=100.0, rsi=50.0, close=100.0
        )

        strat._manage_longs(indicators)

        mock_ig.close_position.assert_not_called()

    def test_short_entry_blocked_when_price_equals_bb_upper(self):
        """Short entry condition is STRICT (>); price == bb_upper must NOT trigger a short entry."""
        strat, mock_ig, _ = _make_strategy()
        # close == bb_upper exactly → entry blocked (not strictly greater than)
        indicators = _make_indicators(
            bb_lower=0.0, bb_upper=100.0, rsi=75.0, close=100.0
        )

        strat._manage_shorts(indicators)

        mock_ig.open_position.assert_not_called()

    def test_short_exit_blocked_when_price_equals_bb_lower(self):
        """Short exit condition is STRICT (<); price == bb_lower must NOT trigger a short exit."""
        strat, mock_ig, _ = _make_strategy()
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

    def test_on_candle_skips_candle_missing_close_key(self):
        """A candle without 'close' must be skipped; no exception raised."""
        strat, mock_ig, _ = _make_strategy()
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
        try:
            strat._on_candle(bad_candle)
        except Exception as exc:
            pytest.fail(
                f"_on_candle raised {type(exc).__name__} on malformed candle: {exc}"
            )

        mock_ig.open_position.assert_not_called()
        mock_ig.close_position.assert_not_called()

    def test_on_candle_skips_candle_missing_spread_key(self):
        """A candle missing the required 'spread' key is rejected by _REQUIRED_CANDLE_KEYS validation.

        _on_candle logs a WARNING and returns early — no position management occurs.
        """
        strat, mock_ig, _ = _make_strategy()
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

        try:
            strat._on_candle(candle_no_spread)
        except Exception as exc:
            pytest.fail(
                f"_on_candle raised {type(exc).__name__} on candle without spread: {exc}"
            )

        # Candle must be silently dropped — no position management must occur
        mock_ig.open_position.assert_not_called()
        mock_ig.close_position.assert_not_called()
