"""Tests for RSIBollingerStrategy — run() sub-methods, signal logic, and risk controls.

Covers manage_positions entry conditions (RSI + BB signals), martingale sizing,
ATR-based stop-loss, drawdown freeze (75% threshold), and basket take-profit.
Also covers load_params() schema validation and __init__ trading_config wiring.
IGClient and talib functions are mocked. time.sleep is always patched.

Design decision: we test manage_positions() and close_all_positions() directly
rather than the run() loop (which is an infinite while True). This matches the
approach used in test_strategy_v2.py for V2.
"""

import json
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from strategies.RSIBollingerStrategy import RSIBollingerStrategy
from strategies.RSIBollingerStrategy import load_params as load_params_v1

# --------------------------------------------------------------------------- #
# Project root for config file discovery                                       #
# --------------------------------------------------------------------------- #

_PROJECT_ROOT = Path(__file__).parent.parent
_V1_JSON = _PROJECT_ROOT / "strategies" / "RSIBollingerStrategy.json"


# --------------------------------------------------------------------------- #
# Module-level helpers (file-specific factory utilities)                       #
# --------------------------------------------------------------------------- #


def _make_candle_df(
    close=19000.0,
    rsi=20.0,
    bb_lower=18900.0,
    bb_upper=19100.0,
    bb_middle=19000.0,
    atr=50.0,
    ema=18950.0,
    n=5,
):
    """Build a minimal candle DataFrame with pre-computed indicators.

    The last row (most recent) is the one manage_positions() reads for signals.

    Args:
        close: Close price for all rows.
        rsi: RSI value for all rows.
        bb_lower: BB lower band for all rows.
        bb_upper: BB upper band for all rows.
        bb_middle: BB middle band for all rows.
        atr: ATR value for all rows.
        ema: EMA value for all rows.
        n: Number of rows.

    Returns:
        DataFrame with OHLC + indicator columns, DatetimeIndex.
    """
    data = {
        "Open": [close] * n,
        "High": [close + 10] * n,
        "Low": [close - 10] * n,
        "Close": [close] * n,
        "rsi": [rsi] * n,
        "bb_lower": [bb_lower] * n,
        "bb_upper": [bb_upper] * n,
        "bb_middle": [bb_middle] * n,
        "atr": [atr] * n,
        "ema": [ema] * n,
    }
    index = pd.date_range("2026-01-01", periods=n, freq="15min")
    return pd.DataFrame(data, index=index)


def _open_position_dict(
    deal_id="DEAL1",
    deal_ref="REF1",
    level=19000.0,
    size=0.13,
    direction="BUY",
    created="2026-01-01T10:00:00",
):
    """Build a position dict matching the shape from IGClient.get_open_positions.

    Args:
        deal_id: Deal identifier.
        deal_ref: Deal reference string.
        level: Entry price level.
        size: Position size in contracts.
        direction: 'BUY' or 'SELL'.
        created: ISO timestamp string.

    Returns:
        Dict with the six expected keys.
    """
    return {
        "dealId": deal_id,
        "dealReference": deal_ref,
        "level": level,
        "size": size,
        "direction": direction,
        "createdDate": created,
    }


def _account_info(
    balance=20000.0,
    deposit=5000.0,
    profit_loss=0.0,
    available=15000.0,
):
    """Build an account summary dict.

    Args:
        balance: Broker-reported account balance.
        deposit: Used margin / deposit.
        profit_loss: Open P&L.
        available: Available free margin.

    Returns:
        Dict with accountId, balance, deposit, profitLoss, available keys.
    """
    return {
        "accountId": "ACC123",
        "balance": balance,
        "deposit": deposit,
        "profitLoss": profit_loss,
        "available": available,
    }


# --------------------------------------------------------------------------- #
# get_candles                                                                  #
# --------------------------------------------------------------------------- #


class TestGetCandles:
    """Tests for RSIBollingerStrategy.get_candles indicator computation."""

    def test_returns_empty_dataframe_when_ig_returns_none(self, make_strategy_v1):
        """get_candles returns the cached (empty) DataFrame when ig.get_candles() is None."""
        strategy, mock_ig = make_strategy_v1()
        mock_ig.get_candles.return_value = None

        result = strategy.get_candles()

        assert result.empty

    def test_returns_cached_dataframe_on_exception(self, make_strategy_v1):
        """get_candles returns cached DataFrame and does not raise on API exception."""
        strategy, mock_ig = make_strategy_v1()
        strategy.candles = pd.DataFrame({"Close": [100.0]})
        mock_ig.get_candles.side_effect = Exception("network error")

        result = strategy.get_candles()

        assert not result.empty
        assert result["Close"].iloc[0] == 100.0

    def test_calls_ig_get_candles_with_correct_args(
        self, make_strategy_v1, make_params_v1
    ):
        """get_candles forwards the correct epic, frequency, and lookback to ig.get_candles."""
        params = make_params_v1(candle_frequency="15min", lookback=100)
        strategy, mock_ig = make_strategy_v1(params=params)
        mock_ig.get_candles.return_value = None

        strategy.get_candles()

        mock_ig.get_candles.assert_called_once_with("IX.D.NASDAQ.IFMM.IP", "15min", 100)


# --------------------------------------------------------------------------- #
# manage_positions — no candle data                                            #
# --------------------------------------------------------------------------- #


class TestManagePositionsNoCandleData:
    """Tests for manage_positions when candle data is absent."""

    def test_skips_cycle_when_candles_empty(self, make_strategy_v1):
        """manage_positions returns early and does not call ig when candles is empty."""
        strategy, mock_ig = make_strategy_v1()
        assert strategy.candles.empty

        strategy.manage_positions()

        mock_ig.open_position.assert_not_called()
        mock_ig.close_position.assert_not_called()


# --------------------------------------------------------------------------- #
# manage_positions — take-profit exits                                         #
# --------------------------------------------------------------------------- #


class TestTakeProfitExit:
    """Tests for the basket take-profit exit logic in manage_positions."""

    def test_long_basket_closed_when_take_profit_reached(
        self, make_strategy_v1, make_params_v1
    ):
        """All BUY positions are closed when current_price >= avg_entry + take_profit_ticks."""
        params = make_params_v1(take_profit_ticks=100.0)
        strategy, mock_ig = make_strategy_v1(params=params)

        avg_entry = 19000.0
        current_price = avg_entry + 100.0 + 1.0  # 19101.0 — above TP
        strategy.candles = _make_candle_df(close=current_price)

        position = _open_position_dict(level=avg_entry, size=0.13, direction="BUY")
        mock_ig.get_open_positions.return_value = [position]
        mock_ig.get_account_summary.return_value = _account_info()

        with patch("strategies.RSIBollingerStrategy.time.sleep"):
            strategy.manage_positions()

        mock_ig.close_position.assert_called_once_with(
            position["dealId"], "SELL", position["size"]
        )

    def test_short_basket_closed_when_take_profit_reached(
        self, make_strategy_v1, make_params_v1
    ):
        """All SELL positions are closed when current_price <= avg_entry - take_profit_ticks."""
        params = make_params_v1(take_profit_ticks=100.0)
        strategy, mock_ig = make_strategy_v1(params=params)

        avg_entry = 19000.0
        current_price = avg_entry - 100.0 - 1.0  # 18899.0 — below TP
        strategy.candles = _make_candle_df(
            close=current_price, rsi=80.0, bb_upper=18800.0
        )

        position = _open_position_dict(level=avg_entry, size=0.13, direction="SELL")
        mock_ig.get_open_positions.return_value = [position]
        mock_ig.get_account_summary.return_value = _account_info()

        with patch("strategies.RSIBollingerStrategy.time.sleep"):
            strategy.manage_positions()

        mock_ig.close_position.assert_called_once_with(
            position["dealId"], "BUY", position["size"]
        )

    def test_long_position_not_closed_below_take_profit(
        self, make_strategy_v1, make_params_v1
    ):
        """BUY positions are not closed when price is below avg_entry + take_profit_ticks."""
        params = make_params_v1(take_profit_ticks=240.0)
        strategy, mock_ig = make_strategy_v1(params=params)

        avg_entry = 19000.0
        current_price = avg_entry + 50.0  # well below TP
        strategy.candles = _make_candle_df(
            close=current_price, rsi=30.0, bb_lower=19100.0
        )

        position = _open_position_dict(level=avg_entry, size=0.13, direction="BUY")
        mock_ig.get_open_positions.return_value = [position]
        mock_ig.get_account_summary.return_value = _account_info()

        with patch("strategies.RSIBollingerStrategy.time.sleep"):
            strategy.manage_positions()

        mock_ig.close_position.assert_not_called()


# --------------------------------------------------------------------------- #
# manage_positions — drawdown freeze                                           #
# --------------------------------------------------------------------------- #


class TestDrawdownFreeze:
    """Tests for the max drawdown freeze logic in manage_positions."""

    def test_open_position_blocked_when_drawdown_exceeded(
        self, make_strategy_v1, make_params_v1, make_trading_config
    ):
        """No new entry is made when equity falls below the drawdown floor."""
        params = make_params_v1(max_drawdown_pct=75.0)
        trading_config = make_trading_config(initial_cash_balance=4000.0)
        strategy, mock_ig = make_strategy_v1(
            params=params, trading_config=trading_config
        )

        strategy.candles = _make_candle_df(close=18800.0, rsi=20.0, bb_lower=18900.0)
        mock_ig.get_open_positions.return_value = []
        mock_ig.get_account_summary.return_value = _account_info(
            balance=20000.0,
            profit_loss=-3100.0,  # heavy unrealised loss → equity below floor
            available=5000.0,
        )

        with patch("strategies.RSIBollingerStrategy.time.sleep"):
            strategy.manage_positions()

        mock_ig.open_position.assert_not_called()

    def test_drawdown_flag_set_when_floor_breached(
        self, make_strategy_v1, make_params_v1, make_trading_config
    ):
        """max_drawdown_reached is set to True when equity < floor."""
        params = make_params_v1(max_drawdown_pct=75.0)
        trading_config = make_trading_config(initial_cash_balance=4000.0)
        strategy, mock_ig = make_strategy_v1(
            params=params, trading_config=trading_config
        )

        strategy.candles = _make_candle_df(close=18800.0, rsi=20.0, bb_lower=18900.0)
        mock_ig.get_open_positions.return_value = []
        mock_ig.get_account_summary.return_value = _account_info(
            balance=20000.0, profit_loss=-3100.0
        )

        with patch("strategies.RSIBollingerStrategy.time.sleep"):
            strategy.manage_positions()

        assert strategy.max_drawdown_reached is True

    def test_drawdown_flag_cleared_when_equity_recovers(
        self, make_strategy_v1, make_params_v1, make_trading_config
    ):
        """max_drawdown_reached is reset to False when equity recovers above the floor."""
        params = make_params_v1(max_drawdown_pct=75.0)
        trading_config = make_trading_config(initial_cash_balance=4000.0)
        strategy, mock_ig = make_strategy_v1(
            params=params, trading_config=trading_config
        )
        strategy.max_drawdown_reached = True

        strategy.candles = _make_candle_df(close=19200.0, rsi=50.0, bb_lower=18800.0)
        mock_ig.get_open_positions.return_value = []
        mock_ig.get_account_summary.return_value = _account_info(
            balance=20000.0, profit_loss=0.0
        )

        with patch("strategies.RSIBollingerStrategy.time.sleep"):
            strategy.manage_positions()

        assert strategy.max_drawdown_reached is False


# --------------------------------------------------------------------------- #
# manage_positions — entry signals (LONG)                                      #
# --------------------------------------------------------------------------- #


class TestBuySignal:
    """Tests for buy-signal entry in manage_positions (price < BB lower + RSI < oversold)."""

    def _healthy_account(self):
        """Return an account dict with enough equity and margin for an entry."""
        return _account_info(balance=20000.0, profit_loss=0.0, available=5000.0)

    def test_buy_signal_opens_position_when_rsi_oversold_and_below_bb_lower(
        self, make_strategy_v1, make_params_v1
    ):
        """open_position called with side='BUY' when price < BB lower and RSI < oversold."""
        params = make_params_v1(
            rsi_oversold=25, take_profit_ticks=240.0, atr_sl_multiplier=11.0
        )
        strategy, mock_ig = make_strategy_v1(params=params)

        strategy.candles = _make_candle_df(
            close=18800.0, rsi=20.0, bb_lower=18900.0, bb_upper=19200.0, atr=50.0
        )
        mock_ig.get_open_positions.side_effect = [[], []]
        mock_ig.get_account_summary.return_value = self._healthy_account()

        with patch("strategies.RSIBollingerStrategy.time.sleep"):
            strategy.manage_positions()

        mock_ig.open_position.assert_called_once()
        assert mock_ig.open_position.call_args.kwargs["side"] == "BUY"

    def test_no_buy_when_price_above_bb_lower(self, make_strategy_v1, make_params_v1):
        """open_position NOT called when current price is above BB lower band."""
        params = make_params_v1(rsi_oversold=25)
        strategy, mock_ig = make_strategy_v1(params=params)

        strategy.candles = _make_candle_df(close=19100.0, rsi=20.0, bb_lower=18900.0)
        mock_ig.get_open_positions.return_value = []
        mock_ig.get_account_summary.return_value = self._healthy_account()

        with patch("strategies.RSIBollingerStrategy.time.sleep"):
            strategy.manage_positions()

        mock_ig.open_position.assert_not_called()

    def test_no_buy_when_rsi_above_oversold_threshold(
        self, make_strategy_v1, make_params_v1
    ):
        """open_position NOT called when RSI is above the oversold threshold."""
        params = make_params_v1(rsi_oversold=25)
        strategy, mock_ig = make_strategy_v1(params=params)

        strategy.candles = _make_candle_df(close=18800.0, rsi=30.0, bb_lower=18900.0)
        mock_ig.get_open_positions.return_value = []
        mock_ig.get_account_summary.return_value = self._healthy_account()

        with patch("strategies.RSIBollingerStrategy.time.sleep"):
            strategy.manage_positions()

        mock_ig.open_position.assert_not_called()

    def test_buy_blocked_when_max_positions_reached(
        self, make_strategy_v1, make_params_v1
    ):
        """open_position NOT called when n_trades == max_positions."""
        params = make_params_v1(max_positions=2, rsi_oversold=25)
        strategy, mock_ig = make_strategy_v1(params=params)

        strategy.candles = _make_candle_df(close=18800.0, rsi=20.0, bb_lower=18900.0)
        positions = [
            _open_position_dict("D1", level=18900.0, direction="BUY"),
            _open_position_dict("D2", level=18800.0, direction="BUY"),
        ]
        mock_ig.get_open_positions.return_value = positions
        mock_ig.get_account_summary.return_value = self._healthy_account()

        with patch("strategies.RSIBollingerStrategy.time.sleep"):
            strategy.manage_positions()

        mock_ig.open_position.assert_not_called()


# --------------------------------------------------------------------------- #
# manage_positions — martingale sizing                                         #
# --------------------------------------------------------------------------- #


class TestMartingaleSizing:
    """Tests for the martingale position sizing in manage_positions."""

    def _healthy_account(self):
        return _account_info(balance=20000.0, profit_loss=0.0, available=10000.0)

    def test_first_entry_uses_base_position_size(
        self, make_strategy_v1, make_params_v1
    ):
        """When no positions are open, entry size equals params.position_size."""
        params = make_params_v1(
            position_size=0.13,
            martingale_multiplier=1.5,
            rsi_oversold=25,
            take_profit_ticks=240.0,
            atr_sl_multiplier=11.0,
        )
        strategy, mock_ig = make_strategy_v1(params=params)

        strategy.candles = _make_candle_df(
            close=18800.0, rsi=20.0, bb_lower=18900.0, atr=50.0
        )
        mock_ig.get_open_positions.side_effect = [[], []]
        mock_ig.get_account_summary.return_value = self._healthy_account()

        with patch("strategies.RSIBollingerStrategy.time.sleep"):
            strategy.manage_positions()

        assert mock_ig.open_position.call_args.kwargs["size"] == 0.13

    def test_second_entry_applies_martingale_multiplier(
        self, make_strategy_v1, make_params_v1
    ):
        """With one existing position, entry size = position_size * multiplier^1."""
        params = make_params_v1(
            position_size=0.13,
            martingale_multiplier=1.5,
            rsi_oversold=25,
            min_dist_between_entries_ticks=0.0,
            take_profit_ticks=240.0,
            atr_sl_multiplier=11.0,
        )
        strategy, mock_ig = make_strategy_v1(params=params)

        strategy.candles = _make_candle_df(
            close=18800.0, rsi=20.0, bb_lower=18900.0, atr=50.0
        )
        existing = _open_position_dict(level=18820.0, direction="BUY", size=0.13)
        mock_ig.get_open_positions.side_effect = [[existing], [existing]]
        mock_ig.get_account_summary.return_value = self._healthy_account()

        with patch("strategies.RSIBollingerStrategy.time.sleep"):
            strategy.manage_positions()

        expected_size = round(0.13 * (1.5**1), 2)  # 0.2
        assert mock_ig.open_position.call_args.kwargs["size"] == pytest.approx(
            expected_size, abs=0.01
        )

    def test_third_entry_applies_multiplier_squared(
        self, make_strategy_v1, make_params_v1
    ):
        """With two existing positions, entry size = position_size * multiplier^2."""
        params = make_params_v1(
            position_size=0.13,
            martingale_multiplier=1.5,
            rsi_oversold=25,
            min_dist_between_entries_ticks=0.0,
            take_profit_ticks=240.0,
            atr_sl_multiplier=11.0,
        )
        strategy, mock_ig = make_strategy_v1(params=params)

        strategy.candles = _make_candle_df(
            close=18800.0, rsi=20.0, bb_lower=18900.0, atr=50.0
        )
        positions = [
            _open_position_dict("D1", level=18850.0, direction="BUY", size=0.13),
            _open_position_dict("D2", level=18820.0, direction="BUY", size=0.2),
        ]
        mock_ig.get_open_positions.side_effect = [positions, positions]
        mock_ig.get_account_summary.return_value = self._healthy_account()

        with patch("strategies.RSIBollingerStrategy.time.sleep"):
            strategy.manage_positions()

        expected_size = round(0.13 * (1.5**2), 2)  # ~0.29
        assert mock_ig.open_position.call_args.kwargs["size"] == pytest.approx(
            expected_size, abs=0.01
        )


# --------------------------------------------------------------------------- #
# close_all_positions                                                          #
# --------------------------------------------------------------------------- #


class TestCloseAllPositions:
    """Tests for RSIBollingerStrategy.close_all_positions."""

    def test_closes_all_positions_in_basket(self, make_strategy_v1):
        """close_all_positions calls ig.close_position for every open position."""
        strategy, mock_ig = make_strategy_v1()
        positions = [
            _open_position_dict("D1", direction="BUY", size=0.13),
            _open_position_dict("D2", direction="BUY", size=0.20),
        ]

        with patch("strategies.RSIBollingerStrategy.time.sleep"):
            strategy.close_all_positions(positions, reason="BasketTP")

        assert mock_ig.close_position.call_count == 2

    def test_close_direction_is_opposite_of_open_direction(self, make_strategy_v1):
        """BUY positions are closed with SELL and SELL positions with BUY."""
        strategy, mock_ig = make_strategy_v1()
        buy_position = _open_position_dict("D1", direction="BUY", size=0.13)
        sell_position = _open_position_dict("D2", direction="SELL", size=0.13)

        with patch("strategies.RSIBollingerStrategy.time.sleep"):
            strategy.close_all_positions([buy_position], reason="Test")
            strategy.close_all_positions([sell_position], reason="Test")

        calls = mock_ig.close_position.call_args_list
        assert calls[0].args[1] == "SELL"
        assert calls[1].args[1] == "BUY"

    def test_close_retries_three_times_on_failure(self, make_strategy_v1):
        """close_all_positions retries up to 3 times when close_position raises."""
        strategy, mock_ig = make_strategy_v1()
        mock_ig.close_position.side_effect = Exception("close failed")
        position = _open_position_dict("D1", direction="BUY", size=0.13)

        with patch("strategies.RSIBollingerStrategy.time.sleep"):
            strategy.close_all_positions([position], reason="Test")

        assert mock_ig.close_position.call_count == 3

    def test_close_succeeds_on_second_attempt(self, make_strategy_v1):
        """close_all_positions succeeds when the first attempt fails but second succeeds."""
        strategy, mock_ig = make_strategy_v1()
        mock_ig.close_position.side_effect = [
            Exception("first fail"),
            {"status": "CLOSED"},
        ]
        position = _open_position_dict("D1", direction="BUY", size=0.13)

        with patch("strategies.RSIBollingerStrategy.time.sleep"):
            strategy.close_all_positions([position], reason="Test")

        assert mock_ig.close_position.call_count == 2


# =========================================================================== #
# V1 strategy param loading and constructor wiring                             #
# =========================================================================== #

_VALID_V1_PARAMS: dict = {
    "candle_frequency": "15min",
    "lookback": 300,
    "max_positions": 5,
    "position_size": 0.13,
    "min_dist_between_entries_ticks": 100.0,
    "martingale_multiplier": 1.5,
    "take_profit_ticks": 240.0,
    "max_drawdown_pct": 75.75,
    "bb_period": 20,
    "bb_dev": 1.9,
    "rsi_period": 11,
    "rsi_overbought": 76,
    "rsi_oversold": 25,
    "use_trend_filter": False,
    "atr_period": 12,
    "atr_sl_multiplier": 11.0,
    "ema_period": 200,
}


class TestV1LoadParams:
    """Tests for load_params() with the 17-key reduced schema."""

    def test_valid_17_key_json_loads_without_exit(self, tmp_path):
        """17-key file loads; no sys.exit(1) called."""
        params_file = tmp_path / "strategy.json"
        params_file.write_text(json.dumps(_VALID_V1_PARAMS))

        result = load_params_v1(str(params_file))

        assert isinstance(result, types.SimpleNamespace)

    def test_returned_namespace_excludes_removed_infra_keys(self, tmp_path):
        """removed infra keys are not present in the returned SimpleNamespace."""
        params_file = tmp_path / "strategy.json"
        params_file.write_text(json.dumps(_VALID_V1_PARAMS))

        result = load_params_v1(str(params_file))

        removed_keys = (
            "epic",
            "leverage",
            "demo_starting_balance",
            "initial_cash_balance",
            "security_buffer",
        )
        for key in removed_keys:
            assert not hasattr(
                result, key
            ), f"Expected '{key}' to be absent from params"

    def test_returned_namespace_includes_all_signal_keys(self, tmp_path):
        """all 17 signal/risk keys are present in the returned SimpleNamespace."""
        params_file = tmp_path / "strategy.json"
        params_file.write_text(json.dumps(_VALID_V1_PARAMS))

        result = load_params_v1(str(params_file))

        for key in _VALID_V1_PARAMS:
            assert hasattr(result, key), f"Expected '{key}' to be present in params"

    def test_missing_required_key_triggers_sys_exit(self, tmp_path):
        """A required key missing from JSON triggers sys.exit(1)."""
        bad_params = {k: v for k, v in _VALID_V1_PARAMS.items() if k != "lookback"}
        params_file = tmp_path / "strategy.json"
        params_file.write_text(json.dumps(bad_params))

        with pytest.raises(SystemExit) as exc_info:
            load_params_v1(str(params_file))

        assert exc_info.value.code == 1

    def test_zero_candle_frequency_triggers_sys_exit(self, tmp_path):
        """candle_frequency '0min' must be rejected — it causes ZeroDivisionError at runtime."""
        bad_params = {**_VALID_V1_PARAMS, "candle_frequency": "0min"}
        params_file = tmp_path / "strategy.json"
        params_file.write_text(json.dumps(bad_params))

        with pytest.raises(SystemExit) as exc_info:
            load_params_v1(str(params_file))

        assert exc_info.value.code == 1

    def test_bool_value_for_int_param_triggers_exit(self, tmp_path):
        """bool is a subclass of int in Python; the bool guard must reject it for int fields."""
        bad_params = {**_VALID_V1_PARAMS, "lookback": True}
        params_file = tmp_path / "strategy.json"
        params_file.write_text(json.dumps(bad_params))

        with pytest.raises(SystemExit) as exc_info:
            load_params_v1(str(params_file))

        assert exc_info.value.code == 1


class TestV1StrategyInit:
    """Tests for RSIBollingerStrategy.__init__ with trading_config wiring."""

    def test_infra_attrs_sourced_from_trading_config(
        self, make_params_v1, make_trading_config
    ):
        """Self.epic, leverage, etc. come from trading_config."""
        params = make_params_v1()
        ig_mock = MagicMock()
        trading_config = make_trading_config(
            epic="IX.D.NASDAQ.IFMM.IP",
            leverage=20,
            demo_starting_balance=20000.0,
            initial_cash_balance=4000.0,
            security_buffer=1000.0,
        )

        strat = RSIBollingerStrategy(
            params=params, ig_client=ig_mock, trading_config=trading_config
        )

        assert strat.epic == "IX.D.NASDAQ.IFMM.IP"
        assert strat.leverage == 20
        assert strat.demo_starting_balance == 20000.0
        assert strat.initial_cash_balance == 4000.0
        assert strat.security_buffer == 1000.0

    def test_trading_config_values_initialise_correctly_without_infra_on_params(
        self, make_params_v1, make_trading_config
    ):
        """Params without infra keys still initialises correctly."""
        params = make_params_v1()
        assert not hasattr(params, "epic")
        assert not hasattr(params, "leverage")

        ig_mock = MagicMock()
        trading_config = make_trading_config(
            epic="IX.D.SP500.IFM.IP",
            leverage=10,
            demo_starting_balance=50000.0,
            initial_cash_balance=5000.0,
            security_buffer=500.0,
        )

        strat = RSIBollingerStrategy(
            params=params, ig_client=ig_mock, trading_config=trading_config
        )

        assert strat.epic == "IX.D.SP500.IFM.IP"
        assert strat.leverage == 10

    def test_signal_params_sourced_from_params_not_trading_config(
        self, make_params_v1, make_trading_config
    ):
        """Params namespace still supplies signal/risk attributes to the strategy."""
        params = make_params_v1()
        ig_mock = MagicMock()
        trading_config = make_trading_config()

        strat = RSIBollingerStrategy(
            params=params, ig_client=ig_mock, trading_config=trading_config
        )

        assert strat.rsi_period == params.rsi_period
        assert strat.bb_period == params.bb_period
        assert strat.take_profit_ticks == params.take_profit_ticks
        assert strat.lookback == params.lookback


class TestV1JsonConfig:
    """strategies/RSIBollingerStrategy.json must declare api_mode='rest'."""

    def test_v1_json_file_exists(self):
        """The V1 strategy JSON file must be present at the expected path."""
        assert _V1_JSON.exists(), f"File not found: {_V1_JSON}"

    def test_v1_json_has_api_mode_key(self):
        """api_mode key must be present in V1 JSON."""
        data = json.loads(_V1_JSON.read_text(encoding="utf-8"))
        assert "api_mode" in data, "api_mode key missing from RSIBollingerStrategy.json"

    def test_v1_json_api_mode_value_is_rest(self):
        """Api_mode value must be 'rest'."""
        data = json.loads(_V1_JSON.read_text(encoding="utf-8"))
        assert (
            data["api_mode"] == "rest"
        ), f"Expected api_mode='rest', got {data.get('api_mode')!r}"
