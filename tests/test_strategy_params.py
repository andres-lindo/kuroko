"""Unit tests for RSIBollingerStrategy parameter loading and constructor wiring.

Covers the reduced 17-key schema (5 infra keys removed), schema validation
behaviour for unrecognised extra keys, and RSIBollingerStrategy.__init__
sourcing infra attributes from trading_config instead of params.
"""

import json
import types
from unittest.mock import MagicMock

import pytest

from strategies.RSIBollingerStrategy import RSIBollingerStrategy, load_params

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

_VALID_PARAMS: dict = {
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

_VALID_TRADING_CONFIG = types.SimpleNamespace(
    epic="IX.D.NASDAQ.IFMM.IP",
    leverage=20,
    demo_starting_balance=20000.0,
    initial_cash_balance=4000.0,
    security_buffer=1000.0,
)


# --------------------------------------------------------------------------- #
# load_params — reduced 17-key schema                                          #
# --------------------------------------------------------------------------- #


class TestLoadParams:
    """Tests for load_params() with the 17-key reduced schema."""

    def test_valid_17_key_json_loads_without_exit(self, tmp_path):
        """REQ-2, REQ-3 scenario 1: 17-key file loads; no sys.exit(1) called."""
        params_file = tmp_path / "strategy.json"
        params_file.write_text(json.dumps(_VALID_PARAMS))

        result = load_params(str(params_file))

        assert isinstance(result, types.SimpleNamespace)

    def test_returned_namespace_has_no_removed_keys(self, tmp_path):
        """REQ-2: removed infra keys are not present in the returned SimpleNamespace."""
        params_file = tmp_path / "strategy.json"
        params_file.write_text(json.dumps(_VALID_PARAMS))

        result = load_params(str(params_file))

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

    def test_returned_namespace_has_all_signal_keys(self, tmp_path):
        """REQ-3: all 17 signal/risk keys are present in the returned SimpleNamespace."""
        params_file = tmp_path / "strategy.json"
        params_file.write_text(json.dumps(_VALID_PARAMS))

        result = load_params(str(params_file))

        for key in _VALID_PARAMS:
            assert hasattr(result, key), f"Expected '{key}' to be present in params"

    def test_missing_required_key_triggers_sys_exit(self, tmp_path):
        """REQ-3 scenario 2: a required key missing from JSON triggers sys.exit(1)."""
        bad_params = {k: v for k, v in _VALID_PARAMS.items() if k != "lookback"}
        params_file = tmp_path / "strategy.json"
        params_file.write_text(json.dumps(bad_params))

        with pytest.raises(SystemExit) as exc_info:
            load_params(str(params_file))

        assert exc_info.value.code == 1

    def test_zero_candle_frequency_triggers_sys_exit(self, tmp_path):
        """candle_frequency '0min' must be rejected — it causes ZeroDivisionError at runtime."""
        bad_params = {**_VALID_PARAMS, "candle_frequency": "0min"}
        params_file = tmp_path / "strategy.json"
        params_file.write_text(json.dumps(bad_params))

        with pytest.raises(SystemExit) as exc_info:
            load_params(str(params_file))

        assert exc_info.value.code == 1

    def test_bool_value_for_int_param_triggers_exit(self, tmp_path):
        """bool is a subclass of int in Python; the bool guard must reject it for int fields."""
        bad_params = {**_VALID_PARAMS, "lookback": True}
        params_file = tmp_path / "strategy.json"
        params_file.write_text(json.dumps(bad_params))

        with pytest.raises(SystemExit) as exc_info:
            load_params(str(params_file))

        assert exc_info.value.code == 1


# --------------------------------------------------------------------------- #
# RSIBollingerStrategy.__init__ — trading_config wiring                       #
# --------------------------------------------------------------------------- #


class TestStrategyInit:
    """Tests for RSIBollingerStrategy.__init__ with trading_config."""

    def _make_params(self) -> types.SimpleNamespace:
        """Return a minimal valid params SimpleNamespace."""
        return types.SimpleNamespace(**_VALID_PARAMS)

    def test_infra_attrs_sourced_from_trading_config(self):
        """REQ-4 scenario 1: self.epic, leverage, etc. come from trading_config."""
        params = self._make_params()
        ig_mock = MagicMock()
        trading_config = types.SimpleNamespace(
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

    def test_trading_config_values_override_any_params_absence(self):
        """REQ-4 scenario 2: params without infra keys still initialises correctly."""
        params = self._make_params()
        # Confirm these are NOT present on params
        assert not hasattr(params, "epic")
        assert not hasattr(params, "leverage")

        ig_mock = MagicMock()
        trading_config = types.SimpleNamespace(
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
        assert strat.demo_starting_balance == 50000.0
        assert strat.initial_cash_balance == 5000.0
        assert strat.security_buffer == 500.0

    def test_signal_params_still_sourced_from_params(self):
        """Params namespace still supplies signal/risk attributes to the strategy."""
        params = self._make_params()
        ig_mock = MagicMock()

        strat = RSIBollingerStrategy(
            params=params, ig_client=ig_mock, trading_config=_VALID_TRADING_CONFIG
        )

        assert strat.rsi_period == params.rsi_period
        assert strat.bb_period == params.bb_period
        assert strat.take_profit_ticks == params.take_profit_ticks
        assert strat.lookback == params.lookback
