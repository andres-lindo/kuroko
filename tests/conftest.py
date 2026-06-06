"""Shared pytest fixtures for the kuroko test suite.

Provides factory fixtures for V1/V2 params, trading config, and mock clients —
eliminating duplicated inline helpers across test files.
"""

import os
import types
from unittest.mock import MagicMock, patch

import pytest

# --------------------------------------------------------------------------- #
# V1 param factory (RSIBollingerStrategy — 17-key schema)                     #
# --------------------------------------------------------------------------- #


@pytest.fixture
def make_params_v1():
    """Factory fixture: returns callable(**overrides) → SimpleNamespace (V1 17-key schema).

    Returns:
        A factory function that accepts keyword overrides and returns a
        fully-populated SimpleNamespace with all required V1 fields.
    """

    def _factory(**overrides) -> types.SimpleNamespace:
        """Build a V1 params namespace with optional field overrides.

        Args:
            **overrides: Key/value pairs to override the defaults.

        Returns:
            SimpleNamespace populated with all 17 V1 signal/risk keys.
        """
        defaults = {
            "api_mode": "rest",
            "epic": "IX.D.SPTRD.IFMM.IP",
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
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    return _factory


# --------------------------------------------------------------------------- #
# V2 param factory (RSIBollingerStrategyV2 — 12-key schema)                   #
# --------------------------------------------------------------------------- #


@pytest.fixture
def make_params_v2():
    """Factory fixture: returns callable(**overrides) → SimpleNamespace (V2 13-key schema).

    Returns:
        A factory function that accepts keyword overrides and returns a
        fully-populated SimpleNamespace with all required V2 fields.
    """

    def _factory(**overrides) -> types.SimpleNamespace:
        """Build a V2 params namespace with optional field overrides.

        Args:
            **overrides: Key/value pairs to override the defaults.

        Returns:
            SimpleNamespace populated with all 13 V2 signal/risk keys.
        """
        defaults = {
            "epic": "IX.D.SPTRD.IFMM.IP",
            "api_mode": "streaming",
            "candle_frequency": "5min",
            "bb_period": 20,
            "bb_std": 1.5,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "max_long_positions": 3,
            "max_short_positions": 3,
            "contract_size": 1.0,
            "min_dist_between_entries_ticks": 10,
            "take_profit_ticks": 50.0,
            "stop_loss_ticks": 100.0,
            "operation_mode": "candle",
            "close_on_bb_cross": True,
        }
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    return _factory


# --------------------------------------------------------------------------- #
# Trading config factory                                                       #
# --------------------------------------------------------------------------- #


@pytest.fixture
def make_trading_config():
    """Factory fixture: returns callable(**overrides) → SimpleNamespace (trading config).

    Returns:
        A factory function that accepts keyword overrides and returns a
        SimpleNamespace matching what kuroko.py passes as trading_config.
    """

    def _factory(**overrides) -> types.SimpleNamespace:
        """Build a trading_config namespace with optional field overrides.

        Args:
            **overrides: Key/value pairs to override the defaults.

        Returns:
            SimpleNamespace with epic, leverage, and balance fields.
        """
        defaults = {
            "epic": "IX.D.SPTRD.IFMM.IP",
            "spread": 1.0,
            "leverage": 20,
            "demo_starting_balance": 20000.0,
            "initial_cash_balance": 4000.0,
            "security_buffer": 1000.0,
        }
        defaults.update(overrides)
        return types.SimpleNamespace(**defaults)

    return _factory


# --------------------------------------------------------------------------- #
# Mock IGClient factory                                                        #
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_ig_client():
    """Fixture: returns (IGClient_instance, mock_svc) with env patched and IGService mocked.

    Patches ig_client.IGService and os.environ with minimal credentials so
    IGClient can be instantiated without live credentials.

    Returns:
        Tuple of (ig_client_instance, mock_svc) where mock_svc is the
        MagicMock standing in for the underlying IGService.
    """
    from ig_client import IGClient

    mock_svc = MagicMock()
    mock_svc.create_session.return_value = {"accountType": "DEMO", "accountId": "TEST"}
    mock_svc.fetch_accounts.return_value = {
        "accounts": [{"preferred": True, "accountId": "TEST"}]
    }

    env_vars = {
        "ig_username": "test_user",
        "ig_password": "test_pass",
        "ig_api_key": "test_key",
        "ig_acc_number": "TEST123",
        "ig_acc_type": "DEMO",
    }

    with patch("ig_client.IGService", return_value=mock_svc):
        with patch.dict("os.environ", env_vars):
            client = IGClient()

    return client, mock_svc


# --------------------------------------------------------------------------- #
# V1 strategy factory                                                          #
# --------------------------------------------------------------------------- #


@pytest.fixture
def make_strategy_v1(make_params_v1, make_trading_config):
    """Factory fixture: returns callable(params=None, trading_config=None) → (strat, mock_ig).

    Returns:
        A factory function that builds an RSIBollingerStrategy with a mocked
        IGClient. Accepts optional params and trading_config namespaces.
    """

    def _factory(params=None, trading_config=None):
        """Build a V1 strategy instance with a mocked IGClient.

        Args:
            params: Optional SimpleNamespace of V1 params. Defaults to make_params_v1().
            trading_config: Optional trading_config namespace. Defaults to make_trading_config().

        Returns:
            Tuple of (strategy_instance, mock_ig_client).
        """
        from strategies.RSIBollingerStrategy import RSIBollingerStrategy

        if params is None:
            params = make_params_v1()
        if trading_config is None:
            trading_config = make_trading_config()

        mock_ig = MagicMock()

        with patch.dict(os.environ, {"ig_acc_type": "DEMO"}):
            strategy = RSIBollingerStrategy(
                params=params,
                ig_client=mock_ig,
                trading_config=trading_config,
            )

        return strategy, mock_ig

    return _factory


# --------------------------------------------------------------------------- #
# V2 strategy factory                                                          #
# --------------------------------------------------------------------------- #


@pytest.fixture
def make_strategy_v2(make_params_v2, make_trading_config):
    """Factory fixture: returns callable(params=None, **trading_overrides) → (strat, mock_ig, mock_streaming).

    Returns:
        A factory function that builds an RSIBollingerStrategyV2 with mocked
        clients. Accepts an optional params namespace and trading_config overrides.
    """

    def _factory(params=None, **trading_overrides):
        """Build a strategy instance with mocked IG and streaming clients.

        Args:
            params: Optional SimpleNamespace of V2 params. Defaults to make_params_v2().
            **trading_overrides: Key/value pairs to override trading_config defaults.

        Returns:
            Tuple of (strategy_instance, mock_ig_client, mock_streaming_client).
        """
        from strategies.RSIBollingerStrategyV2 import RSIBollingerStrategyV2

        if params is None:
            params = make_params_v2()
        mock_ig = MagicMock()
        mock_streaming = MagicMock()
        trading_config = make_trading_config(**trading_overrides)
        strat = RSIBollingerStrategyV2(
            params=params,
            ig_client=mock_ig,
            streaming_client=mock_streaming,
            trading_config=trading_config,
        )
        return strat, mock_ig, mock_streaming

    return _factory
