"""Unit tests for kuroko.py api_mode routing logic (REQ-13).

Verifies that kuroko.py correctly reads api_mode from strategy params and:
- Creates IGStreamingClient when api_mode="streaming"
- Does NOT create IGStreamingClient when api_mode="rest"
- Raises a configuration error when api_mode is absent

All external dependencies (IGClient, IGStreamingClient, strategy classes)
are mocked — no live credentials required.
"""

import types
import pytest
from unittest.mock import MagicMock, patch, call

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _make_params(**overrides) -> types.SimpleNamespace:
    """Return a minimal params SimpleNamespace."""
    defaults = {
        "api_mode": "rest",
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _make_trading_config(**overrides) -> types.SimpleNamespace:
    """Return a minimal trading_config SimpleNamespace."""
    defaults = {
        "epic": "IX.D.NASDAQ.IFMM.IP",
        "spread": 1.0,
    }
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


# --------------------------------------------------------------------------- #
# Import the module under test — patches applied per test                      #
# --------------------------------------------------------------------------- #


class TestRestModeRouting:
    """REQ-13/Scenario 2: api_mode='rest' uses existing REST flow only."""

    def test_rest_mode_does_not_instantiate_streaming_client(self):
        """REST mode must not create IGStreamingClient."""
        params = _make_params(api_mode="rest")
        mock_ig = MagicMock()
        mock_strategy_class = MagicMock()
        trading_config = _make_trading_config()

        with patch("kuroko.IGStreamingClient") as mock_streaming_cls:
            from kuroko import _wire_strategy

            _wire_strategy(
                strategy_class=mock_strategy_class,
                params=params,
                ig=mock_ig,
                trading_config=trading_config,
            )

        mock_streaming_cls.assert_not_called()

    def test_rest_mode_instantiates_strategy_without_streaming_client(self):
        """REST mode must instantiate the strategy without streaming_client kwarg."""
        params = _make_params(api_mode="rest")
        mock_ig = MagicMock()
        mock_strategy_class = MagicMock()
        trading_config = _make_trading_config()

        with patch("kuroko.IGStreamingClient"):
            from kuroko import _wire_strategy

            _wire_strategy(
                strategy_class=mock_strategy_class,
                params=params,
                ig=mock_ig,
                trading_config=trading_config,
            )

        mock_strategy_class.assert_called_once_with(
            params=params, ig_client=mock_ig, trading_config=trading_config
        )

    def test_rest_mode_returns_strategy_instance(self):
        """REST mode must return the strategy instance created from strategy_class."""
        params = _make_params(api_mode="rest")
        mock_ig = MagicMock()
        mock_strategy_instance = MagicMock()
        mock_strategy_class = MagicMock(return_value=mock_strategy_instance)
        trading_config = _make_trading_config()

        with patch("kuroko.IGStreamingClient"):
            from kuroko import _wire_strategy

            result = _wire_strategy(
                strategy_class=mock_strategy_class,
                params=params,
                ig=mock_ig,
                trading_config=trading_config,
            )

        assert result is mock_strategy_instance


class TestStreamingModeRouting:
    """REQ-13/Scenario 1: api_mode='streaming' creates IGStreamingClient and wires strategy."""

    def test_streaming_mode_instantiates_streaming_client(self):
        """Streaming mode must create IGStreamingClient with ig_service, epic, and resolution."""
        params = _make_params(api_mode="streaming", candle_frequency="5min")
        mock_ig = MagicMock()
        mock_ig.ig_service = MagicMock(name="fake_ig_service")
        mock_strategy_class = MagicMock()
        trading_config = _make_trading_config(epic="IX.D.NASDAQ.IFMM.IP")

        with patch("kuroko.IGStreamingClient") as mock_streaming_cls:
            from kuroko import _wire_strategy

            _wire_strategy(
                strategy_class=mock_strategy_class,
                params=params,
                ig=mock_ig,
                trading_config=trading_config,
            )

        mock_streaming_cls.assert_called_once_with(
            mock_ig.ig_service, "IX.D.NASDAQ.IFMM.IP", resolution="5MINUTE"
        )

    def test_streaming_mode_passes_streaming_client_to_strategy(self):
        """Streaming mode must pass the streaming client to the strategy constructor."""
        params = _make_params(api_mode="streaming")
        mock_ig = MagicMock()
        mock_streaming_instance = MagicMock(name="streaming_client")
        mock_strategy_class = MagicMock()
        trading_config = _make_trading_config()

        with patch("kuroko.IGStreamingClient", return_value=mock_streaming_instance):
            from kuroko import _wire_strategy

            _wire_strategy(
                strategy_class=mock_strategy_class,
                params=params,
                ig=mock_ig,
                trading_config=trading_config,
            )

        mock_strategy_class.assert_called_once_with(
            params=params,
            ig_client=mock_ig,
            streaming_client=mock_streaming_instance,
            trading_config=trading_config,
        )

    def test_streaming_mode_returns_strategy_instance(self):
        """Streaming mode must return the strategy instance."""
        params = _make_params(api_mode="streaming")
        mock_ig = MagicMock()
        mock_streaming_instance = MagicMock()
        mock_strategy_instance = MagicMock()
        mock_strategy_class = MagicMock(return_value=mock_strategy_instance)
        trading_config = _make_trading_config()

        with patch("kuroko.IGStreamingClient", return_value=mock_streaming_instance):
            from kuroko import _wire_strategy

            result = _wire_strategy(
                strategy_class=mock_strategy_class,
                params=params,
                ig=mock_ig,
                trading_config=trading_config,
            )

        assert result is mock_strategy_instance


class TestMissingApiMode:
    """REQ-13/Scenario 3: absent api_mode must raise a configuration error."""

    def test_missing_api_mode_raises_configuration_error(self):
        """No api_mode attribute must raise ConfigurationError (not silently default)."""
        # params with NO api_mode attribute at all
        params = types.SimpleNamespace(bb_period=20)
        mock_ig = MagicMock()
        mock_strategy_class = MagicMock()
        trading_config = _make_trading_config()

        with patch("kuroko.IGStreamingClient"):
            from kuroko import _wire_strategy, ConfigurationError

            with pytest.raises(ConfigurationError):
                _wire_strategy(
                    strategy_class=mock_strategy_class,
                    params=params,
                    ig=mock_ig,
                    trading_config=trading_config,
                )

    def test_unknown_api_mode_raises_configuration_error(self):
        """An unrecognised api_mode value must raise ConfigurationError."""
        params = _make_params(api_mode="websocket")
        mock_ig = MagicMock()
        mock_strategy_class = MagicMock()
        trading_config = _make_trading_config()

        with patch("kuroko.IGStreamingClient"):
            from kuroko import _wire_strategy, ConfigurationError

            with pytest.raises(ConfigurationError):
                _wire_strategy(
                    strategy_class=mock_strategy_class,
                    params=params,
                    ig=mock_ig,
                    trading_config=trading_config,
                )


class TestGracefulShutdown:
    """Verify streaming client is stopped on graceful shutdown."""

    def test_streaming_client_stop_called_on_shutdown(self):
        """When streaming mode is active and a shutdown occurs, stop() is called."""
        params = _make_params(api_mode="streaming")
        mock_ig = MagicMock()
        mock_streaming_instance = MagicMock()
        mock_strategy_instance = MagicMock()
        mock_strategy_class = MagicMock(return_value=mock_strategy_instance)
        trading_config = _make_trading_config()

        # Simulate strategy.run() raising KeyboardInterrupt
        mock_strategy_instance.run.side_effect = KeyboardInterrupt

        with patch("kuroko.IGStreamingClient", return_value=mock_streaming_instance):
            from kuroko import _wire_strategy, _run_strategy

            strat, streaming = _wire_strategy(
                strategy_class=mock_strategy_class,
                params=params,
                ig=mock_ig,
                trading_config=trading_config,
                return_streaming=True,
            )
            _run_strategy(strat, streaming)

        mock_streaming_instance.stop.assert_called_once()

    def test_rest_mode_no_streaming_stop_on_shutdown(self):
        """REST mode shutdown must not call any streaming stop() — no streaming client."""
        params = _make_params(api_mode="rest")
        mock_ig = MagicMock()
        mock_strategy_instance = MagicMock()
        mock_strategy_class = MagicMock(return_value=mock_strategy_instance)
        trading_config = _make_trading_config()

        mock_strategy_instance.run.side_effect = KeyboardInterrupt

        with patch("kuroko.IGStreamingClient") as mock_streaming_cls:
            from kuroko import _wire_strategy, _run_strategy

            strat, streaming = _wire_strategy(
                strategy_class=mock_strategy_class,
                params=params,
                ig=mock_ig,
                trading_config=trading_config,
                return_streaming=True,
            )
            _run_strategy(strat, streaming)

        # No streaming instance was created
        mock_streaming_cls.assert_not_called()
