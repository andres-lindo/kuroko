"""Tests for kuroko.py — strategy routing, startup wiring, and lifecycle.

Covers:
- _wire_strategy: api_mode routing (rest/streaming), ConfigurationError paths,
  return_streaming tuple convention
- load_strategy: known/unknown/missing-class scenarios
- _run_strategy: normal run, KeyboardInterrupt, streaming stop, exception re-raise
- main(): IGClient instantiation, load_strategy call, ConfigurationError exit

All external dependencies (IGClient, IGStreamingClient, strategy classes,
logging, config) are mocked — no live credentials required.

Note: kuroko.py executes module-level side-effects (load_dotenv, basicConfig,
getLogger) on import. These are harmless in test context because:
- load_dotenv silently does nothing when credentials.env is absent
- basicConfig is idempotent (only runs once per process)
- We patch ig_client.IGService to prevent real authentication
"""

import logging
import sys
import types
from unittest.mock import MagicMock, call, patch

import pytest

from kuroko import (
    ConfigurationError,
    _run_strategy,
    _wire_strategy,
    load_strategy,
    main,
)
from logging_setup import setup_logging
from strategies.RSIBollingerStrategy import RSIBollingerStrategy, load_params

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _make_main_config() -> dict:
    """Return a minimal but complete config dict for main() tests.

    Returns:
        Dict with logging and trading sections matching load_app_config output.
    """
    return {
        "logging": {
            "log_type": [],
            "log_level": "INFO",
            "log_dir": "logs",
            "log_file_name": "kuroko.log",
            "retention_days": 7,
            "console_logging": False,
            "structured_format": False,
            "azure_log_partition_key": "test",
        },
        "trading": {
            "epic": "IX.D.SPTRD.IFMM.IP",
            "leverage": 20,
            "demo_starting_balance": 20000.0,
            "initial_cash_balance": 4000.0,
            "security_buffer": 1000.0,
            "spread": 1.0,
        },
    }


# --------------------------------------------------------------------------- #
# load_strategy                                                                #
# --------------------------------------------------------------------------- #


class TestLoadStrategy:
    """Tests for kuroko.load_strategy() — module discovery and class extraction."""

    def test_returns_class_and_load_params_for_known_strategy(self):
        """load_strategy returns (StrategyClass, load_params_fn) for a valid strategy name."""
        strategy_class, load_params_fn = load_strategy("RSIBollingerStrategy")

        assert strategy_class is RSIBollingerStrategy
        assert load_params_fn is load_params

    def test_exits_with_code_1_for_unknown_strategy_module(self):
        """load_strategy calls sys.exit(1) when the strategy module does not exist."""
        with pytest.raises(SystemExit) as exc_info:
            load_strategy("NonExistentStrategy99")

        assert exc_info.value.code == 1

    def test_exits_with_code_1_when_module_lacks_class(self, tmp_path, monkeypatch):
        """load_strategy exits(1) when the module exists but does not export the class."""
        strategy_dir = tmp_path / "strategies"
        strategy_dir.mkdir()
        (strategy_dir / "__init__.py").touch()
        (strategy_dir / "EmptyStrategy.py").write_text(
            '"""Empty strategy for testing."""\n'
        )

        monkeypatch.syspath_prepend(str(tmp_path))
        for key in list(sys.modules.keys()):
            if key.startswith("strategies"):
                del sys.modules[key]

        with pytest.raises(SystemExit) as exc_info:
            load_strategy("EmptyStrategy")

        assert exc_info.value.code == 1

        for key in list(sys.modules.keys()):
            if key.startswith("strategies"):
                del sys.modules[key]


# --------------------------------------------------------------------------- #
# _wire_strategy — REST mode                                                   #
# --------------------------------------------------------------------------- #


class TestRestModeWiring:
    def test_rest_mode_does_not_instantiate_streaming_client(
        self, make_params_v1, make_trading_config
    ):
        params = make_params_v1(api_mode="rest")
        mock_ig = MagicMock()
        mock_strategy_class = MagicMock()
        trading_config = make_trading_config()

        with patch("kuroko.IGStreamingClient") as mock_streaming_cls:
            _wire_strategy(
                strategy_class=mock_strategy_class,
                params=params,
                ig=mock_ig,
                trading_config=trading_config,
            )

        mock_streaming_cls.assert_not_called()

    def test_rest_mode_instantiates_strategy_without_streaming_client(
        self, make_params_v1, make_trading_config
    ):
        params = make_params_v1(api_mode="rest")
        mock_ig = MagicMock()
        mock_strategy_class = MagicMock()
        trading_config = make_trading_config()

        with patch("kuroko.IGStreamingClient"):
            _wire_strategy(
                strategy_class=mock_strategy_class,
                params=params,
                ig=mock_ig,
                trading_config=trading_config,
            )

        mock_strategy_class.assert_called_once_with(
            params=params, ig_client=mock_ig, trading_config=trading_config
        )

    def test_rest_mode_returns_strategy_instance(
        self, make_params_v1, make_trading_config
    ):
        params = make_params_v1(api_mode="rest")
        mock_ig = MagicMock()
        mock_strategy_instance = MagicMock()
        mock_strategy_class = MagicMock(return_value=mock_strategy_instance)
        trading_config = make_trading_config()

        with patch("kuroko.IGStreamingClient"):
            result = _wire_strategy(
                strategy_class=mock_strategy_class,
                params=params,
                ig=mock_ig,
                trading_config=trading_config,
            )

        assert result is mock_strategy_instance

    def test_rest_mode_return_streaming_true_returns_none_client(
        self, make_params_v1, make_trading_config
    ):
        params = make_params_v1(api_mode="rest")
        mock_ig = MagicMock()
        mock_strategy_class = MagicMock()
        trading_config = make_trading_config()

        result = _wire_strategy(
            mock_strategy_class, params, mock_ig, trading_config, return_streaming=True
        )

        assert isinstance(result, tuple)
        strat, streaming = result
        assert strat is mock_strategy_class.return_value
        assert streaming is None


# --------------------------------------------------------------------------- #
# _wire_strategy — streaming mode                                              #
# --------------------------------------------------------------------------- #


class TestStreamingModeWiring:
    def test_streaming_mode_instantiates_streaming_client_with_resolution(
        self, make_params_v1, make_trading_config
    ):
        params = make_params_v1(api_mode="streaming", candle_frequency="5min")
        mock_ig = MagicMock()
        mock_ig.ig_service = MagicMock(name="fake_ig_service")
        mock_strategy_class = MagicMock()
        trading_config = make_trading_config(epic="IX.D.SPTRD.IFMM.IP")

        with patch("kuroko.IGStreamingClient") as mock_streaming_cls:
            _wire_strategy(
                strategy_class=mock_strategy_class,
                params=params,
                ig=mock_ig,
                trading_config=trading_config,
            )

        mock_streaming_cls.assert_called_once_with(
            mock_ig.ig_service, "IX.D.SPTRD.IFMM.IP", resolution="5MINUTE"
        )

    def test_streaming_mode_passes_streaming_client_to_strategy(
        self, make_params_v1, make_trading_config
    ):
        params = make_params_v1(api_mode="streaming")
        mock_ig = MagicMock()
        mock_streaming_instance = MagicMock(name="streaming_client")
        mock_strategy_class = MagicMock()
        trading_config = make_trading_config()

        with patch("kuroko.IGStreamingClient", return_value=mock_streaming_instance):
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

    def test_streaming_mode_returns_strategy_instance(
        self, make_params_v1, make_trading_config
    ):
        params = make_params_v1(api_mode="streaming")
        mock_ig = MagicMock()
        mock_strategy_instance = MagicMock()
        mock_strategy_class = MagicMock(return_value=mock_strategy_instance)
        trading_config = make_trading_config()

        with patch("kuroko.IGStreamingClient"):
            result = _wire_strategy(
                strategy_class=mock_strategy_class,
                params=params,
                ig=mock_ig,
                trading_config=trading_config,
            )

        assert result is mock_strategy_instance


# --------------------------------------------------------------------------- #
# _wire_strategy — ConfigurationError paths                                    #
# --------------------------------------------------------------------------- #


class TestWireStrategyConfigurationErrors:
    def test_missing_api_mode_raises_configuration_error(self, make_trading_config):
        params = types.SimpleNamespace(bb_period=20)
        mock_ig = MagicMock()
        mock_strategy_class = MagicMock()
        trading_config = make_trading_config()

        with patch("kuroko.IGStreamingClient"):
            with pytest.raises(ConfigurationError):
                _wire_strategy(
                    strategy_class=mock_strategy_class,
                    params=params,
                    ig=mock_ig,
                    trading_config=trading_config,
                )

    def test_unknown_api_mode_raises_configuration_error(
        self, make_params_v1, make_trading_config
    ):
        params = make_params_v1(api_mode="websocket")
        mock_ig = MagicMock()
        mock_strategy_class = MagicMock()
        trading_config = make_trading_config()

        with patch("kuroko.IGStreamingClient"):
            with pytest.raises(ConfigurationError):
                _wire_strategy(
                    strategy_class=mock_strategy_class,
                    params=params,
                    ig=mock_ig,
                    trading_config=trading_config,
                )


# --------------------------------------------------------------------------- #
# _run_strategy                                                                #
# --------------------------------------------------------------------------- #


class TestRunStrategy:
    """Tests for kuroko._run_strategy lifecycle — run, interrupt, stop, re-raise."""

    def test_calls_strat_run_once(self):
        mock_strat = MagicMock()

        _run_strategy(mock_strat)

        mock_strat.run.assert_called_once()

    def test_keyboard_interrupt_handled_without_propagating(self):
        mock_strat = MagicMock()
        mock_strat.run.side_effect = KeyboardInterrupt

        _run_strategy(mock_strat)  # must not raise

    def test_stops_streaming_client_on_normal_exit(self):
        mock_strat = MagicMock()
        mock_streaming = MagicMock()

        _run_strategy(mock_strat, streaming_client=mock_streaming)

        mock_streaming.stop.assert_called_once()

    def test_stops_streaming_client_on_keyboard_interrupt(self):
        mock_strat = MagicMock()
        mock_strat.run.side_effect = KeyboardInterrupt
        mock_streaming = MagicMock()

        _run_strategy(mock_strat, streaming_client=mock_streaming)

        mock_streaming.stop.assert_called_once()

    def test_non_keyboard_exception_reraises_after_cleanup(self):
        mock_strat = MagicMock()
        mock_strat.run.side_effect = RuntimeError("critical error")
        mock_streaming = MagicMock()

        with pytest.raises(RuntimeError, match="critical error"):
            _run_strategy(mock_strat, streaming_client=mock_streaming)

        mock_streaming.stop.assert_called_once()

    def test_streaming_client_stop_called_on_streaming_shutdown(
        self, make_params_v1, make_trading_config
    ):
        params = make_params_v1(api_mode="streaming")
        mock_ig = MagicMock()
        mock_streaming_instance = MagicMock()
        mock_strategy_instance = MagicMock()
        mock_strategy_class = MagicMock(return_value=mock_strategy_instance)
        trading_config = make_trading_config()

        mock_strategy_instance.run.side_effect = KeyboardInterrupt

        with patch("kuroko.IGStreamingClient", return_value=mock_streaming_instance):
            strat, streaming = _wire_strategy(
                strategy_class=mock_strategy_class,
                params=params,
                ig=mock_ig,
                trading_config=trading_config,
                return_streaming=True,
            )
            _run_strategy(strat, streaming)

        mock_streaming_instance.stop.assert_called_once()

    def test_rest_mode_no_streaming_stop_on_shutdown(
        self, make_params_v1, make_trading_config
    ):
        params = make_params_v1(api_mode="rest")
        mock_ig = MagicMock()
        mock_strategy_instance = MagicMock()
        mock_strategy_class = MagicMock(return_value=mock_strategy_instance)
        trading_config = make_trading_config()

        mock_strategy_instance.run.side_effect = KeyboardInterrupt

        with patch("kuroko.IGStreamingClient") as mock_streaming_cls:
            strat, streaming = _wire_strategy(
                strategy_class=mock_strategy_class,
                params=params,
                ig=mock_ig,
                trading_config=trading_config,
                return_streaming=True,
            )
            _run_strategy(strat, streaming)

        mock_streaming_cls.assert_not_called()


# --------------------------------------------------------------------------- #
# main()                                                                       #
# --------------------------------------------------------------------------- #


class TestMain:
    """Tests for kuroko.main() — startup wiring and CLI argument handling."""

    @pytest.fixture
    def run_main(self):
        """Fixture: returns a callable that runs main() with all externals mocked.

        Returns:
            A factory function that accepts an optional strategy_name and runs
            main() with the given strategy, returning the mocks used.
        """

        def _run(strategy_name="RSIBollingerStrategy"):
            mock_strategy_class = MagicMock()
            mock_load_params = MagicMock()
            mock_params = types.SimpleNamespace(
                api_mode="rest", candle_frequency="15min"
            )
            mock_load_params.return_value = mock_params
            mock_ig_instance = MagicMock()
            mock_ig_cls = MagicMock(return_value=mock_ig_instance)
            mock_strategy_class.return_value = MagicMock()

            with patch.object(sys, "argv", ["kuroko.py", "--strategy", strategy_name]):
                with patch(
                    "kuroko.load_strategy",
                    return_value=(mock_strategy_class, mock_load_params),
                ):
                    with patch(
                        "kuroko.load_app_config", return_value=_make_main_config()
                    ):
                        with patch("kuroko.setup_logging"):
                            with patch("kuroko.IGClient", mock_ig_cls):
                                with patch("kuroko._run_strategy"):
                                    main()

            return {
                "strategy_class": mock_strategy_class,
                "load_params": mock_load_params,
                "ig_instance": mock_ig_instance,
                "ig_cls": mock_ig_cls,
            }

        return _run

    def test_main_creates_ig_client_at_startup(self, run_main):
        mocks = run_main()
        # Verify that the IGClient constructor was actually called once
        mocks["ig_cls"].assert_called_once()

    def test_main_calls_load_strategy_with_cli_argument(self):
        with patch.object(
            sys, "argv", ["kuroko.py", "--strategy", "RSIBollingerStrategy"]
        ):
            with patch(
                "kuroko.load_strategy",
                return_value=(
                    MagicMock(),
                    MagicMock(return_value=types.SimpleNamespace(api_mode="rest")),
                ),
            ) as mock_ls:
                with patch("kuroko.load_app_config", return_value=_make_main_config()):
                    with patch("kuroko.setup_logging"):
                        with patch("kuroko.IGClient"):
                            with patch("kuroko._run_strategy"):
                                main()

        mock_ls.assert_called_once_with("RSIBollingerStrategy")

    def test_main_exits_with_1_on_configuration_error(self):
        mock_strategy_class = MagicMock()
        mock_load_params = MagicMock(
            return_value=types.SimpleNamespace(api_mode="invalid_mode")
        )

        with patch.object(
            sys, "argv", ["kuroko.py", "--strategy", "RSIBollingerStrategy"]
        ):
            with patch(
                "kuroko.load_strategy",
                return_value=(mock_strategy_class, mock_load_params),
            ):
                with patch("kuroko.load_app_config", return_value=_make_main_config()):
                    with patch("kuroko.setup_logging"):
                        with patch("kuroko.IGClient"):
                            with pytest.raises(SystemExit) as exc_info:
                                main()

        assert exc_info.value.code == 1


# --------------------------------------------------------------------------- #
# _wire_strategy — return_streaming flag                                       #
# --------------------------------------------------------------------------- #


class TestWireStrategyReturnStreamingFlag:
    """Tests for the return_streaming=True convention in _wire_strategy."""

    def test_streaming_mode_return_streaming_true_returns_streaming_client(
        self, make_params_v1, make_trading_config
    ):
        params = make_params_v1(api_mode="streaming", candle_frequency="5min")
        mock_ig = MagicMock()
        mock_streaming_instance = MagicMock()
        mock_strategy_class = MagicMock()
        trading_config = make_trading_config()

        with patch("kuroko.IGStreamingClient", return_value=mock_streaming_instance):
            result = _wire_strategy(
                mock_strategy_class,
                params,
                mock_ig,
                trading_config,
                return_streaming=True,
            )

        assert isinstance(result, tuple)
        strat, streaming = result
        assert strat is mock_strategy_class.return_value
        assert streaming is mock_streaming_instance

    def test_without_return_streaming_returns_strategy_directly(
        self, make_params_v1, make_trading_config
    ):
        params = make_params_v1(api_mode="rest")
        mock_ig = MagicMock()
        mock_strategy_class = MagicMock()
        trading_config = make_trading_config()

        result = _wire_strategy(mock_strategy_class, params, mock_ig, trading_config)

        # Not a tuple — returns the strategy instance directly
        assert result is mock_strategy_class.return_value
        assert not isinstance(result, tuple)

    def test_rest_mode_return_streaming_streaming_is_none(
        self, make_params_v1, make_trading_config
    ):
        params = make_params_v1(api_mode="rest")
        mock_ig = MagicMock()
        mock_strategy_class = MagicMock()
        trading_config = make_trading_config()

        strat, streaming = _wire_strategy(
            mock_strategy_class, params, mock_ig, trading_config, return_streaming=True
        )

        assert streaming is None

    def test_run_strategy_without_streaming_client_does_not_call_stop(self):
        mock_strat = MagicMock()

        _run_strategy(mock_strat, streaming_client=None)

        # No stop() called — just confirms no AttributeError on None.stop()
        mock_strat.run.assert_called_once()


# --------------------------------------------------------------------------- #
# setup_logging — override_level parameter                                     #
# --------------------------------------------------------------------------- #


class TestSetupLoggingOverrideLevel:
    """Tests for logging_setup.setup_logging(override_level=...) behaviour."""

    def _minimal_config(self) -> dict:
        """Return a minimal log_config that disables all file and Azure handlers."""
        return {
            "log_type": [],
            "log_level": "INFO",
            "log_dir": "logs",
            "log_file_name": "kuroko.log",
            "retention_days": 7,
            "console_logging": False,
            "structured_format": False,
        }

    def test_override_level_takes_precedence_over_config_value(self):
        """When override_level='DEBUG', root logger is set to DEBUG regardless of config."""
        config = self._minimal_config()
        config["log_level"] = "WARNING"

        setup_logging(config, partition_key="", override_level="DEBUG")

        assert logging.getLogger().level == logging.DEBUG

    def test_config_level_used_when_override_level_is_none(self):
        """Without override_level, root logger level comes from log_config."""
        config = self._minimal_config()
        config["log_level"] = "ERROR"

        setup_logging(config, partition_key="")

        assert logging.getLogger().level == logging.ERROR

    def test_main_passes_cli_log_level_to_setup_logging(self):
        """main() passes args.log_level as override_level to setup_logging."""
        mock_strategy_class = MagicMock()
        mock_load_params = MagicMock(
            return_value=types.SimpleNamespace(
                api_mode="rest", candle_frequency="15min"
            )
        )

        with patch.object(
            sys,
            "argv",
            ["kuroko.py", "--strategy", "RSIBollingerStrategy", "--log-level", "DEBUG"],
        ):
            with patch(
                "kuroko.load_strategy",
                return_value=(mock_strategy_class, mock_load_params),
            ):
                with patch("kuroko.load_app_config", return_value=_make_main_config()):
                    with patch("kuroko.setup_logging") as mock_setup:
                        with patch("kuroko.IGClient"):
                            with patch("kuroko._run_strategy"):
                                main()

        _args, _kwargs = mock_setup.call_args
        assert _kwargs.get("override_level") == "DEBUG"
