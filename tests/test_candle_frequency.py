"""Unit tests for candle_frequency configurability (REQ-candle-freq).

Covers:
- candle_frequency key present in RSIBollingerStrategyV2.json
- candle_frequency present in RSIBollingerStrategyV2 _PARAMS_SCHEMA
- _resolution_to_minutes() utility mapping in ig_streaming_client
- _subscribe_tick_fallback uses parsed resolution_minutes (not hardcoded 5)
- kuroko._wire_strategy passes resolution derived from candle_frequency to
  IGStreamingClient when api_mode='streaming'
"""

import json
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
_V2_JSON = _PROJECT_ROOT / "strategies" / "RSIBollingerStrategyV2.json"


# ---------------------------------------------------------------------------
# REQ-candle-freq-1: candle_frequency in V2 JSON
# ---------------------------------------------------------------------------


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

    def test_v2_json_candle_frequency_default_is_5min(self):
        """candle_frequency default value must be '5min'."""
        data = json.loads(_V2_JSON.read_text(encoding="utf-8"))
        assert (
            data.get("candle_frequency") == "5min"
        ), f"Expected candle_frequency='5min', got {data.get('candle_frequency')!r}"


# ---------------------------------------------------------------------------
# REQ-candle-freq-2: candle_frequency in _PARAMS_SCHEMA
# ---------------------------------------------------------------------------


class TestV2ParamsSchema:
    """candle_frequency must be declared in RSIBollingerStrategyV2._PARAMS_SCHEMA."""

    def test_candle_frequency_in_params_schema(self):
        """_PARAMS_SCHEMA must include candle_frequency."""
        from strategies.RSIBollingerStrategyV2 import _PARAMS_SCHEMA

        assert (
            "candle_frequency" in _PARAMS_SCHEMA
        ), "candle_frequency missing from _PARAMS_SCHEMA in RSIBollingerStrategyV2.py"

    def test_candle_frequency_schema_type_is_str(self):
        """candle_frequency schema type must be str (matching V1 convention)."""
        from strategies.RSIBollingerStrategyV2 import _PARAMS_SCHEMA

        assert (
            _PARAMS_SCHEMA.get("candle_frequency") is str
        ), f"Expected candle_frequency schema type str, got {_PARAMS_SCHEMA.get('candle_frequency')!r}"


# ---------------------------------------------------------------------------
# REQ-candle-freq-3: _resolution_to_minutes utility
# ---------------------------------------------------------------------------


class TestResolutionToMinutes:
    """_resolution_to_minutes() must map IG resolution strings to integer minutes."""

    def test_5minute_maps_to_5(self):
        """'5MINUTE' must map to 5."""
        from ig_streaming_client import _resolution_to_minutes

        assert _resolution_to_minutes("5MINUTE") == 5

    def test_1minute_maps_to_1(self):
        """'1MINUTE' must map to 1."""
        from ig_streaming_client import _resolution_to_minutes

        assert _resolution_to_minutes("1MINUTE") == 1

    def test_15minute_maps_to_15(self):
        """'15MINUTE' must map to 15."""
        from ig_streaming_client import _resolution_to_minutes

        assert _resolution_to_minutes("15MINUTE") == 15

    def test_1hour_maps_to_60(self):
        """'1HOUR' must map to 60."""
        from ig_streaming_client import _resolution_to_minutes

        assert _resolution_to_minutes("1HOUR") == 60

    def test_unknown_resolution_raises_value_error(self):
        """Unknown resolution strings must raise ValueError."""
        from ig_streaming_client import _resolution_to_minutes

        with pytest.raises(ValueError):
            _resolution_to_minutes("UNKNOWN")


# ---------------------------------------------------------------------------
# REQ-candle-freq-4: candle_frequency_to_resolution utility
# ---------------------------------------------------------------------------


class TestCandleFrequencyToResolution:
    """candle_frequency_to_resolution() must map 'Nmin' strings to IG resolution strings."""

    def test_5min_maps_to_5minute(self):
        """'5min' must map to '5MINUTE'."""
        from ig_streaming_client import candle_frequency_to_resolution

        assert candle_frequency_to_resolution("5min") == "5MINUTE"

    def test_1min_maps_to_1minute(self):
        """'1min' must map to '1MINUTE'."""
        from ig_streaming_client import candle_frequency_to_resolution

        assert candle_frequency_to_resolution("1min") == "1MINUTE"

    def test_15min_maps_to_15minute(self):
        """'15min' must map to '15MINUTE'."""
        from ig_streaming_client import candle_frequency_to_resolution

        assert candle_frequency_to_resolution("15min") == "15MINUTE"

    def test_60min_maps_to_1hour(self):
        """'60min' must map to '1HOUR'."""
        from ig_streaming_client import candle_frequency_to_resolution

        assert candle_frequency_to_resolution("60min") == "1HOUR"

    def test_invalid_format_raises_value_error(self):
        """Non-'Nmin' strings must raise ValueError."""
        from ig_streaming_client import candle_frequency_to_resolution

        with pytest.raises(ValueError):
            candle_frequency_to_resolution("5minutes")

    def test_non_numeric_minutes_raises_value_error(self):
        """'Xmin' with non-numeric X must raise ValueError."""
        from ig_streaming_client import candle_frequency_to_resolution

        with pytest.raises(ValueError):
            candle_frequency_to_resolution("Xmin")


# ---------------------------------------------------------------------------
# REQ-candle-freq-5: TickAggregator resolution not hardcoded in fallback
# ---------------------------------------------------------------------------


class TestTickFallbackUsesConfiguredResolution:
    """_subscribe_tick_fallback must use self._resolution to drive TickAggregator,
    not a hardcoded 5."""

    def _make_client_with_resolution(self, resolution: str) -> "IGStreamingClient":
        """Build an IGStreamingClient with a given resolution and mocked stream service."""
        from ig_streaming_client import IGStreamingClient

        ig_service = MagicMock()
        client = IGStreamingClient(
            ig_service, "IX.D.NASDAQ.IFMM.IP", resolution=resolution
        )
        mock_stream_svc = MagicMock()
        mock_stream_svc.subscribe.return_value = None
        client._stream_svc = mock_stream_svc
        return client

    def test_tick_fallback_uses_1_minute_resolution(self):
        """TickAggregator must be created with resolution_minutes=1 when resolution='1MINUTE'."""
        from ig_streaming_client import TickAggregator

        client = self._make_client_with_resolution("1MINUTE")

        captured = {}

        original_init = TickAggregator.__init__

        def capturing_init(self_agg, resolution_minutes, on_candle):
            captured["resolution_minutes"] = resolution_minutes
            original_init(self_agg, resolution_minutes, on_candle)

        with patch.object(TickAggregator, "__init__", capturing_init):
            client._subscribe_tick_fallback()

        assert (
            captured.get("resolution_minutes") == 1
        ), f"Expected resolution_minutes=1, got {captured.get('resolution_minutes')!r}"

    def test_tick_fallback_uses_15_minute_resolution(self):
        """TickAggregator must be created with resolution_minutes=15 when resolution='15MINUTE'."""
        from ig_streaming_client import TickAggregator

        client = self._make_client_with_resolution("15MINUTE")

        captured = {}

        original_init = TickAggregator.__init__

        def capturing_init(self_agg, resolution_minutes, on_candle):
            captured["resolution_minutes"] = resolution_minutes
            original_init(self_agg, resolution_minutes, on_candle)

        with patch.object(TickAggregator, "__init__", capturing_init):
            client._subscribe_tick_fallback()

        assert (
            captured.get("resolution_minutes") == 15
        ), f"Expected resolution_minutes=15, got {captured.get('resolution_minutes')!r}"

    def test_tick_fallback_uses_5_minute_resolution_by_default(self):
        """TickAggregator must be created with resolution_minutes=5 for default '5MINUTE'."""
        from ig_streaming_client import TickAggregator

        client = self._make_client_with_resolution("5MINUTE")

        captured = {}

        original_init = TickAggregator.__init__

        def capturing_init(self_agg, resolution_minutes, on_candle):
            captured["resolution_minutes"] = resolution_minutes
            original_init(self_agg, resolution_minutes, on_candle)

        with patch.object(TickAggregator, "__init__", capturing_init):
            client._subscribe_tick_fallback()

        assert (
            captured.get("resolution_minutes") == 5
        ), f"Expected resolution_minutes=5, got {captured.get('resolution_minutes')!r}"


# ---------------------------------------------------------------------------
# REQ-candle-freq-6: kuroko._wire_strategy passes resolution to IGStreamingClient
# ---------------------------------------------------------------------------


class TestWireStrategyPassesResolution:
    """kuroko._wire_strategy must derive resolution from candle_frequency and pass
    it to IGStreamingClient when api_mode='streaming'."""

    def _make_streaming_params(
        self, candle_frequency: str = "5min"
    ) -> types.SimpleNamespace:
        """Return a minimal streaming-mode params namespace."""
        return types.SimpleNamespace(
            api_mode="streaming",
            candle_frequency=candle_frequency,
        )

    def test_wire_strategy_passes_5min_resolution_to_streaming_client(self):
        """candle_frequency='5min' must result in IGStreamingClient called with resolution='5MINUTE'."""
        params = self._make_streaming_params("5min")
        mock_ig = MagicMock()
        mock_ig.ig_service = MagicMock(name="fake_ig_service")
        mock_strategy_class = MagicMock()
        trading_config = types.SimpleNamespace(epic="IX.D.NASDAQ.IFMM.IP")

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

    def test_wire_strategy_passes_15min_resolution_to_streaming_client(self):
        """candle_frequency='15min' must result in IGStreamingClient called with resolution='15MINUTE'."""
        params = self._make_streaming_params("15min")
        mock_ig = MagicMock()
        mock_ig.ig_service = MagicMock(name="fake_ig_service")
        mock_strategy_class = MagicMock()
        trading_config = types.SimpleNamespace(epic="IX.D.NASDAQ.IFMM.IP")

        with patch("kuroko.IGStreamingClient") as mock_streaming_cls:
            from kuroko import _wire_strategy

            _wire_strategy(
                strategy_class=mock_strategy_class,
                params=params,
                ig=mock_ig,
                trading_config=trading_config,
            )

        mock_streaming_cls.assert_called_once_with(
            mock_ig.ig_service, "IX.D.NASDAQ.IFMM.IP", resolution="15MINUTE"
        )

    def test_wire_strategy_passes_1min_resolution_to_streaming_client(self):
        """candle_frequency='1min' must result in IGStreamingClient called with resolution='1MINUTE'."""
        params = self._make_streaming_params("1min")
        mock_ig = MagicMock()
        mock_ig.ig_service = MagicMock(name="fake_ig_service")
        mock_strategy_class = MagicMock()
        trading_config = types.SimpleNamespace(epic="IX.D.NASDAQ.IFMM.IP")

        with patch("kuroko.IGStreamingClient") as mock_streaming_cls:
            from kuroko import _wire_strategy

            _wire_strategy(
                strategy_class=mock_strategy_class,
                params=params,
                ig=mock_ig,
                trading_config=trading_config,
            )

        mock_streaming_cls.assert_called_once_with(
            mock_ig.ig_service, "IX.D.NASDAQ.IFMM.IP", resolution="1MINUTE"
        )
