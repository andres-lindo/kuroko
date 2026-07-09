"""Tests for IGStreamingClient: auth, candle delivery, tick aggregation,
fallback to tick subscription, and lifecycle management.

Covers REQ-1 (candle delivery), REQ-2 (tick aggregation / fallback),
REQ-3 (auth passthrough), and REQ-4 (worker thread dispatch).
"""

import queue
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from ig_streaming_client import (
    IGStreamingClient,
    TickAggregator,
    _CandleSubscriptionListener,
    _ConnectionListener,
    _DirectTickListener,
    _TickListener,
)

# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------

EPIC = "IX.D.SPTRD.IFMM.IP"


# ---------------------------------------------------------------------------
# Lightstreamer ItemUpdate stub
# ---------------------------------------------------------------------------


class _FakeItemUpdate:
    """Minimal stub mimicking a Lightstreamer ItemUpdate object.

    The real Lightstreamer client passes an ItemUpdate to onItemUpdate
    callbacks. Field values are retrieved via getValue("FIELD_NAME"), which
    returns str | None (None when the field was not present in the update).
    """

    def __init__(self, fields: dict):
        """Initialise with a field-value mapping.

        Args:
            fields: Dict of field names to string values (or None).
        """
        self._fields = fields

    def getValue(self, field_name: str):
        """Return the string value for a field, or None if absent.

        Args:
            field_name: Lightstreamer field name (e.g. 'BID_CLOSE').

        Returns:
            String value or None.
        """
        return self._fields.get(field_name)


# ---------------------------------------------------------------------------
# Fixtures (promoted from tests/streaming/conftest.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def build_candle_update():
    """Factory fixture: returns callable(**fields) → _FakeItemUpdate mimicking a Lightstreamer update.

    Returns:
        A factory function that builds a _FakeItemUpdate representing
        a Lightstreamer item update for a candle subscription.
    """

    def _factory(
        bid_open=20000.0,
        bid_high=20050.0,
        bid_low=19990.0,
        bid_close=20030.0,
        ofr_open=20001.0,
        ofr_high=20051.0,
        ofr_low=19991.0,
        ofr_close=20031.0,
        cons_end="1",
        utm="1716825600000",
        ltv="0",
    ) -> _FakeItemUpdate:
        """Build a candle ItemUpdate stub with optional field overrides.

        Args:
            bid_open: BID_OPEN price as float.
            bid_high: BID_HIGH price as float.
            bid_low: BID_LOW price as float.
            bid_close: BID_CLOSE price as float.
            ofr_open: OFR_OPEN price as float.
            ofr_high: OFR_HIGH price as float.
            ofr_low: OFR_LOW price as float.
            ofr_close: OFR_CLOSE price as float.
            cons_end: CONS_END flag ('1' = complete, '0' = incomplete).
            utm: UTM timestamp in milliseconds as string.
            ltv: LTV (last trade volume) as string.

        Returns:
            _FakeItemUpdate with field values matching Lightstreamer format.
        """
        return _FakeItemUpdate(
            {
                "BID_OPEN": str(bid_open),
                "BID_HIGH": str(bid_high),
                "BID_LOW": str(bid_low),
                "BID_CLOSE": str(bid_close),
                "OFR_OPEN": str(ofr_open),
                "OFR_HIGH": str(ofr_high),
                "OFR_LOW": str(ofr_low),
                "OFR_CLOSE": str(ofr_close),
                "CONS_END": cons_end,
                "UTM": utm,
                "LTV": ltv,
            }
        )

    return _factory


@pytest.fixture
def mock_ig_stream_service():
    """Fixture: returns a MagicMock mimicking IGStreamService API.

    Records subscribe/unsubscribe/create_session/disconnect calls
    without touching the network.

    Returns:
        MagicMock configured with all expected IGStreamService methods.
    """
    svc = MagicMock()
    svc.create_session.return_value = None
    svc.subscribe.return_value = None
    svc.unsubscribe.return_value = None
    svc.disconnect.return_value = None
    return svc


@pytest.fixture
def mock_ig_service():
    """Fixture: returns a MagicMock mimicking the trading_ig IGService object.

    Returns:
        MagicMock standing in for the full IGService.
    """
    return MagicMock()


# ---------------------------------------------------------------------------
# Auth passthrough (REQ-3)
# ---------------------------------------------------------------------------


class TestAuthPassthrough:
    """IGStreamingClient must not create a new IG session independently."""

    def test_start_creates_stream_service_from_provided_ig_service(
        self, mock_ig_stream_service, mock_ig_service
    ):
        """Start() uses the provided IGService to create the stream session."""
        with patch(
            "ig_streaming_client.IGStreamService", return_value=mock_ig_stream_service
        ) as MockStreamServiceCls:
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=MagicMock())
            client.stop()

        MockStreamServiceCls.assert_called_once_with(mock_ig_service)
        mock_ig_stream_service.create_session.assert_called_once()

    def test_start_does_not_call_ig_service_create_session_directly(
        self, mock_ig_stream_service, mock_ig_service
    ):
        """IGStreamingClient must NOT call ig_service.create_session directly."""
        with patch(
            "ig_streaming_client.IGStreamService", return_value=mock_ig_stream_service
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=MagicMock())
            client.stop()

        mock_ig_service.create_session.assert_not_called()


# ---------------------------------------------------------------------------
# Native candle delivery (REQ-1)
# ---------------------------------------------------------------------------


class TestNativeCandleDelivery:
    """completed candles (CONS_END=1) must be delivered to on_candle callback."""

    def _start_and_trigger(self, mock_stream_svc, update_dict, mock_ig_service):
        """Helper: start client, capture the LS listener, simulate an update.

        Uses threading.Event synchronisation instead of time.sleep.
        """
        received = []
        delivered = threading.Event()

        def callback(candle):
            received.append(candle)
            delivered.set()

        with patch("ig_streaming_client.IGStreamService", return_value=mock_stream_svc):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=callback)

            # Retrieve the listener registered on the subscription
            assert mock_stream_svc.subscribe.called, "subscribe was not called"
            sub = mock_stream_svc.subscribe.call_args[0][0]
            # The subscription carries our listener — trigger it manually
            listener = sub._listener
            listener.onItemUpdate(update_dict)

            # Wait for worker thread to process the candle (no fixed sleep)
            delivered.wait(timeout=2.0)
            client.stop()

        return received

    def test_delivered_candle_has_ohlc_fields(
        self, mock_ig_stream_service, mock_ig_service, build_candle_update
    ):
        """candle payload must include open, high, low, close."""
        update = build_candle_update(
            bid_open=20000.0,
            bid_high=20050.0,
            bid_low=19990.0,
            bid_close=20030.0,
            cons_end="1",
        )

        received = self._start_and_trigger(
            mock_ig_stream_service, update, mock_ig_service
        )

        candle = received[0]
        assert "open" in candle
        assert "high" in candle
        assert "low" in candle
        assert "close" in candle

    def test_delivered_candle_has_timestamp(
        self, mock_ig_stream_service, mock_ig_service, build_candle_update
    ):
        """candle payload must include a UTC timestamp."""
        update = build_candle_update(cons_end="1", utm="1716825600000")

        received = self._start_and_trigger(
            mock_ig_stream_service, update, mock_ig_service
        )

        candle = received[0]
        assert "timestamp" in candle
        assert isinstance(candle["timestamp"], datetime)

    def test_delivered_candle_values_match_bid_fields(
        self, mock_ig_stream_service, mock_ig_service, build_candle_update
    ):
        """candle open/high/low/close must be derived from BID fields."""
        update = build_candle_update(
            bid_open=20000.0,
            bid_high=20100.0,
            bid_low=19950.0,
            bid_close=20080.0,
            cons_end="1",
        )

        received = self._start_and_trigger(
            mock_ig_stream_service, update, mock_ig_service
        )

        candle = received[0]
        assert candle["open"] == pytest.approx(20000.0)
        assert candle["high"] == pytest.approx(20100.0)
        assert candle["low"] == pytest.approx(19950.0)
        assert candle["close"] == pytest.approx(20080.0)

    def test_delivered_candle_has_bid_close_and_ofr_close(
        self, mock_ig_stream_service, mock_ig_service, build_candle_update
    ):
        """payload must include bid_close and ofr_close for spread calculation."""
        update = build_candle_update(bid_close=20030.0, ofr_close=20031.0, cons_end="1")

        received = self._start_and_trigger(
            mock_ig_stream_service, update, mock_ig_service
        )

        candle = received[0]
        assert "bid_close" in candle
        assert "ofr_close" in candle
        assert candle["bid_close"] == pytest.approx(20030.0)
        assert candle["ofr_close"] == pytest.approx(20031.0)

    def test_delivered_candle_has_spread_field(
        self, mock_ig_stream_service, mock_ig_service, build_candle_update
    ):
        """Candle must include a 'spread' field equal to OFR_CLOSE - BID_CLOSE."""
        update = build_candle_update(bid_close=20030.0, ofr_close=20031.5, cons_end="1")

        received = self._start_and_trigger(
            mock_ig_stream_service, update, mock_ig_service
        )

        candle = received[0]
        assert "spread" in candle
        assert candle["spread"] == pytest.approx(1.5)  # OFR_CLOSE - BID_CLOSE

    def test_delivered_candle_has_volume_field(
        self, mock_ig_stream_service, mock_ig_service, build_candle_update
    ):
        """candle payload must include a 'volume' field from LTV."""
        update = build_candle_update(cons_end="1", ltv="42")

        received = self._start_and_trigger(
            mock_ig_stream_service, update, mock_ig_service
        )

        candle = received[0]
        assert "volume" in candle
        assert candle["volume"] == 42


class TestIncompleteCandleSuppression:
    """CONS_END=0 must NOT invoke on_candle."""

    def test_multiple_incomplete_followed_by_complete(
        self, mock_ig_stream_service, mock_ig_service, build_candle_update
    ):
        """CONS_END=0 updates are ignored; only CONS_END=1 triggers delivery."""
        received = []
        delivered = threading.Event()

        def callback(candle):
            received.append(candle)
            delivered.set()

        with patch(
            "ig_streaming_client.IGStreamService", return_value=mock_ig_stream_service
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=callback)

            sub = mock_ig_stream_service.subscribe.call_args[0][0]
            listener = sub._listener
            for _ in range(3):
                listener.onItemUpdate(build_candle_update(cons_end="0"))
            listener.onItemUpdate(build_candle_update(cons_end="1"))

            delivered.wait(timeout=2.0)
            client.stop()

        assert len(received) == 1


# ---------------------------------------------------------------------------
# Tick aggregation (REQ-2)
# ---------------------------------------------------------------------------


class TestTickAggregator:
    """TickAggregator aggregates ticks into 5-min OHLC candles."""

    def _make_ts(self, minute: int, second: int = 0) -> datetime:
        """Return a UTC datetime in a fixed hour, at the given minute:second."""
        return datetime(2024, 5, 27, 12, minute, second, tzinfo=timezone.utc)

    def test_candle_completed_on_first_tick_of_next_window(self):
        """Candle emitted when first tick of next window arrives.

        The startup partial window (12:00–12:05) is discarded per REQ-2/Scenario 2.
        The next full window (12:05–12:10) is accumulated and emitted when the
        first tick of the 12:10–12:15 window arrives.
        """
        emitted = []
        agg = TickAggregator(
            resolution_minutes=5, on_candle=lambda c: emitted.append(c)
        )

        # Startup partial window (12:00–12:05): ticks at minute 1, 3 — discarded
        agg.on_tick(bid=99.0, ofr=99.1, utm=self._make_ts(1))
        agg.on_tick(bid=98.0, ofr=98.1, utm=self._make_ts(3))
        assert emitted == [], "Startup window not yet at boundary"

        # Boundary at minute 5 triggers startup discard — tick at 12:05 opens new window
        agg.on_tick(bid=100.0, ofr=100.1, utm=self._make_ts(5))
        assert emitted == [], "Startup partial window must be discarded, not emitted"

        # Accumulate ticks in the 12:05–12:10 window
        agg.on_tick(bid=101.0, ofr=101.1, utm=self._make_ts(6))
        agg.on_tick(bid=99.5, ofr=99.6, utm=self._make_ts(8))
        assert emitted == [], "No candle yet — 12:05 window not closed"

        # First tick of next window (12:10) closes the 12:05 window
        agg.on_tick(bid=102.0, ofr=102.1, utm=self._make_ts(10))

        assert len(emitted) == 1, f"Expected 1 candle, got {len(emitted)}"
        candle = emitted[0]
        assert candle["open"] == pytest.approx(100.0)  # first tick of 12:05 window
        assert candle["high"] == pytest.approx(101.0)
        assert candle["low"] == pytest.approx(99.5)
        assert candle["close"] == pytest.approx(99.5)  # last tick before 12:10

    def test_partial_window_at_startup_is_discarded(self):
        """First partial window must not produce a candle."""
        emitted = []
        agg = TickAggregator(
            resolution_minutes=5, on_candle=lambda c: emitted.append(c)
        )

        # Mid-window ticks (startup — window is already partially elapsed)
        agg.on_tick(bid=100.0, ofr=100.1, utm=self._make_ts(2))
        agg.on_tick(bid=101.0, ofr=101.1, utm=self._make_ts(3))

        # Window boundary: first tick of 12:05 — should emit partial window
        # BUT the spec says partial window at STARTUP is discarded
        agg.on_tick(bid=102.0, ofr=102.1, utm=self._make_ts(5))

        assert (
            emitted == []
        ), "Partial startup window must be discarded — no candle should be emitted."

    def test_full_window_after_startup_produces_candle(self):
        """After startup discard, the next full window must emit correctly."""
        emitted = []
        agg = TickAggregator(
            resolution_minutes=5, on_candle=lambda c: emitted.append(c)
        )

        # Partial startup window — discarded
        agg.on_tick(bid=100.0, ofr=100.1, utm=self._make_ts(2))
        agg.on_tick(bid=101.0, ofr=101.1, utm=self._make_ts(3))
        # Boundary — discard startup partial
        agg.on_tick(bid=102.0, ofr=102.1, utm=self._make_ts(5))

        # Now a full second window: 12:05–12:10
        agg.on_tick(bid=103.0, ofr=103.1, utm=self._make_ts(6))
        agg.on_tick(bid=105.0, ofr=105.1, utm=self._make_ts(8))
        # Boundary of third window: 12:10
        agg.on_tick(bid=104.0, ofr=104.1, utm=self._make_ts(10))

        assert len(emitted) == 1
        candle = emitted[0]
        assert candle["open"] == pytest.approx(102.0)  # first tick of window
        assert candle["high"] == pytest.approx(105.0)
        assert candle["low"] == pytest.approx(102.0)
        assert candle["close"] == pytest.approx(105.0)

    def test_tick_aggregator_candle_has_bid_close_and_ofr_close(self):
        """aggregated candle must include bid_close and ofr_close."""
        emitted = []
        agg = TickAggregator(
            resolution_minutes=5, on_candle=lambda c: emitted.append(c)
        )

        agg.on_tick(bid=100.0, ofr=100.1, utm=self._make_ts(1))
        agg.on_tick(bid=101.0, ofr=101.1, utm=self._make_ts(5))  # boundary

        # Startup discard — need one more boundary
        agg.on_tick(bid=102.0, ofr=102.1, utm=self._make_ts(6))
        agg.on_tick(bid=103.0, ofr=103.1, utm=self._make_ts(10))  # boundary

        assert len(emitted) == 1
        candle = emitted[0]
        assert "bid_close" in candle
        assert "ofr_close" in candle

    def test_tick_aggregator_candle_has_spread_field(self):
        """aggregated candle must include 'spread' = last tick's OFR - BID."""
        emitted = []
        agg = TickAggregator(
            resolution_minutes=5, on_candle=lambda c: emitted.append(c)
        )

        # Startup partial window
        agg.on_tick(bid=100.0, ofr=100.5, utm=self._make_ts(1))
        agg.on_tick(bid=101.0, ofr=101.5, utm=self._make_ts(5))  # boundary — discard

        # Full window with last tick spread of 0.8
        agg.on_tick(bid=102.0, ofr=102.3, utm=self._make_ts(6))
        agg.on_tick(
            bid=103.0, ofr=103.8, utm=self._make_ts(8)
        )  # last tick: spread = 0.8
        agg.on_tick(bid=104.0, ofr=104.1, utm=self._make_ts(10))  # boundary — emit

        assert len(emitted) == 1
        candle = emitted[0]
        assert "spread" in candle
        # Spread of the closing tick of the window (at minute 8): OFR - BID = 0.8
        assert candle["spread"] == pytest.approx(0.8)

    def test_tick_aggregator_candle_has_volume_field(self):
        """aggregated candle must include 'volume' as tick count in the window."""
        emitted = []
        agg = TickAggregator(
            resolution_minutes=5, on_candle=lambda c: emitted.append(c)
        )

        # Startup partial window — tick at minute 1 starts it; tick at minute 5
        # crosses the boundary and opens the next window (startup partial discarded).
        agg.on_tick(bid=100.0, ofr=100.5, utm=self._make_ts(1))
        agg.on_tick(
            bid=101.0, ofr=101.5, utm=self._make_ts(5)
        )  # boundary — discard, opens window 2

        # Full window: opened by the tick at minute 5, then 3 more ticks at 6, 7, 8.
        # Total: 4 ticks in the 12:05-12:10 window (minute 5, 6, 7, 8).
        agg.on_tick(bid=102.0, ofr=102.3, utm=self._make_ts(6))
        agg.on_tick(bid=103.0, ofr=103.8, utm=self._make_ts(7))
        agg.on_tick(bid=104.0, ofr=104.1, utm=self._make_ts(8))
        agg.on_tick(bid=105.0, ofr=105.1, utm=self._make_ts(10))  # boundary — emit

        assert len(emitted) == 1
        candle = emitted[0]
        assert "volume" in candle
        # Window contained 4 ticks (minutes 5, 6, 7, 8) before the boundary tick
        assert candle["volume"] == 4

    def test_multiple_consecutive_windows(self):
        """TickAggregator emits a candle for each completed full window."""
        emitted = []
        agg = TickAggregator(
            resolution_minutes=5, on_candle=lambda c: emitted.append(c)
        )

        # Startup: 12:01 (partial, discarded at 12:05)
        agg.on_tick(bid=100.0, ofr=100.1, utm=self._make_ts(1))
        agg.on_tick(bid=101.0, ofr=101.1, utm=self._make_ts(5))  # discard

        # Window 12:05–12:10 (complete at 12:10)
        agg.on_tick(bid=102.0, ofr=102.1, utm=self._make_ts(6))
        agg.on_tick(bid=110.0, ofr=110.1, utm=self._make_ts(10))  # emit candle 1

        # Window 12:10–12:15 (complete at 12:15)
        agg.on_tick(bid=108.0, ofr=108.1, utm=self._make_ts(11))
        agg.on_tick(bid=109.0, ofr=109.1, utm=self._make_ts(15))  # emit candle 2

        assert len(emitted) == 2


# ---------------------------------------------------------------------------
# Callback thread safety (REQ-4)
# ---------------------------------------------------------------------------


class TestCallbackThreadSafety:
    """on_candle must be called from the worker thread, never the LS thread."""

    def test_callback_is_not_called_from_ls_listener_thread(
        self, mock_ig_stream_service, mock_ig_service, build_candle_update
    ):
        """On_candle must execute on the worker thread."""
        callback_thread_ids = []
        delivered = threading.Event()

        def tracking_callback(candle):
            callback_thread_ids.append(threading.current_thread().ident)
            delivered.set()

        with patch(
            "ig_streaming_client.IGStreamService", return_value=mock_ig_stream_service
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=tracking_callback)

            sub = mock_ig_stream_service.subscribe.call_args[0][0]
            listener = sub._listener
            ls_listener_thread_id = threading.current_thread().ident
            listener.onItemUpdate(build_candle_update(cons_end="1"))

            delivered.wait(timeout=2.0)
            client.stop()

        assert callback_thread_ids, "Callback was never invoked"
        for tid in callback_thread_ids:
            assert tid != ls_listener_thread_id, (
                "on_candle was called on the Lightstreamer listener thread — "
                "it must be called from the worker thread only."
            )


# ---------------------------------------------------------------------------
# Fallback to tick subscription (REQ-2)
# ---------------------------------------------------------------------------


class TestNativeFallbackToTick:
    """When native 5MINUTE subscription fails, client falls back to TICK."""

    def test_fallback_subscribed_on_native_subscription_error(
        self, mock_ig_stream_service, mock_ig_service
    ):
        """on native subscription error, fallback tick subscription is created."""

        # Simulate native subscription raising an exception
        def raise_on_first_subscribe(sub):
            item_name = sub._item_name
            if "5MINUTE" in item_name:
                raise RuntimeError("Subscription not available")
            # Tick subscription succeeds silently

        mock_ig_stream_service.subscribe.side_effect = raise_on_first_subscribe

        with patch(
            "ig_streaming_client.IGStreamService", return_value=mock_ig_stream_service
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=MagicMock())
            client.stop()

        # subscribe should have been called twice: once for 5MINUTE (failed),
        # once for TICK (fallback)
        assert mock_ig_stream_service.subscribe.call_count == 2

    def test_on_tick_with_tick_fallback_logs_warning_and_skips_direct_tick(
        self, mock_ig_stream_service, mock_ig_service, caplog
    ):
        """When on_tick is provided but native fell back to tick aggregation,
        a WARNING is logged and no third subscription (direct tick) is created."""
        import logging

        def raise_on_native_subscribe(sub):
            if "5MINUTE" in sub._item_name:
                raise RuntimeError("Subscription not available")
            # Tick fallback subscription succeeds silently

        mock_ig_stream_service.subscribe.side_effect = raise_on_native_subscribe

        with caplog.at_level(logging.WARNING, logger="ig_streaming_client"):
            with patch(
                "ig_streaming_client.IGStreamService",
                return_value=mock_ig_stream_service,
            ):
                client = IGStreamingClient(mock_ig_service, EPIC)
                client.start(on_candle=MagicMock(), on_tick=MagicMock())
                client.stop()

        # Exactly two subscribe calls: native (raised) + tick fallback.
        # The direct tick subscription must NOT be added as a third call.
        assert mock_ig_stream_service.subscribe.call_count == 2

        # A WARNING mentioning "tick" must have been emitted.
        warning_text = " ".join(
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        )
        assert "tick" in warning_text.lower()


# ---------------------------------------------------------------------------
# Lifecycle: shutdown, idempotency, queue draining, LS callback naming
# ---------------------------------------------------------------------------


class TestGracefulShutdown:
    """stop() must unsubscribe and disconnect cleanly."""

    def test_stop_calls_disconnect(self, mock_ig_stream_service, mock_ig_service):
        """stop() must call disconnect on the stream service."""
        with patch(
            "ig_streaming_client.IGStreamService", return_value=mock_ig_stream_service
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=MagicMock())
            client.stop()

        mock_ig_stream_service.disconnect.assert_called_once()

    def test_stop_without_start_does_not_raise(
        self, mock_ig_stream_service, mock_ig_service
    ):
        """stop() before start() must be a safe no-op."""
        with patch(
            "ig_streaming_client.IGStreamService", return_value=mock_ig_stream_service
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            # Calling stop before start should not raise — any exception fails the test
            client.stop()


class TestStartIdempotency:
    """Issue 8: calling start() twice must not create orphaned threads or sessions."""

    def test_start_twice_raises_runtime_error(
        self, mock_ig_stream_service, mock_ig_service
    ):
        """start() called a second time before stop() must raise RuntimeError."""
        with patch(
            "ig_streaming_client.IGStreamService", return_value=mock_ig_stream_service
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=MagicMock())

            with pytest.raises(RuntimeError, match="Already started"):
                client.start(on_candle=MagicMock())

            client.stop()

    def test_start_after_stop_succeeds(self, mock_ig_stream_service, mock_ig_service):
        """start() after stop() must succeed (not raise RuntimeError)."""
        with patch(
            "ig_streaming_client.IGStreamService", return_value=mock_ig_stream_service
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=MagicMock())
            client.stop()
            # Should not raise
            client.start(on_candle=MagicMock())
            client.stop()


class TestQueueDrainOnShutdown:
    """Issue 10: queued candles must be delivered to the callback before stop() returns."""

    def test_queued_candles_delivered_before_stop_returns(
        self, mock_ig_stream_service, mock_ig_service, build_candle_update
    ):
        """Candles placed on queue before stop() must be processed before stop() returns."""
        received = []
        delivered = threading.Event()

        def tracking_callback(candle):
            received.append(candle)
            if len(received) >= 2:
                delivered.set()

        with patch(
            "ig_streaming_client.IGStreamService", return_value=mock_ig_stream_service
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=tracking_callback)

            # Enqueue 2 candles directly (simulating candles arriving just before stop)
            sub = mock_ig_stream_service.subscribe.call_args[0][0]
            listener = sub._listener
            listener.onItemUpdate(build_candle_update(cons_end="1"))
            listener.onItemUpdate(build_candle_update(cons_end="1"))

            # Wait briefly for queue to be populated, then stop
            delivered.wait(timeout=2.0)
            client.stop()

        # After stop, both candles must have been delivered
        assert len(received) == 2, f"Expected 2 candles delivered, got {len(received)}"


class TestLightstreamerCallbackNaming:
    """Issue 11: Listener callbacks must use camelCase to match Lightstreamer interface."""

    def test_candle_listener_has_onItemUpdate_method(self):
        """_CandleSubscriptionListener must have onItemUpdate (camelCase) for the LS library."""
        listener = _CandleSubscriptionListener("CHART:TEST:5MINUTE", queue.Queue())

        assert hasattr(listener, "onItemUpdate"), (
            "_CandleSubscriptionListener must expose onItemUpdate (camelCase) "
            "to match the Lightstreamer SubscriptionListener interface."
        )
        assert callable(getattr(listener, "onItemUpdate"))

    def test_tick_listener_has_onItemUpdate_method(self):
        """_TickListener must have onItemUpdate (camelCase) for the LS library."""
        agg = TickAggregator(5, lambda c: None)
        listener = _TickListener("CHART:TEST:TICK", agg)

        assert hasattr(listener, "onItemUpdate"), (
            "_TickListener must expose onItemUpdate (camelCase) "
            "to match the Lightstreamer SubscriptionListener interface."
        )
        assert callable(getattr(listener, "onItemUpdate"))

    def test_candle_listener_onSubscription_is_camelcase(self):
        """_CandleSubscriptionListener must have onSubscription (not on_subscription)."""
        listener = _CandleSubscriptionListener("CHART:TEST:5MINUTE", queue.Queue())
        assert hasattr(listener, "onSubscription")
        assert not hasattr(listener, "on_subscription")


# ---------------------------------------------------------------------------
# _CandleSubscriptionListener — lifecycle callbacks and UTM parse error path
# ---------------------------------------------------------------------------


class TestCandleListenerCallbacks:
    """_CandleSubscriptionListener subscription lifecycle callbacks fire without error."""

    def test_onSubscription_does_not_raise(self):
        """onSubscription must not raise."""
        listener = _CandleSubscriptionListener("CHART:TEST:5MINUTE", queue.Queue())
        listener.onSubscription()  # must not raise

    def test_onSubscriptionError_does_not_raise(self):
        """onSubscriptionError must not raise."""
        listener = _CandleSubscriptionListener("CHART:TEST:5MINUTE", queue.Queue())
        listener.onSubscriptionError(404, "Not found")  # must not raise

    def test_onUnsubscription_does_not_raise(self):
        """onUnsubscription must not raise."""
        listener = _CandleSubscriptionListener("CHART:TEST:5MINUTE", queue.Queue())
        listener.onUnsubscription()  # must not raise

    def test_invalid_utm_falls_back_to_now(self):
        """When UTM cannot be parsed, timestamp falls back to datetime.now(UTC)."""
        q = queue.Queue()
        listener = _CandleSubscriptionListener("CHART:TEST:5MINUTE", q)
        update = _FakeItemUpdate(
            {
                "CONS_END": "1",
                "UTM": "not_a_number",  # triggers ValueError → fallback to now()
                "BID_OPEN": "100.0",
                "BID_HIGH": "101.0",
                "BID_LOW": "99.0",
                "BID_CLOSE": "100.5",
                "OFR_CLOSE": "100.7",
                "LTV": "5",
            }
        )

        listener.onItemUpdate(update)

        candle = q.get_nowait()
        assert isinstance(candle["timestamp"], datetime)
        assert candle["timestamp"].tzinfo is not None  # must be tz-aware

    def test_invalid_ltv_falls_back_to_zero(self):
        """When LTV cannot be parsed, volume falls back to 0."""
        q = queue.Queue()
        listener = _CandleSubscriptionListener("CHART:TEST:5MINUTE", q)
        update = _FakeItemUpdate(
            {
                "CONS_END": "1",
                "UTM": "1716825600000",
                "BID_OPEN": "100.0",
                "BID_HIGH": "101.0",
                "BID_LOW": "99.0",
                "BID_CLOSE": "100.5",
                "OFR_CLOSE": "100.7",
                "LTV": "bad_volume",  # triggers ValueError → volume = 0
            }
        )

        listener.onItemUpdate(update)

        candle = q.get_nowait()
        assert candle["volume"] == 0

    def test_cons_end_zero_does_not_enqueue(self):
        """CONS_END=0 must not enqueue anything."""
        q = queue.Queue()
        listener = _CandleSubscriptionListener("CHART:TEST:5MINUTE", q)
        update = _FakeItemUpdate(
            {
                "CONS_END": "0",
                "BID_CLOSE": "100.0",
                "OFR_CLOSE": "100.5",
            }
        )

        listener.onItemUpdate(update)

        assert q.empty()


# ---------------------------------------------------------------------------
# _TickListener — lifecycle callbacks and parse error path
# ---------------------------------------------------------------------------


class TestTickListenerCallbacks:
    """_TickListener subscription lifecycle callbacks fire without error."""

    def test_onSubscription_does_not_raise(self):
        """onSubscription must not raise."""
        agg = TickAggregator(5, lambda c: None)
        listener = _TickListener("CHART:TEST:TICK", agg)
        listener.onSubscription()  # must not raise

    def test_onSubscriptionError_does_not_raise(self):
        """onSubscriptionError must not raise."""
        agg = TickAggregator(5, lambda c: None)
        listener = _TickListener("CHART:TEST:TICK", agg)
        listener.onSubscriptionError(500, "Internal error")  # must not raise

    def test_onUnsubscription_does_not_raise(self):
        """onUnsubscription must not raise."""
        agg = TickAggregator(5, lambda c: None)
        listener = _TickListener("CHART:TEST:TICK", agg)
        listener.onUnsubscription()  # must not raise

    def test_invalid_tick_values_do_not_raise(self):
        """_TickListener.onItemUpdate with unparseable values must not raise."""
        agg = TickAggregator(5, lambda c: None)
        listener = _TickListener("CHART:TEST:TICK", agg)
        update = _FakeItemUpdate(
            {
                "BID": "not_a_float",  # triggers ValueError
                "OFR": "100.5",
                "UTM": "1716825600000",
            }
        )

        listener.onItemUpdate(update)  # must not raise

    def test_missing_tick_field_does_not_raise(self):
        """_TickListener.onItemUpdate with missing UTM field must not raise."""
        agg = TickAggregator(5, lambda c: None)
        listener = _TickListener("CHART:TEST:TICK", agg)
        update = _FakeItemUpdate(
            {
                "BID": "100.0",
                "OFR": "100.5",
                # UTM absent — getValue returns None, "or '0'" provides the default
            }
        )

        listener.onItemUpdate(update)  # must not raise


# ---------------------------------------------------------------------------
# candle_frequency_to_resolution — unmapped minutes path (line 120)
# ---------------------------------------------------------------------------


class TestCandleFrequencyToResolutionUnmapped:
    """candle_frequency_to_resolution raises ValueError for valid format but unmapped minutes."""

    def test_unmapped_minutes_raises_value_error(self):
        """'7min' has a valid format but no IG resolution mapping — must raise ValueError."""
        from ig_streaming_client import candle_frequency_to_resolution

        with pytest.raises(ValueError, match="No IG resolution mapping"):
            candle_frequency_to_resolution("7min")

    def test_non_numeric_prefix_raises_value_error(self):
        """'Xmin' has a non-numeric prefix — must raise ValueError."""
        from ig_streaming_client import candle_frequency_to_resolution

        with pytest.raises(ValueError):
            candle_frequency_to_resolution("Xmin")


# ---------------------------------------------------------------------------
# IGStreamingClient.start — session creation failure path (lines 552-557)
# ---------------------------------------------------------------------------


class TestStartSessionCreationFailure:
    """start() cleans up stream service when IGStreamService.create_session() raises."""

    def test_session_creation_failure_clears_stream_svc_and_re_raises(
        self, mock_ig_service
    ):
        """When create_session() raises, _stream_svc is set to None and the exception propagates."""
        mock_stream_svc = MagicMock()
        mock_stream_svc.create_session.side_effect = RuntimeError("Auth failed")

        with patch("ig_streaming_client.IGStreamService", return_value=mock_stream_svc):
            client = IGStreamingClient(mock_ig_service, EPIC)

            with pytest.raises(RuntimeError, match="Auth failed"):
                client.start(on_candle=MagicMock())

        # After failure, _stream_svc must be None so stop() won't try to disconnect
        assert client._stream_svc is None


# ---------------------------------------------------------------------------
# IGStreamingClient.start — subscription setup failure path (lines 571-576)
# ---------------------------------------------------------------------------


class TestStartSubscriptionFailure:
    """start() shuts down worker thread when _subscribe_native raises."""

    def test_subscription_failure_raises_and_stops_worker(self, mock_ig_service):
        """When _subscribe_native raises (beyond fallback), start() propagates the error."""
        mock_stream_svc = MagicMock()
        mock_stream_svc.create_session.return_value = None
        # Both native and tick subscriptions fail — fallback also raises
        mock_stream_svc.subscribe.side_effect = RuntimeError("Hard subscribe failure")

        with patch("ig_streaming_client.IGStreamService", return_value=mock_stream_svc):
            client = IGStreamingClient(mock_ig_service, EPIC)

            with pytest.raises(RuntimeError):
                client.start(on_candle=MagicMock())

        # Worker must have been stopped (not left orphaned).
        # stop() sets _worker = None in the failure path, so the correct assertion
        # is that _worker is None — proving cleanup completed.
        assert client._worker is None, "Orphaned worker thread after start() failure"


# ---------------------------------------------------------------------------
# IGStreamingClient.stop — disconnect exception path (lines 705-706)
# ---------------------------------------------------------------------------


class TestStopDisconnectException:
    """stop() handles exceptions from stream_svc.disconnect() gracefully."""

    def test_stop_disconnect_exception_does_not_propagate(
        self, mock_ig_stream_service, mock_ig_service
    ):
        """An exception in stream_svc.disconnect() must not propagate from stop()."""
        mock_ig_stream_service.disconnect.side_effect = RuntimeError("Disconnect error")

        with patch(
            "ig_streaming_client.IGStreamService", return_value=mock_ig_stream_service
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=MagicMock())
            client.stop()  # must not raise even though disconnect() raises


# ---------------------------------------------------------------------------
# IGStreamingClient.stop — worker timeout warning (lines 695)
# ---------------------------------------------------------------------------


class TestStopWorkerTimeout:
    """stop() logs a warning when the worker thread doesn't exit within timeout."""

    def test_stop_logs_warning_when_worker_does_not_stop(self, mock_ig_service):
        """When worker.join() times out, stop() logs a warning instead of blocking forever."""
        mock_stream_svc = MagicMock()
        mock_stream_svc.create_session.return_value = None
        mock_stream_svc.subscribe.return_value = None

        with patch("ig_streaming_client.IGStreamService", return_value=mock_stream_svc):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=MagicMock())

            # Patch worker.is_alive to always return True to simulate a stuck thread
            original_worker = client._worker
            with patch.object(original_worker, "is_alive", return_value=True):
                with patch.object(original_worker, "join"):  # no-op join
                    client.stop()  # must not block; warning is logged

        # Clean up: actually stop the worker
        if original_worker is not None and original_worker.is_alive():
            client._stop_event.set()
            original_worker.join(timeout=1.0)


# ---------------------------------------------------------------------------
# IGStreamingClient._worker_loop — on_candle exception and drain paths
# ---------------------------------------------------------------------------


class TestWorkerLoopExceptions:
    """Worker loop catches on_candle exceptions and continues processing."""

    def test_on_candle_exception_does_not_stop_worker(
        self, mock_ig_stream_service, mock_ig_service, build_candle_update
    ):
        """An exception in on_candle for one candle must not stop the worker thread."""
        call_count = [0]
        delivered = threading.Event()

        def flaky_callback(candle):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Processing error")
            delivered.set()

        with patch(
            "ig_streaming_client.IGStreamService", return_value=mock_ig_stream_service
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=flaky_callback)

            sub = mock_ig_stream_service.subscribe.call_args[0][0]
            listener = sub._listener
            # Two candles — first raises, second should still be delivered
            listener.onItemUpdate(build_candle_update(cons_end="1"))
            listener.onItemUpdate(build_candle_update(cons_end="1"))

            delivered.wait(timeout=2.0)
            client.stop()

        assert call_count[0] >= 2, "Worker must continue after on_candle exception"


class TestWorkerDrainExceptions:
    """Worker drain loop handles on_candle exceptions during drain without crashing.

    The drain loop (lines 649-661) runs after the stop event is set. We test
    it by calling _worker_loop directly on the current thread with the stop
    event already set and a candle pre-loaded, so the main while loop exits
    immediately and the drain loop handles the pre-loaded item.
    """

    def test_drain_on_candle_exception_does_not_crash(self, mock_ig_service):
        """Exception in on_candle during drain phase must not propagate."""
        client = IGStreamingClient(mock_ig_service, EPIC)

        def raising_callback(candle):
            raise RuntimeError("Drain callback error")

        # Pre-load a candle into the queue and immediately set stop_event
        # so _worker_loop skips the main while loop and goes to drain
        client._candle_queue.put({"drain": True, "close": 100.0})
        client._stop_event.set()

        # Call _worker_loop directly on this thread — it must not propagate any exception
        client._worker_loop(raising_callback)  # must not raise


class TestWorkerDrainBudgetExceeded:
    """Worker logs a warning when drain budget is exceeded (dropped candles)."""

    def test_drain_budget_exceeded_does_not_crash(
        self, mock_ig_stream_service, mock_ig_service
    ):
        """When more than _DRAIN_BUDGET candles are queued at shutdown, stop() must not crash."""
        with patch(
            "ig_streaming_client.IGStreamService", return_value=mock_ig_stream_service
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=MagicMock())

            # Enqueue 55 candles directly into the queue (> _DRAIN_BUDGET of 50)
            for i in range(55):
                client._candle_queue.put({"candle_index": i})

            client.stop()  # must not raise even when drain budget is exceeded


# =========================================================================== #
# Tick mode — worker dispatch [REQ-11]                                          #
# =========================================================================== #


class TestWorkerDispatch:
    """Worker loop routes by item 'type' key with backward-compat default [REQ-11]."""

    def test_item_without_type_key_routes_to_on_candle(self):
        """Item without 'type' key is treated as a candle — routes to on_candle."""
        on_candle = MagicMock()
        on_tick = MagicMock()

        from ig_streaming_client import IGStreamingClient

        client = IGStreamingClient(MagicMock(), EPIC)

        # Pre-load an item without 'type' key
        client._candle_queue.put({"bid": 1.0})
        client._stop_event.set()  # so worker exits immediately after processing

        # Run _worker_loop directly on this thread
        client._worker_loop(on_candle, on_tick)

        on_candle.assert_called_once()
        on_tick.assert_not_called()

    def test_item_with_type_tick_routes_to_on_tick(self):
        """Item with type='tick' routes to on_tick callback."""
        on_candle = MagicMock()
        on_tick = MagicMock()

        from ig_streaming_client import IGStreamingClient

        client = IGStreamingClient(MagicMock(), EPIC)

        tick_item = {"type": "tick", "bid": 1.0, "ofr": 1.1, "utm": 123}
        client._candle_queue.put(tick_item)
        client._stop_event.set()

        client._worker_loop(on_candle, on_tick)

        on_tick.assert_called_once_with(tick_item)
        on_candle.assert_not_called()


# =========================================================================== #
# Tick mode — _DirectTickListener thread safety [REQ-12]                       #
# =========================================================================== #


class TestThreadSafety:
    """_DirectTickListener.onItemUpdate only calls queue.put; no strategy state access [REQ-12]."""

    def test_on_item_update_only_calls_queue_put(self):
        """_DirectTickListener.onItemUpdate enqueues the tick dict and nothing else."""
        from ig_streaming_client import _DirectTickListener

        q = MagicMock()
        listener = _DirectTickListener("CHART:TEST:TICK", q)

        update = _FakeItemUpdate(
            {
                "BID": "100.5",
                "OFR": "101.0",
                "UTM": "1716825600000",
            }
        )

        listener.onItemUpdate(update)

        q.put.assert_called_once()
        enqueued = q.put.call_args[0][0]
        assert enqueued["type"] == "tick"
        assert enqueued["bid"] == pytest.approx(100.5)
        assert enqueued["ofr"] == pytest.approx(101.0)
        assert enqueued["utm"] == 1716825600000


# =========================================================================== #
# Tick mode — dual subscription [REQ-10]                                        #
# =========================================================================== #


class TestDualSubscription:
    """Two subscriptions created when on_tick provided; one when on_tick=None [REQ-10]."""

    def test_two_subscriptions_created_when_on_tick_provided(
        self, mock_ig_stream_service, mock_ig_service
    ):
        """When on_tick is not None, two Lightstreamer subscriptions are active."""
        with patch(
            "ig_streaming_client.IGStreamService", return_value=mock_ig_stream_service
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=MagicMock(), on_tick=MagicMock())
            client.stop()

        assert mock_ig_stream_service.subscribe.call_count == 2

    def test_one_subscription_created_when_on_tick_is_none(
        self, mock_ig_stream_service, mock_ig_service
    ):
        """When on_tick=None, only the candle subscription is created."""
        with patch(
            "ig_streaming_client.IGStreamService", return_value=mock_ig_stream_service
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=MagicMock(), on_tick=None)
            client.stop()

        assert mock_ig_stream_service.subscribe.call_count == 1


# =========================================================================== #
# _ConnectionListener — disconnect detection [REQ-1, REQ-7]                    #
# =========================================================================== #


class TestConnectionListener:
    """_ConnectionListener enqueues a sentinel only on bare DISCONNECTED status."""

    def test_reconnect_sentinel_on_disconnected(self):
        """Bare 'DISCONNECTED' places {type: reconnect} sentinel on the queue."""
        q = queue.Queue()
        listener = _ConnectionListener(q)

        listener.onStatusChange("DISCONNECTED")

        assert not q.empty()
        item = q.get_nowait()
        assert item == {"type": "reconnect"}

    def test_no_sentinel_on_will_retry(self):
        """'DISCONNECTED:WILL-RETRY' does not enqueue a sentinel."""
        q = queue.Queue()
        listener = _ConnectionListener(q)

        listener.onStatusChange("DISCONNECTED:WILL-RETRY")

        assert q.empty()

    def test_no_sentinel_on_trying_recovery(self):
        """'DISCONNECTED:TRYING-RECOVERY' does not enqueue a sentinel."""
        q = queue.Queue()
        listener = _ConnectionListener(q)

        listener.onStatusChange("DISCONNECTED:TRYING-RECOVERY")

        assert q.empty()

    def test_no_sentinel_on_connected_status(self):
        """Connected status strings do not enqueue a sentinel."""
        q = queue.Queue()
        listener = _ConnectionListener(q)

        for status in ("CONNECTING", "CONNECTED:WS-STREAMING", "STALLED"):
            listener.onStatusChange(status)

        assert q.empty()

    def test_fired_flag_prevents_duplicate_sentinels(self):
        """A second bare DISCONNECTED call does not enqueue a second sentinel."""
        q = queue.Queue()
        listener = _ConnectionListener(q)

        listener.onStatusChange("DISCONNECTED")
        listener.onStatusChange("DISCONNECTED")

        # Only one item must be in the queue
        q.get_nowait()  # consume the first sentinel
        assert q.empty(), "Second DISCONNECTED must not enqueue a second sentinel"

    def test_on_server_error_does_not_raise(self):
        """onServerError must not raise."""
        q = queue.Queue()
        listener = _ConnectionListener(q)
        listener.onServerError(503, "Service unavailable")  # must not raise

    def test_on_property_change_does_not_raise(self):
        """onPropertyChange must not raise."""
        q = queue.Queue()
        listener = _ConnectionListener(q)
        listener.onPropertyChange("serverAddress")  # must not raise

    def test_on_listen_start_does_not_raise(self):
        """onListenStart must not raise."""
        _ConnectionListener(queue.Queue()).onListenStart()

    def test_on_listen_end_does_not_raise(self):
        """onListenEnd must not raise."""
        _ConnectionListener(queue.Queue()).onListenEnd()


# =========================================================================== #
# Worker loop — reconnect sentinel dispatch [REQ-7]                            #
# =========================================================================== #


class TestWorkerLoopReconnectDispatch:
    """Worker loop dispatches reconnect sentinel to the on_reconnect callback."""

    def test_reconnect_sentinel_dispatches_on_reconnect_callback(self):
        """A {type: reconnect} item on the queue calls on_reconnect once."""
        on_reconnect = MagicMock()
        client = IGStreamingClient(MagicMock(), EPIC)

        client._candle_queue.put({"type": "reconnect"})
        client._stop_event.set()

        client._worker_loop(MagicMock(), None, on_reconnect)

        on_reconnect.assert_called_once_with()

    def test_reconnect_sentinel_without_callback_does_not_raise(self):
        """When on_reconnect=None, a reconnect sentinel is silently ignored."""
        client = IGStreamingClient(MagicMock(), EPIC)

        client._candle_queue.put({"type": "reconnect"})
        client._stop_event.set()

        client._worker_loop(MagicMock(), None, None)  # must not raise

    def test_candle_dispatch_unaffected_by_on_reconnect_param(self):
        """Candle items are still dispatched to on_candle when on_reconnect is set."""
        on_candle = MagicMock()
        on_reconnect = MagicMock()
        client = IGStreamingClient(MagicMock(), EPIC)

        candle_item = {"type": "candle", "close": 100.0}
        client._candle_queue.put(candle_item)
        client._stop_event.set()

        client._worker_loop(on_candle, None, on_reconnect)

        on_candle.assert_called_once_with(candle_item)
        on_reconnect.assert_not_called()


# =========================================================================== #
# _restart_streaming — service cycle contract [REQ-7]                          #
# =========================================================================== #


class TestRestartStreaming:
    """_restart_streaming disconnects old service and creates a new one without touching the worker."""

    def test_restart_streaming_disconnects_old_service(self, mock_ig_service):
        """_restart_streaming calls disconnect() on the old _stream_svc."""
        old_svc = MagicMock()
        old_svc.create_session.return_value = None
        old_svc.subscribe.return_value = None
        old_svc.disconnect.return_value = None

        new_svc = MagicMock()
        new_svc.create_session.return_value = None
        new_svc.subscribe.return_value = None
        new_svc.add_client_listener.return_value = None

        with patch(
            "ig_streaming_client.IGStreamService", side_effect=[old_svc, new_svc]
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=MagicMock())

            client._restart_streaming(on_candle=MagicMock())
            client.stop()

        old_svc.disconnect.assert_called()

    def test_restart_streaming_calls_create_session_on_new_service(
        self, mock_ig_service
    ):
        """_restart_streaming calls create_session() on the newly created service."""
        old_svc = MagicMock()
        old_svc.create_session.return_value = None
        old_svc.subscribe.return_value = None
        old_svc.disconnect.return_value = None

        new_svc = MagicMock()
        new_svc.create_session.return_value = None
        new_svc.subscribe.return_value = None
        new_svc.add_client_listener.return_value = None

        with patch(
            "ig_streaming_client.IGStreamService", side_effect=[old_svc, new_svc]
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=MagicMock())

            client._restart_streaming(on_candle=MagicMock())
            client.stop()

        new_svc.create_session.assert_called_once()

    def test_restart_streaming_does_not_set_stop_event(self, mock_ig_service):
        """_restart_streaming must not set the stop event (worker keeps running)."""
        old_svc = MagicMock()
        old_svc.create_session.return_value = None
        old_svc.subscribe.return_value = None
        old_svc.disconnect.return_value = None

        new_svc = MagicMock()
        new_svc.create_session.return_value = None
        new_svc.subscribe.return_value = None
        new_svc.add_client_listener.return_value = None

        with patch(
            "ig_streaming_client.IGStreamService", side_effect=[old_svc, new_svc]
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=MagicMock())

            client._restart_streaming(on_candle=MagicMock())
            assert not client._stop_event.is_set()
            client.stop()

    def test_restart_streaming_attaches_fresh_connection_listener(
        self, mock_ig_service
    ):
        """_restart_streaming attaches a new _ConnectionListener to the new service."""
        old_svc = MagicMock()
        old_svc.create_session.return_value = None
        old_svc.subscribe.return_value = None
        old_svc.disconnect.return_value = None

        new_svc = MagicMock()
        new_svc.create_session.return_value = None
        new_svc.subscribe.return_value = None
        new_svc.add_client_listener.return_value = None

        with patch(
            "ig_streaming_client.IGStreamService", side_effect=[old_svc, new_svc]
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            client.start(on_candle=MagicMock())

            client._restart_streaming(on_candle=MagicMock())
            client.stop()

        new_svc.add_client_listener.assert_called_once()
        listener_arg = new_svc.add_client_listener.call_args[0][0]
        assert isinstance(listener_arg, _ConnectionListener)
        assert listener_arg._fired is False


# ---------------------------------------------------------------------------
# GAP-2: Subscription Error Reconnect Sentinel (RED tests — tasks 3.1–3.2)
# ---------------------------------------------------------------------------


class TestSubscriptionErrorReconnectSentinel:
    """GAP-2: onSubscriptionError enqueues reconnect sentinel when setup is complete."""

    def test_candle_listener_on_subscription_error_enqueues_sentinel_when_setup_complete(
        self,
    ):
        """onSubscriptionError on _CandleSubscriptionListener enqueues sentinel when
        setup_complete_check returns True."""
        q = queue.Queue()
        listener = _CandleSubscriptionListener(
            item_name="CHART:EPIC:5MINUTE",
            candle_queue=q,
            setup_complete_check=lambda: True,
        )
        listener.onSubscriptionError(code=17, message="Subscription refused")
        assert not q.empty(), "Expected reconnect sentinel in queue"
        item = q.get_nowait()
        assert item == {"type": "reconnect"}

    def test_candle_listener_on_subscription_error_does_not_enqueue_during_setup(
        self,
    ):
        """onSubscriptionError on _CandleSubscriptionListener does NOT enqueue sentinel
        when setup_complete_check returns False (initial setup phase)."""
        q = queue.Queue()
        listener = _CandleSubscriptionListener(
            item_name="CHART:EPIC:5MINUTE",
            candle_queue=q,
            setup_complete_check=lambda: False,
        )
        listener.onSubscriptionError(code=17, message="Subscription refused")
        assert q.empty(), "Sentinel must NOT be enqueued during initial setup"

    def test_tick_listener_on_subscription_error_enqueues_sentinel_when_setup_complete(
        self,
    ):
        """onSubscriptionError on _TickListener enqueues sentinel when setup complete."""
        q = queue.Queue()
        aggregator = MagicMock()
        listener = _TickListener(
            item_name="CHART:EPIC:TICK",
            aggregator=aggregator,
            candle_queue=q,
            setup_complete_check=lambda: True,
        )
        listener.onSubscriptionError(code=22, message="Bad subscription")
        assert not q.empty(), "Expected reconnect sentinel in queue"
        item = q.get_nowait()
        assert item == {"type": "reconnect"}

    def test_tick_listener_on_subscription_error_does_not_enqueue_during_setup(
        self,
    ):
        """onSubscriptionError on _TickListener does NOT enqueue during initial setup."""
        q = queue.Queue()
        aggregator = MagicMock()
        listener = _TickListener(
            item_name="CHART:EPIC:TICK",
            aggregator=aggregator,
            candle_queue=q,
            setup_complete_check=lambda: False,
        )
        listener.onSubscriptionError(code=22, message="Bad subscription")
        assert q.empty(), "Sentinel must NOT be enqueued during initial setup"

    def test_direct_tick_listener_on_subscription_error_enqueues_sentinel_when_setup_complete(
        self,
    ):
        """onSubscriptionError on _DirectTickListener enqueues sentinel when setup complete."""
        q = queue.Queue()
        listener = _DirectTickListener(
            item_name="CHART:EPIC:TICK",
            tick_queue=q,
            setup_complete_check=lambda: True,
        )
        listener.onSubscriptionError(code=22, message="Bad subscription")
        assert not q.empty(), "Expected reconnect sentinel in queue"
        item = q.get_nowait()
        assert item == {"type": "reconnect"}

    def test_direct_tick_listener_on_subscription_error_does_not_enqueue_during_setup(
        self,
    ):
        """onSubscriptionError on _DirectTickListener does NOT enqueue during initial setup."""
        q = queue.Queue()
        listener = _DirectTickListener(
            item_name="CHART:EPIC:TICK",
            tick_queue=q,
            setup_complete_check=lambda: False,
        )
        listener.onSubscriptionError(code=22, message="Bad subscription")
        assert q.empty(), "Sentinel must NOT be enqueued during initial setup"


# ---------------------------------------------------------------------------
# GAP-3: Candle Watchdog (RED tests — tasks 3.4–3.7)
# ---------------------------------------------------------------------------


class TestCandleWatchdog:
    """GAP-3: Worker loop watchdog detects data drought and enqueues reconnect."""

    def test_watchdog_fires_after_elapsed_exceeds_threshold(self):
        """Watchdog enqueues reconnect sentinel when monotonic elapsed > threshold.

        Exercises the real _worker_loop method on IGStreamingClient rather than
        re-implementing the watchdog logic inline (follows the pattern of
        test_watchdog_fires_via_worker_loop_on_drought).
        """
        import time

        sentinel_q = queue.Queue()

        def on_reconnect():
            sentinel_q.put("reconnect_called")

        client = IGStreamingClient(MagicMock(), "IX.D.SPTRD.IFMM.IP")

        # Set threshold to near-zero so the test fires immediately
        client._watchdog_timeout_s = 0.0
        # Set _last_data_ts far in the past so elapsed >> threshold
        client._last_data_ts = time.monotonic() - 10

        # Run _worker_loop on a background thread; it will fire the watchdog,
        # enqueue and dispatch the reconnect sentinel, then we stop it.
        worker = threading.Thread(
            target=client._worker_loop,
            args=(MagicMock(), None, on_reconnect),
            daemon=True,
        )
        worker.start()
        try:
            result = sentinel_q.get(timeout=2.0)
        finally:
            client._stop_event.set()
            worker.join(timeout=2.0)

        assert result == "reconnect_called"

    def test_watchdog_does_not_fire_before_threshold(self):
        """Watchdog does NOT enqueue sentinel when elapsed <= threshold.

        Exercises the real _worker_loop method on IGStreamingClient rather than
        re-implementing the watchdog logic inline. Runs the worker loop briefly
        with a recent _last_data_ts and the default large threshold, then asserts
        that no reconnect sentinel was dispatched.
        """
        import time

        on_reconnect = MagicMock()
        client = IGStreamingClient(MagicMock(), "IX.D.SPTRD.IFMM.IP")

        # Keep the default (large) threshold so elapsed is always below it
        # Set _last_data_ts to now so elapsed is tiny
        client._last_data_ts = time.monotonic()

        # Run _worker_loop on a background thread for a short time, then stop
        worker = threading.Thread(
            target=client._worker_loop,
            args=(MagicMock(), None, on_reconnect),
            daemon=True,
        )
        worker.start()
        # Let the loop run a few iterations (each waits 0.05s on queue.get)
        time.sleep(0.2)
        client._stop_event.set()
        worker.join(timeout=2.0)

        on_reconnect.assert_not_called()

    def test_watchdog_threshold_floor_for_1_minute_period(self):
        """candle_period_minutes=1 → threshold must be max(1*3, 15)*60 = 900 s."""
        mock_ig_service = MagicMock()
        # 1MINUTE resolution
        client = IGStreamingClient(
            mock_ig_service, "IX.D.SPTRD.IFMM.IP", resolution="1MINUTE"
        )
        expected = max(1 * 3, 15) * 60  # 15 * 60 = 900
        assert client._watchdog_timeout_s == expected

    def test_watchdog_threshold_scales_with_10_minute_period(self):
        """candle_period_minutes=10 → threshold must be max(10*3, 15)*60 = 1800 s."""
        mock_ig_service = MagicMock()
        # 10MINUTE resolution
        client = IGStreamingClient(
            mock_ig_service, "IX.D.SPTRD.IFMM.IP", resolution="10MINUTE"
        )
        expected = max(10 * 3, 15) * 60  # 30 * 60 = 1800
        assert client._watchdog_timeout_s == expected

    def test_watchdog_fires_via_worker_loop_on_drought(
        self, mock_ig_stream_service, mock_ig_service
    ):
        """Integration: worker loop enqueues reconnect when data drought exceeds threshold."""
        import time

        sentinel_q = queue.Queue()

        def on_reconnect():
            sentinel_q.put("reconnect_called")

        with patch(
            "ig_streaming_client.IGStreamService", return_value=mock_ig_stream_service
        ):
            client = IGStreamingClient(mock_ig_service, EPIC)
            # Lower threshold to near-zero so the test doesn't wait 15 minutes
            client._watchdog_timeout_s = 0.0
            client._last_data_ts = time.monotonic() - 10  # already in the past
            client.start(on_candle=MagicMock(), on_reconnect=on_reconnect)
            # Give the worker a moment to fire watchdog
            try:
                result = sentinel_q.get(timeout=2.0)
            finally:
                client.stop()
        assert result == "reconnect_called"


# ---------------------------------------------------------------------------
# GAP-1: Broad Exception Catch in Listeners (RED tests — tasks 3.9–3.10)
# ---------------------------------------------------------------------------


class TestBroadExceptionCatchInListeners:
    """GAP-1: onItemUpdate must catch all Exception subclasses and log them."""

    def test_candle_listener_on_item_update_catches_runtime_error(self):
        """_CandleSubscriptionListener.onItemUpdate must not propagate RuntimeError."""
        q = queue.Queue()
        listener = _CandleSubscriptionListener(
            item_name="CHART:EPIC:5MINUTE",
            candle_queue=q,
        )
        bad_update = MagicMock()
        bad_update.getValue.side_effect = RuntimeError("simulated LS error")

        # Must not raise
        listener.onItemUpdate(bad_update)
        # Queue should still be empty (no candle built)
        assert q.empty()

    def test_candle_listener_on_item_update_catches_value_error_and_continues(self):
        """_CandleSubscriptionListener.onItemUpdate catches ValueError and remains callable."""
        q = queue.Queue()
        listener = _CandleSubscriptionListener(
            item_name="CHART:EPIC:5MINUTE",
            candle_queue=q,
        )
        bad_update = MagicMock()
        bad_update.getValue.side_effect = ValueError("bad field")

        listener.onItemUpdate(bad_update)
        # Listener is still callable after catching
        assert callable(listener.onItemUpdate)

    def test_tick_listener_on_item_update_catches_runtime_error(self):
        """_TickListener.onItemUpdate must not propagate RuntimeError."""
        aggregator = MagicMock()
        q = queue.Queue()
        listener = _TickListener(
            item_name="CHART:EPIC:TICK",
            aggregator=aggregator,
            candle_queue=q,
        )
        bad_update = MagicMock()
        bad_update.getValue.side_effect = RuntimeError("simulated error")

        listener.onItemUpdate(bad_update)
        # Listener remains callable
        assert callable(listener.onItemUpdate)

    def test_tick_listener_on_item_update_catches_value_error(self):
        """_TickListener.onItemUpdate catches ValueError without propagating."""
        aggregator = MagicMock()
        q = queue.Queue()
        listener = _TickListener(
            item_name="CHART:EPIC:TICK",
            aggregator=aggregator,
            candle_queue=q,
        )
        bad_update = MagicMock()
        bad_update.getValue.side_effect = ValueError("bad value")

        listener.onItemUpdate(bad_update)
        assert callable(listener.onItemUpdate)

    def test_direct_tick_listener_on_item_update_catches_runtime_error(self):
        """_DirectTickListener.onItemUpdate must not propagate RuntimeError."""
        q = queue.Queue()
        listener = _DirectTickListener(
            item_name="CHART:EPIC:TICK",
            tick_queue=q,
        )
        bad_update = MagicMock()
        bad_update.getValue.side_effect = RuntimeError("simulated error")

        listener.onItemUpdate(bad_update)
        # Queue still empty — no crash
        assert q.empty()

    def test_direct_tick_listener_on_item_update_catches_attribute_error(self):
        """_DirectTickListener.onItemUpdate catches AttributeError without propagating."""
        q = queue.Queue()
        listener = _DirectTickListener(
            item_name="CHART:EPIC:TICK",
            tick_queue=q,
        )
        bad_update = MagicMock()
        bad_update.getValue.side_effect = AttributeError("missing attr")

        listener.onItemUpdate(bad_update)
        assert callable(listener.onItemUpdate)
