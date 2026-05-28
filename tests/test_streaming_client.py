"""Unit tests for IGStreamingClient (REQ-1 through REQ-4, REQ-16).

Covers:
- REQ-1: Native candle subscription via CHART:{epic}:5MINUTE
  - Scenario 1: completed candle (CONS_END=1) delivered to callback
  - Scenario 2: incomplete candle (CONS_END=0) does NOT trigger callback
- REQ-2: Tick aggregation fallback via CHART:{epic}:TICK
  - Scenario 1: candle completed when first tick of next window arrives
  - Scenario 2: partial window at startup is discarded
- REQ-3: Streaming session created from existing IGService, no re-auth
- REQ-4: Callback dispatched from worker thread, never from LS listener thread

All IGStreamService and Subscription instances are mocked (no live credentials).

Note: tests that call start() mock IGStreamService and the Lightstreamer
Subscription class to avoid triggering deferred imports of
lightstreamer-client-lib. The library is a pinned runtime dependency
(requirements.txt) and will be installed in any environment running the bot,
but the mock boundary ensures unit tests are self-contained.
"""

import queue
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

import pytest

from ig_streaming_client import IGStreamingClient, TickAggregator

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_EPIC = "IX.D.NASDAQ.IFMM.IP"


def _make_mock_ig_stream_service():
    """Return a MagicMock that mimics IGStreamService API.

    The mock records subscribe/unsubscribe/create_session/disconnect calls
    without touching the network.
    """
    svc = MagicMock()
    svc.create_session.return_value = None
    svc.subscribe.return_value = None
    svc.unsubscribe.return_value = None
    svc.disconnect.return_value = None
    return svc


def _make_mock_ig_service():
    """Return a MagicMock that mimics the trading_ig IGService object."""
    ig_service = MagicMock()
    return ig_service


def _build_candle_update(
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
):
    """Return a dict mimicking the field-value map from a Lightstreamer update."""
    return {
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


# ---------------------------------------------------------------------------
# REQ-3: Auth passthrough — session created from existing IGService, no re-auth
# ---------------------------------------------------------------------------


class TestAuthPassthrough:
    """REQ-3: IGStreamingClient must not create a new IG session independently."""

    def test_start_creates_stream_service_from_provided_ig_service(self):
        """REQ-3 scenario: start() uses the provided IGService to create the stream session."""
        ig_service = _make_mock_ig_service()
        mock_stream_svc = _make_mock_ig_stream_service()

        with patch(
            "ig_streaming_client.IGStreamService", return_value=mock_stream_svc
        ) as MockStreamServiceCls:
            client = IGStreamingClient(ig_service, _EPIC)
            client.start(on_candle=MagicMock())
            client.stop()

        MockStreamServiceCls.assert_called_once_with(ig_service)
        mock_stream_svc.create_session.assert_called_once()

    def test_start_does_not_call_ig_service_create_session_directly(self):
        """REQ-3: IGStreamingClient must NOT call ig_service.create_session directly."""
        ig_service = _make_mock_ig_service()
        mock_stream_svc = _make_mock_ig_stream_service()

        with patch("ig_streaming_client.IGStreamService", return_value=mock_stream_svc):
            client = IGStreamingClient(ig_service, _EPIC)
            client.start(on_candle=MagicMock())
            client.stop()

        ig_service.create_session.assert_not_called()


# ---------------------------------------------------------------------------
# REQ-1 / Scenario 1: CONS_END=1 → candle delivered via callback
# ---------------------------------------------------------------------------


class TestNativeCandleDelivery:
    """REQ-1: completed candles (CONS_END=1) must be delivered to on_candle callback."""

    def _start_and_trigger(self, mock_stream_svc, update_dict):
        """Helper: start client, capture the LS listener, simulate an update.

        Uses threading.Event synchronisation instead of time.sleep.
        """
        ig_service = _make_mock_ig_service()
        received = []
        delivered = threading.Event()

        def callback(candle):
            received.append(candle)
            delivered.set()

        with patch("ig_streaming_client.IGStreamService", return_value=mock_stream_svc):
            client = IGStreamingClient(ig_service, _EPIC)
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

    def test_delivered_candle_has_ohlc_fields(self):
        """REQ-1: candle payload must include open, high, low, close."""
        mock_stream_svc = _make_mock_ig_stream_service()
        update = _build_candle_update(
            bid_open=20000.0,
            bid_high=20050.0,
            bid_low=19990.0,
            bid_close=20030.0,
            cons_end="1",
        )

        received = self._start_and_trigger(mock_stream_svc, update)

        candle = received[0]
        assert "open" in candle
        assert "high" in candle
        assert "low" in candle
        assert "close" in candle

    def test_delivered_candle_has_timestamp(self):
        """REQ-1: candle payload must include a UTC timestamp."""
        mock_stream_svc = _make_mock_ig_stream_service()
        update = _build_candle_update(cons_end="1", utm="1716825600000")

        received = self._start_and_trigger(mock_stream_svc, update)

        candle = received[0]
        assert "timestamp" in candle
        assert isinstance(candle["timestamp"], datetime)

    def test_delivered_candle_values_match_bid_fields(self):
        """REQ-1: candle open/high/low/close must be derived from BID fields."""
        mock_stream_svc = _make_mock_ig_stream_service()
        update = _build_candle_update(
            bid_open=20000.0,
            bid_high=20100.0,
            bid_low=19950.0,
            bid_close=20080.0,
            cons_end="1",
        )

        received = self._start_and_trigger(mock_stream_svc, update)

        candle = received[0]
        assert candle["open"] == pytest.approx(20000.0)
        assert candle["high"] == pytest.approx(20100.0)
        assert candle["low"] == pytest.approx(19950.0)
        assert candle["close"] == pytest.approx(20080.0)

    def test_delivered_candle_has_bid_close_and_ofr_close(self):
        """REQ-1: payload must include bid_close and ofr_close for spread calculation."""
        mock_stream_svc = _make_mock_ig_stream_service()
        update = _build_candle_update(
            bid_close=20030.0, ofr_close=20031.0, cons_end="1"
        )

        received = self._start_and_trigger(mock_stream_svc, update)

        candle = received[0]
        assert "bid_close" in candle
        assert "ofr_close" in candle
        assert candle["bid_close"] == pytest.approx(20030.0)
        assert candle["ofr_close"] == pytest.approx(20031.0)

    def test_delivered_candle_has_spread_field(self):
        """Candle must include a 'spread' field equal to OFR_CLOSE - BID_CLOSE."""
        mock_stream_svc = _make_mock_ig_stream_service()
        update = _build_candle_update(
            bid_close=20030.0, ofr_close=20031.5, cons_end="1"
        )

        received = self._start_and_trigger(mock_stream_svc, update)

        candle = received[0]
        assert "spread" in candle
        assert candle["spread"] == pytest.approx(1.5)  # OFR_CLOSE - BID_CLOSE

    def test_delivered_candle_has_volume_field(self):
        """REQ-1: candle payload must include a 'volume' field from LTV."""
        mock_stream_svc = _make_mock_ig_stream_service()
        update = _build_candle_update(cons_end="1", ltv="42")

        received = self._start_and_trigger(mock_stream_svc, update)

        candle = received[0]
        assert "volume" in candle
        assert candle["volume"] == 42


# ---------------------------------------------------------------------------
# REQ-1 / Scenario 2: CONS_END=0 → callback must NOT be called
# (Event-based equivalents are in TestNativeCandleDeliveryEventBased)
# ---------------------------------------------------------------------------


class TestIncompleteCandleSuppression:
    """REQ-1/Scenario 2: CONS_END=0 must NOT invoke on_candle."""

    def test_multiple_incomplete_followed_by_complete(self):
        """CONS_END=0 updates are ignored; only CONS_END=1 triggers delivery."""
        ig_service = _make_mock_ig_service()
        mock_stream_svc = _make_mock_ig_stream_service()
        received = []
        delivered = threading.Event()

        def callback(candle):
            received.append(candle)
            delivered.set()

        with patch("ig_streaming_client.IGStreamService", return_value=mock_stream_svc):
            client = IGStreamingClient(ig_service, _EPIC)
            client.start(on_candle=callback)

            sub = mock_stream_svc.subscribe.call_args[0][0]
            listener = sub._listener
            for _ in range(3):
                listener.onItemUpdate(_build_candle_update(cons_end="0"))
            listener.onItemUpdate(_build_candle_update(cons_end="1"))

            delivered.wait(timeout=2.0)
            client.stop()

        assert len(received) == 1


# ---------------------------------------------------------------------------
# REQ-4: Candle dispatched from worker thread, not Lightstreamer listener thread
# ---------------------------------------------------------------------------


class TestCallbackThreadSafety:
    """REQ-4: on_candle must be called from the worker thread, never the LS thread."""

    def test_callback_is_not_called_from_ls_listener_thread(self):
        """REQ-4 scenario: on_candle must execute on the worker thread."""
        ig_service = _make_mock_ig_service()
        mock_stream_svc = _make_mock_ig_stream_service()
        callback_thread_ids = []
        delivered = threading.Event()

        def tracking_callback(candle):
            callback_thread_ids.append(threading.current_thread().ident)
            delivered.set()

        with patch("ig_streaming_client.IGStreamService", return_value=mock_stream_svc):
            client = IGStreamingClient(ig_service, _EPIC)
            client.start(on_candle=tracking_callback)

            sub = mock_stream_svc.subscribe.call_args[0][0]
            listener = sub._listener
            ls_listener_thread_id = threading.current_thread().ident
            listener.onItemUpdate(_build_candle_update(cons_end="1"))

            delivered.wait(timeout=2.0)
            client.stop()

        assert callback_thread_ids, "Callback was never invoked"
        for tid in callback_thread_ids:
            assert tid != ls_listener_thread_id, (
                "on_candle was called on the Lightstreamer listener thread — "
                "it must be called from the worker thread only."
            )


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------


class TestGracefulShutdown:
    """stop() must unsubscribe and disconnect cleanly."""

    def test_stop_calls_disconnect(self):
        """stop() must call disconnect on the stream service."""
        ig_service = _make_mock_ig_service()
        mock_stream_svc = _make_mock_ig_stream_service()

        with patch("ig_streaming_client.IGStreamService", return_value=mock_stream_svc):
            client = IGStreamingClient(ig_service, _EPIC)
            client.start(on_candle=MagicMock())
            client.stop()

        mock_stream_svc.disconnect.assert_called_once()

    def test_stop_without_start_does_not_raise(self):
        """stop() before start() must be a safe no-op."""
        ig_service = _make_mock_ig_service()
        mock_stream_svc = _make_mock_ig_stream_service()

        with patch("ig_streaming_client.IGStreamService", return_value=mock_stream_svc):
            client = IGStreamingClient(ig_service, _EPIC)
            # Calling stop before start should not raise
            try:
                client.stop()
            except Exception as e:
                pytest.fail(f"stop() before start() raised: {e}")


# ---------------------------------------------------------------------------
# REQ-2: TickAggregator unit tests
# ---------------------------------------------------------------------------


class TestTickAggregator:
    """REQ-2: TickAggregator aggregates ticks into 5-min OHLC candles."""

    def _make_ts(self, minute: int, second: int = 0) -> datetime:
        """Return a UTC datetime in a fixed hour, at the given minute:second."""
        return datetime(2024, 5, 27, 12, minute, second, tzinfo=timezone.utc)

    def test_candle_completed_on_first_tick_of_next_window(self):
        """REQ-2/Scenario 1: candle emitted when first tick of next window arrives.

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
        """REQ-2/Scenario 2: first partial window must not produce a candle."""
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
        """REQ-2: aggregated candle must include bid_close and ofr_close."""
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
        """REQ-2: aggregated candle must include 'spread' = last tick's OFR - BID."""
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
        """REQ-2: aggregated candle must include 'volume' as tick count in the window."""
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
# REQ-2: Native subscription fallback to tick aggregation on subscription error
# ---------------------------------------------------------------------------


class TestNativeFallbackToTick:
    """REQ-2: When native 5MINUTE subscription fails, client falls back to TICK."""

    def test_fallback_subscribed_on_native_subscription_error(self):
        """REQ-2: on native subscription error, fallback tick subscription is created."""
        ig_service = _make_mock_ig_service()
        mock_stream_svc = _make_mock_ig_stream_service()

        # Simulate native subscription raising an exception
        def raise_on_first_subscribe(sub):
            item_name = sub._item_name
            if "5MINUTE" in item_name:
                raise RuntimeError("Subscription not available")
            # Tick subscription succeeds silently

        mock_stream_svc.subscribe.side_effect = raise_on_first_subscribe

        with patch("ig_streaming_client.IGStreamService", return_value=mock_stream_svc):
            client = IGStreamingClient(ig_service, _EPIC)
            client.start(on_candle=MagicMock())
            client.stop()

        # subscribe should have been called twice: once for 5MINUTE (failed),
        # once for TICK (fallback)
        assert mock_stream_svc.subscribe.call_count == 2


# ---------------------------------------------------------------------------
# Issue 8: start() idempotency guard
# ---------------------------------------------------------------------------


class TestStartIdempotency:
    """Issue 8: calling start() twice must not create orphaned threads or sessions."""

    def test_start_twice_raises_runtime_error(self):
        """start() called a second time before stop() must raise RuntimeError."""
        ig_service = _make_mock_ig_service()
        mock_stream_svc = _make_mock_ig_stream_service()

        with patch("ig_streaming_client.IGStreamService", return_value=mock_stream_svc):
            client = IGStreamingClient(ig_service, _EPIC)
            client.start(on_candle=MagicMock())

            with pytest.raises(RuntimeError, match="Already started"):
                client.start(on_candle=MagicMock())

            client.stop()

    def test_start_after_stop_succeeds(self):
        """start() after stop() must succeed (not raise RuntimeError)."""
        ig_service = _make_mock_ig_service()
        mock_stream_svc = _make_mock_ig_stream_service()

        with patch("ig_streaming_client.IGStreamService", return_value=mock_stream_svc):
            client = IGStreamingClient(ig_service, _EPIC)
            client.start(on_candle=MagicMock())
            client.stop()
            # Should not raise
            client.start(on_candle=MagicMock())
            client.stop()


# ---------------------------------------------------------------------------
# Issue 10: Queue draining on shutdown
# ---------------------------------------------------------------------------


class TestQueueDrainOnShutdown:
    """Issue 10: queued candles must be delivered to the callback before stop() returns."""

    def test_queued_candles_delivered_before_stop_returns(self):
        """Candles placed on queue before stop() must be processed before stop() returns."""
        ig_service = _make_mock_ig_service()
        mock_stream_svc = _make_mock_ig_stream_service()
        received = []
        delivered = threading.Event()

        def tracking_callback(candle):
            received.append(candle)
            if len(received) >= 2:
                delivered.set()

        with patch("ig_streaming_client.IGStreamService", return_value=mock_stream_svc):
            client = IGStreamingClient(ig_service, _EPIC)
            client.start(on_candle=tracking_callback)

            # Enqueue 2 candles directly (simulating candles arriving just before stop)
            sub = mock_stream_svc.subscribe.call_args[0][0]
            listener = sub._listener
            listener.onItemUpdate(_build_candle_update(cons_end="1"))
            listener.onItemUpdate(_build_candle_update(cons_end="1"))

            # Wait briefly for queue to be populated, then stop
            delivered.wait(timeout=2.0)
            client.stop()

        # After stop, both candles must have been delivered
        assert len(received) == 2, f"Expected 2 candles delivered, got {len(received)}"


# ---------------------------------------------------------------------------
# Issue 5: Replace time.sleep(0.1) with threading.Event in existing tests
# These tests verify the same behaviour but use Event-based synchronisation
# ---------------------------------------------------------------------------


class TestNativeCandleDeliveryEventBased:
    """Verify candle delivery using threading.Event instead of time.sleep."""

    def _start_and_trigger_event(self, mock_stream_svc, update_dict):
        """Helper: start client, trigger update, wait via Event instead of sleep."""
        ig_service = _make_mock_ig_service()
        received = []
        delivered = threading.Event()

        def callback(candle):
            received.append(candle)
            delivered.set()

        with patch("ig_streaming_client.IGStreamService", return_value=mock_stream_svc):
            client = IGStreamingClient(ig_service, _EPIC)
            client.start(on_candle=callback)

            sub = mock_stream_svc.subscribe.call_args[0][0]
            listener = sub._listener
            listener.onItemUpdate(update_dict)

            delivered.wait(timeout=2.0)
            client.stop()

        return received

    def test_cons_end_1_triggers_callback_event_based(self):
        """CONS_END=1 triggers on_candle (Event-based wait — no sleep)."""
        mock_stream_svc = _make_mock_ig_stream_service()
        update = _build_candle_update(cons_end="1")

        received = self._start_and_trigger_event(mock_stream_svc, update)

        assert len(received) == 1

    def test_cons_end_0_no_callback_event_based(self):
        """CONS_END=0 does not trigger on_candle (Event-based wait — no sleep)."""
        ig_service = _make_mock_ig_service()
        mock_stream_svc = _make_mock_ig_stream_service()
        received = []
        # Use a short timeout Event; after 0.2s assume no delivery
        no_delivery = threading.Event()

        def callback(candle):
            received.append(candle)
            no_delivery.set()

        with patch("ig_streaming_client.IGStreamService", return_value=mock_stream_svc):
            client = IGStreamingClient(ig_service, _EPIC)
            client.start(on_candle=callback)

            sub = mock_stream_svc.subscribe.call_args[0][0]
            listener = sub._listener
            listener.onItemUpdate(_build_candle_update(cons_end="0"))

            # Wait briefly; callback should NOT fire
            no_delivery.wait(timeout=0.2)
            client.stop()

        assert received == [], f"Expected no candles but got {received}"


# ---------------------------------------------------------------------------
# Issue 11: Lightstreamer camelCase callback naming verification
# ---------------------------------------------------------------------------


class TestLightstreamerCallbackNaming:
    """Issue 11: Listener callbacks must use camelCase to match Lightstreamer interface."""

    def test_candle_listener_has_onItemUpdate_method(self):
        """_CandleSubscriptionListener must have onItemUpdate (camelCase) for the LS library."""
        from ig_streaming_client import _CandleSubscriptionListener
        import queue as q

        listener = _CandleSubscriptionListener("CHART:TEST:5MINUTE", q.Queue())

        assert hasattr(listener, "onItemUpdate"), (
            "_CandleSubscriptionListener must expose onItemUpdate (camelCase) "
            "to match the Lightstreamer SubscriptionListener interface."
        )
        assert callable(getattr(listener, "onItemUpdate"))

    def test_tick_listener_has_onItemUpdate_method(self):
        """_TickListener must have onItemUpdate (camelCase) for the LS library."""
        from ig_streaming_client import _TickListener, TickAggregator

        agg = TickAggregator(5, lambda c: None)
        listener = _TickListener("CHART:TEST:TICK", agg)

        assert hasattr(listener, "onItemUpdate"), (
            "_TickListener must expose onItemUpdate (camelCase) "
            "to match the Lightstreamer SubscriptionListener interface."
        )
        assert callable(getattr(listener, "onItemUpdate"))

    def test_candle_listener_onSubscription_is_camelcase(self):
        """_CandleSubscriptionListener must have onSubscription (not on_subscription)."""
        from ig_streaming_client import _CandleSubscriptionListener
        import queue as q

        listener = _CandleSubscriptionListener("CHART:TEST:5MINUTE", q.Queue())
        assert hasattr(listener, "onSubscription")
        assert not hasattr(listener, "on_subscription")
