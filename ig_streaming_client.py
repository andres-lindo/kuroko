"""IG Markets Lightstreamer streaming client with candle delivery.

Wraps trading_ig's IGStreamService to subscribe to native OHLC candles via
CHART:{epic}:{resolution} items. Resolution is configurable (e.g. '5MINUTE',
'15MINUTE', '1HOUR'). When a completed candle arrives (CONS_END=1), it is
enqueued and delivered to the registered callback from a dedicated worker
thread (never from the Lightstreamer listener thread).

If the native subscription is unavailable (subscription error), the client
falls back automatically to CHART:{epic}:TICK and aggregates ticks into OHLC
candles of the configured resolution in-process using TickAggregator.

Use candle_frequency_to_resolution() to convert strategy candle_frequency
strings ('5min', '15min', etc.) to IG resolution strings before passing them
to IGStreamingClient.
"""

import logging
import queue
import threading
from datetime import datetime, timezone
from typing import Callable, Optional

from trading_ig.stream import IGStreamService

logger = logging.getLogger(__name__)

# Candle field names used in the Lightstreamer CHART:epic:5MINUTE items
_NATIVE_FIELDS = [
    "BID_OPEN",
    "BID_HIGH",
    "BID_LOW",
    "BID_CLOSE",
    "OFR_OPEN",
    "OFR_HIGH",
    "OFR_LOW",
    "OFR_CLOSE",
    "CONS_END",
    "UTM",
    "LTV",
]

# Tick field names for CHART:epic:TICK fallback
_TICK_FIELDS = [
    "BID",
    "OFR",
    "UTM",
]

# Mapping from IG resolution string to minutes
_RESOLUTION_MINUTES: dict[str, int] = {
    "1MINUTE": 1,
    "2MINUTE": 2,
    "3MINUTE": 3,
    "5MINUTE": 5,
    "10MINUTE": 10,
    "15MINUTE": 15,
    "30MINUTE": 30,
    "1HOUR": 60,
    "2HOUR": 120,
    "3HOUR": 180,
    "4HOUR": 240,
    "1DAY": 1440,
}


def _resolution_to_minutes(resolution: str) -> int:
    """Convert an IG Lightstreamer resolution string to integer minutes.

    Args:
        resolution: IG resolution string (e.g. '5MINUTE', '1HOUR').

    Returns:
        Number of minutes per candle.

    Raises:
        ValueError: If the resolution string is not recognised.
    """
    try:
        return _RESOLUTION_MINUTES[resolution]
    except KeyError:
        raise ValueError(
            f"Unrecognised resolution string: {resolution!r}. "
            f"Valid values: {sorted(_RESOLUTION_MINUTES)}"
        )


def candle_frequency_to_resolution(candle_frequency: str) -> str:
    """Convert a candle_frequency string ('Nmin') to an IG resolution string.

    Maps values like '5min' to '5MINUTE', '60min' to '1HOUR'.

    Args:
        candle_frequency: Frequency string in 'Nmin' format (e.g. '5min', '15min').

    Returns:
        IG resolution string (e.g. '5MINUTE', '15MINUTE', '1HOUR').

    Raises:
        ValueError: If the string is not in 'Nmin' format or the number of
            minutes is not recognised.
    """
    if not candle_frequency.endswith("min"):
        raise ValueError(
            f"candle_frequency must be in 'Nmin' format (e.g. '5min'), got {candle_frequency!r}"
        )
    minutes_str = candle_frequency[:-3]
    try:
        minutes = int(minutes_str)
    except ValueError:
        raise ValueError(
            f"candle_frequency prefix must be numeric, got {candle_frequency!r}"
        )

    # Build reverse lookup from minutes to IG resolution string
    minutes_to_resolution = {v: k for k, v in _RESOLUTION_MINUTES.items()}

    # For minute-based resolutions, prefer 'NMINUTEformat; 1-hour = 60
    if minutes not in minutes_to_resolution:
        raise ValueError(
            f"No IG resolution mapping for {minutes} minutes (from {candle_frequency!r}). "
            f"Valid minute values: {sorted(minutes_to_resolution)}"
        )
    return minutes_to_resolution[minutes]


class _CandleSubscriptionListener:
    """Lightstreamer SubscriptionListener for native 5-minute candle items.

    Receives item updates from the Lightstreamer adapter and enqueues
    completed candles (CONS_END=1) onto the provided queue. All other
    updates are silently discarded.

    Attributes:
        _q: The queue onto which completed candle dicts are placed.
        item_name: The Lightstreamer item name this listener handles.
    """

    def __init__(self, item_name: str, candle_queue: queue.Queue):
        """Initialise the listener with a target item name and delivery queue.

        Args:
            item_name: Lightstreamer item identifier (e.g. 'CHART:epic:5MINUTE').
            candle_queue: Queue onto which completed candle payloads are placed.
        """
        self._item_name = item_name
        self._q = candle_queue

    # NOTE: The Lightstreamer Python client library (lightstreamer-client-lib) uses
    # camelCase callback names that mirror the Java SubscriptionListener interface:
    # onItemUpdate, onSubscription, onSubscriptionError, onUnsubscription, etc.
    # These MUST be camelCase — the library dispatches to these exact method names.
    # snake_case versions would be silently ignored by the Lightstreamer dispatcher.

    def onItemUpdate(self, update) -> None:
        """Process a Lightstreamer item update.

        Called by the Lightstreamer library dispatcher (camelCase required).
        Enqueues a completed candle dict when CONS_END is '1'. Ignores all
        other updates (CONS_END=0 or missing).

        Args:
            update: Lightstreamer ItemUpdate object. Use update.getValue("FIELD")
                to retrieve field values; returns str or None.
        """
        cons_end = update.getValue("CONS_END")
        if cons_end != "1":
            return

        try:
            utm_ms = int(update.getValue("UTM") or "0")
            timestamp = datetime.fromtimestamp(utm_ms / 1000.0, tz=timezone.utc)
        except (ValueError, TypeError):
            timestamp = datetime.now(tz=timezone.utc)

        bid_close = float(update.getValue("BID_CLOSE") or "0")
        ofr_close = float(update.getValue("OFR_CLOSE") or "0")
        try:
            volume = int(update.getValue("LTV") or "0")
        except (ValueError, TypeError):
            volume = 0
        candle = {
            "open": float(update.getValue("BID_OPEN") or "0"),
            "high": float(update.getValue("BID_HIGH") or "0"),
            "low": float(update.getValue("BID_LOW") or "0"),
            "close": bid_close,
            "bid_close": bid_close,
            "ofr_close": ofr_close,
            "spread": ofr_close - bid_close,
            "volume": volume,
            "timestamp": timestamp,
        }
        self._q.put(candle)

    def onSubscription(self) -> None:
        """Called when the subscription is confirmed by the server."""
        logger.info(f"Subscribed to {self._item_name}")

    def onSubscriptionError(self, code: int, message: str) -> None:
        """Called when the server reports a subscription error.

        Args:
            code: IG/Lightstreamer error code.
            message: Human-readable error description.
        """
        logger.error(f"Subscription error on {self._item_name}: {code} {message}")

    def onUnsubscription(self) -> None:
        """Called when the subscription is confirmed as removed."""
        logger.info(f"Unsubscribed from {self._item_name}")


class _TickListener:
    """Lightstreamer SubscriptionListener for tick items (fallback mode).

    Forwards each tick to a TickAggregator which builds 5-minute OHLC
    candles in-process.

    Attributes:
        _aggregator: The TickAggregator instance processing tick data.
        item_name: The Lightstreamer item name this listener handles.
    """

    def __init__(self, item_name: str, aggregator: "TickAggregator"):
        """Initialise the tick listener.

        Args:
            item_name: Lightstreamer item identifier (e.g. 'CHART:epic:TICK').
            aggregator: TickAggregator instance that builds candles from ticks.
        """
        self._item_name = item_name
        self._aggregator = aggregator

    # NOTE: The Lightstreamer Python client library uses camelCase callback names.
    # onItemUpdate MUST be camelCase — the library dispatches to this exact method name.

    def onItemUpdate(self, update) -> None:
        """Forward a tick update to the aggregator.

        Called by the Lightstreamer library dispatcher (camelCase required).

        Args:
            update: Lightstreamer ItemUpdate object. Use update.getValue("FIELD")
                to retrieve field values; returns str or None.
        """
        try:
            bid = float(update.getValue("BID") or "0")
            ofr = float(update.getValue("OFR") or "0")
            utm_ms = int(update.getValue("UTM") or "0")
            utm = datetime.fromtimestamp(utm_ms / 1000.0, tz=timezone.utc)
            self._aggregator.on_tick(bid=bid, ofr=ofr, utm=utm)
        except (ValueError, TypeError) as e:
            logger.debug(f"Tick parse error on {self._item_name}: {e}")

    def onSubscription(self) -> None:
        """Called when the tick subscription is confirmed by the server."""
        logger.info(f"Tick subscription active on {self._item_name}")

    def onSubscriptionError(self, code: int, message: str) -> None:
        """Called when the server reports a tick subscription error.

        Args:
            code: Error code from server.
            message: Human-readable error description.
        """
        logger.error(f"Tick subscription error on {self._item_name}: {code} {message}")

    def onUnsubscription(self) -> None:
        """Called when the tick subscription is removed."""
        logger.info(f"Tick subscription removed from {self._item_name}")


def _make_native_subscription(
    epic: str, resolution: str, listener: _CandleSubscriptionListener
):
    """Build a Lightstreamer Subscription for native 5-minute candle items.

    Creates a MERGE-mode Subscription for CHART:{epic}:{resolution} and
    attaches the provided listener to it. The subscription object is returned
    for passing to IGStreamService.subscribe().

    The subscription object is augmented with _item_name and _listener
    attributes so that tests can inspect the registered listener via the
    object returned from mock_stream_svc.subscribe.call_args.

    Args:
        epic: Instrument identifier (e.g. 'IX.D.NASDAQ.IFMM.IP').
        resolution: Candle resolution string (e.g. '5MINUTE').
        listener: Update handler to attach to the subscription.

    Returns:
        A lightstreamer.client.Subscription with listener attached.
    """
    from lightstreamer.client import Subscription

    item_name = f"CHART:{epic}:{resolution}"
    sub = Subscription(
        mode="MERGE",
        items=[item_name],
        fields=_NATIVE_FIELDS,
    )
    sub.addListener(listener)
    # Attach inspection attributes for testability
    sub._item_name = item_name
    sub._listener = listener
    return sub


def _make_tick_subscription(epic: str, listener: _TickListener):
    """Build a Lightstreamer Subscription for tick items (fallback).

    Creates a DISTINCT-mode Subscription for CHART:{epic}:TICK and attaches
    the provided listener to it.

    Args:
        epic: Instrument identifier.
        listener: Tick update handler to attach.

    Returns:
        A lightstreamer.client.Subscription with listener attached.
    """
    from lightstreamer.client import Subscription

    item_name = f"CHART:{epic}:TICK"
    sub = Subscription(
        mode="DISTINCT",
        items=[item_name],
        fields=_TICK_FIELDS,
    )
    sub.addListener(listener)
    # Attach inspection attributes for testability
    sub._item_name = item_name
    sub._listener = listener
    return sub


class TickAggregator:
    """Aggregates tick data into OHLC candles of a fixed resolution.

    Ticks are accumulated within a time window. When the first tick
    timestamped at or beyond the next window boundary arrives, the
    accumulated window is emitted as a candle dict and a new window begins.

    The first window at startup is always considered partial and is
    discarded — the first candle emitted is always from a complete window.

    Attributes:
        _resolution_minutes: Width of each candle window in minutes.
        _on_candle: Callback invoked with completed candle dicts.
        _window_open: Price of the first tick in the current window.
        _window_high: Highest bid price seen in the current window.
        _window_low: Lowest bid price seen in the current window.
        _window_close: Price of the most recent tick in the current window.
        _window_close_ofr: Offer price of the most recent tick.
        _window_start: Datetime of the window boundary for the current window.
        _window_tick_count: Number of ticks accumulated in the current window.
        _is_startup: True until the first window boundary has been crossed.
    """

    def __init__(
        self,
        resolution_minutes: int,
        on_candle: Callable[[dict], None],
    ):
        """Initialise the aggregator.

        Args:
            resolution_minutes: Candle width in minutes (typically 5).
            on_candle: Callback called with completed candle dicts.
        """
        self._resolution_minutes = resolution_minutes
        self._on_candle = on_candle

        self._window_open: Optional[float] = None
        self._window_high: Optional[float] = None
        self._window_low: Optional[float] = None
        self._window_close: Optional[float] = None
        self._window_close_ofr: Optional[float] = None
        self._window_start: Optional[datetime] = None
        self._window_tick_count: int = 0
        self._is_startup: bool = True

    def _window_boundary(self, ts: datetime) -> datetime:
        """Compute the start of the resolution window containing ts.

        Args:
            ts: A tick timestamp.

        Returns:
            The floored datetime representing the start of the window.
        """
        total_minutes = ts.hour * 60 + ts.minute
        window_index = total_minutes // self._resolution_minutes
        window_minute = window_index * self._resolution_minutes
        hour = window_minute // 60
        minute = window_minute % 60
        return ts.replace(hour=hour, minute=minute, second=0, microsecond=0)

    def on_tick(self, bid: float, ofr: float, utm: datetime) -> None:
        """Process one tick and emit a candle if a window boundary is crossed.

        Args:
            bid: Bid price of the tick.
            ofr: Offer price of the tick.
            utm: UTC timestamp of the tick.
        """
        current_window = self._window_boundary(utm)

        if self._window_start is None:
            # First tick ever — set initial window
            self._window_start = current_window
            self._window_open = bid
            self._window_high = bid
            self._window_low = bid
            self._window_close = bid
            self._window_close_ofr = ofr
            self._window_tick_count = 1
            return

        if current_window > self._window_start:
            # Window boundary crossed — decide whether to emit or discard
            if self._is_startup:
                # Startup partial window: discard accumulated data
                self._is_startup = False
                logger.debug(
                    f"Discarding startup partial window starting at {self._window_start}"
                )
            else:
                # Full window: emit candle
                candle = {
                    "open": self._window_open,
                    "high": self._window_high,
                    "low": self._window_low,
                    "close": self._window_close,
                    "bid_close": self._window_close,
                    "ofr_close": self._window_close_ofr,
                    "spread": self._window_close_ofr - self._window_close,
                    "volume": self._window_tick_count,
                    "timestamp": self._window_start,
                }
                logger.debug(
                    f"Emitting tick-aggregated candle for window {self._window_start}"
                )
                self._on_candle(candle)

            # Start new window with the triggering tick as first data point
            self._window_start = current_window
            self._window_open = bid
            self._window_high = bid
            self._window_low = bid
            self._window_close = bid
            self._window_close_ofr = ofr
            self._window_tick_count = 1
        else:
            # Same window — update running OHLC
            self._window_high = max(self._window_high, bid)
            self._window_low = min(self._window_low, bid)
            self._window_close = bid
            self._window_close_ofr = ofr
            self._window_tick_count += 1


class IGStreamingClient:
    """Lightstreamer streaming client that delivers closed OHLC candles.

    Connects to IG Markets Lightstreamer service using an existing IGService
    authentication session. Subscribes to native CHART:{epic}:{resolution}
    candle items. Resolution defaults to '5MINUTE' but is configurable via
    the ``resolution`` constructor argument (see ``candle_frequency_to_resolution()``
    to convert strategy ``candle_frequency`` strings). When a candle is
    complete (CONS_END=1), it is enqueued and delivered to the registered
    callback from a dedicated worker thread.

    If the native subscription fails, the client falls back to
    CHART:{epic}:TICK and aggregates ticks into candles of the configured
    resolution in-process using TickAggregator.

    Usage::

        resolution = candle_frequency_to_resolution("5min")  # -> "5MINUTE"
        client = IGStreamingClient(ig.ig_service, epic="IX.D.NASDAQ.IFMM.IP",
                                   resolution=resolution)
        client.start(on_candle=my_callback)
        # ... strategy runs ...
        client.stop()

    Attributes:
        _ig_service: The IGService instance providing authentication.
        _epic: Instrument identifier.
        _resolution: Candle resolution for the native subscription.
        _stream_svc: IGStreamService instance (created on start).
        _candle_queue: Queue carrying completed candle dicts.
        _worker: Background thread that dequeues and dispatches candles.
        _stop_event: Event signalling the worker thread to terminate.
        _active_subscription: The current subscription wrapper (native or tick).
    """

    def __init__(self, ig_service, epic: str, resolution: str = "5MINUTE"):
        """Initialise the streaming client.

        Does not connect or subscribe — call start() to begin streaming.

        Args:
            ig_service: Authenticated IGService instance from IGClient.ig_service.
            epic: Instrument identifier (e.g. 'IX.D.NASDAQ.IFMM.IP').
            resolution: Candle resolution for the native subscription (default '5MINUTE').
        """
        self._ig_service = ig_service
        self._epic = epic
        self._resolution = resolution

        self._stream_svc: Optional[IGStreamService] = None
        self._candle_queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._stop_event: threading.Event = threading.Event()
        self._active_subscription = None

    def start(self, on_candle: Callable[[dict], None]) -> None:
        """Subscribe to candle data and begin delivering events to on_candle.

        Creates an IGStreamService from the provided IGService, establishes
        a Lightstreamer session, subscribes to native 5-minute candles, and
        starts a worker thread that dispatches completed candles to on_candle.

        Falls back to tick aggregation if the native subscription fails.

        Notes:
            open/high/low/close are derived from BID prices (not mid-market).
            spread = OFR_CLOSE - BID_CLOSE at candle close.
            volume = LTV (Last Traded Volume) for native candles; tick count
            for tick-aggregated candles.

        Args:
            on_candle: Callable invoked with each completed candle dict.
                       The dict has keys: open, high, low, close, bid_close,
                       ofr_close, spread, volume, timestamp.

        Raises:
            RuntimeError: If start() is called while the client is already running.
                          Call stop() first to restart.
        """
        if self._worker is not None and self._worker.is_alive():
            raise RuntimeError(
                "Already started. Call stop() before calling start() again."
            )
        self._stop_event.clear()

        try:
            self._stream_svc = IGStreamService(self._ig_service)
            self._stream_svc.create_session()
        except Exception:
            # Session creation failed — null out the partially-constructed
            # stream service so stop() does not attempt to disconnect it.
            logger.error("Lightstreamer session creation failed.")
            self._stream_svc = None
            raise

        # Start the worker thread before subscribing so it is ready to
        # receive candles immediately when the subscription confirms.
        self._worker = threading.Thread(
            target=self._worker_loop,
            args=(on_candle,),
            daemon=True,
            name="ig-streaming-worker",
        )
        self._worker.start()

        try:
            self._subscribe_native(on_candle)
        except Exception:
            # Subscription failed — shut down the worker thread and stream
            # service so the orphaned thread does not run indefinitely.
            logger.error("Subscription setup failed; shutting down worker thread.")
            self.stop()
            raise

    def _subscribe_native(self, on_candle: Callable[[dict], None]) -> None:
        """Attempt to subscribe to native 5-minute candle items.

        Falls back to tick aggregation if the subscription raises an error.

        Args:
            on_candle: Candle delivery callback (passed through for fallback).
        """
        item_name = f"CHART:{self._epic}:{self._resolution}"
        listener = _CandleSubscriptionListener(item_name, self._candle_queue)
        sub = _make_native_subscription(self._epic, self._resolution, listener)

        try:
            self._stream_svc.subscribe(sub)
            self._active_subscription = sub
            logger.info(f"Native candle subscription active: {item_name}")
        except Exception as e:
            # Broad catch is intentional — Lightstreamer can raise various exception types
            # (network errors, subscription errors, adapter errors). Log the type and message
            # so the fallback reason is always visible in logs.
            logger.warning(
                f"Native subscription failed "
                f"({type(e).__name__}: {e}); falling back to tick aggregation."
            )
            self._subscribe_tick_fallback()

    def _subscribe_tick_fallback(self) -> None:
        """Subscribe to CHART:{epic}:TICK and aggregate ticks into candles."""
        item_name = f"CHART:{self._epic}:TICK"
        aggregator = TickAggregator(
            resolution_minutes=_resolution_to_minutes(self._resolution),
            on_candle=lambda candle: self._candle_queue.put(candle),
        )
        listener = _TickListener(item_name, aggregator)
        sub = _make_tick_subscription(self._epic, listener)

        self._stream_svc.subscribe(sub)
        self._active_subscription = sub
        logger.info(f"Tick fallback subscription active: {item_name}")

    def _worker_loop(self, on_candle: Callable[[dict], None]) -> None:
        """Dequeue completed candles and invoke on_candle.

        Runs on a dedicated worker thread. Blocks on queue.get() with a
        short timeout so the stop event is checked regularly. After the stop
        event is set, any remaining items in the queue are drained and
        delivered to on_candle before the loop exits — this ensures in-flight
        candles are not silently dropped at shutdown.

        Args:
            on_candle: Callback to invoke for each completed candle.
        """
        logger.debug("Streaming worker thread started.")
        while not self._stop_event.is_set():
            try:
                candle = self._candle_queue.get(timeout=0.05)
                try:
                    on_candle(candle)
                except Exception as e:
                    logger.error(f"Error in on_candle callback: {e}", exc_info=True)
                finally:
                    self._candle_queue.task_done()
            except queue.Empty:
                continue

        # Drain remaining items so in-flight candles are not lost on shutdown.
        # Cap at _DRAIN_BUDGET items to prevent indefinite blocking under
        # thundering-herd conditions; any remaining items are discarded.
        _DRAIN_BUDGET = 50
        drained = 0
        dropped = 0
        while drained < _DRAIN_BUDGET:
            try:
                candle = self._candle_queue.get(timeout=0.01)
                drained += 1
                try:
                    on_candle(candle)
                except Exception as e:
                    logger.error(
                        f"Error in on_candle callback during drain: {e}", exc_info=True
                    )
                finally:
                    self._candle_queue.task_done()
            except queue.Empty:
                break

        # Count and discard any items beyond the budget
        while True:
            try:
                self._candle_queue.get_nowait()
                self._candle_queue.task_done()
                dropped += 1
            except queue.Empty:
                break

        if dropped:
            logger.warning(
                f"Drain budget ({_DRAIN_BUDGET}) exceeded on shutdown; "
                f"dropped {dropped} candle(s) from queue."
            )

        logger.debug("Streaming worker thread stopped.")

    def stop(self) -> None:
        """Disconnect the Lightstreamer session and shut down the worker thread.

        Safe to call even if start() was never called. Blocks until the
        worker thread has exited cleanly.
        """
        self._stop_event.set()

        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=2.0)
            if self._worker.is_alive():
                # Thread did not exit within the timeout; leave _worker set so
                # a subsequent start() call will see it as alive and refuse to
                # start a second worker.
                logger.warning(
                    "Worker thread did not stop within timeout — "
                    "it may still be running."
                )
            else:
                self._worker = None

        if self._stream_svc is not None:
            try:
                self._stream_svc.disconnect()
            except Exception as e:
                logger.warning(f"Error during stream service disconnect: {e}")
            self._stream_svc = None

        self._active_subscription = None
        logger.info("IGStreamingClient stopped.")
