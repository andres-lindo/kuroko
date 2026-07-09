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
import time
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

    Any unexpected exception in onItemUpdate is caught and logged at ERROR
    level with traceback — the listener never terminates due to a single
    bad update (GAP-1).

    When a mid-session subscription error occurs and setup_complete_check
    returns True, onSubscriptionError enqueues a reconnect sentinel (GAP-2).

    Attributes:
        _q: The queue onto which completed candle dicts and reconnect sentinels
            are placed.
        _item_name: The Lightstreamer item name this listener handles.
        _setup_complete_check: Optional callable; see constructor for details.
    """

    def __init__(
        self,
        item_name: str,
        candle_queue: queue.Queue,
        setup_complete_check: Optional[Callable[[], bool]] = None,
    ):
        """Initialise the listener with a target item name and delivery queue.

        Args:
            item_name: Lightstreamer item identifier (e.g. 'CHART:epic:5MINUTE').
            candle_queue: Queue onto which completed candle payloads are placed.
            setup_complete_check: Optional callable that returns True when the
                streaming client has completed initial setup. When not None and
                returning True, onSubscriptionError enqueues a reconnect sentinel.
                When None or returning False, onSubscriptionError only logs.
        """
        self._item_name = item_name
        self._q = candle_queue
        self._setup_complete_check = setup_complete_check

    # NOTE: The Lightstreamer Python client library (lightstreamer-client-lib) uses
    # camelCase callback names that mirror the Java SubscriptionListener interface:
    # onItemUpdate, onSubscription, onSubscriptionError, onUnsubscription, etc.
    # These MUST be camelCase — the library dispatches to these exact method names.
    # snake_case versions would be silently ignored by the Lightstreamer dispatcher.

    def onItemUpdate(self, update) -> None:
        """Process a Lightstreamer item update.

        Called by the Lightstreamer library dispatcher (camelCase required).
        Enqueues a completed candle dict when CONS_END is '1'. Ignores all
        other updates (CONS_END=0 or missing). Any unexpected exception is
        caught, logged at ERROR level with traceback, and processing continues
        (GAP-1: broad exception catch).

        Args:
            update: Lightstreamer ItemUpdate object. Use update.getValue("FIELD")
                to retrieve field values; returns str or None.
        """
        try:
            cons_end = update.getValue("CONS_END")
            if cons_end != "1":
                return

            try:
                utm_ms = int(update.getValue("UTM") or "0")
                timestamp = datetime.fromtimestamp(utm_ms / 1000.0, tz=timezone.utc)
            except (ValueError, TypeError):
                timestamp = datetime.now(tz=timezone.utc)

            try:
                bid_close = float(update.getValue("BID_CLOSE") or "0")
                ofr_close = float(update.getValue("OFR_CLOSE") or "0")
                try:
                    volume = int(update.getValue("LTV") or "0")
                except (ValueError, TypeError):
                    volume = 0
                candle = {
                    "type": "candle",
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
            except (ValueError, TypeError) as e:
                logger.debug(f"Candle field parse error on {self._item_name}: {e}")
                return
            self._q.put(candle)
        except Exception as e:
            logger.error(
                f"Unexpected error in _CandleSubscriptionListener.onItemUpdate "
                f"on {self._item_name}: {e}",
                exc_info=True,
            )

    def onSubscription(self) -> None:
        """Called when the subscription is confirmed by the server."""
        logger.info(f"Subscribed to {self._item_name}")

    def onSubscriptionError(self, code: int, message: str) -> None:
        """Called when the server reports a subscription error.

        When the streaming client has completed initial setup (setup_complete_check
        returns True), enqueues a ``{"type": "reconnect"}`` sentinel to trigger
        reconnection. During initial setup failures, only logs (GAP-2).

        Args:
            code: IG/Lightstreamer error code.
            message: Human-readable error description.
        """
        logger.error(f"Subscription error on {self._item_name}: {code} {message}")
        if self._setup_complete_check is not None and self._setup_complete_check():
            logger.warning(
                f"Mid-session subscription error on {self._item_name} — "
                "enqueuing reconnect sentinel."
            )
            self._q.put({"type": "reconnect"})

    def onUnsubscription(self) -> None:
        """Called when the subscription is confirmed as removed."""
        logger.info(f"Unsubscribed from {self._item_name}")


class _TickListener:
    """Lightstreamer SubscriptionListener for tick items (fallback mode).

    Forwards each tick to a TickAggregator which builds 5-minute OHLC
    candles in-process.

    Attributes:
        _aggregator: The TickAggregator instance processing tick data.
        _item_name: The Lightstreamer item name this listener handles.
        _q: Queue for enqueuing reconnect sentinels on subscription errors.
        _setup_complete_check: Optional callable returning True when initial
            setup is complete. Used by onSubscriptionError for GAP-2.
    """

    def __init__(
        self,
        item_name: str,
        aggregator: "TickAggregator",
        candle_queue: Optional[queue.Queue] = None,
        setup_complete_check: Optional[Callable[[], bool]] = None,
    ):
        """Initialise the tick listener.

        Args:
            item_name: Lightstreamer item identifier (e.g. 'CHART:epic:TICK').
            aggregator: TickAggregator instance that builds candles from ticks.
            candle_queue: Queue onto which reconnect sentinels are placed when
                a mid-session subscription error occurs. When None,
                onSubscriptionError only logs.
            setup_complete_check: Optional callable that returns True when the
                streaming client has completed initial setup. When not None and
                returning True, onSubscriptionError enqueues a reconnect sentinel.
        """
        self._item_name = item_name
        self._aggregator = aggregator
        self._q = candle_queue
        self._setup_complete_check = setup_complete_check

    # NOTE: The Lightstreamer Python client library uses camelCase callback names.
    # onItemUpdate MUST be camelCase — the library dispatches to this exact method name.

    def onItemUpdate(self, update) -> None:
        """Forward a tick update to the aggregator.

        Called by the Lightstreamer library dispatcher (camelCase required).
        Any unexpected exception is caught, logged at ERROR level with traceback,
        and processing continues (GAP-1: broad exception catch).

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
        except Exception as e:
            logger.error(
                f"Unexpected error in _TickListener.onItemUpdate "
                f"on {self._item_name}: {e}",
                exc_info=True,
            )

    def onSubscription(self) -> None:
        """Called when the tick subscription is confirmed by the server."""
        logger.info(f"Tick subscription active on {self._item_name}")

    def onSubscriptionError(self, code: int, message: str) -> None:
        """Called when the server reports a tick subscription error.

        When the streaming client has completed initial setup (setup_complete_check
        returns True), enqueues a ``{"type": "reconnect"}`` sentinel to trigger
        reconnection. During initial setup failures, only logs (GAP-2).

        Args:
            code: Error code from server.
            message: Human-readable error description.
        """
        logger.error(f"Tick subscription error on {self._item_name}: {code} {message}")
        if (
            self._q is not None
            and self._setup_complete_check is not None
            and self._setup_complete_check()
        ):
            logger.warning(
                f"Mid-session tick subscription error on {self._item_name} — "
                "enqueuing reconnect sentinel."
            )
            self._q.put({"type": "reconnect"})

    def onUnsubscription(self) -> None:
        """Called when the tick subscription is removed."""
        logger.info(f"Tick subscription removed from {self._item_name}")


class _ConnectionListener:
    """Lightstreamer ClientListener that detects terminal disconnects.

    Monitors the Lightstreamer connection status and enqueues a reconnect
    sentinel onto the shared worker queue when a bare ``DISCONNECTED`` status
    is received. ``DISCONNECTED:WILL-RETRY`` and ``DISCONNECTED:TRYING-RECOVERY``
    are intentionally ignored — those transitions are handled internally by the
    Lightstreamer library.

    A ``_fired`` flag prevents duplicate sentinels from a single disconnect event,
    since the library may fire ``onStatusChange`` multiple times for one physical drop.

    Attributes:
        _q: The shared worker queue.
        _fired: True once the sentinel has been enqueued for this listener instance.
    """

    def __init__(self, reconnect_queue: queue.Queue):
        """Initialise the connection listener.

        Args:
            reconnect_queue: The shared worker queue onto which the reconnect
                sentinel is placed.
        """
        self._q = reconnect_queue
        self._fired: bool = False

    def onStatusChange(self, status: str) -> None:
        """React to Lightstreamer connection status changes.

        Places a ``{"type": "reconnect"}`` sentinel on the queue only when
        ``status`` is the bare string ``"DISCONNECTED"`` and no sentinel has
        been placed yet for this listener instance.

        Args:
            status: The new Lightstreamer connection status string.
        """
        if status == "DISCONNECTED" and not self._fired:
            self._fired = True
            logger.warning(
                f"Lightstreamer terminal disconnect detected (status={status!r})"
            )
            self._q.put({"type": "reconnect"})

    def onServerError(self, code: int, message: str) -> None:
        """Log Lightstreamer server errors.

        Args:
            code: IG/Lightstreamer error code.
            message: Human-readable error description.
        """
        logger.error(f"Lightstreamer server error: {code} {message}")

    def onPropertyChange(self, property: str) -> None:
        """No-op — property changes do not require action.

        Args:
            property: The name of the changed connection property.
        """

    def onListenStart(self) -> None:
        """No-op — required by the ClientListener interface."""

    def onListenEnd(self) -> None:
        """No-op — required by the ClientListener interface."""


class _DirectTickListener:
    """Lightstreamer SubscriptionListener for raw tick items in tick mode.

    Unlike _TickListener (which feeds a TickAggregator for the candle-fallback
    path), this listener enqueues raw tick dicts directly onto the shared queue
    so the worker thread can dispatch them to the on_tick callback.

    All strategy state mutations happen on the single worker thread — this
    listener's onItemUpdate MUST only call queue.put() and nothing else.

    Attributes:
        _item_name: The Lightstreamer item name this listener handles.
        _q: The shared queue onto which tick dicts are placed.
        _setup_complete_check: Optional callable returning True when initial
            setup is complete. Used by onSubscriptionError for GAP-2.
    """

    def __init__(
        self,
        item_name: str,
        tick_queue: queue.Queue,
        setup_complete_check: Optional[Callable[[], bool]] = None,
    ):
        """Initialise the listener.

        Args:
            item_name: Lightstreamer item identifier (e.g. 'CHART:epic:TICK').
            tick_queue: Shared queue for both candle and tick items.
            setup_complete_check: Optional callable that returns True when the
                streaming client has completed initial setup. When not None and
                returning True, onSubscriptionError enqueues a reconnect sentinel.
        """
        self._item_name = item_name
        self._q = tick_queue
        self._setup_complete_check = setup_complete_check

    # NOTE: The Lightstreamer Python client library uses camelCase callback names.
    # onItemUpdate MUST be camelCase — the library dispatches to this exact method name.

    def onItemUpdate(self, update) -> None:
        """Enqueue a raw tick dict onto the shared queue.

        Called by the Lightstreamer library dispatcher (camelCase required).
        The only side effect is a queue.put() call — no strategy attributes
        are read or written (thread-safety requirement, REQ-12). Any unexpected
        exception is caught, logged at ERROR level with traceback, and processing
        continues (GAP-1: broad exception catch).

        Args:
            update: Lightstreamer ItemUpdate object. Use update.getValue("FIELD")
                to retrieve field values; returns str or None.
        """
        try:
            bid = float(update.getValue("BID") or "0")
            ofr = float(update.getValue("OFR") or "0")
            utm = int(update.getValue("UTM") or "0")
        except (ValueError, TypeError) as e:
            logger.debug(f"Tick parse error on {self._item_name}: {e}")
            return
        except Exception as e:
            logger.error(
                f"Unexpected error in _DirectTickListener.onItemUpdate "
                f"on {self._item_name}: {e}",
                exc_info=True,
            )
            return
        self._q.put({"type": "tick", "bid": bid, "ofr": ofr, "utm": utm})

    def onSubscription(self) -> None:
        """Called when the tick subscription is confirmed by the server."""
        logger.info(f"Direct tick subscription active on {self._item_name}")

    def onSubscriptionError(self, code: int, message: str) -> None:
        """Called when the server reports a subscription error.

        When the streaming client has completed initial setup (setup_complete_check
        returns True), enqueues a ``{"type": "reconnect"}`` sentinel to trigger
        reconnection. During initial setup failures, only logs (GAP-2).

        Args:
            code: Error code from server.
            message: Human-readable error description.
        """
        logger.error(
            f"Direct tick subscription error on {self._item_name}: {code} {message}"
        )
        if self._setup_complete_check is not None and self._setup_complete_check():
            logger.warning(
                f"Mid-session direct tick subscription error on {self._item_name} — "
                "enqueuing reconnect sentinel."
            )
            self._q.put({"type": "reconnect"})

    def onUnsubscription(self) -> None:
        """Called when the tick subscription is removed."""
        logger.info(f"Direct tick subscription removed from {self._item_name}")


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
        epic: Instrument identifier (e.g. 'IX.D.SPTRD.IFMM.IP').
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


def _make_tick_subscription(epic: str, listener: _TickListener | _DirectTickListener):
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
        client = IGStreamingClient(ig.ig_service, epic="IX.D.SPTRD.IFMM.IP",
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
        _setup_complete: True after start() or _restart_streaming() completes
            successfully; False initially, during stop(), and at the entry of
            _restart_streaming(). Used by listeners to distinguish mid-session
            subscription errors from initial-setup failures (GAP-2).
        _last_data_ts: Monotonic timestamp of the last candle or tick dispatched
            to the callback. Updated after every successful queue.get(). Used
            by the watchdog in the queue.Empty branch (GAP-3).
        _watchdog_timeout_s: Number of seconds without data before the watchdog
            enqueues a reconnect sentinel. Computed as
            ``max(resolution_minutes * 3, 15) * 60`` (GAP-3).
    """

    def __init__(self, ig_service, epic: str, resolution: str = "5MINUTE"):
        """Initialise the streaming client.

        Does not connect or subscribe — call start() to begin streaming.

        Args:
            ig_service: Authenticated IGService instance from IGClient.ig_service.
            epic: Instrument identifier (e.g. 'IX.D.SPTRD.IFMM.IP').
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
        self._using_tick_fallback: bool = False
        self._setup_complete: bool = False
        self._last_data_ts: float = time.monotonic()
        resolution_minutes = _resolution_to_minutes(resolution)
        self._watchdog_timeout_s: float = max(resolution_minutes * 3, 15) * 60.0

    def start(
        self,
        on_candle: Callable[[dict], None],
        on_tick: Optional[Callable[[dict], None]] = None,
        on_reconnect: Optional[Callable[[], None]] = None,
    ) -> None:
        """Subscribe to candle data and begin delivering events to on_candle.

        Creates an IGStreamService from the provided IGService, establishes
        a Lightstreamer session, subscribes to native 5-minute candles, and
        starts a worker thread that dispatches completed candles to on_candle.

        Falls back to tick aggregation if the native subscription fails.

        When on_tick is provided (tick mode), a second Lightstreamer subscription
        to CHART:{epic}:TICK is created using _DirectTickListener. Raw ticks are
        enqueued onto the same shared queue and dispatched to on_tick from the
        single worker thread — guaranteeing no concurrency between on_candle and
        on_tick.

        When on_reconnect is provided, a _ConnectionListener is attached to the
        Lightstreamer client. On terminal disconnect (bare ``DISCONNECTED`` status),
        the listener places a reconnect sentinel on the queue and the worker thread
        dispatches it to on_reconnect. If on_reconnect is None, sentinel items are
        silently ignored.

        Notes:
            open/high/low/close are derived from BID prices (not mid-market).
            spread = OFR_CLOSE - BID_CLOSE at candle close.
            volume = LTV (Last Traded Volume) for native candles; tick count
            for tick-aggregated candles.

        Args:
            on_candle: Callable invoked with each completed candle dict.
                       The dict has keys: type, open, high, low, close, bid_close,
                       ofr_close, spread, volume, timestamp.
            on_tick: Optional callable invoked with each raw tick dict.
                     The dict has keys: type, bid, ofr, utm. When None (default),
                     no tick subscription is created and candle mode is unchanged.
            on_reconnect: Optional callable invoked with no arguments when a terminal
                          Lightstreamer disconnect is detected. Runs on the worker
                          thread. When None (default), reconnect sentinels are ignored.

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

        # Attach a connection listener to detect terminal disconnects before
        # starting the worker, so no status change is missed.
        self._stream_svc.add_client_listener(_ConnectionListener(self._candle_queue))

        # Start the worker thread before subscribing so it is ready to
        # receive candles immediately when the subscription confirms.
        self._worker = threading.Thread(
            target=self._worker_loop,
            args=(on_candle, on_tick, on_reconnect),
            daemon=True,
            name="ig-streaming-worker",
        )
        self._worker.start()

        try:
            self._subscribe_native(on_candle)
            if on_tick is not None and not self._using_tick_fallback:
                self._subscribe_tick_direct()
            elif on_tick is not None and self._using_tick_fallback:
                logger.warning(
                    "Tick mode requested but native subscription fell back to tick aggregation. "
                    "Direct tick subscription skipped — tick-mode trade signals are disabled. "
                    "Strategy will receive synthetic candles only."
                )
        except Exception:
            # Subscription failed — shut down the worker thread and stream
            # service so the orphaned thread does not run indefinitely.
            logger.error("Subscription setup failed; shutting down worker thread.")
            self.stop()
            raise

        # Mark setup complete so mid-session subscription errors (GAP-2) trigger
        # a reconnect sentinel rather than being silently swallowed.
        self._setup_complete = True

    def _subscribe_native(self, on_candle: Callable[[dict], None]) -> None:
        """Attempt to subscribe to native 5-minute candle items.

        Falls back to tick aggregation if the subscription raises an error.

        Args:
            on_candle: Candle delivery callback (passed through for fallback).
        """
        item_name = f"CHART:{self._epic}:{self._resolution}"
        listener = _CandleSubscriptionListener(
            item_name,
            self._candle_queue,
            setup_complete_check=lambda: self._setup_complete,
        )
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

    def _subscribe_tick_direct(self) -> None:
        """Subscribe to CHART:{epic}:TICK for raw tick delivery in tick mode.

        Creates a _DirectTickListener that enqueues raw tick dicts onto the
        shared queue with type='tick'. The worker thread dispatches these to
        the on_tick callback. This subscription is created in addition to the
        candle subscription — not instead of it.
        """
        item_name = f"CHART:{self._epic}:TICK"
        listener = _DirectTickListener(
            item_name,
            self._candle_queue,
            setup_complete_check=lambda: self._setup_complete,
        )
        sub = _make_tick_subscription(self._epic, listener)
        self._stream_svc.subscribe(sub)
        logger.info(f"Direct tick subscription created: {item_name}")

    def _subscribe_tick_fallback(self) -> None:
        """Subscribe to CHART:{epic}:TICK and aggregate ticks into candles."""
        item_name = f"CHART:{self._epic}:TICK"
        self._using_tick_fallback = True
        aggregator = TickAggregator(
            resolution_minutes=_resolution_to_minutes(self._resolution),
            on_candle=lambda candle: self._candle_queue.put(candle),
        )
        listener = _TickListener(
            item_name,
            aggregator,
            candle_queue=self._candle_queue,
            setup_complete_check=lambda: self._setup_complete,
        )
        sub = _make_tick_subscription(self._epic, listener)

        self._stream_svc.subscribe(sub)
        self._active_subscription = sub
        logger.info(f"Tick fallback subscription active: {item_name}")

    def _worker_loop(
        self,
        on_candle: Callable[[dict], None],
        on_tick: Optional[Callable[[dict], None]] = None,
        on_reconnect: Optional[Callable[[], None]] = None,
    ) -> None:
        """Dequeue items and dispatch to on_candle, on_tick, or on_reconnect by item type.

        Runs on a dedicated worker thread. Blocks on queue.get() with a
        short timeout so the stop event is checked regularly. After the stop
        event is set, any remaining items in the queue are drained and
        delivered before the loop exits — this ensures in-flight items are
        not silently dropped at shutdown.

        Dispatch rules:
        - ``item.get("type", "candle") == "candle"`` → ``on_candle(item)``
        - ``item.get("type", "candle") == "tick"`` and on_tick is not None
          → ``on_tick(item)``
        - ``item.get("type") == "reconnect"`` and on_reconnect is not None
          → ``on_reconnect()``
        - Items without a ``"type"`` key default to ``"candle"`` for
          backward compatibility with any producer that predates this change.

        Args:
            on_candle: Callback to invoke for each completed candle.
            on_tick: Optional callback to invoke for each raw tick. When None,
                tick items are silently discarded (candle mode).
            on_reconnect: Optional callback invoked when a reconnect sentinel
                is dequeued. When None, reconnect sentinels are logged as a
                warning and discarded.
        """
        logger.debug("Streaming worker thread started.")

        def _dispatch(item: dict) -> None:
            item_type = item.get("type", "candle")
            if item_type == "tick":
                if on_tick is not None:
                    on_tick(item)
            elif item_type == "reconnect":
                if on_reconnect is not None:
                    on_reconnect()
                else:
                    logger.warning(
                        "Reconnect sentinel received but no on_reconnect callback is registered."
                    )
            else:
                on_candle(item)

        while not self._stop_event.is_set():
            try:
                item = self._candle_queue.get(timeout=0.05)
                try:
                    _dispatch(item)
                    # Update last-data timestamp after every successful dispatch (GAP-3).
                    self._last_data_ts = time.monotonic()
                except Exception as e:
                    logger.error(f"Error in callback: {e}", exc_info=True)
                finally:
                    self._candle_queue.task_done()
            except queue.Empty:
                # GAP-3: Candle watchdog — detect data drought and enqueue reconnect.
                elapsed = time.monotonic() - self._last_data_ts
                if elapsed > self._watchdog_timeout_s:
                    logger.warning(
                        f"Candle watchdog triggered: no data for {elapsed:.0f}s "
                        f"(threshold={self._watchdog_timeout_s:.0f}s). "
                        "Enqueuing reconnect sentinel."
                    )
                    self._candle_queue.put({"type": "reconnect"})
                    self._last_data_ts = time.monotonic()
                continue

        # Drain remaining items so in-flight items are not lost on shutdown.
        # Cap at _DRAIN_BUDGET items to prevent indefinite blocking under
        # thundering-herd conditions; any remaining items are discarded.
        _DRAIN_BUDGET = 50
        drained = 0
        dropped = 0
        while drained < _DRAIN_BUDGET:
            try:
                item = self._candle_queue.get(timeout=0.01)
                drained += 1
                try:
                    _dispatch(item)
                except Exception as e:
                    logger.error(f"Error in callback during drain: {e}", exc_info=True)
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
                f"dropped {dropped} item(s) from queue."
            )

        logger.debug("Streaming worker thread stopped.")

    def _restart_streaming(
        self,
        on_candle: Callable[[dict], None],
        on_tick: Optional[Callable[[dict], None]] = None,
    ) -> None:
        """Cycle the Lightstreamer service without stopping the worker thread.

        Intended to be called FROM the worker thread (inside the on_reconnect
        callback). Disconnects the current Lightstreamer service, creates a new
        one, re-subscribes, and attaches a fresh _ConnectionListener. The
        worker thread continues polling the shared queue after this method
        returns.

        Does NOT touch ``_worker`` or ``_stop_event``. The worker thread stays
        alive throughout — only the LS service is cycled.

        Args:
            on_candle: Candle delivery callback for the new subscription.
            on_tick: Optional tick delivery callback. When not None and the
                native subscription succeeds, a direct tick subscription is
                also created.

        Raises:
            Exception: Any exception from ``create_session()`` or subscription
                setup is re-raised so the caller (reconnect backoff loop) can
                retry.
            SystemExit: ``trading_ig`` calls ``sys.exit(1)`` on auth failure
                inside ``create_session()``. The caller must catch this.
        """
        # Reset setup flag at entry — listeners created during this restart will not
        # enqueue reconnect sentinels until setup completes (GAP-2).
        self._setup_complete = False

        # Disconnect old service — best-effort, ignore errors
        if self._stream_svc is not None:
            try:
                self._stream_svc.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting old stream service: {e}")
            self._stream_svc = None

        # Clear subscription state so _subscribe_native starts clean
        self._active_subscription = None
        self._using_tick_fallback = False

        # Create a new Lightstreamer session — may raise or sys.exit on auth failure
        self._stream_svc = IGStreamService(self._ig_service)
        self._stream_svc.create_session()

        # Subscribe to candles (and optional direct ticks)
        self._subscribe_native(on_candle)
        if on_tick is not None and not self._using_tick_fallback:
            self._subscribe_tick_direct()

        # Attach a fresh ConnectionListener with _fired=False so the new session
        # can detect terminal disconnects again.
        self._stream_svc.add_client_listener(_ConnectionListener(self._candle_queue))

        # Mark setup complete after successful re-subscription (GAP-2).
        self._setup_complete = True
        logger.info("_restart_streaming completed — new Lightstreamer session active.")

    def is_worker_alive(self) -> bool:
        """Check whether the internal worker thread is running.

        Returns:
            True if the worker thread exists and is alive, False otherwise.
        """
        return self._worker is not None and self._worker.is_alive()

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
        self._using_tick_fallback = False
        self._setup_complete = False
        logger.info("IGStreamingClient stopped.")
