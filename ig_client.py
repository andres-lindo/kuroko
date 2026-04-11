"""IG Markets API client with retry logic and candle caching.

Wraps the trading_ig library to provide authenticated REST API access,
exponential-backoff retries, and a two-tier candle cache (in-memory +
parquet on disk) that minimises API calls across restarts.
"""
import json
import os
import logging
import pandas as pd
from pathlib import Path
from time import sleep
from datetime import datetime, timedelta
from requests.exceptions import ConnectionError, RequestException

from trading_ig import IGService
from trading_ig.rest import IGException

log = logging.getLogger(__name__)


class IGClient:
    """Wrapper around the IG Markets REST API.

    Handles authentication, session refresh, retry logic on transient
    errors, and candle caching (in-memory dict backed by parquet files).

    Attributes:
        accountId: The IG account identifier loaded from credentials.
        candles_cache: In-memory dict keyed by '{epic}_{resolution}'.
        cache_dir: Path to the directory used for parquet persistence.
    """

    def __init__(self):
        """Authenticate with IG Markets and initialise the candle cache.

        Reads credentials from environment variables and creates a live
        session. Raises RuntimeError if any required credential is missing.

        Raises:
            RuntimeError: If ig_username, ig_password, or ig_api_key are
                not set in the environment.
        """
        user = os.getenv("ig_username")
        pwd  = os.getenv("ig_password")
        key  = os.getenv("ig_api_key")
        accn = os.getenv("ig_acc_number")
        acc_type = os.getenv("ig_acc_type")  # DEMO | LIVE

        if not (user and pwd and key):
            raise RuntimeError("Missing IG credentials (username/password/api_key).")

        self._svc = IGService(user, pwd, key, acc_type, accn)
        self._svc.create_session()
        self.accountId = accn

        # Simple in-memory cache keyed by '{epic}_{resolution}'
        self.candles_cache = {}  # {epic_res: DataFrame}

        # Directory for parquet persistence across restarts
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)

    def _safe_api_call(self, func, *args, max_retries=3, **kwargs):
        """Execute an API call with retry and session-refresh logic.

        Retries up to max_retries times on connection or IG API errors.
        If the error message indicates an expired token, the session is
        refreshed before retrying. Waits use exponential backoff (1, 2, 4 s).

        Args:
            func: Callable from self._svc to invoke.
            *args: Positional arguments forwarded to func.
            max_retries: Maximum number of attempts before raising.
            **kwargs: Keyword arguments forwarded to func.

        Returns:
            The return value of func on success.

        Raises:
            ConnectionError: If all retry attempts are exhausted.
            RequestException: If all retry attempts are exhausted.
            IGException: If all retry attempts are exhausted.
            json.JSONDecodeError: If all retry attempts are exhausted on an
                empty/malformed HTTP body. Session refresh is NOT triggered
                for this error type — it is a content error, not an auth error.
            Exception: For any unexpected error not covered by the retry logic.
        """
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (
                ConnectionError,
                RequestException,
                IGException,
                json.JSONDecodeError,
            ) as e:
                # Session refresh is only valid for connection/auth errors.
                # json.JSONDecodeError indicates an empty body from the server
                # (e.g. during a maintenance window) — refreshing the session
                # would be incorrect and could trigger spurious re-auth.
                if not isinstance(e, json.JSONDecodeError):
                    if "token" in str(e).lower():
                        log.warning("Token expired. Refreshing session...")
                        try:
                            self._svc.create_session()
                            log.info("Session refreshed successfully.")
                            continue  # Retry with the new session
                        except Exception as refresh_error:
                            log.error(
                                "Error refreshing session: %s", refresh_error
                            )

                log.debug(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")

                if attempt < max_retries - 1:
                    # Exponential backoff: 1 s, 2 s, 4 s between attempts
                    wait_time = 2 ** attempt
                    log.debug(f"Waiting {wait_time} seconds before retrying...")
                    sleep(wait_time)
                else:
                    log.error(f"Failed after {max_retries} attempts.")
                    raise
            except Exception as e:
                log.error(f"Unexpected error: {e}")
                raise

    def _remove_incomplete_candle(self, df: pd.DataFrame, resolution: str) -> pd.DataFrame:
        """Strip the current (incomplete) candle from a price DataFrame.

        A candle whose timestamp matches the current execution minute is still
        forming and must not be used for signal calculation. A candle whose
        timestamp equals (now - timeframe) is closed and is safe to use.

        Args:
            df: Raw OHLC DataFrame indexed by timestamp.
            resolution: Candle resolution string, e.g. '15min'.

        Returns:
            DataFrame with the incomplete candle removed if detected,
            otherwise the original DataFrame unchanged.
        """
        if df.empty:
            return df

        # Normalise execution time to the current minute boundary
        exec_time = datetime.now().replace(second=0, microsecond=0)
        last_candle_time = pd.to_datetime(df.index[-1]).replace(second=0, microsecond=0)

        # Derive timeframe width from resolution string
        timeframe_minutes = int(resolution.replace("min", "")) if "min" in resolution else 60
        expected_complete_time = exec_time - timedelta(minutes=timeframe_minutes)

        # Last candle == now → it is still forming → drop it
        if last_candle_time == exec_time:
            log.debug(f"Dropping incomplete candle: {last_candle_time}")
            return df.iloc[:-1]
        # Last candle == now - timeframe → it is closed → keep it
        elif last_candle_time == expected_complete_time:
            log.debug(f"Last candle is complete: {last_candle_time}")
            return df
        else:
            log.debug(f"Unexpected timestamp. Last candle: {last_candle_time}, expected: {expected_complete_time}")
            return df

    def get_candles(self, epic: str, res: str, num_points: int = 200) -> pd.DataFrame:
        """Return OHLC candles for the given epic, using cached data when possible.

        On the first call for an epic/resolution pair, fetches num_points + 1
        candles from the API (the extra candle guards against the incomplete
        current bar being the only one). Subsequent calls fetch only 3 candles
        and merge them into the cache, so the API is queried minimally.

        Args:
            epic: Instrument identifier (e.g. 'IX.D.NASDAQ.IFMM.IP').
            res: Candle resolution string (e.g. '15min').
            num_points: Number of candles to return.

        Returns:
            DataFrame with columns Open/High/Low/Close and a datetime index,
            containing at most num_points rows. Returns None when the initial
            load fails after all retries — callers must handle None explicitly
            and treat it as "no data available for this tick."
        """
        cache_key = f"{epic}_{res}"

        # Try to warm the cache from disk on first access
        if cache_key not in self.candles_cache:
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            if cache_file.exists():
                try:
                    df = pd.read_parquet(cache_file)
                    self.candles_cache[cache_key] = df
                    log.info(f"Cache loaded from disk for {epic} {res}")
                except Exception as e:
                    log.warning(f"Error loading cache from disk: {e}")

        # Full initial load when cache is absent or empty
        if cache_key not in self.candles_cache or self.candles_cache[cache_key].empty:
            log.info(f"Initial load: fetching {num_points} candles for {epic} {res}")

            try:
                resp = self._safe_api_call(
                    self._svc.fetch_historical_prices_by_epic_and_num_points,
                    epic,
                    res,
                    num_points + 1,  # one extra to guard against an incomplete bar
                )
                df = resp["prices"]["bid"]

                # Drop the current (incomplete) candle if present
                df = self._remove_incomplete_candle(df, res)

                self.candles_cache[cache_key] = df

                # Persist to disk so the cache survives restarts
                try:
                    df.to_parquet(self.cache_dir / f"{cache_key}.parquet")
                except Exception as e:
                    log.warning(f"Error saving cache to disk: {e}")

                return df.tail(num_points).copy()
            except Exception as e:
                log.error("Initial candle load failed: %s", e, exc_info=True)
                return None

        # Incremental update: fetch only the 3 most recent candles
        existing_df = self.candles_cache[cache_key]

        log.debug(f"Checking for updates for {epic} {res}")

        try:
            resp = self._safe_api_call(
                self._svc.fetch_historical_prices_by_epic_and_num_points,
                epic, res, 3
            )
            new_df = resp['prices']['bid']

            # Drop the incomplete candle from the freshly fetched slice
            new_df = self._remove_incomplete_candle(new_df, res)

            # Only merge when the slice contains valid OHLC rows
            if not new_df.empty and not new_df.isna().all().all():
                # Merge new rows into the cache, deduplicate (keep last seen),
                # drop all-NaN rows, sort chronologically, and trim to num_points
                combined_df = pd.concat([existing_df, new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df = combined_df.dropna(how='all')
                combined_df = combined_df.sort_index().tail(num_points)

                # Persist only when the cache has actually changed
                if len(combined_df) > len(existing_df) or not combined_df.equals(existing_df.tail(num_points)):
                    log.info(f"Updating cache for {epic} {res}")
                    self.candles_cache[cache_key] = combined_df
                    try:
                        combined_df.to_parquet(self.cache_dir / f"{cache_key}.parquet")
                    except Exception as e:
                        log.warning(f"Error saving updated cache: {e}")

                    return combined_df.copy()

            return existing_df.tail(num_points).copy()

        except Exception as e:
            log.error(f"Error fetching update, returning existing cache: {e}")
            return existing_df.tail(num_points).copy()

    def clear_cache(self):
        """Clear the in-memory and on-disk candle cache."""
        self.candles_cache.clear()
        for file in self.cache_dir.glob("*.parquet"):
            try:
                file.unlink()
            except Exception as e:
                log.warning(f"Error deleting cache file {file}: {e}")
        log.info("Cache cleared.")

    def get_open_positions(self):
        """Fetch all currently open positions for the account.

        Returns:
            List of dicts with keys: dealReference, dealId, level, size,
            createdDate, direction. Returns an empty list on error or if
            there are no open positions.
        """
        try:
            open_positions = self._safe_api_call(self._svc.fetch_open_positions)

            if open_positions.empty:
                return []

            return open_positions[["dealReference", "dealId", "level", "size", "createdDate", "direction"]].to_dict(orient="records")
        except Exception as e:
            log.error(f"Error fetching open positions: {e}")
            return []

    def get_account_summary(self):
        """Return balance, deposit, P&L, and available margin for the account.

        Returns:
            Dict with keys: accountId, balance, deposit, profitLoss,
            available. Returns an empty dict on error.
        """
        try:
            accounts = self._safe_api_call(self._svc.fetch_accounts)
            cols = ["accountId", "balance", "deposit", "profitLoss", "available"]

            return accounts.loc[
                accounts["accountId"] == self.accountId, cols
            ].to_dict(orient="records")[0]
        except Exception as e:
            log.error(f"Error fetching account summary: {e}")
            return {}

    def open_position(self, epic: str, size: float, side: str, currency: str = 'USD', stop: float = None, limit: float = None):
        """Open a new market-order position.

        Args:
            epic: Instrument identifier (e.g. 'IX.D.NASDAQ.IFMM.IP').
            size: Number of contracts to trade.
            side: Trade direction, either 'BUY' or 'SELL'.
            currency: Currency code for the deal (default 'USD').
            stop: Stop-loss distance in points (optional).
            limit: Take-profit distance in points (optional).

        Returns:
            API response dict from trading_ig containing dealReference
            and confirmation status.
        """
        return self._safe_api_call(
            self._svc.create_open_position,
            currency_code=currency,
            direction=side,
            epic=epic,
            order_type='MARKET',
            expiry='-',
            force_open='true',
            guaranteed_stop='false',
            size=float(size),
            level=None,
            limit_distance=limit,
            limit_level=None,
            quote_id=None,
            stop_level=None,
            stop_distance=stop,
            trailing_stop=None,
            trailing_stop_increment=None
        )

    def update_position(self, dealid: str, stop: float = None, limit: float = None):
        """Update the stop-loss and/or take-profit level on an open position.

        Args:
            dealid: Deal identifier for the position to update.
            stop: New stop-loss price level (absolute, not distance).
            limit: New take-profit price level (absolute, not distance).

        Returns:
            API response dict from trading_ig.
        """
        return self._safe_api_call(
            self._svc.update_open_position,
            limit_level=limit,
            stop_level=stop,
            deal_id=dealid
        )

    def close_position(self, deal_id: str, side: str, size: float):
        """Close an open position with a market order.

        Args:
            deal_id: Deal identifier for the position to close.
            side: Closing direction (opposite of the open direction).
            size: Number of contracts to close.

        Returns:
            API response dict from trading_ig.
        """
        return self._safe_api_call(
            self._svc.close_open_position,
            deal_id=deal_id,
            direction=side,
            epic=None,
            expiry='-',
            level=None,
            order_type='MARKET',
            quote_id=None,
            size=float(size),
        )
