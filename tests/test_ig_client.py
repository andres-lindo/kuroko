"""Tests for IGClient — retry logic, cache paths, and broker API wrappers.

Covers _safe_api_call retry loop, token-refresh on transient auth errors,
get_candles cache-miss/hit/incremental paths, _remove_incomplete_candle,
open_position, close_position, get_open_positions, and get_account_summary.
All network calls and time.sleep are mocked.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest
from requests.exceptions import ConnectionError, RequestException
from trading_ig.rest import IGException

from ig_client import IGClient

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

_ENV = {
    "ig_username": "test_user",
    "ig_password": "test_pass",
    "ig_api_key": "test_key",
    "ig_acc_number": "ACC123",
    "ig_acc_type": "DEMO",
}


def _make_client(tmp_path, mock_svc=None):
    """Build an IGClient with mocked IGService and a tmp cache dir.

    Args:
        tmp_path: pytest tmp_path fixture for the cache directory.
        mock_svc: Optional pre-configured MagicMock. A fresh one is created
            when not provided.

    Returns:
        Tuple of (IGClient, mock_svc).
    """
    if mock_svc is None:
        mock_svc = MagicMock()
        mock_svc.create_session.return_value = {
            "accountType": "DEMO",
            "accountId": "ACC123",
        }

    with patch("ig_client.IGService", return_value=mock_svc):
        with patch.dict("os.environ", _ENV):
            client = IGClient()

    # Redirect cache to tmp_path so tests do not touch the real cache dir
    client.cache_dir = tmp_path
    return client, mock_svc


def _make_ohlc_df(n=5, base_price=19000.0, freq="15min"):
    """Build a minimal OHLC DataFrame with a DatetimeIndex.

    Args:
        n: Number of rows to generate.
        base_price: Starting close price.
        freq: pandas frequency string for the DatetimeIndex.

    Returns:
        DataFrame with Open, High, Low, Close columns.
    """
    now = datetime.now().replace(second=0, microsecond=0)
    # Place all candles well in the past so none match 'now'
    start = now - timedelta(minutes=15 * (n + 2))
    index = pd.date_range(start=start, periods=n, freq=freq)
    data = {
        "Open": [base_price] * n,
        "High": [base_price + 10] * n,
        "Low": [base_price - 10] * n,
        "Close": [base_price + i for i in range(n)],
    }
    return pd.DataFrame(data, index=index)


# --------------------------------------------------------------------------- #
# _safe_api_call                                                               #
# --------------------------------------------------------------------------- #


class TestSafeApiCall:
    """Tests for IGClient._safe_api_call retry and session-refresh logic."""

    def test_success_on_first_attempt_returns_value(self, tmp_path):
        client, _ = _make_client(tmp_path)
        func = MagicMock(return_value={"ok": True})

        result = client._safe_api_call(func, "arg1", kwarg="val")

        assert result == {"ok": True}
        func.assert_called_once_with("arg1", kwarg="val")

    def test_retries_on_connection_error_then_succeeds(self, tmp_path):
        client, _ = _make_client(tmp_path)
        func = MagicMock(side_effect=[ConnectionError("timeout"), {"ok": True}])

        with patch("ig_client.sleep") as mock_sleep:
            result = client._safe_api_call(func, max_retries=3)

        assert result == {"ok": True}
        assert func.call_count == 2
        mock_sleep.assert_called_once_with(1)  # 2^0 = 1 second

    def test_retries_on_request_exception_then_succeeds(self, tmp_path):
        client, _ = _make_client(tmp_path)
        func = MagicMock(side_effect=[RequestException("network"), {"ok": True}])

        with patch("ig_client.sleep"):
            result = client._safe_api_call(func, max_retries=3)

        assert result == {"ok": True}
        assert func.call_count == 2

    def test_raises_after_max_retries_exceeded(self, tmp_path):
        client, _ = _make_client(tmp_path)
        func = MagicMock(side_effect=ConnectionError("always fails"))

        with patch("ig_client.sleep"):
            with pytest.raises(ConnectionError):
                client._safe_api_call(func, max_retries=3)

        assert func.call_count == 3

    def test_exponential_backoff_sleep_values(self, tmp_path):
        client, _ = _make_client(tmp_path)
        func = MagicMock(side_effect=ConnectionError("fail"))

        with patch("ig_client.sleep") as mock_sleep:
            with pytest.raises(ConnectionError):
                client._safe_api_call(func, max_retries=3)

        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_calls == [1, 2]  # 2^0, 2^1

    def test_token_refresh_triggered_on_token_error(self, tmp_path):
        client, mock_svc = _make_client(tmp_path)
        token_error = IGException("error: token expired")
        func = MagicMock(side_effect=[token_error, {"ok": True}])

        with patch("ig_client.sleep"):
            result = client._safe_api_call(func, max_retries=3)

        assert result == {"ok": True}
        # create_session was called once on init and once on token refresh
        assert mock_svc.create_session.call_count == 2

    def test_json_decode_error_does_not_trigger_token_refresh(self, tmp_path):
        client, mock_svc = _make_client(tmp_path)
        init_call_count = mock_svc.create_session.call_count
        func = MagicMock(side_effect=json.JSONDecodeError("expecting value", "", 0))

        with patch("ig_client.sleep"):
            with pytest.raises(json.JSONDecodeError):
                client._safe_api_call(func, max_retries=1)

        # create_session count must not have increased beyond init
        assert mock_svc.create_session.call_count == init_call_count

    def test_unexpected_exception_reraises_immediately(self, tmp_path):
        client, _ = _make_client(tmp_path)
        func = MagicMock(side_effect=ValueError("bad input"))

        with pytest.raises(ValueError, match="bad input"):
            client._safe_api_call(func)

        assert func.call_count == 1


# --------------------------------------------------------------------------- #
# _remove_incomplete_candle                                                    #
# --------------------------------------------------------------------------- #


class TestRemoveIncompleteCandle:
    """Tests for IGClient._remove_incomplete_candle."""

    def test_empty_dataframe_returned_unchanged(self, tmp_path):
        client, _ = _make_client(tmp_path)
        df = pd.DataFrame()

        result = client._remove_incomplete_candle(df, "15min")

        assert result.empty

    def test_last_candle_matching_now_is_dropped(self, tmp_path):
        client, _ = _make_client(tmp_path)
        now = datetime.now().replace(second=0, microsecond=0)
        # Place last candle exactly at the current minute
        past = now - timedelta(minutes=15)
        index = pd.DatetimeIndex([past, now])
        df = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [110, 111],
                "Low": [90, 91],
                "Close": [105, 106],
            },
            index=index,
        )

        result = client._remove_incomplete_candle(df, "15min")

        assert len(result) == 1
        assert result.index[-1] == past

    def test_last_candle_matching_previous_period_is_kept(self, tmp_path):
        client, _ = _make_client(tmp_path)
        now = datetime.now().replace(second=0, microsecond=0)
        expected_closed = now - timedelta(minutes=15)
        earlier = expected_closed - timedelta(minutes=15)
        index = pd.DatetimeIndex([earlier, expected_closed])
        df = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [110, 111],
                "Low": [90, 91],
                "Close": [105, 106],
            },
            index=index,
        )

        result = client._remove_incomplete_candle(df, "15min")

        assert len(result) == 2

    def test_unexpected_timestamp_is_returned_unchanged(self, tmp_path):
        client, _ = _make_client(tmp_path)
        now = datetime.now().replace(second=0, microsecond=0)
        # Timestamp 7 minutes ago — not 'now' and not 'now-15min'
        odd_time = now - timedelta(minutes=7)
        earlier = odd_time - timedelta(minutes=15)
        index = pd.DatetimeIndex([earlier, odd_time])
        df = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [110, 111],
                "Low": [90, 91],
                "Close": [105, 106],
            },
            index=index,
        )

        result = client._remove_incomplete_candle(df, "15min")

        assert len(result) == 2


# --------------------------------------------------------------------------- #
# get_candles                                                                  #
# --------------------------------------------------------------------------- #


class TestGetCandles:
    """Tests for IGClient.get_candles cache-miss, cache-hit, and incremental paths."""

    def _make_api_response(self, df):
        """Wrap a DataFrame in the structure returned by fetch_historical_prices.

        Args:
            df: OHLC DataFrame to wrap.

        Returns:
            Dict matching the trading_ig response shape.
        """
        return {"prices": {"bid": df}}

    def test_cache_miss_fetches_from_api_and_stores(self, tmp_path):
        client, mock_svc = _make_client(tmp_path)
        df = _make_ohlc_df(n=5)
        mock_svc.fetch_historical_prices_by_epic_and_num_points.return_value = (
            self._make_api_response(df)
        )

        result = client.get_candles("EPIC.TEST", "15min", num_points=5)

        assert result is not None
        assert not result.empty
        mock_svc.fetch_historical_prices_by_epic_and_num_points.assert_called_once()

    def test_cache_miss_persists_to_parquet(self, tmp_path):
        client, mock_svc = _make_client(tmp_path)
        df = _make_ohlc_df(n=5)
        mock_svc.fetch_historical_prices_by_epic_and_num_points.return_value = (
            self._make_api_response(df)
        )

        client.get_candles("EPIC.TEST", "15min", num_points=5)

        parquet_file = tmp_path / "EPIC.TEST_15min.parquet"
        assert parquet_file.exists()

    def test_cache_hit_skips_api_call(self, tmp_path):
        client, mock_svc = _make_client(tmp_path)
        df = _make_ohlc_df(n=5)
        # Pre-populate the in-memory cache
        client.candles_cache["EPIC.TEST_15min"] = df

        result = client.get_candles("EPIC.TEST", "15min", num_points=5)

        assert result is not None
        # fetch_historical should only be called for the incremental update (3 candles)
        # but NOT for the initial full load
        args_list = (
            mock_svc.fetch_historical_prices_by_epic_and_num_points.call_args_list
        )
        # Ensure no call with num_points+1 (=6) was made — that's the initial-load signature
        for c in args_list:
            positional = c.args
            # The initial load passes num_points+1 as the third positional arg
            if len(positional) >= 3:
                assert (
                    positional[2] != 6
                ), "Initial full-load must not be triggered on cache hit"

    def test_parquet_cache_loaded_on_startup(self, tmp_path):
        client, mock_svc = _make_client(tmp_path)
        df = _make_ohlc_df(n=5)
        # Write parquet directly to simulate a previous run
        df.to_parquet(tmp_path / "EPIC.DISK_15min.parquet")

        # Incremental fetch returns the same df (to satisfy the update merge)
        mock_svc.fetch_historical_prices_by_epic_and_num_points.return_value = (
            self._make_api_response(df)
        )

        result = client.get_candles("EPIC.DISK", "15min", num_points=5)

        assert result is not None
        # Cache should now be populated from disk
        assert "EPIC.DISK_15min" in client.candles_cache

    def test_initial_load_failure_returns_none(self, tmp_path):
        client, mock_svc = _make_client(tmp_path)
        mock_svc.fetch_historical_prices_by_epic_and_num_points.side_effect = (
            ConnectionError("network down")
        )

        with patch("ig_client.sleep"):
            result = client.get_candles("EPIC.FAIL", "15min", num_points=5)

        assert result is None

    def test_incremental_update_merges_new_candles(self, tmp_path):
        client, mock_svc = _make_client(tmp_path)
        base_df = _make_ohlc_df(n=5)
        client.candles_cache["EPIC.MERGE_15min"] = base_df

        # The incremental fetch returns one new candle beyond the current cache
        now = datetime.now().replace(second=0, microsecond=0)
        new_ts = base_df.index[-1] + timedelta(minutes=15)
        # Ensure new candle is not 'now' (so _remove_incomplete_candle keeps it)
        if new_ts >= now:
            new_ts = base_df.index[-1] - timedelta(minutes=15)
        new_candle = pd.DataFrame(
            {"Open": [19100], "High": [19120], "Low": [19080], "Close": [19110]},
            index=pd.DatetimeIndex([new_ts]),
        )
        mock_svc.fetch_historical_prices_by_epic_and_num_points.return_value = (
            self._make_api_response(new_candle)
        )

        result = client.get_candles("EPIC.MERGE", "15min", num_points=5)

        assert result is not None


# --------------------------------------------------------------------------- #
# open_position                                                                #
# --------------------------------------------------------------------------- #


class TestOpenPosition:
    """Tests for IGClient.open_position."""

    def test_open_position_calls_create_open_position(self, tmp_path):
        client, mock_svc = _make_client(tmp_path)
        mock_svc.create_open_position.return_value = {
            "status": "OPEN",
            "dealReference": "REF1",
        }

        result = client.open_position("EPIC.TEST", size=0.5, side="BUY")

        assert result["dealReference"] == "REF1"
        mock_svc.create_open_position.assert_called_once()
        call_kwargs = mock_svc.create_open_position.call_args.kwargs
        assert call_kwargs["direction"] == "BUY"
        assert call_kwargs["epic"] == "EPIC.TEST"
        assert call_kwargs["size"] == 0.5

    def test_open_position_passes_stop_and_limit(self, tmp_path):
        client, mock_svc = _make_client(tmp_path)
        mock_svc.create_open_position.return_value = {}

        client.open_position("EPIC.TEST", size=0.13, side="BUY", stop=50.0, limit=100.0)

        call_kwargs = mock_svc.create_open_position.call_args.kwargs
        assert call_kwargs["stop_distance"] == 50.0
        assert call_kwargs["limit_distance"] == 100.0


# --------------------------------------------------------------------------- #
# close_position                                                               #
# --------------------------------------------------------------------------- #


class TestClosePosition:
    """Tests for IGClient.close_position."""

    def test_close_position_calls_close_open_position(self, tmp_path):
        client, mock_svc = _make_client(tmp_path)
        mock_svc.close_open_position.return_value = {"status": "CLOSED"}

        result = client.close_position("DEAL123", side="SELL", size=0.5)

        assert result["status"] == "CLOSED"
        mock_svc.close_open_position.assert_called_once()
        call_kwargs = mock_svc.close_open_position.call_args.kwargs
        assert call_kwargs["deal_id"] == "DEAL123"
        assert call_kwargs["direction"] == "SELL"
        assert call_kwargs["size"] == 0.5


# --------------------------------------------------------------------------- #
# get_open_positions                                                           #
# --------------------------------------------------------------------------- #


class TestGetOpenPositions:
    """Tests for IGClient.get_open_positions."""

    def test_returns_list_of_position_dicts(self, tmp_path):
        client, mock_svc = _make_client(tmp_path)
        positions_df = pd.DataFrame(
            [
                {
                    "dealReference": "REF1",
                    "dealId": "DEAL1",
                    "level": 19000.0,
                    "size": 0.13,
                    "createdDate": "2026-05-28T10:00:00",
                    "direction": "BUY",
                }
            ]
        )
        mock_svc.fetch_open_positions.return_value = positions_df

        result = client.get_open_positions()

        assert len(result) == 1
        assert result[0]["dealId"] == "DEAL1"
        assert result[0]["direction"] == "BUY"

    def test_returns_empty_list_when_no_positions(self, tmp_path):
        client, mock_svc = _make_client(tmp_path)
        mock_svc.fetch_open_positions.return_value = pd.DataFrame()

        result = client.get_open_positions()

        assert result == []

    def test_returns_empty_list_on_api_error(self, tmp_path):
        client, mock_svc = _make_client(tmp_path)
        mock_svc.fetch_open_positions.side_effect = ConnectionError("network fail")

        with patch("ig_client.sleep"):
            result = client.get_open_positions()

        assert result == []

    def test_returns_empty_list_on_unexpected_schema(self, tmp_path):
        client, mock_svc = _make_client(tmp_path)
        # DataFrame with wrong column names
        bad_df = pd.DataFrame([{"wrong_col": "x"}])
        mock_svc.fetch_open_positions.return_value = bad_df

        result = client.get_open_positions()

        assert result == []


# --------------------------------------------------------------------------- #
# ig_service property — REQ-14                                                 #
# --------------------------------------------------------------------------- #


class TestIGServiceProperty:
    """Tests for the ig_service property on IGClient."""

    def test_ig_service_property_returns_underlying_svc(self, tmp_path):
        client, _ = _make_client(tmp_path)

        assert client.ig_service is client._svc

    def test_ig_service_property_does_not_re_authenticate(self, tmp_path):
        client, mock_svc = _make_client(tmp_path)
        initial_call_count = mock_svc.create_session.call_count

        _ = client.ig_service

        assert mock_svc.create_session.call_count == initial_call_count

    def test_ig_service_returns_same_instance_on_multiple_accesses(self, tmp_path):
        """ig_service must return the same object reference on repeated access."""
        client, _ = _make_client(tmp_path)

        assert client.ig_service is client.ig_service


# --------------------------------------------------------------------------- #
# get_account_summary                                                          #
# --------------------------------------------------------------------------- #


class TestGetAccountSummary:
    """Tests for IGClient.get_account_summary."""

    def test_returns_account_dict_for_matching_account(self, tmp_path):
        """get_account_summary returns a dict keyed by accountId etc."""
        client, mock_svc = _make_client(tmp_path)
        accounts_df = pd.DataFrame(
            [
                {
                    "accountId": "ACC123",
                    "balance": 4000.0,
                    "deposit": 500.0,
                    "profitLoss": 50.0,
                    "available": 3500.0,
                }
            ]
        )
        mock_svc.fetch_accounts.return_value = accounts_df

        result = client.get_account_summary()

        assert result["accountId"] == "ACC123"
        assert result["balance"] == 4000.0
        assert result["profitLoss"] == 50.0

    def test_returns_empty_dict_on_api_error(self, tmp_path):
        """get_account_summary returns {} and does not raise on API error."""
        client, mock_svc = _make_client(tmp_path)
        mock_svc.fetch_accounts.side_effect = ConnectionError("network fail")

        with patch("ig_client.sleep"):
            result = client.get_account_summary()

        assert result == {}
