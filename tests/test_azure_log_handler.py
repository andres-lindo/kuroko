"""Tests for logging infrastructure — AzureBlobHandler and logging_setup.

AzureBlobHandler tests cover: init, emit, midnight rotation, error fallback.
All Azure SDK calls are mocked at the module boundary using
patch('azure_log_handler.BlobServiceClient.from_connection_string').
The date.today() calls are patched at 'azure_log_handler.date' to control
midnight rotation without real time passage.

logging_setup tests cover: config loading (valid, missing, malformed),
handler attachment per log_type combinations, formatter selection,
retention cleanup, and handler clearing on setup_logging() re-entry.
"""

import json
import logging
import os
import sys
from datetime import date, timedelta
from logging.handlers import TimedRotatingFileHandler
from unittest.mock import MagicMock, call, patch

import pytest

from azure_log_handler import AzureBlobHandler
from logging_setup import (
    _DEFAULTS,
    _cleanup_old_logs,
    load_app_config,
    setup_logging,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

_CONN_STR = "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=abc123;"
_CONTAINER = "test-logs"
_BLOB_NAME = "kuroko"


def _make_handler(conn_str=_CONN_STR, blob_name=_BLOB_NAME, container=_CONTAINER):
    """Build an AzureBlobHandler with a fully mocked BlobServiceClient chain.

    Patches BlobServiceClient.from_connection_string so no real Azure calls
    are made. Returns (handler, mock_blob_service_client, mock_blob_client).

    Args:
        conn_str: Azure connection string (fake value for tests).
        blob_name: Base blob name prefix.
        container: Target container name.

    Returns:
        Tuple of (AzureBlobHandler, mock_blob_service, mock_blob_client).
    """
    mock_blob_service = MagicMock()
    mock_blob_client = MagicMock()

    # get_blob_properties raises to trigger create_append_blob
    mock_blob_client.get_blob_properties.side_effect = Exception("BlobNotFound")
    mock_blob_service.get_blob_client.return_value = mock_blob_client

    with patch(
        "azure_log_handler.BlobServiceClient.from_connection_string",
        return_value=mock_blob_service,
    ):
        handler = AzureBlobHandler(conn_str, blob_name, container)

    return handler, mock_blob_service, mock_blob_client


def _make_log_record(msg="test message", level=logging.INFO):
    """Build a minimal LogRecord for emit() testing.

    Args:
        msg: Log message string.
        level: Logging level (default INFO).

    Returns:
        logging.LogRecord instance.
    """
    record = logging.LogRecord(
        name="test",
        level=level,
        pathname="test.py",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    return record


# --------------------------------------------------------------------------- #
# Initialisation                                                               #
# --------------------------------------------------------------------------- #


class TestAzureBlobHandlerInit:
    """Tests for AzureBlobHandler.__init__ and _create_blob_client."""

    def test_from_connection_string_called_with_conn_str(self):
        """__init__ calls BlobServiceClient.from_connection_string with the provided string."""
        with patch(
            "azure_log_handler.BlobServiceClient.from_connection_string"
        ) as mock_factory:
            mock_service = MagicMock()
            mock_blob_client = MagicMock()
            mock_blob_client.get_blob_properties.side_effect = Exception("not found")
            mock_service.get_blob_client.return_value = mock_blob_client
            mock_factory.return_value = mock_service

            AzureBlobHandler(_CONN_STR, _BLOB_NAME, _CONTAINER)

        mock_factory.assert_called_once_with(_CONN_STR)

    def test_container_name_is_lowercased(self):
        """Container name is forced to lowercase during init."""
        handler, _, _ = _make_handler(container="UPPER-CASE-CONTAINER")

        assert handler.container_name == "upper-case-container"

    def test_blob_client_is_created_for_today(self):
        """__init__ creates a blob client named with today's date."""
        today = date.today()
        handler, mock_service, _ = _make_handler()

        expected_blob_name = f"{_BLOB_NAME}_{today}.log"
        mock_service.get_blob_client.assert_called_with(
            container=_CONTAINER, blob=expected_blob_name
        )

    def test_create_append_blob_called_when_blob_not_found(self):
        """_create_blob_client calls create_append_blob when the blob does not exist."""
        _, _, mock_blob_client = _make_handler()

        mock_blob_client.create_append_blob.assert_called_once()

    def test_base_blob_name_stored(self):
        """The base_blob_name attribute is set to the provided blob_name."""
        handler, _, _ = _make_handler()

        assert handler.base_blob_name == _BLOB_NAME

    def test_connection_string_stored(self):
        """The connection_string attribute is set to the provided value."""
        handler, _, _ = _make_handler()

        assert handler.connection_string == _CONN_STR


# --------------------------------------------------------------------------- #
# emit                                                                         #
# --------------------------------------------------------------------------- #


class TestAzureBlobHandlerEmit:
    """Tests for AzureBlobHandler.emit."""

    def test_emit_calls_append_block_with_encoded_message(self):
        """emit() appends the formatted log record to the blob as UTF-8 bytes."""
        handler, _, mock_blob_client = _make_handler()
        # Reset call history from __init__
        mock_blob_client.append_block.reset_mock()
        record = _make_log_record("hello world")

        handler.emit(record)

        mock_blob_client.append_block.assert_called_once()
        call_args = mock_blob_client.append_block.call_args.args[0]
        assert isinstance(call_args, bytes)
        assert b"hello world" in call_args

    def test_emit_appends_newline_to_message(self):
        """emit() ensures the message ends with a newline before writing."""
        handler, _, mock_blob_client = _make_handler()
        mock_blob_client.append_block.reset_mock()
        record = _make_log_record("no trailing newline")

        handler.emit(record)

        call_args = mock_blob_client.append_block.call_args.args[0]
        # Decoded bytes should end with a newline
        assert call_args.decode("utf-8").endswith("\n")

    def test_emit_with_existing_newline_does_not_double_newline(self):
        """emit() does not add a second newline if the formatted message already has one."""
        handler, _, mock_blob_client = _make_handler()
        mock_blob_client.append_block.reset_mock()
        # Override format to return a message that already has \n
        handler.format = MagicMock(return_value="msg with newline\n")
        record = _make_log_record("ignored by mock format")

        handler.emit(record)

        call_args = mock_blob_client.append_block.call_args.args[0]
        decoded = call_args.decode("utf-8")
        # Should not end with two consecutive newlines
        assert not decoded.endswith("\n\n")


# --------------------------------------------------------------------------- #
# Midnight rotation                                                            #
# --------------------------------------------------------------------------- #


class TestMidnightRotation:
    """Tests for the midnight date-rotation logic in emit()."""

    def test_rotation_creates_new_blob_client_on_date_change(self):
        """emit() creates a new blob client when the calendar date advances."""
        handler, mock_service, old_blob_client = _make_handler()
        old_blob_client.append_block.reset_mock()

        # New future date — clearly different from real today
        new_date = date(2099, 1, 2)
        new_blob_client = MagicMock()
        new_blob_client.get_blob_properties.side_effect = Exception("not found")
        mock_service.get_blob_client.return_value = new_blob_client
        # Reset the call count AFTER setting up the new_blob_client return value
        mock_service.get_blob_client.reset_mock()

        with patch("azure_log_handler.date") as mock_date:
            mock_date.today.return_value = new_date
            record = _make_log_record("after midnight")
            handler.emit(record)

        # A new blob client must have been fetched (rotation triggered)
        mock_service.get_blob_client.assert_called_once()

    def test_rotation_updates_current_date(self):
        """emit() updates handler.current_date after midnight rotation."""
        handler, mock_service, _ = _make_handler()
        new_blob_client = MagicMock()
        new_blob_client.get_blob_properties.side_effect = Exception("not found")
        mock_service.get_blob_client.return_value = new_blob_client

        new_date = date(2099, 1, 2)
        with patch("azure_log_handler.date") as mock_date:
            mock_date.today.return_value = new_date
            handler.emit(_make_log_record("rotation test"))

        assert handler.current_date == new_date

    def test_no_rotation_when_date_unchanged(self):
        """emit() does not create a new blob client when the date has not changed."""
        handler, mock_service, mock_blob_client = _make_handler()
        initial_get_blob_count = mock_service.get_blob_client.call_count
        mock_blob_client.append_block.reset_mock()

        today = date.today()
        with patch("azure_log_handler.date") as mock_date:
            mock_date.today.return_value = today
            handler.emit(_make_log_record("same day"))

        # get_blob_client count should not have increased
        assert mock_service.get_blob_client.call_count == initial_get_blob_count


# --------------------------------------------------------------------------- #
# Error fallback                                                               #
# --------------------------------------------------------------------------- #


class TestErrorFallback:
    """Tests for emit() error fallback to stdout."""

    def test_emit_prints_to_stderr_on_append_block_failure(self, capsys):
        """When append_block raises, emit() prints the error to stdout and does not crash."""
        handler, _, mock_blob_client = _make_handler()
        mock_blob_client.append_block.side_effect = Exception("azure unavailable")

        record = _make_log_record("will fail")
        # Should not raise
        handler.emit(record)

        captured = capsys.readouterr()
        # emit() uses print() on failure, which writes to stdout
        assert "azure unavailable" in captured.out

    def test_emit_does_not_raise_on_azure_failure(self):
        """emit() never propagates exceptions from Azure storage failures."""
        handler, _, mock_blob_client = _make_handler()
        mock_blob_client.append_block.side_effect = Exception("storage error")
        record = _make_log_record("fail silently")

        # Must not raise — any raised exception fails the test automatically
        handler.emit(record)


# =========================================================================== #
# logging_setup tests                                                          #
# =========================================================================== #


def _minimal_log_config(**overrides) -> dict:
    """Return a minimal log_config dict with all required keys, allowing overrides.

    Args:
        **overrides: Key/value pairs to override the defaults.

    Returns:
        Dict suitable for passing to setup_logging().
    """
    defaults = {
        "log_type": [],
        "log_level": "WARNING",
        "log_dir": "logs",
        "log_file_name": "test.log",
        "retention_days": 7,
        "console_logging": False,
        "structured_format": True,
    }
    defaults.update(overrides)
    return defaults


def _reset_root_logger():
    """Detach all handlers from the root logger and reset its level."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        h.close()
        root.removeHandler(h)
    root.setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
# load_app_config                                                              #
# --------------------------------------------------------------------------- #


class TestLoadAppConfig:
    """Tests for load_app_config() — config parsing and fallback behaviour."""

    def test_valid_config_returns_all_values(self, tmp_path):
        """valid scenario: returns values matching the file."""
        config_data = {
            "logging": {
                "log_type": ["file"],
                "log_level": "DEBUG",
                "log_dir": "custom_logs",
                "log_file_name": "myapp.log",
                "retention_days": 14,
                "console_logging": False,
                "structured_format": False,
            },
            "trading": {
                "epic": "IX.D.SP500.IFM.IP",
                "leverage": 10,
                "demo_starting_balance": 50000.0,
                "initial_cash_balance": 5000.0,
                "security_buffer": 500.0,
            },
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        result = load_app_config(str(config_file))

        assert result["logging"]["log_level"] == "DEBUG"
        assert result["logging"]["log_dir"] == "custom_logs"
        assert result["logging"]["retention_days"] == 14
        assert result["trading"]["epic"] == "IX.D.SP500.IFM.IP"
        assert result["trading"]["leverage"] == 10

    def test_missing_file_returns_defaults_and_warns(self, tmp_path, caplog):
        """missing scenario: returns defaults and emits a warning."""
        missing_path = str(tmp_path / "nonexistent.json")

        with caplog.at_level(logging.WARNING):
            result = load_app_config(missing_path)

        assert result == _DEFAULTS
        assert any("not found" in record.message for record in caplog.records)

    def test_malformed_json_returns_defaults_and_warns(self, tmp_path, caplog):
        """malformed scenario: returns defaults and emits a warning."""
        bad_file = tmp_path / "config.json"
        bad_file.write_text("{this is not valid json")

        with caplog.at_level(logging.WARNING):
            result = load_app_config(str(bad_file))

        assert result == _DEFAULTS
        assert any("Could not parse" in record.message for record in caplog.records)

    def test_empty_config_returns_all_defaults_and_warns(self, tmp_path, caplog):
        """empty config: valid JSON with no sections falls back to all defaults."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({}))

        with caplog.at_level(logging.WARNING):
            result = load_app_config(str(config_file))

        assert result == _DEFAULTS
        assert any("missing or invalid 'logging'" in r.message for r in caplog.records)
        assert any("missing or invalid 'trading'" in r.message for r in caplog.records)

    def test_returns_independent_copy_of_defaults(self, tmp_path):
        """Mutating the returned defaults dict must not affect future calls."""
        missing_path = str(tmp_path / "nonexistent.json")
        first = load_app_config(missing_path)
        first["logging"]["log_level"] = "CRITICAL"

        second = load_app_config(missing_path)
        assert second["logging"]["log_level"] == "INFO"


class TestLoadAppConfigTradingSection:
    """Tests for load_app_config() trading section handling."""

    def test_valid_trading_section_returns_all_keys(self, tmp_path):
        """scenario 1: valid trading section — all 5 keys present and typed."""
        config_data = {
            "logging": {
                "log_type": ["file"],
                "log_level": "INFO",
                "log_dir": "logs",
                "log_file_name": "kuroko.log",
                "retention_days": 7,
                "console_logging": True,
                "structured_format": True,
                "azure_log_partition_key": "PROD_NQ100",
            },
            "trading": {
                "epic": "IX.D.NASDAQ.IFMM.IP",
                "leverage": 20,
                "demo_starting_balance": 20000.0,
                "initial_cash_balance": 4000.0,
                "security_buffer": 1000.0,
            },
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        result = load_app_config(str(config_file))

        assert result["logging"]["azure_log_partition_key"] == "PROD_NQ100"
        trading = result["trading"]
        assert trading["epic"] == "IX.D.NASDAQ.IFMM.IP"
        assert trading["leverage"] == 20
        assert trading["initial_cash_balance"] == 4000.0

    def test_missing_trading_section_returns_defaults_and_warns(self, tmp_path, caplog):
        """scenario 2: missing 'trading' key — WARNING logged, defaults returned."""
        config_data = {
            "logging": {
                "log_type": ["file"],
                "log_level": "INFO",
                "log_dir": "logs",
                "log_file_name": "kuroko.log",
                "retention_days": 7,
                "console_logging": True,
                "structured_format": True,
            }
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        with caplog.at_level(logging.WARNING):
            result = load_app_config(str(config_file))

        assert isinstance(result["trading"], dict)
        assert result["trading"]["epic"] == "IX.D.NASDAQ.IFMM.IP"
        assert any("missing or invalid 'trading'" in r.message for r in caplog.records)

    def test_null_trading_section_returns_defaults_and_warns(self, tmp_path, caplog):
        """scenario 3: trading is null — WARNING logged, defaults returned."""
        config_data = {
            "logging": {
                "log_type": ["file"],
                "log_level": "INFO",
                "log_dir": "logs",
                "log_file_name": "kuroko.log",
                "retention_days": 7,
                "console_logging": True,
                "structured_format": True,
            },
            "trading": None,
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        with caplog.at_level(logging.WARNING):
            result = load_app_config(str(config_file))

        assert isinstance(result["trading"], dict)
        assert any("missing or invalid 'trading'" in r.message for r in caplog.records)

    def test_wrong_type_trading_key_replaced_with_default_and_warns(
        self, tmp_path, caplog
    ):
        """type coercion: wrong-type trading key is replaced with default and warns."""
        config_data = {
            "logging": {
                "log_type": ["file"],
                "log_level": "INFO",
                "log_dir": "logs",
                "log_file_name": "kuroko.log",
                "retention_days": 7,
                "console_logging": True,
                "structured_format": True,
            },
            "trading": {
                "epic": "IX.D.NASDAQ.IFMM.IP",
                "leverage": "twenty",
                "demo_starting_balance": 20000.0,
                "initial_cash_balance": 4000.0,
                "security_buffer": 1000.0,
            },
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        with caplog.at_level(logging.WARNING):
            result = load_app_config(str(config_file))

        assert result["trading"]["leverage"] == _DEFAULTS["trading"]["leverage"]
        assert any("trading.leverage" in r.message for r in caplog.records)

    def test_bool_leverage_replaced_with_default_and_warns(self, tmp_path, caplog):
        """bool is a subclass of int — True/False must not pass numeric validation."""
        config_data = {
            "logging": {
                "log_type": [],
                "log_level": "INFO",
                "log_dir": "logs",
                "log_file_name": "kuroko.log",
                "retention_days": 7,
                "console_logging": False,
                "structured_format": True,
            },
            "trading": {
                "epic": "IX.D.NASDAQ.IFMM.IP",
                "leverage": True,
                "demo_starting_balance": 20000.0,
                "initial_cash_balance": 4000.0,
                "security_buffer": 1000.0,
            },
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        with caplog.at_level(logging.WARNING):
            result = load_app_config(str(config_file))

        assert result["trading"]["leverage"] == _DEFAULTS["trading"]["leverage"]
        assert any("trading.leverage" in r.message for r in caplog.records)

    def test_bool_retention_days_replaced_with_default_and_warns(
        self, tmp_path, caplog
    ):
        """bool masquerading as int must be rejected for bare-int fields like retention_days."""
        config_data = {
            "logging": {
                "log_type": [],
                "log_level": "INFO",
                "log_dir": "logs",
                "log_file_name": "kuroko.log",
                "retention_days": True,
                "console_logging": False,
                "structured_format": True,
            },
            "trading": {
                "epic": "IX.D.NASDAQ.IFMM.IP",
                "leverage": 20,
                "demo_starting_balance": 20000.0,
                "initial_cash_balance": 4000.0,
                "security_buffer": 1000.0,
            },
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        with caplog.at_level(logging.WARNING):
            result = load_app_config(str(config_file))

        assert (
            result["logging"]["retention_days"]
            == _DEFAULTS["logging"]["retention_days"]
        )
        assert any("logging.retention_days" in r.message for r in caplog.records)

    def test_non_dict_trading_section_returns_defaults_and_warns(
        self, tmp_path, caplog
    ):
        """scenario 4: trading is a non-dict scalar — WARNING logged, defaults returned."""
        config_data = {
            "logging": {
                "log_type": ["file"],
                "log_level": "INFO",
                "log_dir": "logs",
                "log_file_name": "kuroko.log",
                "retention_days": 7,
                "console_logging": True,
                "structured_format": True,
            },
            "trading": 42,
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config_data))

        with caplog.at_level(logging.WARNING):
            result = load_app_config(str(config_file))

        assert isinstance(result["trading"], dict)
        assert result["trading"]["epic"] == "IX.D.NASDAQ.IFMM.IP"
        assert any("missing or invalid 'trading'" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# setup_logging — handler attachment                                           #
# --------------------------------------------------------------------------- #


class TestSetupLoggingHandlerAttachment:
    """Tests for handler attachment logic in setup_logging()."""

    @pytest.fixture(autouse=True)
    def reset_logger(self):
        """Reset root logger before and after each test."""
        _reset_root_logger()
        yield
        _reset_root_logger()

    def test_file_only_log_type_attaches_single_file_handler(self, tmp_path):
        """single handler: only a TimedRotatingFileHandler is attached."""
        log_dir = str(tmp_path / "logs")
        cfg = _minimal_log_config(
            log_type=["file"],
            log_dir=log_dir,
            log_file_name="test.log",
            console_logging=False,
        )

        setup_logging(cfg, partition_key="test-key")

        root_handlers = logging.getLogger().handlers
        file_handlers = [
            h for h in root_handlers if isinstance(h, TimedRotatingFileHandler)
        ]
        stream_handlers = [
            h
            for h in root_handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, TimedRotatingFileHandler)
        ]
        assert len(file_handlers) == 1
        assert len(stream_handlers) == 0

    def test_file_and_azure_log_types_attach_both_handlers(self, tmp_path):
        """multiple handler types: both file and Azure handlers attached."""
        log_dir = str(tmp_path / "logs")
        cfg = _minimal_log_config(
            log_type=["file", "azure_table"],
            log_dir=log_dir,
            log_file_name="test.log",
            console_logging=False,
        )
        mock_azure_handler = MagicMock(spec=logging.Handler)

        with patch(
            "azure_log_handler.AzureBlobHandler", return_value=mock_azure_handler
        ):
            with patch.dict(
                os.environ, {"table_storage_connection": "fake-connection-string"}
            ):
                setup_logging(cfg, partition_key="test-key")

        root_handlers = logging.getLogger().handlers
        file_handlers = [
            h for h in root_handlers if isinstance(h, TimedRotatingFileHandler)
        ]
        assert len(file_handlers) == 1
        assert mock_azure_handler in root_handlers

    def test_empty_log_type_attaches_no_file_or_azure_handlers(self, tmp_path):
        """empty log_type: no file or Azure handlers attached."""
        cfg = _minimal_log_config(log_type=[], console_logging=False)

        setup_logging(cfg, partition_key="test-key")

        assert len(logging.getLogger().handlers) == 0

    def test_console_logging_enabled_attaches_stream_handler(self):
        """console enabled: StreamHandler present."""
        cfg = _minimal_log_config(log_type=[], console_logging=True)

        setup_logging(cfg, partition_key="test-key")

        stream_handlers = [
            h
            for h in logging.getLogger().handlers
            if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) == 1

    def test_console_logging_disabled_omits_stream_handler(self):
        """console disabled: no StreamHandler attached."""
        cfg = _minimal_log_config(log_type=[], console_logging=False)

        setup_logging(cfg, partition_key="test-key")

        stream_handlers = [
            h
            for h in logging.getLogger().handlers
            if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) == 0


# --------------------------------------------------------------------------- #
# setup_logging — formatter selection                                          #
# --------------------------------------------------------------------------- #


class TestSetupLoggingFormatter:
    """Tests for the formatter applied to handlers by setup_logging()."""

    @pytest.fixture(autouse=True)
    def reset_logger(self):
        """Reset root logger before and after each test."""
        _reset_root_logger()
        yield
        _reset_root_logger()

    def test_structured_format_uses_pipe_delimited_pattern(self):
        """structured: formatter uses pipe-delimited pattern."""
        cfg = _minimal_log_config(
            log_type=[], console_logging=True, structured_format=True
        )

        setup_logging(cfg, partition_key="test-key")

        handler = logging.getLogger().handlers[0]
        assert "|" in handler.formatter._fmt

    def test_legacy_format_uses_bracket_pattern(self):
        """legacy: formatter uses bracket pattern without pipes."""
        cfg = _minimal_log_config(
            log_type=[], console_logging=True, structured_format=False
        )

        setup_logging(cfg, partition_key="test-key")

        handler = logging.getLogger().handlers[0]
        assert "[" in handler.formatter._fmt
        assert "|" not in handler.formatter._fmt


# --------------------------------------------------------------------------- #
# setup_logging — log directory creation                                       #
# --------------------------------------------------------------------------- #


class TestSetupLoggingDirectoryCreation:
    """Tests for log directory auto-creation."""

    @pytest.fixture(autouse=True)
    def reset_logger(self):
        """Reset root logger before and after each test."""
        _reset_root_logger()
        yield
        _reset_root_logger()

    def test_log_dir_created_when_missing(self, tmp_path):
        """directory creation: log_dir is created when it does not exist."""
        log_dir = str(tmp_path / "new_logs" / "nested")
        cfg = _minimal_log_config(
            log_type=["file"],
            log_dir=log_dir,
            log_file_name="test.log",
            console_logging=False,
        )

        setup_logging(cfg, partition_key="test-key")

        assert os.path.isdir(log_dir)


# --------------------------------------------------------------------------- #
# setup_logging — pre-existing handler removal                                 #
# --------------------------------------------------------------------------- #


class TestSetupLoggingHandlerClear:
    """Tests for pre-existing handler removal on re-entry."""

    @pytest.fixture(autouse=True)
    def reset_logger(self):
        """Reset root logger before and after each test."""
        _reset_root_logger()
        yield
        _reset_root_logger()

    def test_pre_existing_handlers_removed_and_closed(self):
        """dummy handler attached before setup_logging() is removed and closed."""
        dummy = MagicMock(spec=logging.Handler)
        logging.getLogger().addHandler(dummy)

        cfg = _minimal_log_config(log_type=[], console_logging=False)
        setup_logging(cfg, partition_key="test-key")

        assert dummy not in logging.getLogger().handlers
        dummy.close.assert_called_once()


# --------------------------------------------------------------------------- #
# _cleanup_old_logs                                                            #
# --------------------------------------------------------------------------- #


class TestCleanupOldLogs:
    """Tests for _cleanup_old_logs() retention logic."""

    def test_files_older_than_retention_days_are_deleted(self, tmp_path):
        """retention: files older than retention_days are deleted."""
        base = "kuroko.log"
        old_date = (date.today() - timedelta(days=10)).isoformat()
        recent_date = (date.today() - timedelta(days=2)).isoformat()
        old_file = tmp_path / f"{base}.{old_date}"
        recent_file = tmp_path / f"{base}.{recent_date}"
        old_file.write_text("old log")
        recent_file.write_text("recent log")

        _cleanup_old_logs(str(tmp_path), retention_days=7, base_filename=base)

        assert not old_file.exists()
        assert recent_file.exists()

    def test_files_within_retention_days_are_preserved(self, tmp_path):
        """retention: files within retention_days are NOT deleted."""
        base = "kuroko.log"
        recent_date = (date.today() - timedelta(days=2)).isoformat()
        recent_file = tmp_path / f"{base}.{recent_date}"
        recent_file.write_text("recent log")

        _cleanup_old_logs(str(tmp_path), retention_days=7, base_filename=base)

        assert recent_file.exists()

    def test_no_error_when_log_directory_is_missing(self, tmp_path):
        """no-files scenario: no error when log directory does not exist."""
        missing_dir = str(tmp_path / "does_not_exist")

        _cleanup_old_logs(missing_dir, retention_days=7, base_filename="kuroko.log")

    def test_no_error_when_log_directory_is_empty(self, tmp_path):
        """no-files scenario: empty directory handled gracefully."""
        _cleanup_old_logs(str(tmp_path), retention_days=7, base_filename="kuroko.log")
