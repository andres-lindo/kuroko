"""Unit tests for logging_setup.py.

Covers config loading (valid, missing, malformed), handler attachment per
log_type combinations, formatter selection, retention cleanup, and handler
clearing on setup_logging() re-entry.
"""

import json
import logging
import os
from datetime import date, timedelta
from logging.handlers import TimedRotatingFileHandler
from unittest.mock import MagicMock, patch

import pytest

from logging_setup import (
    _DEFAULTS,
    _cleanup_old_logs,
    load_app_config,
    setup_logging,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _minimal_log_config(**overrides) -> dict:
    """Return a minimal log_config dict with all required keys, allowing overrides."""
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
    """Tests for load_app_config()."""

    def test_valid_config(self, tmp_path):
        """REQ-1 valid scenario: returns values matching the file."""
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
        assert result["logging"]["console_logging"] is False
        assert result["logging"]["structured_format"] is False
        assert result["trading"]["epic"] == "IX.D.SP500.IFM.IP"
        assert result["trading"]["leverage"] == 10
        assert result["trading"]["initial_cash_balance"] == 5000.0

    def test_missing_file_returns_defaults(self, tmp_path, caplog):
        """REQ-1 missing scenario: returns defaults and emits a warning."""
        missing_path = str(tmp_path / "nonexistent.json")

        with caplog.at_level(logging.WARNING):
            result = load_app_config(missing_path)

        assert result == _DEFAULTS
        assert any("not found" in record.message for record in caplog.records)

    def test_malformed_json_returns_defaults(self, tmp_path, caplog):
        """REQ-1 malformed scenario: returns defaults and emits a warning."""
        bad_file = tmp_path / "config.json"
        bad_file.write_text("{this is not valid json")

        with caplog.at_level(logging.WARNING):
            result = load_app_config(str(bad_file))

        assert result == _DEFAULTS
        assert any("Could not parse" in record.message for record in caplog.records)

    def test_empty_config_returns_all_defaults(self, tmp_path, caplog):
        """REQ-1 empty config: valid JSON with no sections falls back to all defaults."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({}))

        with caplog.at_level(logging.WARNING):
            result = load_app_config(str(config_file))

        assert result == _DEFAULTS
        assert any(
            "missing or invalid 'logging'" in record.message
            for record in caplog.records
        )
        assert any(
            "missing or invalid 'trading'" in record.message
            for record in caplog.records
        )

    def test_returns_independent_copy_of_defaults(self, tmp_path):
        """Mutating the returned defaults dict must not affect future calls."""
        missing_path = str(tmp_path / "nonexistent.json")
        first = load_app_config(missing_path)
        first["logging"]["log_level"] = "CRITICAL"

        second = load_app_config(missing_path)
        assert second["logging"]["log_level"] == "INFO"


# --------------------------------------------------------------------------- #
# load_app_config — trading section                                            #
# --------------------------------------------------------------------------- #


class TestLoadAppConfigTrading:
    """Tests for load_app_config() trading section handling."""

    def test_valid_trading_section_returns_all_keys(self, tmp_path):
        """REQ-1 scenario 1: valid trading section — all 5 keys present and typed."""
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
        assert trading["demo_starting_balance"] == 20000.0
        assert trading["initial_cash_balance"] == 4000.0
        assert trading["security_buffer"] == 1000.0

    def test_missing_trading_section_returns_defaults_and_warns(self, tmp_path, caplog):
        """REQ-1 scenario 2: missing 'trading' key — WARNING logged, defaults returned."""
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
        assert any(
            "missing or invalid 'trading'" in record.message
            for record in caplog.records
        )

    def test_trading_section_null_returns_defaults_and_warns(self, tmp_path, caplog):
        """REQ-1 scenario 3: trading is null — WARNING logged, defaults returned."""
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
        assert result["trading"]["epic"] == "IX.D.NASDAQ.IFMM.IP"
        assert any(
            "missing or invalid 'trading'" in record.message
            for record in caplog.records
        )

    def test_trading_wrong_type_replaced_with_default(self, tmp_path, caplog):
        """REQ-1 type coercion: wrong-type trading key is replaced with default and warns."""
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
        assert any("trading.leverage" in record.message for record in caplog.records)

    def test_trading_bool_leverage_replaced_with_default(self, tmp_path, caplog):
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
        assert any("trading.leverage" in record.message for record in caplog.records)

    def test_logging_bool_retention_days_replaced_with_default(self, tmp_path, caplog):
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
        assert any(
            "logging.retention_days" in record.message for record in caplog.records
        )

    def test_trading_section_non_dict_returns_defaults_and_warns(
        self, tmp_path, caplog
    ):
        """REQ-1 scenario 4: trading is a non-dict scalar — WARNING logged, defaults returned."""
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
        assert any(
            "missing or invalid 'trading'" in record.message
            for record in caplog.records
        )


# --------------------------------------------------------------------------- #
# setup_logging — handler attachment                                           #
# --------------------------------------------------------------------------- #


class TestSetupLoggingHandlers:
    """Tests for handler attachment logic in setup_logging()."""

    def setup_method(self):
        _reset_root_logger()

    def teardown_method(self):
        _reset_root_logger()

    def test_file_only(self, tmp_path):
        """REQ-2 single handler: only a TimedRotatingFileHandler is attached."""
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
        assert len(file_handlers) == 1
        # No StreamHandler
        stream_handlers = [
            h
            for h in root_handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, TimedRotatingFileHandler)
        ]
        assert len(stream_handlers) == 0

    def test_multiple_handlers(self, tmp_path):
        """REQ-2 multiple handler types: both file and Azure handlers attached."""
        log_dir = str(tmp_path / "logs")
        cfg = _minimal_log_config(
            log_type=["file", "azure_table"],
            log_dir=log_dir,
            log_file_name="test.log",
            console_logging=False,
        )
        mock_azure_handler = MagicMock(spec=logging.Handler)

        # Patch target is 'azure_log_handler.AzureBlobHandler' (not
        # 'logging_setup.AzureBlobHandler') because the import is deferred
        # inside the function body — the name lives in the azure_log_handler
        # module's own namespace at the point of use.
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

    def test_empty_log_type_no_file_or_azure(self, tmp_path):
        """REQ-2 empty log_type: no file or Azure handlers attached."""
        cfg = _minimal_log_config(log_type=[], console_logging=False)

        setup_logging(cfg, partition_key="test-key")

        root_handlers = logging.getLogger().handlers
        assert len(root_handlers) == 0

    def test_console_logging_enabled(self, tmp_path):
        """REQ-5 console enabled: StreamHandler present."""
        cfg = _minimal_log_config(log_type=[], console_logging=True)

        setup_logging(cfg, partition_key="test-key")

        root_handlers = logging.getLogger().handlers
        stream_handlers = [
            h for h in root_handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) == 1

    def test_console_logging_disabled(self, tmp_path):
        """REQ-5 console disabled: no StreamHandler attached."""
        cfg = _minimal_log_config(log_type=[], console_logging=False)

        setup_logging(cfg, partition_key="test-key")

        root_handlers = logging.getLogger().handlers
        stream_handlers = [
            h for h in root_handlers if isinstance(h, logging.StreamHandler)
        ]
        assert len(stream_handlers) == 0


# --------------------------------------------------------------------------- #
# setup_logging — formatter selection                                          #
# --------------------------------------------------------------------------- #


class TestSetupLoggingFormatter:
    """Tests for the formatter applied to handlers by setup_logging()."""

    def setup_method(self):
        _reset_root_logger()

    def teardown_method(self):
        _reset_root_logger()

    def test_structured_format(self, tmp_path):
        """REQ-6 structured: formatter uses pipe-delimited pattern."""
        cfg = _minimal_log_config(
            log_type=[], console_logging=True, structured_format=True
        )

        setup_logging(cfg, partition_key="test-key")

        handler = logging.getLogger().handlers[0]
        assert "|" in handler.formatter._fmt

    def test_legacy_format(self, tmp_path):
        """REQ-6 legacy: formatter uses bracket pattern."""
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


class TestSetupLoggingDirCreation:
    """Tests for log directory auto-creation."""

    def setup_method(self):
        _reset_root_logger()

    def teardown_method(self):
        _reset_root_logger()

    def test_log_dir_created_if_missing(self, tmp_path):
        """REQ-3 directory creation: log_dir is created when it does not exist."""
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
# setup_logging — handler clearing                                             #
# --------------------------------------------------------------------------- #


class TestSetupLoggingHandlerClear:
    """Tests for REQ-7 pre-existing handler removal."""

    def setup_method(self):
        _reset_root_logger()

    def teardown_method(self):
        _reset_root_logger()

    def test_pre_existing_handlers_are_removed(self):
        """REQ-7: dummy handler attached before setup_logging() must be gone after,
        and close() must be called on it."""
        dummy = MagicMock(spec=logging.Handler)
        logging.getLogger().addHandler(dummy)
        assert dummy in logging.getLogger().handlers

        cfg = _minimal_log_config(log_type=[], console_logging=False)
        setup_logging(cfg, partition_key="test-key")

        assert dummy not in logging.getLogger().handlers
        dummy.close.assert_called_once()


# --------------------------------------------------------------------------- #
# _cleanup_old_logs                                                            #
# --------------------------------------------------------------------------- #


class TestCleanupOldLogs:
    """Tests for _cleanup_old_logs()."""

    def test_deletes_old_files(self, tmp_path):
        """REQ-3 retention: files older than retention_days are deleted."""
        log_dir = str(tmp_path)
        base = "kuroko.log"

        # Create two rotated files — one 10 days old (stale), one 2 days old (recent)
        old_date = (date.today() - timedelta(days=10)).isoformat()
        recent_date = (date.today() - timedelta(days=2)).isoformat()
        old_file = tmp_path / f"{base}.{old_date}"
        recent_file = tmp_path / f"{base}.{recent_date}"
        old_file.write_text("old log")
        recent_file.write_text("recent log")

        _cleanup_old_logs(log_dir, retention_days=7, base_filename=base)

        assert not old_file.exists()
        assert recent_file.exists()

    def test_preserves_recent_files(self, tmp_path):
        """REQ-3 retention: files within retention_days are NOT deleted."""
        log_dir = str(tmp_path)
        base = "kuroko.log"

        recent_date = (date.today() - timedelta(days=2)).isoformat()
        recent_file = tmp_path / f"{base}.{recent_date}"
        recent_file.write_text("recent log")

        _cleanup_old_logs(log_dir, retention_days=7, base_filename=base)

        assert recent_file.exists()

    def test_no_error_when_dir_missing(self, tmp_path):
        """REQ-3 no-files scenario: no error when log directory does not exist."""
        missing_dir = str(tmp_path / "does_not_exist")

        # Must not raise
        _cleanup_old_logs(missing_dir, retention_days=7, base_filename="kuroko.log")

    def test_no_error_when_dir_is_empty(self, tmp_path):
        """REQ-3 no-files scenario: empty directory handled gracefully."""
        _cleanup_old_logs(str(tmp_path), retention_days=7, base_filename="kuroko.log")
