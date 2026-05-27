"""Logging bootstrap for the Kuroko trading bot.

Provides load_app_config() to parse config.json (with graceful fallback to
defaults) and setup_logging() to wire all configured handlers onto the root
logger, replacing any handlers that were attached earlier (e.g. by basicConfig
at import time).
"""

import copy
import json
import logging
import os
import re
import sys
from datetime import date, timedelta
from logging.handlers import TimedRotatingFileHandler

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Defaults                                                                     #
# --------------------------------------------------------------------------- #

_DEFAULTS: dict = {
    "logging": {
        "log_type": ["file"],
        "log_level": "INFO",
        "log_dir": "logs",
        "log_file_name": "kuroko.log",
        "retention_days": 7,
        "console_logging": True,
        "structured_format": True,
        "azure_log_partition_key": "DEV_NQ100",
    },
    "trading": {
        "epic": "IX.D.NASDAQ.IFMM.IP",
        "leverage": 20,
        "demo_starting_balance": 20000.0,
        "initial_cash_balance": 4000.0,
        "security_buffer": 1000.0,
    },
}

_FORMAT_STRUCTURED = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_FORMAT_LEGACY = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


# Expected types for each key in the trading section
_TRADING_TYPES: dict = {
    "epic": str,
    "leverage": (int, float),
    "demo_starting_balance": (int, float),
    "initial_cash_balance": (int, float),
    "security_buffer": (int, float),
}

# Expected types for each key in the logging section
_LOGGING_TYPES: dict = {
    "log_type": list,
    "log_level": str,
    "log_dir": str,
    "log_file_name": str,
    "retention_days": int,
    "console_logging": bool,
    "structured_format": bool,
    "azure_log_partition_key": str,
}


def _validate_section_types(
    data: dict, types_dict: dict, defaults: dict, section_name: str
) -> None:
    """Check each known key in a config section against its expected type.

    Logs a WARNING for each key whose value does not match the expected type and
    replaces it with the default value. Missing keys are also warned about and
    filled from defaults. Unknown extra keys are left untouched.

    Args:
        data: The config section dict, mutated in place if corrections are made.
        types_dict: Mapping of key → expected type (or tuple of types).
        defaults: The defaults dict for this section, used for replacements.
        section_name: Section label used in warning messages (e.g. 'trading').
    """
    for key, expected in types_dict.items():
        if key not in data:
            logger.warning(
                f"config.json {section_name}.{key} is missing — using default value"
            )
            data[key] = copy.deepcopy(defaults[key])
            continue
        value = data[key]
        # bool is a subclass of int in Python; reject it when the expected type is numeric
        is_numeric_expected = expected in (int, (int, float))
        is_bool_masquerading = isinstance(value, bool) and is_numeric_expected
        if is_bool_masquerading or not isinstance(value, expected):
            if isinstance(expected, type):
                expected_name = expected.__name__
            else:
                expected_name = "/".join(t.__name__ for t in expected)
            logger.warning(
                f"config.json {section_name}.{key} has unexpected type "
                f"'{type(value).__name__}' — expected '{expected_name}'; "
                f"using default value"
            )
            data[key] = copy.deepcopy(defaults[key])


# --------------------------------------------------------------------------- #
# Config loading                                                               #
# --------------------------------------------------------------------------- #


def load_app_config(config_path: str) -> dict:
    """Load config.json from config_path and return its contents as a dict.

    Falls back to _DEFAULTS when the file is missing or contains invalid JSON,
    and emits a WARNING in both fallback cases.

    Args:
        config_path: Absolute path to config.json.

    Returns:
        Parsed config dict (same structure as _DEFAULTS) or _DEFAULTS on error.
    """
    if not os.path.exists(config_path):
        logger.warning(
            f"config.json not found at '{config_path}'. Using default config."
        )
        return copy.deepcopy(_DEFAULTS)

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for section in ("logging", "trading"):
            if not isinstance(data.get(section), dict):
                logger.warning(
                    f"config.json missing or invalid '{section}' section — using defaults"
                )
                data[section] = copy.deepcopy(_DEFAULTS[section])
        _validate_section_types(
            data["logging"], _LOGGING_TYPES, _DEFAULTS["logging"], "logging"
        )
        _validate_section_types(
            data["trading"], _TRADING_TYPES, _DEFAULTS["trading"], "trading"
        )
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            f"Could not parse config.json at '{config_path}': {exc}. "
            "Using default config."
        )
        return copy.deepcopy(_DEFAULTS)


# --------------------------------------------------------------------------- #
# Retention cleanup                                                            #
# --------------------------------------------------------------------------- #


def _cleanup_old_logs(log_dir: str, retention_days: int, base_filename: str) -> None:
    """Delete rotated log files in log_dir that are older than retention_days.

    Skips silently when log_dir does not exist (nothing to clean up yet).

    Args:
        log_dir: Directory that contains the rotated log files.
        retention_days: Files older than this many days are deleted.
        base_filename: Base filename of the log (e.g. 'kuroko.log') — used to
            restrict deletion to files that belong to this log series.
    """
    if not os.path.isdir(log_dir):
        return

    # Pattern: exactly {base_filename}.YYYY-MM-DD — excludes the base file
    # itself and any unrelated files that share a common prefix.
    _rotated_pattern = re.compile(
        r"^" + re.escape(base_filename) + r"\.\d{4}-\d{2}-\d{2}$"
    )
    cutoff_date = date.today() - timedelta(days=retention_days)

    try:
        with os.scandir(log_dir) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                m = _rotated_pattern.match(entry.name)
                if not m:
                    continue
                # Parse date from the filename suffix (YYYY-MM-DD) rather than
                # relying on mtime, which can lag hours behind actual rotation.
                try:
                    suffix = entry.name[len(base_filename) + 1 :]  # skip leading dot
                    file_date = date.fromisoformat(suffix)
                except ValueError:
                    continue
                if file_date < cutoff_date:
                    try:
                        os.remove(entry.path)
                        logger.debug(f"Deleted old log file: {entry.path}")
                    except OSError as exc:
                        logger.warning(
                            f"Could not remove log file '{entry.path}': {exc}"
                        )
    except OSError as exc:
        logger.warning(f"Could not scan log directory '{log_dir}': {exc}")


# --------------------------------------------------------------------------- #
# Handler setup                                                                #
# --------------------------------------------------------------------------- #


def setup_logging(log_config: dict, partition_key: str) -> None:
    """Configure the root logger with the handlers specified in log_config.

    Clears all existing root handlers before attaching the new ones to prevent
    duplicate log entries from any earlier basicConfig() calls.

    Handler attachment order:
    1. TimedRotatingFileHandler — when "file" is in log_type
    2. AzureBlobHandler        — when "azure_table" is in log_type
    3. StreamHandler(stdout)   — when console_logging is True

    Args:
        log_config: The "logging" sub-dict from config.json (or defaults).
        partition_key: Partition/blob name passed to AzureBlobHandler. Must
            come from config.json["logging"]["azure_log_partition_key"].
    """
    log_type: list = log_config.get("log_type", _DEFAULTS["logging"]["log_type"])
    log_level_str: str = log_config.get("log_level", _DEFAULTS["logging"]["log_level"])
    log_dir: str = log_config.get("log_dir", _DEFAULTS["logging"]["log_dir"])
    log_file_name: str = log_config.get(
        "log_file_name", _DEFAULTS["logging"]["log_file_name"]
    )
    retention_days: int = log_config.get(
        "retention_days", _DEFAULTS["logging"]["retention_days"]
    )
    console_logging: bool = log_config.get(
        "console_logging", _DEFAULTS["logging"]["console_logging"]
    )
    structured_format: bool = log_config.get(
        "structured_format", _DEFAULTS["logging"]["structured_format"]
    )

    level = getattr(logging, log_level_str.upper(), None)
    if level is None:
        logger.warning(f"Unknown log_level '{log_level_str}', defaulting to INFO")
        level = logging.INFO

    # Close and remove any handlers attached before setup_logging() was called
    # to avoid file-descriptor leaks.
    root = logging.getLogger()
    for h in root.handlers[:]:
        h.close()
        root.removeHandler(h)
    root.setLevel(level)

    # Single formatter applied to every handler (REQ-6)
    fmt = _FORMAT_STRUCTURED if structured_format else _FORMAT_LEGACY
    formatter = logging.Formatter(fmt)

    # File handler with daily rotation (REQ-3)
    if "file" in log_type:
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as exc:
            logger.warning(
                f"Could not create log directory '{log_dir}': {exc} — file logging disabled"
            )
            log_type = [t for t in log_type if t != "file"]
        if "file" in log_type:
            log_path = os.path.join(log_dir, log_file_name)
            file_handler = TimedRotatingFileHandler(
                filename=log_path,
                when="midnight",
                backupCount=0,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
            _cleanup_old_logs(log_dir, retention_days, log_file_name)

    # Azure Blob handler (REQ-4)
    if "azure_table" in log_type:
        if not partition_key:
            logger.warning("partition_key is empty — Azure Blob logging disabled")
        else:
            conn_str = os.getenv("table_storage_connection")
            if not conn_str:
                logger.warning(
                    "table_storage_connection not set — Azure Blob logging disabled"
                )
            else:
                try:
                    # Import is deferred here because azure_log_handler is an
                    # optional dependency — the module may not be installed in
                    # environments that do not use Azure logging.
                    from azure_log_handler import AzureBlobHandler

                    azure_handler = AzureBlobHandler(
                        connection_string=conn_str,
                        blob_name=partition_key,
                        container_name="logs",
                    )
                    azure_handler.setFormatter(formatter)
                    root.addHandler(azure_handler)
                    # Confirmation is logged AFTER the handler is attached so the
                    # message is guaranteed to reach the Azure handler itself.
                    logger.info(
                        f"Azure Blob logging configured for partition: {partition_key}"
                    )
                except Exception as exc:
                    logger.warning(f"Could not configure Azure Blob logging: {exc}")

    # Console handler (REQ-5)
    if console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)

    # Route unhandled exceptions through the logger (REQ-7)
    sys.excepthook = _handle_uncaught_exception


def _handle_uncaught_exception(exc_type, exc_value, exc_traceback):
    """Log unhandled exceptions via the root logger before the process exits.

    Passes KeyboardInterrupt through to the default handler so CTRL+C
    still terminates the process cleanly.

    Args:
        exc_type: Exception class of the unhandled exception.
        exc_value: Exception instance.
        exc_traceback: Traceback object.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Unhandled exception:", exc_info=(exc_type, exc_value, exc_traceback))
