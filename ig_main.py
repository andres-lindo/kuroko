"""Entry point for the Kuroko live trading bot.

Loads strategy parameters from strategy_parameters.json, initialises logging
(console and Azure Blob), connects to IG Markets, and starts the strategy loop.
"""
import os
import sys
import logging
import argparse
import json
import types
import re

from dotenv import load_dotenv

from ig_client import IGClient
from ig_strategy import Strategy
from azure_log_handler import AzureBlobHandler
import warnings

# Route unhandled exceptions through the standard logger instead of stderr
def handle_exception(exc_type, exc_value, exc_traceback):
    """Log unhandled exceptions before the interpreter exits.

    Allows KeyboardInterrupt to pass through to the default handler so
    CTRL+C still terminates the process cleanly.

    Args:
        exc_type: Exception class of the unhandled exception.
        exc_value: Exception instance.
        exc_traceback: Traceback object.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Unhandled exception:", exc_info=(exc_type, exc_value, exc_traceback))


# Load secrets from credentials.env before any module reads environment variables
load_dotenv("credentials.env")

# Configure root logger so all modules emit to stdout with a consistent format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Suppress verbose INFO output from third-party libraries that are not actionable
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("trading_ig.rest").setLevel(logging.WARNING)

# Suppress FutureWarnings from trading_ig until the upstream library is updated
warnings.filterwarnings("ignore", category=FutureWarning, module="trading_ig.utils")
warnings.filterwarnings("ignore", category=FutureWarning, module="trading_ig.rest")



# Expected type for each parameter key.
# float fields accept int values (e.g. 240 is valid for take_profit_ticks).
# bool fields are checked before int because bool is a subclass of int in Python.
_PARAMS_SCHEMA: dict[str, type] = {
    "epic":                           str,
    "candle_frecuency":               str,
    "is_live_account":                bool,
    "leverage":                       int,
    "lookback":                       int,
    "demo_starting_balance":          float,
    "initial_cash_balance":           float,
    "security_buffer":                float,
    "max_positions":                  int,
    "position_size":                  float,
    "min_dist_between_entries_ticks": float,
    "martingale_multiplier":          float,
    "take_profit_ticks":              float,
    "max_drawdown_pct":               float,
    "bb_period":                      int,
    "bb_dev":                         float,
    "rsi_period":                     int,
    "rsi_overbought":                 int,
    "rsi_oversold":                   int,
    "use_trend_filter":               bool,
    "atr_period":                     int,
    "atr_sl_multiplier":              float,
    "ema_period":                     int,
}


def _validate_params(data: dict, path: str) -> None:
    """Validate that all required keys are present and correctly typed.

    Collects every missing key and every type mismatch before logging them
    all at once, so a single bad file produces a complete error report.

    Args:
        data: Parsed JSON dict to validate.
        path: File path used in error messages.

    Raises:
        SystemExit: If any key is missing or has the wrong type.
    """
    errors: list[str] = []

    for key, expected in _PARAMS_SCHEMA.items():
        if key not in data:
            errors.append(f"  missing key: '{key}'")
            continue

        value = data[key]

        if expected is bool:
            if not isinstance(value, bool):
                errors.append(
                    f"  '{key}': expected bool, got {type(value).__name__} ({value!r})"
                )
        elif expected is int:
            # Reject bool (subclass of int) and non-int values
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(
                    f"  '{key}': expected int, got {type(value).__name__} ({value!r})"
                )
        elif expected is float:
            # Accept int as float (e.g. 240 is valid for take_profit_ticks)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(
                    f"  '{key}': expected float, got {type(value).__name__} ({value!r})"
                )
        elif not isinstance(value, expected):
            errors.append(
                f"  '{key}': expected {expected.__name__}, got {type(value).__name__} ({value!r})"
            )

    if errors:
        logging.critical(
            "Parameter validation failed for %s — %d error(s):\n%s",
            path,
            len(errors),
            "\n".join(errors),
        )
        sys.exit(1)

    if not re.match(r"^\d+min$", data["candle_frecuency"]):
        logging.critical(
            "Invalid candle_frecuency in %s — must match '<N>min' (e.g. '15min'), got: %r",
            path,
            data["candle_frecuency"],
        )
        sys.exit(1)


def load_params(path: str = "strategy_parameters.json") -> types.SimpleNamespace:
    """Load and validate strategy parameters from a JSON file.

    Reads the JSON file at ``path``, validates all required keys and their
    types, and returns the parameters as a SimpleNamespace for attribute-style
    access.

    Args:
        path: Path to the JSON parameters file. Defaults to
            ``strategy_parameters.json`` in the working directory.

    Returns:
        SimpleNamespace with one attribute per JSON key.

    Raises:
        SystemExit: If the file is missing, unreadable, contains invalid JSON,
            has missing keys, type mismatches, or an invalid ``candle_frecuency``.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        logging.critical("Parameters file not found: %s", path)
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.critical("Invalid JSON in %s: %s", path, e)
        sys.exit(1)
    except OSError as e:
        logging.critical("Could not read %s: %s", path, e)
        sys.exit(1)

    _validate_params(data, path)

    logging.info("Parameters loaded from %s.", path)
    return types.SimpleNamespace(**data)


def main():
    """Parse CLI arguments, bootstrap the bot, and start the strategy loop.

    Loads strategy parameters from ``strategy_parameters.json``, attaches
    Azure Blob log shipping, and then runs the strategy until interrupted.

    If parameter loading fails for any reason, a CRITICAL log entry is written
    and the process exits with code 1. No IGClient or Strategy initialisation
    is attempted in that case.
    """
    # The partition key is used as the log blob label in Azure Blob Storage
    parser = argparse.ArgumentParser(description="Start the trading bot")
    parser.add_argument(
        'partition_key', nargs='?', default="DEV_US500",
        help="Label used to identify this deployment's log blob in Azure Blob Storage"
    )
    args = parser.parse_args()

    params = load_params()

    # Attach Azure Blob log handler so logs are shipped to cloud storage
    try:
        conn_str = os.getenv("table_storage_connection")
        azure_handler = AzureBlobHandler(
            connection_string=conn_str,
            blob_name=args.partition_key,
            container_name="logs"
        )

        # Reuse the same format as the console handler for consistency
        azure_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))

        # Attach to root logger so all modules ship their output to the blob
        logging.getLogger().addHandler(azure_handler)
        logging.info(f"Azure Blob logging configured for partition: {args.partition_key}")

        # Register the unhandled-exception hook after the blob handler is ready
        sys.excepthook = handle_exception
    except Exception as e:
        logging.warning(f"Could not configure Azure Blob logging: {e}")

    # Initialise broker client and strategy, then enter the main loop
    ig = IGClient()
    strat = Strategy(params=params, ig_client=ig)

    logging.info('IG bot started. Press CTRL+C to stop.')
    try:
        strat.run()
    except KeyboardInterrupt:
        logging.info('CTRL+C detected. Exiting...')
    except Exception as e:
        logging.exception("Critical application error:")
        raise


if __name__ == '__main__':
    main()
