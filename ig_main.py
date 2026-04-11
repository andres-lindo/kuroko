"""Entry point for the Kuroko live trading bot.

Loads strategy parameters from strategy_parameters.json, initialises logging
(console and Azure Blob), connects to IG Markets, and starts the strategy loop.
"""
import os
import sys
import logging
import argparse

from dotenv import load_dotenv

from ig_client import IGClient
from ig_strategy import Strategy, load_params
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



def setup_azure_logging(partition_key: str) -> None:
    """Attach the AzureBlobHandler to the root logger for cloud log shipping.

    Also registers the unhandled-exception hook so crashes are captured in the
    blob before the process exits. Fails gracefully — a warning is logged and
    the bot continues without cloud shipping if the handler cannot be created.

    Args:
        partition_key: Label used to identify this deployment's log blob.
    """
    try:
        conn_str = os.getenv("table_storage_connection")
        azure_handler = AzureBlobHandler(
            connection_string=conn_str,
            blob_name=partition_key,
            container_name="logs"
        )
        azure_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))
        logging.getLogger().addHandler(azure_handler)
        logging.info(f"Azure Blob logging configured for partition: {partition_key}")
        sys.excepthook = handle_exception
    except Exception as e:
        logging.warning(f"Could not configure Azure Blob logging: {e}")


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
        'partition_key', nargs='?', default="DEV_NQ100",
        help="Label used to identify this deployment's log blob in Azure Blob Storage"
    )
    args = parser.parse_args()

    params = load_params()
    setup_azure_logging(args.partition_key)

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
