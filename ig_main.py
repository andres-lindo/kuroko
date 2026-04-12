"""Entry point for the Kuroko live trading bot.

Dynamically loads the configured strategy module, initialises logging
(console and Azure Blob), connects to IG Markets, and starts the strategy loop.
"""

import importlib
import logging
import os
import sys
import warnings

import argparse
from dotenv import load_dotenv

from azure_log_handler import AzureBlobHandler
from ig_client import IGClient

BANNER = r"""
  ██╗  ██╗██╗   ██╗██████╗  ██████╗ ██╗  ██╗ ██████╗
  ██║ ██╔╝██║   ██║██╔══██╗██╔═══██╗██║ ██╔╝██╔═══██╗
  █████╔╝ ██║   ██║██████╔╝██║   ██║█████╔╝ ██║   ██║
  ██╔═██╗ ██║   ██║██╔══██╗██║   ██║██╔═██╗ ██║   ██║
  ██║  ██╗╚██████╔╝██║  ██║╚██████╔╝██║  ██╗╚██████╔╝
  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              OPERATE IN THE SHADOWS
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


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
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Suppress verbose INFO output from third-party libraries that are not actionable
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("trading_ig.rest").setLevel(logging.WARNING)

# Suppress FutureWarnings from trading_ig until the upstream library is updated
warnings.filterwarnings("ignore", category=FutureWarning, module="trading_ig.utils")
warnings.filterwarnings("ignore", category=FutureWarning, module="trading_ig.rest")

logger = logging.getLogger(__name__)


def load_strategy(strategy_name: str) -> tuple[type, callable]:
    """Dynamically load a strategy module and return the class and load_params callable.

    The module file must be named after the strategy class (e.g.
    'RSIBollingerStrategy' is loaded from 'strategies/RSIBollingerStrategy.py').
    Exits the process with code 1 if the module is not found or does not
    export the expected names.

    Args:
        strategy_name: Strategy class name (e.g. 'RSIBollingerStrategy').

    Returns:
        A tuple of (StrategyClass, load_params_fn).

    Raises:
        SystemExit: If the module cannot be imported or does not export the
            expected class or 'load_params' callable.
    """
    try:
        module = importlib.import_module(f"strategies.{strategy_name}")
    except ModuleNotFoundError as e:
        if e.name not in (f"strategies.{strategy_name}", "strategies"):
            raise
        logger.critical(
            f"Strategy module not found for '{strategy_name}'. "
            f"Check that strategies/{strategy_name}.py exists."
        )
        sys.exit(1)

    try:
        strategy_class = getattr(module, strategy_name)
    except AttributeError:
        logger.critical(
            f"Module 'strategies.{strategy_name}' does not export class '{strategy_name}'."
        )
        sys.exit(1)

    try:
        load_params_fn = getattr(module, "load_params")
    except AttributeError:
        logger.critical(
            f"Module 'strategies.{strategy_name}' does not export 'load_params'."
        )
        sys.exit(1)

    return strategy_class, load_params_fn


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
            connection_string=conn_str, blob_name=partition_key, container_name="logs"
        )
        azure_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(azure_handler)
        logging.info(f"Azure Blob logging configured for partition: {partition_key}")
        sys.excepthook = handle_exception
    except Exception as e:
        logging.warning(f"Could not configure Azure Blob logging: {e}")


def main():
    """Parse CLI arguments, bootstrap the bot, and start the strategy loop.

    Dynamically loads the configured strategy and its parameters, attaches
    Azure Blob log shipping (using ``log_partition_key`` from the strategy
    JSON), and then runs the strategy until interrupted.

    If the strategy module cannot be loaded or parameter loading fails, a
    CRITICAL log entry is written and the process exits with code 1.
    No IGClient or strategy instantiation is attempted in that case.
    """
    print(BANNER)
    parser = argparse.ArgumentParser(description="Kuroko live trading bot")
    parser.add_argument(
        "--strategy",
        required=True,
        help="Strategy class name (e.g. RSIBollingerStrategy)",
    )
    args = parser.parse_args()

    strategy_class, load_params = load_strategy(args.strategy)
    params = load_params(f"strategies/{args.strategy}.json")
    setup_azure_logging(params.log_partition_key)

    # Initialise broker client and strategy, then enter the main loop
    ig = IGClient()
    strat = strategy_class(params=params, ig_client=ig)

    logging.info("Kuroko started. Press CTRL+C to stop.")
    try:
        strat.run()
    except KeyboardInterrupt:
        logging.info("CTRL+C detected. Exiting...")
    except Exception:
        logging.exception("Critical application error:")
        raise


if __name__ == "__main__":
    main()
