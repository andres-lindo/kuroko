"""Entry point for the Kuroko live trading bot.

Dynamically loads the configured strategy module, initialises logging via
logging_setup (file, Azure Blob, and/or console per config.json), connects to
IG Markets, and starts the strategy loop.
"""

import importlib
import logging
import os
import sys
import warnings

import argparse
from dotenv import load_dotenv

from logging_setup import load_app_config, setup_logging
from ig_client import IGClient

BANNER = r"""
  ██╗  ██╗██╗   ██╗██████╗  ██████╗ ██╗  ██╗ ██████╗
  ██║ ██╔╝██║   ██║██╔══██╗██╔═══██╗██║ ██╔╝██╔═══██╗
  █████╔╝ ██║   ██║██████╔╝██║   ██║█████╔╝ ██║   ██║
  ██╔═██╗ ██║   ██║██╔══██╗██║   ██║██╔═██╗ ██║   ██║
  ██║  ██╗╚██████╔╝██║  ██║╚██████╔╝██║  ██╗╚██████╔╝
  ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              OPERATES IN THE SHADOWS
       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# Load secrets from credentials.env before any module reads environment variables
load_dotenv("credentials.env")

# Configure a minimal root logger so early-startup warnings (e.g. from
# load_app_config) are visible before setup_logging() replaces the handlers.
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

    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, "config.json")
    config = load_app_config(config_path)

    # Resolve log_dir to an absolute path anchored at the project root so the
    # bot creates logs relative to its own directory regardless of CWD.
    log_dir = config["logging"].get("log_dir", "logs")
    if not os.path.isabs(log_dir):
        config["logging"]["log_dir"] = os.path.join(project_root, log_dir)

    strategy_class, load_params = load_strategy(args.strategy)
    strategy_path = os.path.join(project_root, "strategies", f"{args.strategy}.json")
    params = load_params(strategy_path)

    # Configure all handlers once, after params are loaded so the Azure Blob
    # handler uses the correct partition_key from the strategy JSON.
    # Console output during the bootstrap phase above is handled by basicConfig.
    setup_logging(config["logging"], params.log_partition_key)

    # Initialise broker client and strategy, then enter the main loop
    ig = IGClient()
    strat = strategy_class(params=params, ig_client=ig)

    logger.info("Kuroko started. Press CTRL+C to stop.")
    try:
        strat.run()
    except KeyboardInterrupt:
        logger.info("CTRL+C detected. Exiting...")
    except Exception:
        logger.exception("Critical application error:")
        raise


if __name__ == "__main__":
    main()
