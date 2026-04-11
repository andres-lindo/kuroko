"""Entry point for the Kuroko live trading bot.

Loads configuration from Azure Table Storage, initialises logging (console
and Azure Blob), connects to IG Markets, and starts the strategy loop.
"""
import os
import ast
import sys
import logging
import argparse

from dotenv import load_dotenv
from azure.data.tables import TableServiceClient

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


def load_params(partition_key: str):
    """Load strategy configuration from Azure Table Storage.

    Queries the ConfigParameters table for rows matching partition_key or
    the shared 'BASE_CONF' partition, then deserialises each value into the
    correct Python type based on the key name.

    Args:
        partition_key: Azure Table PartitionKey that selects the config set
            (e.g. 'DEV_US500').

    Returns:
        Config object whose attributes correspond to each RowKey in the table,
        with values cast to str, float, bool, or int as appropriate.
    """
    # Connect to Azure Table Storage using the connection string from the environment
    conn_str = os.getenv('table_storage_connection')
    table_service = TableServiceClient.from_connection_string(conn_str=conn_str)
    table_client = table_service.get_table_client(table_name="ConfigParameters")

    # Fetch rows for the requested partition and the shared base config
    entities = table_client.query_entities(f"PartitionKey eq '{partition_key}' or PartitionKey eq 'BASE_CONF'")

    # Flatten entity list into a {RowKey: Value} mapping
    raw = {e['RowKey']: e['Value'] for e in entities}

    class Config:
        """Namespace object that holds configuration values as attributes."""
        pass

    cfg = Config()

    # Dispatch each key to the correct Python type; keys not listed default to int
    for key, val in raw.items():
        if key in ('table_storage_name', 'table_log_name', 'cfd_symbol', 'candle_frecuency'):
            setattr(cfg, key, val)
        elif key in ('position_size_long', 'position_size_short',
                     'default_volume', 'profit_threshold',
                     'stop_loss_long', 'stop_loss_short',
                     'take_profit_long', 'take_profit_short'):
            setattr(cfg, key, float(val))
        elif key == 'ema_crossover':
            setattr(cfg, key, ast.literal_eval(val))
        else:
            setattr(cfg, key, int(val))

    # Attach the raw connection string so downstream components can reuse it
    setattr(cfg, 'table_storage_connection', conn_str)

    return cfg


def main():
    """Parse CLI arguments, bootstrap the bot, and start the strategy loop.

    Reads the partition_key from the command line, loads parameters from
    Azure Table Storage, attaches Azure Blob log shipping, and then runs
    the strategy until interrupted.
    """
    # The partition key selects the Azure Table row set for this deployment
    parser = argparse.ArgumentParser(description="Start the trading bot")
    parser.add_argument(
        'partition_key', nargs='?', default="DEV_US500",
        help="PartitionKey to filter parameters from Azure Table Storage"
    )
    args = parser.parse_args()

    # Load all strategy parameters from Azure Table Storage
    params = load_params(args.partition_key)
    logging.info("Parameters loaded.")

    # Attach Azure Blob log handler so logs are shipped to cloud storage
    try:
        azure_handler = AzureBlobHandler(
            connection_string=params.table_storage_connection,
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
