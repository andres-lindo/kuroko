# main.py
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

# Función para capturar excepciones no manejadas
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    logging.error("Excepción no manejada:", exc_info=(exc_type, exc_value, exc_traceback))

# Carga variables de entorno de .env
load_dotenv("credentials.env")

# Logging del módulo principal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# Se establece el nivel de logging a WARNING para evitar mensajes innecesarios
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("trading_ig.rest").setLevel(logging.WARNING)

# Ignorar advertencias
warnings.filterwarnings("ignore", category=FutureWarning, module="trading_ig.utils")
warnings.filterwarnings("ignore", category=FutureWarning, module="trading_ig.rest")

def load_params(partition_key: str):
    """
    Carga los parámetros de configuración desde Azure Table Storage filtrando
    por PartitionKey == partition_key o 'BASE_CONF', y retorna un objeto con atributos.
    """
    # Conectar al servicio de tablas de Azure
    conn_str = os.getenv('table_storage_connection')
    table_service = TableServiceClient.from_connection_string(conn_str=conn_str)
    table_client = table_service.get_table_client(table_name="ConfigParameters")

    # Consultar las entidades deseadas
    entities = table_client.query_entities(f"PartitionKey eq '{partition_key}' or PartitionKey eq 'BASE_CONF'")

    # Leer todos los valores
    raw = {e['RowKey']: e['Value'] for e in entities}

    # Objeto contenedor de configuración
    class Config: pass
    cfg = Config()

    # Asignar atributos, convirtiendo tipos según la clave
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

    # También guardamos la cadena de conexión
    setattr(cfg, 'table_storage_connection', conn_str)

    return cfg

def main():
    # Se espera que el primer argumento sea la clave de partición para Azure Table
    parser = argparse.ArgumentParser(description="Inicia el bot de trading")
    parser.add_argument(
        'partition_key', nargs='?', default="DEV_US500",
        help="PartitionKey para filtrar parámetros en Azure Table"
    )
    args = parser.parse_args()

    # Parámetros globales de Azure Table
    params = load_params(args.partition_key)
    logging.info("Parámetros cargados")
    
    # Configurar Azure Blob Handler para logging
    try:
        azure_handler = AzureBlobHandler(
            connection_string=params.table_storage_connection,
            blob_name=args.partition_key,
            container_name="logs"
        )

        # Usar el mismo formato que el handler de consola
        azure_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        ))

        # Añadir el handler al root logger
        logging.getLogger().addHandler(azure_handler)
        logging.info(f"Logging configurado para Azure Blob Storage - {args.partition_key}")

        # Configurar el manejador de excepciones no manejadas
        sys.excepthook = handle_exception
    except Exception as e:
        logging.warning(f"No se pudo configurar Azure Blob logging: {e}")

    # Cliente IG
    ig = IGClient()
    strat = Strategy(params=params, ig_client=ig)

    logging.info('Bot IG iniciado. CTRL+C para detener.')
    try:
        strat.run()
    except KeyboardInterrupt:
        logging.info('CTRL+C detectado. Saliendo...')
    except Exception as e:
        logging.exception("Error crítico en la aplicación:")
        raise

if __name__ == '__main__':
    main()
