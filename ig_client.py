import os
import logging
import pandas as pd
from pathlib import Path
from time import sleep
from datetime import datetime, timedelta
from requests.exceptions import ConnectionError, RequestException

from trading_ig import IGService
from trading_ig.rest import IGException

log = logging.getLogger(__name__)

class IGClient:

    def __init__(self):
        user = os.getenv("ig_username")
        pwd  = os.getenv("ig_password")
        key  = os.getenv("ig_api_key")
        accn = os.getenv("ig_acc_number")
        acc_type = os.getenv("ig_acc_type")  # DEMO | LIVE

        if not (user and pwd and key):
            raise RuntimeError("Faltan credenciales IG (usuario/clave/api_key).")

        self._svc = IGService(user, pwd, key, acc_type, accn)
        self._svc.create_session()
        self.accountId = accn

        # Caché simple en memoria
        self.candles_cache = {}  # {epic_res: DataFrame}
        
        # Directorio para persistencia
        self.cache_dir = Path("./cache")
        self.cache_dir.mkdir(exist_ok=True)

    def _safe_api_call(self, func, *args, max_retries=3, **kwargs):
        """
        Ejecuta una llamada a la API con reintentos en caso de error de conexión.
        """
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except (ConnectionError, RequestException, IGException) as e:
                # Refrescar sesión si token expirado
                if "token" in str(e).lower():
                    log.warning("Token expirado. Refrescando sesión...")
                    try:
                        self._svc.create_session()
                        log.info("Sesión refrescada exitosamente")
                        continue  # Reintentar con nueva sesión
                    except Exception as refresh_error:
                        log.error(f"Error refrescando sesión: {refresh_error}")
                
                log.debug(f"Error de conexión (intento {attempt + 1}/{max_retries}): {e}")
                
                if attempt < max_retries - 1:
                    # Espera progresiva entre reintentos
                    wait_time = 2 ** attempt  # 1, 2, 4 segundos
                    log.debug(f"Esperando {wait_time} segundos antes de reintentar...")
                    sleep(wait_time)
                else:
                    log.error(f"Fallo después de {max_retries} intentos")
                    raise
            except Exception as e:
                log.error(f"Error inesperado: {e}")
                raise

    def _remove_incomplete_candle(self, df: pd.DataFrame, resolution: str) -> pd.DataFrame:
        """
        Elimina la última vela si está incompleta.
        Una vela está incompleta si su timestamp == hora actual.
        Una vela está completa si su timestamp == (hora actual - timeframe).
        """
        if df.empty:
            return df
        
        # Normalizar hora de ejecución al minuto
        exec_time = datetime.now().replace(second=0, microsecond=0)
        last_candle_time = pd.to_datetime(df.index[-1]).replace(second=0, microsecond=0)
        
        # Obtener minutos del timeframe
        timeframe_minutes = int(resolution.replace("min", "")) if "min" in resolution else 60
        expected_complete_time = exec_time - timedelta(minutes=timeframe_minutes)
        
        # Si última vela == hora ejecución → incompleta → eliminar
        if last_candle_time == exec_time:
            log.debug(f"Eliminando vela incompleta: {last_candle_time}")
            return df.iloc[:-1]
        # Si última vela == (hora ejecución - timeframe) → completa → mantener
        elif last_candle_time == expected_complete_time:
            log.debug(f"Última vela completa: {last_candle_time}")
            return df
        else:
            log.debug(f"Timestamp inesperado. Última vela: {last_candle_time}, Esperado: {expected_complete_time}")
            return df

    def get_candles(self, epic: str, res: str, num_points: int = 200) -> pd.DataFrame:
        """
        Obtiene velas optimizando el uso de la API.
        Primera vez: pide 200 velas
        Actualizaciones: pide solo 2 velas y las agrega al caché
        Nota: Verifica si la última vela está completa comparando con hora de ejecución
        """
        cache_key = f"{epic}_{res}"
        
        # Si no hay caché, intenta cargar desde disco
        if cache_key not in self.candles_cache:
            cache_file = self.cache_dir / f"{cache_key}.parquet"
            if cache_file.exists():
                try:
                    df = pd.read_parquet(cache_file)
                    self.candles_cache[cache_key] = df
                    log.info(f"Caché cargado desde disco para {epic} {res}")
                except Exception as e:
                    log.warning(f"Error cargando caché: {e}")
        
        # Si aún no hay caché o está vacío, carga inicial completa
        if cache_key not in self.candles_cache or self.candles_cache[cache_key].empty:
            log.info(f"Carga inicial: pidiendo {num_points} velas para {epic} {res}")
            
            resp = self._safe_api_call(
                self._svc.fetch_historical_prices_by_epic_and_num_points,
                epic, res, num_points + 1  # Pide una extra por si necesitamos eliminar
            )
            df = resp['prices']['bid']
            
            # Verificar y eliminar vela incompleta si es necesario
            df = self._remove_incomplete_candle(df, res)
            
            self.candles_cache[cache_key] = df
            
            # Guarda en disco
            try:
                df.to_parquet(self.cache_dir / f"{cache_key}.parquet")
            except Exception as e:
                log.warning(f"Error guardando caché en disco: {e}")
            
            return df.tail(num_points).copy()
        
        # Actualización incremental
        existing_df = self.candles_cache[cache_key]
        
        # Pide solo 3 velas recientes
        log.debug(f"Verificando actualizaciones para {epic} {res}")
        
        try:
            resp = self._safe_api_call(
                self._svc.fetch_historical_prices_by_epic_and_num_points,
                epic, res, 3
            )
            new_df = resp['prices']['bid']
            
            # Verificar y eliminar vela incompleta si es necesario
            new_df = self._remove_incomplete_candle(new_df, res)
            
            # Solo combinar si hay datos válidos
            if not new_df.empty and not new_df.isna().all().all():
                # Combina con el caché existente (sin duplicados)
                combined_df = pd.concat([existing_df, new_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df = combined_df.dropna(how='all')
                combined_df = combined_df.sort_index().tail(num_points)
                
                # Solo actualizar si hay cambios
                if len(combined_df) > len(existing_df) or not combined_df.equals(existing_df.tail(num_points)):
                    log.info(f"Actualizando caché para {epic} {res}")
                    self.candles_cache[cache_key] = combined_df
                    try:
                        combined_df.to_parquet(self.cache_dir / f"{cache_key}.parquet")
                    except Exception as e:
                        log.warning(f"Error guardando caché actualizado: {e}")
                    
                    return combined_df.copy()
            
            return existing_df.tail(num_points).copy()
            
        except Exception as e:
            log.error(f"Error obteniendo actualización, devolviendo caché existente: {e}")
            return existing_df.tail(num_points).copy()

    def clear_cache(self):
        """Limpia todo el caché."""
        self.candles_cache.clear()
        for file in self.cache_dir.glob("*.parquet"):
            try:
                file.unlink()
            except Exception as e:
                log.warning(f"Error eliminando archivo de caché {file}: {e}")
        log.info("Caché limpiado")

    def get_open_positions(self):
        try:
            open_positions = self._safe_api_call(self._svc.fetch_open_positions)
            
            if open_positions.empty:
                return []
            
            return open_positions[["dealReference", "dealId", "level", "size", "createdDate", "direction"]].to_dict(orient="records")
        except Exception as e:
            log.error(f"Error obteniendo posiciones abiertas: {e}")
            return []

    def get_account_summary(self):
        try:
            accounts = self._safe_api_call(self._svc.fetch_accounts)
            cols = ["accountId", "balance", "deposit", "profitLoss", "available"]
            
            return accounts.loc[
                accounts["accountId"] == self.accountId, cols
            ].to_dict(orient="records")[0]
        except Exception as e:
            log.error(f"Error obteniendo resumen de cuenta: {e}")
            return {}

    def open_position(self, epic: str, size: float, side: str, currency: str = 'USD', stop: float = None, limit: float = None):
        return self._safe_api_call(
            self._svc.create_open_position,
            currency_code=currency,
            direction=side,
            epic=epic,
            order_type='MARKET',
            expiry='-',
            force_open='true',
            guaranteed_stop='false',
            size=float(size), 
            level=None,
            limit_distance=limit,
            limit_level=None,
            quote_id=None,
            stop_level=None,
            stop_distance=stop,
            trailing_stop=None,
            trailing_stop_increment=None
        )

    def update_position(self, dealid: str, stop: float = None, limit: float = None):
        return self._safe_api_call(
            self._svc.update_open_position,
            limit_level=limit,
            stop_level=stop,
            deal_id=dealid
        )

    def close_position(self, deal_id: str, side: str, size: float):
        return self._safe_api_call(
            self._svc.close_open_position,
            deal_id=deal_id,
            direction=side,
            epic=None,
            expiry='-',
            level=None,
            order_type='MARKET',
            quote_id=None,
            size=float(size),
        )