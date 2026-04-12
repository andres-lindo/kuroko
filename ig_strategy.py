import time
import logging
import talib as ta
import pandas as pd

from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

class Strategy:
    def __init__(self, params, ig_client):
        self.params = params
        self.ig = ig_client
        self.candles = pd.DataFrame()

        # Parámetros operativos
        self.max_positions = 5
        self.min_dist_between_entries_ticks = 100.0
        self.martingale_multiplier = 1.5
        self.bb_dev = 1.9
        self.bb_period = 20
        self.rsi_period = 11
        self.rsi_overbought = 76
        self.rsi_oversold = 25
        self.use_trend_filter = False
        self.take_profit_ticks = 240.0
        self.atr_period = 12
        self.atr_sl_multiplier = 11.0

        # Variables internas
        self.position_size = 0.13
        self.max_drawdown_pct = 75.75
        self.ema_period = 200
        self.max_drawdown_reached = False

        # Variables sistema
        self.is_live_account = False          # Cambiar a True SOLO cuando se opere con dinero REAL
        self.demo_starting_balance = 20000.0  # Balance demo de IG
        self.initial_cash_balance = 4000.0    # Capital inicial
        
        self.leverage = 20
        self.epic = "IX.D.NASDAQ.IFMM.IP"
        self.lookback = 300
        self.security_buffer = 1000.0

    def get_candles(self):
        df = self.ig.get_candles(self.epic, '15min', self.lookback).copy()

        df['rsi'] = ta.RSI(df["Close"], timeperiod=self.rsi_period)
        logger.info(f"RSI calculado: period={self.rsi_period}")

        df['ema'] = ta.EMA(df["Close"], timeperiod=self.ema_period)
        logger.info(f"EMA calculada: period={self.ema_period}")

        df['atr'] = ta.ATR(df["High"], df["Low"], df["Close"], timeperiod=self.atr_period)
        logger.info(f"ATR calculado: period={self.atr_period}")

        df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.BBANDS(
            df["Close"], timeperiod=self.bb_period,
            nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0
        )
        logger.info(f"BBANDS calculadas: period={self.bb_period}")

        self.candles = df.tail(self.lookback)

    def manage_positions(self):
        if self.candles.empty:
            logger.warning("No hay datos de velas para gestionar posiciones.")
            return

        current_candle = self.candles.iloc[-1]
        current_price = current_candle['Close']

        positions = self.ig.get_open_positions()
        n_trades = len(positions)

        # --- 0. SALIDAS (TIME STOP & BASKET TP) ---
        if n_trades > 0:
            positions.sort(key=lambda x: x['createdDate'])
            first_trade = positions[0]

            total_size = sum(p['size'] for p in positions)
            avg_price = sum(p['size'] * p['level'] for p in positions) / total_size
            is_long = first_trade['direction'] == 'BUY'

            if is_long:
                if current_price >= avg_price + self.take_profit_ticks:
                    profit = (current_price - avg_price) * total_size
                    logger.info(f"💰 WIN (LONG) | Size: {total_size} | Avg: {avg_price:.2f} | Curr: {current_price:.2f} | Profit: {profit:.2f}")
                    self.close_all_positions(positions, reason="BasketTP")
                    return
            else:
                if current_price <= avg_price - self.take_profit_ticks:
                    profit = (avg_price - current_price) * total_size
                    logger.info(f"💰 WIN (SHORT) | Size: {total_size} | Avg: {avg_price:.2f} | Curr: {current_price:.2f} | Profit: {profit:.2f}")
                    self.close_all_positions(positions, reason="BasketTP")
                    return

        # Leemos la realidad del broker
        account_info = self.ig.get_account_summary()
        open_pnl = account_info['profitLoss']

        # --- 1. SEGURIDAD Y EQUIDAD (DEMO vs REAL) ---
        if self.is_live_account:
            # MODO REAL: Confiamos 100% en los números del broker
            current_equity = account_info['balance'] + open_pnl
            free_margin = account_info['available']
        else:
            # MODO DEMO: Espejo de ganancias para simular estrictamente tus $3,000
            realized_profit = account_info['balance'] - self.demo_starting_balance
            virtual_balance = self.initial_cash_balance + realized_profit
            
            current_equity = virtual_balance + open_pnl
            used_margin = sum((p['size'] * p['level'] / self.leverage) for p in positions) if n_trades > 0 else 0
            free_margin = current_equity - used_margin

        # Cálculo del Drawdown Dinámico
        floor_value = self.initial_cash_balance * (1 - (self.max_drawdown_pct / 100))

        if current_equity < floor_value:
            if not self.max_drawdown_reached:
                logger.error(f"⚠️ MAX DRAWDOWN | Equity ${current_equity:.2f} < Suelo ${floor_value:.2f}. Congelando.")
                self.max_drawdown_reached = True
        elif self.max_drawdown_reached and current_equity > floor_value:
            logger.info(f"✅ RECUPERADO | Equity ${current_equity:.2f} > Suelo. Reactivando.")
            self.max_drawdown_reached = False

        # --- 2. ENTRADAS (GRID) ---
        if self.max_drawdown_reached or n_trades >= self.max_positions:
            return

        if n_trades > 0:
            dist_to_last = abs(current_price - positions[-1]['level'])
            if dist_to_last < self.min_dist_between_entries_ticks:
                return

        # Filtros e indicadores
        current_atr = current_candle['atr']
        sl_dist = current_atr * self.atr_sl_multiplier

        if self.use_trend_filter:
            current_ema = current_candle['ema']
            can_buy = current_price > current_ema
            can_sell = current_price < current_ema
        else:
            can_buy = True
            can_sell = True

        # Sizing (Martingala Fraccionada a 2 decimales)
        current_size = max(self.position_size, round(self.position_size * (self.martingale_multiplier ** n_trades), 2))

        # --- 3. MARGIN CHECK (Unificado) ---
        cost_to_open = (current_price / self.leverage) * current_size

        if cost_to_open > (free_margin - self.security_buffer):
            modo = "REAL" if self.is_live_account else "DEMO/VIRTUAL"
            logger.warning(
                f"🚫 MARGEN {modo} INSUFICIENTE | Req: ${cost_to_open:.2f} | "
                f"Libre: ${free_margin:.2f} | Buffer: ${self.security_buffer} | "
                f"Intento: x{current_size} @ {current_price:.2f}"
            )
            return

        # --- 4. CÁLCULO DE TARGET PRICE DINÁMICO (PREVIO A ENTRAR) ---
        # Calculamos cuál será el promedio SI entramos ahora mismo
        if n_trades > 0:
            total_size = sum(p['size'] for p in positions)
            total_value = sum(p['size'] * p['level'] for p in positions)
        else:
            total_size = 0
            total_value = 0

        futuro_total_size = total_size + current_size
        futuro_total_value = total_value + (current_size * current_price)
        futuro_avg_price = futuro_total_value / futuro_total_size

        # --- 5. EJECUCIÓN ---
        bb_lower = current_candle['bb_lower']
        bb_upper = current_candle['bb_upper']
        current_rsi = current_candle['rsi']

        # LONG
        if can_buy and current_price < bb_lower and current_rsi < self.rsi_oversold:
            if n_trades > 0:
                if positions[0]['direction'] == 'SELL':
                    return
                if current_price >= positions[-1]['level']:
                    return

            # 1. Target y Stop como Nivel/Precio Exacto (Para Actualizar posiciones viejas)
            target_price = round(futuro_avg_price + self.take_profit_ticks, 2)
            stop_price = round(current_price - sl_dist, 2) 
            
            # 2. Target y Stop como Distancia (Para Abrir la nueva posición en IG)
            limit_dist = round(abs(target_price - current_price), 2)
            stop_dist = round(sl_dist, 2)

            try:
                # Abrimos la nueva posición
                self.ig.open_position(epic=self.epic, size=current_size, side='BUY', stop=stop_dist, limit=limit_dist)
                logger.info(f"⬆️ BUY #{n_trades + 1} | x{current_size} @ {current_price:.2f} | SL Nivel: {stop_price} | TP Nivel: {target_price}") 
                
                # Pausa para evitar conflictos al actualizar posiciones justo después de abrir
                time.sleep(2)

                # Actualizamos TP y SL de todas las posiciones Viejas
                upd_df = self.ig.get_open_positions()

                if len(upd_df) > 0:
                    # Calculamos el PROMEDIO REAL EXACTO
                    real_size = sum(p['size'] for p in upd_df)
                    real_avg = sum(p['size'] * p['level'] for p in upd_df) / real_size

                    # Calculamos el TP y SL REAL basados en el promedio real
                    real_tp = round(real_avg + self.take_profit_ticks, 2)
                    real_sl = round(current_price - sl_dist, 2)

                    for p in upd_df:
                        deal_id = p['dealId']
                        # Forzamos los floats con 2 decimales para evitar problemas
                        safe_limit = round(float(real_tp), 2)
                        safe_stop = round(float(real_sl), 2)

                        self.ig.update_position(dealid=deal_id, limit=safe_limit, stop=safe_stop)
                        logger.info(f"🔄 Posición {deal_id} actualizada -> Nuevo TP: {safe_limit} | Nuevo SL: {safe_stop}")
            except Exception as e:
                logger.error(f"Error abriendo/actualizando posiciones LONG: {e}")

        # SHORT
        elif can_sell and current_price > bb_upper and current_rsi > self.rsi_overbought:
            if n_trades > 0:
                if positions[0]['direction'] == 'BUY':
                    return
                if current_price <= positions[-1]['level']:
                    return

            # 1. Target y Stop como Nivel/Precio Exacto (Para Actualizar posiciones viejas)
            target_price = round(futuro_avg_price - self.take_profit_ticks, 2)
            stop_price = round(current_price + sl_dist, 2)
            
            # 2. Target y Stop como Distancia (Para Abrir la nueva posición en IG)
            limit_dist = round(abs(current_price - target_price), 2)
            stop_dist = round(sl_dist, 2)

            try:
                # Abrimos la nueva posición
                self.ig.open_position(epic=self.epic, size=current_size, side='SELL', stop=stop_dist, limit=limit_dist)
                logger.info(f"⬇️ SELL #{n_trades + 1} | x{current_size} @ {current_price:.2f} | SL Nivel: {stop_price} | TP Nivel: {target_price}")

                # Pausa para evitar conflictos al actualizar posiciones justo después de abrir
                time.sleep(2)

                # Actualizamos TP y SL de todas las posiciones Viejas
                upd_df = self.ig.get_open_positions()

                # Actualizamos TP y SL de todas las posiciones Viejas
                if len(upd_df) > 0:
                    # Calculamos el PROMEDIO REAL EXACTO
                    real_size = sum(p['size'] for p in upd_df)
                    real_avg = sum(p['size'] * p['level'] for p in upd_df) / real_size

                    # Calculamos el TP y SL REAL basados en el promedio real
                    real_tp = round(real_avg - self.take_profit_ticks, 2)
                    real_sl = round(current_price + sl_dist, 2)

                    for p in upd_df:
                        deal_id = p['dealId']
                        # Forzamos los floats con 2 decimales para evitar problemas
                        safe_limit = round(float(real_tp), 2)
                        safe_stop = round(float(real_sl), 2)

                        self.ig.update_position(dealid=deal_id, limit=safe_limit, stop=safe_stop)
                        logger.info(f"🔄 Posición {deal_id} actualizada -> Nuevo TP: {safe_limit} | Nuevo SL: {safe_stop}")
            except Exception as e:
                logger.error(f"Error abriendo/actualizando posiciones SHORT: {e}")

    def close_all_positions(self, positions, reason):
        for p in positions:
            close_direction = 'SELL' if p['direction'] == 'BUY' else 'BUY'
            self.ig.close_position(p['dealId'], close_direction, p['size'])            
            logger.info(f"Cerrando {p['dealId']} ({p['direction']}) -> {close_direction} por {reason}")

    def log_account_status(self):
        try:
            account_info = self.ig.get_account_summary()
            open_pnl = account_info['profitLoss']
            positions = self.ig.get_open_positions()
            n_trades = len(positions) if positions else 0

            if self.is_live_account:
                # Datos crudos del broker para cuando pases a REAL
                current_equity = account_info['balance'] + open_pnl
                used_margin = account_info['margin']
                free_margin = account_info['available']
                modo = "REAL"
            else:
                # Matemática estricta para tu simulación a 1:20
                realized_profit = account_info['balance'] - self.demo_starting_balance
                virtual_balance = self.initial_cash_balance + realized_profit
                
                current_equity = virtual_balance + open_pnl
                used_margin = sum((p['size'] * p['level'] / self.leverage) for p in positions) if n_trades > 0 else 0
                free_margin = current_equity - used_margin
                modo = f"VIRTUAL (1:{self.leverage})"

            # --- CÁLCULO DEL NIVEL DE MARGEN (%) ---
            if used_margin > 0:
                margin_level_pct = (current_equity / used_margin) * 100
                margin_str = f"{margin_level_pct:.2f}%"
                
                if margin_level_pct < 120:
                    health_icon = "🚨 PELIGRO"
                elif margin_level_pct < 200:
                    health_icon = "⚠️ ALERTA"
                else:
                    health_icon = "✅ SANO"
            else:
                margin_str = "N/A"
                health_icon = "💤 REPOSO"

            # Imprimimos el bloque de resumen
            logger.info(
                f"📊 ESTADO {modo} | "
                f"Equidad: ${current_equity:.2f} | "
                f"Margen Usado: ${used_margin:.2f} | "
                f"Nivel Margen: {margin_str} {health_icon} | "
                f"Libre: ${free_margin:.2f} | "
                f"Posiciones: {n_trades}"
            )
        except Exception as e:
            logger.error(f"Error generando el reporte de estado de cuenta: {e}")

    def run(self):
        freq = int(self.params.candle_frecuency.replace("min", ""))
        logger.info("Strategy corriendo. Ejecución cada %d minutos.", freq)
        next_tick = (datetime.now() + timedelta(minutes=1)).replace(second=0, microsecond=0)

        while True:
            sleep_seconds = (next_tick - datetime.now()).total_seconds()
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            now = datetime.now()

            if now.minute % freq == 0:
                logger.info(f"Ejecución de estrategia a las {now.strftime('%H:%M:%S')}")
                self.get_candles()

                if self.candles.empty:
                    logger.info("No hay velas para operar.")
                else:
                    logger.info("Últimas 5 velas.\n%s", self.candles.tail(5).to_string())
                    self.manage_positions()

                    # Reporte de estado de cuenta
                    self.log_account_status()

            next_tick += timedelta(minutes=1)