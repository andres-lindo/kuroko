from backtesting import Strategy
import talib
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class RSIBollingerStrategy(Strategy):
    # Parámetros que también configuran el motor
    initial_cash_balance = 300000
    leverage = 20
    commission = 0.00012
    silent_mode = False

    # Parámetros operativos (Optimizables)
    max_positions = 5
    min_dist_between_entries_ticks = 16.0
    martingale_multiplier = 1.5
    bb_dev = 1.5
    bb_period = 23
    rsi_period = 14
    rsi_overbought = 77
    rsi_oversold = 40
    use_trend_filter = False
    take_profit_ticks = 60.0
    atr_period = 11
    atr_sl_multiplier = 12.0
    
    # Variables internas (no se optimizan normalmente)
    contract_multiplier = 1
    position_size = 13           # 20 enteros simulando 0.2
    max_drawdown_pct = 80
    ema_period = 200
    security_buffer = 100000.0    # 50000 simulando 500

    # Debugging
    log_all_candles = False

    @classmethod
    def prepare_data(cls, df, start_date, end_date):
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df.set_index('Time', inplace=True)

        rename_map = {
            'Last': 'Close', 'last': 'Close', 'close': 'Close', 
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'volume': 'Volume', 'vol': 'Volume'
        }
        df.rename(columns={c: rename_map.get(c, c) for c in df.columns}, inplace=True)
        return df[(df.index >= start_date) & (df.index <= end_date)].copy()

    def init(self):
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=self.bb_period, nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0
        )
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.max_drawdown_reached = False 

        # NUEVO: Memoria para el detector de TP/SL
        self.last_closed_count = 0

    def log(self, txt):
         if not self.silent_mode:
            dt = self.data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            price = self.data.Close[-1]
            used_margin = sum((price / self.leverage) * abs(t.size) for t in self.trades)
            margin_level_pct = (self.equity / used_margin) * 100 if used_margin > 0 else 1000
            free_margin = self.equity - used_margin
            logger.info(f"{dt}, {txt}, Price: {price:.2f}, Equity: {self.equity:.2f}. Used Margin: {abs(used_margin):.2f}. Margin Level: {margin_level_pct:.2f}%. Free Margin: {free_margin:.2f}")

    def next(self):
        if self.max_drawdown_reached:
            return

        # --- DETECTOR DE TP/SL DEL BROKER SIMULADO ---
        current_closed_count = len(self.closed_trades)
        if current_closed_count > self.last_closed_count:
            # Obtenemos solo las operaciones que se acaban de cerrar en esta vela
            newly_closed = self.closed_trades[self.last_closed_count:]
            
            # Sumamos los resultados de la canasta
            total_profit = sum(t.pl for t in newly_closed)
            total_size = sum(abs(t.size) for t in newly_closed)
            
            # Si el PnL es positivo, tocó TP. Si es negativo, tocó SL.
            if total_profit > 0:
                self.log(f"TAKE PROFIT AUTOMATICO | Cerradas: {len(newly_closed)} pos | Size: {total_size} | Profit Neto: ${total_profit:.2f}")
            else:
                self.log(f"STOP LOSS AUTOMATICO | Cerradas: {len(newly_closed)} pos | Size: {total_size} | Pérdida Neta: ${total_profit:.2f}")
            
            # Actualizamos la memoria
            self.last_closed_count = current_closed_count
        # ---------------------------------------------

        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]
        trades = self.trades
        n_trades = len(trades)

        if self.log_all_candles:
            self.log("OK") 

        # --- 0. SALIDAS (BASKET TP MANUAL FALLBACK) ---
        if n_trades > 0:
            avg_price = sum(abs(t.size) * t.entry_price for t in trades) / sum(abs(t.size) for t in trades)
            total_size_abs = sum(abs(t.size) for t in trades)

            if trades[0].is_long:
                if price >= avg_price + self.take_profit_ticks:
                    self.log(f"WIN (LONG) | Size: {total_size_abs} | Profit: ${(price - avg_price) * total_size_abs:.2f}")
                    for t in trades: t.close()
                    return
            else:
                if price <= avg_price - self.take_profit_ticks:
                    self.log(f"WIN (SHORT) | Size: {total_size_abs} | Profit: ${(avg_price - price) * total_size_abs:.2f}")
                    for t in trades: t.close()
                    return

        # MARGIN CALL AND LIQUIDATION
        used_margin = sum((price / self.leverage) * abs(t.size) for t in trades)
        margin_level_pct = (self.equity / used_margin) * 100 if used_margin > 0 else 1000

        if margin_level_pct <= 100 and margin_level_pct > 50:
            self.log(f"MARGIN CALL: Level {margin_level_pct:.2f}% <= 100%. Equity: ${self.equity:.2f}, Used Margin: ${used_margin:.2f}")

        # Stop Out: El broker cierra las posiciones a la fuerza
        if margin_level_pct <= 50:
            self.log(f"LIQUIDATION: Level {margin_level_pct:.2f}% <= 50%. Equity: ${self.equity:.2f}, Used Margin: ${used_margin:.2f}")
            for t in trades: 
                t.close()  # Ejecución forzosa del broker
            self.max_drawdown_reached = True 
            return

        # --- 1. SEGURIDAD: MAX DRAWDOWN (FREEZE) ---
        floor_value = self.initial_cash_balance * (1 - (self.max_drawdown_pct / 100))
        if self.equity < floor_value:
            if not self.max_drawdown_reached:
                self.log(f"MAX DD HIT! Equity ${self.equity:.2f} < ${floor_value:.2f}. Congelando.")
                self.max_drawdown_reached = True
        elif self.max_drawdown_reached and self.equity > floor_value:
            self.log(f"RECUPERADO. Reactivando compras.")
            self.max_drawdown_reached = False

        # --- 2. ENTRADAS (GRID) ---
        # Bloqueo si el Drawdown saltó o si ya agotamos las balas permitidas
        if self.max_drawdown_reached or n_trades >= self.max_positions:
            return

        if n_trades > 0:
            if abs(price - trades[-1].entry_price) < self.min_dist_between_entries_ticks: return

        # Filtros
        sl_dist = atr * self.atr_sl_multiplier
        can_buy = price > self.ema[-1] if self.use_trend_filter else True
        can_sell = price < self.ema[-1] if self.use_trend_filter else True

        # Sizing (Martingala fraccionada a 2 decimales)
        current_size = max(self.position_size, int(round(self.position_size * (self.martingale_multiplier ** n_trades))))

        # --- 3. MARGIN CHECK (REALITY CHECK) ---
        cost_to_open = (price / self.leverage) * current_size
        used_margin = sum((price / self.leverage) * abs(t.size) for t in trades)
        free_margin = self.equity - used_margin

        # MODIFICADO: Ahora respeta el security_buffer igual que IG
        if cost_to_open > (free_margin - self.security_buffer):
            self.log(f"NO MARGIN | Req: ${cost_to_open:.2f} | Libre: ${free_margin:.2f} | Buffer: ${self.security_buffer}")
            return

        # --- 4. CÁLCULO DE TARGET PRICE DINÁMICO ---
        if n_trades > 0:
            total_size = sum(abs(t.size) for t in trades)
            total_value = sum(abs(t.size) * t.entry_price for t in trades)
        else:
            total_size = 0
            total_value = 0

        futuro_total_size = total_size + current_size
        futuro_total_value = total_value + (current_size * price)
        futuro_avg_price = futuro_total_value / futuro_total_size

        # --- 5. EJECUCIÓN (CON ACTUALIZACIÓN DE TP Y SL EN LA LIBRERÍA) ---
        if can_buy and price < self.bb_lower[-1] and rsi < self.rsi_oversold:
            if n_trades > 0 and (trades[0].is_short or price >= trades[-1].entry_price): return 
            
            # Calculamos los niveles exactos simulando la orden a IG
            target_price = round(futuro_avg_price + self.take_profit_ticks, 2)
            stop_price = round(price - sl_dist, 2)

            self.buy(size=current_size, sl=stop_price, tp=target_price)
            self.log(f"BUY #{n_trades + 1} (x{current_size}) @ {price:.2f} | SL: {stop_price} | TP: {target_price}")
            
            for t in self.trades:
                t.sl = stop_price
                t.tp = target_price

        elif can_sell and price > self.bb_upper[-1] and rsi > self.rsi_overbought:
            if n_trades > 0 and (trades[0].is_long or price <= trades[-1].entry_price): return 
            
            target_price = round(futuro_avg_price - self.take_profit_ticks, 2)
            stop_price = round(price + sl_dist, 2)

            self.sell(size=current_size, sl=stop_price, tp=target_price)
            self.log(f"SELL #{n_trades + 1} (x{current_size}) @ {price:.2f} | SL: {stop_price} | TP: {target_price}")
            
            for t in self.trades:
                t.sl = stop_price
                t.tp = target_price