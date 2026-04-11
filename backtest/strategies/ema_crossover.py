from backtesting import Strategy
from enum import Enum
import pandas as pd
import numpy as np
import talib

# Enum para las razones de cierre
class CloseReason(Enum):
    CROSS_OVER = "Crossover"
    TAKE_PROFIT = "Take Profit"
    STOP_LOSS = "Stop Loss"

class EMACrossoverStrategy(Strategy):
    # Parámetros que también configuran el motor
    initial_cash_balance = 3000
    leverage = 20
    commission = 0.00012
    silent_mode = False

    # Parámetros operativos
    fast_ema = 8
    take_profit_long = 0.15
    take_profit_short = 0.11
    stop_loss_long = 0.98     
    stop_loss_short = 0.89
    max_long_positions = 1
    max_short_positions = 0
    rsi_overbought = 70
    rsi_oversold = 60
    atr_percentile = 11
    
    # Variables internas (no se optimizan normalmente)
    contract_multiplier = 1     
    position_size = 1
    max_margin_equity_pct = 80    
    
    @classmethod
    def prepare_data(cls, df, start_date, end_date):
        """Prepara RSI 1h y resampleo específico para EMA Crossover"""
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df.set_index('Time', inplace=True)
            
        if 'Last' in df.columns:
            df.rename(columns={'Last': 'Close'}, inplace=True)

        df = df[(df.index >= start_date) & (df.index <= end_date)].copy()

        df_1h = df.resample('1h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        df_1h['RSI'] = talib.RSI(df_1h['Close'].values, timeperiod=14)
        
        df = df.join(df_1h[['RSI']], how='left')
        df['RSI'] = df['RSI'].ffill()
        df = df.dropna(subset=['RSI'])
        
        return df

    def init(self):
        self.max_drawdown_reached = False
    
    def log(self, txt):
        """Log messages"""
        if not self.silent_mode:
            dt = self.data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            used_margin = sum(t.size * t.entry_price * self.contract_multiplier / self.leverage 
                        for t in self.trades if t.entry_price)
            print(f"{dt}, {txt}, Equity: {self.equity:.2f}. Used Margin: {used_margin:.2f}")

    def calculate_indicators(self):
        """Calcula todos los indicadores necesarios"""
        # ATR(7) con velas de 5 min
        atr_values = talib.ATR(self.data.High, self.data.Low, self.data.Close, timeperiod=7)
        
        # EMA con velas de 5 min - devolver array completo
        ema_values = talib.EMA(self.data.Close, timeperiod=self.fast_ema)
        
        # RSI(14) de 1 hora (ya viene pre-calculado)
        rsi_1h = self.data.RSI[-1]
        
        return atr_values, ema_values, rsi_1h

    def manage_positions(self, last_close, cross_up, cross_down):
        """Gestiona las posiciones LONG y SHORT activas"""
        for trade in self.trades:
            # Solo trades activos
            if trade.entry_price is None:
                continue
            
            entry_price = trade.entry_price
            size = trade.size
            
            # LONG positions
            if trade.is_long:
                profit_usd = (last_close - entry_price) * self.contract_multiplier * size
                profit_pct = (last_close / entry_price - 1) * 100

                # Exit conditions
                positive_gain = last_close > entry_price
                tp_long = positive_gain and (profit_pct >= self.take_profit_long)
                sl_long = not positive_gain and (profit_pct <= -self.stop_loss_long)

                if tp_long or sl_long:
                    trade.close()

                    if tp_long:
                        exit_reason = CloseReason.TAKE_PROFIT.value
                    elif sl_long:
                        exit_reason = CloseReason.STOP_LOSS.value    

                    if exit_reason == CloseReason.STOP_LOSS.value:
                        self.log(f"🔴 SL LONG. Reason: {exit_reason}. Exit Price: {last_close:.2f}. Entry Price: {entry_price:.2f}. Size: {size}. Est. Loss: {-profit_usd:.2f}. Pct: {-profit_pct:.2f}")
                    else:    
                        self.log(f"🟢 TP LONG. Reason: {exit_reason}. Exit Price: {last_close:.2f}. Entry Price: {entry_price:.2f}. Size: {size}. Est. Profit: {profit_usd:.2f}. Pct: {profit_pct:.2f}")
            
            # SHORT positions
            elif trade.is_short:
                profit_usd = (entry_price - last_close) * self.contract_multiplier * size
                profit_pct = (entry_price / last_close - 1) * 100

                # Exit conditions
                positive_gain = last_close < entry_price
                tp_short = positive_gain and (profit_pct >= self.take_profit_short)
                sl_short = not positive_gain and (profit_pct <= -self.stop_loss_short)

                if tp_short or sl_short:
                    trade.close()

                    if tp_short:
                        exit_reason = CloseReason.TAKE_PROFIT.value
                    elif sl_short:
                        exit_reason = CloseReason.STOP_LOSS.value

                    if exit_reason == CloseReason.STOP_LOSS.value:
                        self.log(f"🔴 SL SHORT. Reason: {exit_reason}. Exit Price: {last_close:.2f}. Entry Price: {entry_price:.2f}. Size: {size}. Est. Loss: {-profit_usd:.2f}. Pct: {-profit_pct:.2f}")
                    else:    
                        self.log(f"🟢 TP SHORT. Reason: {exit_reason}. Exit Price: {last_close:.2f}. Entry Price: {entry_price:.2f}. Size: {size}. Est. Profit: {abs(profit_usd):.2f}. Pct: {profit_pct:.2f}")

    def open_positions(self, last_close, cross_up, cross_down, atr, last_valid_rsi, atr_values):
        """Abre nuevas posiciones LONG o SHORT según las condiciones"""
        # Count active trades
        open_long = [t for t in self.trades if t.is_long and t.entry_price]
        open_short = [t for t in self.trades if t.is_short and t.entry_price]

        # Position size: siempre 1 contrato completo
        #position_size_contracts = self.position_size

        # Position size dinamico
        dynamic_size = int(self.equity // 3000) 
        position_size_contracts = max(1, dynamic_size)
        
        # Get position size in USD
        position_size_usd = position_size_contracts * last_close * self.contract_multiplier

        # Calculate required margin
        required_margin = position_size_usd / self.leverage
        
        # Calcular margen usado correctamente
        used_margin = sum(t.size * t.entry_price * self.contract_multiplier / self.leverage 
                        for t in self.trades if t.entry_price)
        
        # Margen disponible = equity - margen usado
        available_margin = self.equity - used_margin

        is_valid_margin = True

        if (used_margin + required_margin) > (self.equity * (self.max_margin_equity_pct / 100)):
            is_valid_margin = False
        
        # RSI filter conditions
        is_good_long = True
        is_good_short = True

        if last_valid_rsi is not None:
            is_good_long = last_valid_rsi <= self.rsi_overbought
            is_good_short = last_valid_rsi >= self.rsi_oversold

        # ATR filter conditions
        is_good_volatility = True
        atr_last_100 = atr_values[-100:]
        
        if len(atr_last_100) >= 10:
            last_atr = atr
            if last_atr < np.percentile(atr_last_100, self.atr_percentile):
                is_good_volatility = False    

        # Enter LONG
        if cross_up:           
            if len(open_long) >= self.max_long_positions:
                self.log(f"⚠️  Max number of long positions reached. {self.max_long_positions}.")
                return
            
            if not is_good_volatility:
                self.log(f"⚠️  Low volatility. ATR(7): {last_atr:.2f}.")
                return
            
            if not is_good_long:
                self.log(f"⚠️  RSI on 1h is not optimal for LONG. RSI(14): {last_valid_rsi:.1f}")
                return
            
            if not is_valid_margin:
                self.log(f"⚠️ SKIP LONG: Margin usage too high. Required Margin with new position: {used_margin + required_margin:.2f}. Equity ({self.max_margin_equity_pct}%): {(self.equity * (self.max_margin_equity_pct / 100)):.2f} Available Margin: {available_margin:.2f}.")
                return

            self.buy(size=position_size_contracts)
            self.log(f"↗️  Opening LONG. Price: {last_close:.2f}. Contracts: {position_size_contracts}. Available Margin: {available_margin:.2f}. Size USD: {position_size_usd:.2f}. RSI: {last_valid_rsi:.1f}")

        # Enter SHORT
        if cross_down:           
            if len(open_short) >= self.max_short_positions:
                self.log(f"⚠️  Max number of SHORT positions reached. {self.max_short_positions}.")
                return
            
            if not is_good_volatility:
                self.log(f"⚠️  Low volatility. ATR(7): {last_atr:.2f}.")
                return
            
            if not is_good_short:
                self.log(f"⚠️  RSI on 1h is not optimal for SHORT. RSI(14): {last_valid_rsi:.1f}")
                return
            
            if not is_valid_margin:
                self.log(f"⚠️ SKIP SHORT: Margin usage too high. Required Margin with new position: {used_margin + required_margin:.2f}. Equity ({self.max_margin_equity_pct}%): {(self.equity * (self.max_margin_equity_pct / 100)):.2f} Available Margin: {available_margin:.2f}.")
                return
            
            self.sell(size=position_size_contracts)
            self.log(f"↘️  Opening SHORT. Price: {last_close:.2f}. Contracts: {position_size_contracts}. Available Margin: {available_margin:.2f}. Size USD: {position_size_usd:.2f}. RSI: {last_valid_rsi:.1f}")

    def next(self):
        if self.max_drawdown_reached:
            return
        
        if self.equity < self.initial_cash_balance * 0.3:
            self.log(f"🛑 BAD STRATEGY: Equity below {30}% of intial balance ({self.initial_cash_balance:.2f}).")
            for trade in self.trades:
                trade.close()
            self.max_drawdown_reached = True
            return

        used_margin = sum(t.size * t.entry_price * self.contract_multiplier / self.leverage 
                        for t in self.trades if t.entry_price)
        
        margin_threshold = self.equity * (self.max_margin_equity_pct / 100)
        if used_margin > margin_threshold:
            self.log(f"🛑 MARGIN CRITICAL: Used margin ({used_margin:.2f}) exceeds {self.max_margin_equity_pct}% of equity ({margin_threshold:.2f}).")
            for trade in self.trades:
                trade.close()
            self.max_drawdown_reached = True
            return

        # if self.equity < (self.initial_cash_balance * (1 - self.max_drawdown_pct / 100)):
        #     self.log(f"⚠️  MAX DRAWDOWN REACHED! Portfolio value ({self.equity:.2f}) dropped below {self.max_drawdown_pct}% of the initial balance.")
        #     for trade in self.trades:
        #         trade.close()
        #     self.max_drawdown_reached = True
        #     return
        
        # Calcular indicadores
        atr_values, ema_values, rsi_1h = self.calculate_indicators()

        # Necesitamos al menos 3 barras para comparar
        if len(self.data.Close) < 3:
            return
        
        # Current and previous values
        fast_ema = ema_values[-1]
        prev_fast_ema = ema_values[-2]
        prev2_fast_ema = ema_values[-3]
        last_close = self.data.Close[-1]
        prev_close = self.data.Close[-2]
        prev2_close = self.data.Close[-3]

        atr = atr_values[-1]
        last_valid_rsi = rsi_1h

        # Entry conditions - EMA crossover with 2-bar confirmation
        cross_up = (
            (prev2_close <= prev2_fast_ema) and
            (prev_close > prev_fast_ema) and
            (last_close > fast_ema)
        )

        cross_down = (
            (prev2_close >= prev2_fast_ema) and
            (prev_close < prev_fast_ema) and
            (last_close < fast_ema)
        )

        # Gestionar posiciones activas
        self.manage_positions(last_close, cross_up, cross_down)

        # Abrir nuevas posiciones
        self.open_positions(last_close, cross_up, cross_down, atr, last_valid_rsi, atr_values)