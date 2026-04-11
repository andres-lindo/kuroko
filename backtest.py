from backtesting import Strategy
import talib
import pandas as pd

class RSIBollingerStrategy(Strategy):
    # Parámetros que también configuran el motor
    initial_cash_balance = 3000
    leverage = 20
    commission = 0.00012
    silent_mode = False

    # Parámetros operativos
    max_positions = 5
    min_dist_between_entries_ticks = 16.0
    martingale_multiplier = 1.5
    bb_dev = 1.8
    bb_period = 22
    rsi_period = 10
    rsi_overbought = 80
    rsi_oversold = 38
    use_trend_filter = False
    take_profit_ticks = 60.0
    atr_period = 20
    atr_sl_multiplier = 12.0
    max_holding_hours = 72.0
    
    # Variables internas (no se optimizan normalmente)
    contract_multiplier = 1
    position_size = 1
    max_drawdown_pct = 80
    ema_period = 200    

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

    def log(self, txt):
        if not self.silent_mode:
            print(f"[{self.data.index[-1]}] {txt}")

    def next(self):
        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]
        trades = self.trades
        n_trades = len(trades)

        # --- 0. SALIDAS (TIME STOP & BASKET TP) ---
        if n_trades > 0:
            # Time Stop
            duration = (self.data.index[-1] - self.data.index[trades[0].entry_bar]).total_seconds() / 3600
            if duration > self.max_holding_hours:
                self.log(f"⏰ TIME STOP | {duration:.1f}h | PnL: ${sum(t.pl for t in trades):.2f}")
                for t in trades: t.close()
                return

            # Basket TP
            avg_price = sum(t.size * t.entry_price for t in trades) / sum(t.size for t in trades)
            total_size = sum(t.size for t in trades)

            if trades[0].is_long:
                if price >= avg_price + self.take_profit_ticks:
                    self.log(f"💰 WIN (LONG) | Size: {total_size} | Profit: ${(price - avg_price) * total_size:.2f}")
                    for t in trades: t.close()
                    return
            else:
                if price <= avg_price - self.take_profit_ticks:
                    self.log(f"💰 WIN (SHORT) | Size: {total_size} | Profit: ${(avg_price - price) * total_size:.2f}")
                    for t in trades: t.close()
                    return

        # --- 1. SEGURIDAD: MAX DRAWDOWN (FREEZE) ---
        floor_value = self.initial_cash_balance * (1 - (self.max_drawdown_pct / 100))
        if self.equity < floor_value:
            if not self.max_drawdown_reached:
                self.log(f"⚠️ MAX DD HIT! Equity ${self.equity:.2f} < ${floor_value:.2f}. Congelando.")
                self.max_drawdown_reached = True
        elif self.max_drawdown_reached and self.equity > floor_value:
            self.log(f"✅ RECUPERADO. Reactivando compras.")
            self.max_drawdown_reached = False

        # --- 2. ENTRADAS (GRID) ---
        if self.max_drawdown_reached or n_trades >= self.max_positions: return

        if n_trades > 0:
            if abs(price - trades[-1].entry_price) < self.min_dist_between_entries_ticks: return

        # Filtros
        sl_dist = atr * self.atr_sl_multiplier
        can_buy = price > self.ema[-1] if self.use_trend_filter else True
        can_sell = price < self.ema[-1] if self.use_trend_filter else True

        # Sizing
        current_size = max(1, round(self.position_size * (self.martingale_multiplier ** n_trades)))

        # --- 3. MARGIN CHECK (REALITY CHECK) ---
        cost_to_open = (price / self.leverage) * current_size
        used_margin = sum((t.entry_price / self.leverage) * t.size for t in trades)
        free_margin = self.equity - used_margin

        if cost_to_open > free_margin:
            self.log(f"🚫 NO MARGIN | Req: ${cost_to_open:.2f} > Free: ${free_margin:.2f}")
            return

        # --- 4. EJECUCIÓN ---
        if can_buy and price < self.bb_lower[-1] and rsi < self.rsi_oversold:
            if n_trades > 0 and (trades[0].is_short or price >= trades[-1].entry_price): return 
            self.buy(size=current_size, sl=price - sl_dist)
            self.log(f"⬆️  BUY #{n_trades + 1} (x{current_size}) @ {price:.2f}")

        elif can_sell and price > self.bb_upper[-1] and rsi > self.rsi_overbought:
            if n_trades > 0 and (trades[0].is_long or price <= trades[-1].entry_price): return 
            self.sell(size=current_size, sl=price + sl_dist)
            self.log(f"⬇️  SELL #{n_trades + 1} (x{current_size}) @ {price:.2f}")
