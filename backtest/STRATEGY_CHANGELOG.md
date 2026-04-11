## 2025-05-12

```python
params = {
	'initial_cash_balance': 2000,
	'leverage': 200.0,
	'contract_multiplier': 50,
	'max_drawdown_pct': 80,
	'position_size': 0.02,
	'fast_ema': 11,
	'slow_ema': 32,
	'trend_ema': 200,
	'take_profit_long': 4.4,
	'take_profit_short': 4.1,
	'stop_loss_long': 2.9,
	'stop_loss_short': 3.8,
	'max_long_positions': 6,
	'max_short_positions': 15
}
```

## 2025-05-19

```python
params = {
	'initial_cash_balance': 2000,
	'leverage': 200.0,
	'contract_multiplier': 50,
	'max_drawdown_pct': 80,
	'position_size': 0.03,
	'fast_ema': 5,
	'slow_ema': 66,
	'trend_ema': 200,
	'take_profit_long': 3.3,
	'take_profit_short': 3.7,
	'stop_loss_long': 3.2,
	'stop_loss_short': 3.7,
	'max_long_positions': 5,
	'max_short_positions': 11
}
```

## 2025-06-10

```python
params = {
	'initial_cash_balance': 3000,
	'leverage': 200.0,
	'contract_multiplier': 50,
	'max_drawdown_pct': 80,
	'position_size': 0.03,
	'fast_ema': 5,
	'slow_ema': 65,
	'trend_ema': 200,
	'take_profit_long': 5.0,
	'take_profit_short': 4.5,
	'stop_loss_long': 2.1,
	'stop_loss_short': 3.5,
	'max_long_positions': 5,
	'max_short_positions': 14
}
```

## 2025-06-10 (IG First Version)

```python
params = {
	'initial_cash_balance': 3000,
	'leverage': 200.0,
	'contract_multiplier': 1,
	'max_drawdown_pct': 80,
	'position_size': 1.5,
	'fast_ema': 6,
	'slow_ema': 61,
	'trend_ema': 200,
	'take_profit_long': 2.8,
	'take_profit_short': 1.6,
	'stop_loss_long': 3.4,
	'stop_loss_short': 3.5,
	'max_long_positions': 5,
	'max_short_positions': 14
}
```

## 2025-10-17 (New strategy with only 1 EMA price crossover)

```python
    params = {
        'initial_cash_balance': 3000,
        'leverage': 200.0,
        'contract_multiplier': 1,
        'max_drawdown_pct': 80,
        'position_size': 1,
        'fast_ema': 21,
        'slow_ema': 61,
        'trend_ema': 200,
        'take_profit_long': 0.32,
	    'take_profit_short': 0.5,
        'stop_loss_long': 2.51,
        'stop_loss_short': 5.3,
        'max_long_positions': 10,
        'max_short_positions': 8
    }
```

## 2025-10-22

```python
params = {
        'initial_cash_balance': 3000,
        'leverage': 200.0,
        'contract_multiplier': 1,
        'max_drawdown_pct': 80,
        'position_size': 2,
        'fast_ema': 8,
        'take_profit_long': 0.17,
	    'take_profit_short': 0.17,
        'stop_loss_long': 1.6,
        'stop_loss_short': 1.7,
        'max_long_positions': 20,
        'max_short_positions': 5
    }
```

## 2025-10-27 (Optimized strategy with overbought, oversold, and ATR levels to avoid no-optimimal entries)

```python
params = {
        'initial_cash_balance': 10000,
        'leverage': 200.0,
        'contract_multiplier': 1,
        'max_drawdown_pct': 80,
        'position_size': 2,
        'fast_ema': 13,
        'take_profit_long': 0.2,
	    'take_profit_short': 0.1,
        'stop_loss_long': 1.96,
        'stop_loss_short': 2.0,
        'max_long_positions': 19,
        'max_short_positions': 1,
        'rsi_overbought':70,
        'rsi_oversold': 34,
        'atr_percentile': 14,
        'silent_mode': False,
        'objective_type': 'single',
    }
```