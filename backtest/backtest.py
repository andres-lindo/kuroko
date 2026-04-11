from backtesting import Backtest
import pandas as pd
import os
import argparse
import importlib
import warnings
import logging
import sys

def setup_logging(filename="backtest-last-execution.log"):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(filename, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# Suprimir warning de fractional trading cuando usamos leverage
warnings.filterwarnings('ignore', category=UserWarning, message='.*fractional trading.*')

# --- Registro de Estrategias ---
STRATEGY_REGISTRY = {
    "EMACrossoverStrategy": "strategies.ema_crossover",
    "RSIBollingerStrategy": "strategies.rsi_bollinger",
}

def load_raw_data(csv_file):
    """Solo carga el CSV en memoria, sin procesar"""
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")
    
    # Carga rápida, sin parsear fechas aun para no perder tiempo si la estrategia no lo necesita
    return pd.read_csv(csv_file, low_memory=False)

def load_strategy_class(strategy_name):
    """Convierte el string 'strategy_name' en la clase real"""
    if strategy_name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Estrategia '{strategy_name}' no registrada. Disponibles: {available}")
    
    # Magia de importación dinámica
    module_path = STRATEGY_REGISTRY[strategy_name]
    module = importlib.import_module(module_path)
    return getattr(module, strategy_name)

def get_strategy_params(strategy_name):
    """Devuelve los parámetros por defecto para la estrategia especificada"""
    STRATEGY_PARAMS = {
        'EMACrossoverStrategy': {
            # --- Fechas de Backtest ---
            'start_date': '2025-06-01',
            'end_date': '2026-02-13',
            'dataset': 'es_intraday-5min.csv',
            # --- Configuración del Motor ---
            'initial_cash_balance': 3000,
            'leverage': 20,
            'commission': 0.00012,
            'silent_mode': False,
            'max_margin_equity_pct': 85,
            'objective_type': 'single',
            # --- Parámetros de la Estrategia ---
            "fast_ema": 19,
            "take_profit_long": 0.31,
            "take_profit_short": 0.31,
            "stop_loss_long": 1.6800000000000002,
            "stop_loss_short": 1.4200000000000002,
            "max_long_positions": 1,
            "max_short_positions": 6,
            "rsi_overbought": 67.0,
            "rsi_oversold": 62.0,
            "atr_percentile": 17.0
        },
        "RSIBollingerStrategy": {
            # --- Fechas de Backtest ---
            "start_date": "2026-01-01",  # Fecha de inicio de la simulación.
            "end_date": "2026-04-10",  # Fecha de fin de la simulación.
            "dataset": "nq_intraday-15min.csv",  # Archivo de datos (Futuros ES 15min).
            # --- Configuración del Motor ---
            "initial_cash_balance": 400000,  # Capital inicial en USD.
            "leverage": 20.0,  # CRÍTICO: Apalancamiento RETAIL (1:20). Requiere 5% de margen ($345/contrato).
            "commission": 0.00012,  # Comisión simulada por operación.
            "silent_mode": False,  # False = Muestra logs detallados de cada compra/venta en consola.
            "objective_type": "single",  # Optimización enfocada en un solo objetivo (Maximizar Equity).
            # --- Parámetros de la Estrategia ---
            "max_positions": 5,  # Límite de seguridad: Máximo 5 niveles de compras escalonadas para no quemar margen.
            "min_dist_between_entries_ticks": 100.0,  # Espacio entre compras: 16 ticks = 4 Puntos ES. Evita comprar muy seguido.
            "martingale_multiplier": 1.5,  # Factor de Martingala: Aumenta el tamaño (x1, x1.5, x2.25...) para promediar precio agresivamente.
            "bb_dev": 1.9,  # Desviación Estándar. 1.7 es "sensible", entra antes de llegar a los extremos absolutos (2.0).
            "bb_period": 20,  # Periodo base para las Bandas de Bollinger (Medición de volatilidad media).
            "rsi_period": 11,  # Periodo estándar del oscilador RSI.
            "rsi_overbought": 76,  # Umbral de VENTA (Short). Solo vende si el RSI sube de 77 (mercado muy caliente).
            "rsi_oversold": 25,  # Umbral de COMPRA (Long). Solo compra si el RSI baja de 39 (mercado barato).
            "use_trend_filter": False,  # False = Estrategia "Mean Reversion" pura (apuesta al rebote, ignora la tendencia general).
            "take_profit_ticks": 240.0,  # Objetivo de Ganancia: 56 ticks = 14 Puntos desde el PRECIO PROMEDIO de la cesta.
            "atr_period": 12,  # Periodo del ATR para medir qué tan "nervioso" está el mercado hoy.
            "atr_sl_multiplier": 11.0,  # Stop Loss Dinámico: 12 veces el ATR. Es un Stop muy lejano para dejar respirar a la Martingala.
        },
        # Add more strategy configurations here
    }
    
    if strategy_name not in STRATEGY_PARAMS:
        raise ValueError(f"No parameters configured for strategy '{strategy_name}'")
    
    return STRATEGY_PARAMS[strategy_name].copy()

def generate_plot(bt_instance, filename="backtest_result.html"):
    """Generates the interactive HTML plot without resampling data."""
    try:
        # resample=False is critical to avoid 'Length of values' errors
        bt_instance.plot(filename=filename, open_browser=False, resample=False) 
        full_path = os.path.abspath(filename)
        
        print(f"📈 Plot generated successfully!")
        print(f"🔗 Path: {full_path}", end="\n\n")
    except Exception as e:
        print(f"❌ Could not generate plot: {e}", end="\n\n")

def run(data, params, strategy_class=None):
    """Ejecuta el backtest con los parámetros especificados"""

    # 1. Extraer configuración del motor (con valores por defecto si no están en params)
    initial_cash = params.get('initial_cash_balance')
    leverage = params.get('leverage')
    commission = params.get('commission')

    silent_mode = params.pop('silent_mode')
    objective_type = params.pop('objective_type')

    # Inyectar silent_mode a la clase (truco para que el log interno de la estrategia lo vea)
    strategy_class.silent_mode = silent_mode

    # 2. Instanciar Motor
    bt = Backtest(
        data,
        strategy_class,
        cash=initial_cash,
        commission=commission,
        margin=1/leverage,
        exclusive_orders=False,
        trade_on_close=True,
        hedging=True
    )

    # 3. Ejecutar Backtest (pasando solo los parámetros de la estrategia)
    stats = bt.run(**params)

    trades_df = stats._trades
    strategy_instance = stats['_strategy']
    
    if getattr(strategy_instance, 'max_drawdown_reached', False):
        if silent_mode:
            if objective_type == 'single':
                return 0.0 
            return {
                "portfolio_value": -100.0,
                "win_rate": -100.0,
                "max_drawdown": -100.0,
                "avg_loss": -100,
                "sortino_ratio": -100.0,
                "sharpe_ratio": -100.0,
            }
        else:
            return stats

    # Caso 1: Silent mode + Objective Single (Optimización rápida)
    if silent_mode and objective_type == 'single':
        portfolio_value = stats['Equity Final [$]']
        print(f"Equity: {portfolio_value:.2f}")
        return round(portfolio_value, 1)

    # Cálculos de métricas
    closed_trades = trades_df[trades_df["ExitTime"].notna()]
    n_winning = len(closed_trades[closed_trades["PnL"] > 0])
    n_losing = len(closed_trades[closed_trades["PnL"] < 0])
    win_rate = (n_winning / len(closed_trades) * 100) if len(closed_trades) > 0 else 0
    avg_loss = (closed_trades[closed_trades["PnL"] < 0]["PnL"].mean() if n_losing > 0 else 0)

    final_equity = stats['Equity Final [$]']
    max_dd = stats['Max. Drawdown [%]']
    sharpe_ratio = stats['Sharpe Ratio']
    sortino_ratio = stats['Sortino Ratio']

    # Silent mode: objective multiple/weighted
    if silent_mode and objective_type in ("multiple", "weighted"):
        print(
            f"Portfolio Value: {final_equity:.2f}. Trades: {len(trades_df)}. Closed: {len(closed_trades)}. Win: {n_winning} ({win_rate:.2f}%). Loss: {n_losing}. Drawdown: {max_dd:.2f}. Net Profit: {final_equity - params['initial_cash_balance']:.2f}."
        )
        return {
            "portfolio_value": round(final_equity, 1),
            "win_rate": round(win_rate, 1),
            "max_drawdown": round(max_dd, 1),
            "avg_loss": round(avg_loss, 0),
            "sortino_ratio": sortino_ratio,
            "sharpe_ratio": sharpe_ratio,
        }

    # Caso 4: Modo Normal (Reporte Completo)
    print(f"\n{stats}\n")
    print("="*60)
    print("Operations Summary:")
    print("="*60)
    print(f"Total Trades: {len(trades_df)}")
    print(f"Closed Trades: {len(closed_trades)}")
    print(f"Winning: {n_winning} ({win_rate:.2f}%)")
    print(f"Losing: {n_losing}")
    print(f"Avg Win: {closed_trades[closed_trades['PnL'] > 0]['PnL'].mean():.2f}" if n_winning > 0 else "Avg Win: 0.00")
    print(f"Avg Loss: {avg_loss:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Sortino Ratio: {sortino_ratio}")
    print(f"Initial Capital: {params['initial_cash_balance']:.2f}")
    print(f"Final Capital: {final_equity:.2f}")
    print(f"Net Profit: {final_equity - params['initial_cash_balance']:.2f}")
    print("="*60)

    # Generar gráfico interactivo
    if not silent_mode:
        generate_plot(bt)

    return stats

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', nargs='?', default='EMACrossoverStrategy')
    args = parser.parse_args()

    setup_logging(f"{args.strategy}-backtest-last-execution.log")

    # Obtener Clase
    StrategyClass = load_strategy_class(args.strategy)
    
    # Obtener Parámetros del Panel de Control
    params = get_strategy_params(args.strategy)

    # Obtener configuracion de dataset y fechas
    csv_filename = params.pop('dataset')
    start_date = params.pop('start_date')
    end_date = params.pop('end_date')

    # Rutas y Fechas
    script_dir = os.path.dirname(__file__)
    csv_file = os.path.join(script_dir, 'datasets', csv_filename)

    print(f"Running backtest with strategy: {args.strategy}")
    
    # 3. Cargar Data Cruda
    raw_df = load_raw_data(csv_file)

    # 4. Preparar Data (Responsabilidad de la Estrategia)
    # Si la estrategia no tiene prepare_data, fallará aquí (lo cual es deseado para forzar la implementación)
    data = StrategyClass.prepare_data(raw_df, start_date, end_date)

    # 5. Ejecutar
    run(data, params, StrategyClass)
