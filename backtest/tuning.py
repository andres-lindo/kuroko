import optuna
import json
import os
import logging
import argparse
from datetime import datetime
import random
import numpy as np
from backtest import load_raw_data, run, load_strategy_class

# --- Global Configurations ---
FILE_CONFIG = {
    'dataset_name': 'nq_intraday-15min.csv',
    'output_folder': 'tuning_output',
    'datasets_folder': 'datasets'
}

ENGINE_CONFIG = {
    'initial_cash_balance': 400000,
    'leverage': 20.0,
    'commission': 0.00012,
    'contract_multiplier': 1,
    'max_drawdown_pct': 80,
    'silent_mode': True
}

STRATEGY_SEARCH_SPACES = {
    'RSIBollingerStrategy': {
        'position_size':                  {'type': 'int',   'low': 10,    'high': 15},
        'max_positions':                  {'type': 'int',   'low': 3,    'high': 5},
        'min_dist_between_entries_ticks': {'type': 'float', 'low': 60.0,  'high': 150.0, 'step': 10.0},        
        'martingale_multiplier':          {'type': 'float', 'low': 1.0,  'high': 1.5,  'step': 0.1},

        'bb_dev':                         {'type': 'float', 'low': 1.5,  'high': 2.5,  'step': 0.1},
        'bb_period':                      {'type': 'int',   'low': 18,   'high': 24},
        'rsi_period':                     {'type': 'int',   'low': 10,   'high': 16},
        'rsi_overbought':                 {'type': 'float', 'low': 65,   'high': 80,   'step': 1.0},
        'rsi_oversold':                   {'type': 'float', 'low': 20,   'high': 40,   'step': 1.0},
        'use_trend_filter':               {'type': 'categorical', 'choices': [True, False]},
        
        'take_profit_ticks':              {'type': 'float', 'low': 80.0, 'high': 250.0, 'step': 10.0},        
        'atr_period':                     {'type': 'int', 'low': 10, 'high': 20},        
        'atr_sl_multiplier':              {'type': 'float', 'low': 6.0,  'high': 15.0, 'step': 1.0},
    }
}

# --- Setup & Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, default="RSIBollingerStrategy")
parser.add_argument("--start_date", type=str, required=True)
parser.add_argument("--end_date", type=str, required=True)
parser.add_argument("--objective_type", type=str, choices=["single", "multiple", "weighted"], required=True)
parser.add_argument("--trials", type=int, required=True)
args = parser.parse_args()

def get_trial_params(trial, strategy_name):
    """Convierte el diccionario declarativo en llamadas de Optuna"""
    config = STRATEGY_SEARCH_SPACES.get(strategy_name)
    params = {}
    
    for param_name, specs in config.items():
        p_type = specs['type']
        
        if p_type == 'int':
            params[param_name] = trial.suggest_int(param_name, specs['low'], specs['high'], step=specs.get('step', 1))
        elif p_type == 'float':
            params[param_name] = trial.suggest_float(param_name, specs['low'], specs['high'], step=specs.get('step'))
        elif p_type == 'categorical':
            params[param_name] = trial.suggest_categorical(param_name, specs['choices'])
            
    return params

if args.strategy not in STRATEGY_SEARCH_SPACES:
    raise ValueError(f"Optimization config not found for '{args.strategy}'")

script_dir = os.path.dirname(__file__)
output_dir = os.path.join(script_dir, FILE_CONFIG['output_folder'])
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M')
tuning_output_file = os.path.join(output_dir, f"tuning_{args.strategy}_{args.objective_type}_{timestamp}.json")
log_file = os.path.join(output_dir, f"tuning_{args.strategy}_{args.objective_type}_log_{timestamp}.txt")

logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
optuna.logging.enable_propagation()
optuna.logging.set_verbosity(optuna.logging.INFO)

# --- Data Loading ---
print(f"Loading Strategy: {args.strategy}...")
StrategyClass = load_strategy_class(args.strategy)

csv_path = os.path.join(script_dir, FILE_CONFIG['datasets_folder'], FILE_CONFIG['dataset_name'])
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found: {csv_path}")

print(f"Processing Data ({args.start_date} to {args.end_date})...")
raw_df = load_raw_data(csv_path)
DATA_PROCESSED = StrategyClass.prepare_data(raw_df, args.start_date, args.end_date)
print("Data ready.")

# --- Optimization Loop ---
def objective(trial):
    seed = trial.number
    random.seed(seed)
    np.random.seed(seed)
    
    dynamic_params = get_trial_params(trial, args.strategy)
    
    # Merge global static params + dynamic params + current objective type
    params = ENGINE_CONFIG.copy()
    params.update(dynamic_params)
    params['objective_type'] = args.objective_type
    
    result = run(DATA_PROCESSED, params, StrategyClass)

    # Handle missing or failed backtest results
    if result is None:
        if args.objective_type == "single":
            return float("-inf")
        elif args.objective_type == "multiple":
            # Se requieren 6 valores de retorno (coincidiendo con las direcciones de Optuna)
            return float("-inf"), -100.0, 0.0, -9999.0, 0.0, 0.0
        elif args.objective_type == "weighted":
            return float("-inf")

    # Single-objective
    if args.objective_type == "single":
        return result

    # Multi-objective (Pareto optimization)
    elif args.objective_type == "multiple":
        portfolio_value = result.get('portfolio_value', 0)
        max_drawdown = result.get('max_drawdown', -100)
        win_rate = result.get('win_rate', 0)
        avg_loss = result.get('avg_loss', -9999)
        sortino_ratio = result.get('sortino_ratio', 0)
        sharpe_ratio = result.get('sharpe_ratio', 0)

        # Si la estrategia es irreal (ej. 0 pérdidas genera Infinito), la castigamos con 0.0
        if np.isinf(sortino_ratio) or np.isnan(sortino_ratio):
            sortino_ratio = 0.0
        if np.isinf(sharpe_ratio) or np.isnan(sharpe_ratio):
            sharpe_ratio = 0.0

        return portfolio_value, max_drawdown, win_rate, avg_loss, sortino_ratio, sharpe_ratio

    # Weighted combination
    elif args.objective_type == "weighted":
        portfolio_value = result.get('portfolio_value', 0)
        max_drawdown = result.get('max_drawdown', 100)
        win_rate = result.get('win_rate', 0)
        score = (0.7 * portfolio_value) + (0.3 * -max_drawdown)
        return score

if __name__ == '__main__':

    logging.info("="*80)
    logging.info(f"STARTING OPTIMIZATION STUDY")
    logging.info(f"Strategy: {args.strategy}")
    logging.info(f"Objective Type: {args.objective_type}")
    logging.info(f"Trials: {args.trials}")
    logging.info(f"Date Range: {args.start_date} to {args.end_date}")
    
    search_space = STRATEGY_SEARCH_SPACES.get(args.strategy)
    logging.info(f"Search Space: {json.dumps(search_space, indent=2)}")
    logging.info(f"Engine Config: {json.dumps(ENGINE_CONFIG, indent=2)}")
    logging.info("="*80)

    print(f"Starting optimization ({args.trials} trials)...")
    
    if args.objective_type == "multiple":
        #Metric Optimization Direction: portfolio_value, max_drawdown, win_rate, avg_loss, sortino_ratio, sharpe_ratio
        study = optuna.create_study(directions=['maximize', 'maximize', 'maximize', 'maximize', 'maximize', 'maximize'])
        study.optimize(objective, n_trials=args.trials, n_jobs=-1)
        
        best_trials = sorted(
            [{'trial_number': t.number, 'values': t.values, 'params': t.params} for t in study.best_trials],
            key=lambda x: (
                # PRIORIDAD 1: "Ratio de Eficiencia" (Capital / Peor Drawdown)
                
                # Entre más dinero gane con menos drawdown, mayor será este puntaje.
                (x['values'][0] - ENGINE_CONFIG['initial_cash_balance']) / abs(x['values'][1] - 0.0001),

                # PRIORIDAD 2: Capital Neto (En caso de empate en eficiencia, dame el que da más dinero)
                x['values'][0],

                # PRIORIDAD 3: Sortino Ratio
                x['values'][4]
            ), 
            reverse=True 
        )
        with open(tuning_output_file, 'w') as f:
            json.dump(best_trials, f, indent=4)
        print(f"Optimization complete. Solutions: {len(best_trials)}")
        
    else:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=args.trials, n_jobs=-1)
        
        with open(tuning_output_file, 'w') as f:
            json.dump({'best_params': study.best_params, 'best_value': study.best_value}, f, indent=4)
        print(f"Optimization complete. Best Value: {study.best_value}")