"""Optuna-based hyperparameter optimisation driver for Kuroko backtest strategies.

Supports three objective modes:

- ``single``: maximise final equity (single scalar, fastest).
- ``multiple``: Pareto-front optimisation across six metrics
  (equity, drawdown, win rate, avg loss, Sortino, Sharpe).
- ``weighted``: weighted linear combination of equity and drawdown.

Run via:
    python tuning.py --strategy RSIBollingerStrategy \\
        --start_date 2026-01-01 --end_date 2026-04-10 \\
        --objective_type single --trials 100
"""

import optuna
import json
import os
import logging
import argparse
from datetime import datetime
import random
import numpy as np
from backtest import load_raw_data, run, load_strategy_class


def load_tuning_params() -> dict:
    """Load tuning configuration from tuning_params.json."""
    params_path = os.path.join(os.path.dirname(__file__), "tuning_params.json")
    with open(params_path) as f:
        return json.load(f)


# --- Global Configurations ---
_params = load_tuning_params()
FILE_CONFIG = _params["file"]
ENGINE_CONFIG = _params["engine"]
STRATEGY_SEARCH_SPACES = _params["search_spaces"]

# --- Setup & Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--strategy", type=str, default="RSIBollingerStrategy")
parser.add_argument("--start_date", type=str, required=True)
parser.add_argument("--end_date", type=str, required=True)
parser.add_argument(
    "--objective_type",
    type=str,
    choices=["single", "multiple", "weighted"],
    required=True,
)
parser.add_argument("--trials", type=int, required=True)
args = parser.parse_args()


def get_trial_params(trial, strategy_name):
    """Convert the declarative search-space config into Optuna suggest calls.

    Iterates over :data:`STRATEGY_SEARCH_SPACES` for the given strategy and
    dispatches each parameter to the appropriate ``trial.suggest_*`` method
    based on its ``type`` key (``'int'``, ``'float'``, or ``'categorical'``).

    Args:
        trial: The current :class:`optuna.Trial` object provided by the
            optimisation loop.
        strategy_name: Registered strategy name used to look up the search
            space in :data:`STRATEGY_SEARCH_SPACES`.

    Returns:
        Dict mapping parameter names to the values sampled by Optuna for
        this trial.
    """
    config = STRATEGY_SEARCH_SPACES.get(strategy_name)
    params = {}

    for param_name, specs in config.items():
        p_type = specs["type"]

        if p_type == "int":
            params[param_name] = trial.suggest_int(
                param_name, specs["low"], specs["high"], step=specs.get("step", 1)
            )
        elif p_type == "float":
            params[param_name] = trial.suggest_float(
                param_name, specs["low"], specs["high"], step=specs.get("step")
            )
        elif p_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, specs["choices"])

    return params


if args.strategy not in STRATEGY_SEARCH_SPACES:
    raise ValueError(f"Optimization config not found for '{args.strategy}'")

script_dir = os.path.dirname(__file__)
output_dir = os.path.join(script_dir, FILE_CONFIG["output_folder"])
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
tuning_output_file = os.path.join(
    output_dir, f"tuning_{args.strategy}_{args.objective_type}_{timestamp}.json"
)
log_file = os.path.join(
    output_dir, f"tuning_{args.strategy}_{args.objective_type}_log_{timestamp}.txt"
)

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
optuna.logging.enable_propagation()
optuna.logging.set_verbosity(optuna.logging.INFO)

# --- Data Loading ---
print(f"Loading Strategy: {args.strategy}...")
StrategyClass = load_strategy_class(args.strategy)

csv_path = os.path.join(
    script_dir, FILE_CONFIG["datasets_folder"], FILE_CONFIG["dataset_name"]
)
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found: {csv_path}")

print(f"Processing Data ({args.start_date} to {args.end_date})...")
raw_df = load_raw_data(csv_path)
DATA_PROCESSED = StrategyClass.prepare_data(raw_df, args.start_date, args.end_date)
print("Data ready.")


# --- Optimization Loop ---
def objective(trial):
    """Optuna objective function — run one backtest trial and return its score.

    Seeds both ``random`` and ``numpy`` with the trial number so results
    are reproducible when Optuna re-evaluates the same parameter combination.
    The return value matches the study's direction(s):

    - ``single``: a single float (final equity).
    - ``multiple``: a six-tuple of floats for Pareto optimisation.
    - ``weighted``: a single float combining equity and drawdown.

    Args:
        trial: The current :class:`optuna.Trial` provided by
            ``study.optimize``.

    Returns:
        Score(s) as described above. Returns a penalty (``float('-inf')``
        or ``-100`` values) if the backtest returned ``None``.
    """
    seed = trial.number
    random.seed(seed)
    np.random.seed(seed)

    dynamic_params = get_trial_params(trial, args.strategy)

    # Merge global static params + dynamic params + current objective type
    params = ENGINE_CONFIG.copy()
    params.update(dynamic_params)
    params["objective_type"] = args.objective_type

    result = run(DATA_PROCESSED, params, StrategyClass)

    # Handle missing or failed backtest results
    if result is None:
        if args.objective_type == "single":
            return float("-inf")
        elif args.objective_type == "multiple":
            # Must return 6 values matching the study's direction tuple.
            return float("-inf"), -100.0, 0.0, -9999.0, 0.0, 0.0
        elif args.objective_type == "weighted":
            return float("-inf")

    # Single-objective
    if args.objective_type == "single":
        return result

    # Multi-objective (Pareto optimization)
    elif args.objective_type == "multiple":
        portfolio_value = result.get("portfolio_value", 0)
        max_drawdown = result.get("max_drawdown", -100)
        win_rate = result.get("win_rate", 0)
        avg_loss = result.get("avg_loss", -9999)
        sortino_ratio = result.get("sortino_ratio", 0)
        sharpe_ratio = result.get("sharpe_ratio", 0)

        # Clamp degenerate ratios (e.g. zero losses → Inf) to 0 to avoid
        # poisoning the Pareto front with mathematically undefined values.
        if np.isinf(sortino_ratio) or np.isnan(sortino_ratio):
            sortino_ratio = 0.0
        if np.isinf(sharpe_ratio) or np.isnan(sharpe_ratio):
            sharpe_ratio = 0.0

        return (
            portfolio_value,
            max_drawdown,
            win_rate,
            avg_loss,
            sortino_ratio,
            sharpe_ratio,
        )

    # Weighted combination
    elif args.objective_type == "weighted":
        portfolio_value = result.get("portfolio_value", 0)
        max_drawdown = result.get("max_drawdown", 100)
        win_rate = result.get("win_rate", 0)
        score = (0.7 * portfolio_value) + (0.3 * -max_drawdown)
        return score


if __name__ == "__main__":

    logging.info("=" * 80)
    logging.info(f"STARTING OPTIMIZATION STUDY")
    logging.info(f"Strategy: {args.strategy}")
    logging.info(f"Objective Type: {args.objective_type}")
    logging.info(f"Trials: {args.trials}")
    logging.info(f"Date Range: {args.start_date} to {args.end_date}")

    search_space = STRATEGY_SEARCH_SPACES.get(args.strategy)
    logging.info(f"Search Space: {json.dumps(search_space, indent=2)}")
    logging.info(f"Engine Config: {json.dumps(ENGINE_CONFIG, indent=2)}")
    logging.info("=" * 80)

    print(f"Starting optimization ({args.trials} trials)...")

    if args.objective_type == "multiple":
        # Metric Optimization Direction: portfolio_value, max_drawdown, win_rate, avg_loss, sortino_ratio, sharpe_ratio
        study = optuna.create_study(
            directions=[
                "maximize",
                "maximize",
                "maximize",
                "maximize",
                "maximize",
                "maximize",
            ]
        )
        study.optimize(objective, n_trials=args.trials, n_jobs=-1)

        best_trials = sorted(
            [
                {"trial_number": t.number, "values": t.values, "params": t.params}
                for t in study.best_trials
            ],
            # Three-priority sort key for Pareto-front ranking:
            #   Priority 1 — Efficiency ratio (net profit / worst drawdown):
            #     higher profit with lower drawdown scores better. The 0.0001
            #     offset prevents division by zero when drawdown is exactly 0.
            #   Priority 2 — Net capital: breaks ties in efficiency by
            #     preferring the trial that earned the most absolute profit.
            #   Priority 3 — Sortino ratio: secondary risk-adjusted tiebreaker.
            key=lambda x: (
                (x["values"][0] - ENGINE_CONFIG["initial_cash_balance"])
                / abs(x["values"][1] - 0.0001),
                x["values"][0],
                x["values"][4],
            ),
            reverse=True,
        )
        with open(tuning_output_file, "w") as f:
            json.dump(best_trials, f, indent=4)
        print(f"Optimization complete. Solutions: {len(best_trials)}")

    else:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.trials, n_jobs=-1)

        with open(tuning_output_file, "w") as f:
            json.dump(
                {"best_params": study.best_params, "best_value": study.best_value},
                f,
                indent=4,
            )
        print(f"Optimization complete. Best Value: {study.best_value}")
