"""Backtest runner and helper utilities for the Kuroko strategy framework.

Provides functions to load data, resolve strategy classes by name, retrieve
default parameters, and execute a ``backtesting.py`` backtest. Intended to be
invoked directly (``python backtest.py --strategy <Name>``) or imported by the
Optuna tuning script.
"""

from backtesting import Backtest
import pandas as pd
import os
import json
import argparse
import importlib
import warnings
import logging
import sys


def setup_logging(filename="backtest-last-execution.log"):
    """Configure root logger to write to a file and stdout simultaneously.

    Clears any handlers registered by a previous call so the function is
    safe to call multiple times (e.g. when the CLI re-runs with a different
    strategy). The log file is opened in write mode, so each run starts with
    a fresh file rather than appending to an old one.

    Args:
        filename: Path to the log file. Defaults to
            ``'backtest-last-execution.log'`` in the current directory.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(filename, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


# Suppress the fractional-trading UserWarning that backtesting.py emits when
# leverage-based margin accounts hold non-integer contract sizes.
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*fractional trading.*"
)

# --- Strategy Registry ---
STRATEGY_REGISTRY = {
    "RSIBollingerStrategy": "strategies.RSIBollingerStrategy",
}


def load_raw_data(csv_file):
    """Load a CSV dataset into a DataFrame without any preprocessing.

    Date parsing is intentionally deferred to the strategy's
    ``prepare_data`` classmethod so that this function remains cheap even
    when the caller only needs to inspect column names or row counts.

    Args:
        csv_file: Absolute or relative path to the OHLC CSV file.

    Returns:
        Raw :class:`pandas.DataFrame` with all columns as read from disk.

    Raises:
        FileNotFoundError: If ``csv_file`` does not exist.
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File not found: {csv_file}")

    # Skip date parsing here — the strategy's prepare_data is responsible
    # for interpreting and filtering the index.
    return pd.read_csv(csv_file, low_memory=False)


def load_strategy_class(strategy_name):
    """Resolve a strategy name string to the corresponding class object.

    Uses :data:`STRATEGY_REGISTRY` to look up the module path, then imports
    it dynamically. This keeps the backtest runner decoupled from concrete
    strategy imports — adding a new strategy only requires a registry entry.

    Args:
        strategy_name: Registered name of the strategy (e.g.
            ``'RSIBollingerStrategy'``).

    Returns:
        The strategy class (a subclass of
        :class:`backtesting.Strategy`).

    Raises:
        ValueError: If ``strategy_name`` is not present in
            :data:`STRATEGY_REGISTRY`.
    """
    if strategy_name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Strategy '{strategy_name}' not registered. Available: {available}"
        )

    # Dynamic import — the registry maps name → dotted module path.
    module_path = STRATEGY_REGISTRY[strategy_name]
    module = importlib.import_module(module_path)
    return getattr(module, strategy_name)


def load_backtest_params(strategy_name):
    """Load backtest parameters for a strategy from its JSON config file.

    Resolves the path relative to this file's directory:
    ``backtest/strategies/{strategy_name}.json``.

    Args:
        strategy_name: Name of the strategy class (e.g.
            ``'RSIBollingerStrategy'``). Used to build the file name.

    Returns:
        A dict with all parameters loaded from the JSON file.

    Raises:
        FileNotFoundError: If no JSON file exists for ``strategy_name``
            at the expected path.
    """
    config_path = os.path.join(
        os.path.dirname(__file__), "strategies", f"{strategy_name}.json"
    )
    with open(config_path, "r") as f:
        return json.load(f)


def get_strategy_params(strategy_name):
    """Return the default run parameters for a strategy from its JSON config.

    Delegates to :func:`load_backtest_params` to read
    ``backtest/strategies/{strategy_name}.json``. Parameters cover two
    groups: engine settings (cash, leverage, commission, mode flags) and
    strategy settings (indicator settings, risk limits), returned together
    in a single flat dict so callers can pass them directly to :func:`run`.

    Args:
        strategy_name: Name of the strategy class (e.g.
            ``'RSIBollingerStrategy'``).

    Returns:
        A shallow copy of the parameter dict so callers can safely
        ``pop`` keys without mutating the loaded data.

    Raises:
        FileNotFoundError: If no JSON config exists for ``strategy_name``
            at ``backtest/strategies/{strategy_name}.json``.
    """
    return load_backtest_params(strategy_name).copy()


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
    """Execute a backtest and return results according to the active objective mode.

    The return value varies by the combination of ``silent_mode`` and
    ``objective_type`` so that the Optuna tuning loop can consume results
    directly without post-processing:

    - ``silent_mode=False`` (normal): returns the full ``backtesting.py``
      stats object and prints a human-readable summary.
    - ``silent_mode=True, objective_type='single'``: returns a single
      float (final equity) for fast single-objective Optuna trials.
    - ``silent_mode=True, objective_type in ('multiple', 'weighted')``:
      returns a dict of metric scalars for multi-objective Pareto trials.

    Args:
        data: Preprocessed :class:`pandas.DataFrame` ready for
            ``backtesting.py`` (must have OHLCV columns and a
            datetime index).
        params: Flat parameter dict containing both engine keys
            (``initial_cash_balance``, ``leverage``, ``commission``,
            ``silent_mode``, ``objective_type``) and strategy-specific
            keys. The function mutates this dict by popping engine keys.
        strategy_class: The strategy class to run. Must be a subclass of
            :class:`backtesting.Strategy`.

    Returns:
        Depends on ``silent_mode`` / ``objective_type`` — see above.
        Returns a penalty value (``0.0`` or a dict of ``-100`` values)
        when the strategy triggered its max-drawdown freeze.
    """

    # 1. Extract engine configuration keys from params before passing the
    #    remainder to bt.run(), which only accepts strategy-level parameters.
    initial_cash = params.get("initial_cash_balance")
    leverage = params.get("leverage")
    commission = params.get("commission")

    silent_mode = params.pop("silent_mode", True)
    objective_type = params.pop("objective_type")

    # Inject silent_mode into the class attribute so the strategy's internal
    # log() method can read it without receiving it as a constructor argument.
    strategy_class.silent_mode = silent_mode

    # 2. Instantiate the backtesting engine
    bt = Backtest(
        data,
        strategy_class,
        cash=initial_cash,
        commission=commission,
        margin=1 / leverage,
        exclusive_orders=False,
        trade_on_close=True,
        hedging=True,
    )

    # 3. Run the backtest, forwarding only strategy-level params (engine keys
    #    were already popped above).
    stats = bt.run(**params)

    trades_df = stats._trades
    strategy_instance = stats["_strategy"]

    if getattr(strategy_instance, "max_drawdown_reached", False):
        if silent_mode:
            if objective_type == "single":
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

    # Four-way branch on (silent_mode, objective_type):
    #   silent + single   → return a single float for Optuna to maximise
    #   silent + multiple → return a metric dict for Pareto optimisation
    #   silent + weighted → return a metric dict for weighted scoring
    #   normal (any)      → print the full report and return the stats object

    # Case 1: Silent + single-objective (fast Optuna trial)
    if silent_mode and objective_type == "single":
        portfolio_value = stats["Equity Final [$]"]
        print(f"Equity: {portfolio_value:.2f}")
        return round(portfolio_value, 1)

    # Shared metric calculations used by both silent-multiple and normal modes.
    closed_trades = trades_df[trades_df["ExitTime"].notna()]
    n_winning = len(closed_trades[closed_trades["PnL"] > 0])
    n_losing = len(closed_trades[closed_trades["PnL"] < 0])
    win_rate = (n_winning / len(closed_trades) * 100) if len(closed_trades) > 0 else 0
    avg_loss = (
        closed_trades[closed_trades["PnL"] < 0]["PnL"].mean() if n_losing > 0 else 0
    )

    final_equity = stats["Equity Final [$]"]
    max_dd = stats["Max. Drawdown [%]"]
    sharpe_ratio = stats["Sharpe Ratio"]
    sortino_ratio = stats["Sortino Ratio"]

    # Case 2 & 3: Silent + multiple/weighted — return a metric dict for Optuna.
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

    # Case 4: Normal mode — print full human-readable report.
    print(f"\n{stats}\n")
    print("=" * 60)
    print("Operations Summary:")
    print("=" * 60)
    print(f"Total Trades: {len(trades_df)}")
    print(f"Closed Trades: {len(closed_trades)}")
    print(f"Winning: {n_winning} ({win_rate:.2f}%)")
    print(f"Losing: {n_losing}")
    print(
        f"Avg Win: {closed_trades[closed_trades['PnL'] > 0]['PnL'].mean():.2f}"
        if n_winning > 0
        else "Avg Win: 0.00"
    )
    print(f"Avg Loss: {avg_loss:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio}")
    print(f"Sortino Ratio: {sortino_ratio}")
    print(f"Initial Capital: {params['initial_cash_balance']:.2f}")
    print(f"Final Capital: {final_equity:.2f}")
    print(f"Net Profit: {final_equity - params['initial_cash_balance']:.2f}")
    print("=" * 60)

    # Generate the interactive HTML chart only in normal (non-silent) mode.
    if not silent_mode:
        generate_plot(bt)

    return stats


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", nargs="?", default="RSIBollingerStrategy")
    args = parser.parse_args()

    setup_logging(f"{args.strategy}-backtest-last-execution.log")

    # 1. Resolve the strategy class from the registry.
    StrategyClass = load_strategy_class(args.strategy)

    # 2. Load default parameters from the control panel.
    params = get_strategy_params(args.strategy)

    # 3. Extract dataset path and date range before passing params to run().
    csv_filename = params.pop("dataset")
    start_date = params.pop("start_date")
    end_date = params.pop("end_date")

    # Build the absolute path to the dataset file.
    script_dir = os.path.dirname(__file__)
    csv_file = os.path.join(script_dir, "datasets", csv_filename)

    print(f"Running backtest with strategy: {args.strategy}")

    # 4. Load raw CSV — date parsing is deferred to prepare_data.
    raw_df = load_raw_data(csv_file)

    # 5. Preprocess data via the strategy's classmethod. If the strategy class
    #    does not implement prepare_data, this will raise AttributeError, which
    #    is intentional — every strategy must own its data contract.
    data = StrategyClass.prepare_data(raw_df, start_date, end_date)

    # 6. Execute the backtest.
    run(data, params, StrategyClass)
