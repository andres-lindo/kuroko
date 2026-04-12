"""Strategy classes for the backtesting module.

Each submodule defines one backtesting.py Strategy subclass. Strategies
are registered by name in backtest/backtest.py under STRATEGY_REGISTRY
and are selected at runtime via the --strategy CLI argument.
"""