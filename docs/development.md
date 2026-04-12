# Development Guide

## Branching Model

```
main ──────────────────────────────────────► main (production)
  │                                            ▲
  │                                            │
  └──► your-work-branch ──► dev ───────────────┘
```

| Branch | Purpose |
|---|---|
| `main` | Production-ready code. Never commit directly. |
| `dev` | Integration branch. All work branches merge here first. |
| `your-work-branch` | One branch per feature, fix, or experiment. Branched from `main`. |

### Workflow

1. **Branch from `main`**
   ```bash
   git checkout main
   git pull
   git checkout -b your-work-branch
   ```

2. **Do your work**, committing with [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat: add RSI divergence filter to mean-reversion entry
   fix: correct martingale size calculation on position reload
   docs: update optimization CLI reference
   refactor: extract basket TP logic into separate method
   ```

3. **Keep your branch up to date** with `dev` before opening a PR to minimize divergence:
   ```bash
   git fetch origin
   git rebase origin/dev
   ```

4. **Open a PR into `dev`** — this is where code review happens.

5. **After review and merge into `dev`**, a separate PR is opened from `dev` into `main` to promote to production.

> Direct commits to `main` or `dev` are not allowed. All changes go through a work branch.

---

## Virtual Environments

This project uses **two isolated virtual environments** that must never be merged:

| Environment | Location | Purpose |
|---|---|---|
| Live trading | `venv/` (project root) | Running the bot and development |
| Backtesting | `backtest/venv/` | Historical simulation and optimization |

The dependency versions are intentionally different — `backtesting.py` requires `pandas 1.5.3` and `numpy 1.26.4`, which conflict with the live trading stack.

### Root venv (live trading + development)

See [Prerequisites and Setup](../README.md#setup) in the root README for commands.

### Backtest venv

See [`backtest/README.md`](../backtest/README.md) for commands.

---

## Pre-commit Hooks

This project uses [pre-commit](https://pre-commit.com/) to enforce code formatting with **black** before every commit. Configuration is in `.pre-commit-config.yaml`.

**One-time setup** — run this once after cloning, with the root venv active:
```bash
pre-commit install
```

From that point on, black runs automatically on every `git commit`. If it reformats any file, the commit is aborted — stage the changes and commit again:

```bash
git add -u
git commit -m "your message"
```

**Run manually against all files:**
```bash
pre-commit run --all-files
```

> Each contributor must run `pre-commit install` once. The hook is not active until that command is run. The hook is installed at the root level and covers **all Python files in the repository**, including those under `backtest/` — no separate install is needed inside the backtest venv.

---

## Project Conventions

### General

- **Formatter**: black (enforced via pre-commit, line length default: 88)
- **Language**: all code, variable names, inline comments, docstrings, and log messages must be in English
- **Secrets**: never commit `credentials.env` or any file containing API keys or connection strings
- **Config changes**: strategy parameters for live trading are stored in `strategies/RSIBollingerStrategy.json` and committed to the repository. Edit that file directly and redeploy the bot to apply changes. See [RSIBollingerStrategy documentation](../docs/strategies/RSIBollingerStrategy.md) for the full parameter reference.

> **Note**: `strategies/RSIBollingerStrategy.json` (live) and `backtest/strategies/RSIBollingerStrategy.json` (backtest) are independent files. Tuning results from Optuna must be manually applied to the live config. See [backtest/README.md](../backtest/README.md) for details.

- **Dependencies**: add new external packages to `requirements.txt` (root) or `backtest/requirements.txt` depending on which execution context requires them

### Logging

Use the standard module-level logger in every file:

```python
import logging

logger = logging.getLogger(__name__)
```

Never use `print()` for runtime output. Use `logger.debug/info/warning/error/exception` as appropriate.

Always use f-strings for log message interpolation — never `%s`/`%d` formatting or `.format()`:

```python
# correct
logger.info(f"Cache loaded from disk for {epic} {res}")

# wrong
logger.info("Cache loaded from disk for %s %s", epic, res)
```

### Error Handling

Handle exceptions individually in each function. Do not let exceptions propagate silently or catch broad `Exception` at the top level without logging:

```python
def fetch_data(self):
    try:
        return self._client.get(...)
    except SomeSpecificError as e:
        logger.error(f"Failed to fetch data: {e}")
        return None
```

### Docstrings

All Python files, classes, and functions must use **Google-style docstrings**.

**Module-level** (every `.py` file):
```python
"""Brief one-line description of what this module does.

Longer explanation if needed. Describes the module's responsibility
within the system.
"""
```

**Functions and methods**:
```python
def open_position(self, epic: str, size: float, side: str) -> dict:
    """Opens a new market position via the IG API.

    Args:
        epic: Instrument identifier (e.g. 'IX.D.NASDAQ.IFMM.IP').
        size: Position size in contracts.
        side: Trade direction, either 'BUY' or 'SELL'.

    Returns:
        API response dict containing dealReference and status.

    Raises:
        IGException: If the API rejects the order.
        RuntimeError: If the session is not authenticated.
    """
```

**Classes**:
```python
class IGClient:
    """Wrapper around the IG Markets REST API.

    Handles authentication, session refresh, retry logic,
    and candle caching (in-memory and parquet).

    Attributes:
        accountId: The IG account identifier loaded from credentials.
        candles_cache: In-memory dict keyed by '{epic}_{resolution}'.
    """
```

Omit sections (`Args`, `Returns`, `Raises`) that do not apply. One-liners are acceptable for trivial properties or pass-through methods.

---

## Adding a Strategy

### To the backtest module

1. Create `backtest/strategies/your_strategy.py` following the `backtesting.py` `Strategy` base class:
   ```python
   """One-line description of this strategy.

   Describe the signal logic, timeframe, and intended use case.
   """
   from backtesting import Strategy

   class YourStrategy(Strategy):
       """Brief description of the strategy.

       Attributes:
           param_a: Description of what this parameter controls.
           param_b: Description of what this parameter controls.
       """

       # Class attributes are the tunable parameters — Optuna reads these directly.
       param_a = 10
       param_b = 0.5

       @classmethod
       def prepare_data(cls, df, start_date, end_date):
           """Filter and preprocess raw OHLC data for this strategy.

           Args:
               df: Raw DataFrame loaded from CSV.
               start_date: Inclusive start of the date range.
               end_date: Inclusive end of the date range.

           Returns:
               Preprocessed DataFrame with the required columns and index.
           """
           ...

       def init(self):
           """Initialise indicators. Called once before the first bar."""
           ...

       def next(self):
           """Execute strategy logic for the current bar."""
           ...
   ```

2. Register it in `backtest/backtest.py` under `STRATEGY_REGISTRY`:
   ```python
   STRATEGY_REGISTRY = {
       "YourStrategy": "strategies.your_strategy",
   }
   ```

3. Add its parameter search space to `backtest/tuning.py` under `STRATEGY_SEARCH_SPACES`:
   ```python
   STRATEGY_SEARCH_SPACES = {
       "YourStrategy": {
           "param_a": {"type": "int",   "low": 5,   "high": 20},
           "param_b": {"type": "float", "low": 0.1, "high": 1.0, "step": 0.05},
       }
   }
   ```
   If your strategy uses a different dataset than the default (`nq_intraday-15min.csv`), also update `FILE_CONFIG['dataset_name']` in `tuning.py`.

### To the live trading bot

Live strategies are pluggable via the `--strategy` CLI flag. Each live strategy consists of two artifacts:

- A Python module inside the `strategies/` package named after the class (e.g. `strategies/RSIBollingerStrategy.py`) exporting a strategy class and a `load_params` callable.
- A JSON config file at `strategies/<StrategyClassName>.json`.

The `strategies/` directory is a Python package (contains `__init__.py`). The module file must be named exactly after the strategy class (e.g. `RSIBollingerStrategy.py`), and is imported as `strategies.RSIBollingerStrategy`.

**Always validate logic changes in the backtest module before modifying any live strategy.** Port the change to a backtest strategy class, run it against historical data, and confirm the behaviour is correct before applying it to the live module.

After updating the live strategy module, update the corresponding values in `strategies/<StrategyClassName>.json` and commit before deploying.

See [RSIBollingerStrategy documentation](../docs/strategies/RSIBollingerStrategy.md) for the reference implementation.
