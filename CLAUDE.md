# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Global CLAUDE.md covers:** Personality, tone, language, engram protocol (proactive saves, search, session close, compaction recovery), SDD workflow and commands, model assignments (opus/sonnet/haiku per phase), skill resolver, sub-agent context protocol, delegation rules (base table).

> **Project Rules, Orchestrator Override and others in this file take precedence over their global equivalents.**
---

## Project Overview

**Kuroko** is an algorithmic trading bot that trades financial futures (NASDAQ/S&P 500) via IG Markets. It has two concerns: **live trading** (executes against the IG Markets API) and **backtesting/optimization** (offline strategy validation and parameter tuning).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Note on TA-Lib**: On Windows, the wheel must be installed from a prebuilt GitHub release before `pip install -r requirements.txt`. On macOS/Linux, `brew install ta-lib` / `apt install libta-lib-dev` first.

Environment variables are loaded from `credentials.env` (not committed):
```
ig_username, ig_password, ig_api_key, ig_acc_number, ig_acc_type
table_storage_connection  # Azure Blob Storage (log shipping only)
```

## Running

**Live trading bot:**
```bash
python kuroko.py <partition_key> --strategy <StrategyName>
# e.g. python kuroko.py DEV_NQ100 --strategy RSIBollingerStrategy
```

**Backtest a strategy:**
```bash
python backtest/backtest.py --strategy RSIBollingerStrategy
```

**Optuna parameter optimization:**
```bash
cd backtest
python tuning.py --strategy RSIBollingerStrategy \
  --start_date 2026-01-01 --end_date 2026-04-10 \
  --objective_type single --trials 100
# objective_type: single | multiple | weighted
```

There is no automated test suite, linting config, or CI pipeline.

## Architecture

### Live Trading (`kuroko.py` → `ig_client.py` + `strategies/rsi_bollinger.py`)

```
kuroko.py
  ├── load_strategy(args.strategy) → (RSIBollingerStrategy, load_params)
  ├── load_params("strategies/RSIBollingerStrategy.json") → types.SimpleNamespace
  ├── Setup logging         → AzureBlobHandler (daily append-blob rotation)
  ├── IGClient              → IG Markets REST API session (handles auth + retry)
  └── RSIBollingerStrategy.run() → main loop
```

- **`IGClient`**: Thin wrapper around IG Markets API. Handles session auth, token refresh, exponential backoff on failures, and candle caching (in-memory + parquet).
- **`RSIBollingerStrategy`**: All trading logic lives here. Manages positions, computes signals via TA-Lib, and enforces risk rules. See [RSIBollingerStrategy documentation](docs/strategies/RSIBollingerStrategy.md).
- **`AzureBlobHandler`**: Custom `logging.Handler` that ships logs to Azure Blob Storage.

### Strategy Logic (RSI + Bollinger Bands, mean-reversion)

- **Entry**: Buy when price < BB_lower AND RSI < oversold threshold
- **Sizing**: Martingale grid — each new position uses a 1.5× multiplier (up to 5 open positions)
- **Exit**: Basket take-profit (close all when avg entry + take_profit_ticks is reached)
- **Protection**: ATR-based dynamic stop-loss; max drawdown freeze (75% threshold); margin check before entry
- Incomplete (current) candle is always stripped before signal calculation

### Backtesting (`backtest/`)

Uses the `backtesting.py` framework. Strategy classes follow this pattern:
- **Class attributes** = tunable parameters (read by Optuna)
- `prepare_data(df, start_date, end_date)` = classmethod for preprocessing
- `init()` = indicator setup
- `next()` = per-bar logic

Optuna supports three objective modes:
- `single`: maximize equity
- `multiple`: Pareto front across 6 objectives
- `weighted`: equity − 0.3 × drawdown

Historical datasets live in `backtest/datasets/` (15-min OHLC CSVs for ES and NQ futures).

### Configuration

Strategy parameters for live trading are stored in **`strategies/RSIBollingerStrategy.json`** and committed to the repository. All 22 keys (trading params, risk controls, system config) are loaded at startup into a `types.SimpleNamespace` and passed to `RSIBollingerStrategy`. Changing a parameter requires editing the file and redeploying. Azure Table Storage is no longer used for parameter storage — `table_storage_connection` is retained in `credentials.env` exclusively for `AzureBlobHandler` log shipping.

### Key Design Decisions

- `is_live_account` is `True` only when `ig_acc_type=LIVE` (exact, case-sensitive). Any other value — including `DEMO` or missing — results in `False` (virtual mode). Not stored in the strategy JSON.
- Candle caching uses parquet files to minimize IG API calls across restarts
- `initial_cash_balance = 4000` (in `strategies/RSIBollingerStrategy.json`) maps to the 20,000 IG demo account via 1:20 leverage

---

## Project Rules

### Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code. For non-trivial changes, ask: "Is there a more elegant solution?". If a fix feels hacky, prefer the elegant alternative. Skip for simple, obvious fixes.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Do not refactor the entire project at once — sub-agents modify only what is necessary to solve the task.
- **Architecture**: Before architectural decisions, delegate codebase and docs exploration to a sub-agent (Explore type or `sdd-explore`). Never read `docs/` inline.

### Session Start
At the beginning of every session:
1. Load the skill registry: `mem_search(query: "skill-registry")` → `mem_get_observation(id)` for full compact rules. Fallback: read `.atl/skill-registry.md` if engram returns nothing. Cache the result for the session.
2. Call `mem_context()` to load recent session summaries and observations.
3. Call `mem_search(query: "open SDD changes")` to check for incomplete SDD cycles.
4. `mem_search(query: "corrections feedback pattern")` — review results to avoid repeating past mistakes.

### Task Management
- Use SDD engram artifacts (`sdd-tasks`, `sdd-apply-progress`) as the source of truth for progress tracking.
- Check in with the user before implementation begins; provide a high-level summary at each phase completion.

### Follow Development Standards
- Detailed instructions live in `docs/development.md` -- pass this reference to sub-agents as needed.
- If a sub-agent discovers a better practice, instruct it to update `docs/development.md` as part of its task.
- Documentation: quality over quantity — remove non-useful content; keep it easy to find and understand.

### Autonomous Bug Fixing
- When given a bug report, delegate immediately to a sub-agent with full context (logs, errors, failing test names, stack traces).
- Ask clarifying questions only if the bug description is ambiguous. Otherwise delegate immediately with all available context.
- Zero context switching required from the user -- the sub-agent resolves the issue end-to-end.

### Python Environment
- **Always activate the project venv** before any `python`, `pip`, or `pytest` call: `source venv/bin/activate` (project root).
- Test suites have their own venvs -- use the commands in the Testing section; do NOT mix venvs.
- **Pre-commit hooks auto-reformat Python with `black`** at commit time -- expect reformatting, do not fight it.
- **HARD RULE — Sub-agents MUST read `docs/development.md` before writing any Python code.** Non-compliance is a blocking issue: the orchestrator must reject output that does not follow those standards and re-delegate with the requirement made explicit.

### Testing
- Test execution is delegated to sub-agents. TDD (RED -> GREEN -> REFACTOR) is the default for Python changes.
- If no test exists for a changed area, the sub-agent must create the necessary unit, integration, or system test under `tests/`.

---

## Orchestrator Override — Delegation and Hard Stop

**This section overrides the global `CLAUDE.md` delegation table. These rules are STRICTER — no size-based exceptions.**

### HARD RULE — SDD Before Substantial Features

**NEVER** delegate a substantial feature to a general-purpose sub-agent without first running SDD. Jumping straight to delegation is a violation, not an option.

**Mandatory self-check before every delegation:**
> "Do I need to explore or design before acting?" — if yes, propose `sdd-new <name>` to the user before delegating anything.

**SDD is required when ANY of the following are true:**
- The right approach is not immediately obvious — the work needs analysis or design before implementation
- Multiple files or components are affected and the impact needs to be understood first
- A decision must be made before acting (architecture, approach, integrations, critical infrastructure managed by Terraform)

**SDD is NOT required when:**
- The change is obvious, self-contained, and needs no prior analysis
- It's a simple bug fix, small update scoped to a known area
- It's documentation-only

If in doubt, propose SDD. The cost of a quick proposal is far lower than the cost of unplanned rework.

- If something goes sideways, STOP, re-plan, and re-delegate -- do not keep pushing.
- Ask all required questions until requirements are clear before launching sub-agents.

Example:
> User: "Add support for account type FOO."
> Orchestrator: "This touches IAM, networking, and SSO -- SDD required. Want me to start with `sdd-new account-type-foo`?"

---

## Verification Before Done

- Never mark a task complete without confirmed evidence it works.
- Verification (tests, logs, correctness checks) is delegated to a sub-agent or the `sdd-verify` phase -- never done inline by the orchestrator.
- Ask yourself: "Would a senior staff engineer approve this?" before closing a task.
- **HARD RULE — `sdd-verify` MUST be followed by `judgment-day` before proposing a commit.** After `sdd-verify` passes, launch the `judgment-day` skill against the changed files. Only if both pass does the change proceed to commit proposal. This is not optional.
- **HARD RULE — `judgment-day` MUST include simplicity as a custom criterion:** "Is this the simplest solution that meets the spec requirements? Flag any abstraction, generalization, or complexity not directly justified by a specific requirement."
- **HARD RULE — Documentation MUST be updated as part of closing every task, not deferred.** Once `sdd-verify` passes (or a task is confirmed working), update project `README.md` if needed, `CHANGELOG.md`, and `docs/` files relevant to what changed. A task is NOT done until documentation reflects the change.

---

## Commit

### Git Commits - HARD RULE

**NEVER** run `git commit`, `git push`, or any variant (amend, force-push, auto-commit) without **explicit written confirmation from the user.**

- Verified tasks do NOT grant commit permission
- "Looks good" does NOT grant commit permission
- The orchestrator and ALL sub-agents are bound by this rule

Required flow:
1. Sub-agent proposes the commit message (must match Conventional Commits format — see below) and lists the files that would be staged.
2. Orchestrator surfaces the proposal to the user and waits.
3. The user explicitly says to commit (e.g., "yes, commit it", "go ahead").
4. Only then does a sub-agent execute `git commit`.

"The verify passed" is NOT permission to commit. "Looks good" is NOT permission to commit. Wait for unambiguous confirmation.

Both commit messages and MR titles must follow Conventional Commits format, validated by:

```
^(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test|memory)(\([a-z0-9\._-]+\))?!?: .+
```

| Type | Example |
|------|---------|
| feat | `feat(terraform): add new region` |
| fix | `fix: correct assume role logic` |
| docs | `docs: update workspace guide` |
| chore | `chore: update shellspec version` |
| refactor | `refactor(tui): simplify screen navigation` |
| test | `test(ci): add integration mode tests` |
| ci | `ci: split test jobs by suite` |
| style | `style: normalize output formatting` |
| perf | `perf(cache): improve lookup speed` |
| build | `build: add makefile targets` |
| memory | `memory: update engram memories` |

If a commit message or MR title does not match the regex, reject it and ask for a valid one.

If engram memory was updated during the session, ask the user:
> "Do you want to run `engram sync` and commit your session memories?"
If yes, run `engram sync`, stage all changed files under .engram folder, and commit them.
