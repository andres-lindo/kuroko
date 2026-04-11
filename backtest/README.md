# Set Up Python Virtual Environment (Windows)
```sh
C:\Python311\python.exe -m venv venv  # Create a virtual environment named 'venv'
.\venv\Scripts\activate               # Activate the virtual environment
python -m pip install --upgrade pip   # Upgrade pip to the latest version
pip install -r requirements.txt       # Install required dependencies
```

---

# Install Python 3.11 on Amazon Linux 2023
```sh
sudo dnf install -y python3.11  # Install Python 3.11
sudo alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1  # Set Python 3.11 as an alternative
sudo alternatives --config python3  # Select Python 3.11 as the default version
```

# Install pip for Python 3.11 on Amazon Linux 2023
```sh
curl -O https://bootstrap.pypa.io/get-pip.py  # Download get-pip.py script
python3.11 get-pip.py --user  # Install pip for Python 3.11
```

# Install dependencies on Amazon Linux 2023
```sh
python -m pip install --upgrade pip   # Upgrade pip to the latest version
pip install -r requirements-linux.txt
```

# Invoke Backtesting
```sh
python backtest.py --strategy EMACrossoverStrategy
```

# Run Hyperparameter Tuning with `screen`
## Single-Objective Optimization
```sh
screen -S single_objective  # Create a new screen session named 'single_objective'
python3 tuning.py --strategy EMACrossoverStrategy --start_date 2025-06-01 --end_date 2026-02-06 --objective_type single --trials 1000  # Run tuning
screen -r single_objective  # Reattach to the session
```

## Multi-Objective Optimization
```sh
screen -S multiple_objective  # Create a new screen session named 'multiple_objective'
python3 tuning.py --strategy EMACrossoverStrategy --start_date 2025-06-01 --end_date 2026-02-06 --objective_type multiple --trials 5000  # Run tuning
screen -r multiple_objective  # Reattach to the session
```

## Weighted-Objective Optimization
```sh
screen -S weighted_objective  # Create a new screen session named 'weighted_objective'
python3 tuning.py --strategy EMACrossoverStrategy --start_date 2025-06-01 --end_date 2026-02-06 --objective_type weighted --trials 5000  # Run tuning
screen -r weighted_objective  # Reattach to the session
```

---

## Managing `screen` Sessions
```sh
screen -ls  # List all active screen sessions
```

## Detach from a session
Press `Ctrl + A`, then `D`

## Kill all `screen` Sessions
```sh
screen -ls | awk '/[0-9]+\./ {print $1}' | xargs -I {} screen -S {} -X quit
```

# Use a calculation: 1 contract for every $3,000 of equity
dynamic_size = int(self.equity // 3000) 
position_size_contracts = max(1, dynamic_size)


2026-02-13 23:21:06,010 - INFO - Trial 319 finished with values: [6038.8, -14.4, 72.4, -118.0, 6.575368156425812, 1.7760819640379988] and parameters: {'fast_ema': 6, 'take_profit_long': 0.8700000000000001, 'take_profit_short': 0.98, 'stop_loss_long': 1.97, 'stop_loss_short': 1.52, 'max_long_positions': 3, 'max_short_positions': 3, 'rsi_overbought': 70.0, 'rsi_oversold': 66.0, 'atr_percentile': 18.0}.
2026-02-13 23:47:36,735 - INFO - Trial 414 finished with values: [6335.8, -16.4, 78.3, -122.0, 6.1765959246350794, 1.6672116121158578] and parameters: {'fast_ema': 6, 'take_profit_long': 0.8700000000000001, 'take_profit_short': 0.47, 'stop_loss_long': 1.97, 'stop_loss_short': 1.52, 'max_long_positions': 3, 'max_short_positions': 3, 'rsi_overbought': 67.0, 'rsi_oversold': 66.0, 'atr_percentile': 18.0}.
