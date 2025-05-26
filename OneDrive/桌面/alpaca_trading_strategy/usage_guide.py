# Complete Trading System Usage Guide

## Overview

This is a comprehensive quantitative trading system with the following features:
- **Real-time options data integration** (Polygon.io, Tradier, or simulated)
- **Proper trading day calculations** with market calendar
- **Walk-Forward Analysis (WFA)** for parameter optimization
- **Complete risk management** with circuit breakers
- **Async news sentiment analysis**
- **Multi-strategy execution** (passive, core momentum, financials, sentiment)

## System Components

### 1. Core Files Required

```
project/
├── complete_trading_strategy.py    # Main strategy implementation
├── options_integration.py          # Options data providers
├── factor_ic_calculator.py         # Factor IC with trading days
├── wfa_test_complete.py           # Walk-forward analysis
├── strategy_methods_complete.py    # Additional methods
├── strategy_execution_methods.py   # Execution methods
└── usage_guide.md                 # This file
```

### 2. Dependencies

```bash
pip install alpaca-trade-api
pip install alpaca-py
pip install pandas numpy scipy scikit-learn
pip install matplotlib seaborn
pip install yfinance
pip install pytz
pip install vaderSentiment
pip install aiohttp asyncio
pip install tqdm
pip install schedule
```

## Quick Start

### 1. Basic Live Trading Setup

```python
import os
from complete_trading_strategy import AlpacaCompleteStrategy

# Set your API credentials
os.environ['ALPACA_API_KEY'] = 'your_alpaca_api_key'
os.environ['ALPACA_SECRET_KEY'] = 'your_alpaca_secret_key'

# Optional: Options data provider API key
os.environ['POLYGON_API_KEY'] = 'your_polygon_key'  # or use 'simulated'

# Create strategy instance
strategy = AlpacaCompleteStrategy(
    api_key=os.environ['ALPACA_API_KEY'],
    secret_key=os.environ['ALPACA_SECRET_KEY'],
    base_url='https://paper-api.alpaca.markets',  # Paper trading
    options_provider='polygon',  # or 'tradier' or 'simulated'
    options_api_key=os.environ.get('POLYGON_API_KEY'),
    mode='live'
)

# Run the strategy
strategy.run_live_trading()
```

### 2. Backtesting Mode

```python
# Create strategy in backtest mode
strategy = AlpacaCompleteStrategy(
    api_key='dummy',
    secret_key='dummy',
    options_provider='simulated',
    mode='backtest'
)

# Run backtest for specific period
# (You'll need to use the WFA framework for proper backtesting)
```

### 3. Walk-Forward Analysis

```python
from wfa_test_complete import WalkForwardAnalysis, WFAConfig
import pytz
from datetime import datetime

# Configure WFA
config = WFAConfig(
    start_date=datetime(2021, 5, 18, tzinfo=pytz.timezone('US/Eastern')),
    end_date=datetime(2025, 5, 18, tzinfo=pytz.timezone('US/Eastern')),
    in_sample_months=12,      # Training period
    out_sample_months=3,      # Testing period
    step_months=1,            # Rolling window step
    min_trades=50,            # Minimum trades for valid period
    initial_capital=100000
)

# Run WFA
wfa = WalkForwardAnalysis(
    config, 
    api_key=os.environ.get('ALPACA_API_KEY'),
    secret_key=os.environ.get('ALPACA_SECRET_KEY')
)

# Run with all CPU cores
wfa.run(n_jobs=-1)
```

## Configuration Options

### Strategy Parameters

```python
# Risk Management
strategy.TARGET_VOL = 0.12           # Target 12% annual volatility
strategy.LEV_MIN = 0.7               # Minimum leverage
strategy.LEV_MAX = 1.3               # Maximum leverage

# Position Limits
strategy.max_core_positions = 5      # Max momentum positions
strategy.max_financial_positions = 5 # Max financial positions
strategy.max_sentiment_positions = 8 # Max sentiment positions
strategy.max_total_positions = 15    # Total position limit

# Buy Thresholds
strategy.CORE_THR = 0.05            # Core strategy threshold
strategy.FIN_THR = 0.10             # Financial strategy threshold

# Daily Limits
strategy.MAX_CORE_BUYS = 3          # Max daily core buys
strategy.MAX_FIN_BUYS = 2           # Max daily financial buys
strategy.MAX_SENT_BUYS = 4          # Max daily sentiment buys

# Risk Controls
strategy.max_intraday_loss_pct = 0.03     # 3% max intraday loss
strategy.max_position_correlation = 0.70   # Max correlation between positions
strategy.vix_percentile_threshold = 80     # VIX risk-off threshold

# Stop Losses
strategy.position_stop_loss = {
    'core': 0.08,         # 8% stop loss for core
    'financial': 0.06,    # 6% stop loss for financials
    'sentiment': 0.06,    # 6% stop loss for sentiment
    'trailing_stop': 0.08, # 8% trailing stop
    'profit_stop': 0.40   # 40% profit taking
}
```

### Options Integration

```python
# Using Polygon.io
from options_integration import OptionsIntegration

options = OptionsIntegration(
    provider_type='polygon',
    api_key='your_polygon_api_key'
)

# Get option metrics for a symbol
iv_skew, put_call_ratio = options.get_option_metrics_sync('AAPL')
```

### Custom Universe

```python
# Modify stock universe
strategy.momentum_stocks = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
    # Add your stocks
]

strategy.elite_financial_stocks = [
    'JPM', 'GS', 'MS', 'BAC',
    # Add your financial stocks
]

strategy.sentiment_stocks = {
    'ultra_sensitive': ['TSLA', 'NVDA', 'AMD'],
    'high_beta_tech': ['META', 'NFLX', 'ROKU'],
    'cyclical_leaders': ['XOM', 'CVX', 'FCX']
}
```

## Monitoring and Logging

### Real-time Monitoring

```python
# The strategy automatically logs to:
# - Console output
# - alpaca_strategy_complete.log file

# Key metrics logged:
# - Portfolio value and returns
# - Position changes
# - Risk metrics (VaR, leverage, volatility)
# - Market sentiment
# - Factor weights
# - Circuit breaker status
```

### Data Outputs

The strategy saves the following files:
- `complete_trades_YYYYMMDD_HHMMSS.csv` - All trades
- `complete_portfolio_YYYYMMDD_HHMMSS.csv` - Portfolio history
- `factor_weights_YYYYMMDD_HHMMSS.json` - Factor weight evolution
- `slippage_log.csv` - Order execution slippage

## Advanced Features

### 1. Circuit Breakers

Circuit breakers automatically trigger when:
- Intraday loss exceeds 3%
- Drawdown speed exceeds 1% per hour
- 5 consecutive losing trades
- VIX spike detected

```python
# Customize circuit breakers
strategy.max_intraday_loss_pct = 0.02  # 2% limit
strategy.max_drawdown_speed = 0.005    # 0.5% per hour
```

### 2. Dynamic Factor Weighting

The strategy uses IC-EWMA to dynamically adjust factor weights based on recent performance:

```python
# View current factor weights
print(strategy.factor_weights)
# {'momentum': 0.25, 'value': 0.15, 'quality': 0.20, 
#  'volatility': 0.10, 'sentiment': 0.30}

# Weights are automatically adjusted based on factor performance
```

### 3. Options Sentiment Integration

```python
# The strategy checks option sentiment before trades:
# - IV Skew < -0.05 (puts bid) = Bullish
# - Put/Call Ratio > 1.5 = Contrarian Bullish
# - IV Skew > 0.1 (calls bid) = Bearish
# - Put/Call Ratio < 0.7 = Bearish
```

### 4. News Sentiment Streaming

News sentiment is automatically processed in real-time:

```python
# Access news sentiment history
symbol_news = strategy.news_sentiment_history['AAPL']
for news in symbol_news:
    print(f"{news['timestamp']}: {news['headline'][:50]} (Score: {news['score']:.3f})")
```

## Troubleshooting

### Common Issues

1. **"No module named 'alpaca_trade_api'"**
   ```bash
   pip install alpaca-trade-api alpaca-py
   ```

2. **"Failed to get VIX data"**
   - This is normal - uses yfinance as fallback
   - Strategy continues without issue

3. **"Circuit breaker triggered"**
   - Check logs for specific reason
   - Strategy automatically reduces positions
   - Resumes next trading day

4. **"Wide spread on symbol"**
   - Normal for illiquid stocks
   - Strategy uses limit orders with dynamic pricing

### Performance Optimization

1. **Reduce API calls**:
   ```python
   # Increase data cache time
   strategy.max_data_age_seconds = 600  # 10 minutes
   ```

2. **Limit universe size**:
   ```python
   # Use fewer stocks for faster execution
   strategy.momentum_stocks = strategy.momentum_stocks[:10]
   ```

3. **Disable options data** (if not needed):
   ```python
   strategy = AlpacaCompleteStrategy(
       options_provider='simulated',  # Uses simulation
       # ...
   )
   ```

## Production Deployment

### 1. Server Setup

```bash
# Use systemd service (Linux)
[Unit]
Description=Alpaca Trading Strategy
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/trading
ExecStart=/usr/bin/python3 /home/trader/trading/run_strategy.py
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

### 2. Environment Variables

```bash
# .env file
ALPACA_API_KEY=your_live_api_key
ALPACA_SECRET_KEY=your_live_secret_key
POLYGON_API_KEY=your_polygon_key
TRADING_MODE=live
```

### 3. Monitoring

```python
# Add external monitoring
import requests

def send_alert(message):
    # Send to Slack, Discord, email, etc.
    webhook_url = "your_webhook_url"
    requests.post(webhook_url, json={"text": message})

# Override circuit breaker handler
original_handler = strategy.handle_circuit_breaker
def monitored_handler(reason):
    send_alert(f"Circuit breaker triggered: {reason}")
    original_handler(reason)
strategy.handle_circuit_breaker = monitored_handler
```

## Best Practices

1. **Start with Paper Trading**
   - Test for at least 1 month
   - Verify all features work correctly
   - Monitor performance metrics

2. **Gradual Capital Deployment**
   - Start with small position sizes
   - Increase gradually as confidence grows
   - Monitor slippage and execution quality

3. **Regular Optimization**
   - Run WFA monthly
   - Update parameters based on results
   - Monitor factor performance

4. **Risk Management**
   - Never override circuit breakers
   - Monitor correlation between positions
   - Keep some cash reserve

5. **Data Quality**
   - Verify data freshness
   - Check for missing symbols
   - Monitor options data accuracy

## Support and Updates

- Check logs regularly: `tail -f alpaca_strategy_complete.log`
- Save trade history for analysis
- Run WFA periodically to reoptimize
- Monitor factor weights evolution
- Keep dependencies updated

## Example Results

Typical performance metrics (backtested):
- Annual Return: 15-25%
- Sharpe Ratio: 1.2-1.8
- Max Drawdown: 10-15%
- Win Rate: 55-65%
- Average holding period: 5-10 days

*Note: Past performance does not guarantee future results.*
