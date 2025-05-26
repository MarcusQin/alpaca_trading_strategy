#!/usr/bin/env python3
"""
Walk-Forward Analysis (WFA) Test Framework
==========================================
Complete implementation for testing the Alpaca trading strategy
Period: May 18, 2021 to May 18, 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp
from itertools import product
import warnings
from dataclasses import dataclass
import pickle
import os
import yfinance as yf
from tqdm import tqdm
import alpaca_trade_api as tradeapi
from alpaca.data import StockHistoricalDataClient, StockBarsRequest, TimeFrame

# Import the complete strategy
from complete_trading_strategy import AlpacaCompleteStrategy

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('wfa_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class WFAConfig:
    """Walk-Forward Analysis Configuration"""
    start_date: datetime
    end_date: datetime
    in_sample_months: int = 12    # Training period
    out_sample_months: int = 3     # Testing period
    step_months: int = 1           # Step size
    min_trades: int = 50           # Minimum trades for valid period
    
    # Parameter ranges for optimization
    param_ranges = {
        'TARGET_VOL': [0.08, 0.10, 0.12, 0.15],
        'CORE_THR': [0.03, 0.05, 0.08],
        'FIN_THR': [0.08, 0.10, 0.12],
        'MAX_CORE_BUYS': [2, 3, 4],
        'MAX_FIN_BUYS': [1, 2, 3],
        'MAX_SENT_BUYS': [3, 4, 5],
        'vix_percentile_threshold': [70, 80, 90],
        'max_position_correlation': [0.60, 0.70, 0.80],
        'position_stop_loss_core': [0.06, 0.08, 0.10],
        'position_stop_loss_financial': [0.05, 0.06, 0.08],
        'order_timeout': [20, 30, 45],
        'max_intraday_loss_pct': [0.02, 0.03, 0.04]
    }
    
    # Optimization objectives
    objectives = ['sharpe_ratio', 'calmar_ratio', 'sortino_ratio', 'profit_factor']
    primary_objective = 'sharpe_ratio'
    
    # Backtesting parameters
    initial_capital = 100000
    commission = 0.001  # 10 bps per trade
    slippage = 0.0005   # 5 bps slippage


class DataLoader:
    """Loads and prepares historical data for backtesting"""
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.data_cache = {}
        self.eastern = pytz.timezone('US/Eastern')
        
        if api_key and secret_key:
            self.data_client = StockHistoricalDataClient(api_key, secret_key)
        else:
            self.data_client = None
    
    def load_historical_data(self, symbols: List[str], start_date: datetime, 
                           end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Load historical data for symbols"""
        logger.info(f"Loading data for {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")
        
        data = {}
        failed_symbols = []
        
        # Try to load from cache first
        cache_file = f'data_cache_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.pkl'
        if os.path.exists(cache_file):
            logger.info("Loading from cache...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                logger.info(f"Loaded {len(data)} symbols from cache")
        
        # Load missing symbols
        missing_symbols = [s for s in symbols if s not in data]
        
        if missing_symbols:
            logger.info(f"Loading {len(missing_symbols)} missing symbols...")
            
            for symbol in tqdm(missing_symbols, desc="Loading data"):
                try:
                    if self.data_client:
                        # Use Alpaca data
                        df = self._load_alpaca_data(symbol, start_date, end_date)
                    else:
                        # Fallback to yfinance
                        df = self._load_yfinance_data(symbol, start_date, end_date)
                    
                    if not df.empty:
                        data[symbol] = df
                    else:
                        failed_symbols.append(symbol)
                        
                except Exception as e:
                    logger.error(f"Failed to load {symbol}: {e}")
                    failed_symbols.append(symbol)
            
            # Save cache
            if data:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f"Saved cache with {len(data)} symbols")
        
        if failed_symbols:
            logger.warning(f"Failed to load: {failed_symbols[:10]}...")
        
        return data
    
    def _load_alpaca_data(self, symbol: str, start_date: datetime, 
                         end_date: datetime) -> pd.DataFrame:
        """Load data from Alpaca"""
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        
        bars = self.data_client.get_stock_bars(request)
        
        if symbol in bars.data:
            df_data = bars.data[symbol]
            data = pd.DataFrame([{
                'Open': bar.open,
                'High': bar.high,
                'Low': bar.low,
                'Close': bar.close,
                'Volume': bar.volume,
                'Date': bar.timestamp.tz_convert(self.eastern)
            } for bar in df_data])
            
            if not data.empty:
                data.set_index('Date', inplace=True)
                return data
        
        return pd.DataFrame()
    
    def _load_yfinance_data(self, symbol: str, start_date: datetime, 
                           end_date: datetime) -> pd.DataFrame:
        """Load data from Yahoo Finance"""
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        
        if not df.empty:
            # Ensure timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize('America/New_York')
            else:
                df.index = df.index.tz_convert('America/New_York')
        
        return df


class BacktestEngine:
    """Backtesting engine for the strategy"""
    
    def __init__(self, config: WFAConfig, data_loader: DataLoader):
        self.config = config
        self.data_loader = data_loader
        self.eastern = pytz.timezone('US/Eastern')
    
    def run_backtest(self, params: Dict, start_date: datetime, end_date: datetime,
                    historical_data: Dict[str, pd.DataFrame]) -> Dict:
        """Run backtest with given parameters"""
        logger.info(f"Running backtest from {start_date.date()} to {end_date.date()}")
        
        try:
            # Create strategy instance in backtest mode
            strategy = AlpacaCompleteStrategy(
                api_key='dummy',
                secret_key='dummy',
                mode='backtest'
            )
            
            # Override parameters
            for param, value in params.items():
                if param == 'position_stop_loss_core':
                    strategy.position_stop_loss['core'] = value
                elif param == 'position_stop_loss_financial':
                    strategy.position_stop_loss['financial'] = value
                else:
                    setattr(strategy, param, value)
            
            # Set initial capital
            strategy.initial_capital = self.config.initial_capital
            strategy.portfolio_peak = self.config.initial_capital
            
            # Load historical data into strategy
            strategy.backtest_data = historical_data
            
            # Initialize tracking
            cash = self.config.initial_capital
            positions = {}
            portfolio_history = []
            trades = []
            
            # Generate trading days
            current_date = start_date
            trading_days = []
            
            while current_date <= end_date:
                if strategy.trading_calendar.is_trading_day(current_date):
                    trading_days.append(current_date)
                current_date += timedelta(days=1)
            
            logger.info(f"Simulating {len(trading_days)} trading days...")
            
            # Simulate each trading day
            for day in tqdm(trading_days, desc="Backtesting"):
                strategy.backtest_date = day
                
                # Update market data for the day
                self._update_strategy_data(strategy, historical_data, day)
                
                # Morning preparation
                if day.hour == 9 and day.minute == 30:
                    strategy.reset_daily_counters()
                    strategy.update_portfolio_volatility()
                
                # Execute strategies
                portfolio_value = cash + sum(p['value'] for p in positions.values())
                
                # Update strategy account info for backtesting
                strategy.account = type('obj', (object,), {
                    'cash': cash,
                    'portfolio_value': portfolio_value,
                    'buying_power': cash * 2  # Assume 2x margin
                })
                
                # Execute each strategy component
                strategy.execute_passive_investment(day.date(), portfolio_value)
                strategy.execute_core_active(day.date(), portfolio_value)
                strategy.execute_elite_financial(day.date(), portfolio_value)
                strategy.execute_sentiment_driven_enhanced(day.date(), portfolio_value)
                
                # Process simulated trades
                new_trades = strategy.trades_history[-10:]  # Get recent trades
                for trade in new_trades:
                    if trade not in trades:
                        trades.append(trade)
                        
                        # Update cash and positions
                        if trade['action'] == 'BUY':
                            cost = trade['shares'] * trade.get('price', 100)
                            commission = cost * self.config.commission
                            cash -= (cost + commission)
                            
                            # Add to positions
                            symbol = trade['symbol']
                            if symbol not in positions:
                                positions[symbol] = {
                                    'shares': trade['shares'],
                                    'cost_basis': cost,
                                    'value': cost
                                }
                            else:
                                positions[symbol]['shares'] += trade['shares']
                                positions[symbol]['cost_basis'] += cost
                        
                        elif 'SELL' in trade['action']:
                            proceeds = trade['shares'] * trade.get('price', 100)
                            commission = proceeds * self.config.commission
                            cash += (proceeds - commission)
                            
                            # Update positions
                            symbol = trade['symbol']
                            if symbol in positions:
                                if trade['shares'] >= positions[symbol]['shares']:
                                    del positions[symbol]
                                else:
                                    positions[symbol]['shares'] -= trade['shares']
                                    positions[symbol]['cost_basis'] *= (
                                        positions[symbol]['shares'] / 
                                        (positions[symbol]['shares'] + trade['shares'])
                                    )
                
                # Update position values
                for symbol, pos in positions.items():
                    if symbol in strategy.market_data:
                        current_price = strategy.market_data[symbol]['Close'].iloc[-1]
                        pos['value'] = pos['shares'] * current_price
                
                # Record portfolio state
                portfolio_value = cash + sum(p['value'] for p in positions.values())
                portfolio_history.append({
                    'date': day,
                    'value': portfolio_value,
                    'cash': cash,
                    'positions': len(positions),
                    'leverage': strategy.current_leverage,
                    'vix_5d_ma': strategy.vix_5d_ma
                })
            
            # Calculate final metrics
            results = {
                'trades': trades,
                'portfolio_history': portfolio_history,
                'positions': positions,
                'final_value': portfolio_value,
                'initial_value': self.config.initial_capital
            }
            
            metrics = self.calculate_metrics(results)
            
            return {
                'params': params,
                'results': results,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}", exc_info=True)
            return None
    
    def _update_strategy_data(self, strategy, historical_data: Dict, current_date: datetime):
        """Update strategy with data up to current date"""
        # Clear existing data
        strategy.market_data = {}
        
        # Load data up to current date
        for symbol, df in historical_data.items():
            # Filter data up to current date
            mask = df.index <= current_date
            if mask.any():
                strategy.market_data[symbol] = df[mask].copy()
        
        # Update derived data
        if len(strategy.market_data) > 0:
            strategy.update_market_sentiment()
            strategy.update_factor_scores()
            strategy.update_vix_filter()
            strategy.update_correlation_matrix()
    
    def calculate_metrics(self, results: Dict) -> Dict:
        """Calculate performance metrics"""
        if not results or len(results['portfolio_history']) < 2:
            return self._empty_metrics()
        
        # Convert to pandas for easier calculation
        df = pd.DataFrame(results['portfolio_history'])
        df['returns'] = df['value'].pct_change()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Basic metrics
        total_return = (results['final_value'] / results['initial_value'] - 1)
        trading_days = len(df)
        years = trading_days / 252
        
        # Risk metrics
        daily_returns = df['returns'].dropna()
        sharpe_ratio = self._calculate_sharpe(daily_returns)
        sortino_ratio = self._calculate_sortino(daily_returns)
        calmar_ratio = self._calculate_calmar(df['value'], years)
        max_drawdown = self._calculate_max_drawdown(df['value'])
        
        # Trade metrics
        trades = results.get('trades', [])
        win_rate, profit_factor, avg_win, avg_loss = self._calculate_trade_metrics(trades)
        
        # Risk-adjusted metrics
        downside_deviation = daily_returns[daily_returns < 0].std() * np.sqrt(252)
        var_95 = np.percentile(daily_returns, 5)
        cvar_95 = daily_returns[daily_returns <= var_95].mean()
        
        return {
            'total_return': total_return,
            'annual_return': (1 + total_return) ** (1/years) - 1 if years > 0 else 0,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'num_trades': len(trades),
            'trading_days': trading_days,
            'downside_deviation': downside_deviation,
            'var_95': var_95,
            'cvar_95': cvar_95
        }
    
    def _calculate_sharpe(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0
        excess_returns = returns - risk_free_rate/252
        if returns.std() > 0:
            return np.sqrt(252) * excess_returns.mean() / returns.std()
        return 0
    
    def _calculate_sortino(self, returns, risk_free_rate=0.02):
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0
        excess_returns = returns - risk_free_rate/252
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
        return float('inf') if excess_returns.mean() > 0 else 0
    
    def _calculate_calmar(self, values, years):
        """Calculate Calmar ratio"""
        if years == 0:
            return 0
        annual_return = (values.iloc[-1] / values.iloc[0]) ** (1/years) - 1
        max_dd = self._calculate_max_drawdown(values)
        if max_dd < 0:
            return annual_return / abs(max_dd)
        return float('inf') if annual_return > 0 else 0
    
    def _calculate_max_drawdown(self, values):
        """Calculate maximum drawdown"""
        rolling_max = values.expanding().max()
        drawdowns = (values - rolling_max) / rolling_max
        return drawdowns.min()
    
    def _calculate_trade_metrics(self, trades):
        """Calculate trade-based metrics"""
        if not trades:
            return 0, 0, 0, 0
        
        pnls = []
        for t in trades:
            if 'pnl' in t:
                pnls.append(t['pnl'])
            elif 'SELL' in t.get('action', '') and 'pnl_pct' in t:
                # Estimate PnL from percentage
                pnls.append(t['pnl_pct'] * 1000)  # Rough estimate
        
        if not pnls:
            return 0, 0, 0, 0
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        win_rate = len(wins) / len(pnls) if pnls else 0
        
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        return win_rate, profit_factor, avg_win, avg_loss
    
    def _empty_metrics(self):
        """Return empty metrics dict"""
        return {
            'total_return': 0,
            'annual_return': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'num_trades': 0,
            'trading_days': 0,
            'downside_deviation': 0,
            'var_95': 0,
            'cvar_95': 0
        }


class WalkForwardOptimizer:
    """Handles parameter optimization for each window"""
    
    def __init__(self, config: WFAConfig, backtest_engine: BacktestEngine):
        self.config = config
        self.backtest_engine = backtest_engine
    
    def optimize_parameters(self, start_date: datetime, end_date: datetime,
                          historical_data: Dict[str, pd.DataFrame], 
                          n_jobs: int = -1) -> Dict:
        """Optimize parameters for given period"""
        # Generate parameter combinations
        param_names = list(self.config.param_ranges.keys())
        param_values = list(self.config.param_ranges.values())
        param_combinations = list(product(*param_values))
        
        logger.info(f"Testing {len(param_combinations)} parameter combinations")
        
        # Parallel optimization
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        
        if n_jobs > 1:
            with mp.Pool(n_jobs) as pool:
                tasks = []
                for combo in param_combinations:
                    params = dict(zip(param_names, combo))
                    tasks.append((params, start_date, end_date, historical_data))
                
                results = pool.starmap(self._evaluate_params_wrapper, tasks)
        else:
            results = []
            for combo in tqdm(param_combinations, desc="Optimizing"):
                params = dict(zip(param_names, combo))
                result = self._evaluate_params(params, start_date, end_date, historical_data)
                results.append(result)
        
        # Find best parameters
        valid_results = [r for r in results if r is not None]
        if not valid_results:
            logger.warning("No valid results from optimization")
            return None
        
        # Sort by primary objective
        best_result = max(valid_results, 
                         key=lambda x: x['metrics'][self.config.primary_objective])
        
        logger.info(f"Best {self.config.primary_objective}: {best_result['metrics'][self.config.primary_objective]:.3f}")
        
        return best_result['params']
    
    def _evaluate_params_wrapper(self, params, start_date, end_date, historical_data):
        """Wrapper for multiprocessing"""
        return self._evaluate_params(params, start_date, end_date, historical_data)
    
    def _evaluate_params(self, params: Dict, start_date: datetime, 
                        end_date: datetime, historical_data: Dict) -> Optional[Dict]:
        """Evaluate single parameter set"""
        try:
            result = self.backtest_engine.run_backtest(params, start_date, end_date, historical_data)
            
            # Check minimum trades requirement
            if result and result['metrics']['num_trades'] >= self.config.min_trades:
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to evaluate params {params}: {e}")
            return None


class WalkForwardAnalysis:
    """Main WFA orchestrator"""
    
    def __init__(self, config: WFAConfig, api_key: str = None, secret_key: str = None):
        self.config = config
        self.data_loader = DataLoader(api_key, secret_key)
        self.backtest_engine = BacktestEngine(config, self.data_loader)
        self.optimizer = WalkForwardOptimizer(config, self.backtest_engine)
        self.results = []
        self.best_params_history = []
        self.eastern = pytz.timezone('US/Eastern')
    
    def run(self, n_jobs: int = -1):
        """Run complete walk-forward analysis"""
        logger.info(f"Starting WFA from {self.config.start_date} to {self.config.end_date}")
        
        # Load all required symbols
        all_symbols = self._get_all_symbols()
        
        # Add market indicators
        all_symbols.extend(['^VIX', '^TNX', 'SPY', 'QQQ', 'IWM'])
        all_symbols = list(set(all_symbols))
        
        # Load historical data for entire period
        logger.info("Loading historical data...")
        historical_data = self.data_loader.load_historical_data(
            all_symbols,
            self.config.start_date - timedelta(days=365),  # Extra data for indicators
            self.config.end_date
        )
        
        # Generate windows
        windows = self._generate_windows()
        logger.info(f"Generated {len(windows)} windows")
        
        # Process each window
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"\n{'='*60}")
            logger.info(f"Window {i+1}/{len(windows)}")
            logger.info(f"Training: {train_start.date()} to {train_end.date()}")
            logger.info(f"Testing: {test_start.date()} to {test_end.date()}")
            
            # Optimize on training period
            best_params = self.optimizer.optimize_parameters(
                train_start, train_end, historical_data, n_jobs
            )
            
            if best_params:
                logger.info(f"Best params: {best_params}")
                self.best_params_history.append(best_params)
                
                # Test on out-of-sample period
                test_results = self.backtest_engine.run_backtest(
                    best_params, test_start, test_end, historical_data
                )
                
                if test_results:
                    self.results.append({
                        'window': i,
                        'train_period': (train_start, train_end),
                        'test_period': (test_start, test_end),
                        'best_params': best_params,
                        'test_results': test_results
                    })
                    
                    # Log test results
                    metrics = test_results['metrics']
                    logger.info(f"Test Results:")
                    logger.info(f"  Total Return: {metrics['total_return']:.1%}")
                    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                    logger.info(f"  Max Drawdown: {metrics['max_drawdown']:.1%}")
                    logger.info(f"  Win Rate: {metrics['win_rate']:.1%}")
                    
                    # Save intermediate results
                    self._save_results(f'wfa_results_window_{i}.pkl')
        
        # Generate final report
        self._generate_report()
    
    def _get_all_symbols(self) -> List[str]:
        """Get all symbols used in the strategy"""
        # This should match the symbols in your strategy
        passive_etfs = ['SPY', 'QQQ', 'IWM']
        
        elite_financial = [
            'JPM', 'GS', 'MS', 'BRK.B', 'BAC', 'WFC',
            'BLK', 'SCHW', 'V', 'MA', 'AXP', 'C',
            'PNC', 'USB', 'TFC', 'COF', 'SPGI', 'MCO'
        ]
        
        momentum = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            'ADBE', 'CRM', 'AMD', 'ORCL', 'AVGO', 'QCOM', 'INTC',
            'NFLX', 'DIS', 'HD', 'WMT', 'PG', 'KO', 'PEP', 'UNH'
        ]
        
        sentiment = [
            'NVDA', 'TSLA', 'AMD', 'COIN', 'MSTR', 'RIOT', 'MARA',
            'PLTR', 'GME', 'AMC', 'SOFI', 'LCID', 'RIVN', 'AI', 'SMCI',
            'META', 'NFLX', 'ROKU', 'SNAP', 'PINS', 'PYPL',
            'SHOP', 'TWLO', 'DOCU', 'ZM', 'CRWD', 'DDOG', 'NET', 'SNOW',
            'JPM', 'GS', 'MS', 'BAC', 'XOM', 'CVX', 'FCX',
            'NUE', 'CAT', 'DE', 'BA', 'AAL', 'DAL', 'LUV', 'UAL'
        ]
        
        sectors = ['XLK', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLRE', 'XLU', 'XLF']
        
        all_symbols = list(set(passive_etfs + elite_financial + momentum + sentiment + sectors))
        return all_symbols
    
    def _generate_windows(self) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate train/test windows"""
        windows = []
        current = self.config.start_date
        
        while current < self.config.end_date:
            # Training period
            train_start = current
            train_end = train_start + timedelta(days=30 * self.config.in_sample_months)
            
            # Testing period
            test_start = train_end
            test_end = test_start + timedelta(days=30 * self.config.out_sample_months)
            
            # Check if we have enough data
            if test_end <= self.config.end_date:
                windows.append((
                    self._ensure_tz(train_start),
                    self._ensure_tz(train_end),
                    self._ensure_tz(test_start),
                    self._ensure_tz(test_end)
                ))
            
            # Step forward
            current += timedelta(days=30 * self.config.step_months)
        
        return windows
    
    def _ensure_tz(self, dt):
        """Ensure timezone aware datetime"""
        if dt.tzinfo is None:
            return self.eastern.localize(dt)
        return dt
    
    def _generate_report(self):
        """Generate comprehensive WFA report"""
        if not self.results:
            logger.error("No results to report")
            return
        
        # Aggregate metrics
        all_metrics = []
        for result in self.results:
            window_metrics = result['test_results']['metrics'].copy()
            window_metrics['window'] = result['window']
            window_metrics['test_start'] = result['test_period'][0]
            window_metrics['test_end'] = result['test_period'][1]
            all_metrics.append(window_metrics)
        
        metrics_df = pd.DataFrame(all_metrics)
        
        # Create visualizations
        self._plot_results(metrics_df)
        
        # Summary statistics
        logger.info("\n" + "="*80)
        logger.info("WALK-FORWARD ANALYSIS SUMMARY")
        logger.info("="*80)
        
        logger.info(f"\nPeriod: {self.config.start_date.date()} to {self.config.end_date.date()}")
        logger.info(f"Windows: {len(self.results)}")
        logger.info(f"In-sample months: {self.config.in_sample_months}")
        logger.info(f"Out-sample months: {self.config.out_sample_months}")
        
        logger.info("\nOut-of-Sample Performance:")
        for metric in ['sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate', 'profit_factor']:
            if metric in metrics_df.columns:
                mean_val = metrics_df[metric].mean()
                std_val = metrics_df[metric].std()
                min_val = metrics_df[metric].min()
                max_val = metrics_df[metric].max()
                
                if metric in ['total_return', 'max_drawdown', 'win_rate']:
                    logger.info(f"  {metric}: {mean_val:.1%} ± {std_val:.1%} "
                               f"[{min_val:.1%}, {max_val:.1%}]")
                else:
                    logger.info(f"  {metric}: {mean_val:.3f} ± {std_val:.3f} "
                               f"[{min_val:.3f}, {max_val:.3f}]")
        
        # Parameter stability
        self._analyze_parameter_stability()
        
        # Create combined equity curve
        self._create_equity_curve()
        
        # Save final results
        self._save_results('wfa_final_results.pkl')
        metrics_df.to_csv('wfa_metrics_summary.csv', index=False)
        
        # Save parameter history
        param_df = pd.DataFrame(self.best_params_history)
        param_df.to_csv('wfa_parameter_history.csv', index=False)
    
    def _plot_results(self, metrics_df: pd.DataFrame):
        """Create visualization plots"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # Plot 1: Sharpe ratio over time
        ax1 = axes[0, 0]
        metrics_df.plot(x='test_start', y='sharpe_ratio', ax=ax1, marker='o')
        ax1.axhline(y=metrics_df['sharpe_ratio'].mean(), color='r', linestyle='--', label='Mean')
        ax1.fill_between(range(len(metrics_df)), 
                        metrics_df['sharpe_ratio'].mean() - metrics_df['sharpe_ratio'].std(),
                        metrics_df['sharpe_ratio'].mean() + metrics_df['sharpe_ratio'].std(),
                        alpha=0.2, color='red')
        ax1.set_title('Out-of-Sample Sharpe Ratio')
        ax1.set_xlabel('Test Period Start')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.legend()
        
        # Plot 2: Returns over time
        ax2 = axes[0, 1]
        metrics_df['cumulative_return'] = (1 + metrics_df['total_return']).cumprod() - 1
        metrics_df.plot(x='test_start', y='total_return', ax=ax2, marker='o', color='green')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('Out-of-Sample Returns')
        ax2.set_xlabel('Test Period Start')
        ax2.set_ylabel('Total Return')
        
        # Plot 3: Drawdown over time
        ax3 = axes[1, 0]
        metrics_df.plot(x='test_start', y='max_drawdown', ax=ax3, marker='o', color='red')
        ax3.set_title('Maximum Drawdown')
        ax3.set_xlabel('Test Period Start')
        ax3.set_ylabel('Max Drawdown')
        
        # Plot 4: Win rate over time
        ax4 = axes[1, 1]
        metrics_df.plot(x='test_start', y='win_rate', ax=ax4, marker='o', color='blue')
        ax4.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
        ax4.set_title('Win Rate')
        ax4.set_xlabel('Test Period Start')
        ax4.set_ylabel('Win Rate')
        
        # Plot 5: Rolling performance
        ax5 = axes[2, 0]
        rolling_sharpe = metrics_df['sharpe_ratio'].rolling(3).mean()
        ax5.plot(metrics_df['test_start'], rolling_sharpe, marker='o', color='purple')
        ax5.set_title('3-Window Rolling Sharpe')
        ax5.set_xlabel('Test Period Start')
        ax5.set_ylabel('Rolling Sharpe')
        
        # Plot 6: Performance distribution
        ax6 = axes[2, 1]
        metrics_df['sharpe_ratio'].hist(ax=ax6, bins=15, alpha=0.7, color='orange')
        ax6.axvline(x=metrics_df['sharpe_ratio'].mean(), color='red', linestyle='--')
        ax6.set_title('Sharpe Ratio Distribution')
        ax6.set_xlabel('Sharpe Ratio')
        ax6.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('wfa_results_visualization.png', dpi=300)
        plt.show()
    
    def _analyze_parameter_stability(self):
        """Analyze parameter stability across windows"""
        if not self.best_params_history:
            return
        
        # Collect all parameters
        param_history = pd.DataFrame(self.best_params_history)
        
        logger.info("\nParameter Stability Analysis:")
        for col in param_history.columns:
            unique_vals = param_history[col].nunique()
            most_common = param_history[col].mode()[0]
            frequency = (param_history[col] == most_common).sum() / len(param_history)
            
            logger.info(f"  {col}:")
            logger.info(f"    Unique values: {unique_vals}")
            logger.info(f"    Most common: {most_common} ({frequency:.1%} of windows)")
            logger.info(f"    Mean: {param_history[col].mean():.3f}")
            logger.info(f"    Std: {param_history[col].std():.3f}")
            
            # Plot parameter evolution
            plt.figure(figsize=(10, 4))
            plt.plot(param_history[col], marker='o')
            plt.title(f'Parameter Evolution: {col}')
            plt.xlabel('Window')
            plt.ylabel('Value')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'param_evolution_{col}.png')
            plt.close()
    
    def _create_equity_curve(self):
        """Create combined equity curve from all windows"""
        all_values = []
        all_dates = []
        
        for result in self.results:
            portfolio_history = result['test_results']['results']['portfolio_history']
            for entry in portfolio_history:
                all_dates.append(entry['date'])
                all_values.append(entry['value'])
        
        # Create DataFrame and sort by date
        equity_df = pd.DataFrame({'date': all_dates, 'value': all_values})
        equity_df = equity_df.sort_values('date').drop_duplicates('date')
        equity_df['returns'] = equity_df['value'].pct_change()
        
        # Calculate cumulative metrics
        equity_df['cumulative_return'] = (equity_df['value'] / self.config.initial_capital - 1)
        equity_df['rolling_max'] = equity_df['value'].expanding().max()
        equity_df['drawdown'] = (equity_df['value'] - equity_df['rolling_max']) / equity_df['rolling_max']
        
        # Plot equity curve
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Equity curve
        ax1.plot(equity_df['date'], equity_df['value'], label='Portfolio Value')
        ax1.fill_between(equity_df['date'], self.config.initial_capital, equity_df['value'],
                        where=equity_df['value'] >= self.config.initial_capital,
                        alpha=0.3, color='green', label='Profit')
        ax1.fill_between(equity_df['date'], self.config.initial_capital, equity_df['value'],
                        where=equity_df['value'] < self.config.initial_capital,
                        alpha=0.3, color='red', label='Loss')
        ax1.axhline(y=self.config.initial_capital, color='black', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Walk-Forward Analysis: Combined Equity Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2.fill_between(equity_df['date'], 0, equity_df['drawdown'],
                        alpha=0.7, color='red')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('wfa_equity_curve.png', dpi=300)
        plt.show()
        
        # Save equity curve data
        equity_df.to_csv('wfa_equity_curve.csv', index=False)
    
    def _save_results(self, filename: str):
        """Save results to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'results': self.results,
                'best_params_history': self.best_params_history,
                'timestamp': datetime.now()
            }, f)


def main():
    """Run Walk-Forward Analysis"""
    # API credentials (optional - will use yfinance if not provided)
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    # Configure WFA
    config = WFAConfig(
        start_date=datetime(2021, 5, 18, tzinfo=pytz.timezone('US/Eastern')),
        end_date=datetime(2025, 5, 18, tzinfo=pytz.timezone('US/Eastern')),
        in_sample_months=12,
        out_sample_months=3,
        step_months=1,
        min_trades=50
    )
    
    # Run WFA
    wfa = WalkForwardAnalysis(config, api_key, secret_key)
    
    # Use all CPU cores for optimization
    wfa.run(n_jobs=-1)
    
    print("\nWalk-Forward Analysis Complete!")
    print("Results saved to:")
    print("- wfa_final_results.pkl")
    print("- wfa_metrics_summary.csv")
    print("- wfa_parameter_history.csv")
    print("- wfa_equity_curve.csv")
    print("- wfa_results_visualization.png")
    print("- wfa_equity_curve.png")


if __name__ == "__main__":
    main()

