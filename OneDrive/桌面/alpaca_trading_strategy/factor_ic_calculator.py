#!/usr/bin/env python3
"""
Fixed Factor IC Calculation Module
==================================
Properly handles trading days and market holidays
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
import logging

logger = logging.getLogger(__name__)


class TradingCalendar:
    """Handles trading days and market holidays"""
    
    def __init__(self):
        # US market holidays
        self.calendar = USFederalHolidayCalendar()
        
        # Additional market holidays not in federal calendar
        self.market_holidays = [
            '2021-04-02',  # Good Friday 2021
            '2022-04-15',  # Good Friday 2022
            '2023-04-07',  # Good Friday 2023
            '2024-03-29',  # Good Friday 2024
            '2025-04-18',  # Good Friday 2025
        ]
        
        # Create custom business day
        holidays = self.calendar.holidays(start='2020-01-01', end='2030-01-01')
        holidays = holidays.append(pd.DatetimeIndex(self.market_holidays))
        self.trading_day = CustomBusinessDay(holidays=holidays)
        
    def get_next_n_trading_days(self, start_date: pd.Timestamp, n: int) -> pd.DatetimeIndex:
        """Get next n trading days from start_date"""
        # Ensure timezone aware
        if start_date.tz is None:
            start_date = pytz.timezone('US/Eastern').localize(start_date)
        
        # Generate trading days
        end_date = start_date + pd.Timedelta(days=n*2)  # Rough estimate
        trading_days = pd.date_range(start=start_date, end=end_date, freq=self.trading_day)
        
        # Return exactly n days
        return trading_days[:n]
    
    def get_trading_days_between(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DatetimeIndex:
        """Get all trading days between two dates"""
        return pd.date_range(start=start_date, end=end_date, freq=self.trading_day)
    
    def is_trading_day(self, date: pd.Timestamp) -> bool:
        """Check if a date is a trading day"""
        # Check if it's a weekday
        if date.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check holidays
        holidays = self.calendar.holidays(start=date, end=date)
        if date.normalize() in holidays:
            return False
        
        # Check additional market holidays
        if date.strftime('%Y-%m-%d') in self.market_holidays:
            return False
        
        return True
    
    def get_previous_trading_day(self, date: pd.Timestamp) -> pd.Timestamp:
        """Get previous trading day"""
        prev_day = date - pd.Timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= pd.Timedelta(days=1)
        return prev_day


class ImprovedFactorICCalculator:
    """Improved Factor IC calculation with proper trading day handling"""
    
    def __init__(self):
        self.calendar = TradingCalendar()
        self.ic_history = {}
        self.factor_weights = {}
        
    def calculate_factor_ic_and_weights(self, factor_scores: dict, market_data: dict, 
                                      factor_names: list, ic_ewma_alpha: float = 0.05) -> dict:
        """Calculate factor ICs with proper trading day handling"""
        
        # Initialize if needed
        for factor in factor_names:
            if factor not in self.ic_history:
                self.ic_history[factor] = []
            if factor not in self.factor_weights:
                self.factor_weights[factor] = 1.0 / len(factor_names)
        
        # Calculate ICs for each factor
        for factor in factor_names:
            factor_data = []
            return_data = []
            
            # Collect factor scores and forward returns
            for symbol in factor_scores:
                if symbol not in market_data:
                    continue
                
                df = market_data[symbol]
                if len(df) < 20:  # Need sufficient history
                    continue
                
                # Get all dates with factor scores
                score_dates = sorted(factor_scores[symbol].keys())
                
                for score_date in score_dates[:-5]:  # Leave room for forward returns
                    if factor not in factor_scores[symbol][score_date].get('scores', {}):
                        continue
                    
                    factor_score = factor_scores[symbol][score_date]['scores'][factor]
                    
                    # Find score date in market data
                    try:
                        # Convert score_date to timestamp if needed
                        if isinstance(score_date, pd.Timestamp):
                            score_timestamp = score_date
                        else:
                            score_timestamp = pd.Timestamp(score_date)
                        
                        # Ensure timezone consistency
                        if score_timestamp.tz is None and df.index.tz is not None:
                            score_timestamp = df.index.tz.localize(score_timestamp)
                        elif score_timestamp.tz is not None and df.index.tz is None:
                            score_timestamp = score_timestamp.tz_localize(None)
                        
                        # Find exact or nearest date in dataframe
                        if score_timestamp in df.index:
                            start_idx = df.index.get_loc(score_timestamp)
                        else:
                            # Find nearest trading day
                            nearest_idx = df.index.get_indexer([score_timestamp], method='nearest')[0]
                            if nearest_idx >= 0 and nearest_idx < len(df):
                                start_idx = nearest_idx
                            else:
                                continue
                        
                        # Get next 5 trading days
                        start_date = df.index[start_idx]
                        next_trading_days = self.calendar.get_next_n_trading_days(start_date, 6)[1:]  # Skip first day
                        
                        # Find these days in the dataframe
                        future_prices = []
                        for td in next_trading_days:
                            # Normalize to date for comparison
                            td_date = td.date()
                            mask = df.index.date == td_date
                            if mask.any():
                                idx = df.index[mask][0]
                                future_prices.append(df.loc[idx, 'Close'])
                        
                        # Calculate return if we have enough future prices
                        if len(future_prices) >= 5:
                            start_price = df.iloc[start_idx]['Close']
                            end_price = future_prices[4]  # 5th trading day
                            forward_return = (end_price / start_price - 1)
                            
                            factor_data.append(factor_score)
                            return_data.append(forward_return)
                        
                    except Exception as e:
                        logger.debug(f"Error calculating forward return for {symbol} on {score_date}: {e}")
                        continue
            
            # Calculate IC if enough data points
            if len(factor_data) >= 20:
                try:
                    # Calculate information coefficient (rank correlation)
                    ic = pd.Series(factor_data).corr(pd.Series(return_data), method='spearman')
                    
                    if not np.isnan(ic):
                        self.ic_history[factor].append(ic)
                        
                        # Calculate IC-EWMA
                        if len(self.ic_history[factor]) >= 5:
                            ic_series = pd.Series(self.ic_history[factor])
                            ic_ewma = ic_series.ewm(alpha=ic_ewma_alpha, adjust=False).mean().iloc[-1]
                            ic_std = ic_series.std()
                            
                            # Update weight based on IC-EWMA
                            if ic_std > 0:
                                # Information ratio
                                ir = ic_ewma / ic_std
                                
                                # Weight multiplier (bounded)
                                weight_multiplier = np.clip(0.5 + ir, 0.1, 2.0)
                                self.factor_weights[factor] *= weight_multiplier
                                
                                # Apply minimum weight threshold
                                self.factor_weights[factor] = max(0.05, self.factor_weights[factor])
                            
                            logger.info(f"Factor {factor}: IC={ic:.3f}, IC-EWMA={ic_ewma:.3f}, "
                                      f"IC-Std={ic_std:.3f}, Weight={self.factor_weights[factor]:.3f}")
                        else:
                            logger.info(f"Factor {factor}: IC={ic:.3f} (building history)")
                
                except Exception as e:
                    logger.error(f"Error calculating IC for {factor}: {e}")
        
        # Normalize weights
        total_weight = sum(self.factor_weights.values())
        if total_weight > 0:
            self.factor_weights = {k: v/total_weight for k, v in self.factor_weights.items()}
        
        return self.factor_weights
    
    def calculate_rolling_ic(self, symbol: str, factor_scores: pd.Series, 
                           returns: pd.Series, window: int = 60) -> pd.Series:
        """Calculate rolling IC for a single symbol/factor combination"""
        
        # Ensure proper alignment
        aligned_scores, aligned_returns = factor_scores.align(returns, join='inner')
        
        if len(aligned_scores) < window:
            return pd.Series()
        
        # Calculate rolling correlation
        rolling_ic = aligned_scores.rolling(window).corr(aligned_returns)
        
        return rolling_ic
    
    def analyze_factor_stability(self, ic_history: dict, min_periods: int = 20) -> dict:
        """Analyze factor stability over time"""
        stability_metrics = {}
        
        for factor, ic_values in ic_history.items():
            if len(ic_values) < min_periods:
                continue
            
            ic_series = pd.Series(ic_values)
            
            # Calculate stability metrics
            stability_metrics[factor] = {
                'mean_ic': ic_series.mean(),
                'std_ic': ic_series.std(),
                'sharpe_ic': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
                'min_ic': ic_series.min(),
                'max_ic': ic_series.max(),
                'positive_rate': (ic_series > 0).sum() / len(ic_series),
                'current_ic': ic_series.iloc[-1],
                'trend': ic_series.iloc[-10:].mean() - ic_series.iloc[-20:-10].mean() if len(ic_series) >= 20 else 0
            }
        
        return stability_metrics


class FactorScoreCalculator:
    """Enhanced factor score calculation with proper date handling"""
    
    def __init__(self):
        self.calendar = TradingCalendar()
        
    def calculate_momentum_score_improved(self, df: pd.DataFrame, 
                                        lookback_days: list = [5, 20, 60]) -> float:
        """Calculate momentum score with proper trading day lookback"""
        if len(df) < max(lookback_days):
            return 0
        
        momentum_score = 0
        weights = [0.4, 0.4, 0.2]  # Weights for different lookback periods
        
        current_date = df.index[-1]
        
        for days, weight in zip(lookback_days, weights):
            try:
                # Get the date N trading days ago
                past_trading_days = self.calendar.get_next_n_trading_days(
                    current_date - pd.Timedelta(days=days*2), days + 1
                )
                
                # Find the closest date in our data
                past_date = past_trading_days[0]
                if past_date in df.index:
                    past_idx = df.index.get_loc(past_date)
                else:
                    # Get nearest available date
                    past_idx = df.index.get_indexer([past_date], method='nearest')[0]
                
                if 0 <= past_idx < len(df) - 1:
                    past_price = df.iloc[past_idx]['Close']
                    current_price = df.iloc[-1]['Close']
                    period_return = (current_price / past_price - 1)
                    momentum_score += period_return * weight * 2  # Scale factor
                
            except Exception as e:
                logger.debug(f"Error calculating {days}-day momentum: {e}")
                continue
        
        # Trend consistency bonus
        if all(col in df.columns for col in ['Returns_5', 'Returns_20', 'Returns_60']):
            r5 = df['Returns_5'].iloc[-1]
            r20 = df['Returns_20'].iloc[-1]
            r60 = df['Returns_60'].iloc[-1]
            
            if r5 > 0 and r20 > 0 and r60 > 0:
                momentum_score *= 1.2
            elif r5 < 0 and r20 < 0 and r60 < 0:
                momentum_score *= 1.2
        
        return np.clip(momentum_score, -1, 1)
    
    def calculate_value_reversion_score(self, df: pd.DataFrame, symbol: str, 
                                      sector_data: dict = None) -> float:
        """Calculate value/mean reversion score"""
        value_score = 0
        
        # Price relative to recent range
        if 'Pct_from_52w_High' in df.columns and 'Pct_from_52w_Low' in df.columns:
            pct_high = df['Pct_from_52w_High'].iloc[-1]
            pct_low = df['Pct_from_52w_Low'].iloc[-1]
            
            # Deep value opportunity
            if pct_high < -0.30:  # 30% below 52w high
                value_score += 0.5
            elif pct_high < -0.20:
                value_score += 0.3
            
            # Overextended penalty
            if pct_low > 0.90:  # 90% above 52w low
                value_score -= 0.3
        
        # RSI mean reversion
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            if rsi < 30:
                value_score += 0.3
            elif rsi < 40:
                value_score += 0.1
            elif rsi > 70:
                value_score -= 0.2
        
        # Sector relative value
        if sector_data and symbol in sector_data:
            sector_return = sector_data[symbol].get('sector_relative_return', 0)
            if sector_return < -0.10:  # Underperforming sector by 10%
                value_score += 0.2
        
        # Bollinger band position
        if 'BB_Position' in df.columns:
            bb_pos = df['BB_Position'].iloc[-1]
            if bb_pos < 0.2:  # Near lower band
                value_score += 0.2
            elif bb_pos > 0.8:  # Near upper band
                value_score -= 0.1
        
        return np.clip(value_score, -1, 1)
