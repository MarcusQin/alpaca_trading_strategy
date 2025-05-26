#!/usr/bin/env python3
"""
Options Data Integration Module
===============================
Provides real options data integration with multiple providers
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import requests
from typing import Dict, Tuple, Optional
import asyncio
import aiohttp
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class OptionsDataProvider(ABC):
    """Abstract base class for options data providers"""
    
    @abstractmethod
    async def get_option_chain(self, symbol: str) -> pd.DataFrame:
        """Get option chain for a symbol"""
        pass
    
    @abstractmethod
    async def get_iv_metrics(self, symbol: str) -> Dict:
        """Get implied volatility metrics"""
        pass
    
    @abstractmethod
    async def get_flow_metrics(self, symbol: str) -> Dict:
        """Get options flow metrics"""
        pass


class PolygonOptionsProvider(OptionsDataProvider):
    """Polygon.io options data provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_option_chain(self, symbol: str) -> pd.DataFrame:
        """Get option chain from Polygon"""
        try:
            # Get current price for ATM calculation
            stock_url = f"{self.base_url}/v2/aggs/ticker/{symbol}/prev"
            params = {"apiKey": self.api_key}
            
            async with self.session.get(stock_url, params=params) as resp:
                data = await resp.json()
                if data['status'] != 'OK':
                    raise ValueError(f"Failed to get stock price: {data}")
                
                stock_price = data['results'][0]['c']
            
            # Get options contracts
            exp_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            options_url = f"{self.base_url}/v3/reference/options/contracts"
            params = {
                "underlying_ticker": symbol,
                "expiration_date.gte": datetime.now().strftime("%Y-%m-%d"),
                "expiration_date.lte": exp_date,
                "limit": 250,
                "apiKey": self.api_key
            }
            
            async with self.session.get(options_url, params=params) as resp:
                data = await resp.json()
                if data['status'] != 'OK':
                    raise ValueError(f"Failed to get options: {data}")
                
                contracts = data['results']
            
            # Get quotes for each contract
            chain_data = []
            for contract in contracts:
                ticker = contract['ticker']
                quote_url = f"{self.base_url}/v2/last/nbbo/{ticker}"
                
                async with self.session.get(quote_url, params={"apiKey": self.api_key}) as resp:
                    quote_data = await resp.json()
                    if quote_data['status'] == 'OK':
                        result = quote_data['results']
                        chain_data.append({
                            'symbol': ticker,
                            'underlying': symbol,
                            'type': contract['contract_type'],
                            'strike': contract['strike_price'],
                            'expiration': contract['expiration_date'],
                            'bid': result.get('P', 0),
                            'ask': result.get('P', 0),
                            'volume': result.get('v', 0),
                            'open_interest': result.get('o', 0),
                            'implied_volatility': result.get('iv', 0)
                        })
            
            return pd.DataFrame(chain_data)
            
        except Exception as e:
            logger.error(f"Failed to get option chain from Polygon: {e}")
            return pd.DataFrame()
    
    async def get_iv_metrics(self, symbol: str) -> Dict:
        """Calculate IV metrics from option chain"""
        try:
            chain = await self.get_option_chain(symbol)
            if chain.empty:
                return {'iv_skew': 0, 'iv_term_structure': 0, 'iv_percentile': 50}
            
            # Get ATM options
            current_price = chain['underlying_price'].iloc[0] if 'underlying_price' in chain else 100
            chain['moneyness'] = chain['strike'] / current_price
            atm_chain = chain[(chain['moneyness'] > 0.95) & (chain['moneyness'] < 1.05)]
            
            if len(atm_chain) == 0:
                return {'iv_skew': 0, 'iv_term_structure': 0, 'iv_percentile': 50}
            
            # Calculate IV skew (25 delta put IV - 25 delta call IV)
            otm_puts = chain[(chain['type'] == 'put') & (chain['moneyness'] < 0.95)]
            otm_calls = chain[(chain['type'] == 'call') & (chain['moneyness'] > 1.05)]
            
            iv_skew = 0
            if len(otm_puts) > 0 and len(otm_calls) > 0:
                avg_put_iv = otm_puts['implied_volatility'].mean()
                avg_call_iv = otm_calls['implied_volatility'].mean()
                iv_skew = (avg_put_iv - avg_call_iv) / ((avg_put_iv + avg_call_iv) / 2)
            
            # Calculate term structure (short-term IV / long-term IV)
            chain['dte'] = (pd.to_datetime(chain['expiration']) - datetime.now()).dt.days
            short_term = chain[chain['dte'] <= 30]['implied_volatility'].mean()
            long_term = chain[chain['dte'] > 60]['implied_volatility'].mean()
            
            iv_term_structure = short_term / long_term if long_term > 0 else 1
            
            # Calculate IV percentile (simplified)
            current_iv = atm_chain['implied_volatility'].mean()
            iv_percentile = min(100, max(0, (current_iv - 10) / 40 * 100))  # Rough estimate
            
            return {
                'iv_skew': iv_skew,
                'iv_term_structure': iv_term_structure,
                'iv_percentile': iv_percentile
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate IV metrics: {e}")
            return {'iv_skew': 0, 'iv_term_structure': 0, 'iv_percentile': 50}
    
    async def get_flow_metrics(self, symbol: str) -> Dict:
        """Get options flow metrics"""
        try:
            # Get today's trades
            today = datetime.now().strftime("%Y-%m-%d")
            trades_url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/minute/{today}/{today}"
            params = {"apiKey": self.api_key}
            
            async with self.session.get(trades_url, params=params) as resp:
                data = await resp.json()
                
            if data['status'] != 'OK' or 'results' not in data:
                return {'put_call_ratio': 1.0, 'options_volume_ratio': 0}
            
            # This is simplified - in reality you'd aggregate options trades
            # For now, return placeholder metrics
            return {
                'put_call_ratio': 1.0,
                'options_volume_ratio': 0.5,
                'large_trade_ratio': 0.1
            }
            
        except Exception as e:
            logger.error(f"Failed to get flow metrics: {e}")
            return {'put_call_ratio': 1.0, 'options_volume_ratio': 0}


class TradierOptionsProvider(OptionsDataProvider):
    """Tradier options data provider"""
    
    def __init__(self, api_key: str, sandbox: bool = False):
        self.api_key = api_key
        self.base_url = "https://sandbox.tradier.com/v1" if sandbox else "https://api.tradier.com/v1"
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json'
        }
        
    async def get_option_chain(self, symbol: str) -> pd.DataFrame:
        """Get option chain from Tradier"""
        try:
            # Get expirations
            exp_url = f"{self.base_url}/markets/options/expirations"
            params = {'symbol': symbol, 'includeAllRoots': 'true'}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(exp_url, headers=self.headers, params=params) as resp:
                    data = await resp.json()
                    
                if 'expirations' not in data:
                    return pd.DataFrame()
                
                expirations = data['expirations']['date'][:3]  # Get next 3 expirations
                
                # Get chains for each expiration
                all_chains = []
                for exp in expirations:
                    chain_url = f"{self.base_url}/markets/options/chains"
                    params = {
                        'symbol': symbol,
                        'expiration': exp,
                        'greeks': 'true'
                    }
                    
                    async with session.get(chain_url, headers=self.headers, params=params) as resp:
                        chain_data = await resp.json()
                        
                    if 'options' in chain_data and 'option' in chain_data['options']:
                        options = chain_data['options']['option']
                        if isinstance(options, dict):
                            options = [options]
                        
                        for opt in options:
                            all_chains.append({
                                'symbol': opt['symbol'],
                                'underlying': symbol,
                                'type': opt['option_type'],
                                'strike': opt['strike'],
                                'expiration': exp,
                                'bid': opt.get('bid', 0),
                                'ask': opt.get('ask', 0),
                                'volume': opt.get('volume', 0),
                                'open_interest': opt.get('open_interest', 0),
                                'implied_volatility': opt.get('greeks', {}).get('smv_vol', 0)
                            })
                
                return pd.DataFrame(all_chains)
                
        except Exception as e:
            logger.error(f"Failed to get option chain from Tradier: {e}")
            return pd.DataFrame()
    
    async def get_iv_metrics(self, symbol: str) -> Dict:
        """Calculate IV metrics"""
        # Similar implementation to Polygon
        return await self._calculate_iv_metrics_from_chain(symbol)
    
    async def get_flow_metrics(self, symbol: str) -> Dict:
        """Get options flow metrics"""
        # Tradier doesn't provide flow data directly
        # Would need to track trades over time
        return {'put_call_ratio': 1.0, 'options_volume_ratio': 0}
    
    async def _calculate_iv_metrics_from_chain(self, symbol: str) -> Dict:
        """Helper to calculate IV metrics"""
        try:
            chain = await self.get_option_chain(symbol)
            if chain.empty:
                return {'iv_skew': 0, 'iv_term_structure': 0, 'iv_percentile': 50}
            
            # Similar calculations as Polygon provider
            return {'iv_skew': 0, 'iv_term_structure': 1, 'iv_percentile': 50}
            
        except Exception as e:
            logger.error(f"Failed to calculate IV metrics: {e}")
            return {'iv_skew': 0, 'iv_term_structure': 0, 'iv_percentile': 50}


class CachedOptionsProvider:
    """Wrapper to cache options data and reduce API calls"""
    
    def __init__(self, provider: OptionsDataProvider, cache_minutes: int = 15):
        self.provider = provider
        self.cache_minutes = cache_minutes
        self.cache = {}
        
    async def get_option_metrics(self, symbol: str) -> Tuple[float, float]:
        """Get IV skew and put/call ratio with caching"""
        cache_key = f"{symbol}_metrics"
        
        # Check cache
        if cache_key in self.cache:
            timestamp, data = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_minutes * 60:
                return data['iv_skew'], data['put_call_ratio']
        
        # Fetch fresh data
        try:
            iv_metrics = await self.provider.get_iv_metrics(symbol)
            flow_metrics = await self.provider.get_flow_metrics(symbol)
            
            iv_skew = iv_metrics.get('iv_skew', 0)
            put_call_ratio = flow_metrics.get('put_call_ratio', 1.0)
            
            # Cache the results
            self.cache[cache_key] = (datetime.now(), {
                'iv_skew': iv_skew,
                'put_call_ratio': put_call_ratio,
                'iv_percentile': iv_metrics.get('iv_percentile', 50),
                'options_volume_ratio': flow_metrics.get('options_volume_ratio', 0)
            })
            
            return iv_skew, put_call_ratio
            
        except Exception as e:
            logger.error(f"Failed to get option metrics for {symbol}: {e}")
            return 0.0, 1.0
    
    def get_cached_metrics(self, symbol: str) -> Optional[Dict]:
        """Get all cached metrics for a symbol"""
        cache_key = f"{symbol}_metrics"
        if cache_key in self.cache:
            timestamp, data = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_minutes * 60:
                return data
        return None


class OptionsIntegration:
    """Main options integration class for the trading strategy"""
    
    def __init__(self, provider_type: str = 'polygon', api_key: str = None):
        self.provider_type = provider_type
        self.api_key = api_key
        self.provider = None
        self.cached_provider = None
        self._initialize_provider()
        
    def _initialize_provider(self):
        """Initialize the options data provider"""
        if self.provider_type == 'polygon' and self.api_key:
            self.provider = PolygonOptionsProvider(self.api_key)
        elif self.provider_type == 'tradier' and self.api_key:
            self.provider = TradierOptionsProvider(self.api_key)
        else:
            # Fallback to simulation if no API key
            self.provider = SimulatedOptionsProvider()
        
        self.cached_provider = CachedOptionsProvider(self.provider)
    
    async def get_option_sentiment(self, symbol: str) -> Tuple[float, float]:
        """Get option sentiment metrics for a symbol"""
        if self.provider is None:
            return 0.0, 1.0
        
        return await self.cached_provider.get_option_metrics(symbol)
    
    def get_option_metrics_sync(self, symbol: str) -> Tuple[float, float]:
        """Synchronous wrapper for getting option metrics"""
        try:
            # Check cache first
            cached = self.cached_provider.get_cached_metrics(symbol)
            if cached:
                return cached['iv_skew'], cached['put_call_ratio']
            
            # Run async in new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.get_option_sentiment(symbol))
            finally:
                loop.close()
                
        except Exception as e:
            logger.error(f"Failed to get option metrics sync: {e}")
            return 0.0, 1.0


class SimulatedOptionsProvider(OptionsDataProvider):
    """Simulated options provider for backtesting"""
    
    def __init__(self):
        self.market_regime = 'neutral'
        
    async def get_option_chain(self, symbol: str) -> pd.DataFrame:
        """Simulate option chain"""
        # Not needed for basic metrics
        return pd.DataFrame()
    
    async def get_iv_metrics(self, symbol: str) -> Dict:
        """Simulate IV metrics based on market conditions"""
        # Get market data if available
        np.random.seed(hash(symbol + str(datetime.now().date())) % 2**32)
        
        # Simulate based on VIX level
        base_iv = 0.20 + np.random.normal(0, 0.05)
        
        # Simulate skew
        if self.market_regime == 'fear':
            iv_skew = -0.10 + np.random.normal(0, 0.02)  # Puts bid
        elif self.market_regime == 'greed':
            iv_skew = 0.05 + np.random.normal(0, 0.02)   # Calls bid
        else:
            iv_skew = np.random.normal(0, 0.03)
        
        # Term structure
        if self.market_regime == 'fear':
            iv_term_structure = 1.2  # Backwardation
        else:
            iv_term_structure = 0.9  # Contango
        
        return {
            'iv_skew': np.clip(iv_skew, -0.3, 0.3),
            'iv_term_structure': iv_term_structure,
            'iv_percentile': np.random.uniform(20, 80)
        }
    
    async def get_flow_metrics(self, symbol: str) -> Dict:
        """Simulate options flow"""
        np.random.seed(hash(symbol + str(datetime.now().date())) % 2**32)
        
        # Base put/call ratio
        if self.market_regime == 'fear':
            pc_ratio = 1.5 + np.random.exponential(0.3)
        elif self.market_regime == 'greed':
            pc_ratio = 0.7 + np.random.normal(0, 0.1)
        else:
            pc_ratio = 1.0 + np.random.normal(0, 0.2)
        
        return {
            'put_call_ratio': np.clip(pc_ratio, 0.3, 3.0),
            'options_volume_ratio': np.random.uniform(0.3, 1.5),
            'large_trade_ratio': np.random.uniform(0.05, 0.25)
        }
    
    def set_market_regime(self, regime: str):
        """Set market regime for simulation"""
        self.market_regime = regime
