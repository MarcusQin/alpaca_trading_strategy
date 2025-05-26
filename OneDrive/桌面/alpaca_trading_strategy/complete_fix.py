#!/usr/bin/env python3
"""
complete_fix.py - 完整修复所有缺失的方法
"""
import os
import re

def add_all_missing_methods():
    """添加所有缺失的方法到主策略类"""
    
    print("🔧 开始完整修复...")
    
    # 读取主策略文件
    with open('complete_trading_strategy.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份
    with open('complete_trading_strategy_backup.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✅ 已备份原文件")
    
    # 需要添加的所有方法
    all_methods = '''
    # ========== 缺失的方法完整补充 ==========
    
    def get_top_k_candidates(self, scores_dict, k=5, min_score=0.0):
        """Get top k candidates with score above threshold"""
        # Filter by minimum score
        filtered = {sym: score for sym, score in scores_dict.items() 
                    if score > min_score and not np.isnan(score)}
        
        # Sort and get top k
        sorted_stocks = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        return sorted_stocks[:k]
    
    def get_batch_quotes(self, symbols):
        """Get quotes for multiple symbols efficiently"""
        if self.mode != 'live':
            # Return simulated quotes for backtesting
            quotes = {}
            for symbol in symbols:
                if symbol in self.market_data and len(self.market_data[symbol]) > 0:
                    last_price = self.market_data[symbol]['Close'].iloc[-1]
                    quotes[symbol] = type('obj', (object,), {
                        'ask_price': last_price * 1.0001,
                        'bid_price': last_price * 0.9999,
                        'last_price': last_price
                    })
            return quotes
        
        try:
            # Filter out invalid symbols
            valid_symbols = [s for s in symbols if s and s.strip()]
            
            # Batch API call
            quotes = self.api.get_latest_quotes(valid_symbols)
            return quotes
        except Exception as e:
            logger.error(f"Failed to get batch quotes: {e}")
            # Fallback to individual quotes
            quotes = {}
            for symbol in valid_symbols:
                try:
                    quote = self.api.get_latest_quote(symbol)
                    quotes[symbol] = quote
                except Exception as e:
                    logger.warning(f"Failed to get quote for {symbol}: {e}")
                    continue
            return quotes
    
    def calculate_leveraged_position_size(self, base_size, symbol, market_state, strategy_type='core'):
        """Calculate position size with improved logic"""
        # Start with leveraged base size
        leveraged_size = base_size * self.current_leverage
        
        # Market state adjustment
        state_factors = {
            'euphoria': 0.6,     # Reduce in euphoria
            'optimism': 1.0,     # Neutral in optimism
            'neutral': 1.0,      # Neutral
            'pessimism': 1.2,    # Increase in pessimism (contrarian)
            'panic': 1.4         # Increase more in panic
        }
        factor = state_factors.get(market_state.get('state', 'neutral'), 1.0)
        
        # Option sentiment adjustment
        if symbol in self.option_sentiment:
            iv_skew, cpr = self.option_sentiment[symbol]
            
            # Bullish signals
            if iv_skew < self.IV_SKEW_BULLISH and cpr > self.CPRATIO_BULLISH:
                factor *= 1.2
                logger.debug(f"{symbol}: Bullish options (IV skew: {iv_skew:.3f}, P/C: {cpr:.2f})")
            # Bearish signals
            elif iv_skew > 0.1 or cpr < self.CPRATIO_BEARISH:
                factor *= 0.8
                logger.debug(f"{symbol}: Bearish options (IV skew: {iv_skew:.3f}, P/C: {cpr:.2f})")
        
        # Strategy specific adjustment
        strategy_multipliers = {
            'core': 1.0,
            'financial': 1.1,
            'sentiment': 0.9,
            'passive': 1.0
        }
        factor *= strategy_multipliers.get(strategy_type, 1.0)
        
        # Volatility adjustment
        if symbol in self.market_data and 'Volatility' in self.market_data[symbol].columns:
            vol = self.market_data[symbol]['Volatility'].iloc[-1]
            if vol > 0:
                # Reduce size for high volatility stocks
                vol_factor = min(1.0, 0.20 / vol)  # Target 20% volatility
                factor *= vol_factor
        
        # Circuit breaker check
        triggered, reason = self.check_circuit_breakers()
        if triggered:
            factor *= 0.5  # Reduce size if circuit breakers are close
        
        final_size = leveraged_size * factor
        
        # Ensure we don't exceed margin
        if self.mode == 'live' and self.account:
            max_position_size = float(self.account.buying_power) * 0.20
            final_size = min(final_size, max_position_size)
        
        return final_size
    
    def reset_daily_counters(self):
        """Reset daily buy counters with proper date handling"""
        current_date = self.get_market_time().date()
        
        if self.last_reset_date != current_date:
            self.daily_buys = {'core': 0, 'financial': 0, 'sentiment': 0}
            self.last_reset_date = current_date
            
            # Reset circuit breaker for new day
            self.circuit_breaker_triggered = False
            self.intraday_high_water_mark = None
            
            # Check if pause should be lifted
            if self.paused_until and current_date >= self.paused_until:
                self.paused_until = None
                logger.info("Trading resumed - new trading day")
                
            logger.info("Daily counters reset")
    
    def morning_preparation(self):
        """Morning preparation tasks"""
        logger.info("Performing morning preparation...")
        
        # Reset circuit breaker for new day
        self.circuit_breaker_triggered = False
        self.intraday_high_water_mark = None
        
        # Update all data
        self.update_market_data()
        if self.mode == 'live':
            self.update_account_info()
        self.update_portfolio_volatility()
        self.update_correlation_matrix()
        
        # Print market state
        current_date = self.get_market_time().date()
        if current_date in self.sentiment_history:
            sentiment = self.sentiment_history[current_date]
            logger.info(f"Today's market sentiment: {sentiment['overall']:.3f} ({sentiment['state']})")
        
        logger.info(f"Current leverage: {self.current_leverage:.2f}, VIX 5D MA: {self.vix_5d_ma:.1f}")
        logger.info(f"Risk ON: {self.risk_on}")
        logger.info("Morning preparation complete")
    
    def execute_strategies(self):
        """Execute all trading strategies"""
        logger.info("Executing trading strategies...")
        
        # Update data
        if self.mode == 'live':
            self.update_account_info()
        
        current_date = self.get_market_time().date()
        
        if self.mode == 'live' and self.account:
            total_value = float(self.account.portfolio_value)
        else:
            # For backtesting, calculate from positions
            cash = getattr(self, 'backtest_cash', self.initial_capital)
            positions_value = sum(pos.get('value', 0) for pos in self.positions.values())
            total_value = cash + positions_value
        
        # Execute each strategy
        self.execute_passive_investment(current_date, total_value)
        self.execute_core_active(current_date, total_value)
        self.execute_elite_financial(current_date, total_value)
        self.execute_sentiment_driven_enhanced(current_date, total_value)
        
        logger.info("Strategy execution complete")
    
    def monitor_positions(self):
        """Monitor existing positions"""
        logger.info("Monitoring positions...")
        
        if self.mode == 'live':
            self.update_account_info()
        
        current_date = self.get_market_time().date()
        
        # Check circuit breakers first
        triggered, reason = self.check_circuit_breakers()
        if triggered:
            logger.warning(f"Circuit breaker during monitoring: {reason}")
        
        # Check all positions
        for symbol, position in list(self.positions.items()):
            # Skip ETFs
            if symbol in self.passive_etfs:
                continue
            
            # Determine strategy type and check if should sell
            if symbol in self.elite_financial_stocks:
                should_sell, reason = self.should_sell_financial_position_live(symbol)
                if should_sell:
                    self.execute_sell_live(symbol, 'financial', reason)
            elif symbol in self.momentum_stocks:
                should_sell, reason = self.should_sell_core_position_live(symbol)
                if should_sell:
                    self.execute_sell_live(symbol, 'core', reason)
            else:
                should_sell, reason = self.should_sell_sentiment_position_live(symbol)
                if should_sell:
                    self.execute_sell_live(symbol, 'sentiment', reason)
        
        logger.info("Position monitoring complete")
    
    def end_of_day_tasks(self):
        """End of day tasks"""
        logger.info("Performing end of day tasks...")
        
        # Update account info
        if self.mode == 'live':
            self.update_account_info()
        
        # Record final portfolio value
        self.record_portfolio_value()
        
        # Generate daily report
        self.generate_daily_report()
        
        # Save data
        self.save_data()
        
        logger.info("End of day tasks complete")
    
    def record_portfolio_value(self):
        """Record portfolio value with all metrics"""
        if self.mode == 'live':
            self.update_account_info()
            portfolio_value = float(self.account.portfolio_value)
            cash = float(self.account.cash)
        else:
            # For backtesting
            cash = getattr(self, 'backtest_cash', self.initial_capital)
            positions_value = sum(pos.get('value', 0) for pos in self.positions.values())
            portfolio_value = cash + positions_value
        
        # Calculate component values
        passive_value = sum(
            float(pos.market_value) if hasattr(pos, 'market_value') else pos.get('value', 0)
            for symbol, pos in self.positions.items() 
            if symbol in self.passive_etfs
        )
        
        financial_value = sum(
            float(pos.market_value) if hasattr(pos, 'market_value') else pos.get('value', 0)
            for symbol, pos in self.positions.items() 
            if symbol in self.elite_financial_stocks
        )
        
        core_value = sum(
            float(pos.market_value) if hasattr(pos, 'market_value') else pos.get('value', 0)
            for symbol, pos in self.positions.items() 
            if symbol in self.momentum_stocks and symbol not in self.elite_financial_stocks
        )
        
        sentiment_value = portfolio_value - cash - passive_value - financial_value - core_value
        
        current_date = self.get_market_time().date()
        market_sentiment = self.sentiment_history.get(current_date, {}).get('overall', 0)
        market_state = self.sentiment_history.get(current_date, {}).get('state', 'neutral')
        
        self.portfolio_history.append({
            'timestamp': self.get_market_time(),
            'date': current_date,
            'total_value': portfolio_value,
            'cash': cash,
            'passive_value': passive_value,
            'core_value': core_value,
            'financial_value': financial_value,
            'sentiment_value': sentiment_value,
            'core_pct': core_value / portfolio_value * 100 if portfolio_value > 0 else 0,
            'financial_pct': financial_value / portfolio_value * 100 if portfolio_value > 0 else 0,
            'sentiment_pct': sentiment_value / portfolio_value * 100 if portfolio_value > 0 else 0,
            'market_sentiment': market_sentiment,
            'market_state': market_state,
            'total_positions': len(self.positions),
            'leverage': self.current_leverage,
            'realized_vol': self.realized_vol,
            'vix_5d_ma': self.vix_5d_ma,
            'risk_on': self.risk_on
        })
        
        # Update portfolio peak
        if portfolio_value > self.portfolio_peak:
            self.portfolio_peak = portfolio_value
    
    def generate_daily_report(self):
        """Generate comprehensive daily report"""
        current_date = self.get_market_time().date()
        
        if self.mode == 'live' and self.account:
            portfolio_value = float(self.account.portfolio_value)
            cash = float(self.account.cash)
            buying_power = float(self.account.buying_power)
        else:
            cash = getattr(self, 'backtest_cash', self.initial_capital)
            positions_value = sum(pos.get('value', 0) for pos in self.positions.values())
            portfolio_value = cash + positions_value
            buying_power = cash * 2
        
        # Calculate daily return
        if len(self.portfolio_history) >= 2:
            prev_value = self.portfolio_history[-2]['total_value']
            daily_return = (portfolio_value / prev_value - 1) * 100
        else:
            daily_return = 0
        
        # Calculate total return
        total_return = (portfolio_value / self.initial_capital - 1) * 100
        
        # Calculate drawdown
        drawdown = (portfolio_value - self.portfolio_peak) / self.portfolio_peak * 100
        
        logger.info(f"\\n{'='*60}")
        logger.info(f"Enhanced Strategy Daily Report - {current_date}")
        logger.info(f"{'='*60}")
        logger.info(f"Portfolio Value: ${portfolio_value:,.2f}")
        logger.info(f"Daily Return: {daily_return:+.2f}%")
        logger.info(f"Total Return: {total_return:+.2f}%")
        logger.info(f"Current Drawdown: {drawdown:.2f}%")
        logger.info(f"Cash: ${cash:,.2f}")
        logger.info(f"Buying Power: ${buying_power:,.2f}")
        logger.info(f"Positions: {len(self.positions)}")
        logger.info(f"Leverage: {self.current_leverage:.2f}x")
        logger.info(f"Realized Vol: {self.realized_vol:.1%}")
        logger.info(f"VIX 5D MA: {self.vix_5d_ma:.1f}")
        logger.info(f"Risk Status: {'ON' if self.risk_on else 'OFF'}")
        
        # Strategy allocation
        if self.portfolio_history:
            latest = self.portfolio_history[-1]
            logger.info(f"\\nStrategy Allocation:")
            logger.info(f"  Passive: {latest['passive_value']/portfolio_value*100:.1f}%")
            logger.info(f"  Core Active: {latest['core_pct']:.1f}%")
            logger.info(f"  Financial: {latest['financial_pct']:.1f}%")
            logger.info(f"  Sentiment: {latest['sentiment_pct']:.1f}%")
        
        # Market state
        if current_date in self.sentiment_history:
            sentiment = self.sentiment_history[current_date]
            logger.info(f"\\nMarket Sentiment: {sentiment['overall']:.3f} ({sentiment['state']})")
        
        # Factor weights
        logger.info(f"\\nFactor Weights (IC-EWMA adjusted):")
        for factor, weight in self.factor_weights.items():
            logger.info(f"  {factor}: {weight:.2f}")
        
        # Today's trades
        today_trades = [t for t in self.trades_history 
                       if self.ensure_timezone_aware(t['date']).date() == current_date]
        if today_trades:
            logger.info(f"\\nToday's Trades ({len(today_trades)} total):")
            for trade in today_trades[:10]:  # Show first 10
                logger.info(f"  {trade['action']}: {trade['symbol']} x {trade['shares']} shares "
                           f"(strategy: {trade['strategy']})")
        
        # Daily buy counts
        logger.info(f"\\nDaily Buy Counts:")
        logger.info(f"  Core: {self.daily_buys['core']}/{self.MAX_CORE_BUYS}")
        logger.info(f"  Financial: {self.daily_buys['financial']}/{self.MAX_FIN_BUYS}")
        logger.info(f"  Sentiment: {self.daily_buys['sentiment']}/{self.MAX_SENT_BUYS}")
        
        # Performance metrics
        if len(self.portfolio_history) >= 20:
            var_95 = self.calculate_portfolio_var(0.95)
            logger.info(f"\\nRisk Metrics:")
            logger.info(f"  VaR (95%): {var_95:.2%}")
            logger.info(f"  Consecutive Losses: {self.consecutive_losses}")
        
        logger.info(f"{'='*60}\\n")
    
    def save_data(self):
        """Save data to files"""
        try:
            timestamp = self.get_market_time().strftime("%Y%m%d_%H%M%S")
            
            # Save trade history
            if self.trades_history:
                trades_df = pd.DataFrame(self.trades_history)
                trades_df.to_csv(f'complete_trades_{timestamp}.csv', index=False)
            
            # Save portfolio history
            if self.portfolio_history:
                portfolio_df = pd.DataFrame(self.portfolio_history)
                portfolio_df.to_csv(f'complete_portfolio_{timestamp}.csv', index=False)
            
            # Save slippage log
            if self.slippage_log:
                slippage_df = pd.DataFrame(self.slippage_log)
                slippage_df.to_csv('slippage_log.csv', index=False)
            
            # Save factor weights history
            weights_data = {
                'timestamp': self.get_market_time().isoformat() if hasattr(self.get_market_time(), 'isoformat') else str(self.get_market_time()),
                'weights': self.factor_weights,
                'ic_history': {}
            }
            
            # Check if factor_ic_calculator exists
            if hasattr(self, 'factor_ic_calculator') and hasattr(self.factor_ic_calculator, 'ic_history'):
                weights_data['ic_history'] = {k: list(v) for k, v in self.factor_ic_calculator.ic_history.items()}
            
            with open(f'factor_weights_{timestamp}.json', 'w') as f:
                json.dump(weights_data, f, indent=2, default=str)
            
            logger.info("Data saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save data: {e}")
    
    def handle_circuit_breaker(self, reason):
        """Handle circuit breaker trigger"""
        logger.error(f"EMERGENCY: Circuit breaker triggered - {reason}")
        
        if self.mode == 'live':
            # Cancel all pending orders
            for order_id, order_info in list(self.pending_orders.items()):
                try:
                    self.api.cancel_order(order_id)
                    logger.info(f"Cancelled pending order: {order_id}")
                except:
                    pass
        
        # Clear high-risk positions
        for symbol in list(self.positions.keys()):
            if symbol not in self.passive_etfs:  # Keep passive ETFs
                position = self.positions[symbol]
                
                entry_price = float(position.avg_entry_price) if hasattr(position, 'avg_entry_price') else 100
                current_price = float(position.current_price) if hasattr(position, 'current_price') else 100
                
                if symbol in self.market_data and len(self.market_data[symbol]) > 0:
                    current_price = self.market_data[symbol]['Close'].iloc[-1]
                
                pnl_pct = (current_price / entry_price - 1)
                
                # Sell losing positions and high-volatility positions
                if pnl_pct < -0.03 or symbol in [s for stocks in self.sentiment_stocks.values() for s in stocks]:
                    self.execute_sell_live(symbol, 'circuit_breaker', reason)
        
        # Pause until next day
        self.paused_until = self.get_market_time().date() + timedelta(days=1)
        logger.info(f"Trading paused until {self.paused_until}")
    
    def track_order_for_slippage(self, order, ref_price):
        """Track order for slippage calculation"""
        if self.mode != 'live':
            return
            
        def check_fill():
            try:
                filled_order = self.api.get_order(order.id)
                if filled_order.status == 'filled':
                    fill_price = float(filled_order.filled_avg_price)
                    slippage = (fill_price - ref_price) / ref_price
                    
                    self.slippage_log.append({
                        'timestamp': self.get_market_time(),
                        'symbol': order.symbol,
                        'side': order.side,
                        'ref_price': ref_price,
                        'fill_price': fill_price,
                        'slippage_bps': slippage * 10000
                    })
                    
                    # Save to CSV periodically
                    if len(self.slippage_log) % 10 == 0:
                        pd.DataFrame(self.slippage_log).to_csv('slippage_log.csv', index=False)
                        
            except Exception as e:
                logger.error(f"Failed to track slippage: {e}")
        
        # Check after delay
        timer = threading.Timer(5.0, check_fill)
        timer.daemon = True
        timer.start()
    
    def stop(self):
        """Stop the strategy gracefully"""
        self.is_running = False
        logger.info("Strategy stop signal sent")
'''
    
    # 找到类定义的最后一个方法
    # 查找最后一个方法定义
    last_method_pattern = r'(\n    def \w+\(self[^:]*\):[^}]+?)(?=\n(?:class|\Z))'
    matches = list(re.finditer(last_method_pattern, content, re.DOTALL))
    
    if matches:
        last_match = matches[-1]
        insert_pos = last_match.end()
    else:
        # 如果找不到，在类的末尾添加
        class_pattern = r'class AlpacaCompleteStrategy[^:]*:.*'
        class_match = re.search(class_pattern, content, re.DOTALL)
        if class_match:
            insert_pos = len(content) - 1
        else:
            print("❌ 无法找到合适的插入位置")
            return
    
    # 插入所有方法
    new_content = content[:insert_pos] + all_methods + content[insert_pos:]
    
    # 写回文件
    with open('complete_trading_strategy.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ 所有方法已添加完成")
    
    # 还需要添加执行策略的方法
    print("🔧 添加执行策略方法...")
    
    # 从其他文件读取执行方法
    exec_methods = '''
    # ========== 策略执行方法 ==========
    
    def execute_core_active(self, date, total_value):
        """Execute core active strategy with all fixes"""
        if not self.risk_on:
            logger.info("Risk OFF - skipping core active strategy")
            return
        
        # Check if trading is paused
        if self.paused_until and date <= self.paused_until:
            logger.info(f"Trading paused until {self.paused_until}")
            return
        
        # Check circuit breakers
        triggered, reason = self.check_circuit_breakers()
        if triggered:
            logger.warning(f"Circuit breaker active: {reason}")
            return
        
        # Reset daily counters if needed
        self.reset_daily_counters()
        
        # Check daily buy limit
        if self.daily_buys['core'] >= self.MAX_CORE_BUYS:
            return
        
        # Check sells first
        for symbol in list(self.positions.keys()):
            if symbol in self.momentum_stocks:
                should_sell, reason = self.should_sell_core_position_live(symbol)
                if should_sell:
                    self.execute_sell_live(symbol, 'core', reason)
        
        # Calculate scores for all candidates
        scores = {}
        for symbol in self.momentum_stocks:
            if symbol not in self.positions and symbol in self.market_data:
                df = self.market_data[symbol]
                
                # Validate data
                is_valid, validation_msg = self.validate_market_data(symbol, df)
                if not is_valid:
                    logger.debug(f"Skipping {symbol}: {validation_msg}")
                    continue
                
                if len(df) >= 60:
                    score_data = self.calculate_enhanced_factor_score(symbol, df)
                    scores[symbol] = score_data['total']
        
        # Get top candidates
        candidates = self.get_top_k_candidates(scores, k=5, min_score=self.CORE_THR)
        
        if candidates and len([s for s in self.positions if s in self.momentum_stocks]) < self.max_core_positions:
            # Get batch quotes for efficiency
            candidate_symbols = [sym for sym, _ in candidates]
            quotes = self.get_batch_quotes(candidate_symbols)
            
            # Calculate leveraged allocation
            active_allocation = self.base_active_allocation
            core_capital = total_value * self.core_active_pct * active_allocation
            base_position_size = core_capital / self.max_core_positions
            
            # Execute buys (respecting daily limit)
            buys_remaining = self.MAX_CORE_BUYS - self.daily_buys['core']
            
            for symbol, score in candidates[:buys_remaining]:
                if len(self.positions) >= self.max_total_positions:
                    break
                
                if symbol not in quotes:
                    logger.warning(f"No quote available for {symbol}")
                    continue
                
                # Check correlation
                can_buy, correlation_msg = self.check_position_correlation(symbol)
                if not can_buy:
                    logger.info(f"Skipping {symbol}: {correlation_msg}")
                    continue
                
                # Get options sentiment
                iv_skew, pc_ratio = self.get_option_metrics(symbol)
                
                market_state = self.sentiment_history.get(date, {})
                position_size = self.calculate_leveraged_position_size(
                    base_position_size, symbol, market_state, 'core'
                )
                
                # Calculate shares
                quote = quotes[symbol]
                price = float(quote.ask_price) if hasattr(quote, 'ask_price') else float(quote['ask_price'])
                shares = int(position_size / price)
                
                # Check buying power
                required_capital = shares * price
                buying_power = float(self.account.buying_power) if self.mode == 'live' else total_value * 0.5
                
                if shares > 0 and required_capital <= buying_power * 0.95:
                    if self.use_limit_orders:
                        order = self.execute_limit_order(symbol, 'buy', shares, 'core')
                    else:
                        order = self.execute_order_with_timeout(
                            symbol, 'buy', shares, 'market', None, 'core'
                        )
                    
                    if order:
                        self.daily_buys['core'] += 1
                        self.trades_history.append({
                            'date': self.get_market_time(),
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': price,
                            'strategy': 'core',
                            'score': score,
                            'leverage': self.current_leverage
                        })
                else:
                    logger.warning(f"Insufficient buying power for {symbol}: "
                                 f"need ${required_capital:.2f}, have ${buying_power:.2f}")
    
    def execute_elite_financial(self, date, total_value):
        """Execute elite financial strategy with all fixes"""
        if not self.risk_on:
            logger.info("Risk OFF - skipping elite financial strategy")
            return
        
        # Check if trading is paused
        if self.paused_until and date <= self.paused_until:
            return
        
        # Check circuit breakers
        triggered, reason = self.check_circuit_breakers()
        if triggered:
            return
        
        # Reset daily counters if needed
        self.reset_daily_counters()
        
        # Check daily buy limit
        if self.daily_buys['financial'] >= self.MAX_FIN_BUYS:
            return
        
        # Check sells
        for symbol in list(self.positions.keys()):
            if symbol in self.elite_financial_stocks:
                should_sell, reason = self.should_sell_financial_position_live(symbol)
                if should_sell:
                    self.execute_sell_live(symbol, 'financial', reason)
        
        # Calculate scores
        scores = {}
        for symbol in self.elite_financial_stocks:
            if symbol not in self.positions and symbol in self.market_data:
                # Validate data
                is_valid, validation_msg = self.validate_market_data(symbol, self.market_data[symbol])
                if not is_valid:
                    continue
                    
                score = self.calculate_elite_financial_score(symbol, date)
                if score > self.FIN_THR:
                    scores[symbol] = score
        
        # Get top candidates
        candidates = self.get_top_k_candidates(scores, k=3, min_score=self.FIN_THR)
        
        if candidates and len([s for s in self.positions if s in self.elite_financial_stocks]) < self.max_financial_positions:
            # Get batch quotes
            candidate_symbols = [sym for sym, _ in candidates]
            quotes = self.get_batch_quotes(candidate_symbols)
            
            # Calculate allocation
            active_allocation = self.base_active_allocation
            financial_capital = total_value * self.elite_financial_pct * active_allocation
            base_position_size = financial_capital / self.max_financial_positions
            
            # Execute buys
            buys_remaining = self.MAX_FIN_BUYS - self.daily_buys['financial']
            
            for symbol, score in candidates[:buys_remaining]:
                if len(self.positions) >= self.max_total_positions:
                    break
                
                if symbol not in quotes:
                    continue
                
                # Check correlation
                can_buy, correlation_msg = self.check_position_correlation(symbol)
                if not can_buy:
                    logger.info(f"Skipping {symbol}: {correlation_msg}")
                    continue
                
                market_state = self.sentiment_history.get(date, {})
                position_size = self.calculate_leveraged_position_size(
                    base_position_size, symbol, market_state, 'financial'
                )
                
                # Get option sentiment
                iv_skew, cpr = self.get_option_metrics(symbol)
                
                # Skip if bearish option sentiment
                if iv_skew > 0.1 or cpr < self.CPRATIO_BEARISH:
                    logger.info(f"Skipping {symbol} due to bearish options (IV skew: {iv_skew:.2f}, P/C: {cpr:.2f})")
                    continue
                
                # Calculate shares
                quote = quotes[symbol]
                price = float(quote.ask_price) if hasattr(quote, 'ask_price') else float(quote['ask_price'])
                shares = int(position_size / price)
                
                # Check buying power
                required_capital = shares * price
                buying_power = float(self.account.buying_power) if self.mode == 'live' else total_value * 0.5
                
                if shares > 0 and required_capital <= buying_power * 0.95:
                    if self.use_limit_orders:
                        order = self.execute_limit_order(symbol, 'buy', shares, 'financial')
                    else:
                        order = self.execute_order_with_timeout(
                            symbol, 'buy', shares, 'market', None, 'financial'
                        )
                    
                    if order:
                        self.daily_buys['financial'] += 1
                        self.trades_history.append({
                            'date': self.get_market_time(),
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': price,
                            'strategy': 'financial',
                            'score': score,
                            'leverage': self.current_leverage,
                            'iv_skew': iv_skew,
                            'put_call_ratio': cpr
                        })
    
    def execute_sentiment_driven_enhanced(self, date, total_value):
        """Execute sentiment driven strategy with news integration"""
        if not self.risk_on:
            logger.info("Risk OFF - skipping sentiment strategy")
            return
        
        # Check if trading is paused
        if self.paused_until and date <= self.paused_until:
            return
        
        # Check circuit breakers
        triggered, reason = self.check_circuit_breakers()
        if triggered:
            return
        
        # Reset daily counters if needed
        self.reset_daily_counters()
        
        # Check daily buy limit
        if self.daily_buys['sentiment'] >= self.MAX_SENT_BUYS:
            return
        
        # Check sells
        for symbol in list(self.positions.keys()):
            if symbol not in self.momentum_stocks and symbol not in self.elite_financial_stocks:
                should_sell, reason = self.should_sell_sentiment_position_live(symbol)
                if should_sell:
                    self.execute_sell_live(symbol, 'sentiment', reason)
        
        # Find opportunities with news momentum
        all_scores = {}
        
        for category, stocks in self.sentiment_stocks.items():
            for symbol in stocks:
                if symbol not in self.positions and symbol in self.market_data:
                    # Validate data
                    is_valid, validation_msg = self.validate_market_data(symbol, self.market_data[symbol])
                    if not is_valid:
                        continue
                    
                    # Get technical sentiment
                    technical_sentiment = self.analyze_stock_sentiment_enhanced(symbol)
                    
                    # Get news momentum
                    news_momentum = self.calculate_news_sentiment_momentum(symbol)
                    
                    # Combined score
                    combined_score = technical_sentiment * 0.6 + news_momentum * 0.4
                    
                    # Entry conditions
                    if (abs(combined_score) > 0.3 or
                        abs(news_momentum) > 0.5 or
                        (symbol in self.news_sentiment_history and 
                         len(self.news_sentiment_history[symbol]) >= 3)):
                        
                        all_scores[symbol] = combined_score
        
        # Get top candidates
        candidates = sorted(all_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        if candidates:
            # Get batch quotes
            candidate_symbols = [sym for sym, _ in candidates]
            quotes = self.get_batch_quotes(candidate_symbols)
            
            # Calculate allocation
            active_allocation = self.base_active_allocation
            sentiment_capital = total_value * self.sentiment_driven_pct * active_allocation
            base_position_size = sentiment_capital / self.max_sentiment_positions
            
            # Execute buys
            buys_remaining = self.MAX_SENT_BUYS - self.daily_buys['sentiment']
            
            for symbol, score in candidates[:buys_remaining]:
                if len(self.positions) >= self.max_total_positions:
                    break
                
                if symbol not in quotes:
                    continue
                
                # Check correlation
                can_buy, correlation_msg = self.check_position_correlation(symbol)
                if not can_buy:
                    logger.info(f"Skipping {symbol}: {correlation_msg}")
                    continue
                
                market_state = self.sentiment_history.get(date, {})
                position_size = self.calculate_leveraged_position_size(
                    base_position_size, symbol, market_state, 'sentiment'
                )
                
                # Calculate shares
                quote = quotes[symbol]
                price = float(quote.ask_price) if hasattr(quote, 'ask_price') else float(quote['ask_price'])
                shares = int(position_size / price)
                
                # Check buying power
                required_capital = shares * price
                buying_power = float(self.account.buying_power) if self.mode == 'live' else total_value * 0.5
                
                if shares > 0 and required_capital <= buying_power * 0.95:
                    if self.use_limit_orders:
                        order = self.execute_limit_order(symbol, 'buy', shares, 'sentiment')
                    else:
                        order = self.execute_order_with_timeout(
                            symbol, 'buy', shares, 'market', None, 'sentiment'
                        )
                    
                    if order:
                        self.daily_buys['sentiment'] += 1
                        
                        # Get recent news for logging
                        recent_news = ""
                        if symbol in self.news_sentiment_history and self.news_sentiment_history[symbol]:
                            recent_news = self.news_sentiment_history[symbol][-1]['headline'][:50]
                        
                        self.trades_history.append({
                            'date': self.get_market_time(),
                            'symbol': symbol,
                            'action': 'BUY',
                            'shares': shares,
                            'price': price,
                            'strategy': 'sentiment',
                            'score': score,
                            'leverage': self.current_leverage,
                            'recent_news': recent_news
                        })
    
    def execute_passive_investment(self, date, total_value):
        """Execute passive investment strategy without leverage"""
        market_state = self.sentiment_history.get(date, {'state': 'neutral'})
        
        # Dynamic passive allocation WITHOUT leverage
        if market_state['state'] == 'panic':
            passive_allocation = 0.30
        elif market_state['state'] == 'euphoria':
            passive_allocation = 0.10
        else:
            passive_allocation = self.base_passive_allocation
        
        # No leverage on passive investments
        passive_capital = total_value * passive_allocation
        
        # Rebalancing logic (quarterly)
        need_rebalance = False
        if not self.positions or not any(s in self.passive_etfs for s in self.positions):
            need_rebalance = True
        elif self.last_rebalance_date:
            days_since = (date - self.last_rebalance_date).days
            if days_since >= 90:
                need_rebalance = True
        
        if need_rebalance:
            # Sell all ETFs
            for symbol in list(self.positions.keys()):
                if symbol in self.passive_etfs:
                    self.execute_sell_live(symbol, 'passive', 'Rebalance')
            
            # Buy ETFs
            etf_symbols = list(self.passive_etfs.keys())
            quotes = self.get_batch_quotes(etf_symbols)
            
            for etf, weight in self.passive_etfs.items():
                if etf not in quotes:
                    continue
                    
                allocation = passive_capital * weight
                
                try:
                    quote = quotes[etf]
                    price = float(quote.ask_price) if hasattr(quote, 'ask_price') else float(quote['ask_price'])
                    shares = int(allocation / price)
                    
                    # Check buying power
                    required_capital = shares * price
                    buying_power = float(self.account.buying_power) if self.mode == 'live' else total_value * 0.5
                    
                    if shares > 0 and required_capital <= buying_power * 0.95:
                        if self.use_limit_orders:
                            order = self.execute_limit_order(etf, 'buy', shares, 'passive')
                        else:
                            order = self.execute_order_with_timeout(
                                etf, 'buy', shares, 'market', None, 'passive'
                            )
                        
                        if order:
                            self.trades_history.append({
                                'date': self.get_market_time(),
                                'symbol': etf,
                                'action': 'BUY',
                                'shares': shares,
                                'price': price,
                                'strategy': 'passive'
                            })
                            
                except Exception as e:
                    logger.error(f"Failed to buy {etf}: {e}")
            
            self.last_rebalance_date = date
    
    def should_sell_core_position_live(self, symbol):
        """Check if should sell core position with proper stop loss handling"""
        if symbol not in self.positions or symbol not in self.market_data:
            return False, ""
        
        position = self.positions[symbol]
        df = self.market_data[symbol]
        
        entry_price = float(position.avg_entry_price) if hasattr(position, 'avg_entry_price') else 100
        current_price = float(position.current_price) if hasattr(position, 'current_price') else df['Close'].iloc[-1]
        profit_pct = (current_price / entry_price - 1)
        
        # Stop loss - CRITICAL
        if profit_pct < -self.position_stop_loss['core']:
            return True, f"Stop Loss ({profit_pct*100:.1f}%)"
        
        # Trailing stop for profits
        if profit_pct > 0.15:  # 15% profit
            trailing_stop = entry_price * (1 + profit_pct - self.position_stop_loss['trailing_stop'])
            if current_price < trailing_stop:
                return True, f"Trailing Stop ({profit_pct*100:.1f}%)"
        
        # Factor turned negative
        current_date = self.get_market_time().date()
        if symbol in self.factor_scores and current_date in self.factor_scores[symbol]:
            if self.factor_scores[symbol][current_date]['total'] < -0.1:
                return True, "Factor Negative"
        
        # Trend break
        if len(df) >= 20 and 'SMA_20' in df.columns:
            if current_price < df['SMA_20'].iloc[-1] * 0.97:
                return True, "Trend Break"
        
        # Profit taking
        if profit_pct > self.position_stop_loss['profit_stop']:
            return True, f"Profit Taking (+{profit_pct*100:.1f}%)"
        
        # Time-based exit
        if hasattr(position, 'purchase_date'):
            purchase_date = self.ensure_timezone_aware(position.purchase_date)
            days_held = (self.get_market_time() - purchase_date).days
            if days_held > self.position_params['max_holding_days']:
                return True, f"Max holding period ({days_held} days)"
        
        return False, ""
    
    def should_sell_financial_position_live(self, symbol):
        """Check if should sell financial position"""
        if symbol not in self.positions or symbol not in self.market_data:
            return False, ""
        
        position = self.positions[symbol]
        
        entry_price = float(position.avg_entry_price) if hasattr(position, 'avg_entry_price') else 100
        current_price = float(position.current_price) if hasattr(position, 'current_price') else self.market_data[symbol]['Close'].iloc[-1]
        profit_pct = (current_price / entry_price - 1)
        
        # Stop loss
        if profit_pct < -self.position_stop_loss['financial']:
            return True, f"Stop Loss ({profit_pct*100:.1f}%)"
        
        # Trailing stop for profits
        if profit_pct > 0.12:  # 12% profit
            trailing_stop = entry_price * (1 + profit_pct - self.position_stop_loss['trailing_stop'])
            if current_price < trailing_stop:
                return True, f"Trailing Stop ({profit_pct*100:.1f}%)"
        
        # Sector weakness check
        if 'XLF' in self.market_data and len(self.market_data[symbol]) >= 5:
            stock_r5 = (current_price / self.market_data[symbol]['Close'].iloc[-6] - 1)
            xlf_r5 = (self.market_data['XLF']['Close'].iloc[-1] / 
                     self.market_data['XLF']['Close'].iloc[-6] - 1)
            
            if stock_r5 < xlf_r5 * 0.7 and profit_pct < 0.05:
                return True, "Sector Weakness"
        
        # Profit taking
        if profit_pct > self.position_stop_loss['profit_stop']:
            return True, f"Profit Taking (+{profit_pct*100:.1f}%)"
        
        return False, ""
    
    def should_sell_sentiment_position_live(self, symbol):
        """Check if should sell sentiment position"""
        if symbol not in self.positions:
            return False, ""
        
        position = self.positions[symbol]
        
        entry_price = float(position.avg_entry_price) if hasattr(position, 'avg_entry_price') else 100
        current_price = float(position.current_price) if hasattr(position, 'current_price') else 100
        
        if symbol in self.market_data and len(self.market_data[symbol]) > 0:
            current_price = self.market_data[symbol]['Close'].iloc[-1]
        
        profit_pct = (current_price / entry_price - 1)
        
        # Stop loss
        if profit_pct < -self.position_stop_loss['sentiment']:
            return True, f"Stop Loss ({profit_pct*100:.1f}%)"
        
        # Quick profit for sentiment trades
        if profit_pct > 0.20:
            return True, f"Quick Profit (+{profit_pct*100:.1f}%)"
        
        # News sentiment reversal
        if symbol in self.news_sentiment_history and len(self.news_sentiment_history[symbol]) >= 3:
            recent_scores = [item['score'] for item in list(self.news_sentiment_history[symbol])[-3:]]
            if np.mean(recent_scores) < -0.3:
                return True, "News Sentiment Reversal"
        
        # Time decay for sentiment trades
        if hasattr(position, 'purchase_date'):
            purchase_date = self.ensure_timezone_aware(position.purchase_date)
            days_held = (self.get_market_time() - purchase_date).days
            if days_held > 5 and profit_pct < 0.05:
                return True, f"Time Decay ({days_held} days)"
        
        return False, ""
    
    def execute_sell_live(self, symbol, strategy_type, reason):
        """Execute sell order with proper handling"""
        try:
            if symbol not in self.positions:
                return
            
            position = self.positions[symbol]
            shares = int(position.qty) if hasattr(position, 'qty') else position.get('shares', 0)
            
            if shares > 0:
                # Determine order type based on reason
                if 'Stop Loss' in reason or 'circuit_breaker' in strategy_type:
                    # Use market order for urgent exits
                    order = self.execute_order_with_timeout(
                        symbol, 'sell', shares, 'market', None, 'stop_loss'
                    )
                else:
                    # Use limit order for normal exits
                    if self.use_limit_orders:
                        order = self.execute_limit_order(symbol, 'sell', shares, strategy_type)
                    else:
                        order = self.execute_order_with_timeout(
                            symbol, 'sell', shares, 'market', None, strategy_type
                        )
                
                if order:
                    # Calculate P&L
                    entry_price = float(position.avg_entry_price) if hasattr(position, 'avg_entry_price') else 100
                    current_price = float(position.current_price) if hasattr(position, 'current_price') else 100
                    
                    if symbol in self.market_data and len(self.market_data[symbol]) > 0:
                        current_price = self.market_data[symbol]['Close'].iloc[-1]
                    
                    pnl = (current_price - entry_price) * shares
                    pnl_pct = (current_price / entry_price - 1) * 100
                    
                    logger.info(f"{strategy_type} SELL: {symbol} @ ${current_price:.2f} "
                               f"({pnl_pct:+.1f}%, {reason})")
                    
                    # Record trade
                    self.trades_history.append({
                        'date': self.get_market_time(),
                        'symbol': symbol,
                        'action': f'SELL({reason})',
                        'shares': shares,
                        'price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'strategy': strategy_type
                    })
                    
                    # Update consecutive losses
                    if pnl < 0:
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 0
                    
                    # Update portfolio peak if needed
                    if self.mode == 'live':
                        self.update_account_info()
                        current_value = float(self.account.portfolio_value)
                        if current_value > self.portfolio_peak:
                            self.portfolio_peak = current_value
                    
        except Exception as e:
            logger.error(f"Failed to sell {symbol}: {e}")
'''
    
    # 再次插入执行方法
    with open('complete_trading_strategy.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 在文件末尾之前插入
    insert_pos = content.rfind('\n\n')
    if insert_pos == -1:
        insert_pos = len(content) - 1
    
    new_content = content[:insert_pos] + exec_methods + content[insert_pos:]
    
    # 写回文件
    with open('complete_trading_strategy.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ 执行方法已添加")
    
    # 导入需要的模块
    print("🔧 检查导入语句...")
    
    # 确保所有必要的导入都在文件开头
    with open('complete_trading_strategy.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否缺少导入
    imports_needed = [
        "from datetime import timedelta",
        "import threading",
        "import json"
    ]
    
    imports_to_add = []
    for imp in imports_needed:
        if imp not in content:
            imports_to_add.append(imp)
    
    if imports_to_add:
        # 找到导入部分的结尾
        import_end = content.find('\nwarnings.filterwarnings')
        if import_end != -1:
            import_section = '\n'.join(imports_to_add) + '\n'
            content = content[:import_end] + import_section + content[import_end:]
            
            with open('complete_trading_strategy.py', 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ 添加了缺失的导入: {imports_to_add}")
    
    print("\n🎉 完整修复完成！")
    print("现在运行: python run_weekend_test.py")

if __name__ == "__main__":
    add_all_missing_methods()

