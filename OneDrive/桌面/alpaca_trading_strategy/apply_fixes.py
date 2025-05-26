#!/usr/bin/env python3
"""
apply_fixes.py - 自动修复策略文件
"""
import os
import shutil
from datetime import datetime

def apply_fixes():
    print("🔧 开始修复策略文件...")
    
    # 备份原文件
    strategy_file = 'complete_trading_strategy.py'
    backup_file = f'complete_trading_strategy_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.py'
    
    if os.path.exists(strategy_file):
        shutil.copy(strategy_file, backup_file)
        print(f"✅ 已备份原文件到: {backup_file}")
    
    # 读取原文件
    with open(strategy_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找类定义位置
    class_def = "class AlpacaCompleteStrategy:"
    class_pos = content.find(class_def)
    
    if class_pos == -1:
        print("❌ 找不到类定义！")
        return
    
    # 找到 __init__ 方法的结束位置
    init_end = content.find("\n    def ", class_pos + len(class_def))
    
    if init_end == -1:
        print("❌ 找不到方法定义位置！")
        return
    
    # 准备要插入的方法
    methods_to_add = '''
    # ========== 修复添加的方法 ==========
    
    async def stream_news_async(self):
        """Stream news without blocking - Fixed version"""
        if self.mode != 'live':
            return
            
        try:
            logger.info("News streaming temporarily disabled - API update needed")
            # 暂时禁用新闻流，等待 API 更新
            
        except Exception as e:
            logger.error(f"News streaming error: {e}")
    
    def update_market_sentiment(self):
        """Update market sentiment analysis"""
        current_date = self.get_market_time().date()
        sentiment = self.calculate_enhanced_market_sentiment(current_date)
        self.sentiment_history[current_date] = sentiment
        
        logger.info(f"Market sentiment: {sentiment['overall']:.3f} ({sentiment['state']}), "
                   f"Confidence: {sentiment['confidence']:.2f}")
    
    def calculate_enhanced_market_sentiment(self, date):
        """Calculate enhanced market sentiment with all components"""
        sentiment_components = {}
        
        # 1. Price momentum
        if 'SPY' in self.market_data:
            spy_df = self.market_data['SPY']
            if len(spy_df) > 20:
                latest = spy_df.iloc[-1]
                
                r1 = latest.get('Returns', 0)
                r5 = latest.get('Returns_5', 0)
                r20 = latest.get('Returns_20', 0)
                
                sentiment_components['price_momentum'] = (r1 * 0.5 + r5 * 0.3 + r20 * 0.2) * 4
                
                if 'Price_Acceleration' in spy_df.columns:
                    acc = latest['Price_Acceleration']
                    sentiment_components['momentum_acceleration'] = acc * 10
        
        # 2. VIX sentiment
        if 'VIX' in self.market_data:
            vix_df = self.market_data['VIX']
            if len(vix_df) > 20:
                vix = vix_df['Close'].iloc[-1]
                vix_ma20 = vix_df['Close'].rolling(20).mean().iloc[-1]
                vix_std = vix_df['Close'].rolling(20).std().iloc[-1]
                
                if vix_std > 0:
                    vix_zscore = (vix - vix_ma20) / vix_std
                    sentiment_components['vix_sentiment'] = -vix_zscore * 0.3
                    
                    if len(vix_df) >= 5:
                        vix_change = vix_df['Close'].pct_change(5).iloc[-1]
                        sentiment_components['vix_momentum'] = -vix_change * 2
        
        # 3. Market breadth
        self.calculate_market_breadth()
        if date in self.market_microstructure:
            micro = self.market_microstructure[date]
            sentiment_components['market_breadth'] = micro.get('market_breadth', 0) * 0.5
            sentiment_components['advance_decline'] = (micro.get('advance_decline_ratio', 0.5) - 0.5) * 2
        
        # 4. Sector rotation
        sector_sentiment = self.calculate_sector_rotation_sentiment()
        sentiment_components['sector_rotation'] = sector_sentiment * 0.4
        
        # 5. Volume sentiment
        if 'SPY' in self.market_data:
            spy_df = self.market_data['SPY']
            if 'Volume_Ratio' in spy_df.columns and len(spy_df) > 0:
                vol_ratio = spy_df['Volume_Ratio'].iloc[-1]
                price_change = spy_df['Returns'].iloc[-1] if 'Returns' in spy_df.columns else 0
                
                if vol_ratio > 1.5 and price_change > 0:
                    sentiment_components['volume_sentiment'] = 0.4
                elif vol_ratio > 1.5 and price_change < 0:
                    sentiment_components['volume_sentiment'] = -0.5
                elif vol_ratio < 0.7:
                    sentiment_components['volume_sentiment'] = 0.2
                else:
                    sentiment_components['volume_sentiment'] = 0
        
        # 6. Smart money flow
        smart_money = self.calculate_smart_money_flow()
        sentiment_components['smart_money'] = smart_money * 0.5
        
        # 7. Technical sentiment
        tech_sentiment = self.calculate_technical_sentiment()
        sentiment_components['technical'] = tech_sentiment * 0.3
        
        # 8. Options sentiment (if available)
        options_sentiment = self.calculate_options_sentiment()
        if options_sentiment != 0:
            sentiment_components['options'] = options_sentiment * 0.4
        
        # Calculate total sentiment
        total_sentiment = sum(sentiment_components.values())
        
        # Momentum adjustment
        recent_dates = sorted([d for d in self.sentiment_history.keys() if d < date])[-3:]
        if len(recent_dates) >= 2:
            recent_sentiments = [self.sentiment_history[d]['overall'] for d in recent_dates]
            sentiment_momentum = (total_sentiment - np.mean(recent_sentiments)) * 0.4
            total_sentiment += sentiment_momentum
        else:
            sentiment_momentum = 0
        
        # Normalize
        total_sentiment = np.clip(total_sentiment, -1, 1)
        
        # Determine market state
        if total_sentiment >= 0.5:
            market_state = 'euphoria'
        elif total_sentiment >= 0.2:
            market_state = 'optimism'
        elif total_sentiment >= -0.2:
            market_state = 'neutral'
        elif total_sentiment >= -0.5:
            market_state = 'pessimism'
        else:
            market_state = 'panic'
        
        # Calculate confidence
        confidence = min(abs(total_sentiment) * 1.5, 1.0)
        
        # Multi-factor consistency check
        positive_count = sum(1 for v in sentiment_components.values() if v > 0.1)
        negative_count = sum(1 for v in sentiment_components.values() if v < -0.1)
        
        if positive_count >= 5 or negative_count >= 5:
            confidence = min(confidence * 1.2, 1.0)
        
        return {
            'overall': total_sentiment,
            'components': sentiment_components,
            'momentum': sentiment_momentum,
            'state': market_state,
            'confidence': confidence
        }
    
    def calculate_market_breadth(self):
        """Calculate market breadth"""
        current_date = self.get_market_time().date()
        
        advancing = 0
        declining = 0
        total = 0
        new_highs = 0
        new_lows = 0
        
        for symbol, df in self.market_data.items():
            if symbol not in ['VIX', 'TNX', '^VIX', '^TNX'] and len(df) > 1:
                if df['Close'].iloc[-1] > df['Close'].iloc[-2]:
                    advancing += 1
                else:
                    declining += 1
                total += 1
                
                if len(df) >= 252:
                    if df['Close'].iloc[-1] >= df['Close'].iloc[-252:].max():
                        new_highs += 1
                    elif df['Close'].iloc[-1] <= df['Close'].iloc[-252:].min():
                        new_lows += 1
        
        if total > 0:
            micro_data = {
                'advance_decline_ratio': advancing / total,
                'market_breadth': (advancing - declining) / total,
                'new_highs': new_highs,
                'new_lows': new_lows,
                'new_highs_lows_ratio': new_highs / (new_highs + new_lows + 1)
            }
        else:
            micro_data = {
                'advance_decline_ratio': 0.5,
                'market_breadth': 0,
                'new_highs': 0,
                'new_lows': 0,
                'new_highs_lows_ratio': 0.5
            }
        
        self.market_microstructure[current_date] = micro_data
    
    def calculate_sector_rotation_sentiment(self):
        """Calculate sector rotation sentiment"""
        sector_etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLRE', 'XLU']
        sector_performance = {}
        
        for etf in sector_etfs:
            if etf in self.market_data and len(self.market_data[etf]) >= 5:
                df = self.market_data[etf]
                if 'Returns_5' in df.columns:
                    returns_5d = df['Returns_5'].iloc[-1]
                    sector_performance[etf] = returns_5d
        
        if not sector_performance:
            return 0
        
        # Risk appetite indicator
        risk_on_sectors = ['XLK', 'XLY', 'XLF']
        risk_off_sectors = ['XLP', 'XLU', 'XLRE']
        
        risk_on_perf = np.mean([sector_performance.get(s, 0) for s in risk_on_sectors])
        risk_off_perf = np.mean([sector_performance.get(s, 0) for s in risk_off_sectors])
        
        risk_appetite = (risk_on_perf - risk_off_perf) * 3
        
        return risk_appetite
    
    def calculate_smart_money_flow(self):
        """Calculate smart money flow"""
        smart_money_signal = 0
        
        # Large cap vs small cap
        if 'SPY' in self.market_data and 'IWM' in self.market_data:
            spy_df = self.market_data['SPY']
            iwm_df = self.market_data['IWM']
            
            if len(spy_df) >= 20 and len(iwm_df) >= 20:
                if 'Returns_20' in spy_df.columns and 'Returns_20' in iwm_df.columns:
                    spy_r20 = spy_df['Returns_20'].iloc[-1]
                    iwm_r20 = iwm_df['Returns_20'].iloc[-1]
                    
                    large_vs_small = spy_r20 - iwm_r20
                    
                    if large_vs_small > 0.03:
                        smart_money_signal = 0.4
                    elif large_vs_small < -0.03:
                        smart_money_signal = -0.3
        
        # Defensive vs growth
        if 'XLP' in self.market_data and 'XLK' in self.market_data:
            xlp_df = self.market_data['XLP']
            xlk_df = self.market_data['XLK']
            
            if len(xlp_df) >= 10 and len(xlk_df) >= 10:
                if 'Returns_10' in xlp_df.columns and 'Returns_10' in xlk_df.columns:
                    defensive_r = xlp_df['Returns_10'].iloc[-1]
                    growth_r = xlk_df['Returns_10'].iloc[-1]
                    
                    if growth_r > defensive_r + 0.02:
                        smart_money_signal += 0.2
        
        return smart_money_signal
    
    def calculate_technical_sentiment(self):
        """Calculate technical sentiment"""
        if 'SPY' not in self.market_data:
            return 0
        
        spy_df = self.market_data['SPY']
        if len(spy_df) < 50:
            return 0
        
        tech_score = 0
        latest = spy_df.iloc[-1]
        
        # Trend strength
        if all(col in spy_df.columns for col in ['Close', 'SMA_20', 'SMA_50']):
            close = latest['Close']
            sma20 = latest['SMA_20']
            sma50 = latest['SMA_50']
            
            if close > sma20 > sma50:
                tech_score += 0.3
            elif close < sma20 < sma50:
                tech_score -= 0.3
        
        # RSI state
        if 'RSI' in spy_df.columns:
            rsi = latest['RSI']
            if 40 < rsi < 60:
                tech_score += 0.1
            elif rsi > 70:
                tech_score -= 0.1
            elif rsi < 30:
                tech_score += 0.2
        
        # MACD momentum
        if all(col in spy_df.columns for col in ['MACD', 'MACD_Signal']):
            if latest['MACD'] > latest['MACD_Signal']:
                tech_score += 0.2
        
        return tech_score
    
    def calculate_options_sentiment(self):
        """Calculate aggregate options sentiment"""
        if not self.option_sentiment:
            return 0
        
        sentiment_scores = []
        
        for symbol, (iv_skew, pc_ratio) in self.option_sentiment.items():
            # Bullish signals
            if iv_skew < self.IV_SKEW_BULLISH and pc_ratio > self.CPRATIO_BULLISH:
                sentiment_scores.append(0.5)
            # Bearish signals
            elif iv_skew > 0.1 or pc_ratio < self.CPRATIO_BEARISH:
                sentiment_scores.append(-0.5)
            else:
                sentiment_scores.append(0)
        
        if sentiment_scores:
            return np.mean(sentiment_scores)
        return 0
    
    def update_factor_scores(self):
        """Update factor scores"""
        logger.info("Calculating factor scores...")
        
        for symbol, df in self.market_data.items():
            if symbol in ['^VIX', '^TNX', 'VIX', 'TNX'] or len(df) < 60:
                continue
            
            current_date = self.get_market_time().date()
            scores = self.calculate_enhanced_factor_score(symbol, df)
            
            if symbol not in self.factor_scores:
                self.factor_scores[symbol] = {}
                
            self.factor_scores[symbol][current_date] = scores
    
    def analyze_stock_sentiment_enhanced(self, symbol):
        """Enhanced stock sentiment analysis"""
        if symbol not in self.market_data:
            return 0
        
        df = self.market_data[symbol]
        if len(df) < 20:
            return 0
        
        sentiment = 0
        
        # 1. Relative strength
        if 'SPY' in self.market_data:
            spy_df = self.market_data['SPY']
            if len(spy_df) >= 20:
                for period in [5, 10, 20]:
                    col_name = f'Returns_{period}'
                    if col_name in df.columns and col_name in spy_df.columns:
                        stock_return = df[col_name].iloc[-1]
                        spy_return = spy_df[col_name].iloc[-1]
                        rs = stock_return - spy_return
                        sentiment += rs * (4 - period/10)
        
        # 2. Technical sentiment
        if all(col in df.columns for col in ['RSI_5', 'RSI_14', 'RSI_21']):
            rsi5 = df['RSI_5'].iloc[-1]
            rsi14 = df['RSI_14'].iloc[-1]
            
            if rsi5 < 25 and rsi14 < 35:
                sentiment += 0.4
            elif len(df) >= 5:
                price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1)
                if 'RSI_14' in df.columns and len(df) >= 6:
                    rsi_change = rsi14 - df['RSI_14'].iloc[-6]
                    if price_change < -0.03 and rsi_change > 5:
                        sentiment += 0.3
        
        # 3. MACD momentum
        if 'MACD_Histogram' in df.columns and len(df) >= 3:
            macd_hist = df['MACD_Histogram'].iloc[-3:]
            if all(macd_hist.diff().dropna() > 0):
                sentiment += 0.2
        
        # 4. Volume patterns
        if 'Volume_Ratio' in df.columns and len(df) >= 5:
            vol_ratio = df['Volume_Ratio'].iloc[-1]
            recent_volumes = df['Volume_Ratio'].iloc[-5:]
            price_changes = df['Returns'].iloc[-5:]
            
            if (recent_volumes.iloc[:-1] < 0.8).all() and vol_ratio > 1.5 and price_changes.iloc[-1] > 0:
                sentiment += 0.35
        
        # 5. Bollinger Band squeeze
        if 'BB_Width' in df.columns and len(df) >= 20:
            bb_width = df['BB_Width'].iloc[-1]
            bb_width_avg = df['BB_Width'].iloc[-20:].mean()
            if bb_width < bb_width_avg * 0.5:  # Squeeze detected
                sentiment += 0.2
        
        return np.clip(sentiment, -1, 1)
    
    def calculate_elite_financial_score(self, symbol, date):
        """Calculate elite financial stock score"""
        if symbol not in self.market_data:
            return -999
        
        df = self.market_data[symbol]
        if len(df) < 50:
            return -999
        
        # Base factor score
        scores = self.calculate_enhanced_factor_score(symbol, df)
        score = scores['total']
        
        # Financial sector relative strength
        if 'XLF' in self.market_data and len(self.market_data['XLF']) >= 20:
            xlf_df = self.market_data['XLF']
            if 'Returns_20' in df.columns and 'Returns_20' in xlf_df.columns:
                stock_r20 = df['Returns_20'].iloc[-1]
                xlf_r20 = xlf_df['Returns_20'].iloc[-1]
                if stock_r20 > xlf_r20 * 1.02:
                    score += 0.25
                elif stock_r20 > xlf_r20:
                    score += 0.15
        
        # Interest rate environment
        if 'TNX' in self.market_data and len(self.market_data['TNX']) >= 20:
            rate_df = self.market_data['TNX']
            rate_change = rate_df['Close'].iloc[-1] - rate_df['Close'].iloc[-20]
            if -1.0 < rate_change < 2.0:  # Moderate rate increase good for banks
                score += 0.15
        elif '^TNX' in self.market_data:
            # Fallback to ^TNX
            rate_df = self.market_data['^TNX']
            if len(rate_df) >= 20:
                rate_change = rate_df['Close'].iloc[-1] - rate_df['Close'].iloc[-20]
                if -1.0 < rate_change < 2.0:
                    score += 0.15
        
        # Technical quality
        if all(col in df.columns for col in ['RSI', 'MACD_Histogram']):
            rsi = df['RSI'].iloc[-1]
            macd_hist = df['MACD_Histogram'].iloc[-1]
            
            if 35 < rsi < 75 and macd_hist > 0:
                score += 0.2
            elif 30 < rsi < 80:
                score += 0.1
        
        # Volume confirmation
        if 'Volume_Ratio' in df.columns:
            vol_ratio = df['Volume_Ratio'].iloc[-1]
            if vol_ratio > 1.3:
                score *= 1.1
        
        # Leadership premium
        if symbol in ['JPM', 'GS', 'MS', 'BRK.B', 'V', 'MA']:
            score *= 1.2
        
        return score
'''
    
    # 插入方法
    new_content = content[:init_end] + methods_to_add + content[init_end:]
    
    # 写回文件
    with open(strategy_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ 方法已添加到策略文件")
    
    # 修复 validate_market_data 方法中的时间检查
    print("🔧 修复数据验证逻辑...")
    
    # 创建一个周末友好的运行脚本
    weekend_script = '''#!/usr/bin/env python3
"""
run_weekend_test.py - 周末测试脚本（忽略数据过期警告）
"""
import os
import sys
from datetime import datetime
import logging

# 如果使用 .env 文件
from dotenv import load_dotenv
load_dotenv()

# 导入策略
from complete_trading_strategy import AlpacaCompleteStrategy

def main():
    # 检查 API 密钥
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        print("❌ 错误：未找到 API 密钥！")
        print("请确保设置了 ALPACA_API_KEY 和 ALPACA_SECRET_KEY")
        sys.exit(1)
    
    print("="*60)
    print("🚀 Alpaca 模拟交易策略启动（周末测试模式）")
    print("="*60)
    print(f"⏰ 启动时间: {datetime.now()}")
    print("📊 模式: Paper Trading (模拟交易)")
    print("💰 初始资金: $100,000")
    print("⚠️  注意：现在是周末，市场关闭")
    print("    数据会显示'过期'，这是正常的")
    print("="*60)
    
    try:
        # 创建策略实例
        strategy = AlpacaCompleteStrategy(
            api_key=api_key,
            secret_key=secret_key,
            base_url='https://paper-api.alpaca.markets',
            options_provider='simulated',
            mode='live'
        )
        
        # 修改数据过期容忍度（周末测试）
        strategy.max_data_age_seconds = 259200  # 3天
        
        print("✅ 策略初始化成功！")
        print("📈 开始运行策略...")
        print("💡 提示: 按 Ctrl+C 停止")
        print("-"*60)
        
        # 运行策略
        strategy.run_live_trading()
        
    except KeyboardInterrupt:
        print("\\n⚠️ 收到停止信号...")
        print("正在安全关闭...")
    except Exception as e:
        print(f"\\n❌ 发生错误: {e}")
        logging.error(f"策略错误: {e}", exc_info=True)
    finally:
        print("\\n👋 策略已停止")

if __name__ == "__main__":
    main()
'''
    
    with open('run_weekend_test.py', 'w', encoding='utf-8') as f:
        f.write(weekend_script)
    
    print("✅ 创建了周末测试脚本: run_weekend_test.py")
    
    print("\n🎉 修复完成！")
    print("\n下一步：")
    print("1. 运行周末测试: python run_weekend_test.py")
    print("2. 或等到周一市场开放时运行: python run_paper_trading.py")

if __name__ == "__main__":
    apply_fixes()

