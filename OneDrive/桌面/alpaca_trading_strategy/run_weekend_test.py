#!/usr/bin/env python3
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
        print("\n⚠️ 收到停止信号...")
        print("正在安全关闭...")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        logging.error(f"策略错误: {e}", exc_info=True)
    finally:
        print("\n👋 策略已停止")

if __name__ == "__main__":
    main()
