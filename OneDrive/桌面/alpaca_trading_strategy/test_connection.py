import os
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# 加载环境变量
load_dotenv()

# 获取 API 密钥
api_key = os.getenv('ALPACA_API_KEY')
secret_key = os.getenv('ALPACA_SECRET_KEY')

print("🔍 检查 API 密钥...")
print(f"API Key 前5位: {api_key[:5] if api_key else '未找到'}")
print(f"Secret Key 前5位: {secret_key[:5] if secret_key else '未找到'}")

if not api_key or not secret_key:
    print("\n❌ 错误：未找到 API 密钥！")
    print("请检查 .env 文件是否正确")
else:
    print("\n✅ API 密钥已加载")
    print("\n📡 正在连接 Alpaca...")
    
    try:
        # 连接到 Alpaca
        api = tradeapi.REST(
            api_key, 
            secret_key, 
            'https://paper-api.alpaca.markets',
            api_version='v2'
        )
        
        # 获取账户信息
        account = api.get_account()
        
        print("\n✅ 连接成功！")
        print("\n📊 账户信息：")
        print(f"  💰 现金余额: ${float(account.cash):,.2f}")
        print(f"  💼 总资产: ${float(account.portfolio_value):,.2f}")
        print(f"  📈 买入能力: ${float(account.buying_power):,.2f}")
        print(f"  🏦 账户状态: {account.status}")
        print(f"  🎯 账户类型: Paper Trading (模拟)")
        
        # 检查市场状态
        clock = api.get_clock()
        print(f"\n⏰ 市场状态: {'开市' if clock.is_open else '休市'}")
        print(f"  下次开市: {clock.next_open}")
        print(f"  下次休市: {clock.next_close}")
        
    except Exception as e:
        print(f"\n❌ 连接失败: {e}")
        print("\n可能的原因：")
        print("1. API 密钥不正确")
        print("2. 网络连接问题")
        print("3. Alpaca 服务暂时不可用")
