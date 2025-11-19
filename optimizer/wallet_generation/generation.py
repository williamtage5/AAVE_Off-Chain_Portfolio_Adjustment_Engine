import os
import json
from decimal import Decimal, getcontext

# --- 0. 设置精度 ---
# 设置Decimal的精度以进行精确的金融计算
getcontext().prec = 50

# --- 1. 定义文件路径 ---
BASE_PATH = r"F:\Learning_journal_at_CUHK\FTEC5520_Appl Blockchain & Cryptocur\Simulation\optimizer\wallet_generation"
DATA_DIR = os.path.join(BASE_PATH, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "wallets.json")

# 确保输出目录存在
os.makedirs(DATA_DIR, exist_ok=True)

# --- 2. 静态仓位和价格数据 (基于您的 outline.json 和提示) ---

# 仓位 (来自 outline.json)
POSITION = {
    "WETH": {
        "amount_raw": "630624808395278241",
        "decimals": 18
    },
    "cbBTC": {
        "amount_raw": "58445",
        "decimals": 8
    },
    "USDC": {
        "amount_raw": "2040843772",
        "decimals": 6
    }
}

# 价格 (来自 2025-10-30 12:00:00+00:00 的数据)
PRICES_USD = {
    "WETH": "3885.469893419255186018",
    "cbBTC": "109864.189461083398782648",
    "USDC": "0.999865197543199802"
}

def main():
    """
    主执行函数：计算仓位价值，生成钱包情景，并保存为JSON。
    """
    print("--- 钱包情景生成脚本 ---")

    # --- 3. 将所有数据转换为 Decimal ---
    print("正在计算基础仓位价值 (USD)...")
    
    # 资产数量
    q_weth = Decimal(POSITION['WETH']['amount_raw']) / (Decimal('10') ** POSITION['WETH']['decimals'])
    q_cbtc = Decimal(POSITION['cbBTC']['amount_raw']) / (Decimal('10') ** POSITION['cbBTC']['decimals'])
    q_usdc = Decimal(POSITION['USDC']['amount_raw']) / (Decimal('10') ** POSITION['USDC']['decimals'])
    
    # 价格
    p_weth = Decimal(PRICES_USD['WETH'])
    p_cbtc = Decimal(PRICES_USD['cbBTC'])
    p_usdc = Decimal(PRICES_USD['USDC'])

    # --- 4. 计算总价值 (USD) ---
    
    # 总债务价值 (USD)
    total_debt_usd = q_usdc * p_usdc
    
    # 总抵押品价值 (USD)
    value_weth_usd = q_weth * p_weth
    value_cbtc_usd = q_cbtc * p_cbtc
    total_collateral_usd = value_weth_usd + value_cbtc_usd
    
    print(f"  总债务价值 (Total Debt):   {total_debt_usd:,.2f} USD")
    print(f"  总抵押品价值 (Total Collateral): {total_collateral_usd:,.2f} USD")

    # --- 5. 定义钱包情景 ---
    print("\n正在生成钱包情景 (可用的预备金)...")
    
    wallets = []

    # 情景 1: 部分 (25% 债务) - 无法还清债务
    val_1 = total_debt_usd * Decimal('0.25')
    wallets.append({
        "wallet_id": "Partial_25pct_Debt",
        "description": "Partial (Low-Liquidity). Cannot cover full debt.",
        "available_usd": f"{val_1:.18f}" # 保存为字符串以保持精度
    })
    
    # 情景 2: 平衡 (75% 债务) - 无法还清债务
    val_2 = total_debt_usd * Decimal('0.75')
    wallets.append({
        "wallet_id": "Balanced_75pct_Debt",
        "description": "Balanced (Medium-Liquidity). Cannot cover full debt.",
        "available_usd": f"{val_2:.18f}"
    })
    
    # 情景 3: 高流动性 (110% 债务) - 可以还清债务
    val_3 = total_debt_usd * Decimal('1.10')
    wallets.append({
        "wallet_id": "FullDebt_110pct_Debt",
        "description": "High-Liquidity. Can cover full debt with surplus.",
        "available_usd": f"{val_3:.18f}"
    })

    # 情景 4: 超高流动性 (100% 抵押品) - 远超债务
    val_4 = total_collateral_usd * Decimal('1.00')
    wallets.append({
        "wallet_id": "FullCollateral_100pct_Coll",
        "description": "Very High-Liquidity. Based on 100% of collateral value.",
        "available_usd": f"{val_4:.18f}"
    })
    
    print(f"已生成 {len(wallets)} 个钱包情景。")
    for wallet in wallets:
        print(f"  - {wallet['wallet_id']}: {Decimal(wallet['available_usd']):,.2f} USD")

    # --- 6. 保存到 JSON 文件 ---
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(wallets, f, indent=4)
        print(f"\n--- 成功 ---")
        print(f"已将钱包数据保存到:\n{OUTPUT_FILE}")
    except Exception as e:
        print(f"\n[Error] 保存 JSON 文件失败: {e}")

# --- 脚本入口 ---
if __name__ == "__main__":
    main()