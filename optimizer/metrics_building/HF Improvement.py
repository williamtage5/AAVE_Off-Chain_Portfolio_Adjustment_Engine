import pandas as pd
import os
import json
from decimal import Decimal, getcontext

# --- 0. 设置精度 ---
# 设置Decimal的精度以进行精确的金融计算
getcontext().prec = 50

# --- 1. 定义文件路径 ---
BASE_PATH = r"F:\Learning_journal_at_CUHK\FTEC5520_Appl Blockchain & Cryptocur\Simulation"
MERGED_DATA_FILE = os.path.join(BASE_PATH, "optimizer", "data", "merged_hf_and_prices.csv")
OUTLINE_FILE = os.path.join(BASE_PATH, "optimizer", "data", "outline.json")


def load_static_data(json_path: str) -> tuple[dict, Decimal]:
    """
    加载 outline.json 文件.
    解析所有金额, LTs, 和 GAF 为 Decimal.
    """
    print(f"Loading static portfolio data from: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. 提取 GAF
        GAF = Decimal(data['health_factor_anchors']['Adjustment_Factor_GAF'])
        print(f"  Loaded Global Adjustment Factor (GAF): {GAF}")

        # 2. 提取持仓
        portfolio = {'collaterals': {}, 'debts': {}}
        
        for item in data['numerator_collaterals']:
            symbol = item['symbol']
            amount = Decimal(item['amount_raw']) / (Decimal('10') ** item['decimals'])
            lt = Decimal(item['liquidation_threshold'])
            portfolio['collaterals'][symbol] = {'amount': amount, 'lt': lt}
            print(f"  Collateral: {symbol}, Amount: {amount}, LT: {lt}")

        for item in data['denominator_debts']:
            symbol = item['symbol']
            amount = Decimal(item['amount_raw']) / (Decimal('10') ** item['decimals'])
            portfolio['debts'][symbol] = {'amount': amount}
            print(f"  Debt: {symbol}, Amount: {amount}")

        return portfolio, GAF

    except FileNotFoundError:
        print(f"  [Error] Static data file not found: {json_path}")
        exit()
    except Exception as e:
        print(f"  [Error] Failed to parse static data: {e}")
        exit()

def calculate_hf(current_amounts: dict, static_portfolio: dict, prices_t: dict, gaf: Decimal) -> Decimal:
    """
    根据当前资产和价格计算调整后的HF。
    
    :param current_amounts: {'WETH': Decimal, 'cbBTC': Decimal, 'USDC': Decimal}
    :param static_portfolio: 从 load_static_data() 加载的完整持仓 (用于 LTs)
    :param prices_t: {'WETH_price_usd': Decimal, 'cbBTC_price_usd': Decimal, 'USDC_price_usd': Decimal}
    :param gaf: 全局调整因子
    :return: 调整后的 HF (Decimal)
    """
    
    # --- 计算分子 (抵押品) ---
    numerator = Decimal('0')
    try:
        numerator += current_amounts['WETH'] * \
                     prices_t['WETH_price_usd'] * \
                     static_portfolio['collaterals']['WETH']['lt']
        
        numerator += current_amounts['cbBTC'] * \
                     prices_t['cbBTC_price_usd'] * \
                     static_portfolio['collaterals']['cbBTC']['lt']
                     
    except KeyError as e:
        print(f"  [Error] Missing collateral info in calculate_hf: {e}")
        return Decimal('0')

    # --- 计算分母 (债务) ---
    denominator = Decimal('0')
    try:
        denominator = current_amounts['USDC'] * prices_t['USDC_price_usd']
    except KeyError as e:
        print(f"  [Error] Missing debt info in calculate_hf: {e}")
        return Decimal('0')

    # --- 计算HF ---
    if denominator == Decimal('0'):
        return Decimal('inf') # HF 无穷大
        
    hf_unadjusted = numerator / denominator
    
    # (!!!) 应用调整因子
    hf_adjusted = hf_unadjusted * gaf
    
    return hf_adjusted


def main():
    """
    主模拟函数
    """
    
    # --- 1. 加载所有数据 ---
    static_portfolio, GAF = load_static_data(OUTLINE_FILE)
    
    print(f"Loading merged time-series data from: {MERGED_DATA_FILE}")
    try:
        df = pd.read_csv(MERGED_DATA_FILE)
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    except FileNotFoundError:
        print(f"  [Error] Merged data file not found: {MERGED_DATA_FILE}")
        return
    except Exception as e:
        print(f"  [Error] Failed to load merged data: {e}")
        return
        
    print("All data loaded successfully.\n")

    # --- 2. 定义模拟场景 ---
    print("--- SIMULATION SETUP ---")
    
    # a. 找到时间点 't' (HF 最低点)
    target_row = df.loc[df['HF_adjusted'].idxmin()]
    target_time = target_row['datetime_utc']
    HF_before = Decimal(target_row['HF_adjusted'])
    
    print(f"Simulating at time t = {target_time} (Lowest HF)")
    print(f"Baseline HF Before (Adjusted): {HF_before:.8f}")

    # b. 获取 't' 时刻的价格
    prices_t = {
        'WETH_price_usd': Decimal(target_row['WETH_price_usd']),
        'cbBTC_price_usd': Decimal(target_row['cbBTC_price_usd']),
        'USDC_price_usd': Decimal(target_row['USDC_price_usd'])
    }
    
    # c. 获取调仓前的资产数量
    amounts_before = {
        'WETH': static_portfolio['collaterals']['WETH']['amount'],
        'cbBTC': static_portfolio['collaterals']['cbBTC']['amount'],
        'USDC': static_portfolio['debts']['USDC']['amount']
    }
    
    # d. 定义示例操作
    action_add_weth = Decimal('0.1')
    action_repay_usdc = Decimal('200.0')
    
    print(f"Defined Action 1: Add {action_add_weth} WETH")
    print(f"Defined Action 2: Repay {action_repay_usdc} USDC")


    # --- 3. 执行计算 (三个场景) ---
    print("\n--- RUNNING SIMULATIONS (Metric 1: HF Improvement) ---")

    # --- SCENARIO 1: 增加抵押品 (Add Collateral) ---
    amounts_scen_1 = amounts_before.copy()
    amounts_scen_1['WETH'] += action_add_weth
    
    HF_after_scen_1 = calculate_hf(amounts_scen_1, static_portfolio, prices_t, GAF)
    improvement_1 = HF_after_scen_1 - HF_before

    # --- SCENARIO 2: 偿还债务 (Repay Debt) ---
    amounts_scen_2 = amounts_before.copy()
    amounts_scen_2['USDC'] -= action_repay_usdc
    
    HF_after_scen_2 = calculate_hf(amounts_scen_2, static_portfolio, prices_t, GAF)
    improvement_2 = HF_after_scen_2 - HF_before

    # --- SCENARIO 3: 融合 (Fusion: Add Collateral + Repay Debt) ---
    amounts_scen_3 = amounts_before.copy()
    amounts_scen_3['WETH'] += action_add_weth
    amounts_scen_3['USDC'] -= action_repay_usdc
    
    HF_after_scen_3 = calculate_hf(amounts_scen_3, static_portfolio, prices_t, GAF)
    improvement_3 = HF_after_scen_3 - HF_before
    
    
    # --- 4. 打印结果 ---
    print(f"\n--- Simulation Results at t = {target_time} ---")
    print(f"Baseline HF Before: {HF_before: .8f}\n")

    print(f"  SCENARIO 1: Add {action_add_weth} WETH")
    print(f"    HF After:       {HF_after_scen_1: .8f}")
    print(f"    HF Improvement: {improvement_1: .8f}\n")

    print(f"  SCENARIO 2: Repay {action_repay_usdc} USDC")
    print(f"    HF After:       {HF_after_scen_2: .8f}")
    print(f"    HF Improvement: {improvement_2: .8f}\n")

    print(f"  SCENARIO 3: Add {action_add_weth} WETH AND Repay {action_repay_usdc} USDC")
    print(f"    HF After:       {HF_after_scen_3: .8f}")
    print(f"    HF Improvement: {improvement_3: .8f}\n")
    
    print("--- END OF SIMULATION ---")


# --- 脚本入口 ---
if __name__ == "__main__":
    main()