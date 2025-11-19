import pandas as pd
import os
import json
import numpy as np # (!!!) 导入 numpy
from decimal import Decimal, getcontext
from datetime import datetime # (!!!) 导入 datetime

# --- 0. 设置精度 ---
getcontext().prec = 50

# --- 1. 定义文件路径 ---
BASE_PATH = r"F:\Learning_journal_at_CUHK\FTEC5520_Appl Blockchain & Cryptocur\Simulation"
OPTIMIZER_PATH = os.path.join(BASE_PATH, "optimizer")
OPERATION_POINTS_DATA = os.path.join(BASE_PATH, "operation_points", "data")

# 输入文件
TRIGGER_FILE = os.path.join(OPERATION_POINTS_DATA, "risk_escalation_points.csv")
WALLETS_FILE = os.path.join(OPTIMIZER_PATH, "wallet_generation", "data", "wallets.json")
OUTLINE_FILE = os.path.join(OPTIMIZER_PATH, "data", "outline.json")
MERGED_DATA_FILE = os.path.join(OPTIMIZER_PATH, "data", "merged_hf_and_prices.csv")
RISK_CLASSIFICATION_FILE = os.path.join(OPERATION_POINTS_DATA, "risk_classification_per_timestamp.csv")

# 输出目录和文件
OUTPUT_DIR = os.path.join(OPTIMIZER_PATH, "Strategy_A", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "strategy_A_simulation_results.json")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_static_data(json_path: str) -> tuple[dict, Decimal]:
    """
    加载 outline.json 文件.
    解析所有金额, LTs, 和 GAF 为 Decimal.
    """
    print(f"Loading static portfolio data from: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        GAF = Decimal(data['health_factor_anchors']['Adjustment_Factor_GAF'])
        portfolio = {'collaterals': {}, 'debts': {}}
        
        for item in data['numerator_collaterals']:
            symbol = item['symbol']
            amount = Decimal(item['amount_raw']) / (Decimal('10') ** item['decimals'])
            lt = Decimal(item['liquidation_threshold'])
            portfolio['collaterals'][symbol] = {'amount': amount, 'lt': lt}

        for item in data['denominator_debts']:
            symbol = item['symbol']
            amount = Decimal(item['amount_raw']) / (Decimal('10') ** item['decimals'])
            portfolio['debts'][symbol] = {'amount': amount}

        print("  Portfolio data and GAF loaded.")
        return portfolio, GAF

    except FileNotFoundError:
        print(f"  [Error] Static data file not found: {json_path}")
        exit()

def load_wallets(json_path: str) -> list:
    """
    加载 wallets.json 文件.
    将 available_usd 转换为 Decimal.
    """
    print(f"Loading wallet scenarios from: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            wallets = json.load(f)
        
        for wallet in wallets:
            wallet['available_usd'] = Decimal(wallet['available_usd'])
            
        print(f"  Loaded {len(wallets)} wallet scenarios.")
        return wallets

    except FileNotFoundError:
        print(f"  [Error] Wallets file not found: {json_path}")
        exit()
        
def calculate_hf(current_amounts: dict, static_portfolio: dict, prices_t: dict, gaf: Decimal) -> Decimal:
    """
    根据当前资产和价格计算调整后的HF。
    (与前一个脚本相同)
    """
    numerator = Decimal('0')
    numerator += current_amounts['WETH'] * \
                 prices_t['WETH_price_usd'] * \
                 static_portfolio['collaterals']['WETH']['lt']
    numerator += current_amounts['cbBTC'] * \
                 prices_t['cbBTC_price_usd'] * \
                 static_portfolio['collaterals']['cbBTC']['lt']
    
    denominator = current_amounts['USDC'] * prices_t['USDC_price_usd']

    if denominator == Decimal('0'):
        return Decimal('inf')
        
    hf_unadjusted = numerator / denominator
    hf_adjusted = hf_unadjusted * gaf
    
    return hf_adjusted

# (!!!) 修复: 新的递归函数以解决 JSON 序列化错误
def convert_to_json_serializable(obj):
    """
    递归地将 Decimal, Timestamp, 和 numpy floats 转换为
    JSON 兼容的字符串。
    """
    if isinstance(obj, (Decimal, np.float64, np.float32)):
        return f"{obj:.18f}"
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def main():
    """
    主模拟函数：运行策略A（模拟对比）
    """
    print("--- Strategy A Simulation Script (Comparative Simulation) ---")
    
    # --- 1. 加载所有数据 ---
    try:
        static_portfolio, GAF = load_static_data(OUTLINE_FILE)
        wallets = load_wallets(WALLETS_FILE)
        
        df_triggers_raw = pd.read_csv(TRIGGER_FILE)
        df_triggers_raw['datetime_utc'] = pd.to_datetime(df_triggers_raw['datetime_utc'])
        
        df_prices = pd.read_csv(MERGED_DATA_FILE)
        df_prices['datetime_utc'] = pd.to_datetime(df_prices['datetime_utc'])

        df_risk_class = pd.read_csv(RISK_CLASSIFICATION_FILE)
        
    except FileNotFoundError as e:
        print(f"  [Error] 必需文件未找到: {e.filename}")
        return
    except Exception as e:
        print(f"  [Error] 加载数据时出错: {e}")
        return

    # --- 2. 准备模拟环境 ---
    
    # a. 计算成功阈值 (q30)
    HF_STATIC_THRESH = Decimal(df_risk_class['Weighted_HF'].quantile(0.30))
    print(f"Success Threshold (HF_TARGET = q30) set to: {HF_STATIC_THRESH:.8f}")

    # b. 筛选出 "中风险触发" 的时间点
    df_triggers = df_triggers_raw[
        (df_triggers_raw['Previous_Risk_Level'] == 'Low Risk') &
        (df_triggers_raw['New_Risk_Level'] == 'Medium Risk')
    ]
    
    if df_triggers.empty:
        print("\nNo 'Low -> Medium' risk triggers found. No simulations to run.")
        return
        
    print(f"Found {len(df_triggers)} 'Low -> Medium' risk escalation points to simulate.")
    
    # c. 将触发器与价格数据合并
    sim_jobs = pd.merge(df_triggers, df_prices, on='datetime_utc', how='left')
    
    # --- 3. 运行模拟 ---
    
    results_list = []
    
    print(f"Running {len(sim_jobs) * len(wallets)} simulations...")

    for index, row in sim_jobs.iterrows():
        
        # a. 获取基准状态
        t_datetime = row['datetime_utc']
        HF_before = Decimal(row['HF_adjusted']) 
        
        prices_t = {
            'WETH_price_usd': Decimal(row['WETH_price_usd']),
            'cbBTC_price_usd': Decimal(row['cbBTC_price_usd']),
            'USDC_price_usd': Decimal(row['USDC_price_usd'])
        }
        
        amounts_before = {
            'WETH': static_portfolio['collaterals']['WETH']['amount'],
            'cbBTC': static_portfolio['collaterals']['cbBTC']['amount'],
            'USDC': static_portfolio['debts']['USDC']['amount']
        }

        # b. 针对每个钱包情景进行模拟
        for wallet in wallets:
            wallet_id = wallet['wallet_id']
            wallet_value_usd = wallet['available_usd'] # 这是一个 Decimal
            
            # --- 模拟行动1: 偿还债务 ---
            delta_q_repay = wallet_value_usd / prices_t['USDC_price_usd']
            # (!!!) 增加: 计算操作的USD价值 (应几乎等于 wallet_value_usd)
            usd_value_repay = delta_q_repay * prices_t['USDC_price_usd']
            amounts_A = amounts_before.copy()
            amounts_A['USDC'] -= delta_q_repay
            
            HF_after_A = calculate_hf(amounts_A, static_portfolio, prices_t, GAF)
            HF_Improve_A = HF_after_A - HF_before
            Success_A = (HF_after_A >= HF_STATIC_THRESH)

            # --- 模拟行动2: 增加抵押 (WETH) ---
            delta_q_add = wallet_value_usd / prices_t['WETH_price_usd']
            # (!!!) 增加: 计算操作的USD价值
            usd_value_add = delta_q_add * prices_t['WETH_price_usd']
            amounts_B = amounts_before.copy()
            amounts_B['WETH'] += delta_q_add
            
            HF_after_B = calculate_hf(amounts_B, static_portfolio, prices_t, GAF)
            HF_Improve_B = HF_after_B - HF_before
            Success_B = (HF_after_B >= HF_STATIC_THRESH)

            # --- c. 决策逻辑 ---
            if not Success_A and not Success_B:
                Final_Decision = "ESCALATE_TO_STRATEGY_B"
                Decision_Reason = "Neither 'Repay Debt' nor 'Add Collateral' achieved the HF_TARGET (q30)."
                Final_HF_After = HF_before
                Chosen_Improvement = Decimal('0')
            
            elif Success_A and not Success_B:
                Final_Decision = "Repay_Debt"
                Decision_Reason = "'Repay Debt' achieved the HF_TARGET while 'Add Collateral' did not."
                Final_HF_After = HF_after_A
                Chosen_Improvement = HF_Improve_A
            
            elif not Success_A and Success_B:
                Final_Decision = "Add_Collateral"
                Decision_Reason = "'Add Collateral' achieved the HF_TARGET while 'Repay Debt' did not."
                Final_HF_After = HF_after_B
                Chosen_Improvement = HF_Improve_B
            
            else: 
                if HF_Improve_A >= HF_Improve_B:
                    Final_Decision = "Repay_Debt"
                    Decision_Reason = "Both actions succeeded. 'Repay Debt' was chosen (Better HF Gain)."
                    Final_HF_After = HF_after_A
                    Chosen_Improvement = HF_Improve_A
                else:
                    Final_Decision = "Add_Collateral"
                    Decision_Reason = "Both actions succeeded. 'Add Collateral' was chosen (Better HF Gain)."
                    Final_HF_After = HF_after_B
                    Chosen_Improvement = HF_Improve_B
            
            # --- d. 存储结果 ---
            results_list.append({
                "datetime_utc": t_datetime,
                "wallet_id": wallet_id,
                "available_usd": wallet_value_usd,
                "Reason_For_Trigger": row['Reason_For_Change (Quadrant)'],
                "HF_before": HF_before,
                "HF_TARGET (q30)": HF_STATIC_THRESH,
                
                "Final_Decision": Final_Decision,
                "Decision_Reason": Decision_Reason,
                
                "HF_after": Final_HF_After,
                "HF_Improvement": Chosen_Improvement,
                
                # (!!!) 已扩展指标
                "metrics_Repay_Debt": {
                    "Operation_Asset": "USDC",
                    "Operation_Quantity": delta_q_repay,
                    "Operation_USD_Value": usd_value_repay,
                    "HF_after": HF_after_A,
                    "HF_Improvement": HF_Improve_A,
                    "Success": Success_A
                },
                "metrics_Add_Collateral": {
                    "Operation_Asset": "WETH",
                    "Operation_Quantity": delta_q_add,
                    "Operation_USD_Value": usd_value_add,
                    "HF_after": HF_after_B,
                    "HF_Improvement": HF_Improve_B,
                    "Success": Success_B
                }
            })

    # --- 4. 保存结果 (已修改为 JSON) ---
    print("\nSimulation complete.")
    
    if not results_list:
        print("No simulation results to save.")
        return

    # (!!!) 修复: 使用新的递归函数
    serializable_results = convert_to_json_serializable(results_list)
            
    try:
        # 保存为 JSON
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"\nSuccessfully saved {len(serializable_results)} simulation results to:\n{OUTPUT_FILE}")
    except Exception as e:
        print(f"\n[Error] Failed to save output file: {e}")


# --- 脚本入口 ---
if __name__ == "__main__":
    main()