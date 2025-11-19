import pandas as pd
import os
import json
import numpy as np
from scipy.stats import norm
from decimal import Decimal, getcontext
from datetime import datetime

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
# (!!!) 新增: 用于 Metric 2
TRUE_HF_FILE = os.path.join(BASE_PATH, "HF", "data", "true_HF", "true_hf_adjusted.csv")


# (!!!) 输出目录和文件 (新名称)
OUTPUT_DIR = os.path.join(OPTIMIZER_PATH, "Strategy_A", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "strategy_A_full_metrics_results.json")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- 2. 辅助函数 (来自所有脚本) ---

def load_static_data(json_path: str) -> tuple[dict, Decimal]:
    """加载 outline.json (来自 HF Improvement.py)"""
    print(f"Loading static portfolio data from: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        GAF = Decimal(data['health_factor_anchors']['Adjustment_Factor_GAF'])
        portfolio = {'collaterals': {}, 'debts': {}}
        for item in data['numerator_collaterals']:
            symbol = item['symbol']
            amount = Decimal(item['amount_raw']) / (Decimal('10')**item['decimals'])
            lt = Decimal(item['liquidation_threshold'])
            portfolio['collaterals'][symbol] = {'amount': amount, 'lt': lt}
        for item in data['denominator_debts']:
            symbol = item['symbol']
            amount = Decimal(item['amount_raw']) / (Decimal('10')**item['decimals'])
            portfolio['debts'][symbol] = {'amount': amount}
        print("  Portfolio data and GAF loaded.")
        return portfolio, GAF
    except FileNotFoundError:
        print(f"  [Error] Static data file not found: {json_path}")
        exit()

def load_wallets(json_path: str) -> list:
    """加载 wallets.json"""
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
    """计算 HF (来自 HF Improvement.py)"""
    numerator = Decimal('0')
    numerator += current_amounts['WETH'] * prices_t['WETH_price_usd'] * static_portfolio['collaterals']['WETH']['lt']
    numerator += current_amounts['cbBTC'] * prices_t['cbBTC_price_usd'] * static_portfolio['collaterals']['cbBTC']['lt']
    denominator = current_amounts['USDC'] * prices_t['USDC_price_usd']
    if denominator == Decimal('0'):
        return Decimal('inf')
    hf_unadjusted = numerator / denominator
    hf_adjusted = hf_unadjusted * gaf
    return hf_adjusted

def calculate_mu_sigma_from_history(true_hf_file: str) -> tuple[float, float]:
    """(来自 Liquidation Rate Reduction.py)"""
    print(f"Loading historical HF data from: {true_hf_file}")
    try:
        df_hist = pd.read_csv(true_hf_file)
        log_returns = np.log(df_hist['HF_adjusted'] / df_hist['HF_adjusted'].shift(1)).dropna()
        mu_hourly = log_returns.mean()
        sigma_hourly = log_returns.std()
        print(f"  Calculated Historical Hourly Mu (μ): {mu_hourly: .8f}")
        print(f"  Calculated Historical Hourly Sigma (σ): {sigma_hourly: .8f}")
        return float(mu_hourly), float(sigma_hourly)
    except FileNotFoundError:
        print(f"  [Error] Historical HF file not found: {true_hf_file}")
        exit()

def calculate_liquidation_probability(S0: Decimal, K: float, T: float, mu: float, sigma: float) -> Decimal:
    """(来自 Liquidation Rate Reduction.py)"""
    S0 = float(S0)
    if S0 <= K:
        return Decimal('1.0')
    if sigma == 0.0:
        return Decimal('0.0') # 简化处理

    v = mu - 0.5 * (sigma**2)
    d = np.log(S0 / K)
    sqrt_T = np.sqrt(T)
    den = sigma * sqrt_T
    
    d1 = ((v * T) - d) / den
    d2 = ((-v * T) - d) / den
    
    term1 = norm.cdf(d1)
    term2 = np.exp((2 * v * d) / (sigma**2)) * norm.cdf(d2)
    
    return Decimal(str(term1 + term2)) # 转换为 Decimal

def convert_to_json_serializable(obj):
    """递归地将 Decimal/Timestamp/Numpy 转换为 JSON 兼容的字符串"""
    if isinstance(obj, (Decimal, np.float64, np.float32)):
        return f"{obj:.18f}"
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


def main():
    """
    主模拟函数：运行策略A（模拟对比）并计算所有3个指标
    """
    print("--- Strategy A Full Metrics Simulation Script ---")
    
    # --- 1. 加载所有数据 ---
    try:
        static_portfolio, GAF = load_static_data(OUTLINE_FILE)
        wallets = load_wallets(WALLETS_FILE)
        
        df_triggers_raw = pd.read_csv(TRIGGER_FILE)
        df_triggers_raw['datetime_utc'] = pd.to_datetime(df_triggers_raw['datetime_utc'])
        
        df_prices = pd.read_csv(MERGED_DATA_FILE)
        df_prices['datetime_utc'] = pd.to_datetime(df_prices['datetime_utc'])
        # (!!!) 将价格表建立索引，以便于 t-1 查找
        df_prices = df_prices.set_index('datetime_utc').sort_index()

        df_risk_class = pd.read_csv(RISK_CLASSIFICATION_FILE)
        
    except FileNotFoundError as e:
        print(f"  [Error] 必需文件未找到: {e.filename}")
        return
    except Exception as e:
        print(f"  [Error] 加载数据时出错: {e}")
        return

    # --- 2. 准备模拟环境 ---
    
    # a. 计算 Metric 1 的成功阈值 (q30)
    HF_STATIC_THRESH = Decimal(df_risk_class['Weighted_HF'].quantile(0.30))
    print(f"Success Threshold (HF_TARGET = q30) set to: {HF_STATIC_THRESH:.8f}")

    # b. 计算 Metric 2 的参数 (mu, sigma, T)
    mu_hist_hourly, sigma_hist_hourly = calculate_mu_sigma_from_history(TRUE_HF_FILE)
    T_horizon = 6.0  # 6 小时
    K_threshold = 1.0  # 清算HF

    # c. 筛选出 "中风险触发" 的时间点
    df_triggers = df_triggers_raw[
        (df_triggers_raw['Previous_Risk_Level'] == 'Low Risk') &
        (df_triggers_raw['New_Risk_Level'] == 'Medium Risk')
    ]
    
    if df_triggers.empty:
        print("\nNo 'Low -> Medium' risk triggers found. No simulations to run.")
        return
        
    print(f"Found {len(df_triggers)} 'Low -> Medium' risk escalation points to simulate.")
    
    # d. 将触发器与价格数据合并
    # (!!!) 注意: df_prices 已经是一个索引DF
    sim_jobs = df_triggers.join(df_prices, on='datetime_utc', how='left')
    
    # --- 3. 运行模拟 ---
    
    results_list = []
    
    print(f"Running {len(sim_jobs) * len(wallets)} simulations...")

    for index, row in sim_jobs.iterrows():
        
        # a. 获取基准状态
        t_datetime = row['datetime_utc']
        HF_before = Decimal(row['HF_adjusted']) 
        
        # b. (!!!) 获取 t 和 t-1 的数据 (用于 Metric 3)
        try:
            t_index_pos = df_prices.index.get_loc(t_datetime)
            if t_index_pos == 0:
                print(f"  Skipping {t_datetime}: Cannot calculate t-1 cost for first record.")
                continue
            
            prices_t = {
                'WETH_price_usd': Decimal(row['WETH_price_usd']),
                'cbBTC_price_usd': Decimal(row['cbBTC_price_usd']),
                'USDC_price_usd': Decimal(row['USDC_price_usd'])
            }
            
            prices_t_minus_1_row = df_prices.iloc[t_index_pos - 1]
            prices_t_minus_1 = {
                'WETH_price_usd': Decimal(prices_t_minus_1_row['WETH_price_usd']),
                'USDC_price_usd': Decimal(prices_t_minus_1_row['USDC_price_usd'])
            }
            
        except KeyError:
            print(f"  Skipping {t_datetime}: Could not find price data.")
            continue
        
        amounts_before = {
            'WETH': static_portfolio['collaterals']['WETH']['amount'],
            'cbBTC': static_portfolio['collaterals']['cbBTC']['amount'],
            'USDC': static_portfolio['debts']['USDC']['amount']
        }
        
        # c. (!!!) 计算基准 P(Liq) (Metric 2)
        P_Liq_before = calculate_liquidation_probability(HF_before, K_threshold, T_horizon, mu_hist_hourly, sigma_hist_hourly)

        # d. 针对每个钱包情景进行模拟
        for wallet in wallets:
            wallet_id = wallet['wallet_id']
            wallet_value_usd = wallet['available_usd']
            
            # --- 模拟行动1: 偿还债务 ---
            delta_q_repay = wallet_value_usd / prices_t['USDC_price_usd']
            usd_value_repay = delta_q_repay * prices_t['USDC_price_usd']
            amounts_A = amounts_before.copy()
            amounts_A['USDC'] -= delta_q_repay
            
            HF_after_A = calculate_hf(amounts_A, static_portfolio, prices_t, GAF)
            # Metric 1 (HF Imprv)
            HF_Improve_A = HF_after_A - HF_before
            Success_A = (HF_after_A >= HF_STATIC_THRESH)
            # Metric 2 (Liq Rate)
            P_Liq_after_A = calculate_liquidation_probability(HF_after_A, K_threshold, T_horizon, mu_hist_hourly, sigma_hist_hourly)
            Liq_Rate_Reduction_A = P_Liq_before - P_Liq_after_A
            # Metric 3 (Cost)
            Cost_A = (prices_t_minus_1['USDC_price_usd'] - prices_t['USDC_price_usd']) * delta_q_repay


            # --- 模拟行动2: 增加抵押 (WETH) ---
            delta_q_add = wallet_value_usd / prices_t['WETH_price_usd']
            usd_value_add = delta_q_add * prices_t['WETH_price_usd']
            amounts_B = amounts_before.copy()
            amounts_B['WETH'] += delta_q_add
            
            HF_after_B = calculate_hf(amounts_B, static_portfolio, prices_t, GAF)
            # Metric 1 (HF Imprv)
            HF_Improve_B = HF_after_B - HF_before
            Success_B = (HF_after_B >= HF_STATIC_THRESH)
            # Metric 2 (Liq Rate)
            P_Liq_after_B = calculate_liquidation_probability(HF_after_B, K_threshold, T_horizon, mu_hist_hourly, sigma_hist_hourly)
            Liq_Rate_Reduction_B = P_Liq_before - P_Liq_after_B
            # Metric 3 (Cost)
            Cost_B = (prices_t_minus_1['WETH_price_usd'] - prices_t['WETH_price_usd']) * delta_q_add

            # --- e. 决策逻辑 (基于 Metric 1 的成功) ---
            if not Success_A and not Success_B:
                Final_Decision = "ESCALATE_TO_STRATEGY_B"
                Decision_Reason = "Neither action achieved the HF_TARGET (q30)."
                Final_HF_After = HF_before
            elif Success_A and not Success_B:
                Final_Decision = "Repay_Debt"
                Decision_Reason = "'Repay Debt' succeeded while 'Add Collateral' did not."
                Final_HF_After = HF_after_A
            elif not Success_A and Success_B:
                Final_Decision = "Add_Collateral"
                Decision_Reason = "'Add Collateral' succeeded while 'Repay Debt' did not."
                Final_HF_After = HF_after_B
            else: 
                if HF_Improve_A >= HF_Improve_B: # 比较 Metric 1
                    Final_Decision = "Repay_Debt"
                    Decision_Reason = "Both succeeded. 'Repay Debt' chosen (Better HF Gain)."
                    Final_HF_After = HF_after_A
                else:
                    Final_Decision = "Add_Collateral"
                    Decision_Reason = "Both succeeded. 'Add Collateral' chosen (Better HF Gain)."
                    Final_HF_After = HF_after_B
            
            # --- f. 存储所有结果 ---
            results_list.append({
                "datetime_utc": t_datetime,
                "wallet_id": wallet_id,
                "available_usd": wallet_value_usd,
                "Reason_For_Trigger": row['Reason_For_Change (Quadrant)'],
                
                "Baseline_State": {
                    "HF_before": HF_before,
                    "P_Liq_before_6h": P_Liq_before,
                    "HF_TARGET (q30)": HF_STATIC_THRESH
                },
                
                "Final_Decision": Final_Decision,
                "Decision_Reason": Decision_Reason,
                "HF_after_Action": Final_HF_After,
                
                "metrics_Repay_Debt": {
                    "Operation_Asset": "USDC",
                    "Operation_Quantity": delta_q_repay,
                    "Operation_USD_Value": usd_value_repay,
                    "Success": Success_A,
                    "Metric1_HF_Improvement": HF_Improve_A,
                    "Metric2_P_Liq_After": P_Liq_after_A,
                    "Metric2_Liq_Rate_Reduction": Liq_Rate_Reduction_A,
                    "Metric3_Operation_Cost_USD": Cost_A
                },
                "metrics_Add_Collateral": {
                    "Operation_Asset": "WETH",
                    "Operation_Quantity": delta_q_add,
                    "Operation_USD_Value": usd_value_add,
                    "Success": Success_B,
                    "Metric1_HF_Improvement": HF_Improve_B,
                    "Metric2_P_Liq_After": P_Liq_after_B,
                    "Metric2_Liq_Rate_Reduction": Liq_Rate_Reduction_B,
                    "Metric3_Operation_Cost_USD": Cost_B
                }
            })

    # --- 4. 保存结果 (JSON) ---
    print("\nSimulation complete.")
    
    if not results_list:
        print("No simulation results to save.")
        return

    serializable_results = convert_to_json_serializable(results_list)
            
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=4)
        print(f"\nSuccessfully saved {len(serializable_results)} simulation results to:\n{OUTPUT_FILE}")
    except Exception as e:
        print(f"\n[Error] Failed to save output file: {e}")

# --- 脚本入口 ---
if __name__ == "__main__":
    main()