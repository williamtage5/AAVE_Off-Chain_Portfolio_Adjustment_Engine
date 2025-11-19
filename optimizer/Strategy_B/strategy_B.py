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
TRIGGER_FILE = os.path.join(OPERATION_POINTS_DATA, "risk_escalation_points.csv") #
WALLETS_FILE = os.path.join(OPTIMIZER_PATH, "wallet_generation", "data", "wallets.json") #
OUTLINE_FILE = os.path.join(OPTIMIZER_PATH, "data", "outline.json") #
MERGED_DATA_FILE = os.path.join(OPTIMIZER_PATH, "data", "merged_hf_and_prices.csv") #
TRUE_HF_FILE = os.path.join(BASE_PATH, "HF", "data", "true_HF", "true_hf_adjusted.csv") #

# 输出目录和文件
OUTPUT_DIR = os.path.join(OPTIMIZER_PATH, "Strategy_B", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "strategy_B_optimization_results.json")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. 定义模拟超参数 ---

# (!!!) 关键超参数: 风险货币化因子 (来自您的运筹优化模型)
# 含义: HF 提升 1.0 (例如从 1.0 到 2.0) 值多少美元？
# (HF 提升 0.01 约值 1 USD)
LAMBDA = Decimal('100.0') 

# Metric 2 (清算率) 的时间范围
T_HORIZON = 6.0  # 6 小时
K_THRESHOLD = 1.0  # 清算HF

# 运筹优化网格搜索的粒度 (10 = 10% 步长)
GRID_STEP = 10


# --- 3. 辅助函数 (来自所有脚本) ---

def load_static_data(json_path: str) -> tuple[dict, Decimal]:
    """加载 outline.json"""
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
    """计算 HF"""
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
        return Decimal('0.0')
    v = mu - 0.5 * (sigma**2)
    d = np.log(S0 / K)
    sqrt_T = np.sqrt(T)
    den = sigma * sqrt_T
    d1 = ((v * T) - d) / den
    d2 = ((-v * T) - d) / den
    term1 = norm.cdf(d1)
    term2 = np.exp((2 * v * d) / (sigma**2)) * norm.cdf(d2)
    return Decimal(str(term1 + term2))

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
    主模拟函数：运行策略B（运筹优化）
    """
    print("--- Strategy B Simulation Script (Optimization) ---")
    
    # --- 1. 加载所有数据 ---
    try:
        static_portfolio, GAF = load_static_data(OUTLINE_FILE)
        wallets = load_wallets(WALLETS_FILE)
        
        df_triggers_raw = pd.read_csv(TRIGGER_FILE) #
        df_triggers_raw['datetime_utc'] = pd.to_datetime(df_triggers_raw['datetime_utc'])
        
        df_prices = pd.read_csv(MERGED_DATA_FILE) #
        df_prices['datetime_utc'] = pd.to_datetime(df_prices['datetime_utc'])
        df_prices = df_prices.set_index('datetime_utc').sort_index()

    except FileNotFoundError as e:
        print(f"  [Error] 必需文件未找到: {e.filename}")
        return
    except Exception as e:
        print(f"  [Error] 加载数据时出错: {e}")
        return

    # --- 2. 准备模拟环境 ---
    
    # a. 计算 Metric 2 的参数 (mu, sigma)
    mu_hist_hourly, sigma_hist_hourly = calculate_mu_sigma_from_history(TRUE_HF_FILE)

    # b. (!!!) 筛选出 "高风险触发" 的时间点
    df_triggers = df_triggers_raw[
        (df_triggers_raw['New_Risk_Level'] == 'High Risk')
    ]
    
    if df_triggers.empty:
        print("\nNo 'High Risk' triggers found. No simulations to run.")
        return
        
    print(f"Found {len(df_triggers)} 'High Risk' escalation points to simulate.")
    
    # c. 将触发器与价格数据合并
    sim_jobs = df_triggers.join(df_prices, on='datetime_utc', how='left')
    
    # --- 3. 运行模拟 ---
    
    results_list = []
    
    print(f"Running {len(sim_jobs) * len(wallets)} optimization simulations...")

    for index, row in sim_jobs.iterrows():
        
        # a. 获取基准状态
        t_datetime = row['datetime_utc']
        HF_before = Decimal(row['HF_adjusted']) #
        
        # b. 获取 t 和 t-1 的数据 (用于 Metric 3)
        try:
            t_index_pos = df_prices.index.get_loc(t_datetime)
            if t_index_pos == 0:
                print(f"  Skipping {t_datetime}: Cannot calculate t-1 cost.")
                continue
            
            prices_t = {
                'WETH_price_usd': Decimal(row['WETH_price_usd']),
                'cbBTC_price_usd': Decimal(row['cbBTC_price_usd']),
                'USDC_price_usd': Decimal(row['USDC_price_usd'])
            }
            
            prices_t_minus_1_row = df_prices.iloc[t_index_pos - 1]
            prices_t_minus_1 = {
                'WETH_price_usd': Decimal(prices_t_minus_1_row['WETH_price_usd']),
                'cbBTC_price_usd': Decimal(prices_t_minus_1_row['cbBTC_price_usd']),
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
        
        # c. 计算基准 P(Liq) (Metric 2)
        P_Liq_before = calculate_liquidation_probability(HF_before, K_THRESHOLD, T_HORIZON, mu_hist_hourly, sigma_hist_hourly)

        # d. 针对每个钱包情景进行模拟
        for wallet in wallets:
            wallet_id = wallet['wallet_id']
            Wallet_USD = wallet['available_usd'] # 这是一个 Decimal
            
            print(f"  Optimizing for t={t_datetime}, wallet={wallet_id} (${Wallet_USD:,.2f})...")
            
            best_Z = Decimal('-inf')
            best_action_details = {}

            # --- e. (!!!) 运筹优化: 网格搜索 ---
            # 遍历分配给 USDC 的百分比
            for pct_usdc in range(0, 101, GRID_STEP):
                
                # 遍历分配给 WETH 的百分比
                for pct_weth in range(0, 101 - pct_usdc, GRID_STEP):
                    
                    # 剩余的分配给 cbBTC
                    pct_cbtc = 100 - pct_usdc - pct_weth
                    
                    # 1. 计算此组合的USD价值和数量
                    V_repay_USDC = (Decimal(pct_usdc) / Decimal(100)) * Wallet_USD
                    V_add_WETH = (Decimal(pct_weth) / Decimal(100)) * Wallet_USD
                    V_add_cbBTC = (Decimal(pct_cbtc) / Decimal(100)) * Wallet_USD

                    ΔQ_repay_USDC = V_repay_USDC / prices_t['USDC_price_usd']
                    ΔQ_add_WETH = V_add_WETH / prices_t['WETH_price_usd']
                    ΔQ_add_cbBTC = V_add_cbBTC / prices_t['cbBTC_price_usd']
                    
                    # 2. 计算 Metric 3 (Operation Cost)
                    Cost_USDC = (prices_t['USDC_price_usd'] - prices_t_minus_1['USDC_price_usd']) * ΔQ_repay_USDC
                    Cost_WETH = (prices_t['WETH_price_usd'] - prices_t_minus_1['WETH_price_usd']) * ΔQ_add_WETH
                    Cost_cbBTC = (prices_t['cbBTC_price_usd'] - prices_t_minus_1['cbBTC_price_usd']) * ΔQ_add_cbBTC
                    Total_Operation_Cost = Cost_USDC + Cost_WETH + Cost_cbBTC

                    # 3. 计算 Metric 1 (HF Improvement)
                    amounts_after = {
                        'WETH': amounts_before['WETH'] + ΔQ_add_WETH,
                        'cbBTC': amounts_before['cbBTC'] + ΔQ_add_cbBTC,
                        'USDC': amounts_before['USDC'] - ΔQ_repay_USDC
                    }
                    HF_after = calculate_hf(amounts_after, static_portfolio, prices_t, GAF)
                    HF_Improvement = HF_after - HF_before
                    
                    # 4. (!!!) 计算目标函数 Z
                    Z = (HF_Improvement * LAMBDA) - Total_Operation_Cost

                    # 5. 存储最佳结果
                    if Z > best_Z:
                        best_Z = Z
                        best_action_details = {
                            "Z_Objective_Score": Z,
                            "Allocation_pct": {
                                "Repay_USDC": pct_usdc,
                                "Add_WETH": pct_weth,
                                "Add_cbBTC": pct_cbtc
                            },
                            "Allocation_usd": {
                                "Repay_USDC": V_repay_USDC,
                                "Add_WETH": V_add_WETH,
                                "Add_cbBTC": V_add_cbBTC
                            },
                            "Allocation_quantity": {
                                "Repay_USDC": ΔQ_repay_USDC,
                                "Add_WETH": ΔQ_add_WETH,
                                "Add_cbBTC": ΔQ_add_cbBTC
                            },
                            "Result_HF_after": HF_after,
                            "Metric1_HF_Improvement": HF_Improvement,
                            "Metric3_Operation_Cost_USD": Total_Operation_Cost
                        }

            # --- f. (!!!) 评估最优解的 Metric 2 ---
            if best_action_details: # 确保我们找到了一个解
                HF_after_best = best_action_details['Result_HF_after']
                P_Liq_after = calculate_liquidation_probability(HF_after_best, K_THRESHOLD, T_HORIZON, mu_hist_hourly, sigma_hist_hourly)
                Liq_Rate_Reduction = P_Liq_before - P_Liq_after
                
                best_action_details["Metric2_P_Liq_After"] = P_Liq_after
                best_action_details["Metric2_Liq_Rate_Reduction"] = Liq_Rate_Reduction
            
            # --- g. 存储最终结果 ---
            results_list.append({
                "datetime_utc": t_datetime,
                "wallet_id": wallet_id,
                "available_usd": Wallet_USD,
                "Reason_For_Trigger": row['Reason_For_Change (Quadrant)'],
                "Baseline_State": {
                    "HF_before": HF_before,
                    "P_Liq_before_6h": P_Liq_before,
                },
                "Optimization_Parameters": {
                    "LAMBDA": LAMBDA,
                    "GRID_STEP_pct": GRID_STEP
                },
                "Optimal_Action": best_action_details
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
        print(f"\nSuccessfully saved {len(serializable_results)} optimization results to:\n{OUTPUT_FILE}")
    except Exception as e:
        print(f"\n[Error] Failed to save output file: {e}")

# --- 脚本入口 ---
if __name__ == "__main__":
    main()