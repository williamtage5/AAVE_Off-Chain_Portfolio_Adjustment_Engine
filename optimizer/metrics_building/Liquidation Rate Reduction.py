import pandas as pd
import os
import json
import numpy as np
from scipy.stats import norm
from decimal import Decimal, getcontext

# --- 0. 设置精度 ---
getcontext().prec = 50

# --- 1. 定义文件路径 ---
BASE_PATH = r"F:\Learning_journal_at_CUHK\FTEC5520_Appl Blockchain & Cryptocur\Simulation"
MERGED_DATA_FILE = os.path.join(BASE_PATH, "optimizer", "data", "merged_hf_and_prices.csv")
OUTLINE_FILE = os.path.join(BASE_PATH, "optimizer", "data", "outline.json")

# (!!!) 新增文件: 用于计算历史 mu 和 sigma
TRUE_HF_FILE = os.path.join(BASE_PATH, "HF", "data", "true_HF", "true_hf_adjusted.csv")


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

        for item in data['denominator_debts']:
            symbol = item['symbol']
            amount = Decimal(item['amount_raw']) / (Decimal('10') ** item['decimals'])
            portfolio['debts'][symbol] = {'amount': amount}

        print("  Portfolio data loaded.")
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
    (与前一个脚本相同)
    """
    
    # --- 计算分子 (抵押品) ---
    numerator = Decimal('0')
    numerator += current_amounts['WETH'] * \
                 prices_t['WETH_price_usd'] * \
                 static_portfolio['collaterals']['WETH']['lt']
    numerator += current_amounts['cbBTC'] * \
                 prices_t['cbBTC_price_usd'] * \
                 static_portfolio['collaterals']['cbBTC']['lt']

    # --- 计算分母 (债务) ---
    denominator = current_amounts['USDC'] * prices_t['USDC_price_usd']

    # --- 计算HF ---
    if denominator == Decimal('0'):
        return Decimal('inf') # HF 无穷大
        
    hf_unadjusted = numerator / denominator
    hf_adjusted = hf_unadjusted * gaf
    
    return hf_adjusted

# --- (!!!) 新函数: 计算历史 Mu 和 Sigma ---
def calculate_mu_sigma_from_history(true_hf_file: str) -> tuple[float, float]:
    """
    (选项 A) 从 true_hf_adjusted.csv 计算历史(每小时) mu 和 sigma
    """
    print(f"\nLoading historical HF data from: {true_hf_file}")
    try:
        df_hist = pd.read_csv(true_hf_file)
        if 'HF_adjusted' not in df_hist.columns:
            print("  [Error] 'HF_adjusted' column not found in historical file.")
            exit()
            
        # 计算对数回报率
        log_returns = np.log(df_hist['HF_adjusted'] / df_hist['HF_adjusted'].shift(1))
        log_returns = log_returns.dropna() # 移除第一个NaN

        # 计算每小时的 mu (漂移率) 和 sigma (波动率)
        mu_hourly = log_returns.mean()
        sigma_hourly = log_returns.std()
        
        print(f"  Calculated Historical Hourly Mu (μ): {mu_hourly: .8f}")
        print(f"  Calculated Historical Hourly Sigma (σ): {sigma_hourly: .8f}")
        
        return float(mu_hourly), float(sigma_hourly)

    except FileNotFoundError:
        print(f"  [Error] Historical HF file not found: {true_hf_file}")
        exit()
    except Exception as e:
        print(f"  [Error] Failed to calculate mu/sigma: {e}")
        exit()

# --- (!!!) 新函数: 计算清算概率 ---
def calculate_liquidation_probability(S0: Decimal, K: float, T: float, mu: float, sigma: float) -> float:
    """
    使用反高斯分布(Inverse Gaussian)CDF计算清算概率 P(liq)
    基于您提供的公式: P(T) = Φ(d1) + exp(2vd/σ^2) * Φ(d2)
    
    :param S0: 起始健康因子 (HF_before 或 HF_after)
    :param K: 触发阈值 (1.0)
    :param T: 时间范围 (6 小时)
    :param mu: 每小时漂移率 (μ)
    :param sigma: 每小时波动率 (σ)
    :return: 0到1之间的概率
    """
    
    # 转换为 float 进行数学运算
    S0 = float(S0) 
    
    # --- 边缘情况处理 ---
    if S0 <= K:
        return 1.0 # 已经处于清算状态
    if sigma == 0.0:
        if S0 > K and mu >= 0:
            return 0.0 # 不会下降
        elif S0 > K and mu < 0:
            # 检查确定性路径是否会触及
            return 1.0 if (S0 * np.exp(mu * T)) <= K else 0.0
        else:
            return 0.0

    # --- 1. 定义公式变量 ---
    v = mu - 0.5 * (sigma**2)  # 有效漂移项
    d = np.log(S0 / K)           # 对数价格距离 (d > 0)
    
    # --- 2. 计算 d1 和 d2 ---
    # (根据您的公式: Φ((vT-d)/...) 和 Φ((-vT-d)/...))
    sqrt_T = np.sqrt(T)
    den = sigma * sqrt_T
    
    d1_num = (v * T) - d
    d2_num = (-v * T) - d
    
    d1 = d1_num / den
    d2 = d2_num / den
    
    # --- 3. 计算两个主要项 ---
    term1 = norm.cdf(d1)
    
    exp_term = np.exp((2 * v * d) / (sigma**2))
    cdf_term = norm.cdf(d2)
    term2 = exp_term * cdf_term
    
    # --- 4. 最终概率 ---
    P_liq = term1 + term2
    
    return P_liq


def main():
    """
    主模拟函数
    """
    
    # --- 1. 加载所有数据 ---
    static_portfolio, GAF = load_static_data(OUTLINE_FILE)
    
    print(f"\nLoading merged time-series data from: {MERGED_DATA_FILE}")
    try:
        df = pd.read_csv(MERGED_DATA_FILE)
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    except FileNotFoundError:
        print(f"  [Error] Merged data file not found: {MERGED_DATA_FILE}")
        return
        
    # (!!!) 计算历史 mu 和 sigma
    mu_hist_hourly, sigma_hist_hourly = calculate_mu_sigma_from_history(TRUE_HF_FILE)
    
    print("\nAll data loaded successfully.")

    # --- 2. 定义模拟场景 ---
    print("\n--- SIMULATION SETUP ---")
    
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
    
    # e. (!!!) 定义 GBM 模型参数
    T_horizon = 6.0  # 6 小时
    K_threshold = 1.0  # 清算HF
    
    print(f"Defined Time Horizon (T): {T_horizon} hours")
    
    # --- 3. 执行计算 (三个场景) ---
    print("\n--- RUNNING SIMULATIONS (Metric 2: Liquidation Rate Reduction) ---")

    # (!!!) 首先，计算 "Before" 状态的清算概率
    P_liq_before = calculate_liquidation_probability(HF_before, K_threshold, T_horizon, mu_hist_hourly, sigma_hist_hourly)

    # --- SCENARIO 1: 增加抵押品 (Add Collateral) ---
    amounts_scen_1 = amounts_before.copy()
    amounts_scen_1['WETH'] += action_add_weth
    HF_after_scen_1 = calculate_hf(amounts_scen_1, static_portfolio, prices_t, GAF)
    
    P_liq_after_scen_1 = calculate_liquidation_probability(HF_after_scen_1, K_threshold, T_horizon, mu_hist_hourly, sigma_hist_hourly)
    reduction_1 = P_liq_before - P_liq_after_scen_1

    # --- SCENARIO 2: 偿还债务 (Repay Debt) ---
    amounts_scen_2 = amounts_before.copy()
    amounts_scen_2['USDC'] -= action_repay_usdc
    HF_after_scen_2 = calculate_hf(amounts_scen_2, static_portfolio, prices_t, GAF)
    
    P_liq_after_scen_2 = calculate_liquidation_probability(HF_after_scen_2, K_threshold, T_horizon, mu_hist_hourly, sigma_hist_hourly)
    reduction_2 = P_liq_before - P_liq_after_scen_2

    # --- SCENARIO 3: 融合 (Fusion: Add Collateral + Repay Debt) ---
    amounts_scen_3 = amounts_before.copy()
    amounts_scen_3['WETH'] += action_add_weth
    amounts_scen_3['USDC'] -= action_repay_usdc
    HF_after_scen_3 = calculate_hf(amounts_scen_3, static_portfolio, prices_t, GAF)
    
    P_liq_after_scen_3 = calculate_liquidation_probability(HF_after_scen_3, K_threshold, T_horizon, mu_hist_hourly, sigma_hist_hourly)
    reduction_3 = P_liq_before - P_liq_after_scen_3
    
    
    # --- 4. 打印结果 ---
    print(f"\n--- Simulation Results at t = {target_time} ---")
    print(f"Model Params: T={T_horizon}h, μ={mu_hist_hourly: .6f}, σ={sigma_hist_hourly: .6f}")
    print(f"Baseline HF Before: {HF_before: .8f}")
    print(f"Baseline P(Liq) Before: {P_liq_before: .8f} ({(P_liq_before*100):.4f} %)\n")

    print(f"  SCENARIO 1: Add {action_add_weth} WETH")
    print(f"    HF After:          {HF_after_scen_1: .8f}")
    print(f"    P(Liq) After:      {P_liq_after_scen_1: .8f} ({(P_liq_after_scen_1*100):.4f} %)")
    print(f"    Liq Rate Reduction: {reduction_1: .8f} ({(reduction_1*100):.4f} %)\n")

    print(f"  SCENARIO 2: Repay {action_repay_usdc} USDC")
    print(f"    HF After:          {HF_after_scen_2: .8f}")
    print(f"    P(Liq) After:      {P_liq_after_scen_2: .8f} ({(P_liq_after_scen_2*100):.4f} %)")
    print(f"    Liq Rate Reduction: {reduction_2: .8f} ({(reduction_2*100):.4f} %)\n")

    print(f"  SCENARIO 3: Fusion (Add WETH + Repay USDC)")
    print(f"    HF After:          {HF_after_scen_3: .8f}")
    print(f"    P(Liq) After:      {P_liq_after_scen_3: .8f} ({(P_liq_after_scen_3*100):.4f} %)")
    print(f"    Liq Rate Reduction: {reduction_3: .8f} ({(reduction_3*100):.4f} %)\n")
    
    print("--- END OF SIMULATION ---")


# --- 脚本入口 ---
if __name__ == "__main__":
    main()