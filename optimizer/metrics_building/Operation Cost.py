import pandas as pd
import os
import json
from decimal import Decimal, getcontext

# --- 0. 设置精度 ---
getcontext().prec = 50

# --- 1. 定义文件路径 ---
BASE_PATH = r"F:\Learning_journal_at_CUHK\FTEC5520_Appl Blockchain & Cryptocur\Simulation"
MERGED_DATA_FILE = os.path.join(BASE_PATH, "optimizer", "data", "merged_hf_and_prices.csv")
# (outline.json 不是计算此指标所必需的)


def main():
    """
    主模拟函数
    """
    
    # --- 1. 加载所有数据 ---
    print(f"Loading merged time-series data from: {MERGED_DATA_FILE}")
    try:
        df = pd.read_csv(MERGED_DATA_FILE)
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
        # 使用 datetime 作为索引以便于 .loc 和 shift
        df = df.set_index('datetime_utc').sort_index() 
    except FileNotFoundError:
        print(f"  [Error] Merged data file not found: {MERGED_DATA_FILE}")
        return
    except Exception as e:
        print(f"  [Error] Failed to load merged data: {e}")
        return
        
    print("All data loaded successfully.\n")

    # --- 2. 定义模拟场景 ---
    print("--- SIMULATION SETUP (Metric 3: Operation Cost) ---")
    
    # a. 找到时间点 't' (执行点 @ HF 最低点)
    target_time_t = df['HF_adjusted'].idxmin()
    target_row_t = df.loc[target_time_t]
    
    # b. 找到时间点 't-1' (决策点)
    # 我们使用 shift(-1) 来获取 t-1 的索引
    # 我们需要找到 t 在其索引中的位置
    try:
        t_index_pos = df.index.get_loc(target_time_t)
        if t_index_pos == 0:
            print(f"  [Error] Lowest HF is at the first data point. Cannot simulate t-1 cost.")
            return
        # 获取 t-1 的时间戳
        target_time_t_minus_1 = df.index[t_index_pos - 1]
        target_row_t_minus_1 = df.loc[target_time_t_minus_1]
        
    except Exception as e:
        print(f"  [Error] Failed to find t-1 row: {e}")
        return
    
    print(f"Decision Time (t-1): {target_time_t_minus_1}")
    print(f"Execution Time (t):  {target_time_t}")

    # c. 获取 't' 和 't-1' 时刻的价格 (P_t 和 P_t-1)
    prices_t = {
        'WETH_price_usd': Decimal(target_row_t['WETH_price_usd']),
        'USDC_price_usd': Decimal(target_row_t['USDC_price_usd'])
    }
    prices_t_minus_1 = {
        'WETH_price_usd': Decimal(target_row_t_minus_1['WETH_price_usd']),
        'USDC_price_usd': Decimal(target_row_t_minus_1['USDC_price_usd'])
    }
    
    # d. 定义示例操作 (与前一个脚本一致)
    action_add_weth = Decimal('0.1')
    action_repay_usdc = Decimal('200.0')

    # --- 3. 执行计算 (三个场景) ---
    print("\n--- RUNNING SIMULATIONS (Metric 3: Operation Cost) ---")
    print("Formula: Cost = (Predicted_Price_at_t-1 - Actual_Price_at_t) * Quantity_Purchased")
    print("(Cost > 0 means SAVING, Cost < 0 means LOSS)\n")

    # --- SCENARIO 1: 购买 WETH (Add Collateral) ---
    q_weth = action_add_weth
    p_weth_t1 = prices_t_minus_1['WETH_price_usd']
    p_weth_t = prices_t['WETH_price_usd']
    
    predicted_cost_1 = p_weth_t1 * q_weth
    actual_cost_1 = p_weth_t * q_weth
    cost_metric_1 = (p_weth_t1 - p_weth_t) * q_weth # (Pred Price - Actual Price) * Q

    # --- SCENARIO 2: 购买 USDC (Repay Debt) ---
    q_usdc = action_repay_usdc
    p_usdc_t1 = prices_t_minus_1['USDC_price_usd']
    p_usdc_t = prices_t['USDC_price_usd']

    predicted_cost_2 = p_usdc_t1 * q_usdc
    actual_cost_2 = p_usdc_t * q_usdc
    cost_metric_2 = (p_usdc_t1 - p_usdc_t) * q_usdc # (Pred Price - Actual Price) * Q
    
    # --- SCENARIO 3: 融合 (Fusion: Buy WETH + Buy USDC) ---
    # 总成本是两者之和
    cost_metric_3 = cost_metric_1 + cost_metric_2
    
    
    # --- 4. 打印结果 ---
    print(f"\n--- Simulation Results (Cost) at t = {target_time_t} ---")

    print(f"  SCENARIO 1: Buy {q_weth} WETH")
    print(f"    Predicted Price (at t-1): {p_weth_t1: .8f} USD")
    print(f"    Actual Price (at t):    {p_weth_t: .8f} USD")
    print(f"    Predicted Cost:         {(predicted_cost_1): .8f} USD")
    print(f"    Actual Cost:            {(actual_cost_1): .8f} USD")
    print(f"    Operation Cost (Saving/Loss): {cost_metric_1: .8f} USD\n")

    print(f"  SCENARIO 2: Buy {q_usdc} USDC")
    print(f"    Predicted Price (at t-1): {p_usdc_t1: .8f} USD")
    print(f"    Actual Price (at t):    {p_usdc_t: .8f} USD")
    print(f"    Predicted Cost:         {(predicted_cost_2): .8f} USD")
    print(f"    Actual Cost:            {(actual_cost_2): .8f} USD")
    print(f"    Operation Cost (Saving/Loss): {cost_metric_2: .8f} USD\n")

    print(f"  SCENARIO 3: Fusion (Buy Both)")
    print(f"    Total Operation Cost:   {cost_metric_3: .8f} USD\n")
    
    if cost_metric_3 > 0:
        print("Result: The 1-hour time delay resulted in a net SAVING.")
    else:
        print("Result: The 1-hour time delay resulted in a net LOSS (Slippage).")
        
    print("--- END OF SIMULATION ---")


# --- 脚本入口 ---
if __name__ == "__main__":
    main()