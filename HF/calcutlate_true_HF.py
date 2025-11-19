import pandas as pd
import os
from decimal import Decimal, getcontext

# 设置Decimal的精度
getcontext().prec = 50

# --- 1. 定义文件路径 ---
BASE_PROJECT_PATH = r"F:\Learning_journal_at_CUHK\FTEC5520_Appl Blockchain & Cryptocur\Simulation"

# 输入: 包含WETH, cbBTC, USDC 美元价格的CSV文件目录
PRICE_DATA_DIR = os.path.join(BASE_PROJECT_PATH, "HF", "data", "true_data_in_usd")

# 输出: 存放计算出的HF Csv文件的目录
OUTPUT_HF_DIR = os.path.join(BASE_PROJECT_PATH, "HF", "data", "true_HF")

# --- 2. 用户的持仓数据 (直接从您的提示中复制) ---
USER_PORTFOLIO = {
    "txHash": "0xa0a5a91430df9443b4565ed44c3d21afc1ae142d74dc5fbfd86748db8d708299",
    "user_id": "0x33cf8f585e7063e31ec34f85721a65f4659f7172",
    "simulation_time_range_utc": {
        "start_utc": "2025-10-21T17:00:00+00:00",
        "end_utc": "2025-10-30T12:00:00+00:00",
        "duration_days": 8.814583333333333,
        "is_truncated_90d": False
    },
    "health_factor_anchors": {
        "HF_True_OnChain": 0.9947210949149672,
        "HF_Simulated_Final_Unadjusted": 1.021191354327786,
        "Adjustment_Factor_GAF": 0.9740790408177288
    },
    "numerator_collaterals": [
        {
            "symbol": "WETH",
            "amount_raw": "630624808395278241",
            "decimals": 18,
            "liquidation_threshold": "0.8300"
        },
        {
            "symbol": "cbBTC",
            "amount_raw": "58445",
            "decimals": 8,
            "liquidation_threshold": "0.7800"
        }
    ],
    "denominator_debts": [
        {
            "symbol": "USDC",
            "amount_raw": "2040843772",
            "decimals": 6
        }
    ]
}

def load_and_merge_prices(price_dir: str) -> pd.DataFrame | None:
    """
    加载三种资产的USD价格CSV，并按时间戳合并为一个DataFrame。
    """
    print(f"Loading price data from {price_dir}...")
    try:
        # 定义文件路径
        weth_path = os.path.join(price_dir, "WETH_in_usd.csv")
        cbbtc_path = os.path.join(price_dir, "cbBTC_in_usd.csv")
        usdc_path = os.path.join(price_dir, "USDC_in_usd.csv")

        # 加载CSV
        df_weth = pd.read_csv(weth_path)
        df_cbbtc = pd.read_csv(cbbtc_path)
        df_usdc = pd.read_csv(usdc_path)

        # 为合并做准备，重命名价格列
        df_weth = df_weth.rename(columns={'price_usd': 'WETH_price'})
        df_cbbtc = df_cbbtc.rename(columns={'price_usd': 'cbBTC_price'})
        df_usdc = df_usdc.rename(columns={'price_usd': 'USDC_price'})

        # 合并
        df_merged = pd.merge(df_weth, df_cbbtc, on='datetime_utc', how='inner')
        df_merged = pd.merge(df_merged, df_usdc, on='datetime_utc', how='inner')

        # 转换时间列为datetime对象
        df_merged['datetime_utc'] = pd.to_datetime(df_merged['datetime_utc'])
        
        print(f"Successfully loaded and merged {len(df_merged)} price records.")
        return df_merged
    
    except FileNotFoundError as e:
        print(f"Error: Price file not found. {e}")
        return None
    except Exception as e:
        print(f"Error loading and merging prices: {e}")
        return None

def main():
    """
    主执行函数
    """
    
    # --- 1. 加载并合并价格数据 ---
    price_df = load_and_merge_prices(PRICE_DATA_DIR)
    if price_df is None:
        print("Failed to load price data. Exiting.")
        return

    # --- 2. 解析持仓数据 (使用Decimal进行精确计算) ---
    print("Parsing user portfolio data...")
    portfolio = {}
    
    # a) 抵押品 (分子)
    for item in USER_PORTFOLIO['numerator_collaterals']:
        symbol = item['symbol']
        amount = Decimal(item['amount_raw']) / (Decimal('10') ** item['decimals'])
        lt = Decimal(item['liquidation_threshold'])
        portfolio[symbol] = {'amount': amount, 'lt': lt}
        print(f"  Collateral: {symbol}, Amount: {amount}, LT: {lt}")

    # b) 债务 (分母)
    for item in USER_PORTFOLIO['denominator_debts']:
        symbol = item['symbol']
        amount = Decimal(item['amount_raw']) / (Decimal('10') ** item['decimals'])
        portfolio[symbol] = {'amount': amount}
        print(f"  Debt: {symbol}, Amount: {amount}")

    # c) HF 锚定值
    hf_true_onchain = Decimal(USER_PORTFOLIO['health_factor_anchors']['HF_True_OnChain'])
    sim_end_time = pd.to_datetime(USER_PORTFOLIO['simulation_time_range_utc']['end_utc'])
    print(f"  Anchor HF_True_OnChain: {hf_true_onchain}")
    print(f"  Anchor Time (end_utc): {sim_end_time}")

    # --- 3. 矢量化计算 (未调整的HF) ---
    # HFt = Σ(Q_Collateral * Pt * LT) / Σ(Q_Debt * Pt)
    print("Calculating unadjusted HF time series...")

    # 为确保精度，将Pandas列转换为Decimal
    price_df['WETH_price_d'] = price_df['WETH_price'].apply(Decimal)
    price_df['cbBTC_price_d'] = price_df['cbBTC_price'].apply(Decimal)
    price_df['USDC_price_d'] = price_df['USDC_price'].apply(Decimal)

    # 计算分子: Total_Collateral_Value_Series
    numerator_series = (price_df['WETH_price_d'] * portfolio['WETH']['amount'] * portfolio['WETH']['lt']) + \
                       (price_df['cbBTC_price_d'] * portfolio['cbBTC']['amount'] * portfolio['cbBTC']['lt'])
    
    # 计算分母: Total_Debt_Value_Series
    denominator_series = (price_df['USDC_price_d'] * portfolio['USDC']['amount'])

    # 计算HF
    price_df['HF_unadjusted'] = numerator_series / denominator_series
    print("Unadjusted HF calculated.")

    # --- 4. 计算调整因子 (GAF) ---
    # GAF = HF_TrueOnChain / HF_SimulatedFinal
    print("Calculating Adjustment Factor (GAF)...")
    try:
        # 找到模拟的最终HF (在 'end_utc' 时间点)
        hf_simulated_final_row = price_df[price_df['datetime_utc'] == sim_end_time]
        
        if hf_simulated_final_row.empty:
            print(f"Warning: No data found at exact end_utc ({sim_end_time}). Using last available record.")
            hf_simulated_final = price_df['HF_unadjusted'].iloc[-1]
        else:
            hf_simulated_final = hf_simulated_final_row['HF_unadjusted'].values[0]
            
        print(f"  HF_Simulated_Final at {sim_end_time}: {hf_simulated_final}")
        
        # 确保 hf_simulated_final 不是0
        if hf_simulated_final == Decimal(0):
            print("Error: HF_Simulated_Final is zero. Cannot calculate GAF. Defaulting GAF to 1.0")
            GAF = Decimal('1.0')
        else:
            GAF = hf_true_onchain / hf_simulated_final
            print(f"  GAF calculated: {GAF}")
            
    except Exception as e:
        print(f"Error calculating GAF: {e}. Defaulting GAF to 1.0")
        GAF = Decimal('1.0')

    # --- 5. 计算调整后的HF ---
    price_df['HF_adjusted'] = price_df['HF_unadjusted'] * GAF
    print("Adjusted HF calculated.")

    # --- 6. 保存结果 ---
    print(f"Saving results to {OUTPUT_HF_DIR}...")
    os.makedirs(OUTPUT_HF_DIR, exist_ok=True)

    # 准备输出的DataFrame
    # 将Decimal转换回float以便CSV标准存储
    df_unadjusted_out = price_df[['datetime_utc', 'HF_unadjusted']].copy()
    df_unadjusted_out['HF_unadjusted'] = df_unadjusted_out['HF_unadjusted'].astype(float)
    
    df_adjusted_out = price_df[['datetime_utc', 'HF_adjusted']].copy()
    df_adjusted_out['HF_adjusted'] = df_adjusted_out['HF_adjusted'].astype(float)

    # 定义输出路径
    output_unadjusted_path = os.path.join(OUTPUT_HF_DIR, "true_hf_unadjusted.csv")
    output_adjusted_path = os.path.join(OUTPUT_HF_DIR, "true_hf_adjusted.csv")

    # 保存
    df_unadjusted_out.to_csv(output_unadjusted_path, index=False)
    print(f"Saved unadjusted HF to {output_unadjusted_path}")
    
    df_adjusted_out.to_csv(output_adjusted_path, index=False)
    print(f"Saved adjusted HF to {output_adjusted_path}")

    print("\nProcess completed successfully.")


# --- 脚本入口 ---
if __name__ == "__main__":
    main()