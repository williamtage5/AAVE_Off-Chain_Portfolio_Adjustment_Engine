import pandas as pd
import os
from decimal import Decimal, getcontext

# --- 0. 设置精度 ---
# 设置Decimal的精度以进行精确的金融计算
getcontext().prec = 50

# --- 1. 定义文件路径 ---
BASE_PATH = r"F:\Learning_journal_at_CUHK\FTEC5520_Appl Blockchain & Cryptocur\Simulation\HF"
PREDICTION_DIR = os.path.join(BASE_PATH, "data", "every_timestape_prediction", "target_time_range_prediction")
OUTPUT_DIR = os.path.join(BASE_PATH, "data", "predict_HF")
OUTPUT_FILENAME = os.path.join(OUTPUT_DIR, "predicted_hf_t1_to_t6.csv")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. 用户的持仓数据 (来自之前的提示) ---
USER_PORTFOLIO_DATA = {
    "health_factor_anchors": {
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

# 预测的步数
LOOKAHEAD_STEPS = 6
COIN_LIST = ["cbBTC", "USDC", "WETH"]

def load_and_parse_portfolio(portfolio_data: dict) -> tuple[dict, Decimal]:
    """
    解析持仓数据，提取数量、LTs和GAF。
    """
    print("Parsing user portfolio data...")
    portfolio = {}
    
    # 1. 提取GAF
    gaf = Decimal(portfolio_data['health_factor_anchors']['Adjustment_Factor_GAF'])
    print(f"  Loaded Global Adjustment Factor (GAF): {gaf}")

    # 2. 提取抵押品 (分子)
    for item in portfolio_data['numerator_collaterals']:
        symbol = item['symbol']
        amount = Decimal(item['amount_raw']) / (Decimal('10') ** item['decimals'])
        lt = Decimal(item['liquidation_threshold'])
        portfolio[symbol] = {'amount': amount, 'lt': lt}
        print(f"  Collateral: {symbol}, Amount: {amount}, LT: {lt}")

    # 3. 提取债务 (分母)
    for item in portfolio_data['denominator_debts']:
        symbol = item['symbol']
        amount = Decimal(item['amount_raw']) / (Decimal('10') ** item['decimals'])
        portfolio[symbol] = {'amount': amount}
        print(f"  Debt: {symbol}, Amount: {amount}")
        
    return portfolio, gaf

def load_and_merge_predictions(prediction_dir: str, coin_list: list) -> pd.DataFrame | None:
    """
    加载所有币种的预测CSV，重命名列，并按datetime_utc合并。
    """
    print(f"Loading predicted prices from {prediction_dir}...")
    merged_df = None

    try:
        for coin in coin_list:
            file_path = os.path.join(prediction_dir, f"{coin}_target_range_prediction.csv")
            df_coin = pd.read_csv(file_path)
            df_coin['datetime_utc'] = pd.to_datetime(df_coin['datetime_utc'])
            
            # 创建一个重命名映射
            # 'predicted_price_t_plus_1' -> 'WETH_pred_t_plus_1'
            rename_map = {
                f'predicted_price_t_plus_{i}': f'{coin}_pred_t_plus_{i}'
                for i in range(1, LOOKAHEAD_STEPS + 1)
            }
            df_coin = df_coin.rename(columns=rename_map)
            
            # 保留需要的列
            columns_to_keep = ['datetime_utc'] + list(rename_map.values())
            df_coin = df_coin[columns_to_keep]

            # 合并
            if merged_df is None:
                merged_df = df_coin
            else:
                merged_df = pd.merge(merged_df, df_coin, on='datetime_utc', how='inner')
        
        if merged_df is None or merged_df.empty:
            print("  [Error] No data loaded or dataframes are empty.")
            return None
            
        print(f"  Successfully loaded and merged {len(merged_df)} timestamps for {coin_list}.")
        return merged_df.set_index('datetime_utc')

    except FileNotFoundError as e:
        print(f"  [Error] File not found: {e.filename}")
        return None
    except Exception as e:
        print(f"  [Error] Failed to load prices: {e}")
        return None

def main():
    """
    主执行函数
    """
    
    # --- 1. 加载持仓和GAF ---
    portfolio, GAF = load_and_parse_portfolio(USER_PORTFOLIO_DATA)
    
    # --- 2. 加载所有预测价格 ---
    df_prices = load_and_merge_predictions(PREDICTION_DIR, COIN_LIST)
    if df_prices is None:
        print("Failed to load prediction data. Exiting.")
        return

    # --- 3. 循环计算每个t+i的HF ---
    print("Calculating predicted Health Factors for t+1 to t+6...")
    
    # 最终结果的DataFrame
    df_results = pd.DataFrame(index=df_prices.index)

    for i in range(1, LOOKAHEAD_STEPS + 1):
        print(f"  Calculating for t+{i}...")
        
        # a. 提取该步的价格 (并转换为Decimal)
        try:
            p_weth = df_prices[f'WETH_pred_t_plus_{i}'].apply(Decimal)
            p_cbbtc = df_prices[f'cbBTC_pred_t_plus_{i}'].apply(Decimal)
            p_usdc = df_prices[f'USDC_pred_t_plus_{i}'].apply(Decimal)
        except KeyError as e:
            print(f"  [Error] Missing prediction column: {e}. Skipping t+{i}.")
            continue
            
        # b. 计算HF = Σ(Q_Collateral * P * LT) / Σ(Q_Debt * P)
        
        # 计算分子 (抵押品)
        numerator = (p_weth * portfolio['WETH']['amount'] * portfolio['WETH']['lt']) + \
                    (p_cbbtc * portfolio['cbBTC']['amount'] * portfolio['cbBTC']['lt'])
        
        # 计算分母 (债务)
        denominator = (p_usdc * portfolio['USDC']['amount'])
        
        # c. 计算HF (未调整)
        
        # 1. 将两个Series合并到一个临时DataFrame中
        df_temp = pd.DataFrame({'num': numerator, 'den': denominator})

        # 2. 定义一个安全的逐行除法函数
        def safe_decimal_divide(row):
            if row['den'] == Decimal(0):
                # 如果分母为0, HF 在 Aave 中被视为无穷大
                return Decimal('inf') 
            
            # --- (!!!) 修复! ---
            # 'Decimal' 对象使用标准除法运算符 '/'
            return row['num'] / row['den']
            # --- (!!!) 修复结束 ---

        # 3. 沿 axis=1 (逐行) 应用此函数
        hf_unadjusted = df_temp.apply(safe_decimal_divide, axis=1)
        
        # d. 计算HF (调整后)
        hf_adjusted = hf_unadjusted * GAF

        # e. 存入结果
        df_results[f'HF_unadjusted_t_plus_{i}'] = hf_unadjusted
        df_results[f'HF_adjusted_t_plus_{i}'] = hf_adjusted

    print("HF calculations complete.")
    
    # --- 4. 保存结果 ---
    
    # 将Decimal转换回float以便CSV标准存储
    for col in df_results.columns:
        df_results[col] = df_results[col].astype(float)
        
    # 重置索引，使 'datetime_utc' 成为一列
    df_results = df_results.reset_index()

    try:
        df_results.to_csv(OUTPUT_FILENAME, index=False, float_format='%.18f')
        print(f"\nSuccessfully saved all predicted HFs to:\n{OUTPUT_FILENAME}")
    except Exception as e:
        print(f"\n[Error] Failed to save final CSV: {e}")


# --- 脚本入口 ---
if __name__ == "__main__":
    main()