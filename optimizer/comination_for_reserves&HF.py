import pandas as pd
import os

# --- 1. 定义文件路径 ---
BASE_PATH = r"F:\Learning_journal_at_CUHK\FTEC5520_Appl Blockchain & Cryptocur\Simulation"

# 输入文件
HF_FILE = os.path.join(BASE_PATH, "HF", "data", "true_HF", "true_hf_adjusted.csv")
PRICE_DIR = os.path.join(BASE_PATH, "HF", "data", "true_data_in_usd")
CBTC_FILE = os.path.join(PRICE_DIR, "cbBTC_in_usd.csv")
USDC_FILE = os.path.join(PRICE_DIR, "USDC_in_usd.csv")
WETH_FILE = os.path.join(PRICE_DIR, "WETH_in_usd.csv")

# 输出目录和文件
OUTPUT_DIR = os.path.join(BASE_PATH, "optimizer", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "merged_hf_and_prices.csv")


def main():
    """
    主函数：加载、合并和保存数据。
    """
    print("Starting data merging process...")

    try:
        # --- 2. 确保输出目录存在 ---
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Ensured output directory exists: {OUTPUT_DIR}")

        # --- 3. 加载所有CSV文件 ---
        print(f"Loading HF data from: {HF_FILE}")
        df_hf = pd.read_csv(HF_FILE)
        
        print(f"Loading cbBTC data from: {CBTC_FILE}")
        df_cbtc = pd.read_csv(CBTC_FILE)
        
        print(f"Loading USDC data from: {USDC_FILE}")
        df_usdc = pd.read_csv(USDC_FILE)
        
        print(f"Loading WETH data from: {WETH_FILE}")
        df_weth = pd.read_csv(WETH_FILE)
        
        print("All files loaded successfully.")

        # --- 4. 预处理 ---
        # 重命名价格列以避免合并时冲突
        df_cbtc = df_cbtc.rename(columns={'price_usd': 'cbBTC_price_usd'})
        df_usdc = df_usdc.rename(columns={'price_usd': 'USDC_price_usd'})
        df_weth = df_weth.rename(columns={'price_usd': 'WETH_price_usd'})
        
        # 确保 'datetime_utc' 列是 datetime 对象，以便正确合并
        df_hf['datetime_utc'] = pd.to_datetime(df_hf['datetime_utc'])
        df_cbtc['datetime_utc'] = pd.to_datetime(df_cbtc['datetime_utc'])
        df_usdc['datetime_utc'] = pd.to_datetime(df_usdc['datetime_utc'])
        df_weth['datetime_utc'] = pd.to_datetime(df_weth['datetime_utc'])

        # --- 5. 合并数据 ---
        print("Merging dataframes on 'datetime_utc'...")
        
        # 从 HF 数据开始
        df_merged = df_hf
        
        # 逐个合并价格
        # 使用 'inner' join 来确保我们只保留所有文件中共有的时间戳
        df_merged = pd.merge(df_merged, df_cbtc, on='datetime_utc', how='inner')
        df_merged = pd.merge(df_merged, df_usdc, on='datetime_utc', how='inner')
        df_merged = pd.merge(df_merged, df_weth, on='datetime_utc', how='inner')
        
        print(f"Merging complete. Final dataframe has {len(df_merged)} rows.")
        
        # --- 6. 保存到CSV ---
        df_merged.to_csv(OUTPUT_FILE, index=False, float_format='%.18f')
        print(f"\nSuccessfully saved merged data to:\n{OUTPUT_FILE}")

    except FileNotFoundError as e:
        print(f"\n[Error] File not found. Please check your paths.")
        print(f"Missing file: {e.filename}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

# --- 脚本入口 ---
if __name__ == "__main__":
    main()