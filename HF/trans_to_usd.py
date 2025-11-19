import pandas as pd
import json
import os

# --- 1. 定义绝对路径 ---
# 基础路径 (根据您的描述)
BASE_PROJECT_PATH = r"F:\Learning_journal_at_CUHK\FTEC5520_Appl Blockchain & Cryptocur\Simulation"

# --- 这是修复! ---
# 输入文件1: 路径已更正，添加了 "true_data_in_usd"
FLUCTUATION_JSON_PATH = os.path.join(BASE_PROJECT_PATH, "HF", "data", "true_data_in_usd", "fluctuation.json")

# 输入文件2: ETH的美元价格CSV
ETH_PRICE_FILE_PATH = os.path.join(BASE_PROJECT_PATH, "data", "coin", "USD", "---ETH_hourly_price_data.csv")

# 输出目录: 存放转换后的USD计价CSV
OUTPUT_DIR = os.path.join(BASE_PROJECT_PATH, "HF", "data", "true_data_in_usd")


def load_eth_price_map(eth_csv_path: str) -> dict:
    """
    加载ETH/USD价格数据，并将其处理为每小时一个价格的查找字典。
    它将时间戳向下取整到小时，并保留该小时的最后一条价格记录。
    """
    print(f"Loading ETH/USD price data from {eth_csv_path}...")
    try:
        df = pd.read_csv(eth_csv_path)
        
        # 转换时间戳为datetime对象
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], format='ISO8601')
        
        # --- 这是修复! ---
        # 将时间戳向下取整到最近的小时 (H -> h 修复 FutureWarning)
        df['hourly_timestamp'] = df['datetime_utc'].dt.floor('h')
        
        # 对于每个小时，我们只保留最后一条记录（以防一小时内有多个价格点）
        df = df.drop_duplicates(subset='hourly_timestamp', keep='last')
        
        # 创建一个字典以便快速查找: {pd.Timestamp: price_usd}
        price_map = pd.Series(df['price_usd'].values, index=df['hourly_timestamp']).to_dict()
        
        print(f"Successfully created ETH/USD price map with {len(price_map)} entries.")
        return price_map
        
    except FileNotFoundError:
        print(f"Error: ETH price file not found at {eth_csv_path}")
        return {}
    except Exception as e:
        print(f"Error loading ETH price data: {e}")
        return {}

def process_and_save_data(json_path: str, eth_price_map: dict, output_dir: str):
    """
    处理fluctuation.json文件，将其从ETH计价转换为USD计价，并保存三个CSV文件。
    """
    print(f"Loading fluctuation data from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            fluctuation_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Fluctuation JSON file not found at {json_path}")
        print("Please check the path. Is the file really there?")
        return
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensured output directory exists: {output_dir}")

    # 准备列表来存储转换后的数据
    weth_usd_records = []
    cbbtc_usd_records = []
    usdc_usd_records = []

    missing_timestamps = 0

    # 遍历JSON中的每条记录
    for record in fluctuation_data:
        # 获取记录的时间戳 (已经是每小时整点)
        record_timestamp = pd.to_datetime(record['datetime_utc'])
        
        # 从我们的map中查找对应的ETH/USD价格
        # (map中的键已经是被floor('h')处理过的小时)
        eth_usd_price = eth_price_map.get(record_timestamp)
        
        if eth_usd_price is None:
            # print(f"Warning: No ETH/USD price found for {record_timestamp}. Skipping this entry.")
            missing_timestamps += 1
            continue
            
        # 获取以ETH计价的价格
        prices_eth = record['price_details_eth']
        
        # 计算USD价格
        weth_in_usd = prices_eth['WETH'] * eth_usd_price
        cbbtc_in_usd = prices_eth['cbBTC'] * eth_usd_price
        usdc_in_usd = prices_eth['USDC'] * eth_usd_price
        
        # 添加到各自的列表中
        weth_usd_records.append({'datetime_utc': record_timestamp, 'price_usd': weth_in_usd})
        cbbtc_usd_records.append({'datetime_utc': record_timestamp, 'price_usd': cbbtc_in_usd})
        usdc_usd_records.append({'datetime_utc': record_timestamp, 'price_usd': usdc_in_usd})

    if missing_timestamps > 0:
        print(f"Warning: Skipped {missing_timestamps} records due to missing ETH/USD price data at that hour.")

    # --- 3. 保存为DataFrame并导出到CSV ---
    try:
        # WETH
        df_weth = pd.DataFrame(weth_usd_records)
        weth_output_path = os.path.join(output_dir, "WETH_in_usd.csv")
        df_weth.to_csv(weth_output_path, index=False)
        print(f"Successfully saved WETH USD data to {weth_output_path}")

        # cbBTC
        df_cbbtc = pd.DataFrame(cbbtc_usd_records)
        cbbtc_output_path = os.path.join(output_dir, "cbBTC_in_usd.csv")
        df_cbbtc.to_csv(cbbtc_output_path, index=False)
        print(f"Successfully saved cbBTC USD data to {cbbtc_output_path}")

        # USDC
        df_usdc = pd.DataFrame(usdc_usd_records)
        usdc_output_path = os.path.join(output_dir, "USDC_in_usd.csv")
        df_usdc.to_csv(usdc_output_path, index=False)
        print(f"Successfully saved USDC USD data to {usdc_output_path}")
        
        print("\nAll files processed and saved successfully.")

    except Exception as e:
        print(f"Error saving CSV files: {e}")


def main():
    """
    主执行函数
    """
    # 1. 加载ETH价格数据并创建查找表
    eth_price_map = load_eth_price_map(ETH_PRICE_FILE_PATH)
    
    if not eth_price_map:
        print("Could not load ETH price map. Exiting.")
        return

    # 2. 处理数据并保存
    process_and_save_data(FLUCTUATION_JSON_PATH, eth_price_map, OUTPUT_DIR)


# --- 脚本入口 ---
if __name__ == "__main__":
    main()