import pandas as pd
import os

# --- 1. 定义文件路径 ---
BASE_PATH = r"F:\Learning_journal_at_CUHK\FTEC5520_Appl Blockchain & Cryptocur\Simulation\operation_points\data"

INPUT_FILE = os.path.join(BASE_PATH, "predicted_hf_t1_to_t6.csv")
OUTPUT_FILE = os.path.join(BASE_PATH, "weighted_hf_t1_t3.csv") # 新文件名

# --- 2. 定义权重 ---
WEIGHTS = {
    'HF_adjusted_t_plus_1': 0.6,
    'HF_adjusted_t_plus_2': 0.3,
    'HF_adjusted_t_plus_3': 0.1
}

# 需要的列
COLUMNS_TO_LOAD = ['datetime_utc', 'HF_adjusted_t_plus_1', 'HF_adjusted_t_plus_2', 'HF_adjusted_t_plus_3']

def main():
    """
    主执行函数：加载数据，计算加权HF，并保存。
    """
    print(f"Loading data from: {INPUT_FILE}")
    
    try:
        # --- 3. 加载数据 ---
        # 只加载我们需要的列，以节省内存
        df = pd.read_csv(INPUT_FILE, usecols=COLUMNS_TO_LOAD)
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
        print(f"Successfully loaded {len(df)} records.")

    except FileNotFoundError:
        print(f"  [Error] File not found: {INPUT_FILE}")
        return
    except ValueError as e:
        print(f"  [Error] Missing one of the required columns in the CSV.")
        print(f"  Details: {e}")
        return
    except Exception as e:
        print(f"  [Error] Failed to load data: {e}")
        return

    # --- 4. 计算加权平均HF ---
    print("Calculating weighted average HF indicator...")
    
    # 初始化新列
    df['Weighted_HF'] = 0.0
    
    # 逐列乘以权重并累加
    for col, weight in WEIGHTS.items():
        df['Weighted_HF'] += df[col] * weight
        
    print("Calculation complete.")

    # --- 5. 保存结果 ---
    # 只保存时间和新的指标列
    df_output = df[['datetime_utc', 'Weighted_HF']]
    
    try:
        df_output.to_csv(OUTPUT_FILE, index=False, float_format='%.18f')
        print(f"\nSuccessfully saved weighted HF indicator to:\n{OUTPUT_FILE}")
    except Exception as e:
        print(f"\n[Error] Failed to save output file: {e}")


# --- 脚本入口 ---
if __name__ == "__main__":
    main()