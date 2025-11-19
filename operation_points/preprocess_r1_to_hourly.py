import pandas as pd
import os

# --- 1. 定义文件路径 ---
BASE_PATH = r"F:\Learning_journal_at_CUHK\FTEC5520_Appl Blockchain & Cryptocur\Simulation\operation_points\data"

# (输入1) 包含目标时间戳的主文件
HF_FILE = os.path.join(BASE_PATH, "weighted_hf_t1_t3.csv")

# (输入2) 需要被清理和对齐的原始 R1 文件
R1_RAW_FILE = os.path.join(BASE_PATH, "r1_simulation_results.csv")

# (输出) 清理和对齐后的新 R1 文件
OUTPUT_FILE = os.path.join(BASE_PATH, "r1_hourly_aligned.csv")

# 定义 R1 文件中的列名
R1_COLUMN_NAME_RAW = 'R1_simulated'
R1_COLUMN_NAME_FINAL = 'R1' # 与 judge_phase.py 脚本中的期望一致

def main():
    """
    主执行函数：加载 R1，修复时间，降采样，
    然后加载 HF，对齐时间，最后保存。
    """
    print("--- R1 数据预处理脚本 ---")
    
    # --- 1. 加载 HF 主文件 (获取时间戳) ---
    try:
        df_hf_master = pd.read_csv(HF_FILE, usecols=['datetime_utc'])
        df_hf_master['datetime_utc'] = pd.to_datetime(df_hf_master['datetime_utc'])
        print(f"已加载 {len(df_hf_master)} 条目标时间戳 (来自 {HF_FILE})")
    except FileNotFoundError:
        print(f"  [Error] 文件未找到: {HF_FILE}")
        return
    except Exception as e:
        print(f"  [Error] 加载 HF 文件失败: {e}")
        return

    # --- 2. 加载原始 R1 文件 (修复时间错误) ---
    try:
        df_r1_raw = pd.read_csv(R1_RAW_FILE)
        
        # (!!!) 修复 1: 使用 format='ISO8601' 处理不一致的时间戳
        df_r1_raw['datetime_utc'] = pd.to_datetime(df_r1_raw['datetime_utc'], format='ISO8601')
        print(f"已加载 {len(df_r1_raw)} 条原始 R1 记录 (来自 {R1_RAW_FILE})")
    except FileNotFoundError:
        print(f"  [Error] 文件未找到: {R1_RAW_FILE}")
        return
    except Exception as e:
        print(f"  [Error] 加载 R1 文件失败: {e}")
        return
        
    # 检查 R1 列是否存在
    if R1_COLUMN_NAME_RAW not in df_r1_raw.columns:
         print(f"  [Error] 在 R1 文件中未找到列: '{R1_COLUMN_NAME_RAW}'。")
         return

    # --- 3. 降采样 R1 数据至小时粒度 ---
    # (!!!) 修复 2: 将时间戳向下取整到小时
    df_r1_raw['hourly_timestamp'] = df_r1_raw['datetime_utc'].dt.floor('h')
    
    # 保留每个小时的最后一条记录
    df_r1_hourly_map = df_r1_raw.drop_duplicates(subset='hourly_timestamp', keep='last').copy()
    
    # 重命名为您在 judge_phase.py 中使用的列名
    df_r1_hourly_map = df_r1_hourly_map.rename(columns={R1_COLUMN_NAME_RAW: R1_COLUMN_NAME_FINAL})
    
    # 只保留用于映射的列
    df_r1_hourly_map = df_r1_hourly_map[['hourly_timestamp', R1_COLUMN_NAME_FINAL]]
    print(f"已将 R1 数据降采样至 {len(df_r1_hourly_map)} 条小时记录。")

    # --- 4. 合并 (对齐时间范围) ---
    # (!!!) 修复 3: 使用 'left' 合并, 确保我们只保留 HF 文件中存在的时间戳
    print(f"正在将 R1 数据与 {len(df_hf_master)} 条目标时间戳对齐...")
    df_final = pd.merge(
        df_hf_master,            # (左) 包含目标时间戳
        df_r1_hourly_map,        # (右) R1 的小时地图
        left_on='datetime_utc',  # HF 文件中的小时时间戳
        right_on='hourly_timestamp', # R1 文件中的小时时间戳
        how='left'
    )
    
    # --- 5. 清理合并后的数据 ---
    df_final = df_final.drop(columns=['hourly_timestamp']) # 移除多余的时间戳列
    
    # 检查 R1 数据是否完全覆盖了 HF 时间范围
    missing_r1 = df_final[R1_COLUMN_NAME_FINAL].isna().sum()
    if missing_r1 > 0:
        print(f"  [Warning] 在合并后, 有 {missing_r1} 个时间点缺少 R1 数据。")
        # 假设 R1 是一个缓慢变化的宏观指标，使用前向填充 (ffill)
        print("  ...正在使用前向填充 (forward-fill) 填充缺失值。")
        df_final[R1_COLUMN_NAME_FINAL] = df_final[R1_COLUMN_NAME_FINAL].ffill()
        
        # 填充后再次检查
        missing_r1_after = df_final[R1_COLUMN_NAME_FINAL].isna().sum()
        if missing_r1_after > 0:
            print(f"  [Warning] 仍有 {missing_r1_after} 个缺失值 (可能在数据开头), 正在丢弃这些行。")
            df_final = df_final.dropna()

    if df_final.empty:
        print("  [Error] 合并后数据为空。请检查两个文件的时间范围。")
        return

    # --- 6. 保存新的 R1 CSV 文件 ---
    try:
        df_final.to_csv(OUTPUT_FILE, index=False, float_format='%.18f')
        print(f"\n--- 成功 ---")
        print(f"已创建对齐的小时 R1 文件于:\n{OUTPUT_FILE}")
        print(f"\n下一步: 请修改 'judge_phase.py' 脚本,")
        print(f"使其从 '{os.path.basename(OUTPUT_FILE)}' (而不是 '{os.path.basename(R1_RAW_FILE)}') 加载 R1 数据。")
    except Exception as e:
        print(f"\n[Error] 保存输出文件失败: {e}")

# --- 脚本入口 ---
if __name__ == "__main__":
    main()