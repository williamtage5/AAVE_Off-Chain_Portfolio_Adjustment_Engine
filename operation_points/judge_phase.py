import pandas as pd
import os
import numpy as np

# --- 1. 定义常量和路径 ---
BASE_PATH = r"F:\Learning_journal_at_CUHK\FTEC5520_Appl Blockchain & Cryptocur\Simulation\operation_points\data"

# 输入文件
HF_FILE = os.path.join(BASE_PATH, "weighted_hf_t1_t3.csv")
R1_FILE = os.path.join(BASE_PATH, "r1_hourly_aligned.csv")

# 输出文件
OUTPUT_FILE = os.path.join(BASE_PATH, "risk_classification_per_timestamp.csv")

# --- 2. 定义风险阈值 ---
# 固定阈值
R1_THRESHOLD = 0.63
LIQUIDATION_LIMIT = 1.0

# 动态分位数阈值
HF_STATIC_QUANTILE = 0.30  # (q30) 静态水平阈值
HF_VELOCITY_QUANTILE = 0.20 # (q20) 动态速度阈值 (捕捉下降)

# R1 文件中的列名
R1_COLUMN_NAME = 'R1' 


def classify_risk(row, hf_static_thresh, hf_vel_thresh, r1_thresh, liq_limit):
    """
    根据新的三维风险模型（R1, HF水平, HF速度）应用风险分类。
    """
    # 1. 定义所有风险条件
    hf_is_low = row['Weighted_HF'] < hf_static_thresh
    hf_is_dropping = row['HF_pct_change'] < hf_vel_thresh
    r1_is_high = row[R1_COLUMN_NAME] > r1_thresh
    hf_is_liquidated = row['Weighted_HF'] < liq_limit

    # 2. 按优先级分配风险
    
    # 🔴 高风险 (High Risk)
    if hf_is_liquidated:
        quadrant = "High_Personal_Risk_Liquidated (HF < 1.0)"
        level = "High Risk"
    elif r1_is_high and hf_is_dropping:
        quadrant = "High_System_Risk_AND_HF_Dropping"
        level = "High Risk"
        
    # 🟠 中风险 (Medium Risk)
    elif r1_is_high:
        quadrant = "High_System_Risk_Only"
        level = "Medium Risk"
    elif hf_is_dropping:
        quadrant = "HF_Dropping_Only"
        level = "Medium Risk"
    elif hf_is_low:
        quadrant = "Low_Static_HF_Only"
        level = "Medium Risk"
        
    # 🟢 低风险 (Low Risk)
    else:
        quadrant = "Low_Risk_All_Stable"
        level = "Low Risk"
        
    return pd.Series([quadrant, level])

def main():
    """
    主执行函数：加载、合并、计算指标和阈值、分类和保存。
    """
    print("--- 风险象限划分脚本 (动态三维模型) ---")
    
    # --- 1. 加载数据 ---
    try:
        print(f"Loading Weighted HF data from: {HF_FILE}")
        df_hf = pd.read_csv(HF_FILE)
        df_hf['datetime_utc'] = pd.to_datetime(df_hf['datetime_utc'])
    except FileNotFoundError:
        print(f"  [Error] 文件未找到: {HF_FILE}")
        return
    except Exception as e:
        print(f"  [Error] 加载 HF 文件失败: {e}")
        return

    try:
        print(f"Loading CLEANED R1 data from: {R1_FILE}")
        df_r1 = pd.read_csv(R1_FILE)
        df_r1['datetime_utc'] = pd.to_datetime(df_r1['datetime_utc'])
    except FileNotFoundError:
        print(f"  [Error] 文件未找到: {R1_FILE}")
        return
    except Exception as e:
        print(f"  [Error] 加载 R1 文件失败: {e}")
        return
        
    # --- 2. 合并数据 ---
    print("Merging HF and R1 data on 'datetime_utc'...")
    df_merged = pd.merge(df_hf, df_r1, on='datetime_utc', how='inner')
    
    if df_merged.empty:
        print("  [Error] 合并后数据为空。")
        return
    if R1_COLUMN_NAME not in df_merged.columns:
        print(f"  [Error] R1 文件中未找到列: '{R1_COLUMN_NAME}'。")
        return
        
    print(f"合并成功。共有 {len(df_merged)} 个重叠的时间点。")

    # --- 3. (!!!) 新增: 计算动态指标和阈值 ---
    
    # a. 计算 HF 1小时百分比变化
    df_merged['HF_pct_change'] = df_merged['Weighted_HF'].pct_change()
    # 填充第一个 NaN 值为 0 (代表无变化)
    df_merged['HF_pct_change'] = df_merged['HF_pct_change'].fillna(0.0)

    # b. 计算动态阈值
    HF_STATIC_THRESHOLD = df_merged['Weighted_HF'].quantile(HF_STATIC_QUANTILE)
    HF_VELOCITY_THRESHOLD = df_merged['HF_pct_change'].quantile(HF_VELOCITY_QUANTILE)
    
    print("\n--- 风险阈值定义 (动态模型) ---")
    print(f"系统风险 (R1) 阈值: > {R1_THRESHOLD}")
    print(f"绝对清算 (HF) 阈值: < {LIQUIDATION_LIMIT}")
    print(f"静态HF (q{int(HF_STATIC_QUANTILE*100)}) 阈值: < {HF_STATIC_THRESHOLD:.6f}")
    print(f"动态HF (q{int(HF_VELOCITY_QUANTILE*100)}) 阈值: < {HF_VELOCITY_THRESHOLD:.6f} (即下降超过 {abs(HF_VELOCITY_THRESHOLD*100):.3f}%)")

    # --- 4. 应用分类 ---
    print("\nApplying classification to each timestamp...")
    
    df_merged[['Quadrant', 'Risk_Level']] = df_merged.apply(
        classify_risk, 
        axis=1, 
        hf_static_thresh=HF_STATIC_THRESHOLD, 
        hf_vel_thresh=HF_VELOCITY_THRESHOLD,
        r1_thresh=R1_THRESHOLD,
        liq_limit=LIQUIDATION_LIMIT
    )
    
    print("Classification complete.")
    
    # 检查风险事件的分布
    print(f"\n--- 结果: 风险等级分布 ---")
    risk_distribution = df_merged['Risk_Level'].value_counts()
    print(risk_distribution)
    
    if "High Risk" not in risk_distribution:
        print("\n(提示: 仍未找到高风险事件。这表明系统脆弱 (R1 > 0.63) 时，")
        print(" HF 从未同时经历快速下降 (低于 q20)。这是一个真实的数据发现。)")


    # --- 5. 保存结果 ---
    columns_order = [
        'datetime_utc', 
        'Risk_Level', 
        'Quadrant', 
        'Weighted_HF', 
        'HF_pct_change',
        R1_COLUMN_NAME
    ]
    other_cols = [col for col in df_merged.columns if col not in columns_order]
    final_df = df_merged[columns_order + other_cols]
    
    try:
        final_df.to_csv(OUTPUT_FILE, index=False, float_format='%.18f')
        print(f"\nSuccessfully saved risk classification to:\n{OUTPUT_FILE}")
    except Exception as e:
        print(f"\n[Error] Failed to save output file: {e}")


# --- 脚本入口 ---
if __name__ == "__main__":
    main()