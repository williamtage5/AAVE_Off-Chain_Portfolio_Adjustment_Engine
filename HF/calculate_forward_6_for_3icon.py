import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# --- 全局配置: 预测未来 N 步 ---
LOOKAHEAD_STEPS = 6

# 确保 PyTorch 使用 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 特征工程函数 (from Cell 3) ---
# 为防止RSI计算中除以零，添加了一个小的 epsilon
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    RS = gain / (loss + 1e-10)  # 添加 epsilon 防止除以零
    return 100 - (100 / (1 + RS))

def feature_engineering(df):
    """根据 'Close' 列计算所有技术指标"""
    # 您的基础代码在外部处理时间，这里直接使用
    df_feat = df[['Open Time', 'Close']].copy()
    
    # 确保'Close'是数值型
    df_feat['Close'] = pd.to_numeric(df_feat['Close'])
    
    df_feat['Return'] = df_feat['Close'].pct_change()
    
    # MA 均线
    df_feat['MA5'] = df_feat['Close'].rolling(window=5).mean()
    df_feat['MA10'] = df_feat['Close'].rolling(window=10).mean()
    df_feat['MA20'] = df_feat['Close'].rolling(window=20).mean()
    
    # RSI
    df_feat['RSI14'] = compute_RSI(df_feat['Close'], 14)
    
    # MACD
    ema12 = df_feat['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df_feat['Close'].ewm(span=26, adjust=False).mean()
    df_feat['MACD'] = ema12 - ema26
    df_feat['Signal'] = df_feat['MACD'].ewm(span=9, adjust=False).mean()
    df_feat['MACD_Hist'] = df_feat['MACD'] - df_feat['Signal']
    
    # Bollinger Bands
    window_bb = 20
    std = df_feat['Close'].rolling(window_bb).std()
    df_feat['BB_Mid'] = df_feat['Close'].rolling(window_bb).mean()
    df_feat['BB_Upper'] = df_feat['BB_Mid'] + 2 * std
    df_feat['BB_Lower'] = df_feat['BB_Mid'] - 2 * std
    
    # 删除指标计算产生的 NaN
    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat

# --- (!!!) 滑动窗口函数 (已修改) ---
def create_dataset(dataset, window=5, lookahead_steps=1):
    """
    创建X（特征）和y（标签）
    y 现在是 [t+1, t+2, ..., t+lookahead_steps] 的价格
    """
    X, y = [], []
    # 调整循环范围以适应 lookahead
    # (i + window + lookahead_steps - 1) 必须是有效索引
    for i in range(len(dataset) - window - lookahead_steps + 1):
        X.append(dataset[i:i+window, :])  # 所有特征
        
        # y 的目标是 [t+1, ..., t+6] 的 'Close' 价格
        # 'Close' 价格在第 0 列
        y.append(dataset[i + window : i + window + lookahead_steps, 0])
    return np.array(X), np.array(y)

# --- (!!!) LSTM 模型定义 (已修改) ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            batch_first=True, 
                            dropout=dropout, 
                            num_layers=num_layers)
        # (!!!) 输出层现在预测 'output_size' (即 6) 个值
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # out[:, -1, :] 得到最后一个时间步的隐藏状态
        return self.fc(out[:, -1, :])

# --- 权重初始化 (from Cell 6) ---
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# --- (!!!) 新增：反归一化辅助函数 ---
def inverse_transform_column(scaled_data, scaler, feature_count, target_col_index=0):
    """
    将一维的缩放数据（代表目标列）放回其原始位置
    并进行反归一化。
    """
    dummy = np.zeros((len(scaled_data), feature_count))
    dummy[:, target_col_index] = scaled_data.flatten()
    return scaler.inverse_transform(dummy)[:, target_col_index]

# --- 1. 路径配置 (已修改) ---
BASE_PATH = "F:/Learning_journal_at_CUHK/FTEC5520_Appl Blockchain & Cryptocur/Simulation/HF"
# (!!!) 使用新的 "long_range" 数据
INPUT_DIR = os.path.join(BASE_PATH, "data", "every_timestape_prediction", "long_range")
# (!!!) 新的输出目录
OUTPUT_DIR = os.path.join(BASE_PATH, "data", "every_timestape_prediction", "target_time_range_prediction")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. 币种列表 (已修改) ---
COIN_FILES = [
    "cbBTC_hourly_price_data.csv",
    "USDC_hourly_price_data.csv",
    "WETH_hourly_price_data.csv"
]

# --- 3. (!!!) 预测日期范围 (新) ---
# 我们将使用这个来分割 训练集/测试集
PREDICTION_START_DATE = "2025-10-21T17:00:00+00:00"
PREDICTION_END_DATE = "2025-10-30T12:00:00+00:00"

# --- 4. 特征列表 (from Cell 4) ---
# (!!!) 确保 'Close' 是第一个, 这对反归一化至关重要
FEATURES_LIST = [
    'Close', 'Return', 'MA5', 'MA10', 'MA20',
    'RSI14', 'MACD', 'Signal', 'MACD_Hist',
    'BB_Mid', 'BB_Upper', 'BB_Lower'
]
INPUT_SIZE = len(FEATURES_LIST)

# --- 5. 超参数调优 (键已更新) ---
hyperparameters = {
    "default": {
        "hidden_size": 64, "num_layers": 2, "dropout": 0.2,
        "epochs": 60, "lr": 1e-3, "window": 5, "batch_size": 32
    },
    "cbBTC": {
        "hidden_size": 128, "num_layers": 2, "dropout": 0.2,
        "epochs": 70, "lr": 1e-3, "window": 10, "batch_size": 32
    },
    "USDC": { # 稳定币
        "hidden_size": 32, "num_layers": 1, "dropout": 0.1,
        "epochs": 40, "lr": 5e-4, "window": 3, "batch_size": 16
    },
    "WETH": { # 使用默认值
        "hidden_size": 64, "num_layers": 2, "dropout": 0.2,
        "epochs": 60, "lr": 1e-3, "window": 5, "batch_size": 32
    }
}

print("配置加载完毕。")

for coin_file in COIN_FILES:
    # (!!!) 修改了名称提取以适应新文件名
    coin_name = coin_file.split('_')[0] 
    print(f"\n{'='*50}\nProcessing: {coin_name}\n{'='*50}")

    # --- 1. 加载参数 ---
    params = hyperparameters.get(coin_name, hyperparameters['default'])
    window = params['window']
    
    # --- 2. 加载与特征工程 ---
    try:
        file_path = os.path.join(INPUT_DIR, coin_file)
        data_df_raw = pd.read_csv(file_path)
        
        # 重命名列以匹配特征工程函数的期望
        data_df_raw = data_df_raw.rename(columns={'datetime_utc': 'Open Time', 'price_usd': 'Close'})
        
        # (!!!) 时间聚合 (与您之前的脚本一致)
        data_df_raw['Open Time'] = pd.to_datetime(data_df_raw['Open Time'], format='ISO8601')
        data_df_raw['Open Time'] = data_df_raw['Open Time'].dt.floor('h')
        data_df_raw = data_df_raw.drop_duplicates(subset='Open Time', keep='last')
        data_df_raw = data_df_raw.sort_values('Open Time').reset_index(drop=True)
        # --- 聚合结束 ---
        
        # 应用特征工程
        data_df_feat = feature_engineering(data_df_raw)
        
        if data_df_feat.empty or len(data_df_feat) < (window + LOOKAHEAD_STEPS + 1):
            print(f"  [Warning] No data remaining for {coin_name} after feature engineering. Skipping.")
            continue
            
        print(f"  Data loaded, aggregated, and features engineered. Total rows: {len(data_df_feat)}")

    except Exception as e:
        print(f"  [Error] Failed to load or process data for {coin_name}: {e}")
        continue

    # --- 3. (!!!) 预处理 (基于时间的分割) ---
    start_dt = pd.to_datetime(PREDICTION_START_DATE)
    
    # 找到分割点
    # 我们需要在 'start_dt' 之前至少有 'window' 个数据点来制作第一个预测
    # 训练集 = 'Open Time' < 'start_dt'
    train_split_index = data_df_feat[data_df_feat['Open Time'] < start_dt].index.max()
    
    # 确保我们有足够的训练数据
    if train_split_index is np.nan or train_split_index < (window + LOOKAHEAD_STEPS):
         print(f"  [Error] Not enough training data before {PREDICTION_START_DATE}. Skipping.")
         continue

    # 训练数据 = 从开始到分割点
    train_data = data_df_feat.loc[:train_split_index, FEATURES_LIST].values
    
    # 测试数据 = 从 (分割点 - 窗口 + 1) 开始到结束
    # 我们需要 'window' 的重叠来创建第一个 'X_test'
    test_data_start_index = train_split_index - window + 1
    test_data = data_df_feat.loc[test_data_start_index:, FEATURES_LIST].values
    
    # 保留测试集的时间戳用于最终输出
    # 预测 't' 的时间戳是窗口的最后一个点
    # X_test[0] 窗口是 [test_data_start_index ... train_split_index]
    # 因此, 第一个预测的 't' 是在 'train_split_index' (即 < start_dt 的最后一行)
    # 这意味着第一个 t+1 预测是针对 'start_dt' 的
    
    # X_test[0] -> test_data[0 : window] -> 对应 data_df_feat[test_data_start_index : train_split_index + 1]
    # 这个窗口的最后一个时间戳是 data_df_feat['Open Time'].iloc[train_split_index]
    # 这个窗口的预测 (t+1..t+6) 是针对 [train_split_index+1 ... train_split_index+6]
    
    # 让我们获取 X_test 对应的所有 't' 时间戳
    # create_dataset 将从 test_data[0] 开始
    # X_test[0] -> 窗口在 i=0
    # X_test[1] -> 窗口在 i=1
    # ...
    # X_test 的 't' 时间戳 (窗口的最后一行)
    test_event_timestamps = data_df_feat['Open Time'].iloc[test_data_start_index + window - 1 : -LOOKAHEAD_STEPS].values

    if len(test_data) < window + LOOKAHEAD_STEPS:
        print(f"  [Warning] Test data for {coin_name} is too small. Skipping.")
        continue

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = create_dataset(train_scaled, window, lookahead_steps=LOOKAHEAD_STEPS)
    X_test, y_test = create_dataset(test_scaled, window, lookahead_steps=LOOKAHEAD_STEPS)
    
    if len(X_test) == 0:
        print(f"  [Warning] No test samples created for {coin_name}. Skipping.")
        continue

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    # y 的形状现在是 (N, 6)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=params['batch_size'], 
                              shuffle=True)
    
    print(f"  Train shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"  Test shapes: X={X_test.shape}, y={y_test.shape}")

    # --- 4. (!!!) 模型初始化 (已修改) ---
    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        dropout=params['dropout'],
        output_size=LOOKAHEAD_STEPS  # (!!!) 告知模型输出6个值
    ).to(device)
    
    model.apply(init_weights)
    criterion = nn.MSELoss() # MSE 可以直接处理 (N, 6) vs (N, 6) 的比较
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)

    # --- 5. 训练 ---
    print(f"  Starting training for {params['epochs']} epochs...")
    epochs = params['epochs']
    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(Xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if epochs > 0 and ((epoch + 1) % max(1, (epochs // 5)) == 0 or epoch == epochs - 1):
            print(f"  ...Epoch {epoch+1}/{epochs}, Loss = {loss.item():.6f}")

    # --- 6. (!!!) 预测与反归一化 (新逻辑) ---
    print("  Training complete. Generating predictions for test set...")
    model.eval()
    
    with torch.no_grad():
        y_pred_scaled = model(X_test.to(device)).cpu().numpy()

    # (!!!) 确保预测和时间戳的长度一致
    # len(X_test) == len(y_pred_scaled) == len(test_event_timestamps)
    if len(y_pred_scaled) != len(test_event_timestamps):
        min_len = min(len(y_pred_scaled), len(test_event_timestamps))
        print(f"  [Warning] Mismatch in prediction/timestamp length. Trimming to {min_len}.")
        y_pred_scaled = y_pred_scaled[:min_len]
        test_event_timestamps = test_event_timestamps[:min_len]
        
    results_df = pd.DataFrame({'datetime_utc': test_event_timestamps})
    metrics = {}

    y_test_numpy = y_test.numpy() # (N, 6)

    # 循环 t+1 到 t+6
    for i in range(LOOKAHEAD_STEPS):
        step = i + 1
        
        # 反归一化预测值
        pred_scaled_col = y_pred_scaled[:, i]
        pred_actual_col = inverse_transform_column(pred_scaled_col, scaler, INPUT_SIZE, 0)
        
        # 反归一化真实值 (用于评估)
        true_scaled_col = y_test_numpy[:, i]
        true_actual_col = inverse_transform_column(true_scaled_col, scaler, INPUT_SIZE, 0)

        # 添加到 DataFrame
        results_df[f'predicted_price_t_plus_{step}'] = pred_actual_col
        # results_df[f'actual_price_t_plus_{step}'] = true_actual_col # (可选, 用于验证)

        # 计算评估指标
        try:
            metrics[f'rmse_t{step}'] = np.sqrt(mean_squared_error(true_actual_col, pred_actual_col))
            metrics[f'r2_t{step}'] = r2_score(true_actual_col, pred_actual_col)
        except Exception:
            pass # 可能会失败 (例如 R2)

    print(f"  Prediction Metrics ({coin_name}):")
    print("    " + ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items() if 'rmse' in k]))

    # --- 7. (!!!) 过滤并保存结果 ---
    
    # 将结果中的 'datetime_utc' 设为带时区的 (它们是 naive 的)
    # data_df_feat['Open Time'] 是带时区的
    # test_event_timestamps 是从 .values 中提取的, 所以是 naive
    results_df['datetime_utc'] = pd.to_datetime(results_df['datetime_utc']).dt.tz_localize('UTC')

    # 将筛选日期也设置为带时区的
    filter_start_dt = pd.to_datetime(PREDICTION_START_DATE)
    filter_end_dt = pd.to_datetime(PREDICTION_END_DATE)

    # 筛选 (现在是 'aware' vs 'aware' 比较)
    filtered_results_df = results_df[
        (results_df['datetime_utc'] >= filter_start_dt) & 
        (results_df['datetime_utc'] <= filter_end_dt)
    ].copy()

    # 保存到 CSV
    output_filename = os.path.join(OUTPUT_DIR, f"{coin_name}_target_range_prediction.csv")
    filtered_results_df.to_csv(output_filename, index=False, float_format='%.8f')
    
    print(f"  Successfully saved {len(filtered_results_df)} predictions to {output_filename}")

print(f"\n{'='*50}\nAll processing complete.\n{'='*50}")