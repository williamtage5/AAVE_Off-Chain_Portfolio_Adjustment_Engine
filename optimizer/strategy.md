好的，我完全理解。您的新要求——“如果（中风险）策略A达不到目的，就升级到（高风险）策略B（运筹优化）”——是一个非常智能的升级。这为“中风险”策略增加了一个明确的**“成功/失败”标准**。

我们现在将这个“升级（Escalation）”逻辑和**明确的数值计算**（用于“钱包”和“操作”）整合到您的最终方案中。

---

### 方案名称：三维风险-行动模拟框架 (v2 - 带升级逻辑)

### 1. 核心目标 (不变)
模拟一个主动的、基于风险状态转变的调仓策略。

### 2. 核心数据源 (不变)
* **仓位与配置 (`outline.json`):** `Q_Collaterals`, `Q_Debts`, `LT`, `GAF`。
* **价格与真实HF (`merged_hf_and_prices.csv`):** `P_t` (价格), `P_t-1` (用于成本), `HF_adjusted` (基准HF)。
* **系统风险 (`r1_hourly_aligned.csv`):** `R1`。
* **行动触发点 (`risk_escalation_points.csv`):** 触发模拟的时间点 `t`。

### 3. 策略支柱一：风险等级划分 (不变)

风险等级（高、中、低）的定义不变，它基于 R1、HF 静态水平 (q30) 和 HF 动态速度 (q20) 来确定。**此等级仅用于“触发”**。

* **`HF_STATIC_THRESH`**: `HF_adjusted` 序列的 **30分位数 (q30)**。
* **`R1_THRESH`**: `0.63`

### 4. 策略支柱二：行动触发器 (不变)

调仓动作（Action）**仅**在风险等级**恶化**时触发。

* **中风险触发:** `Low Risk` (t-1) ➔ `Medium Risk` (t)。
* **高风险触发:** `Low Risk` (t-1) 或 `Medium Risk` (t-1) ➔ `High Risk` (t)。

---

### 5. 策略支柱三：预备金（钱包）模拟

这是对您“模拟几种情况”请求的明确数值计算。在**任何**触发点（中或高），我们都将**分别**运行这三种情景。

**A. 计算基础（在触发点 `t`）:**
* `Total_Debt_Value = Q_Debt_USDC * P_t_USDC`
    * *（来自 `outline.json` 和 `merged_hf_and_prices.csv`）*

**B. 三种钱包情景 (Scenarios):**
1.  **情景 A：“保守型” (10% 预备金)**
    * **`Wallet_Value (USD) = 0.10 * Total_Debt_Value`**
    * *（例如：`0.10 * 2040.84 = 204.08 USD`）*
2.  **情景 B：“平衡型” (50% 预备金)**
    * **`Wallet_Value (USD) = 0.50 * Total_Debt_Value`**
    * *（例如：`0.50 * 2040.84 = 1020.42 USD`）*
3.  **情景 C：“激进型” (100% 预备金)**
    * **`Wallet_Value (USD) = 1.00 * Total_Debt_Value`**
    * *（例如：`1.00 * 2040.84 = 2040.84 USD`）*

---

### 6. 策略支柱四：调仓行动模型 (已更新升级逻辑)

这是对您策略的核心完善。

#### A. 针对“中风险触发” (Low ➔ Medium)
* **模型:** **“模拟对比” (Comparative Simulation)**
* **目标 (Success Target):** 行动后的 `HF_after` 必须**大于** `HF_STATIC_THRESH` (q30)。如果达不到，则策略“失败”。
* **流程:** 在 `t` 时刻，针对**每种钱包情景** (A, B, C) 分别执行：

    1.  **获取 `Wallet_Value`** (例如，情景 A 为 `$204.08`)。
    2.  **获取 `HF_before`** (即 `t` 时刻的 `HF_adjusted`)。
    3.  **获取 `HF_TARGET`** (即 `HF_STATIC_THRESH (q30)`)。

    4.  **模拟“行动1：偿还债务”:**
        * **`ΔQ_Repay = Wallet_Value / P_t_USDC`**
        * `HF_after_Repay = calculate_hf(Q_Debt - ΔQ_Repay, ...)`
        * `HF_Improve_Repay = HF_after_Repay - HF_before`
        * `Repay_Success = (HF_after_Repay >= HF_TARGET)`

    5.  **模拟“行动2：增加抵押 (WETH)”:**
        * **`ΔQ_Add = Wallet_Value / P_t_WETH`**
        * `HF_after_Add = calculate_hf(Q_Coll + ΔQ_Add, ...)`
        * `HF_Improve_Add = HF_after_Add - HF_before`
        * `Add_Success = (HF_after_Add >= HF_TARGET)`

    6.  **决策逻辑:**
        * **IF `(Repay_Success == True)` AND `(Add_Success == True)`:**
            * *（两者都成功）* 选择 `HF_Improvement` 更高的那个（遵循您的“单位提升幅度”原则）。
        * **IF `(Repay_Success == True)` AND `(Add_Success == False)`:**
            * *（只有偿还成功）* 选择“偿还债务”。
        * **IF `(Repay_Success == False)` AND `(Add_Success == True)`:**
            * *（只有增加成功）* 选择“增加抵押”。
        * **IF `(Repay_Success == False)` AND `(Add_Success == False)`:**
            * *（两者都失败）* **策略升级！** 记录“中风险策略失败”，并**立即触发 B 模型的运筹优化**。

#### B. 针对“高风险触发” (或“中风险升级”)
* **模型:** **“运筹优化” (Optimization)**
* **目标:** 找到 `ΔQ_Repay` 和 `ΔQ_Add` 的最佳组合，以最大化 `Z`。
* **流程:** 在 `t` 时刻，针对**每种钱包情景** (A, B, C) 分别执行：

    1.  **获取 `Wallet_Value`** (例如，情景 A 为 `$204.08`)。
    2.  **获取 `HF_before`** (即 `t` 时刻的 `HF_adjusted`)。
    3.  **定义 `λ`** (例如，`λ = 100`，此为必须设置的参数)。
    4.  **运行网格搜索 (Grid Search):**
        * `Best_Z = -infinity`
        * `for Pct_to_Repay in [0, 25, 50, 75, 100]:`
            * `Pct_to_Add = 100 - Pct_to_Repay`
            *
            * // **明确的数值计算**
            * `Value_Repay = (Pct_to_Repay / 100) * Wallet_Value`
            * `Value_Add = (Pct_to_Add / 100) * Wallet_Value`
            * **`ΔQ_Repay = Value_Repay / P_t_USDC`**
            * **`ΔQ_Add = Value_Add / P_t_WETH`**
            *
            * // **计算收益 (Metric 1)**
            * `HF_after = calculate_hf(Q_Debt - ΔQ_Repay, Q_Coll + ΔQ_Add, ...)`
            * `HF_Improvement = HF_after - HF_before`
            *
            * // **计算成本 (Metric 3)**
            * `Cost_Repay = (P_t-1_USDC - P_t_USDC) * ΔQ_Repay`
            * `Cost_Add = (P_t-1_WETH - P_t_WETH) * ΔQ_Add`
            * `Total_Cost = Cost_Repay + Cost_Add`
            *
            * // **计算目标函数 Z**
            * `Z = (HF_Improvement * λ) - Total_Cost`
            *
            * // **存储最佳结果**
            * `if Z > Best_Z: ...`
    5.  **决策:** 记录 `Z` 值最高的那组 `(Pct_to_Repay, Pct_to_Add)` 组合为“最优行动”。