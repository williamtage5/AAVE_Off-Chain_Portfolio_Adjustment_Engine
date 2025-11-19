这是一个出色的问题。您已经从“中风险”（策略A）的“二选一”逻辑，演进到了“高风险”（策略B）的“多维优化”问题。您需要一个数学模型来描述如何**分配**有限的“预备金”（`Wallet_USD`）以实现最佳效果，并且要同时考虑**两种**抵押品（WETH 和 cbBTC）。

您在 `outline.json` 中提到的运筹优化目标函数 `max Z = (HF_Improvement * λ) - Total_Operation_Cost` 是这个方案的核心。

以下是该运筹优化模型的完整数学定义方案。

---

### 方案名称：策略B - 净效益最大化（NMV）模型

本方案定义了一个**非线性约束优化问题**。

**目标：** 在“高风险”触发点 `t`，给定一个固定的“预备金”`Wallet_USD`（来自您的 `wallets.json` 文件），求解三个决策变量（`V_repay_USDC`, `V_add_WETH`, `V_add_cbBTC`），以最大化目标函数 `Z`（净货币价值）。

---

### 1. 已知参数（在触发点 `t`）

在运行模型之前，我们从您的数据文件中提取以下**常量**：

* **基准仓位 (来自 `outline.json`)**:
    * `Q_WETH_base`: 用户的 WETH 初始数量 (0.6306...)
    * `Q_cbBTC_base`: 用户的 cbBTC 初始数量 (0.00058...)
    * `Q_USDC_base`: 用户的 USDC 初始债务 (2040.84...)
* **Aave 协议参数 (来自 `outline.json`)**:
    * `LT_WETH`: WETH的清算阈值 (0.83)
    * `LT_cbBTC`: cbBTC的清算阈值 (0.78)
    * `GAF`: 全局调整因子 (0.974...)
* **市场价格 (来自 `merged_hf_and_prices.csv`)**:
    * `P_WETH_t`, `P_cbBTC_t`, `P_USDC_t`: `t` 时刻的实际价格（用于执行）。
    * `P_WETH_t-1`, `P_cbBTC_t-1`, `P_USDC_t-1`: `t-1` 时刻的价格（用于计算成本）。
* **基准状态 (来自 `merged_hf_and_prices.csv`)**:
    * `HF_before`: `t` 时刻的 `HF_adjusted` 值。
* **模拟参数 (来自 `wallets.json` 和您的定义)**:
    * `Wallet_USD`: 来自所选情景的可用总预备金（例如 `$510.14`）。
    * `λ`: **风险货币化因子** (例如 `λ=100`，**这是您必须在代码中定义的超参数**)。

---

### 2. 决策变量 (Decision Variables)

我们要优化的对象是**USD的分配**。我们定义三个变量：

* `V_repay_USDC`: 决定分配多少USD用于**偿还USDC债务**。
* `V_add_WETH`: 决定分配多少USD用于**购买并补充WETH抵押品**。
* `V_add_cbBTC`: 决定分配多少USD用于**购买并补充cbBTC抵押品**。

---

### 3. 约束条件 (Constraints)

1.  **钱包预算约束 (Wallet Budget Constraint):**
    * 您分配的总金额不能超过您的预备金。
    * `V_repay_USDC + V_add_WETH + V_add_cbBTC <= Wallet_USD`

2.  **非负约束 (Non-Negativity):**
    * 您不能凭空卖出资产或增加债务。
    * `V_repay_USDC >= 0`
    * `V_add_WETH >= 0`
    * `V_add_cbBTC >= 0`

---

### 4. 目标函数 (Objective Function)

我们的目标是： **`Maximize Z = (Metric 1 * λ) - Metric 3`**

我们将 `Metric 1` 和 `Metric 3` 分解为与决策变量相关的数学表达式。

#### 4A. 定义 Metric 3: `Total_Operation_Cost`

我们使用您的**定义A（时间延迟成本）**，并将其定义为“滑点损失”（Slippage Loss）。
`Slippage_Loss = (P_t - P_t-1) * ΔQ`

1.  **计算资产数量变化 (`ΔQ`)**:
    * `ΔQ_repay_USDC = V_repay_USDC / P_USDC_t`
    * `ΔQ_add_WETH = V_add_WETH / P_WETH_t`
    * `ΔQ_add_cbBTC = V_add_cbBTC / P_cbBTC_t`

2.  **计算 `Total_Operation_Cost`**:
    * `Cost_USDC = (P_USDC_t - P_USDC_t-1) * ΔQ_repay_USDC`
    * `Cost_WETH = (P_WETH_t - P_WETH_t-1) * ΔQ_add_WETH`
    * `Cost_cbBTC = (P_cbBTC_t - P_cbBTC_t-1) * ΔQ_add_cbBTC`
    * **`Total_Operation_Cost = Cost_USDC + Cost_WETH + Cost_cbBTC`**

    *（注意：如果 `P_t < P_t-1`，这个“成本”将为负，从而变成“收益”，并增加 `Z`）*

#### 4B. 定义 Metric 1: `HF_Improvement`

1.  **计算调仓后的新仓位 (`Q_after`)**:
    * `Q_USDC_after = Q_USDC_base - ΔQ_repay_USDC`
    * `Q_WETH_after = Q_WETH_base + ΔQ_add_WETH`
    * `Q_cbBTC_after = Q_cbBTC_base + ΔQ_add_cbBTC`

2.  **计算 `HF_after`**:
    * `Numerator_after = (Q_WETH_after * P_WETH_t * LT_WETH) + (Q_cbBTC_after * P_cbBTC_t * LT_cbBTC)`
    * `Denominator_after = (Q_USDC_after * P_USDC_t)`
    * `HF_after = GAF * (Numerator_after / Denominator_after)`

3.  **计算 `HF_Improvement`**:
    * **`HF_Improvement = HF_after - HF_before`**

---

### 5. 完整的优化问题（总结）

**求解：**
`V_repay_USDC`, `V_add_WETH`, `V_add_cbBTC`

**最大化：**
`Z = ( (HF_after - HF_before) * λ ) - (Cost_USDC + Cost_WETH + Cost_cbBTC)`

**约束于：**
1.  `V_repay_USDC + V_add_WETH + V_add_cbBTC <= Wallet_USD`
2.  `V_repay_USDC >= 0`
3.  `V_add_WETH >= 0`
4.  `V_add_cbBTC >= 0`

*（其中 `HF_after`, `Cost_USDC`, `Cost_WETH`, `Cost_cbBTC` 都是基于 `V...` 变量计算的复杂非线性函数）*

---

### 6. 求解方案

由于目标函数 `Z` 是非线性的（因为 `HF_after` 是一个分数，其分母中包含 `V_repay_USDC`），您不能使用简单的线性规划求解器。

您之前“网格搜索”的思路是解决这个问题的**最实用、最稳健**的方法。

**建议的求解策略（网格搜索）：**
1.  定义一个“分配粒度”，例如 `[0%, 10%, 20%, ..., 100%]`。
2.  创建一个三层嵌套循环，在 `Wallet_USD` 的约束下，遍历 `V_repay_USDC`, `V_add_WETH`, 和 `V_add_cbBTC` 的所有组合。
    * `for pct_usdc in [0, 10, ..., 100]:`
    * `for pct_weth in [0, 10, ..., 100 - pct_usdc]:`
    * `pct_cbtc = 100 - pct_usdc - pct_weth`
    * `V_repay_USDC = (pct_usdc / 100) * Wallet_USD`
    * `V_add_WETH = (pct_weth / 100) * Wallet_USD`
    * `V_add_cbBTC = (pct_cbtc / 100) * Wallet_USD`
    * `... 计算 Z ...`
    * `... 存储 Z 最高的 (V...) 组合 ...`
3.  在所有组合中 `Z` 值最高的那个，就是您的“最优调仓决策”。