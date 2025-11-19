# 🧠 Off-Chain Decision Engine: Codebase Documentation

## 🗺 System Blueprint

The codebase implements the **Multi-Factor Dynamic Risk Model** described in our research. The system operates in a closed loop:
1.  **Data Ingestion:** Fetches raw on-chain data and normalizes prices.
2.  **Predictive Modeling:** Forecasts asset prices and Health Factor (HF) trends for $T+N$ hours.
3.  **Risk Classification:** Calculates the Systemic Risk ($R_1$) and Individual Risk (Velocity/Weighted HF) to determine the "Risk Zone."
4.  **Strategy Simulation:** Simulates two competing strategies (Repay Debt vs. Add Collateral) using Operations Research.
5.  **Decision Making:** Selects the optimal action based on Net Monetary Value (NMV) and generates the execution payload.

---

## 📂 Directory Structure & Module Breakdown

### 1. Data Processing & Utilities
*Foundational scripts for normalizing data from The Graph/RPC and preparing it for the model.*

* **`trans_to_usd.py`**
    * **Function:** A utility module to convert various token prices (ETH, WBTC, Stablecoins) into a unified USD standard. This ensures all Health Factor calculations use a consistent denominator.
* **`calculate_from_aave.ipynb`**
    * **Function:** Connects to AAVE protocol data to retrieve user-specific parameters: Liquidation Thresholds, LTV (Loan-to-Value), and current asset balances.
* **`preprocess_r1_to_hourly.py`**
    * **Function:** Processes macro-market data to compute the **$R_1$ Indicator** (Rigid Debt / Risk-Weighted Net Collateral) on an hourly basis. This serves as the "systemic risk thermometer."

### 2. Predictive Modeling (The "Eye")
*Machine Learning modules that forecast future market states to preempt liquidation risks.*

* **`BTC_test-小时级-PytorchLSTM+技术指标.ipynb`**
    * **Function:** The core PyTorch LSTM model. It uses historical hourly data and technical indicators (RSI, MACD, Bollinger Bands) to predict asset price volatility.
* **`prediction_for_each_coin.ipynb`**
    * **Function:** A batch processing script that applies the trained LSTM/XGBoost models to every asset in the user's portfolio to generate price vectors for future timestamps.
* **`forward_6_HF_simulation.py`**
    * **Function:** Uses the predicted prices to simulate the user's Health Factor for the next 6 hours ($T+1$ to $T+6$).
* **`calculate_forward_6_for_3icon.py`**
    * **Function:** A specialized helper script to compute forward-looking metrics specifically for the 3 core assets (likely BTC, ETH, Stablecoins) heavily used in the portfolio.

### 3. Risk Assessment Core
*The logic engine that interprets data to categorize the current threat level.*

* **`calcutlate_true_HF.py`**
    * **Function:** Implements the precise AAVE V3 Health Factor formula: $\frac{\sum \text{Collateral} \times \text{Liquidation Threshold}}{\text{Total Debt}}$.
* **`weighed_HF.py`**
    * **Function:** Calculates the **3-Month Weighted Predicted HF**. It applies a weighting mechanism to historical and predicted data to establish a "Safety Floor" (e.g., P30 percentile).
* **`judge_phase.py`**
    * **Function:** The Classifier. It takes inputs ($R_1$, Velocity, Predicted HF) and outputs the **Risk Zone**:
        * 🟢 **Low Risk:** Passive monitoring.
        * 🟠 **Medium Risk:** Preparation / Opportunistic adjustment.
        * 🔴 **High Risk:** Immediate intervention triggered.

### 4. Strategy Simulation & Optimization
*The "Solver" that determines the best course of action when risk is detected.*

* **`strategy_A_with_metrics.py` (Repay Debt)**
    * **Function:** Simulates **Strategy One**. It calculates the impact of repaying debt, prioritizing "isolated asset debt" and assets with the highest HF improvement per unit[cite: 141, 156].
* **`strategy_B.py` (Add Collateral)**
    * **Function:** Simulates **Strategy Two**. It calculates the impact of supplying additional collateral (e.g., WETH, WBTC) to the user's position.
* **`amount_simulation.ipynb`**
    * **Function:** An optimization loop that tests various amounts for Strategy A and B to find the local maximum for the Objective Function.

### 5. Evaluation Metrics
*Quantitative modules to score Strategy A vs. Strategy B.*

* **`HF Improvement.py` (Metric 1)**
    * **Function:** Calculates the raw efficiency of the move: $\Delta HF / \$ \text{Input}$.
* **`Liquidation Rate Reduction.py` (Metric 2)**
    * **Function:** Uses the *Belenko & Vosorov (2025)* Zero-Drift Brownian Motion model to estimate how much the probability of liquidation drops after the proposed action.
* **`Operation Cost.py` (Metric 3)**
    * **Function:** Estimates the total cost of the action, including Gas Fees (Sepolia/Mainnet) and Slippage.

### 6. Orchestration (The Main Loop)
* **`generation.py`**
    * **Function:** The entry point.
        1.  Calls **Prediction** modules.
        2.  Runs **Risk Assessment** (`judge_phase.py`).
        3.  If Risk > Threshold, runs **Simulation** (`strategy_A` vs `strategy_B`).
        4.  Calculates **Metrics** for both.
        5.  Outputs the final decision (JSON/TX payload) to be sent to the On-Chain Executor.

---

## 🔄 Application Workflow

The following sequence diagrams how the code files interact during a single execution cycle:

1.  **Start:** `generation.py` initiates the cycle.
2.  **Fetch:** `calculate_from_aave.ipynb` gets current portfolio state; `trans_to_usd.py` standardizes prices.
3.  **Predict:** `prediction_for_each_coin.ipynb` + `forward_6_HF_simulation.py` generate the $HF_{t+1...t+6}$ vector.
4.  **Assess:**
    * `preprocess_r1_to_hourly.py` provides current Systemic Risk ($R_1$).
    * `weighed_HF.py` provides the Safety Floor.
    * `judge_phase.py` combines these to return a **Risk Flag**.
5.  **Optimize (if Risk Flag == True):**
    * **Path A:** `strategy_A_with_metrics.py` calculates Repayment metrics.
    * **Path B:** `strategy_B.py` calculates Collateral metrics.
    * **Scoring:** `HF Improvement.py` + `Liquidation Rate Reduction.py` - `Operation Cost.py`.
6.  **Execute:** The strategy with the highest **Net Monetary Value ($Z$)** is selected, and `generation.py` formats the transaction data.

---