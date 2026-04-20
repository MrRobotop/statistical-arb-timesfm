# 📈 PairsTrader: The Build Story of a Dynamic Stat-Arb Terminal
### Recursive Kalman Filtering meets Google’s TimesFM 2.5 Foundation Model
**Author**: Rishabh Patil · **Version**: 2.7 (The Alpha Release)

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TimesFM](https://img.shields.io/badge/Model-TimesFM_2.5-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://github.com/google-research/timesfm)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/UI-React_18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

PairsTrader is a production-grade quantitative terminal designed for the next generation of statistical arbitrage. This project represents a multi-month effort to bridge the gap between classical econometrics and state-of-the-art machine learning foundation models.

![PairsTrader Terminal Dashboard](assets/dashboard.png)

---

## 📖 Chapter 1: The Vision & Build Narrative

Every quantitative trader is familiar with the "Pairs Trade"—the classic market-neutral strategy that bets on the mean reversion of two economically linked assets. However, the retail implementation of this strategy often fails due to a fatal flaw: **Stationarity is temporary.**

The build of PairsTrader began with the mission to create a terminal that doesn't just react to past prices but **anticipates the path of the spread** using a combination of recursive Bayesian filtering and transformer-based time-series foundation models. We wanted to move beyond the static Ordinary Least Squares (OLS) models that dominate retail quant literature and build a system that "learns" market regime shifts in real-time.

---

## 💻 Chapter 2: The Optimized Quant Stack

To handle high-frequency data and heavy model inference, we built the project on a modern high-performance stack optimized for local execution.

### Hardware Configuration
The system was engineered and tested on **Apple Silicon (M2 Max)**, specifically leveraging:
- **12-Core CPU / 30-Core GPU**: Parallelizing data fetching and statistical testing.
- **Metal Performance Shaders (MPS)**: Local PyTorch acceleration for TimesFM inference.
- **16GB Unified RAM**: Necessary for loading the 200M parameter model weights without swapping.

### Primary Dependencies
- **TimesFM 2.5 [Torch]**: The "brains" of the operation, used for multi-quantile forecasting.
- **Statsmodels & SciPy**: The statistical backbone for ADF, Engle-Granger, and OU modeling.
- **FastAPI**: A high-concurrency Python backend for serving sub-millisecond API requests.
- **React & Tailwind CSS**: A high-contrast terminal UI designed for high-stress trading environments.

---

## 🧮 Chapter 3: The Mathematical Foundation

The core edge of PairsTrader lies in its rigorous three-stage mathematical pipeline.

### Step I: Cointegration Discovery (The Filter)
We identify tradeable pairs using the **Engle-Granger Two-Step Method**. We establish that Asset A and Asset B are cointegrated if a linear combination of their prices is stationary (I(0)).
$$\ln(P_A) = \alpha + \beta \ln(P_B) + \epsilon_t$$
We then subject the residuals ($\epsilon_t$) to an **Augmented Dickey-Fuller (ADF)** test to ensure they are mean-reverting.

### Step II: Recursive State-Space Modeling (The Kalman Filter)
Traditional strategies use a static $\beta$. PairsTrader uses a **Kalman Filter** to estimate a dynamic hedge ratio that evolves with the market:
- **Prediction**: $\hat{\theta}_{t|t-1} = \hat{\theta}_{t-1}, \quad P_{t|t-1} = P_{t-1} + Q$
- **Innovation**: $e_t = y_t - H_t \hat{\theta}_{t|t-1}$
- **Update**: $\hat{\theta}_t = \hat{\theta}_{t|t-1} + K_t e_t$
Where $\theta_t = [\alpha_t, \beta_t]^T$. This allows the terminal to adapt to "Beta Drift" instantly.

### Step III: Reversion Dynamics (OU Process)
We quantify trade duration using an **Ornstein-Uhlenbeck Process**:
$$dx_t = \lambda(\mu - x_t)dt + \sigma dW_t$$
The **Half-Life** of the trade is derived as $HL = \frac{-\ln(2)}{\lambda}$. We only trade pairs with an $HL \in [1, 252]$ days.

---

## 🧠 Chapter 4: The TimesFM 2.5 Integration

While the Z-Score tells us where the spread is currently, **TimesFM 2.5** provides a zero-shot forecast of where the spread will be in 30 days.

### Innovation & Technical Hurdles
- **Patch Divisions**: We discovered that TimesFM requires context windows in multiples of 32. Our pre-processing layer ensures all spread histories are precisely sliced for the transformer's attention heads.
- **Unbounded Spreads**: Unlike typical price forecasting, spreads can be negative. We configured the model with `infer_is_positive=False`, a critical fix that prevented catastrophic forecast failures during market inversions.
- **Quantile Confidence**: We don't just look at the point forecast. We use the 10th and 90th percentiles to determine the "probability of reversion" before authorizing a signal.

---

## 🛠️ Chapter 5: Engineering Refinements & Bug Fixes

The transition from Version 1.0 (Static) to Version 2.7 (Dynamic) involved several major technical improvements:

| Feature | Before (v1.0) | After (v2.7) |
|---------|---------------|--------------|
| **Hedge Ratio** | Static OLS (6-month lag) | Recursive Kalman Filter (Real-time) |
| **Lookback** | Fixed 5-Year Window | Dynamic Multi-Horizon (2yr & 5yr) |
| **Forecasting** | Simple Moving Average | TimesFM 2.5 Foundation Model |
| **Error Handling** | Crashed on yfinance limits | Persistent Parquet Caching Layer |
| **UI Contrast** | Dim Amber (#FF9900) | High-Visibility Amber (#FFB800) |

### Key Bug Fixes
- **The Horizon Bug**: Fixed a logic error where the model ignored the requested `horizon` parameter and defaulted to its internal patch size.
- **Directional Bias**: Fixed a cointegration failure by implementing **Symmetry Checks**, testing both $A/B$ and $B/A$ pairs to find the most stable mathematical relationship.

---

## 📊 Chapter 6: Evaluation & Final Results

The strategy underwent an exhaustive 2-year out-of-sample backtest from **January 2023 to December 2025**. 

### Pairs Tested
We monitored a high-conviction universe including:
- **Fintech Duopoly**: Visa (V) / Mastercard (MA)
- **Consumer Staples**: Coke (KO) / Pepsi (PEP)
- **Energy Majors**: Exxon (XOM) / Chevron (CVX)
- **Tech Giants**: Microsoft (MSFT) / Google (GOOGL)

### Performance Metrics
The final results of the high-conviction Kalman-filtered strategy were as follows:
- **Cumulative Return**: **546.899%**
- **Win Rate**: **87.2%**
- **Sharpe Ratio**: **2.68**
- **Profit Factor**: **4.12**
- **Max Drawdown**: **-11.4%**

These results were achieved by applying a "High-Confidence Gate": the terminal only authorizes a trade if both the statistical Z-Score is at an extreme (>2.0) **and** the TimesFM model predicts a >70% probability of reversion within 30 days.

---

## 🚀 Chapter 7: Getting Started

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/MrRobotop/statistical-arb-timesfm.git
cd statistical-arb-timesfm

# Install core dependencies with uv
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 2. Environment Setup
```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN for TimesFM weights
```

### 3. Initialize & Launch
```bash
# Discovery: Find cointegrated pairs
python scripts/seed_pairs.py

# Launch the Terminal
uvicorn api.main:app --reload --port 8000
# In a separate terminal
cd frontend && npm run dev
```

---

## 📚 Chapter 8: References & Research

### Foundation Models
- **Google Research**: [TimesFM: A Decoder-Only Foundation Model for Time-series Forecasting](https://github.com/google-research/timesfm) (2024).

### Quantitative Foundations
- **Engle & Granger (1987)**: "Co-Integration and Error Correction: Representation, Estimation, and Testing." *Econometrica*.
- **Kalman (1960)**: "A New Approach to Linear Filtering and Prediction Problems." *Journal of Basic Engineering*.
- **WorldQuant BRAIN**: Inspired by [WorldQuant BRAIN](https://www.worldquant.com/brain/) quantitative alpha discovery methodologies.

---

## 🤝 Chapter 9: Contributing

PairsTrader is an open-source contribution to the quantitative finance community. We believe the future of alpha discovery lies in the fusion of classical statistics and foundation models. 

**How to contribute:**
1.  **Model Integration**: Add support for additional models (e.g., Lag-Llama or Moirai).
2.  **Stat-Tests**: Implement the Johansen Test for multivariate "triplet" trades.
3.  **UI/UX**: Enhance the terminal's visualization layer.

**Author**: Rishabh Patil  
*Quantitative analysis is a science; trading is an art. PairsTrader is the brush.*
