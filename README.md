# 📈 PairsTrader: Dynamic Statistical Arbitrage Terminal
### Powered by Google’s TimesFM 2.5 & Recursive Kalman Filters
**Author**: Rishabh Patil · **Version**: 2.6 (Kalman Enhanced)

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TimesFM](https://img.shields.io/badge/Model-TimesFM_2.5-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://github.com/google-research/timesfm)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/UI-React_18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

PairsTrader is a production-grade quantitative terminal designed for the next generation of statistical arbitrage. While traditional pairs trading relies on static historical averages, this project implements a **recursive state-space model (Kalman Filter)** coupled with **Google’s TimesFM 2.5 foundation model** to predict mean-reversion with unprecedented precision.

![PairsTrader Terminal Dashboard](assets/dashboard.png)

---

## 📖 About the Project

Statistical arbitrage, specifically pairs trading, is one of the most enduring strategies in quantitative finance. However, it often fails in high-volatility environments due to **Beta Drift**—where the relationship between two assets changes before the trade can close.

PairsTrader was built to solve this by treating the market as a non-stationary system. By integrating the **Kalman Filter** for real-time hedge ratio adaptation and **TimesFM 2.5** for sequence-aware forecasting, the terminal provides a "Dynamic Edge" that traditional OLS-based systems lack.

### Core Features
- **Dynamic Hedge Ratios**: Recursive estimation of $\beta$ using a Kalman state-space model.
- **Foundation Model Forecasts**: leveraging Google’s [TimesFM](https://github.com/google-research/timesfm) for 30-day spread projections.
- **Multi-Horizon Discovery**: Cointegration testing across 2-year and 5-year regimes.
- **High-Contrast Terminal**: A Bloomberg-inspired, low-latency UI for real-time monitoring.
- **Vectorized Backtester**: Full simulation suite accounting for slippage and transaction costs.

---

## 🧮 Mathematical Foundations

### I. Cointegration Discovery
We identify tradeable pairs using the **Engle-Granger Two-Step Method**. A pair is valid if the residuals of their price relationship are stationary.
$$\Delta \epsilon_t = \zeta \epsilon_{t-1} + \sum_{i=1}^p \delta_i \Delta \epsilon_{t-i} + \nu_t$$
If $p < 0.05$ (ADF test), we reject the null hypothesis of a unit root (non-stationarity).

### II. Recursive State-Space (Kalman Filter)
Traditional strategies use static $\beta$. We model the relationship as a hidden state that updates with every price candle:
*   **Prediction Step**:
    $$\hat{\theta}_{t|t-1} = \hat{\theta}_{t-1}, \quad P_{t|t-1} = P_{t-1} + Q$$
*   **Update Step**:
    $$K_t = P_{t|t-1} H_t^T (H_t P_{t|t-1} H_t^T + R)^{-1}$$
    $$\hat{\theta}_t = \hat{\theta}_{t|t-1} + K_t (y_t - H_t \hat{\theta}_{t|t-1})$$
Where $\theta_t = [\alpha_t, \beta_t]^T$. This allows the model to "learn" regime shifts instantly.

### III. Mean Reversion Speed
We derive the **Half-Life** of the trade by modeling the spread as an **Ornstein-Uhlenbeck (OU) Process**:
$$dx_t = \lambda(\mu - x_t)dt + \sigma dW_t \implies HL = \frac{-\ln(2)}{\lambda}$$

---

## ⚙️ Hardware & Configuration

### Optimized Stack
This build was optimized on **Apple Silicon (M2 Max)** to leverage **MPS (Metal Performance Shaders)** for GPU-accelerated inference.

| Requirement | Minimum Spec | Recommended |
|-------------|--------------|-------------|
| **CPU** | 8-Core Intel/M1 | 12-Core M2/M3 |
| **RAM** | 8GB Unified | 16GB Unified |
| **Storage** | 1GB SSD | 2GB NVMe |
| **GPU** | Integrated | 30-Core+ MPS/CUDA |

---

## 🚀 Getting Started

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
Configure your HuggingFace token in `.env` to access TimesFM weights:
```bash
cp .env.example .env
# Edit .env and add HF_TOKEN=your_token_here
```

### 3. Initialize & Launch
```bash
# Hardware preflight
python scripts/check_system.py

# Seed cointegration database
python scripts/seed_pairs.py

# Start Backend
uvicorn api.main:app --reload

# In another terminal: Start Frontend
cd frontend && npm install && npm run dev
```

---

## 📚 References & Research

### Foundation Models
- **Google Research**: [TimesFM: A Decoder-Only Foundation Model for Time-series Forecasting](https://github.com/google-research/timesfm) (2024).

### Quantitative Foundations
- **Engle & Granger (1987)**: "Co-Integration and Error Correction: Representation, Estimation, and Testing." *Econometrica*.
- **Kalman (1960)**: "A New Approach to Linear Filtering and Prediction Problems." *Journal of Basic Engineering*.
- **WorldQuant BRAIN**: Inspired by [WorldQuant BRAIN](https://www.worldquant.com/brain/) quantitative alpha discovery methodologies.

---

## 🤝 Contributing & Support
We welcome contributions from the Quant and ML community. Please open a PR to suggest new cointegration tests (Johansen, CADF) or to integrate additional foundation models.

**Project maintained by Rishabh Patil.**  
*Quantitative analysis is a science; trading is an art. PairsTrader is the brush.*
