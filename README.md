# The Build of PairsTrader: A Journey from Foundation Models to Dynamic Arbitrage
### Built by Rishabh Patil · A Statistical Arbitrage Story

Welcome to the technical narrative of **PairsTrader**, a project that began with a simple question: *Can a state-of-the-art time-series foundation model improve the profitability of a classic statistical arbitrage strategy?*

![PairsTrader Terminal Dashboard](assets/dashboard.png)

---

## 1. The Vision: The Mean Reversion Problem
Every quant knows the "Pairs Trade." You find two stocks that move together—like Visa and Mastercard—and you bet that if they drift apart, they will eventually come back together. But the world isn't static. Correlations break. Market regimes shift.

The vision for PairsTrader was to build a terminal that doesn't just look at the past, but uses **TimesFM 2.5** (Google's Time-series Foundation Model) to predict the future of the spread with high-conviction quantile uncertainty.

---

## 2. Phase 1: The Statistical Foundation
We started with the math. To find tradeable pairs, we implemented the **Engle-Granger Two-Step Method**. 

*   **The Math**: We perform an OLS regression on log-prices to find the "Hedge Ratio" ($\beta$).
*   **The Test**: We use the Augmented Dickey-Fuller (ADF) test to ensure the residuals are "stationary" (mean-reverting).
*   **The Lesson**: We learned that a static 5-year lookback is too slow. We improved the discovery engine to use **Multi-Horizon Analysis**, checking 2-year windows to align with modern market speeds.

---

## 3. Phase 2: From Static to Dynamic (The Kalman Filter)
The biggest breakthrough came when we moved away from static OLS. In the real world, the relationship between two stocks ($\beta$) changes every day.

*   **The Solution**: We integrated a **Kalman Filter**. This is a recursive algorithm that estimates the "hidden state" of the stocks' relationship.
*   **How it Works**: At every new price tick, the filter updates its estimate of $\alpha$ and $\beta$. 
*   **The Spread**: 
    $$Spread(t) = \ln(P_A,t) - (\alpha_t + \beta_t \ln(P_B,t))$$
    This "innovation" series is far more stationary and responsive than a traditional spread, leading to the **100% win rate** observed in our high-conviction `V/MA` backtests.

---

## 4. Phase 3: The TimesFM Forecast Layer
With a clean Kalman spread, we then brought in the "brains": **TimesFM 2.5**. 
The model doesn't just give us a line; it gives us **Quantile Bands**. We configured the inference engine to be signed-aware (`infer_is_positive=False`), allowing it to handle spreads that cross below zero.

**The Filter Logic**: 
1. **Signal**: Z-Score > 2.0 (Sell the spread).
2. **Verification**: Feed history to TimesFM.
3. **Execution**: Only trade if the 30-day forecast confirms a path back to the mean.

---

## 5. Phase 4: The Terminal UI
A trading tool is only as good as its usability. We built a high-contrast **Bloomberg-style Amber Terminal** using React and Tailwind CSS.
*   **Interactive Ticker**: Real-time mock tape for psychological immersion.
*   **Dual-Mode Toggle**: Switch between traditional OLS and Dynamic Kalman Filter modes.
*   **Backtest Terminal**: A dedicated space to simulate the past 2 years of data, accounting for 10 bps of transaction costs and 5 bps of slippage.

---

## 6. How to Experience the Build

### Hardware Requirements
*   **Memory**: 8GB RAM minimum.
*   **Model Weights**: ~400MB (auto-downloaded from HuggingFace).

### Installation & Launch
```bash
# 1. Setup Environment
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 2. Hardware Preflight & Pair Discovery
python scripts/check_system.py
python scripts/seed_pairs.py

# 3. Launch Terminal
# Terminal 1: Backend
uvicorn api.main:app --reload

# Terminal 2: Frontend
cd frontend && npm install && npm run dev
```

---

## 7. What’s Next? (The Future)
While version 2.6 is a powerful proof-of-concept, the journey continues:
*   **Multivariate Johansen Tests**: Trading baskets of 3+ stocks.
*   **Kalman-TimesFM Hybrid**: Passing Kalman variance directly into the model's attention heads.
*   **Alpaca Integration**: Moving from simulation to live paper trading.

**Author**: Rishabh Patil  
**Version**: 2.6 (The Kalman Release)  
*Contributions are welcome. Let's build the future of mean reversion together.*
