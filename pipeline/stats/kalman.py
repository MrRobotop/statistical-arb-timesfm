"""Kalman Filter for dynamic hedge ratio estimation.

This module implements a recursive Kalman Filter to estimate the relationship 
between two time-series (Pairs) where the hedge ratio and intercept evolve 
over time (dynamic regime adaptation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class KalmanHedgeRatio:
    """Uses a Kalman Filter to estimate a dynamic hedge ratio and intercept.
    
    The state is represented as [alpha, beta], where:
    y = alpha + beta * x + epsilon
    
    This implementation uses a recursive approach, updating its belief of alpha 
    and beta at each time step based on the prediction error (Innovation).
    """

    def __init__(self, delta: float = 1e-5, R: float = 1e-3):
        """Initialise the Kalman Filter parameters.
        
        Args:
            delta: System noise covariance parameter (process noise).
                Smaller = slower adaptation, smoother beta.
                Larger = faster adaptation to regime shifts, but more noise.
            R: Measurement noise covariance (confidence in the observation).
        """
        self.delta = delta
        self.R = R

    def estimate(self, x: pd.Series, y: pd.Series) -> pd.DataFrame:
        """Estimate dynamic alpha and beta for the relationship y = alpha + beta * x.
        
        Args:
            x: Predictor series (usually log-prices of Stock B).
            y: Dependent series (usually log-prices of Stock A).
            
        Returns:
            DataFrame with 'alpha' (intercept) and 'beta' (hedge ratio) for each time step.
        """
        x_vals = np.log(x.values.astype(float))
        y_vals = np.log(y.values.astype(float))
        n = len(x_vals)

        # Initialise state: [alpha, beta]
        # Start with a neutral assumption (intercept 0, ratio 1)
        theta = np.zeros(2)
        
        # State covariance P: Initial uncertainty in our state estimate
        P = np.eye(2)
        
        # Process noise covariance Q: Represents how much we expect alpha/beta to drift
        Q = self.delta / (1 - self.delta) * np.eye(2)

        alphas = np.zeros(n)
        betas = np.zeros(n)

        for t in range(n):
            # 1. Prediction step: Predict state and covariance for current step
            # We assume theta_t = theta_t-1 (Random Walk model)
            P = P + Q

            # 2. Observation update: Refine prediction with actual data
            # H_t is the measurement matrix mapping state to observation
            H = np.array([1, x_vals[t]])
            
            # Innovation: Difference between actual y and our prediction
            y_hat = np.dot(H, theta)
            error = y_vals[t] - y_hat
            
            # Innovation covariance (uncertainty in the error)
            S = np.dot(H, np.dot(P, H.T)) + self.R
            
            # Kalman gain: How much to trust the new observation vs the prediction
            K = np.dot(P, H.T) / S
            
            # Update state estimate with the innovation weighted by Kalman Gain
            theta = theta + K * error
            
            # Update state covariance (reduce uncertainty)
            P = P - np.outer(K, np.dot(H, P))

            alphas[t] = theta[0]
            betas[t] = theta[1]

        return pd.DataFrame(
            {"alpha": alphas, "beta": betas},
            index=x.index
        )
