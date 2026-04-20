#!/usr/bin/env python
"""Standalone integration test for SpreadForecaster.

Downloads the TimesFM checkpoint on first run (~400 MB).
Run with:  uv run python tests/test_forecaster.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure the project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from pipeline.model.forecaster import ForecastResult, SpreadForecaster
from pipeline.model.loader import TimesFMLoader


def make_ou_spread(n: int = 252, phi: float = 0.95, sigma: float = 0.1, seed: int = 42) -> np.ndarray:
    """Generate a stationary OU process: spread(t) = phi * spread(t-1) + N(0, sigma)."""
    rng = np.random.default_rng(seed)
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = phi * spread[t - 1] + rng.normal(0, sigma)
    return spread


def assert_eq(label: str, actual, expected) -> None:
    if actual != expected:
        print(f"  FAIL  {label}: got {actual!r}, expected {expected!r}")
        sys.exit(1)
    print(f"  pass  {label}")


def assert_true(label: str, condition: bool, detail: str = "") -> None:
    if not condition:
        msg = f"  FAIL  {label}"
        if detail:
            msg += f": {detail}"
        print(msg)
        sys.exit(1)
    print(f"  pass  {label}")


def run_tests(result: ForecastResult, horizon: int) -> None:
    print("\n── Assertions ─────────────────────────────────")

    assert_eq("horizon field", result.horizon, horizon)
    assert_eq("point_forecast length", len(result.point_forecast), horizon)
    assert_eq("q10 length", len(result.q10), horizon)
    assert_eq("q50 length", len(result.q50), horizon)
    assert_eq("q90 length", len(result.q90), horizon)
    assert_eq("mean_forecast length", len(result.mean_forecast), horizon)

    assert_true(
        "point_forecast no NaN",
        not any(np.isnan(v) for v in result.point_forecast),
    )
    assert_true(
        "q10 no NaN",
        not any(np.isnan(v) for v in result.q10),
    )
    assert_true(
        "q90 no NaN",
        not any(np.isnan(v) for v in result.q90),
    )

    # Quantile ordering: q10[i] <= q50[i] <= q90[i]
    q10 = np.array(result.q10)
    q50 = np.array(result.q50)
    q90 = np.array(result.q90)

    assert_true(
        "q10 ≤ q50 for all steps",
        bool(np.all(q10 <= q50 + 1e-6)),
        f"max violation: {float(np.max(q10 - q50)):.4f}",
    )
    assert_true(
        "q50 ≤ q90 for all steps",
        bool(np.all(q50 <= q90 + 1e-6)),
        f"max violation: {float(np.max(q50 - q90)):.4f}",
    )

    assert_true(
        "confidence_interval_width ≥ 0",
        result.confidence_interval_width >= 0,
    )
    assert_true(
        "forecast_direction is valid",
        result.forecast_direction in {"MEAN_REVERTING", "DIVERGING", "FLAT"},
        f"got {result.forecast_direction!r}",
    )


def main() -> None:
    horizon = 30

    print("── Generating synthetic OU spread ──────────────")
    spread = make_ou_spread(n=252, phi=0.95, sigma=0.1)
    print(f"  shape: {spread.shape},  mean: {spread.mean():.4f},  std: {spread.std():.4f}")

    print("\n── Loading TimesFM ─────────────────────────────")
    loader = TimesFMLoader.get_instance()
    loader.load(max_context=512, max_horizon=64)
    print("  Model loaded OK")

    print(f"\n── Running forecast (horizon={horizon}) ────────")
    forecaster = SpreadForecaster(loader)
    result = forecaster.forecast(spread_values=spread, horizon=horizon)

    run_tests(result, horizon)

    print("\n── Visual summary ──────────────────────────────")
    print(f"  current_spread      : {result.current_spread:+.4f}")
    print(f"  forecast_endpoint   : {result.forecast_endpoint:+.4f}")
    print(f"  forecast_direction  : {result.forecast_direction}")
    print(f"  CI width (mean)     : {result.confidence_interval_width:.4f}")
    print(f"  point_forecast[0:5] : {[f'{v:+.4f}' for v in result.point_forecast[:5]]}")
    print(f"  q10[0:5]            : {[f'{v:+.4f}' for v in result.q10[:5]]}")
    print(f"  q90[0:5]            : {[f'{v:+.4f}' for v in result.q90[:5]]}")

    print("\n  ALL TESTS PASSED")


if __name__ == "__main__":
    main()
