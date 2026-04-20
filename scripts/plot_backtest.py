"""Terminal-based ASCII plotter for backtest equity curve."""

from __future__ import annotations

import json
from pathlib import Path


def plot_ascii(values: list[float], width: int = 60, height: int = 15) -> None:
    """Print a simple ASCII line chart of the given values."""
    if not values:
        return

    min_v = min(values)
    max_v = max(values)
    rng = max_v - min_v or 1.0

    # Scale values to fit height
    scaled = [int((v - min_v) / rng * (height - 1)) for v in values]
    
    # Resample to fit width if necessary
    if len(scaled) > width:
        indices = [int(i * (len(scaled) - 1) / (width - 1)) for i in range(width)]
        points = [scaled[i] for i in indices]
    else:
        points = scaled

    print(f"\nEquity Curve (Range: ${min_v:.2f} to ${max_v:.2f})")
    print("-" * (len(points) + 2))
    for h in range(height - 1, -1, -1):
        line = ""
        for p in points:
            if p == h:
                line += "*"
            elif p > h:
                line += "|"
            else:
                line += " "
        print(f"|{line}|")
    print("-" * (len(points) + 2))


def main() -> None:
    results_path = Path("data/backtest_kalman_result.json")
    if not results_path.exists():
        print(f"Error: {results_path} not found. Run backtest_kalman.py first.")
        return

    with open(results_path) as f:
        data = json.load(f)

    equity_curve = data.get("equity_curve", [])
    if not equity_curve:
        print("No equity curve found in results.")
        return

    plot_ascii(equity_curve)


if __name__ == "__main__":
    main()
