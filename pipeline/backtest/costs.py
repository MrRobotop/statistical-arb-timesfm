"""Transaction cost and slippage models for backtesting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostModel:
    """Per-trade transaction cost and market impact calculator."""

    round_trip_bps: float = 10.0
    slippage_bps: float = 5.0

    def total_cost(self, notional: float) -> float:
        """Dollar cost for one completed round-trip trade (entry + exit)."""
        return notional * (self.round_trip_bps + self.slippage_bps) / 10_000

    def one_way_cost(self, notional: float) -> float:
        """Dollar cost for a single leg (entry only or exit only)."""
        return notional * (self.round_trip_bps / 2 + self.slippage_bps) / 10_000
