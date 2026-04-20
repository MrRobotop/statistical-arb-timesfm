"""Curated pairs universe with validation metadata."""

from __future__ import annotations

from pydantic import BaseModel


class PairConfig(BaseModel):
    name: str
    ticker_a: str
    ticker_b: str
    sector: str
    rationale: str
    default_lookback_days: int = 504


PAIRS_UNIVERSE: dict[str, PairConfig] = {
    "KO_PEP": PairConfig(
        name="KO_PEP",
        ticker_a="KO",
        ticker_b="PEP",
        sector="Consumer Staples",
        rationale="Cola duopoly",
    ),
    "XOM_CVX": PairConfig(
        name="XOM_CVX",
        ticker_a="XOM",
        ticker_b="CVX",
        sector="Energy",
        rationale="Integrated majors",
    ),
    "JPM_BAC": PairConfig(
        name="JPM_BAC",
        ticker_a="JPM",
        ticker_b="BAC",
        sector="Financials",
        rationale="US bank leaders",
    ),
    "MSFT_GOOGL": PairConfig(
        name="MSFT_GOOGL",
        ticker_a="MSFT",
        ticker_b="GOOGL",
        sector="Technology",
        rationale="Cloud hyperscalers",
    ),
    "WMT_TGT": PairConfig(
        name="WMT_TGT",
        ticker_a="WMT",
        ticker_b="TGT",
        sector="Retail",
        rationale="US discount retail",
    ),
    "MCD_YUM": PairConfig(
        name="MCD_YUM",
        ticker_a="MCD",
        ticker_b="YUM",
        sector="Restaurants",
        rationale="QSR competitors",
    ),
    "GLD_SLV": PairConfig(
        name="GLD_SLV",
        ticker_a="GLD",
        ticker_b="SLV",
        sector="Commodities",
        rationale="Precious metals",
    ),
    "SPY_IVV": PairConfig(
        name="SPY_IVV",
        ticker_a="SPY",
        ticker_b="IVV",
        sector="ETF",
        rationale="S&P 500 trackers",
    ),
    "HD_LOW": PairConfig(
        name="HD_LOW",
        ticker_a="HD",
        ticker_b="LOW",
        sector="Retail",
        rationale="Home improvement",
    ),
    "V_MA": PairConfig(
        name="V_MA",
        ticker_a="V",
        ticker_b="MA",
        sector="Fintech",
        rationale="Payment duopoly",
    ),
}


def get_pair(name: str) -> PairConfig:
    """Return PairConfig by name, e.g. 'KO_PEP'."""
    try:
        return PAIRS_UNIVERSE[name]
    except KeyError:
        raise KeyError(f"Unknown pair '{name}'. Available: {list(PAIRS_UNIVERSE)}")


def list_pairs() -> list[PairConfig]:
    """Return all pairs in the universe."""
    return list(PAIRS_UNIVERSE.values())
