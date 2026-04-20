"""Pre-compute cointegration results for the curated pairs universe and cache to disk."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from datetime import date, timedelta

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("seed_pairs")


def main() -> None:
    from pipeline.data.fetcher import StockDataFetcher
    from pipeline.data.universe import PAIRS_UNIVERSE
    from pipeline.stats.cointegration import CointegrationAnalyzer

    fetcher = StockDataFetcher()
    analyzer = CointegrationAnalyzer()

    output_path = Path("data/seed_cointegration.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    # Test both 2-year and 5-year horizons; keep the best result that is cointegrated
    horizons = [2, 5]
    
    for pair in PAIRS_UNIVERSE.values():
        ticker_a, ticker_b = pair.ticker_a, pair.ticker_b
        log.info("Analysing %s / %s ...", ticker_a, ticker_b)
        
        best_result = None
        
        for years in horizons:
            try:
                end_date = date.today().isoformat()
                start_date = (date.today() - timedelta(days=years * 365)).isoformat()
                
                df = fetcher.fetch([ticker_a, ticker_b], start_date=start_date, end_date=end_date)
                prices_a = df[ticker_a]["Close"]
                prices_b = df[ticker_b]["Close"]
                
                result = analyzer.analyze(prices_a, prices_b, ticker_a, ticker_b)
                
                # Update best_result if this one is cointegrated, or if we don't have one yet
                if best_result is None or (result.is_cointegrated and not best_result.is_cointegrated):
                    best_result = result
                elif result.is_cointegrated and best_result.is_cointegrated:
                    # Both cointegrated? Keep the one with lower p-value
                    if result.eg_pvalue < best_result.eg_pvalue:
                        best_result = result
                elif not result.is_cointegrated and not best_result.is_cointegrated:
                    # Neither cointegrated? Keep the one with lower p-value anyway for the log
                    if result.eg_pvalue < best_result.eg_pvalue:
                        best_result = result
                        
            except Exception as exc:
                log.warning("  [ERROR] %s / %s (%dy): %s", ticker_a, ticker_b, years, exc)

        if best_result:
            entry = {
                "ticker_a": best_result.ticker_a,
                "ticker_b": best_result.ticker_b,
                "sector": pair.sector,
                "eg_pvalue": round(best_result.eg_pvalue, 6),
                "hedge_ratio": round(best_result.hedge_ratio, 6),
                "half_life_days": round(best_result.half_life_days, 2),
                "is_cointegrated": best_result.is_cointegrated,
            }
            results.append(entry)
            status = "PASS" if best_result.is_cointegrated else "FAIL"
            log.info(
                "  [%s] p=%.4f  half-life=%.1f days  β=%.4f",
                status, best_result.eg_pvalue, best_result.half_life_days, best_result.hedge_ratio,
            )

    output_path.write_text(json.dumps(results, indent=2))
    cointegrated = sum(1 for r in results if r["is_cointegrated"])
    log.info(
        "\nDone. %d / %d pairs cointegrated. Results saved to %s",
        cointegrated, len(results), output_path,
    )


if __name__ == "__main__":
    main()
